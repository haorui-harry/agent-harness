"""Auto-tuning utility for choosing best mode/recipe configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.harness.models import HarnessConstraints

if TYPE_CHECKING:  # pragma: no cover
    from app.harness.engine import HarnessEngine


@dataclass
class OptimizationCandidate:
    """One candidate config for harness optimization."""

    mode: str
    recipe: str


class HarnessOptimizer:
    """Try multiple mode/recipe combinations and rank by value index."""

    def optimize(
        self,
        engine: HarnessEngine,
        query: str,
        candidates: list[OptimizationCandidate] | None = None,
        constraints: HarnessConstraints | None = None,
        live_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active = candidates or self._default_candidates()
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for item in active:
            run = engine.run(
                query=query,
                mode=item.mode,
                recipe=item.recipe or None,
                constraints=constraints,
                live_model=live_model,
            )
            card = engine.build_value_card(run)
            value_index = float(card.get("value_index", 0.0))
            row = {
                "mode": item.mode,
                "recipe": item.recipe,
                "value_index": value_index,
                "band": card.get("band", ""),
                "tool_calls": run.eval_metrics.get("tool_calls", 0.0),
                "tool_success_rate": run.eval_metrics.get("tool_success_rate", 0.0),
                "safety": self._dimension(card, "safety"),
                "innovation": self._dimension(card, "innovation"),
                "completion": run.eval_metrics.get("completion_score", 0.0),
                "session_id": run.metadata.get("session_id", ""),
            }
            rows.append(row)
            if best is None or value_index > float(best.get("value_index", 0.0)):
                best = row

        rows.sort(key=lambda item: float(item.get("value_index", 0.0)), reverse=True)
        return {
            "query": query,
            "candidates": [item.__dict__ for item in active],
            "leaderboard": rows,
            "best": best or {},
            "recommendation": self._recommend(best or {}),
        }

    @staticmethod
    def _dimension(card: dict[str, Any], name: str) -> float:
        for item in card.get("dimensions", []):
            if not isinstance(item, dict):
                continue
            if item.get("name") == name:
                return float(item.get("score", 0.0))
        return 0.0

    @staticmethod
    def _recommend(best: dict[str, Any]) -> str:
        if not best:
            return "No valid candidate found."
        return (
            f"Use mode={best.get('mode')} recipe={best.get('recipe')} "
            f"(value_index={round(float(best.get('value_index', 0.0)), 2)})."
        )

    @staticmethod
    def _default_candidates() -> list[OptimizationCandidate]:
        return [
            OptimizationCandidate(mode="balanced", recipe="risk-radar"),
            OptimizationCandidate(mode="deep", recipe="ecosystem-hunter"),
            OptimizationCandidate(mode="balanced", recipe="router-forge"),
            OptimizationCandidate(mode="safety_critical", recipe="risk-radar"),
            OptimizationCandidate(mode="fast", recipe="router-forge"),
        ]
