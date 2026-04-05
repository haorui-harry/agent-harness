"""Auto-tuning utility for choosing best mode/recipe configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.harness.models import HarnessConstraints
from app.harness.task_profile import analyze_task_request

if TYPE_CHECKING:  # pragma: no cover
    from app.harness.engine import HarnessEngine


@dataclass
class OptimizationCandidate:
    """One candidate config for harness optimization."""

    mode: str
    recipe: str
    auto_recipe: bool = True
    label: str = ""


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
        active = candidates or self._default_candidates(engine=engine, query=query)
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for item in active:
            effective_constraints = self._candidate_constraints(base=constraints, candidate=item)
            run = engine.run(
                query=query,
                mode=item.mode,
                recipe=item.recipe or None,
                constraints=effective_constraints,
                live_model=live_model,
            )
            card = engine.build_value_card(run)
            value_index = float(card.get("value_index", 0.0))
            row = {
                "mode": item.mode,
                "recipe": item.recipe,
                "auto_recipe": item.auto_recipe,
                "label": item.label or self._candidate_label(item),
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
        auto_recipe = bool(best.get("auto_recipe", True))
        recipe = str(best.get("recipe", "") or ("auto" if auto_recipe else "none"))
        return (
            f"Use mode={best.get('mode')} recipe={recipe} "
            f"(value_index={round(float(best.get('value_index', 0.0)), 2)})."
        )

    def _default_candidates(self, *, engine: HarnessEngine, query: str) -> list[OptimizationCandidate]:
        profile = analyze_task_request(query=query, target="general")
        task_spec = profile.task_spec if isinstance(profile.task_spec, dict) else {}
        channels = {
            str(item).strip()
            for item in task_spec.get("required_channels", [])
            if str(item).strip()
        }
        domains = {
            str(item).strip()
            for item in task_spec.get("domains", [])
            if str(item).strip()
        }
        primary = str(task_spec.get("primary_artifact_kind", "")).strip()
        suggested_recipe = engine.recipes.suggest_from_profile(profile.to_dict())
        candidates: list[OptimizationCandidate] = [
            OptimizationCandidate(mode="balanced", recipe="", auto_recipe=False, label="direct-balanced"),
            OptimizationCandidate(mode="balanced", recipe="", auto_recipe=True, label="auto-balanced"),
        ]

        if "web" in channels or primary in {"deliverable_report", "benchmark_manifest", "benchmark_run_config"}:
            candidates.append(OptimizationCandidate(mode="deep", recipe="", auto_recipe=True, label="auto-deep"))
        if "risk" in channels or "risk" in domains:
            candidates.append(OptimizationCandidate(mode="safety_critical", recipe="", auto_recipe=True, label="auto-safe"))
        if primary in {"patch_draft", "patch_plan"}:
            candidates.append(OptimizationCandidate(mode="fast", recipe="", auto_recipe=True, label="auto-fast"))
        if suggested_recipe is not None:
            candidates.append(
                OptimizationCandidate(
                    mode=suggested_recipe.default_mode or "balanced",
                    recipe=suggested_recipe.name,
                    auto_recipe=False,
                    label=f"explicit-{suggested_recipe.name}",
                )
            )
        deduped: list[OptimizationCandidate] = []
        seen: set[tuple[str, str, bool]] = set()
        for item in candidates:
            key = (item.mode, item.recipe, item.auto_recipe)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _candidate_constraints(*, base: HarnessConstraints | None, candidate: OptimizationCandidate) -> HarnessConstraints:
        base_constraints = base or HarnessConstraints()
        return HarnessConstraints(
            max_steps=base_constraints.max_steps,
            max_tool_calls=base_constraints.max_tool_calls,
            allow_write_actions=base_constraints.allow_write_actions,
            allow_network_actions=base_constraints.allow_network_actions,
            allow_browser_actions=base_constraints.allow_browser_actions,
            allow_code_execution=base_constraints.allow_code_execution,
            require_approval_on_high_risk=base_constraints.require_approval_on_high_risk,
            enable_security_scan=base_constraints.enable_security_scan,
            enable_dynamic_discovery=base_constraints.enable_dynamic_discovery,
            auto_recipe=candidate.auto_recipe,
            security_strictness=base_constraints.security_strictness,
            enable_live_agent=base_constraints.enable_live_agent,
            max_live_agent_calls=base_constraints.max_live_agent_calls,
            live_agent_temperature=base_constraints.live_agent_temperature,
            live_agent_timeout_seconds=base_constraints.live_agent_timeout_seconds,
            live_agent_fail_open=base_constraints.live_agent_fail_open,
            blocked_tools=list(base_constraints.blocked_tools),
        )

    @staticmethod
    def _candidate_label(candidate: OptimizationCandidate) -> str:
        recipe = candidate.recipe or ("auto" if candidate.auto_recipe else "none")
        return f"{candidate.mode}:{recipe}"
