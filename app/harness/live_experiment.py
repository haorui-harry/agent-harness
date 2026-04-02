"""A/B experiment runner for baseline vs live-agent harness runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.harness.models import HarnessConstraints

if TYPE_CHECKING:  # pragma: no cover
    from app.harness.engine import HarnessEngine


@dataclass
class LiveExperimentConfig:
    """Experiment configuration with strict call limits."""

    mode: str = "balanced"
    recipe: str = ""
    max_total_calls: int = 30
    max_calls_per_query: int = 8
    limit_queries: int = 6

    def normalize(self) -> LiveExperimentConfig:
        self.max_total_calls = max(1, min(int(self.max_total_calls), 50))
        self.max_calls_per_query = max(1, min(int(self.max_calls_per_query), 50))
        self.limit_queries = max(1, min(int(self.limit_queries), 20))
        return self


class HarnessLiveExperiment:
    """Run side-by-side baseline/live trials and summarize value deltas."""

    def run(
        self,
        engine: HarnessEngine,
        queries: list[str],
        live_model: dict[str, Any] | None = None,
        config: LiveExperimentConfig | None = None,
        constraints: HarnessConstraints | None = None,
    ) -> dict[str, Any]:
        exp = (config or LiveExperimentConfig()).normalize()
        active_queries = [item.strip() for item in queries if item.strip()][: exp.limit_queries]
        if not active_queries:
            return {"count": 0, "rows": [], "summary": {"note": "no_queries"}}

        total_calls = 0
        rows: list[dict[str, Any]] = []
        stopped_early = False

        for query in active_queries:
            base_constraints = self._clone_constraints(constraints)
            base_constraints.enable_live_agent = False
            baseline = engine.run(
                query=query,
                mode=exp.mode,
                recipe=exp.recipe or None,
                constraints=base_constraints,
            )
            baseline_card = engine.build_value_card(baseline)
            baseline_value = float(baseline_card.get("value_index", 0.0))

            remaining_budget = exp.max_total_calls - total_calls
            if remaining_budget <= 0:
                rows.append(
                    {
                        "query": query,
                        "baseline_value_index": baseline_value,
                        "live_value_index": None,
                        "delta": None,
                        "calls_used": 0,
                        "status": "skipped_budget_exhausted",
                    }
                )
                stopped_early = True
                break

            live_constraints = self._clone_constraints(constraints)
            live_constraints.enable_live_agent = True
            live_constraints.max_live_agent_calls = min(exp.max_calls_per_query, remaining_budget)
            live = engine.run(
                query=query,
                mode=exp.mode,
                recipe=exp.recipe or None,
                constraints=live_constraints,
                live_model=live_model,
            )
            live_card = engine.build_value_card(live)
            live_value = float(live_card.get("value_index", 0.0))
            live_meta = live.metadata.get("live_agent", {})
            calls_used = int(live_meta.get("calls_used", 0)) if isinstance(live_meta, dict) else 0
            total_calls += calls_used

            rows.append(
                {
                    "query": query,
                    "baseline_value_index": round(baseline_value, 2),
                    "live_value_index": round(live_value, 2),
                    "delta": round(live_value - baseline_value, 2),
                    "calls_used": calls_used,
                    "live_success": bool(live_meta.get("success", False)) if isinstance(live_meta, dict) else False,
                    "live_model": str(live_meta.get("model", "")) if isinstance(live_meta, dict) else "",
                    "status": "ok" if calls_used > 0 else "no_live_calls",
                }
            )

        deltas = [
            float(row["delta"])
            for row in rows
            if isinstance(row.get("delta"), (int, float))
            and isinstance(row.get("baseline_value_index"), (int, float))
            and row.get("live_value_index") is not None
            and float(row.get("calls_used", 0.0)) > 0.0
        ]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

        return {
            "count": len(rows),
            "config": {
                "mode": exp.mode,
                "recipe": exp.recipe,
                "max_total_calls": exp.max_total_calls,
                "max_calls_per_query": exp.max_calls_per_query,
                "limit_queries": exp.limit_queries,
            },
            "rows": rows,
            "summary": {
                "avg_delta_value_index": round(avg_delta, 3),
                "total_calls_used": total_calls,
                "budget_exhausted": total_calls >= exp.max_total_calls,
                "stopped_early": stopped_early,
                "recommendation": self._recommend(avg_delta, total_calls, exp.max_total_calls),
            },
        }

    @staticmethod
    def _clone_constraints(source: HarnessConstraints | None) -> HarnessConstraints:
        if source is None:
            return HarnessConstraints()
        return HarnessConstraints(
            max_steps=source.max_steps,
            max_tool_calls=source.max_tool_calls,
            allow_write_actions=source.allow_write_actions,
            allow_network_actions=source.allow_network_actions,
            allow_browser_actions=source.allow_browser_actions,
            allow_code_execution=source.allow_code_execution,
            require_approval_on_high_risk=source.require_approval_on_high_risk,
            enable_security_scan=source.enable_security_scan,
            enable_dynamic_discovery=source.enable_dynamic_discovery,
            auto_recipe=source.auto_recipe,
            security_strictness=source.security_strictness,
            enable_live_agent=source.enable_live_agent,
            max_live_agent_calls=source.max_live_agent_calls,
            live_agent_temperature=source.live_agent_temperature,
            live_agent_timeout_seconds=source.live_agent_timeout_seconds,
            live_agent_fail_open=source.live_agent_fail_open,
            blocked_tools=list(source.blocked_tools),
        )

    @staticmethod
    def _recommend(avg_delta: float, used_calls: int, max_calls: int) -> str:
        if used_calls <= 0:
            return "No live calls were used. Check model config and retry."
        if avg_delta >= 3.0:
            return "Live agent clearly improves value index; keep it enabled for critical flows."
        if avg_delta >= 1.0:
            return "Live agent provides moderate gains; enable for deep or safety-critical modes."
        if avg_delta >= 0.0:
            return "Gains are small; tighten prompts or reduce call budget."
        if used_calls >= max_calls:
            return "Value dropped before budget exhausted; revise prompts and retry with smaller batches."
        return "Live agent underperformed; fallback to baseline and iterate prompt strategy."
