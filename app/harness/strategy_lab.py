"""Iterative strategy lab for live-agent prompt profile optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.harness.live_strategy import LiveStrategyRegistry, compute_efficiency
from app.harness.models import HarnessConstraints

if TYPE_CHECKING:  # pragma: no cover
    from app.harness.engine import HarnessEngine


@dataclass
class StrategyLabConfig:
    """Config for strategy profile tournament under strict call budget."""

    mode: str = "balanced"
    recipe: str = ""
    max_total_calls: int = 40
    max_calls_per_query: int = 6
    max_queries: int = 4
    strategies: list[str] = field(default_factory=list)

    def normalize(self) -> StrategyLabConfig:
        self.max_total_calls = max(1, min(int(self.max_total_calls), 50))
        self.max_calls_per_query = max(1, min(int(self.max_calls_per_query), 50))
        self.max_queries = max(1, min(int(self.max_queries), 20))
        cleaned: list[str] = []
        seen = set()
        for name in self.strategies:
            key = str(name).strip()
            if key and key not in seen:
                cleaned.append(key)
                seen.add(key)
        self.strategies = cleaned
        return self


class HarnessStrategyLab:
    """Runs multi-strategy experiments and selects champion profile."""

    def __init__(self) -> None:
        self.registry = LiveStrategyRegistry()

    def run(
        self,
        engine: HarnessEngine,
        queries: list[str],
        live_model: dict[str, Any] | None = None,
        config: StrategyLabConfig | None = None,
        constraints: HarnessConstraints | None = None,
    ) -> dict[str, Any]:
        cfg = (config or StrategyLabConfig()).normalize()
        active_queries = [item.strip() for item in queries if item.strip()][: cfg.max_queries]
        if not active_queries:
            return {
                "count": 0,
                "config": self._config_payload(cfg, []),
                "leaderboard": [],
                "champion": {},
                "summary": {"note": "no_queries"},
            }

        strategy_names = self._resolve_strategies(cfg.strategies)
        if not strategy_names:
            return {
                "count": 0,
                "config": self._config_payload(cfg, []),
                "leaderboard": [],
                "champion": {},
                "summary": {"note": "no_valid_strategies"},
            }

        baseline_values = self._run_baseline(
            engine=engine,
            queries=active_queries,
            mode=cfg.mode,
            recipe=cfg.recipe,
            constraints=constraints,
        )

        total_calls = 0
        board: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] = []
        budget_exhausted = False

        for strategy in strategy_names:
            query_rows: list[dict[str, Any]] = []
            for query in active_queries:
                remaining_budget = cfg.max_total_calls - total_calls
                if remaining_budget <= 0:
                    budget_exhausted = True
                    break

                live_constraints = self._clone_constraints(constraints)
                live_constraints.enable_live_agent = True
                live_constraints.max_live_agent_calls = min(cfg.max_calls_per_query, remaining_budget)
                run = engine.run(
                    query=query,
                    mode=cfg.mode,
                    recipe=cfg.recipe or None,
                    constraints=live_constraints,
                    live_model=live_model,
                    live_strategy=strategy,
                    apply_champion_strategy=False,
                )
                card = engine.build_value_card(run)
                value_index = float(card.get("value_index", 0.0))
                live_meta = run.metadata.get("live_agent", {})
                calls_used = int(live_meta.get("calls_used", 0)) if isinstance(live_meta, dict) else 0
                total_calls += calls_used
                baseline = float(baseline_values.get(query, 0.0))
                delta = value_index - baseline

                row = {
                    "strategy": strategy,
                    "query": query,
                    "baseline_value_index": round(baseline, 2),
                    "value_index": round(value_index, 2),
                    "delta": round(delta, 2),
                    "calls_used": calls_used,
                    "live_success": bool(live_meta.get("success", False)) if isinstance(live_meta, dict) else False,
                }
                query_rows.append(row)
                raw_rows.append(row)

            board.append(self._summarize_strategy(strategy=strategy, rows=query_rows))
            if budget_exhausted:
                break

        board.sort(key=lambda item: float(item.get("champion_score", 0.0)), reverse=True)
        champion = board[0] if board else {}
        frontier = self._pareto_frontier(board)

        return {
            "count": len(raw_rows),
            "config": self._config_payload(cfg, strategy_names),
            "leaderboard": board,
            "champion": champion,
            "rows": raw_rows,
            "frontier": frontier,
            "summary": {
                "queries": len(active_queries),
                "strategies_tested": len(board),
                "budget_total": cfg.max_total_calls,
                "total_calls_used": total_calls,
                "budget_exhausted": budget_exhausted,
                "recommendation": self._recommend(champion, board),
            },
        }

    def _resolve_strategies(self, requested: list[str]) -> list[str]:
        if not requested:
            requested = [
                "balanced_orchestrator",
                "risk_sentinel",
                "systems_architect",
                "decision_theater",
            ]
        out = []
        for name in requested:
            if self.registry.get(name):
                out.append(name)
        return out

    def _run_baseline(
        self,
        engine: HarnessEngine,
        queries: list[str],
        mode: str,
        recipe: str,
        constraints: HarnessConstraints | None,
    ) -> dict[str, float]:
        baseline: dict[str, float] = {}
        for query in queries:
            base_constraints = self._clone_constraints(constraints)
            base_constraints.enable_live_agent = False
            run = engine.run(
                query=query,
                mode=mode,
                recipe=recipe or None,
                constraints=base_constraints,
                apply_champion_strategy=False,
            )
            card = engine.build_value_card(run)
            baseline[query] = float(card.get("value_index", 0.0))
        return baseline

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
    def _summarize_strategy(strategy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "strategy": strategy,
                "tested_queries": 0,
                "avg_value_index": 0.0,
                "avg_delta_value_index": 0.0,
                "avg_calls_used": 0.0,
                "success_rate": 0.0,
                "efficiency": 0.0,
                "champion_score": -999.0,
            }

        avg_value = sum(float(item.get("value_index", 0.0)) for item in rows) / len(rows)
        avg_delta = sum(float(item.get("delta", 0.0)) for item in rows) / len(rows)
        avg_calls = sum(float(item.get("calls_used", 0.0)) for item in rows) / len(rows)
        success = sum(1 for item in rows if item.get("live_success")) / len(rows)
        efficiency = compute_efficiency(avg_delta, avg_calls)

        # Champion score intentionally favors value gains, then efficiency, then reliability.
        champion_score = round(4.2 * avg_delta + 7.0 * efficiency + 10.0 * success + avg_value * 0.08, 4)
        return {
            "strategy": strategy,
            "tested_queries": len(rows),
            "avg_value_index": round(avg_value, 3),
            "avg_delta_value_index": round(avg_delta, 3),
            "avg_calls_used": round(avg_calls, 3),
            "success_rate": round(success, 4),
            "efficiency": efficiency,
            "champion_score": champion_score,
        }

    @staticmethod
    def _pareto_frontier(board: list[dict[str, Any]]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for row in board:
            points.append(
                {
                    "strategy": row.get("strategy", ""),
                    "delta": float(row.get("avg_delta_value_index", 0.0)),
                    "avg_calls": float(row.get("avg_calls_used", 0.0)),
                    "efficiency": float(row.get("efficiency", 0.0)),
                }
            )

        frontier: list[dict[str, Any]] = []
        for item in points:
            dominated = False
            for other in points:
                if other is item:
                    continue
                dominates = (
                    other["delta"] >= item["delta"]
                    and other["efficiency"] >= item["efficiency"]
                    and other["avg_calls"] <= item["avg_calls"]
                    and (
                        other["delta"] > item["delta"]
                        or other["efficiency"] > item["efficiency"]
                        or other["avg_calls"] < item["avg_calls"]
                    )
                )
                if dominates:
                    dominated = True
                    break
            if not dominated:
                frontier.append(item)

        frontier.sort(key=lambda item: (item["delta"], item["efficiency"]), reverse=True)
        return frontier

    @staticmethod
    def _recommend(champion: dict[str, Any], board: list[dict[str, Any]]) -> str:
        if not board:
            return "No strategy runs were completed."
        if not champion:
            return "No clear champion. Re-run with tighter scope and cleaner queries."
        if float(champion.get("avg_delta_value_index", 0.0)) >= 2.0:
            return f"Adopt strategy={champion.get('strategy')} for this mode and continue periodic lab refresh."
        if float(champion.get("avg_delta_value_index", 0.0)) >= 0.5:
            return f"Use strategy={champion.get('strategy')} for critical flows and keep baseline fallback."
        return "No strong winner. Revisit prompts and strategy set before enabling globally."

    @staticmethod
    def _config_payload(cfg: StrategyLabConfig, active_strategies: list[str]) -> dict[str, Any]:
        return {
            "mode": cfg.mode,
            "recipe": cfg.recipe,
            "max_total_calls": cfg.max_total_calls,
            "max_calls_per_query": cfg.max_calls_per_query,
            "max_queries": cfg.max_queries,
            "strategies": active_strategies,
        }
