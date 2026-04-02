"""Harness engine orchestrating planner, tools, memory, guardrails, and eval."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.core.state import GraphState
from app.graph import build_graph
from app.harness.discovery import DiscoveredTool, ToolDiscoveryEngine
from app.harness.evaluator import HarnessEvaluator
from app.harness.guardrails import GuardrailEngine
from app.harness.iteration import LiveIterationTracker
from app.harness.live_agent import LiveAgentOrchestrator
from app.harness.live_experiment import HarnessLiveExperiment, LiveExperimentConfig
from app.harness.manifest import ToolManifestRegistry
from app.harness.models import HarnessConstraints, HarnessRun, HarnessStep, ToolCall
from app.harness.optimizer import HarnessOptimizer
from app.harness.planner import HarnessPlanner
from app.harness.presentation import PresentationBlueprintBuilder
from app.harness.recipes import HarnessRecipe, RecipeRegistry
from app.harness.redteam import HarnessRedTeam
from app.harness.report import HarnessReportBuilder
from app.harness.security import SecurityAction, SecurityDecision, SecurityEngine
from app.harness.showcase import HarnessShowcaseBuilder
from app.harness.stream import HarnessEventStreamBuilder
from app.harness.state import HarnessMemoryStore
from app.harness.tools import ToolRegistry
from app.harness.value import HarnessValueScorer
from app.harness.visuals import HarnessVisualProtocol


class HarnessEngine:
    """Top-level harness runner for reliable agent execution."""

    def __init__(self) -> None:
        self.graph = build_graph()
        self.planner = HarnessPlanner()
        self.tools = ToolRegistry()
        self.memory = HarnessMemoryStore()
        self.guardrails = GuardrailEngine()
        self.evaluator = HarnessEvaluator()
        self.live_agent = LiveAgentOrchestrator()
        self.live_experiment = HarnessLiveExperiment()
        self.iteration = LiveIterationTracker()

        self.manifests = ToolManifestRegistry()
        self.discovery = ToolDiscoveryEngine(self.manifests)
        self.security = SecurityEngine()
        self.recipes = RecipeRegistry()
        self.redteam = HarnessRedTeam()
        self.reporter = HarnessReportBuilder()
        self.value = HarnessValueScorer()
        self.visuals = HarnessVisualProtocol()
        self.showcase = HarnessShowcaseBuilder()
        self.presentation = PresentationBlueprintBuilder()
        self.stream = HarnessEventStreamBuilder()
        self.optimizer = HarnessOptimizer()

    def list_tool_catalog(self) -> list[dict[str, Any]]:
        """Return tool manifest catalog and runtime availability."""

        available = set(self.tools.available_tools())
        catalog = []
        for item in self.manifests.list_all():
            payload = item.to_dict()
            payload["available"] = item.name in available
            catalog.append(payload)
        return catalog

    def discover_tools(
        self,
        query: str,
        constraints: HarnessConstraints | None = None,
        mode: str = "balanced",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Discover/rank tools for a query."""

        effective = constraints or HarnessConstraints()
        ranked = self.discovery.discover(
            query=query,
            constraints=effective,
            mode=mode,
            limit=limit,
            available_tools=set(self.tools.available_tools()),
        )
        return [item.to_dict() for item in ranked]

    def list_recipes(self) -> list[dict[str, Any]]:
        """Return recipe cards."""

        return self.recipes.list_recipe_cards()

    def run_recipe(
        self,
        query: str,
        recipe: str | None = None,
        recipe_path: str | None = None,
        constraints: HarnessConstraints | None = None,
        mode: str = "balanced",
        live_model: dict[str, Any] | None = None,
    ) -> HarnessRun:
        """Run harness in recipe-driven mode."""

        return self.run(
            query=query,
            constraints=constraints,
            mode=mode,
            recipe=recipe,
            recipe_path=recipe_path,
            live_model=live_model,
        )

    def run_redteam(
        self,
        mode: str = "balanced",
        constraints: HarnessConstraints | None = None,
        include_runs: bool = False,
    ) -> dict[str, Any]:
        """Run harness red-team suite."""

        return self.redteam.run(
            engine=self,
            mode=mode,
            constraints=constraints,
            include_runs=include_runs,
        )

    def build_value_card(self, run: HarnessRun) -> dict[str, Any]:
        """Build value card used for demo-level storytelling and ranking."""

        return self.value.score_run(run, manifests=self.manifests)

    def build_visual_payload(
        self,
        run: HarnessRun,
        value_card: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build front-end ready visualization payload."""

        effective_card = value_card or self.build_value_card(run)
        payload = self.visuals.build_run_payload(
            run=run,
            value_card=effective_card,
            manifests=self.manifests,
        )
        payload["first_screen_blueprint"] = self.presentation.build_first_screen(payload)
        payload["event_stream"] = self.stream.build(run, payload)
        return payload

    def build_first_screen_blueprint(self, visual_payload: dict[str, Any]) -> dict[str, Any]:
        """Build first-screen dashboard blueprint from visual payload."""

        return self.presentation.build_first_screen(visual_payload)

    def list_showcase_packs(self) -> list[dict[str, Any]]:
        """List built-in showcase packs."""

        return self.showcase.list_packs()

    def run_showcase(
        self,
        pack_name: str = "impact-lens",
        mode_override: str = "",
        constraints: HarnessConstraints | None = None,
        live_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run scenario pack and return comparative showcase payload."""

        return self.showcase.run_pack(
            engine=self,
            pack_name=pack_name,
            mode_override=mode_override,
            constraints=constraints,
            live_model=live_model,
        )

    def optimize_query(
        self,
        query: str,
        constraints: HarnessConstraints | None = None,
        live_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Auto-tune mode/recipe combinations for this query."""

        return self.optimizer.optimize(
            engine=self,
            query=query,
            constraints=constraints,
            live_model=live_model,
        )

    def run_live_experiment(
        self,
        queries: list[str],
        mode: str = "balanced",
        recipe: str = "",
        live_model: dict[str, Any] | None = None,
        max_total_calls: int = 30,
        max_calls_per_query: int = 8,
        constraints: HarnessConstraints | None = None,
    ) -> dict[str, Any]:
        """Run baseline vs live-agent A/B experiment with strict call limits."""

        config = LiveExperimentConfig(
            mode=mode,
            recipe=recipe,
            max_total_calls=max_total_calls,
            max_calls_per_query=max_calls_per_query,
            limit_queries=len(queries) if queries else 0,
        )
        payload = self.live_experiment.run(
            engine=self,
            queries=queries,
            live_model=live_model,
            config=config,
            constraints=constraints,
        )
        model_info = {
            "base_url": str((live_model or {}).get("base_url", "")),
            "model_name": str((live_model or {}).get("model_name", "")),
        }
        self.iteration.record(payload, model_info=model_info)
        payload["history"] = {"latest": self.iteration.latest(limit=6)}
        return payload

    def list_live_experiment_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent live experiment records."""

        return self.iteration.latest(limit=limit)

    def build_report(self, run: HarnessRun, fmt: str = "markdown") -> str | dict[str, Any]:
        """Build markdown/json report from one run."""

        if fmt == "json":
            return self.reporter.summary(run)
        return self.reporter.to_markdown(run)

    def run(
        self,
        query: str,
        constraints: HarnessConstraints | None = None,
        mode: str = "balanced",
        recipe: str | None = None,
        recipe_path: str | None = None,
        live_model: dict[str, Any] | None = None,
    ) -> HarnessRun:
        """Run harness loop around the core agent graph."""

        constraints = constraints or HarnessConstraints()
        session_id = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
        previous_context = self.memory.read_recent(session_id, limit=8)

        preflight = self.security.preflight(query, constraints)
        active_recipe = self._resolve_recipe(
            query=query,
            constraints=constraints,
            recipe_name=recipe,
            recipe_path=recipe_path,
        )

        if preflight.action == SecurityAction.BLOCK:
            return self._build_preflight_block_run(
                query=query,
                mode=mode,
                session_id=session_id,
                previous_context=previous_context,
                preflight=preflight,
                active_recipe=active_recipe,
            )

        safe_query = preflight.redacted_query or query
        graph_result = self.graph.invoke(GraphState(query=safe_query, system_mode=mode))
        payload: dict[str, Any] = graph_result if isinstance(graph_result, dict) else graph_result.model_dump()
        plan = self.planner.build_plan(safe_query)
        if active_recipe:
            plan.append(f"recipe:{active_recipe.name}")

        discovered = self.discovery.discover(
            query=safe_query,
            constraints=constraints,
            mode=mode,
            limit=max(constraints.max_tool_calls + 4, 6),
            available_tools=set(self.tools.available_tools()),
        )
        discovery_trace = [item.to_dict() for item in discovered]
        discovery_map = {item.name: item for item in discovered}

        steps: list[HarnessStep] = []
        session_events: list[dict[str, Any]] = list(previous_context)
        used_tools: set[str] = set()
        high_risk = payload.get("risk_level") in {"high", "critical"}
        tool_calls = 0
        recipe_executed_steps = 0
        tool_security_decisions: list[dict[str, Any]] = []

        step_idx = 1
        if active_recipe:
            for recipe_step in active_recipe.steps:
                if step_idx > constraints.max_steps or tool_calls >= constraints.max_tool_calls:
                    break
                if not recipe_step.applicable(safe_query):
                    continue

                recipe_executed_steps += 1
                candidate_tools = [recipe_step.tool] + recipe_step.fallback_tools
                attempt_success = False

                for candidate in candidate_tools:
                    if step_idx > constraints.max_steps or tool_calls >= constraints.max_tool_calls:
                        break

                    discovered_item = discovery_map.get(candidate)
                    score = discovered_item.score if discovered_item else 0.0
                    tool_call = self.discovery.build_tool_call(
                        tool_name=candidate,
                        query=safe_query,
                        step=step_idx,
                        args_override=recipe_step.args,
                        source=f"recipe:{active_recipe.name}",
                        score=score,
                    )
                    if not tool_call:
                        if candidate == recipe_step.tool and not recipe_step.optional:
                            steps.append(
                                HarnessStep(
                                    step=step_idx,
                                    thought=f"recipe {recipe_step.step_id}: {recipe_step.title}",
                                    decision=f"missing tool {candidate}",
                                    guardrail_notes=[f"BLOCK:unknown_recipe_tool:{candidate}"],
                                )
                            )
                            step_idx += 1
                        continue

                    self._augment_tool_args(tool_call, safe_query, session_events)
                    step = self._execute_step(
                        step=step_idx,
                        query=safe_query,
                        tool_call=tool_call,
                        constraints=constraints,
                        high_risk=high_risk,
                        preflight_action=preflight.action,
                        discovery_item=discovered_item,
                        thought=f"recipe {recipe_step.step_id}: {recipe_step.title}",
                        decision=f"use {candidate}",
                    )
                    steps.append(step)
                    tool_security_decisions.append(step.security)
                    used_tools.add(candidate)
                    step_idx += 1

                    if step.tool_result:
                        tool_calls += 1
                        event = self._build_memory_event(step.step, step.tool_call, step.tool_result)
                        self.memory.append_event(session_id, event)
                        session_events.append(event)
                        attempt_success = True
                        break

                if not attempt_success and not recipe_step.optional:
                    break
        else:
            while step_idx <= constraints.max_steps and tool_calls < constraints.max_tool_calls:
                tool_call: ToolCall | None = None
                discovered_item: DiscoveredTool | None = None

                if constraints.enable_dynamic_discovery:
                    tool_call = self.discovery.recommend_for_step(
                        query=safe_query,
                        step=step_idx,
                        discovered=discovered,
                        used_tools=used_tools,
                        plan=plan,
                    )
                    if tool_call:
                        discovered_item = discovery_map.get(tool_call.name)

                if not tool_call:
                    fallback = self.planner.next_tool_call(query=safe_query, step=step_idx, plan=plan)
                    if not fallback:
                        break
                    tool_call = fallback
                    discovered_item = discovery_map.get(tool_call.name)

                self._augment_tool_args(tool_call, safe_query, session_events)
                step = self._execute_step(
                    step=step_idx,
                    query=safe_query,
                    tool_call=tool_call,
                    constraints=constraints,
                    high_risk=high_risk,
                    preflight_action=preflight.action,
                    discovery_item=discovered_item,
                    thought=f"step-{step_idx}: choose tool for plan segment",
                    decision=f"use {tool_call.name}",
                )
                steps.append(step)
                tool_security_decisions.append(step.security)
                used_tools.add(tool_call.name)
                step_idx += 1

                if step.tool_result:
                    tool_calls += 1
                    event = self._build_memory_event(step.step, step.tool_call, step.tool_result)
                    self.memory.append_event(session_id, event)
                    session_events.append(event)
                elif any(note.startswith("BLOCK") for note in step.guardrail_notes):
                    break

        final_answer = self._compose_final_answer(
            payload=payload,
            steps=steps,
            preflight=preflight,
            discovery=discovery_trace,
            active_recipe=active_recipe,
        )
        live_overrides = dict(live_model or {})
        live_overrides.setdefault("timeout_seconds", constraints.live_agent_timeout_seconds)
        live_overrides.setdefault("temperature", constraints.live_agent_temperature)
        live_result = self.live_agent.enhance(
            query=safe_query,
            mode=mode,
            base_answer=final_answer,
            plan=plan,
            steps=[self._step_to_dict(item) for item in steps],
            discovery=discovery_trace,
            max_calls=max(1, min(constraints.max_live_agent_calls, 50)),
            temperature=constraints.live_agent_temperature,
            live_model_overrides=live_overrides,
        ) if constraints.enable_live_agent else None

        if live_result and live_result.success and live_result.enhanced_answer:
            final_answer = live_result.enhanced_answer
        elif live_result and not live_result.success and not constraints.live_agent_fail_open:
            final_answer = (
                "Live agent enhancement failed and fail-open is disabled.\n"
                "Please retry with a valid model config or enable fail-open."
            )

        run = HarnessRun(
            query=query,
            plan=plan,
            steps=steps,
            final_answer=final_answer,
            completed=not (
                live_result
                and constraints.enable_live_agent
                and not live_result.success
                and not constraints.live_agent_fail_open
            ),
            eval_metrics={},
            memory_snapshot=previous_context,
            metadata={
                "session_id": session_id,
                "mode": mode,
                "risk_level": payload.get("risk_level", "unknown"),
                "selected_agent": payload.get("agent_name", ""),
                "selected_skills": payload.get("selected_skills", []),
                "security": {
                    "preflight_action": preflight.action.value,
                    "preflight_risk_score": preflight.risk_score,
                    "preflight_findings": [item.to_dict() for item in preflight.findings],
                    "tool_decisions": tool_security_decisions,
                },
                "discovery": discovery_trace,
                "recipe": {
                    "name": active_recipe.name if active_recipe else "",
                    "version": active_recipe.version if active_recipe else "",
                    "total_steps": len(active_recipe.steps) if active_recipe else 0,
                    "executed_steps": recipe_executed_steps,
                },
                "live_agent": live_result.to_dict() if live_result else {
                    "enabled": False,
                    "configured": False,
                    "calls_used": 0,
                    "call_budget": 0,
                    "success": False,
                    "notes": [],
                    "errors": [],
                },
            },
        )
        run.eval_metrics = self.evaluator.evaluate(run)
        run.metadata["value_card"] = self.build_value_card(run)
        run.metadata["visual_hooks"] = run.metadata["value_card"].get("visual_hooks", [])
        return run

    @staticmethod
    def run_to_dict(run: HarnessRun) -> dict[str, Any]:
        """Convert run dataclass to JSON-serializable dict."""

        return asdict(run)

    def eval_suite(self, queries: list[str], mode: str = "balanced") -> dict[str, Any]:
        """Run harness eval over multiple queries."""

        if not queries:
            return {"count": 0, "avg": {}}

        runs = [self.run(query=item, mode=mode) for item in queries]
        metrics = [run.eval_metrics for run in runs]
        keys = sorted({key for metric in metrics for key in metric.keys()})

        avg: dict[str, float] = {}
        for key in keys:
            avg[key] = round(sum(float(metric.get(key, 0.0)) for metric in metrics) / len(metrics), 4)

        return {
            "count": len(runs),
            "avg": avg,
            "queries": queries,
        }

    def _resolve_recipe(
        self,
        query: str,
        constraints: HarnessConstraints,
        recipe_name: str | None = None,
        recipe_path: str | None = None,
    ) -> HarnessRecipe | None:
        if recipe_path:
            loaded = self.recipes.load_from_file(Path(recipe_path))
            return loaded
        if recipe_name:
            return self.recipes.get(recipe_name)
        if constraints.auto_recipe:
            return self.recipes.suggest(query)
        return None

    def _execute_step(
        self,
        step: int,
        query: str,
        tool_call: ToolCall,
        constraints: HarnessConstraints,
        high_risk: bool,
        preflight_action: SecurityAction,
        discovery_item: DiscoveredTool | None,
        thought: str,
        decision: str,
    ) -> HarnessStep:
        guardrail_notes = self.guardrails.check_tool_call(
            tool_call=tool_call,
            constraints=constraints,
            step=step,
            high_risk=high_risk,
        )

        manifest = self.manifests.get(tool_call.name)
        security_decision = self.security.evaluate_tool_call(
            tool_call=tool_call,
            constraints=constraints,
            manifest=manifest,
            high_risk=high_risk,
            preflight_action=preflight_action,
        )
        guardrail_notes.extend(security_decision.to_guardrail_notes())

        blocked = security_decision.action == SecurityAction.BLOCK or any(
            note.startswith("BLOCK") for note in guardrail_notes
        )
        if blocked:
            return HarnessStep(
                step=step,
                thought=thought,
                decision=decision,
                tool_call=tool_call,
                tool_result=None,
                guardrail_notes=guardrail_notes,
                discovery_notes=discovery_item.reasons if discovery_item else [],
                security=security_decision.to_dict(),
            )

        result = self.tools.call(tool_call)
        guardrail_notes.extend(self.guardrails.check_tool_result(result))
        if security_decision.action == SecurityAction.CHALLENGE:
            guardrail_notes.append("WARN:security_challenge_continue")

        return HarnessStep(
            step=step,
            thought=thought,
            decision=decision,
            tool_call=tool_call,
            tool_result=result,
            guardrail_notes=guardrail_notes,
            discovery_notes=discovery_item.reasons if discovery_item else [],
            security=security_decision.to_dict(),
        )

    @staticmethod
    def _augment_tool_args(tool_call: ToolCall, query: str, session_events: list[dict[str, Any]]) -> None:
        if tool_call.name == "memory_context_digest":
            tool_call.args.setdefault("events", session_events)
        if "query" not in tool_call.args:
            tool_call.args["query"] = query

    @staticmethod
    def _build_memory_event(step: int, tool_call: ToolCall | None, tool_result: Any) -> dict[str, Any]:
        tool_name = tool_call.name if tool_call else ""
        tool_score = tool_call.score if tool_call else 0.0
        source = tool_call.source if tool_call else "unknown"
        return {
            "step": step,
            "tool": tool_name,
            "source": source,
            "score": round(float(tool_score), 4),
            "success": bool(tool_result.success),
            "latency_ms": round(float(tool_result.latency_ms), 2),
        }

    @staticmethod
    def _step_to_dict(step: HarnessStep) -> dict[str, Any]:
        return {
            "step": step.step,
            "thought": step.thought,
            "decision": step.decision,
            "tool_call": {
                "name": step.tool_call.name if step.tool_call else "",
                "source": step.tool_call.source if step.tool_call else "",
                "score": round(float(step.tool_call.score), 4) if step.tool_call else 0.0,
            },
            "tool_result": {
                "success": bool(step.tool_result.success),
                "latency_ms": round(float(step.tool_result.latency_ms), 2),
            }
            if step.tool_result
            else {},
            "guardrail_notes": list(step.guardrail_notes),
            "discovery_notes": list(step.discovery_notes),
            "security": dict(step.security),
        }

    @staticmethod
    def _compose_final_answer(
        payload: dict[str, Any],
        steps: list[HarnessStep],
        preflight: SecurityDecision,
        discovery: list[dict[str, Any]],
        active_recipe: HarnessRecipe | None,
    ) -> str:
        tool_summaries = []
        for step in steps:
            if step.tool_result:
                tool_summaries.append(
                    f"- {step.tool_result.name}: "
                    f"{'OK' if step.tool_result.success else 'ERR'} "
                    f"({step.tool_result.latency_ms:.1f}ms)"
                )

        top_tools: list[str] = []
        for item in discovery[:3]:
            name = item.get("name")
            if isinstance(name, str):
                top_tools.append(name)

        notes: list[str] = [
            f"- security preflight: {preflight.action.value} (score={preflight.risk_score:.2f})",
            f"- discovered tools: {', '.join(top_tools) if top_tools else 'none'}",
            f"- recipe: {active_recipe.name if active_recipe else 'none'}",
        ]
        if tool_summaries:
            notes.extend(tool_summaries)
        else:
            notes.append("- no harness tools executed")

        return (
            f"{payload.get('final_output', '')}\n\n"
            "Harness Execution Notes:\n"
            f"{chr(10).join(notes)}"
        )

    def _build_preflight_block_run(
        self,
        query: str,
        mode: str,
        session_id: str,
        previous_context: list[dict[str, Any]],
        preflight: SecurityDecision,
        active_recipe: HarnessRecipe | None,
    ) -> HarnessRun:
        step = HarnessStep(
            step=1,
            thought="preflight security scan",
            decision="blocked before tool execution",
            guardrail_notes=preflight.to_guardrail_notes(),
            security=preflight.to_dict(),
        )
        final_answer = (
            "Request blocked by harness security preflight.\n"
            "Please remove prompt-injection/destructive instructions and retry with a safer query."
        )
        run = HarnessRun(
            query=query,
            plan=["security preflight"],
            steps=[step],
            final_answer=final_answer,
            completed=False,
            eval_metrics={},
            memory_snapshot=previous_context,
            metadata={
                "session_id": session_id,
                "mode": mode,
                "security": {
                    "preflight_action": preflight.action.value,
                    "preflight_risk_score": preflight.risk_score,
                    "preflight_findings": [item.to_dict() for item in preflight.findings],
                    "tool_decisions": [],
                },
                "discovery": [],
                "recipe": {
                    "name": active_recipe.name if active_recipe else "",
                    "version": active_recipe.version if active_recipe else "",
                    "total_steps": len(active_recipe.steps) if active_recipe else 0,
                    "executed_steps": 0,
                },
                "live_agent": {
                    "enabled": False,
                    "configured": False,
                    "calls_used": 0,
                    "call_budget": 0,
                    "success": False,
                    "notes": [],
                    "errors": [],
                },
            },
        )
        run.eval_metrics = self.evaluator.evaluate(run)
        run.metadata["value_card"] = self.build_value_card(run)
        run.metadata["visual_hooks"] = run.metadata["value_card"].get("visual_hooks", [])
        self.memory.append_event(
            session_id,
            {
                "step": 0,
                "tool": "preflight",
                "source": "security",
                "score": preflight.risk_score,
                "success": False,
                "latency_ms": 0.0,
                "blocked": True,
            },
        )
        return run
