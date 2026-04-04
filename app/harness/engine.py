"""Harness engine orchestrating planner, tools, memory, guardrails, and eval."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.agents.runtime import AgentThreadRuntime
from app.agents.scheduler import AgentExecutionScheduler
from app.agents.subagents import ParallelSubagentExecutor
from app.agents.task_actions import TaskGraphActionMapper
from app.agents.workspace_view import ThreadWorkspaceStreamBuilder
from app.core.state import GraphState
from app.graph import build_graph
from app.core.mission import MissionRegistry
from app.harness.code_mission import CodeMissionPackBuilder
from app.harness.deep_research import HarnessDeepResearchBuilder
from app.harness.discovery import DiscoveredTool, ToolDiscoveryEngine
from app.harness.evaluator import HarnessEvaluator
from app.harness.guardrails import GuardrailEngine
from app.harness.iteration import LiveIterationTracker
from app.harness.live_agent import LiveAgentOrchestrator
from app.harness.live_experiment import HarnessLiveExperiment, LiveExperimentConfig
from app.harness.lab_product import LabProductBuilder
from app.harness.manifest import ToolManifestRegistry
from app.harness.models import HarnessConstraints, HarnessRun, HarnessStep, ToolCall, ToolType
from app.harness.optimizer import HarnessOptimizer
from app.harness.planner import HarnessPlanner
from app.harness.presentation import PresentationBlueprintBuilder
from app.harness.recipes import HarnessRecipe, RecipeRegistry
from app.harness.redteam import HarnessRedTeam
from app.harness.research_lab import HarnessResearchLab
from app.harness.report import HarnessReportBuilder
from app.harness.runtime_settings import HarnessRuntimeSettings
from app.harness.security import SecurityAction, SecurityDecision, SecurityEngine
from app.harness.showcase import HarnessShowcaseBuilder
from app.harness.super_agent import ThreadFirstSuperAgent
from app.harness.stream import HarnessEventStreamBuilder
from app.harness.state import HarnessMemoryStore
from app.harness.tools import ToolRegistry
from app.harness.value import HarnessValueScorer
from app.harness.visuals import HarnessVisualProtocol
from app.skills.manager import SkillPackageManager
from app.skills.packages import SkillPackageCatalog


class HarnessEngine:
    """Top-level harness runner for reliable agent execution."""

    def __init__(self, settings: HarnessRuntimeSettings | None = None) -> None:
        self.settings = settings or HarnessRuntimeSettings.from_env()
        self.graph = build_graph()
        self.planner = HarnessPlanner()
        self.skill_manager = SkillPackageManager()
        self.skill_packages = SkillPackageCatalog(
            skills_root=self.skill_manager.skills_root,
            state_file=self.skill_manager.state_file,
        )
        self.tools = ToolRegistry(
            evidence_registry=self.settings.build_evidence_registry(),
            package_catalog=self.skill_packages,
            gateway_config=self.settings.gateway.to_dict(),
        )
        self.thread_runtime = AgentThreadRuntime(
            self.settings.threads_root,
            sandbox_provider=self.settings.build_sandbox_provider(),
            action_mapper=TaskGraphActionMapper(tool_registry=self.tools),
        )
        self.scheduler = AgentExecutionScheduler(self.thread_runtime)
        self.subagents = ParallelSubagentExecutor(self.thread_runtime)
        self.workspace_view = ThreadWorkspaceStreamBuilder()
        self.memory = self.settings.build_memory_store()
        self.guardrails = GuardrailEngine()
        self.evaluator = HarnessEvaluator()
        self.live_agent = LiveAgentOrchestrator()
        self.live_experiment = HarnessLiveExperiment()
        self.iteration = LiveIterationTracker()
        self.lab_product = LabProductBuilder()

        self.manifests = ToolManifestRegistry()
        self.discovery = ToolDiscoveryEngine(self.manifests)
        self.security = SecurityEngine()
        self.recipes = RecipeRegistry()
        self.redteam = HarnessRedTeam()
        self.reporter = HarnessReportBuilder()
        self.missions = MissionRegistry()
        self.code_mission = CodeMissionPackBuilder()
        self.deep_research = HarnessDeepResearchBuilder()
        self.value = HarnessValueScorer()
        self.visuals = HarnessVisualProtocol()
        self.showcase = HarnessShowcaseBuilder()
        self.presentation = PresentationBlueprintBuilder()
        self.stream = HarnessEventStreamBuilder()
        self.optimizer = HarnessOptimizer()
        self.research_lab = HarnessResearchLab()
        self.super_agent = ThreadFirstSuperAgent(self, package_catalog=self.skill_packages)

    def list_tool_catalog(self) -> list[dict[str, Any]]:
        """Return tool manifest catalog and runtime availability."""

        available = set(self.tools.available_tools())
        catalog = []
        for item in self.manifests.list_all():
            payload = item.to_dict()
            payload["available"] = item.name in available
            catalog.append(payload)
        return catalog

    def list_evidence_sources(self) -> list[dict[str, Any]]:
        """Return configured evidence providers for evidence-aware tools."""

        return self.tools.list_evidence_sources()

    def list_skill_packages(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        """Return package-style skills available to the thread-first runtime."""

        return [item for item in self.skill_manager.list_packages() if (item.get("enabled", True) or not enabled_only)]

    def get_skill_package(self, name: str) -> dict[str, Any] | None:
        """Return one package-style skill."""

        return self.skill_manager.get_package(name)

    def suggest_skill_packages(
        self,
        query: str,
        *,
        target: str = "general",
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Suggest package-style skills for a task."""

        return self.tools.suggest_skill_packages(query, target=target, limit=limit)

    def update_skill_package(self, name: str, *, enabled: bool) -> dict[str, Any]:
        """Enable or disable one skill package."""

        payload = self.skill_manager.update_package(name, enabled=enabled)
        self._refresh_skill_runtime()
        return payload

    def install_skill_package_archive(self, archive_path: str | Path) -> dict[str, Any]:
        """Install a DeerFlow-style .skill archive."""

        payload = self.skill_manager.install_archive(archive_path)
        self._refresh_skill_runtime()
        return payload

    def _refresh_skill_runtime(self) -> None:
        self.skill_packages = SkillPackageCatalog(
            skills_root=self.skill_manager.skills_root,
            state_file=self.skill_manager.state_file,
        )
        self.tools = ToolRegistry(
            evidence_registry=self.settings.build_evidence_registry(),
            package_catalog=self.skill_packages,
            gateway_config=self.settings.gateway.to_dict(),
        )
        self.thread_runtime.action_mapper = TaskGraphActionMapper(tool_registry=self.tools)

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

    def build_mission_pack(self, run: HarnessRun) -> dict[str, Any]:
        """Build shared mission-pack artifact from a run."""

        summary = self.reporter.summary(run)
        return summary.get("mission", {}) if isinstance(summary, dict) else {}

    def build_code_mission_pack(
        self,
        run: HarnessRun,
        workspace: str | Path = ".",
        execute_validation: bool = False,
        validation_timeout_seconds: int = 180,
        max_validation_commands: int = 3,
    ) -> dict[str, Any]:
        """Build engineering mission pack with patch/tests/trace/validation artifacts."""

        summary = self.reporter.summary(run)
        return self.code_mission.build(
            query=run.query,
            run=run,
            run_summary=summary,
            workspace=workspace,
            execute_validation=execute_validation,
            validation_timeout_seconds=validation_timeout_seconds,
            max_validation_commands=max_validation_commands,
        )

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

    def list_research_scenarios(self) -> list[dict[str, Any]]:
        """List reproducible research-lab scenarios."""

        return self.research_lab.list_scenarios()

    def list_research_presets(self) -> list[dict[str, Any]]:
        """List candidate presets for research-lab experiments."""

        return self.research_lab.list_candidate_presets()

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

    def run_research_lab(
        self,
        preset: str = "core",
        constraints: HarnessConstraints | None = None,
        candidates: list[dict[str, Any]] | None = None,
        scenario_ids: list[str] | None = None,
        repeats: int = 1,
        seed: int = 7,
        include_runs: bool = False,
        live_model: dict[str, Any] | None = None,
        isolate_memory: bool = True,
        fresh_memory_per_candidate: bool = True,
    ) -> dict[str, Any]:
        """Run research-grade multi-scenario evaluation with reproducible presets."""

        return self.research_lab.run(
            engine=self,
            preset=preset,
            constraints=constraints,
            candidates=candidates,
            scenario_ids=scenario_ids,
            repeats=repeats,
            seed=seed,
            include_runs=include_runs,
            live_model=live_model,
            isolate_memory=isolate_memory,
            fresh_memory_per_candidate=fresh_memory_per_candidate,
        )

    def build_lab_product_bundle(
        self,
        lab_payload: dict[str, Any],
        tag: str = "",
    ) -> dict[str, Any]:
        """Build productized bundle (story/scoreboard/trend) from lab payload."""

        return self.lab_product.build_bundle(lab_payload=lab_payload, tag=tag)

    def write_lab_product_bundle(
        self,
        bundle: dict[str, Any],
        output_dir: str = "reports",
    ) -> dict[str, str]:
        """Write product bundle artifacts to disk."""

        paths = self.lab_product.write_bundle(bundle=bundle, output_dir=Path(output_dir))
        return paths.to_dict()

    def list_lab_product_history(self, limit: int = 12) -> list[dict[str, Any]]:
        """List recent productized lab runs."""

        return self.lab_product.list_history(limit=limit)

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

    def create_thread(
        self,
        title: str = "",
        agent_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a persistent generic agent thread."""

        return self.thread_runtime.create_thread(title=title, agent_name=agent_name, metadata=metadata)

    def list_threads(self, limit: int = 20) -> list[dict[str, Any]]:
        """List persistent generic agent threads."""

        return self.thread_runtime.list_threads(limit=limit)

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Load one persistent generic agent thread."""

        return self.thread_runtime.load_thread(thread_id)

    def request_thread_interrupt(self, thread_id: str, reason: str = "manual") -> dict[str, Any]:
        """Request interrupt for one persistent thread."""

        return self.thread_runtime.request_interrupt(thread_id, reason=reason)

    def execute_thread_task_graph(
        self,
        thread_id: str,
        graph: dict[str, Any],
        execution_label: str = "",
        context: dict[str, Any] | None = None,
        max_nodes: int = 0,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute one task graph inside a persistent thread runtime."""

        return self.thread_runtime.execute_task_graph(
            thread_id,
            graph=graph,
            execution_label=execution_label,
            context=context,
            max_nodes=max_nodes,
            execution_id=execution_id,
        )

    def start_thread_task_graph_async(
        self,
        thread_id: str,
        graph: dict[str, Any],
        execution_label: str = "",
        context: dict[str, Any] | None = None,
        max_nodes: int = 0,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Queue one task graph for background execution inside a persistent thread."""

        return self.thread_runtime.start_task_graph_async(
            thread_id,
            graph=graph,
            execution_label=execution_label,
            context=context,
            max_nodes=max_nodes,
            execution_id=execution_id,
        )

    def resume_thread_execution(self, thread_id: str, execution_id: str) -> dict[str, Any]:
        """Resume a paused or interrupted thread execution."""

        return self.thread_runtime.resume_execution(thread_id, execution_id)

    def retry_thread_execution(
        self,
        thread_id: str,
        execution_id: str,
        from_node_id: str = "",
    ) -> dict[str, Any]:
        """Retry a prior execution, optionally from a selected node."""

        return self.thread_runtime.retry_execution(thread_id, execution_id, from_node_id=from_node_id)

    def wait_for_thread_execution(
        self,
        thread_id: str,
        execution_id: str,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Wait for one background execution to finish or return its latest state."""

        return self.thread_runtime.wait_for_execution(thread_id, execution_id, timeout_seconds=timeout_seconds)

    def list_recoverable_thread_executions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List executions that can be recovered or resumed."""

        self.scheduler.runtime = self.thread_runtime
        return self.scheduler.list_recoverable(limit=limit)

    def recover_thread_execution(
        self,
        thread_id: str,
        execution_id: str,
        async_mode: bool = True,
    ) -> dict[str, Any]:
        """Recover one incomplete execution."""

        self.scheduler.runtime = self.thread_runtime
        return self.scheduler.recover_execution(thread_id, execution_id, async_mode=async_mode)

    def recover_all_thread_executions(
        self,
        async_mode: bool = True,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Recover all incomplete executions."""

        self.scheduler.runtime = self.thread_runtime
        return self.scheduler.recover_all(async_mode=async_mode, limit=limit)

    def run_parallel_subagents(
        self,
        thread_id: str,
        subagents: list[dict[str, Any]],
        wait_timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Run multiple subagent graphs concurrently inside one thread."""

        self.subagents.runtime = self.thread_runtime
        return self.subagents.run_parallel(
            thread_id,
            subagents=subagents,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    def build_thread_workspace_stream(self, thread_id: str) -> dict[str, Any]:
        """Build front-end friendly workspace stream payload for one thread."""

        payload = self.get_thread(thread_id)
        if not payload:
            raise ValueError(f"unknown thread: {thread_id}")
        return self.workspace_view.build(payload)

    def render_thread_workspace_html(self, thread_id: str) -> str:
        """Render workspace/artifact HTML snapshot for one thread."""

        return self.workspace_view.to_html(self.build_thread_workspace_stream(thread_id))

    def export_thread_frontend_snapshot(self, thread_id: str) -> dict[str, Any]:
        """Export a DeerFlow-like thread snapshot contract for frontend consumers."""

        return self.thread_runtime.export_frontend_thread_snapshot(thread_id)

    def generate_deep_research_report(
        self,
        topic: str,
        *,
        subject_root: str | Path = ".",
        competitor_root: str | Path | None = None,
        subject_name: str = "agent-harness",
        competitor_name: str = "deer-flow",
        output_dir: str | Path = "reports",
    ) -> dict[str, Any]:
        """Build and write a deep research report bundle for repository comparison."""

        payload = self.deep_research.build(
            topic=topic,
            subject_root=subject_root,
            competitor_root=competitor_root,
            subject_name=subject_name,
            competitor_name=competitor_name,
        )
        payload["paths"] = self.deep_research.write_bundle(payload, output_dir=output_dir)
        return payload

    def build_generic_task_graph(
        self,
        query: str,
        *,
        target: str = "general",
        workspace_root: str = ".",
        live_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a generic executable task graph for non-mission agent work."""

        return self.compile_generic_task_payload(
            query=query,
            target=target,
            workspace_root=workspace_root,
            live_model=live_model,
        )["graph"]

    def compile_generic_task_payload(
        self,
        query: str,
        *,
        target: str = "general",
        workspace_root: str = ".",
        live_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compile a generic task graph plus planning/profile payload."""

        result = self.tools.call(
            ToolCall(
                name="task_graph_builder",
                tool_type=ToolType.CODE,
                args={
                    "query": query,
                    "target": target,
                    "workspace_root": workspace_root,
                    "live_model": dict(live_model or {}),
                },
            )
        )
        if not result.success:
            raise ValueError(result.error or "task_graph_builder failed")
        output = result.output if isinstance(result.output, dict) else {}
        graph = output.get("graph", {})
        if not isinstance(graph, dict):
            raise ValueError("task_graph_builder returned invalid graph")
        return output

    def run_thread_first(
        self,
        thread_id: str,
        query: str,
        *,
        target: str = "auto",
        max_nodes: int = 0,
        live_model: dict[str, Any] | None = None,
        async_mode: bool = False,
    ) -> dict[str, Any]:
        """Run the thread-first super-agent entrypoint."""

        return self.super_agent.run(
            thread_id,
            query,
            target=target,
            max_nodes=max_nodes,
            live_model=live_model,
            async_mode=async_mode,
        )

    def execute_thread_generic_task(
        self,
        thread_id: str,
        query: str,
        *,
        target: str = "general",
        max_nodes: int = 0,
        live_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build and execute a generic cross-scene task graph inside one thread."""

        return self.run_thread_first(
            thread_id,
            query,
            target=target,
            max_nodes=max_nodes,
            live_model=live_model,
        )

    def run(
        self,
        query: str,
        constraints: HarnessConstraints | None = None,
        mode: str = "balanced",
        recipe: str | None = None,
        recipe_path: str | None = None,
        live_model: dict[str, Any] | None = None,
        thread_id: str | None = None,
        thread_title: str = "",
    ) -> HarnessRun:
        """Run harness loop around the core agent graph."""

        constraints = constraints or HarnessConstraints()
        session_id = thread_id or hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
        thread_context: dict[str, Any] = {}
        if thread_id:
            thread_context = self.thread_runtime.ensure_thread(
                thread_id,
                title=thread_title or query[:80],
            )
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
                thread_context=thread_context,
            )

        safe_query = preflight.redacted_query or query
        graph_result = self.graph.invoke(GraphState(query=safe_query, system_mode=mode))
        payload: dict[str, Any] = graph_result if isinstance(graph_result, dict) else graph_result.model_dump()
        plan = self.planner.build_plan(safe_query, live_model_overrides=live_model)
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
                    fallback = self.planner.next_tool_call(
                        query=safe_query,
                        step=step_idx,
                        plan=plan,
                        session_events=session_events,
                        used_tools=used_tools,
                        live_model_overrides=live_model,
                    )
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
        evidence_summary = self._collect_evidence_summary(steps)
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
            evidence=evidence_summary,
            max_calls=max(0, int(constraints.max_live_agent_calls)),
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
                "evidence": evidence_summary,
                "live_agent": live_result.to_dict() if live_result else {
                    "enabled": False,
                    "configured": False,
                    "calls_used": 0,
                    "call_budget": 0,
                    "success": False,
                    "notes": [],
                    "errors": [],
                },
                "thread": {
                    "thread_id": thread_id or "",
                    "workspace": thread_context.get("workspace", {}),
                }
                if thread_id
                else {},
            },
        )
        run.eval_metrics = self.evaluator.evaluate(run)
        run.metadata["value_card"] = self.build_value_card(run)
        run.metadata["visual_hooks"] = run.metadata["value_card"].get("visual_hooks", [])
        run.mission = self.missions.build_runtime_pack(
            query=query,
            run=run,
            run_summary=self.reporter.summary(run),
        )
        if thread_id:
            self.thread_runtime.record_harness_run(
                thread_id,
                query=query,
                run=run,
                mission=run.mission,
                report_json=self.reporter.summary(run),
                report_markdown=self.reporter.to_markdown(run),
            )
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
        metadata = tool_result.metadata if getattr(tool_result, "metadata", None) else {}
        return {
            "step": step,
            "tool": tool_name,
            "source": source,
            "score": round(float(tool_score), 4),
            "success": bool(tool_result.success),
            "latency_ms": round(float(tool_result.latency_ms), 2),
            "evidence_count": int(len(metadata.get("evidence_records", []))) if isinstance(metadata, dict) else 0,
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
                "metadata": dict(step.tool_result.metadata),
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
        evidence_notes: list[str] = []
        for step in steps:
            if step.tool_result:
                tool_summaries.append(
                    f"- {step.tool_result.name}: "
                    f"{'OK' if step.tool_result.success else 'ERR'} "
                    f"({step.tool_result.latency_ms:.1f}ms)"
                )
                metadata = step.tool_result.metadata if isinstance(step.tool_result.metadata, dict) else {}
                citations = metadata.get("evidence_citations", [])
                if isinstance(citations, list) and citations:
                    evidence_notes.append(f"- {step.tool_result.name}: {', '.join(str(x) for x in citations[:2])}")

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
        if evidence_notes:
            notes.append("- evidence highlights:")
            notes.extend(evidence_notes[:4])

        return (
            f"{payload.get('final_output', '')}\n\n"
            "Harness Execution Notes:\n"
            f"{chr(10).join(notes)}"
        )

    @staticmethod
    def _collect_evidence_summary(steps: list[HarnessStep]) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        citations: list[str] = []
        sources: dict[str, int] = {}
        for step in steps:
            if not step.tool_result or not isinstance(step.tool_result.metadata, dict):
                continue
            metadata = step.tool_result.metadata
            for row in metadata.get("evidence_records", [])[:8]:
                if isinstance(row, dict):
                    records.append(row)
                    source_id = str(row.get("source_id", "")).strip()
                    if source_id:
                        sources[source_id] = sources.get(source_id, 0) + 1
            for citation in metadata.get("evidence_citations", [])[:8]:
                text = str(citation).strip()
                if text and text not in citations:
                    citations.append(text)
        return {
            "record_count": len(records),
            "citation_count": len(citations),
            "records": records[:8],
            "citations": citations[:8],
            "sources": [{"source_id": key, "records": value} for key, value in sorted(sources.items())],
        }

    def _build_preflight_block_run(
        self,
        query: str,
        mode: str,
        session_id: str,
        previous_context: list[dict[str, Any]],
        preflight: SecurityDecision,
        active_recipe: HarnessRecipe | None,
        thread_context: dict[str, Any] | None = None,
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
                "thread": {
                    "thread_id": str((thread_context or {}).get("thread_id", "")),
                    "workspace": dict((thread_context or {}).get("workspace", {})),
                }
                if thread_context
                else {},
            },
        )
        run.eval_metrics = self.evaluator.evaluate(run)
        run.metadata["value_card"] = self.build_value_card(run)
        run.metadata["visual_hooks"] = run.metadata["value_card"].get("visual_hooks", [])
        run.mission = self.missions.build_runtime_pack(
            query=query,
            run=run,
            run_summary=self.reporter.summary(run),
        )
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
        if thread_context and thread_context.get("thread_id"):
            self.thread_runtime.record_harness_run(
                str(thread_context["thread_id"]),
                query=query,
                run=run,
                mission=run.mission,
                report_json=self.reporter.summary(run),
                report_markdown=self.reporter.to_markdown(run),
            )
        return run
