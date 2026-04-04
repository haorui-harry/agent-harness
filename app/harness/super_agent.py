"""Thread-first super-agent entrypoint aligned with DeerFlow-style runtime flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.harness.task_profile import analyze_task_request
from app.skills.packages import SkillPackageCatalog

if TYPE_CHECKING:
    from app.harness.engine import HarnessEngine


@dataclass(frozen=True)
class SuperAgentRoute:
    """Planning summary for a thread-first task entry."""

    kind: str
    target: str
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "rationale": list(self.rationale),
        }


class ThreadFirstSuperAgent:
    """Single product entrypoint for thread-bound super-agent execution."""

    def __init__(self, engine: "HarnessEngine", *, package_catalog: SkillPackageCatalog | None = None) -> None:
        self.engine = engine
        self.package_catalog = package_catalog or SkillPackageCatalog()

    def run(
        self,
        thread_id: str,
        query: str,
        *,
        target: str = "auto",
        max_nodes: int = 0,
        live_model: dict[str, Any] | None = None,
        async_mode: bool = False,
    ) -> dict[str, Any]:
        """Compile and execute a thread-bound task through one super-agent entry."""

        thread = self.engine.thread_runtime.ensure_thread(thread_id, title=query[:80], agent_name="super-agent")
        effective_target = self._resolve_target(query=query, target=target, workspace_root=thread.get("workspace", {}).get("workspace", ""))
        package_suggestions = self.package_catalog.suggest(query, target=effective_target, limit=6)
        compilation = self.engine.compile_generic_task_payload(
            query=query,
            target=effective_target,
            workspace_root=str(thread.get("workspace", {}).get("workspace", ".")),
            live_model=live_model,
        )
        graph = dict(compilation.get("graph", {}))
        profile = dict(compilation.get("profile", {}))
        route = self._build_route(profile=profile, packages=package_suggestions, target=effective_target)

        graph.setdefault("metadata", {})
        graph["metadata"]["skill_packages"] = [item.to_dict() for item in package_suggestions]
        graph["metadata"]["super_agent_route"] = route.to_dict()

        self.engine.thread_runtime.append_message(
            thread_id,
            "user",
            query,
            metadata={
                "entrypoint": "thread-first-super-agent",
                "target": effective_target,
            },
        )
        self.engine.thread_runtime.append_event(
            thread_id,
            {
                "event": "super_agent_planned",
                "target": effective_target,
                "packages": [item.name for item in package_suggestions],
                "route": route.to_dict(),
            },
        )

        context = {
            "query": query,
            "target": effective_target,
            "live_model": dict(live_model or {}),
            "skill_packages": [item.to_dict() for item in package_suggestions],
            "route": route.to_dict(),
        }
        if async_mode:
            execution = self.engine.start_thread_task_graph_async(
                thread_id,
                graph,
                execution_label=f"super-agent:{effective_target}",
                context=context,
                max_nodes=max_nodes,
            )
        else:
            execution = self.engine.execute_thread_task_graph(
                thread_id,
                graph,
                execution_label=f"super-agent:{effective_target}",
                context=context,
                max_nodes=max_nodes,
            )

        self.engine.thread_runtime.append_message(
            thread_id,
            "assistant",
            self._build_summary(execution=execution, route=route, packages=package_suggestions),
            metadata={
                "entrypoint": "thread-first-super-agent",
                "execution_id": execution.get("execution_id", ""),
                "status": execution.get("status", ""),
            },
        )

        return {
            "schema": "agent-harness-thread-super-agent/v1",
            "thread": self.engine.get_thread(thread_id),
            "route": route.to_dict(),
            "packages": [item.to_dict() for item in package_suggestions],
            "profile": profile,
            "graph": graph,
            "execution": execution,
        }

    def _resolve_target(self, *, query: str, target: str, workspace_root: str) -> str:
        requested = str(target or "auto").strip().lower()
        if requested and requested != "auto":
            return requested
        profile = analyze_task_request(query=query, target="general", workspace_root=workspace_root)
        selected = set(profile.deliberation.selected)
        intent = str(profile.execution_intent or "general").strip().lower()
        output_mode = str(profile.output_mode or "artifact").strip().lower()

        if output_mode == "patch" and "workspace" in selected and "web" not in selected:
            return "code"
        if output_mode == "runbook" and intent == "ops" and "risk" in selected:
            return "ops"
        if output_mode == "report" and intent == "research" and "web" in selected and "workspace" not in selected:
            return "research"
        if output_mode == "benchmark" and "web" in selected and "workspace" not in selected:
            return "research"
        return "general"

    @staticmethod
    def _build_route(profile: dict[str, Any], packages: list[Any], target: str) -> SuperAgentRoute:
        rationale: list[str] = []
        selected_channels = profile.get("selected_channels", [])
        if isinstance(selected_channels, list) and selected_channels:
            rationale.append(f"selected channels: {', '.join(str(item) for item in selected_channels[:4])}")
        if packages:
            rationale.append(f"package priors: {', '.join(item.name for item in packages[:3])}")
        if profile.get("requires_workspace"):
            rationale.append("thread workspace inspection is required")
        if profile.get("requires_external_evidence"):
            rationale.append("external evidence collection is required")
        return SuperAgentRoute(
            kind="task_graph",
            target=target,
            rationale=rationale or ["generic executable task graph compiled"],
        )

    @staticmethod
    def _build_summary(*, execution: dict[str, Any], route: SuperAgentRoute, packages: list[Any]) -> str:
        status = str(execution.get("status", "unknown"))
        package_names = ", ".join(item.name for item in packages[:4]) if packages else "none"
        return (
            f"Thread-first super-agent route={route.kind} target={route.target} status={status}. "
            f"Packages: {package_names}."
        )
