"""Harness planner and dynamic tool scheduling."""

from __future__ import annotations

from typing import Any

from app.harness.models import ToolCall, ToolType
from app.harness.task_profile import TaskProfile, analyze_task_request


class HarnessPlanner:
    """Generate task-aware plans and fallback tool decisions."""

    def build_plan(
        self,
        query: str,
        *,
        target: str = "general",
        live_model_overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        """Build a lightweight plan from capability-graph planning."""

        profile = analyze_task_request(query, target=target, live_model_overrides=live_model_overrides)
        plan = ["understand goal, constraints, and desired end state"]
        steps = profile.capability_plan.get("steps", []) if isinstance(profile.capability_plan.get("steps", []), list) else []
        for step in steps[:8]:
            if not isinstance(step, dict):
                continue
            title = str(step.get("title", "")).strip()
            reason = str(step.get("reason", "")).strip()
            if title:
                plan.append(f"{title.lower()} because {reason or 'the capability graph selected it'}")
        if profile.requires_validation:
            plan.append("validate the result against the task success criteria and remaining state gap")
        if profile.requires_command_execution:
            plan.append("execute bounded workspace commands only if they reduce the remaining state gap")
        plan.append(f"close the remaining artifact gap and publish a {profile.output_mode} result")
        return plan

    def next_tool_call(
        self,
        query: str,
        step: int,
        plan: list[str],
        *,
        target: str = "general",
        session_events: list[dict[str, object]] | None = None,
        used_tools: set[str] | None = None,
        live_model_overrides: dict[str, Any] | None = None,
    ) -> ToolCall | None:
        """Choose the next fallback tool from missing signals, not fixed steps."""

        del step, plan  # planner is state-driven rather than step-template-driven

        profile = analyze_task_request(query, target=target, live_model_overrides=live_model_overrides)
        used = {str(item).strip() for item in (used_tools or set()) if str(item).strip()}
        events = session_events or []
        event_tools = {
            str(item.get("tool", "")).strip()
            for item in events
            if isinstance(item, dict) and str(item.get("tool", "")).strip()
        }
        seen = used | event_tools

        for tool_name, args in self._candidate_tool_sequence(profile):
            if tool_name in seen:
                continue
            return ToolCall(
                name=tool_name,
                tool_type=self._tool_type(tool_name),
                args=args,
            )
        return None

    @staticmethod
    def _candidate_tool_sequence(profile: TaskProfile) -> list[tuple[str, dict[str, object]]]:
        keywords = profile.keywords or [profile.execution_intent or "task"]
        primary = keywords[0] if keywords else profile.query
        skill_query = " ".join(keywords[:3]) or profile.execution_intent or profile.query
        glob = "*.py" if profile.execution_intent in {"code", "benchmark"} else "*"

        sequence: list[tuple[str, dict[str, object]]] = []
        for step in profile.capability_plan.get("steps", []) if isinstance(profile.capability_plan.get("steps", []), list) else []:
            if not isinstance(step, dict) or str(step.get("node_type", "")) != "tool_call":
                continue
            ref = str(step.get("ref", "")).strip()
            if not ref:
                continue
            args = dict(step.get("default_args", {})) if isinstance(step.get("default_args", {}), dict) else {}
            if ref == "workspace_file_search":
                args.setdefault("query", primary)
                args.setdefault("glob", glob)
                args.setdefault("limit", 8)
            elif ref == "code_skill_search":
                args.setdefault("query", skill_query)
                args.setdefault("limit", 6)
            elif ref == "external_resource_hub":
                args.setdefault("query", profile.query)
                args.setdefault("limit", 6)
            elif ref == "evidence_dossier_builder":
                args.setdefault("query", profile.query)
                args.setdefault("limit", 5)
                args.setdefault("domains", profile.domains or ["general"])
            elif ref == "policy_risk_matrix":
                args.setdefault("query", profile.query)
                args.setdefault("evidence_limit", 4)
            else:
                args.setdefault("query", profile.query)
            sequence.append((ref, args))

        if profile.execution_intent == "benchmark":
            sequence.append(("code_experiment_design", {"query": profile.query, "max_experiments": 5}))
        elif profile.execution_intent == "code" and profile.requires_validation:
            sequence.append(("memory_context_digest", {"events": []}))
        return sequence

    @staticmethod
    def _tool_type(tool_name: str) -> ToolType:
        mapping = {
            "workspace_file_search": ToolType.CODE,
            "external_resource_hub": ToolType.BROWSER,
            "evidence_dossier_builder": ToolType.BROWSER,
            "tool_search": ToolType.CODE,
            "code_skill_search": ToolType.CODE,
            "policy_risk_matrix": ToolType.CODE,
            "code_experiment_design": ToolType.CODE,
            "memory_context_digest": ToolType.CODE,
        }
        return mapping.get(tool_name, ToolType.CODE)
