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
        """Build a lightweight plan from a dynamic task profile."""

        profile = analyze_task_request(query, target=target, live_model_overrides=live_model_overrides)
        plan = ["understand request and success criteria"]

        if "workspace" in profile.deliberation.selected:
            plan.append("inspect workspace context and local artifacts")
        if "web" in profile.deliberation.selected:
            plan.append("collect external evidence and references")
        if "discovery" in profile.deliberation.selected or profile.skill_priors:
            prior_names = ", ".join(item.name for item in profile.skill_priors[:3]) or "general skills"
            plan.append(f"select capabilities and skill priors ({prior_names})")
        if profile.requires_command_execution:
            plan.append("execute bounded actions inside the workspace")
        if profile.requires_validation:
            plan.append("validate claims or outputs before final synthesis")

        plan.append(f"synthesize a {profile.output_mode} artifact")
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
        selected = set(profile.deliberation.selected)

        sequence: list[tuple[str, dict[str, object]]] = []
        if "workspace" in selected:
            sequence.append(
                (
                    "workspace_file_search",
                    {"query": primary, "glob": glob, "limit": 8},
                )
            )
        if "web" in selected:
            sequence.append(
                (
                    "external_resource_hub",
                    {"query": profile.query, "limit": 6},
                )
            )
            sequence.append(
                (
                    "evidence_dossier_builder",
                    {"query": profile.query, "limit": 5, "domains": profile.domains or ["general"]},
                )
            )
        if "discovery" in selected:
            sequence.append(
                (
                    "tool_search",
                    {"query": profile.query, "limit": 6},
                )
            )
        if profile.skill_priors:
            sequence.append(
                (
                    "code_skill_search",
                    {"query": skill_query, "limit": 6},
                )
            )
        if "risk" in selected or "risk" in profile.domains or profile.execution_intent == "ops":
            sequence.append(
                (
                    "policy_risk_matrix",
                    {"query": profile.query, "evidence_limit": 4},
                )
            )
        if profile.execution_intent == "benchmark":
            sequence.append(
                (
                    "code_experiment_design",
                    {"query": profile.query, "max_experiments": 5},
                )
            )
        elif profile.execution_intent == "code" and profile.requires_validation:
            sequence.append(
                (
                    "memory_context_digest",
                    {"events": []},
                )
            )
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
