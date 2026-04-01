"""Harness guardrails for tool execution and loop control."""

from __future__ import annotations

from app.harness.models import HarnessConstraints, ToolCall, ToolResult


class GuardrailEngine:
    """Evaluate constraints and return guardrail notes."""

    def check_tool_call(
        self,
        tool_call: ToolCall,
        constraints: HarnessConstraints,
        step: int,
        high_risk: bool,
    ) -> list[str]:
        notes: list[str] = []

        if step > constraints.max_steps:
            notes.append("BLOCK:max_steps_exceeded")
        if tool_call.name in constraints.blocked_tools:
            notes.append(f"BLOCK:blocked_tool:{tool_call.name}")
        if not constraints.allow_write_actions and "write" in tool_call.name:
            notes.append("BLOCK:write_action_disallowed")
        if high_risk and constraints.require_approval_on_high_risk and tool_call.tool_type.value == "api":
            notes.append("WARN:high_risk_requires_human_review")
        return notes

    @staticmethod
    def check_tool_result(tool_result: ToolResult) -> list[str]:
        notes: list[str] = []
        if not tool_result.success:
            notes.append(f"WARN:tool_failed:{tool_result.name}")
        if tool_result.latency_ms > 250.0:
            notes.append(f"WARN:tool_slow:{tool_result.name}")
        return notes
