"""Harness planner and loop scheduler."""

from __future__ import annotations

from app.harness.models import ToolCall, ToolType


class HarnessPlanner:
    """Generate execution plan and per-step tool decisions."""

    def build_plan(self, query: str) -> list[str]:
        """Build lightweight plan from query intent."""

        lowered = query.lower()
        plan = ["understand intent", "collect supporting signals"]

        if any(token in lowered for token in ["risk", "compliance", "audit", "critical", "safety"]):
            plan.append("run verification-oriented tools")
        if any(token in lowered for token in ["compare", "option", "strategy", "recommend"]):
            plan.append("retrieve alternatives and trade-offs")
        if any(token in lowered for token in ["code", "skill", "tool", "agent"]):
            plan.append("inspect code-level capabilities")

        plan.append("synthesize and evaluate")
        return plan

    def next_tool_call(self, query: str, step: int, plan: list[str]) -> ToolCall | None:
        """Choose next tool call for scheduler loop."""

        lowered = query.lower()

        if step == 1:
            return ToolCall(
                name="api_market_discover",
                tool_type=ToolType.API,
                args={"query": query, "limit": 3},
            )

        if step == 2:
            return ToolCall(
                name="code_skill_search",
                tool_type=ToolType.CODE,
                args={"query": "risk" if "risk" in lowered else query.split()[0] if query.split() else ""},
            )

        if step == 3 and any("alternatives" in item or "trade-offs" in item for item in plan):
            return ToolCall(
                name="browser_trending_scan",
                tool_type=ToolType.BROWSER,
                args={"limit": 3},
            )

        return None
