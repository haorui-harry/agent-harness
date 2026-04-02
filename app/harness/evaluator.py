"""Harness evaluation metrics."""

from __future__ import annotations

from app.harness.models import HarnessRun


class HarnessEvaluator:
    """Compute evaluation metrics for harness runs."""

    def evaluate(self, run: HarnessRun) -> dict[str, float]:
        """Return eval metrics dictionary."""

        tool_steps = [step for step in run.steps if step.tool_call is not None]
        tool_results = [step.tool_result for step in run.steps if step.tool_result is not None]

        success_count = sum(1 for result in tool_results if result and result.success)
        tool_success_rate = success_count / max(len(tool_results), 1)

        guardrail_hits = 0
        for step in run.steps:
            guardrail_hits += len([item for item in step.guardrail_notes if item.startswith("BLOCK")])

        memory_reuse = 1.0 if run.memory_snapshot else 0.0
        completion = 1.0 if run.completed else 0.0

        security_block_count = 0
        security_challenge_count = 0
        for step in run.steps:
            for note in step.guardrail_notes:
                if "SECURITY_BLOCK" in note:
                    security_block_count += 1
                if "SECURITY_CHALLENGE" in note:
                    security_challenge_count += 1

        discovery = run.metadata.get("discovery", [])
        discovery_count = float(len(discovery)) if isinstance(discovery, list) else 0.0

        used_tools = {step.tool_call.name for step in run.steps if step.tool_call}
        discovery_names = set()
        if isinstance(discovery, list):
            for item in discovery:
                if isinstance(item, dict):
                    name = item.get("name")
                    if isinstance(name, str):
                        discovery_names.add(name)
        discovery_utilization = len(used_tools & discovery_names) / max(len(used_tools), 1)

        recipe_meta = run.metadata.get("recipe", {})
        recipe_total_steps = int(recipe_meta.get("total_steps", 0)) if isinstance(recipe_meta, dict) else 0
        recipe_executed_steps = int(recipe_meta.get("executed_steps", 0)) if isinstance(recipe_meta, dict) else 0
        recipe_completion = recipe_executed_steps / max(recipe_total_steps, 1) if recipe_total_steps else 0.0

        live_meta = run.metadata.get("live_agent", {})
        live_enabled = 1.0 if isinstance(live_meta, dict) and live_meta.get("enabled") else 0.0
        live_configured = 1.0 if isinstance(live_meta, dict) and live_meta.get("configured") else 0.0
        live_success = 1.0 if isinstance(live_meta, dict) and live_meta.get("success") else 0.0
        live_calls = float(live_meta.get("calls_used", 0.0)) if isinstance(live_meta, dict) else 0.0

        return {
            "tool_calls": float(len(tool_steps)),
            "tool_success_rate": round(tool_success_rate, 4),
            "guardrail_block_count": float(guardrail_hits),
            "context_reuse_score": memory_reuse,
            "completion_score": completion,
            "security_block_count": float(security_block_count),
            "security_challenge_count": float(security_challenge_count),
            "discovery_count": discovery_count,
            "discovery_utilization": round(discovery_utilization, 4),
            "recipe_completion": round(recipe_completion, 4),
            "live_agent_enabled": live_enabled,
            "live_agent_configured": live_configured,
            "live_agent_success": live_success,
            "live_agent_calls": live_calls,
        }
