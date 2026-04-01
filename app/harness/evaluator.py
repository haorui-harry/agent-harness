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

        return {
            "tool_calls": float(len(tool_steps)),
            "tool_success_rate": round(tool_success_rate, 4),
            "guardrail_block_count": float(guardrail_hits),
            "context_reuse_score": memory_reuse,
            "completion_score": completion,
        }
