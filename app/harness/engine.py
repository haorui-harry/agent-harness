"""Harness engine orchestrating planner, tools, memory, guardrails, and eval."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

from app.core.state import GraphState
from app.graph import build_graph
from app.harness.evaluator import HarnessEvaluator
from app.harness.guardrails import GuardrailEngine
from app.harness.models import HarnessConstraints, HarnessRun, HarnessStep
from app.harness.planner import HarnessPlanner
from app.harness.state import HarnessMemoryStore
from app.harness.tools import ToolRegistry


class HarnessEngine:
    """Top-level harness runner for reliable agent execution."""

    def __init__(self) -> None:
        self.graph = build_graph()
        self.planner = HarnessPlanner()
        self.tools = ToolRegistry()
        self.memory = HarnessMemoryStore()
        self.guardrails = GuardrailEngine()
        self.evaluator = HarnessEvaluator()

    def run(
        self,
        query: str,
        constraints: HarnessConstraints | None = None,
        mode: str = "balanced",
    ) -> HarnessRun:
        """Run harness loop around the core agent graph."""

        constraints = constraints or HarnessConstraints()

        graph_result = self.graph.invoke(GraphState(query=query, system_mode=mode))
        payload: dict[str, Any] = graph_result if isinstance(graph_result, dict) else graph_result.model_dump()

        plan = self.planner.build_plan(query)
        session_id = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
        previous_context = self.memory.read_recent(session_id, limit=8)

        steps: list[HarnessStep] = []
        high_risk = payload.get("risk_level") in {"high", "critical"}
        tool_calls = 0

        for step_idx in range(1, constraints.max_steps + 1):
            tool_call = self.planner.next_tool_call(query=query, step=step_idx, plan=plan)
            if not tool_call:
                break

            thought = f"step-{step_idx}: choose tool for plan segment"
            decision = f"use {tool_call.name}"
            guardrail_notes = self.guardrails.check_tool_call(
                tool_call=tool_call,
                constraints=constraints,
                step=step_idx,
                high_risk=high_risk,
            )

            if tool_calls >= constraints.max_tool_calls:
                guardrail_notes.append("BLOCK:max_tool_calls_exceeded")

            blocked = any(note.startswith("BLOCK") for note in guardrail_notes)
            if blocked:
                steps.append(
                    HarnessStep(
                        step=step_idx,
                        thought=thought,
                        decision=decision,
                        tool_call=tool_call,
                        tool_result=None,
                        guardrail_notes=guardrail_notes,
                    )
                )
                break

            result = self.tools.call(tool_call)
            guardrail_notes.extend(self.guardrails.check_tool_result(result))
            tool_calls += 1

            memory_event = {
                "step": step_idx,
                "tool": tool_call.name,
                "success": result.success,
                "latency_ms": round(result.latency_ms, 2),
            }
            self.memory.append_event(session_id, memory_event)

            steps.append(
                HarnessStep(
                    step=step_idx,
                    thought=thought,
                    decision=decision,
                    tool_call=tool_call,
                    tool_result=result,
                    guardrail_notes=guardrail_notes,
                )
            )

        tool_summaries = []
        for step in steps:
            if step.tool_result:
                tool_summaries.append(
                    f"- {step.tool_result.name}: "
                    f"{'OK' if step.tool_result.success else 'ERR'} "
                    f"({step.tool_result.latency_ms:.1f}ms)"
                )

        harness_notes = "\n".join(tool_summaries) if tool_summaries else "- no harness tools executed"
        final_answer = (
            f"{payload.get('final_output', '')}\n\n"
            "Harness Execution Notes:\n"
            f"{harness_notes}"
        )

        run = HarnessRun(
            query=query,
            plan=plan,
            steps=steps,
            final_answer=final_answer,
            completed=True,
            eval_metrics={},
            memory_snapshot=previous_context,
            metadata={
                "session_id": session_id,
                "mode": mode,
                "risk_level": payload.get("risk_level", "unknown"),
                "selected_agent": payload.get("agent_name", ""),
                "selected_skills": payload.get("selected_skills", []),
            },
        )
        run.eval_metrics = self.evaluator.evaluate(run)
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
