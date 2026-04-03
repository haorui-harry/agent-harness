"""Parallel subagent execution helpers for thread runtime."""

from __future__ import annotations

from typing import Any

from app.agents.runtime import AgentThreadRuntime


class ParallelSubagentExecutor:
    """Run multiple subagent task graphs concurrently inside one thread runtime."""

    def __init__(self, runtime: AgentThreadRuntime) -> None:
        self.runtime = runtime

    def run_parallel(
        self,
        thread_id: str,
        *,
        subagents: list[dict[str, Any]],
        wait_timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        queued: list[dict[str, Any]] = []
        for item in subagents:
            graph = item.get("graph", {}) if isinstance(item.get("graph", {}), dict) else {}
            label = str(item.get("name", "subagent"))
            queued_payload = self.runtime.start_task_graph_async(
                thread_id,
                graph=graph,
                execution_label=f"subagent:{label}",
                context=item.get("context", {}) if isinstance(item.get("context", {}), dict) else {},
            )
            execution = queued_payload.get("execution", {}) if isinstance(queued_payload.get("execution", {}), dict) else {}
            queued.append(
                {
                    "name": label,
                    "execution_id": execution.get("execution_id", ""),
                }
            )

        completed: list[dict[str, Any]] = []
        for item in queued:
            result = self.runtime.wait_for_execution(
                thread_id,
                str(item.get("execution_id", "")),
                timeout_seconds=wait_timeout_seconds,
            )
            completed.append(
                {
                    "name": item["name"],
                    "execution_id": item["execution_id"],
                    "status": result.get("status", ""),
                    "completed_nodes": result.get("graph", {}).get("summary", {}).get("completed_nodes", 0),
                    "node_count": result.get("graph", {}).get("summary", {}).get("node_count", 0),
                }
            )

        return {
            "schema": "agent-harness-subagent-parallel/v1",
            "thread_id": thread_id,
            "subagents": completed,
            "summary": {
                "count": len(completed),
                "completed": sum(1 for item in completed if item.get("status") == "completed"),
                "interrupted": sum(1 for item in completed if item.get("status") == "interrupted"),
            },
        }
