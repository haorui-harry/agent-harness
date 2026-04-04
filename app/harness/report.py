"""Structured reporting for harness runs."""

from __future__ import annotations

from typing import Any

from app.core.mission import MissionRegistry
from app.harness.models import HarnessRun


class HarnessReportBuilder:
    """Build summary reports from harness runs."""

    def __init__(self) -> None:
        self.missions = MissionRegistry()

    def summary(self, run: HarnessRun) -> dict[str, Any]:
        security = run.metadata.get("security", {})
        discovery = run.metadata.get("discovery", [])
        recipe = run.metadata.get("recipe", {})
        value_card = run.metadata.get("value_card", {})
        live_agent = run.metadata.get("live_agent", {})
        evidence = run.metadata.get("evidence", {})

        step_rows: list[dict[str, Any]] = []
        for step in run.steps:
            step_rows.append(
                {
                    "step": step.step,
                    "tool": step.tool_call.name if step.tool_call else "",
                    "source": step.tool_call.source if step.tool_call else "",
                    "score": round(float(step.tool_call.score), 4) if step.tool_call else 0.0,
                    "success": bool(step.tool_result.success) if step.tool_result else False,
                    "latency_ms": round(float(step.tool_result.latency_ms), 2) if step.tool_result else 0.0,
                    "evidence_count": int(len(step.tool_result.metadata.get("evidence_records", []))) if step.tool_result else 0,
                    "notes": step.guardrail_notes,
                }
            )

        top_discovery: list[dict[str, Any]] = []
        if isinstance(discovery, list):
            for item in discovery[:5]:
                if isinstance(item, dict):
                    top_discovery.append(
                        {
                            "name": item.get("name", ""),
                            "score": round(float(item.get("score", 0.0)), 4),
                            "reasons": item.get("reasons", []),
                        }
                    )

        summary = {
            "query": run.query,
            "completed": run.completed,
            "plan": run.plan,
            "final_answer_preview": run.final_answer[:320],
            "metrics": run.eval_metrics,
            "value_card": value_card if isinstance(value_card, dict) else {},
            "security": {
                "preflight_action": security.get("preflight_action", ""),
                "preflight_risk_score": security.get("preflight_risk_score", 0.0),
                "preflight_findings": security.get("preflight_findings", []),
            },
            "recipe": recipe,
            "live_agent": self._sanitize_live_agent(live_agent if isinstance(live_agent, dict) else {}),
            "evidence": evidence if isinstance(evidence, dict) else {},
            "top_discovery": top_discovery,
            "steps": step_rows,
            "memory_snapshot_size": len(run.memory_snapshot),
        }
        summary["mission"] = (
            run.mission
            if isinstance(run.mission, dict) and run.mission
            else self.missions.build_runtime_pack(run.query, run=run, run_summary=summary)
        )
        return summary

    def to_markdown(self, run: HarnessRun) -> str:
        data = self.summary(run)
        mission = data.get("mission", {}) if isinstance(data.get("mission", {}), dict) else {}
        task_graph = mission.get("task_graph", {}) if isinstance(mission.get("task_graph", {}), dict) else {}

        lines = [
            "# Harness Run Report",
            "",
            f"- Query: `{data['query']}`",
            f"- Completed: `{data['completed']}`",
            f"- Preflight: `{data['security']['preflight_action']}` (risk={data['security']['preflight_risk_score']})",
            f"- Value Index: `{data.get('value_card', {}).get('value_index', 0.0)}`",
            f"- Live Agent Calls: `{data.get('live_agent', {}).get('calls_used', 0)}`",
            f"- Evidence Records: `{data.get('evidence', {}).get('record_count', 0)}`",
            "",
            "## Mission Pack",
            f"- Type: `{mission.get('title', '')}`",
            f"- Primary Deliverable: `{mission.get('primary_deliverable', '')}`",
            f"- Decision: `{mission.get('decision', {}).get('status', '')}`",
            f"- Task Graph: `{task_graph.get('summary', {}).get('node_count', 0)}` nodes / `{task_graph.get('summary', {}).get('completed_nodes', 0)}` completed",
            "",
            "## Plan",
        ]
        for item in data["plan"]:
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "## Deliverables",
            ]
        )
        if mission.get("deliverables"):
            for item in mission.get("deliverables", [])[:4]:
                lines.append(
                    f"- `{item.get('title', '')}` status={item.get('status', '')} signal={item.get('evidence_hint', '-')}"
                )
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                "## Benchmark Fit",
            ]
        )
        if mission.get("benchmark_targets"):
            for item in mission.get("benchmark_targets", [])[:4]:
                lines.append(
                    f"- `{item.get('name', '')}` fit={item.get('fit', '')} gap={item.get('gap', '')}"
                )
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                "## Executable Task Graph",
            ]
        )
        if task_graph.get("nodes"):
            for node in task_graph.get("nodes", [])[:6]:
                lines.append(
                    f"- `{node.get('node_id', '')}` type={node.get('node_type', '')} "
                    f"status={node.get('status', '')} depends_on={','.join(node.get('depends_on', [])) or '-'}"
                )
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                "## Top Discovery",
            ]
        )
        if data["top_discovery"]:
            for item in data["top_discovery"]:
                reasons = ", ".join(item["reasons"]) if item["reasons"] else "-"
                lines.append(f"- `{item['name']}` score={item['score']} reasons={reasons}")
        else:
            lines.append("- none")

        recipe = data["recipe"] if isinstance(data["recipe"], dict) else {}
        live = data["live_agent"] if isinstance(data["live_agent"], dict) else {}
        lines.extend(
            [
                "",
                "## Recipe",
                f"- Name: `{recipe.get('name', '')}`",
                f"- Executed Steps: `{recipe.get('executed_steps', 0)}` / `{recipe.get('total_steps', 0)}`",
                "",
                "## Live Agent",
                f"- Enabled: `{live.get('enabled', False)}`",
                f"- Configured: `{live.get('configured', False)}`",
                f"- Model: `{live.get('model', '')}`",
                f"- Calls: `{live.get('calls_used', 0)}` / `{live.get('call_budget', 0)}`",
                f"- Success: `{live.get('success', False)}`",
                "",
                "## Evidence",
                f"- Records: `{data.get('evidence', {}).get('record_count', 0)}`",
                f"- Citations: `{data.get('evidence', {}).get('citation_count', 0)}`",
                *(f"- Citation: `{item}`" for item in data.get("evidence", {}).get("citations", [])[:4]),
                "",
                "## Metrics",
            ]
        )
        metrics = data["metrics"]
        for key in sorted(metrics.keys()):
            lines.append(f"- {key}: `{metrics[key]}`")

        lines.extend(
            [
                "",
                "## Step Timeline",
                "| Step | Tool | Source | Score | Success | Latency(ms) |",
                "|------|------|--------|-------|---------|-------------|",
            ]
        )
        for row in data["steps"]:
            lines.append(
                f"| {row['step']} | {row['tool']} | {row['source']} | "
                f"{row['score']} | {row['success']} | {row['latency_ms']} |"
            )

        lines.extend(
            [
                "",
                "## Final Answer Preview",
                "",
                "```text",
                data["final_answer_preview"],
                "```",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _sanitize_live_agent(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        cleaned = dict(payload)
        cleaned.pop("base_url", None)
        cleaned.pop("transport", None)
        strategy = cleaned.get("strategy")
        analysis = cleaned.get("analysis")
        critique = cleaned.get("critique")
        if isinstance(strategy, dict):
            cleaned["strategy"] = dict(strategy)
        if isinstance(analysis, dict):
            cleaned["analysis"] = dict(analysis)
        if isinstance(critique, dict):
            cleaned["critique"] = dict(critique)
        return cleaned
