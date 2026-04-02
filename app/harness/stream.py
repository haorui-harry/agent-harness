"""Event stream protocol for animated harness visualizations."""

from __future__ import annotations

from typing import Any

from app.harness.models import HarnessRun


class HarnessEventStreamBuilder:
    """Generate replay-friendly event streams from harness runs."""

    def build(self, run: HarnessRun, visual_payload: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        t = 0.0

        events.append(
            {
                "ts_ms": t,
                "event": "run_started",
                "data": {"query": run.query, "mode": run.metadata.get("mode", ""), "plan": run.plan},
            }
        )
        t += 30.0

        security = run.metadata.get("security", {})
        if isinstance(security, dict):
            events.append(
                {
                    "ts_ms": t,
                    "event": "security_preflight",
                    "data": {
                        "action": security.get("preflight_action", "allow"),
                        "risk_score": security.get("preflight_risk_score", 0.0),
                        "findings": security.get("preflight_findings", []),
                    },
                }
            )
            t += 40.0

        discovery = run.metadata.get("discovery", [])
        if isinstance(discovery, list) and discovery:
            events.append(
                {
                    "ts_ms": t,
                    "event": "tools_discovered",
                    "data": {"top": discovery[:6], "count": len(discovery)},
                }
            )
            t += 50.0

        for step in run.steps:
            events.append(
                {
                    "ts_ms": t,
                    "event": "step_started",
                    "data": {
                        "step": step.step,
                        "tool": step.tool_call.name if step.tool_call else "",
                        "source": step.tool_call.source if step.tool_call else "",
                    },
                }
            )
            latency = float(step.tool_result.latency_ms) if step.tool_result else 20.0
            t += max(20.0, latency)
            events.append(
                {
                    "ts_ms": t,
                    "event": "step_finished",
                    "data": {
                        "step": step.step,
                        "tool": step.tool_call.name if step.tool_call else "",
                        "success": bool(step.tool_result.success) if step.tool_result else False,
                        "status": "blocked"
                        if any(note.startswith("BLOCK") for note in step.guardrail_notes)
                        else "ok",
                        "notes": step.guardrail_notes,
                    },
                }
            )
            t += 25.0

        value_card = run.metadata.get("value_card", {})
        if isinstance(value_card, dict):
            events.append(
                {
                    "ts_ms": t,
                    "event": "value_card_ready",
                    "data": {
                        "value_index": value_card.get("value_index", 0.0),
                        "band": value_card.get("band", ""),
                        "dimensions": value_card.get("dimensions", []),
                    },
                }
            )
            t += 35.0

        live = run.metadata.get("live_agent", {})
        if isinstance(live, dict) and live.get("enabled"):
            events.append(
                {
                    "ts_ms": t,
                    "event": "live_agent_cycle",
                    "data": {
                        "configured": bool(live.get("configured", False)),
                        "success": bool(live.get("success", False)),
                        "calls_used": int(live.get("calls_used", 0)),
                        "call_budget": int(live.get("call_budget", 0)),
                        "model": str(live.get("model", "")),
                        "notes": live.get("notes", []),
                        "errors": live.get("errors", []),
                    },
                }
            )
            t += 30.0

        events.append(
            {
                "ts_ms": t,
                "event": "visual_payload_ready",
                "data": {
                    "sections": sorted(visual_payload.keys()),
                    "timeline_steps": len(visual_payload.get("timeline", [])),
                },
            }
        )
        return events
