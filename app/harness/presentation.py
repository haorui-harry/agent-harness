"""Presentation blueprint builder for high-impact first-screen dashboards."""

from __future__ import annotations

from typing import Any


class PresentationBlueprintBuilder:
    """Convert payloads into opinionated dashboard layout configs."""

    def build_first_screen(self, payload: dict[str, Any]) -> dict[str, Any]:
        kpis = payload.get("kpis", {})
        value_index = float(kpis.get("value_index", 0.0))
        reliability = float(kpis.get("reliability", 0.0))
        safety = float(kpis.get("safety", 0.0))
        innovation = float(kpis.get("innovation", 0.0))
        live_calls = float(kpis.get("live_agent_calls", 0.0))
        live_success = float(kpis.get("live_agent_success", 0.0))

        status_badges = []
        if value_index >= 85:
            status_badges.append({"label": "PLATINUM VALUE", "tone": "success"})
        elif value_index >= 72:
            status_badges.append({"label": "GOLD VALUE", "tone": "accent"})
        else:
            status_badges.append({"label": "VALUE IMPROVABLE", "tone": "warning"})

        if safety >= 0.80:
            status_badges.append({"label": "SAFETY STRONG", "tone": "success"})
        if innovation >= 0.75:
            status_badges.append({"label": "INNOVATION HIGH", "tone": "accent"})
        if reliability < 0.65:
            status_badges.append({"label": "RELIABILITY WATCH", "tone": "warning"})
        if live_calls > 0 and live_success > 0:
            status_badges.append({"label": "REAL AGENT ONLINE", "tone": "success"})
        elif live_calls > 0:
            status_badges.append({"label": "REAL AGENT DEGRADED", "tone": "warning"})

        callouts = self._build_callouts(payload)
        panels = [
            {"id": "radar", "title": "Value Radar", "component": "radar_chart", "data_ref": "radar"},
            {"id": "timeline", "title": "Execution Timeline", "component": "timeline_gantt", "data_ref": "timeline"},
            {
                "id": "discovery",
                "title": "Discovery Opportunity Board",
                "component": "bubble_scatter",
                "data_ref": "discovery_board",
            },
            {"id": "security", "title": "Safety Decision Lane", "component": "security_lane", "data_ref": "security_board"},
            {"id": "network", "title": "Tool Relation Graph", "component": "force_graph", "data_ref": "tool_network"},
        ]
        if live_calls > 0:
            panels.insert(
                3,
                {
                    "id": "live-agent",
                    "title": "Real Agent Loop",
                    "component": "agent_loop_board",
                    "data_ref": "live_agent_board",
                },
            )

        return {
            "layout_version": "1.0.0",
            "theme": {
                "name": "harness-impact",
                "style": "high-contrast-clean",
                "primary": "#0E1A2B",
                "accent": "#1E9B8A",
                "warning": "#E8A317",
                "danger": "#C0392B",
                "surface": "#F7F8FA",
            },
            "hero": {
                "title": "Harness Value Lens",
                "subtitle": payload.get("narrative", ""),
                "badges": status_badges,
                "kpi_cards": [
                    {"title": "Value Index", "value": round(value_index, 2), "ref": "kpis.value_index"},
                    {"title": "Reliability", "value": round(reliability * 100, 1), "unit": "%", "ref": "kpis.reliability"},
                    {"title": "Safety", "value": round(safety * 100, 1), "unit": "%", "ref": "kpis.safety"},
                    {"title": "Innovation", "value": round(innovation * 100, 1), "unit": "%", "ref": "kpis.innovation"},
                    {"title": "Live Calls", "value": round(live_calls, 1), "ref": "kpis.live_agent_calls"},
                ],
            },
            "panels": panels,
            "callouts": callouts,
            "motion_guidance": [
                "Hero badges fade-in with 120ms stagger.",
                "Timeline bars animate width from 0 to latency proportion.",
                "Network graph enters with force warm-up before pinning nodes.",
                "Agent loop board reveals analyze->synthesize->critique states sequentially.",
            ],
        }

    @staticmethod
    def _build_callouts(payload: dict[str, Any]) -> list[dict[str, str]]:
        cards = payload.get("hero_cards", [])
        out: list[dict[str, str]] = []
        for item in cards[:3]:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "title": str(item.get("title", "")),
                    "headline": str(item.get("headline", "")),
                    "evidence": str(item.get("evidence", "")),
                }
            )
        return out
