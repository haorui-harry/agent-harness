"""Visualization payload protocol for harness runs."""

from __future__ import annotations

from typing import Any

from app.harness.manifest import ToolManifestRegistry
from app.harness.models import HarnessRun


class HarnessVisualProtocol:
    """Build front-end ready payloads from harness execution data."""

    def build_run_payload(
        self,
        run: HarnessRun,
        value_card: dict[str, Any],
        manifests: ToolManifestRegistry | None = None,
    ) -> dict[str, Any]:
        """Return structured payload for radar/timeline/network/safety charts."""

        timeline = self._build_timeline(run)
        discovery_board = self._build_discovery_board(run, manifests)
        security_board = self._build_security_board(run)
        live_agent_board = self._build_live_agent_board(run)
        evidence_board = self._build_evidence_board(run)
        network = self._build_tool_network(run, manifests)
        hero_cards = self._build_hero_cards(run, value_card)
        radar = self._build_value_radar(value_card)
        kpis = self._build_kpis(run, value_card)

        return {
            "schema_version": "1.0.0",
            "query": run.query,
            "completed": run.completed,
            "kpis": kpis,
            "radar": radar,
            "timeline": timeline,
            "discovery_board": discovery_board,
            "security_board": security_board,
            "live_agent_board": live_agent_board,
            "evidence_board": evidence_board,
            "tool_network": network,
            "hero_cards": hero_cards,
            "narrative": value_card.get("narrative", ""),
            "visual_hooks": value_card.get("visual_hooks", []),
        }

    def build_comparison_payload(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate multiple run payloads into one showcase comparison object."""

        if not items:
            return {"count": 0, "rows": [], "best": {}}

        rows: list[dict[str, Any]] = []
        best: dict[str, tuple[str, float]] = {}
        for item in items:
            kpis = item.get("kpis", {})
            name = str(item.get("title", item.get("query", "scenario")))
            row = {
                "title": name,
                "value_index": float(kpis.get("value_index", 0.0)),
                "reliability": float(kpis.get("reliability", 0.0)),
                "safety": float(kpis.get("safety", 0.0)),
                "innovation": float(kpis.get("innovation", 0.0)),
                "tool_calls": float(kpis.get("tool_calls", 0.0)),
                "live_agent_calls": float(kpis.get("live_agent_calls", 0.0)),
                "live_agent_success": float(kpis.get("live_agent_success", 0.0)),
                "completion": float(kpis.get("completion_score", 0.0)),
            }
            rows.append(row)

            for metric in ["value_index", "reliability", "safety", "innovation"]:
                val = float(row[metric])
                current = best.get(metric, ("", -1.0))
                if val > current[1]:
                    best[metric] = (name, val)

        return {
            "count": len(rows),
            "rows": rows,
            "best": {key: {"title": value[0], "score": round(value[1], 3)} for key, value in best.items()},
        }

    @staticmethod
    def _build_kpis(run: HarnessRun, value_card: dict[str, Any]) -> dict[str, float]:
        metrics = run.eval_metrics
        dims = {
            item.get("name"): float(item.get("score", 0.0))
            for item in value_card.get("dimensions", [])
            if isinstance(item, dict)
        }
        return {
            "value_index": float(value_card.get("value_index", 0.0)),
            "tool_calls": float(metrics.get("tool_calls", 0.0)),
            "tool_success_rate": float(metrics.get("tool_success_rate", 0.0)),
            "completion_score": float(metrics.get("completion_score", 0.0)),
            "live_agent_calls": float(metrics.get("live_agent_calls", 0.0)),
            "live_agent_success": float(metrics.get("live_agent_success", 0.0)),
            "evidence_records": float(run.metadata.get("evidence", {}).get("record_count", 0.0)) if isinstance(run.metadata.get("evidence", {}), dict) else 0.0,
            "reliability": dims.get("reliability", 0.0),
            "observability": dims.get("observability", 0.0),
            "adaptability": dims.get("adaptability", 0.0),
            "safety": dims.get("safety", 0.0),
            "innovation": dims.get("innovation", 0.0),
        }

    @staticmethod
    def _build_value_radar(value_card: dict[str, Any]) -> dict[str, Any]:
        labels: list[str] = []
        values: list[float] = []
        for item in value_card.get("dimensions", []):
            if not isinstance(item, dict):
                continue
            labels.append(str(item.get("name", "")))
            values.append(round(float(item.get("score", 0.0)) * 100.0, 2))
        return {
            "labels": labels,
            "values": values,
            "band": value_card.get("band", ""),
        }

    @staticmethod
    def _build_timeline(run: HarnessRun) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cursor = 0.0
        for step in run.steps:
            latency = float(step.tool_result.latency_ms) if step.tool_result else 0.0
            start_ms = cursor
            end_ms = cursor + latency
            cursor = end_ms

            status = "ok"
            if any(note.startswith("BLOCK") for note in step.guardrail_notes):
                status = "blocked"
            elif step.tool_result and not step.tool_result.success:
                status = "error"
            elif step.security.get("action") == "challenge":
                status = "challenge"

            rows.append(
                {
                    "step": step.step,
                    "tool": step.tool_call.name if step.tool_call else "",
                    "source": step.tool_call.source if step.tool_call else "",
                    "start_ms": round(start_ms, 2),
                    "end_ms": round(end_ms, 2),
                    "latency_ms": round(latency, 2),
                    "status": status,
                    "notes": step.guardrail_notes,
                }
            )
        return rows

    @staticmethod
    def _build_discovery_board(run: HarnessRun, manifests: ToolManifestRegistry | None) -> list[dict[str, Any]]:
        board: list[dict[str, Any]] = []
        discovery = run.metadata.get("discovery", [])
        if not isinstance(discovery, list):
            return board
        for item in discovery[:12]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            manifest = manifests.get(name) if manifests else None
            board.append(
                {
                    "name": name,
                    "score": round(float(item.get("score", 0.0)), 4),
                    "reasons": item.get("reasons", []),
                    "risk_level": manifest.risk_level if manifest else "unknown",
                    "novelty": round(float(manifest.novelty_score), 3) if manifest else 0.0,
                    "reliability": round(float(manifest.reliability_score), 3) if manifest else 0.0,
                }
            )
        return board

    @staticmethod
    def _build_security_board(run: HarnessRun) -> dict[str, Any]:
        security = run.metadata.get("security", {})
        if not isinstance(security, dict):
            return {"preflight_action": "unknown", "findings": [], "step_actions": []}

        step_actions: list[dict[str, Any]] = []
        for step in run.steps:
            action = str(step.security.get("action", "allow"))
            step_actions.append(
                {
                    "step": step.step,
                    "tool": step.tool_call.name if step.tool_call else "",
                    "action": action,
                    "risk_score": round(float(step.security.get("risk_score", 0.0)), 3),
                }
            )

        return {
            "preflight_action": security.get("preflight_action", "allow"),
            "preflight_risk_score": round(float(security.get("preflight_risk_score", 0.0)), 3),
            "findings": security.get("preflight_findings", []),
            "step_actions": step_actions,
        }

    @staticmethod
    def _build_live_agent_board(run: HarnessRun) -> dict[str, Any]:
        live = run.metadata.get("live_agent", {})
        if not isinstance(live, dict):
            return {
                "enabled": False,
                "configured": False,
                "model": "",
                "calls_used": 0,
                "call_budget": 0,
                "success": False,
                "notes": [],
                "errors": [],
            }

        analysis = live.get("analysis", {})
        critique = live.get("critique", {})
        return {
            "enabled": bool(live.get("enabled", False)),
            "configured": bool(live.get("configured", False)),
            "model": str(live.get("model", "")),
            "calls_used": int(live.get("calls_used", 0)),
            "call_budget": int(live.get("call_budget", 0)),
            "success": bool(live.get("success", False)),
            "latency_ms": round(float(live.get("latency_ms", 0.0)), 2),
            "analysis": analysis if isinstance(analysis, dict) else {},
            "critique": critique if isinstance(critique, dict) else {},
            "notes": live.get("notes", []),
            "errors": live.get("errors", []),
        }

    @staticmethod
    def _build_evidence_board(run: HarnessRun) -> dict[str, Any]:
        evidence = run.metadata.get("evidence", {})
        if not isinstance(evidence, dict):
            return {"record_count": 0, "citation_count": 0, "records": [], "citations": [], "sources": []}
        return {
            "record_count": int(evidence.get("record_count", 0)),
            "citation_count": int(evidence.get("citation_count", 0)),
            "records": evidence.get("records", []),
            "citations": evidence.get("citations", []),
            "sources": evidence.get("sources", []),
        }

    @staticmethod
    def _build_tool_network(run: HarnessRun, manifests: ToolManifestRegistry | None) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []

        ordered_tools: list[str] = []
        for step in run.steps:
            if step.tool_call:
                ordered_tools.append(step.tool_call.name)

        for tool in ordered_tools:
            manifest = manifests.get(tool) if manifests else None
            nodes.append(
                {
                    "id": tool,
                    "type": manifest.tool_type.value if manifest else "unknown",
                    "novelty": round(float(manifest.novelty_score), 3) if manifest else 0.0,
                    "risk_level": manifest.risk_level if manifest else "unknown",
                }
            )

        for idx in range(len(ordered_tools) - 1):
            links.append(
                {
                    "source": ordered_tools[idx],
                    "target": ordered_tools[idx + 1],
                    "kind": "execution_order",
                }
            )

        if manifests:
            ordered_set = set(ordered_tools)
            for name in ordered_set:
                manifest = manifests.get(name)
                if not manifest:
                    continue
                for target in manifest.compatible_with:
                    if target in ordered_set:
                        links.append({"source": name, "target": target, "kind": "compatibility"})

        uniq_nodes = {}
        for node in nodes:
            uniq_nodes[node["id"]] = node
        return {"nodes": list(uniq_nodes.values()), "links": links}

    @staticmethod
    def _build_hero_cards(run: HarnessRun, value_card: dict[str, Any]) -> list[dict[str, str]]:
        metrics = run.eval_metrics
        dims = {
            item.get("name"): float(item.get("score", 0.0))
            for item in value_card.get("dimensions", [])
            if isinstance(item, dict)
        }
        cards = [
            {
                "title": "Reliability Signal",
                "headline": f"{dims.get('reliability', 0.0) * 100:.0f}% reliable execution",
                "evidence": f"tool_success_rate={metrics.get('tool_success_rate', 0.0):.2f}, "
                f"completion={metrics.get('completion_score', 0.0):.2f}",
            },
            {
                "title": "Safety Signal",
                "headline": f"{dims.get('safety', 0.0) * 100:.0f}% safety posture",
                "evidence": f"security_blocks={metrics.get('security_block_count', 0.0):.1f}, "
                f"security_challenges={metrics.get('security_challenge_count', 0.0):.1f}",
            },
            {
                "title": "Innovation Signal",
                "headline": f"{dims.get('innovation', 0.0) * 100:.0f}% innovation density",
                "evidence": f"discovery_count={metrics.get('discovery_count', 0.0):.1f}, "
                f"utilization={metrics.get('discovery_utilization', 0.0):.2f}",
            },
        ]
        if float(metrics.get("live_agent_calls", 0.0)) > 0.0:
            cards.append(
                {
                    "title": "Live Agent Signal",
                    "headline": f"{metrics.get('live_agent_calls', 0.0):.0f} real-model calls used",
                    "evidence": f"live_success={metrics.get('live_agent_success', 0.0):.2f}",
                }
            )
        return cards
