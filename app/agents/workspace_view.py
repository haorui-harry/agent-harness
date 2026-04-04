"""Streaming workspace/artifact view builders for thread runtime UI consumers."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


class ThreadWorkspaceStreamBuilder:
    """Build front-end friendly workspace stream payloads and HTML snapshots."""

    SCHEMA = "agent-harness-workspace-stream/v1"

    def build(self, thread_payload: dict[str, Any]) -> dict[str, Any]:
        executions = thread_payload.get("executions", []) if isinstance(thread_payload.get("executions", []), list) else []
        artifacts = thread_payload.get("artifacts", []) if isinstance(thread_payload.get("artifacts", []), list) else []
        messages = thread_payload.get("messages", []) if isinstance(thread_payload.get("messages", []), list) else []
        events = thread_payload.get("events", []) if isinstance(thread_payload.get("events", []), list) else []
        completion_packet = self._read_completion_packet(artifacts)
        delivery_bundle = self._read_delivery_bundle(artifacts)
        showcase = self._build_showcase(thread_payload, executions, artifacts)
        timeline = self._build_timeline(events)
        return {
            "schema": self.SCHEMA,
            "thread_id": thread_payload.get("thread_id", ""),
            "header": {
                "title": thread_payload.get("title", ""),
                "status": thread_payload.get("status", ""),
                "agent_name": thread_payload.get("agent_name", ""),
                "latest_query": thread_payload.get("latest_query", ""),
            },
            "workspace": thread_payload.get("workspace", {}),
            "metrics": {
                "message_count": len(messages),
                "artifact_count": len(artifacts),
                "execution_count": len(executions),
                "event_count": len(events),
            },
            "completion_packet": completion_packet,
            "delivery_bundle": delivery_bundle,
            "showcase": showcase,
            "messages": messages[-8:],
            "artifacts": artifacts[-12:],
            "executions": [
                {
                    "execution_id": item.get("execution_id", ""),
                    "label": item.get("label", ""),
                    "status": item.get("status", ""),
                    "completed_nodes": item.get("graph", {}).get("summary", {}).get("completed_nodes", 0),
                    "node_count": item.get("graph", {}).get("summary", {}).get("node_count", 0),
                    "runnable_nodes": item.get("graph", {}).get("summary", {}).get("runnable_nodes", []),
                }
                for item in executions[-8:]
            ],
            "timeline": timeline,
            "stream_events": events[-20:],
        }

    def to_html(self, payload: dict[str, Any]) -> str:
        header = payload.get("header", {}) if isinstance(payload.get("header", {}), dict) else {}
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
        showcase = payload.get("showcase", {}) if isinstance(payload.get("showcase", {}), dict) else {}
        delivery_bundle = payload.get("delivery_bundle", {}) if isinstance(payload.get("delivery_bundle", {}), dict) else {}
        messages = payload.get("messages", []) if isinstance(payload.get("messages", []), list) else []
        artifacts = payload.get("artifacts", []) if isinstance(payload.get("artifacts", []), list) else []
        executions = payload.get("executions", []) if isinstance(payload.get("executions", []), list) else []
        timeline = payload.get("timeline", []) if isinstance(payload.get("timeline", []), list) else []
        events = payload.get("stream_events", []) if isinstance(payload.get("stream_events", []), list) else []
        workspace = payload.get("workspace", {}) if isinstance(payload.get("workspace", {}), dict) else {}
        deliverable_index = delivery_bundle.get("deliverable_index", []) if isinstance(delivery_bundle.get("deliverable_index", []), list) else []
        artifact_manifest = delivery_bundle.get("artifact_manifest", []) if isinstance(delivery_bundle.get("artifact_manifest", []), list) else []
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent Workspace Stream</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --ink: #14213d;
      --muted: #5e6472;
      --accent: #e76f51;
      --accent-2: #2a9d8f;
      --card: rgba(255,250,243,0.92);
      --line: rgba(20,33,61,0.12);
      --shadow: 0 18px 50px rgba(20,33,61,0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: 'Segoe UI', sans-serif; background:
      radial-gradient(circle at top left, rgba(231,111,81,0.18), transparent 28%),
      radial-gradient(circle at top right, rgba(42,157,143,0.16), transparent 24%),
      linear-gradient(180deg, #fffaf2 0%, var(--bg) 100%);
      color:var(--ink); }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    .hero {{ padding: 32px; border-radius: 28px; background:
      linear-gradient(135deg, rgba(231,111,81,0.16), rgba(42,157,143,0.10) 58%, rgba(20,33,61,0.06));
      border: 1px solid var(--line); box-shadow: var(--shadow); }}
    .kicker {{ text-transform: uppercase; letter-spacing: 0.12em; font-size: 12px; color: var(--accent-2); }}
    h1 {{ margin: 10px 0 10px; font-size: 44px; line-height: 1.05; }}
    .lede {{ max-width: 860px; font-size: 18px; line-height: 1.6; color: var(--muted); }}
    .status {{ display:inline-flex; gap:10px; align-items:center; padding: 10px 14px; border-radius: 999px; background: rgba(255,255,255,0.55); border: 1px solid var(--line); margin-top: 14px; }}
    .grid {{ display:grid; grid-template-columns: repeat(12, 1fr); gap: 16px; margin-top: 20px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 18px; box-shadow: var(--shadow); }}
    .span-3 {{ grid-column: span 3; }} .span-4 {{ grid-column: span 4; }} .span-6 {{ grid-column: span 6; }} .span-8 {{ grid-column: span 8; }} .span-12 {{ grid-column: span 12; }}
    .value {{ font-size: 34px; font-weight: 700; margin-top: 8px; }}
    .title {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: #4f8fb2; }}
    .section-title {{ margin:0 0 10px; font-size: 22px; }}
    .muted {{ color: var(--muted); }}
    ul {{ padding-left: 18px; margin: 10px 0 0; }}
    li {{ margin: 7px 0; line-height: 1.5; }}
    code {{ background: rgba(20,33,61,0.06); padding: 2px 6px; border-radius: 8px; }}
    .artifact-list li, .event-list li {{ font-size: 14px; }}
    .hero-grid {{ display:grid; grid-template-columns: 1.4fr 0.9fr; gap: 18px; margin-top: 18px; }}
    .mini-panel {{ background: rgba(255,255,255,0.52); border: 1px solid var(--line); border-radius: 18px; padding: 16px; }}
    .quote {{ font-size: 17px; line-height: 1.7; white-space: pre-wrap; }}
    table {{ width:100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 8px 6px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .span-3, .span-4, .span-6, .span-8, .span-12 {{ grid-column: span 1; }}
      .hero-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 34px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="kicker">Executable Agent Result</div>
      <h1>{html.escape(str(showcase.get("title", header.get("title", ""))))}</h1>
      <div class="lede">{html.escape(str(showcase.get("summary", "No summary available.")))}</div>
      <div class="status">
        <strong>{html.escape(str(header.get("status", "")))}</strong>
        <span>Task: {html.escape(str(showcase.get("task", header.get("latest_query", ""))))}</span>
      </div>
      <div class="hero-grid">
        <div class="mini-panel">
          <div class="title">Delivery Index</div>
          <ul>{"".join(f"<li>{html.escape(item)}</li>" for item in showcase.get("deliverables", [])) or "<li>No deliverables summarized.</li>"}</ul>
        </div>
        <div class="mini-panel">
          <div class="title">Closure Signals</div>
          <ul>{"".join(f"<li>{html.escape(item)}</li>" for item in showcase.get("value_points", [])) or "<li>No value points summarized.</li>"}</ul>
        </div>
      </div>
    </section>
    <section class="grid">
      <div class="card span-5">
        <div class="title">Delivery Bundle</div>
        <h3 class="section-title">Deliverable Index</h3>
        <table><thead><tr><th>Family</th><th>Count</th><th>Representative Outputs</th></tr></thead><tbody>{"".join(f"<tr><td>{html.escape(str(item.get('family','')))}</td><td>{int(item.get('count',0))}</td><td>{html.escape(', '.join(str(path).split('/')[-1] for path in item.get('paths', [])[:3]))}</td></tr>" for item in deliverable_index) or "<tr><td colspan='3'>No delivery bundle index.</td></tr>"}</tbody></table>
      </div>
      <div class="card span-7">
        <div class="title">Manifest</div>
        <h3 class="section-title">Openable Artifact Manifest</h3>
        <table><thead><tr><th>Artifact</th><th>Family</th><th>Kind</th><th>Summary</th></tr></thead><tbody>{"".join(f"<tr><td><code>{html.escape(str(item.get('path','')).replace('\\\\','/').split('/')[-1])}</code></td><td>{html.escape(str(item.get('family','')))}</td><td>{html.escape(str(item.get('kind','')))}</td><td>{html.escape(str(item.get('summary','')))}</td></tr>" for item in artifact_manifest[:10]) or "<tr><td colspan='4'>No artifact manifest.</td></tr>"}</tbody></table>
      </div>
      <div class="card span-6">
        <div class="title">Agent Computer</div>
        <h3 class="section-title">Workspace Environment</h3>
        <table><thead><tr><th>Area</th><th>Path</th></tr></thead><tbody>
          <tr><td>workspace</td><td><code>{html.escape(str(workspace.get("workspace", "")))}</code></td></tr>
          <tr><td>outputs</td><td><code>{html.escape(str(workspace.get("outputs", "")))}</code></td></tr>
          <tr><td>uploads</td><td><code>{html.escape(str(workspace.get("uploads", "")))}</code></td></tr>
        </tbody></table>
      </div>
      <div class="card span-6">
        <div class="title">Artifact Rail</div>
        <h3 class="section-title">What You Can Open</h3>
        <table><thead><tr><th>Artifact</th><th>Kind</th><th>Summary</th></tr></thead><tbody>{"".join(f"<tr><td><code>{html.escape(str(item.get('relative_path', item.get('name',''))))}</code></td><td>{html.escape(str(item.get('kind','')))}</td><td>{html.escape(str(item.get('summary','')))}</td></tr>" for item in artifacts) or "<tr><td colspan='3'>No artifacts.</td></tr>"}</tbody></table>
      </div>
    </section>
    <section class="grid">
      <div class="card span-3"><div class="title">Messages</div><div class="value">{int(metrics.get("message_count", 0))}</div></div>
      <div class="card span-3"><div class="title">Artifacts</div><div class="value">{int(metrics.get("artifact_count", 0))}</div></div>
      <div class="card span-3"><div class="title">Executions</div><div class="value">{int(metrics.get("execution_count", 0))}</div></div>
      <div class="card span-3"><div class="title">Events</div><div class="value">{int(metrics.get("event_count", 0))}</div></div>
      <div class="card span-8">
        <div class="title">Result Snapshot</div>
        <h3 class="section-title">{html.escape(str(showcase.get("result_title", "Result")))}</h3>
        <div class="quote">{html.escape(str(showcase.get("result_body", "No result body extracted.")))}</div>
      </div>
      <div class="card span-4">
        <div class="title">Execution Summary</div>
        <ul>{"".join(f"<li><code>{html.escape(str(item.get('label','')))}</code> {html.escape(str(item.get('status','')))} {int(item.get('completed_nodes',0))}/{int(item.get('node_count',0))}</li>" for item in executions) or "<li>No executions.</li>"}</ul>
      </div>
      <div class="card span-6">
        <div class="title">Artifacts</div>
        <ul class="artifact-list">{"".join(f"<li><code>{html.escape(str(item.get('name','')))}</code> {html.escape(str(item.get('kind','')))}<br /><span class='muted'>{html.escape(str(item.get('summary','')))}</span></li>" for item in artifacts) or "<li>No artifacts.</li>"}</ul>
      </div>
      <div class="card span-6">
        <div class="title">Node Timeline</div>
        <ul class="event-list">{"".join(f"<li><strong>{html.escape(str(item.get('node_title','')) or str(item.get('node_id','')))}</strong> <code>{html.escape(str(item.get('phase','')))}</code><br /><span class='muted'>{html.escape(str(item.get('summary','')))}</span>{'<br /><code>' + html.escape(str(item.get('artifact_relative_path',''))) + '</code>' if str(item.get('artifact_relative_path','')) else ''}</li>" for item in timeline) or "<li>No node timeline.</li>"}</ul>
      </div>
      <div class="card span-6">
        <div class="title">Event Stream</div>
        <ul class="event-list">{"".join(f"<li><code>{html.escape(str(item.get('event','')))}</code> {html.escape(str(item.get('timestamp','')))}</li>" for item in events) or "<li>No events.</li>"}</ul>
      </div>
      <div class="card span-12">
        <div class="title">Messages</div>
        <ul>{"".join(f"<li><strong>{html.escape(str(item.get('role','')))}</strong>: {html.escape(str(item.get('content',''))[:220])}</li>" for item in messages) or "<li>No messages.</li>"}</ul>
      </div>
    </section>
  </div>
</body>
</html>
"""

    def _build_showcase(
        self,
        thread_payload: dict[str, Any],
        executions: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        latest_execution = executions[-1] if executions else {}
        graph = latest_execution.get("graph", {}) if isinstance(latest_execution.get("graph", {}), dict) else {}
        task = str(graph.get("query", "") or thread_payload.get("latest_query", "")).strip()
        completion_packet = self._read_completion_packet(artifacts)
        delivery_bundle = self._read_delivery_bundle(artifacts)
        report_text = self._read_text_artifact(artifacts, preferred_kind="file_artifact")
        plan_payload = self._read_json_artifact(artifacts, preferred_kind="skill_result")
        workspace_payload = self._read_json_artifact(artifacts, preferred_kind="workspace_snapshot")
        tool_payload = self._read_json_artifact(artifacts, preferred_kind="tool_result")
        deliverables: list[str] = []
        value_points: list[str] = []
        artifact_family_notes = self._artifact_family_notes(artifacts)
        preview_title = "Artifact Preview"
        preview_body = ""
        summary = (
            f"This thread executed a concrete task inside a persistent agent workspace. "
            f"It grounded on local files, generated reviewable artifacts, and exposed the execution computer "
            f"through workspace paths, event history, and artifact outputs."
        )
        if delivery_bundle:
            bundle_summary = delivery_bundle.get("bundle_summary", {}) if isinstance(delivery_bundle.get("bundle_summary", {}), dict) else {}
            bundle_index = delivery_bundle.get("deliverable_index", []) if isinstance(delivery_bundle.get("deliverable_index", []), list) else []
            manifest = delivery_bundle.get("artifact_manifest", []) if isinstance(delivery_bundle.get("artifact_manifest", []), list) else []
            primary = delivery_bundle.get("primary_deliverable", {}) if isinstance(delivery_bundle.get("primary_deliverable", {}), dict) else {}
            baseline = delivery_bundle.get("baseline_comparison", {}) if isinstance(delivery_bundle.get("baseline_comparison", {}), dict) else {}
            primary_name = Path(str(primary.get("path", ""))).name if str(primary.get("path", "")).strip() else ""
            deliverables.append(
                f"Published a delivery bundle with a primary deliverable and {int(bundle_summary.get('artifact_count', len(manifest)))} tracked supporting artifacts."
            )
            if primary_name:
                deliverables.append(f"Primary deliverable: {primary_name}.")
            for item in bundle_index[:3]:
                if not isinstance(item, dict):
                    continue
                deliverables.append(
                    f"{str(item.get('family', '')).title()} bundle: {', '.join(str(path).replace('\\\\','/').split('/')[-1] for path in item.get('paths', [])[:2])}."
                )
            value_points.append(
                f"Bundle validation status: {str(bundle_summary.get('validation_status', 'unknown')).replace('_', ' ')}."
            )
            if isinstance(baseline.get("harness_additions", []), list) and baseline.get("harness_additions", []):
                value_points.append(str(baseline.get("harness_additions", [])[0]))
            else:
                value_points.append(
                    f"Bundle evidence count: {int(bundle_summary.get('evidence_count', 0))}; risk items: {int(bundle_summary.get('risk_count', 0))}."
                )
            preview_title = Path(str(primary.get("path", ""))).name if str(primary.get("path", "")).strip() else "Delivery Bundle"
            preview_body = str(primary.get("excerpt", "")).strip() or self._format_delivery_bundle_preview(delivery_bundle)
            if primary_name:
                summary = (
                    f"This thread culminates in {primary_name} as the main result, with the delivery bundle acting as the inspection rail behind it. "
                    f"The first screen prioritizes the actual deliverable first, then the evidence and artifact manifest needed to verify it."
                )
            else:
                summary = (
                    f"This thread ends in a delivery bundle with one primary deliverable and a support manifest behind it. "
                    f"The first screen now prioritizes the actual result, then the evidence and artifact rail needed to inspect it."
                )
        if completion_packet:
            packet_summary = completion_packet.get("summary", {}) if isinstance(completion_packet.get("summary", {}), dict) else {}
            delivered = completion_packet.get("delivered_artifacts", []) if isinstance(completion_packet.get("delivered_artifacts", []), list) else []
            state_gap = completion_packet.get("state_gap", {}) if isinstance(completion_packet.get("state_gap", {}), dict) else {}
            validation = completion_packet.get("validation", {}) if isinstance(completion_packet.get("validation", {}), dict) else {}
            evidence = completion_packet.get("evidence", {}) if isinstance(completion_packet.get("evidence", {}), dict) else {}
            primary = completion_packet.get("primary_deliverable", {}) if isinstance(completion_packet.get("primary_deliverable", {}), dict) else {}
            deliverables.append(
                f"Closed execution into a completion packet with {int(packet_summary.get('artifact_count', len(delivered)))} tracked artifacts."
            )
            value_points.append(
                f"Packet evidence count: {int(evidence.get('record_count', 0))}, citations: {int(evidence.get('citation_count', 0))}."
            )
            open_gap_count = (
                len(state_gap.get("missing_channels", [])) if isinstance(state_gap.get("missing_channels", []), list) else 0
            ) + (
                len(state_gap.get("missing_artifacts", [])) if isinstance(state_gap.get("missing_artifacts", []), list) else 0
            ) + (
                len(state_gap.get("failure_types", [])) if isinstance(state_gap.get("failure_types", []), list) else 0
            ) + (1 if state_gap.get("missing_validation") else 0)
            value_points.append(f"Open execution gaps: {open_gap_count}.")
            if not delivery_bundle:
                preview_title = Path(str(primary.get("path", ""))).name if str(primary.get("path", "")).strip() else "Completion Packet"
                preview_body = str(primary.get("excerpt", "")).strip() or self._format_completion_packet_preview(completion_packet)
                summary = (
                    f"This thread produced a unified completion packet that centers the final deliverable and records the evidence, "
                    f"validation state, risk notes, and remaining gaps around it."
                )
        if workspace_payload:
            deliverables.append(
                f"Scanned workspace and found {int(workspace_payload.get('file_count', 0))} relevant file(s)."
            )
            files = workspace_payload.get("files", [])
            if isinstance(files, list) and files:
                value_points.append(f"Grounded the task on actual workspace file: {files[0]}.")
        if tool_payload:
            output = tool_payload.get("output", {}) if isinstance(tool_payload.get("output", {}), dict) else {}
            matches = output.get("matches", [])
            if isinstance(matches, list) and matches:
                deliverables.append(f"Selected tools by searching capability catalog ({len(matches)} match(es)).")
        if plan_payload:
            result_text = str(plan_payload.get("output", "")).strip()
            if result_text:
                deliverables.append("Generated an executable engineering brief instead of a generic summary.")
        deliverables.extend(item["deliverable"] for item in artifact_family_notes)
        if report_text:
            deliverables.append("Wrote a final markdown report artifact for handoff and review.")
        value_points.extend(item["value"] for item in artifact_family_notes)
        value_points.extend(
            [
                "This run produced inspectable artifacts, not just a final answer string.",
                "The result can be resumed, retried, and audited inside persistent thread state.",
                "Workspace paths and artifact files are exposed directly so the user can inspect the agent computer state.",
            ]
        )
        plan_output = str(plan_payload.get("output", "")).strip() if plan_payload else ""
        if not preview_body:
            preview = self._read_preview_artifact(artifacts)
            preview_body = report_text or preview.get("content", "") or plan_output or "No report generated."
            preview_title = preview.get("title", "Artifact Preview")
        summary_focus = ", ".join(item["family"] for item in artifact_family_notes[:3])
        if summary_focus:
            summary += f" Primary artifact families: {summary_focus}."
        result_body = preview_body[:1200]
        prioritized_deliverables = list(
            dict.fromkeys([item["deliverable"] for item in artifact_family_notes] + deliverables)
        )
        primary_artifact_kind = ""
        primary_artifact_path = ""
        if delivery_bundle:
            primary = delivery_bundle.get("primary_deliverable", {}) if isinstance(delivery_bundle.get("primary_deliverable", {}), dict) else {}
            primary_artifact_path = str(primary.get("path", "")).strip()
            normalized_primary = primary_artifact_path.replace("\\", "/").lower()
            if "/reports/" in normalized_primary or normalized_primary.endswith(".md"):
                primary_artifact_kind = "report"
            elif "/briefs/" in normalized_primary:
                primary_artifact_kind = "brief"
            elif primary_artifact_path:
                primary_artifact_kind = "deliverable"
            else:
                primary_artifact_kind = "delivery_bundle"
                primary_artifact_path = str(delivery_bundle.get("_artifact_path", ""))
        elif completion_packet:
            primary_artifact_kind = "completion_packet"
            primary_artifact_path = str(completion_packet.get("_artifact_path", ""))
        return {
            "title": str(thread_payload.get("title", "") or "Agent Result"),
            "task": task or "No task captured.",
            "summary": summary,
            "deliverables": prioritized_deliverables[:4],
            "value_points": value_points[:4],
            "result_title": preview_title,
            "result_body": result_body,
            "primary_artifact": {
                "kind": primary_artifact_kind,
                "path": primary_artifact_path,
            },
        }

    @staticmethod
    def _build_timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            if str(event.get("event", "")) not in {"node_started", "node_result", "artifact_written", "node_completed"}:
                continue
            rows.append(
                {
                    "event_id": int(event.get("event_id", 0)),
                    "event": str(event.get("event", "")),
                    "phase": str(event.get("phase", "")),
                    "timestamp": str(event.get("timestamp", "")),
                    "execution_id": str(event.get("execution_id", "")),
                    "node_id": str(event.get("node_id", "")),
                    "node_title": str(event.get("node_title", "")),
                    "node_type": str(event.get("node_type", "")),
                    "summary": str(event.get("summary", "")),
                    "artifact_relative_path": str(event.get("artifact_relative_path", "")),
                }
            )
        return rows[-20:]

    @staticmethod
    def _artifact_family_notes(artifacts: list[dict[str, Any]]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        mapping = [
            ("web", ["web/"], "Prepared a landing-page or webpage blueprint artifact.", "Turns a generic request into a page structure someone can build next."),
            ("slides", ["slides/"], "Prepared a slide deck plan with narrative beats and proof moments.", "Makes the result presentable to executives instead of burying it in prose."),
            ("charts", ["charts/"], "Prepared a chart pack spec with renderable data contracts.", "Creates a visualization-ready surface rather than only describing insights."),
            ("podcast", ["podcast/"], "Prepared a podcast episode plan with segment structure.", "Extends the framework into audio-ready content packaging."),
            ("video", ["video/"], "Prepared a video storyboard with scene-by-scene beats.", "Creates a media artifact that downstream teams can directly produce."),
            ("images", ["images/"], "Prepared an image prompt pack for reusable visual generation.", "Supports visual asset production without locking to one generator."),
            ("analysis", ["analysis/data-analysis-spec"], "Prepared a data-analysis specification with metrics and outputs.", "Pushes the task toward reproducible analytics rather than loose commentary."),
            ("benchmarks", ["benchmarks/"], "Prepared benchmark execution artifacts.", "Makes evaluation portable and reproducible."),
            ("datasets", ["datasets/"], "Prepared external data pull or loader artifacts.", "Grounds research and analysis tasks in reproducible evidence collection."),
        ]
        for artifact in artifacts:
            rel = str(artifact.get("relative_path", artifact.get("name", ""))).lower()
            for family, patterns, deliverable, value in mapping:
                if family in seen or not any(pattern in rel for pattern in patterns):
                    continue
                seen.add(family)
                rows.append({"family": family, "deliverable": deliverable, "value": value})
        return rows

    @staticmethod
    def _read_completion_packet(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        for artifact in reversed(artifacts):
            rel = str(artifact.get("relative_path", artifact.get("name", ""))).replace("\\", "/").lower()
            if "packets/completion-packet.json" not in rel:
                continue
            path = Path(str(artifact.get("path", "")))
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            payload["_artifact_path"] = str(path)
            return payload
        return {}

    @staticmethod
    def _read_delivery_bundle(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        for artifact in reversed(artifacts):
            rel = str(artifact.get("relative_path", artifact.get("name", ""))).replace("\\", "/").lower()
            if "bundles/delivery-bundle.json" not in rel:
                continue
            path = Path(str(artifact.get("path", "")))
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            payload["_artifact_path"] = str(path)
            return payload
        return {}

    @staticmethod
    def _format_completion_packet_preview(packet: dict[str, Any]) -> str:
        summary = packet.get("summary", {}) if isinstance(packet.get("summary", {}), dict) else {}
        validation = packet.get("validation", {}) if isinstance(packet.get("validation", {}), dict) else {}
        evidence = packet.get("evidence", {}) if isinstance(packet.get("evidence", {}), dict) else {}
        risk = packet.get("risk", {}) if isinstance(packet.get("risk", {}), dict) else {}
        next_steps = packet.get("next_steps", []) if isinstance(packet.get("next_steps", []), list) else []
        delivered = packet.get("delivered_artifacts", []) if isinstance(packet.get("delivered_artifacts", []), list) else []
        primary = packet.get("primary_deliverable", {}) if isinstance(packet.get("primary_deliverable", {}), dict) else {}
        lines = [
            f"Task: {packet.get('query', '')}",
            f"Artifacts: {int(summary.get('artifact_count', len(delivered)))}",
            f"Validation: {validation.get('status', 'unknown')}",
            f"Evidence records: {int(evidence.get('record_count', 0))}",
            f"Risk items: {int(risk.get('count', 0))}",
            "",
            "Primary deliverable:",
            f"- {Path(str(primary.get('path', 'deliverable'))).name if str(primary.get('path', '')).strip() else 'No primary deliverable path'}",
        ]
        excerpt = str(primary.get("excerpt", "")).strip()
        if excerpt:
            lines.extend(["", excerpt[:600], ""])
        else:
            lines.append("")
        lines.extend([
            "Delivered artifacts:",
        ])
        for item in delivered[:5]:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {Path(str(item.get('path', 'artifact'))).name}: {item.get('summary', '')}")
        if next_steps:
            lines.append("")
            lines.append("Next steps:")
            for item in next_steps[:4]:
                lines.append(f"- {item}")
        return "\n".join(lines).strip()

    @staticmethod
    def _format_delivery_bundle_preview(bundle: dict[str, Any]) -> str:
        summary = bundle.get("bundle_summary", {}) if isinstance(bundle.get("bundle_summary", {}), dict) else {}
        deliverable_index = bundle.get("deliverable_index", []) if isinstance(bundle.get("deliverable_index", []), list) else []
        manifest = bundle.get("artifact_manifest", []) if isinstance(bundle.get("artifact_manifest", []), list) else []
        checklist = bundle.get("reviewer_checklist", []) if isinstance(bundle.get("reviewer_checklist", []), list) else []
        primary = bundle.get("primary_deliverable", {}) if isinstance(bundle.get("primary_deliverable", {}), dict) else {}
        baseline = bundle.get("baseline_comparison", {}) if isinstance(bundle.get("baseline_comparison", {}), dict) else {}
        lines = [
            f"Task: {bundle.get('query', '')}",
            f"Artifact count: {int(summary.get('artifact_count', len(manifest)))}",
            f"Deliverable families: {int(summary.get('family_count', len(deliverable_index)))}",
            f"Validation: {summary.get('validation_status', 'unknown')}",
            "",
            "Primary deliverable:",
            f"- {Path(str(primary.get('path', 'deliverable'))).name if str(primary.get('path', '')).strip() else 'No primary deliverable path'}",
        ]
        excerpt = str(primary.get("excerpt", "")).strip()
        if excerpt:
            lines.extend(["", excerpt[:700], ""])
        else:
            lines.append("")
        lines.extend([
            "Deliverable index:",
        ])
        for item in deliverable_index[:5]:
            if not isinstance(item, dict):
                continue
            family = str(item.get("family", "")).title()
            paths = ", ".join(str(path).replace("\\", "/").split("/")[-1] for path in item.get("paths", [])[:3])
            lines.append(f"- {family}: {paths}")
        additions = baseline.get("harness_additions", []) if isinstance(baseline.get("harness_additions", []), list) else []
        if additions:
            lines.append("")
            lines.append("What harness added beyond a direct answer:")
            for item in additions[:3]:
                lines.append(f"- {item}")
        if checklist:
            lines.append("")
            lines.append("Reviewer checklist:")
            for item in checklist[:4]:
                lines.append(f"- {item}")
        return "\n".join(lines).strip()

    @staticmethod
    def _read_preview_artifact(artifacts: list[dict[str, Any]]) -> dict[str, str]:
        for artifact in reversed(artifacts):
            path = Path(str(artifact.get("path", "")))
            if not path.exists() or path.suffix.lower() not in {".md", ".txt", ".diff", ".py", ".json"}:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            title = Path(str(artifact.get("relative_path", artifact.get("name", path.name)))).name
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(content)
                    content = json.dumps(payload, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            return {"title": f"Artifact Preview: {title}", "content": content}
        return {"title": "Artifact Preview", "content": ""}

    @staticmethod
    def _read_text_artifact(artifacts: list[dict[str, Any]], preferred_kind: str = "") -> str:
        for artifact in reversed(artifacts):
            if preferred_kind and str(artifact.get("kind", "")) != preferred_kind:
                continue
            path = Path(str(artifact.get("path", "")))
            if not path.exists():
                continue
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                continue
        return ""

    @staticmethod
    def _read_json_artifact(artifacts: list[dict[str, Any]], preferred_kind: str = "") -> dict[str, Any]:
        for artifact in reversed(artifacts):
            if preferred_kind and str(artifact.get("kind", "")) != preferred_kind:
                continue
            path = Path(str(artifact.get("path", "")))
            if not path.exists():
                continue
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
        return {}
