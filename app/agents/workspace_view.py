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
        showcase = self._build_showcase(thread_payload, executions, artifacts)
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
            "stream_events": events[-20:],
        }

    def to_html(self, payload: dict[str, Any]) -> str:
        header = payload.get("header", {}) if isinstance(payload.get("header", {}), dict) else {}
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
        showcase = payload.get("showcase", {}) if isinstance(payload.get("showcase", {}), dict) else {}
        messages = payload.get("messages", []) if isinstance(payload.get("messages", []), list) else []
        artifacts = payload.get("artifacts", []) if isinstance(payload.get("artifacts", []), list) else []
        executions = payload.get("executions", []) if isinstance(payload.get("executions", []), list) else []
        events = payload.get("stream_events", []) if isinstance(payload.get("stream_events", []), list) else []
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
          <div class="title">What Was Produced</div>
          <ul>{"".join(f"<li>{html.escape(item)}</li>" for item in showcase.get("deliverables", [])) or "<li>No deliverables summarized.</li>"}</ul>
        </div>
        <div class="mini-panel">
          <div class="title">Why It Matters</div>
          <ul>{"".join(f"<li>{html.escape(item)}</li>" for item in showcase.get("value_points", [])) or "<li>No value points summarized.</li>"}</ul>
        </div>
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
        report_text = self._read_text_artifact(artifacts, preferred_kind="file_artifact")
        plan_payload = self._read_json_artifact(artifacts, preferred_kind="skill_result")
        workspace_payload = self._read_json_artifact(artifacts, preferred_kind="workspace_snapshot")
        tool_payload = self._read_json_artifact(artifacts, preferred_kind="tool_result")
        deliverables: list[str] = []
        value_points: list[str] = []
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
        if report_text:
            deliverables.append("Wrote a final markdown report artifact for handoff and review.")
        value_points.extend(
            [
                "This run produced inspectable artifacts, not just a final answer string.",
                "The result can be resumed, retried, and audited inside persistent thread state.",
            ]
        )
        summary = (
            f"This thread executed a generic agent task against a real workspace. "
            f"It inspected project files, discovered usable tools, produced a structured task brief, "
            f"and wrote a reviewable report artifact."
        )
        result_body = report_text or str(plan_payload.get("output", "No report generated.")) if plan_payload else "No report generated."
        result_body = result_body[:1200]
        return {
            "title": str(thread_payload.get("title", "") or "Agent Result"),
            "task": task or "No task captured.",
            "summary": summary,
            "deliverables": deliverables[:4],
            "value_points": value_points[:4],
            "result_title": "Final Report Excerpt",
            "result_body": result_body,
        }

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
