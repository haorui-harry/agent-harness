"""Task-graph node to runtime-action mapper."""

from __future__ import annotations

import fnmatch
import json
import re
import time
from pathlib import Path
from typing import Any

from app.agents.sandbox import ThreadSandbox
from app.core.tasking import (
    allowed_workspace_action_kinds,
    build_world_state,
    compute_state_gap,
    default_capability_registry,
    default_workspace_action_specs,
    infer_task_spec,
    plan_capability_path,
    workspace_action_result_field,
)
from app.skills.registry import execute_skill


class TaskGraphActionMapper:
    """Map generic task-graph nodes into concrete thread-runtime actions."""

    def __init__(self, tool_registry: Any | None = None) -> None:
        if tool_registry is None:
            from app.harness.tools import ToolRegistry

            tool_registry = ToolRegistry()
        self.tools = tool_registry

    def execute_node(
        self,
        *,
        sandbox: ThreadSandbox,
        execution_id: str,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        node_id = str(node.get("node_id", "node"))
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        delay_ms = max(0.0, float(metrics.get("delay_ms", 0.0)))
        if delay_ms:
            time.sleep(delay_ms / 1000.0)

        node_type = str(node.get("node_type", "artifact"))
        if node_type in {"tool_call", "tool"}:
            return self._execute_tool_node(sandbox=sandbox, execution_id=execution_id, node=node, graph=graph, context=context)
        if node_type in {"skill_call", "skill"}:
            return self._execute_skill_node(sandbox=sandbox, execution_id=execution_id, node=node, graph=graph, context=context)
        if node_type in {"workspace_snapshot", "workspace_read"}:
            return self._execute_workspace_snapshot(sandbox=sandbox, execution_id=execution_id, node=node, graph=graph, context=context)
        if node_type in {"workspace_action", "workspace_write"}:
            return self._execute_workspace_action(sandbox=sandbox, execution_id=execution_id, node=node, graph=graph, context=context)
        if node_type in {"file_write", "write_file"}:
            return self._execute_file_write(sandbox=sandbox, node=node, context=context)
        if node_type in {"command", "shell_command", "execution", "validation"}:
            return self._execute_command_node(sandbox=sandbox, execution_id=execution_id, node=node, graph=graph, context=context)
        if node_type in {"subagent", "delegated_agent"}:
            return self._execute_subagent_node(sandbox=sandbox, execution_id=execution_id, node=node, graph=graph, context=context)
        if node_type in {"graph_replan", "replan"}:
            return self._execute_graph_replan(sandbox=sandbox, execution_id=execution_id, node=node, graph=graph, context=context)
        return self._execute_static_node(sandbox=sandbox, execution_id=execution_id, node=node, graph=graph, context=context)

    def _execute_tool_node(
        self,
        *,
        sandbox: ThreadSandbox,
        execution_id: str,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        tool_name = str(metrics.get("tool_name", metrics.get("name", ""))).strip()
        tool_args = dict(metrics.get("tool_args", {})) if isinstance(metrics.get("tool_args", {}), dict) else {}
        if not tool_name:
            raise ValueError("tool_call node requires metrics.tool_name")

        workspace = sandbox.workspace_paths()
        if tool_name.startswith("workspace_"):
            tool_args.setdefault("workspace_root", workspace.get("workspace", ""))
        if tool_name == "task_graph_builder":
            tool_args.setdefault("workspace_root", workspace.get("workspace", ""))
        if "query" not in tool_args and context.get("query"):
            tool_args["query"] = context["query"]

        from app.harness.models import ToolCall

        result = self.tools.call(ToolCall(name=tool_name, tool_type=self.tools.infer_tool_type(tool_name), args=tool_args))
        body = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "node_type": node.get("node_type", ""),
            "tool_name": tool_name,
            "tool_args": tool_args,
            "success": bool(result.success),
            "latency_ms": round(float(result.latency_ms), 2),
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
            "graph_id": graph.get("graph_id", ""),
        }
        return self._write_json_result(
            sandbox=sandbox,
            execution_id=execution_id,
            node=node,
            body=body,
            summary=f"tool {tool_name} {'ok' if result.success else 'failed'}",
            kind="tool_result",
        )

    def _execute_skill_node(
        self,
        *,
        sandbox: ThreadSandbox,
        execution_id: str,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        skill_name = str(metrics.get("skill_name", "")).strip()
        if not skill_name:
            raise ValueError("skill_call node requires metrics.skill_name")

        prompt = str(metrics.get("prompt", context.get("query", ""))).strip()
        source_text = self._collect_source_text(metrics=metrics, context=context)
        input_text = "\n\n".join(item for item in [prompt, source_text] if item).strip()
        output = execute_skill(skill_name, input_text)
        body = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "node_type": node.get("node_type", ""),
            "skill_name": skill_name,
            "input": input_text,
            "output": output,
            "graph_id": graph.get("graph_id", ""),
        }
        return self._write_json_result(
            sandbox=sandbox,
            execution_id=execution_id,
            node=node,
            body=body,
            summary=f"skill {skill_name} executed",
            kind="skill_result",
        )

    def _execute_workspace_snapshot(
        self,
        *,
        sandbox: ThreadSandbox,
        execution_id: str,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        area = str(metrics.get("area", "workspace"))
        glob = str(metrics.get("glob", "*")).strip() or "*"
        max_files = max(1, min(int(metrics.get("max_files", 12)), 100))
        preview_limit = max(0, min(int(metrics.get("preview_limit", 4)), 20))
        files = [path for path in sandbox.list_files(area) if fnmatch.fnmatch(path, glob) or fnmatch.fnmatch(Path(path).name, glob)]
        previews: list[dict[str, Any]] = []
        for relative_path in files[:preview_limit]:
            try:
                preview = sandbox.read_text(relative_path, area=area)[:800]
            except Exception:
                preview = ""
            previews.append({"relative_path": relative_path, "preview": preview})
        body = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "node_type": node.get("node_type", ""),
            "area": area,
            "glob": glob,
            "file_count": len(files),
            "files": files[:max_files],
            "previews": previews,
            "graph_id": graph.get("graph_id", ""),
            "context_keys": sorted(context.keys()),
        }
        return self._write_json_result(
            sandbox=sandbox,
            execution_id=execution_id,
            node=node,
            body=body,
            summary=f"workspace snapshot captured with {len(files)} files",
            kind="workspace_snapshot",
        )

    def _execute_file_write(
        self,
        *,
        sandbox: ThreadSandbox,
        node: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        area = str(metrics.get("area", "outputs"))
        relative_path = str(metrics.get("relative_path", f"{node.get('node_id', 'node')}.txt")).strip()
        prefix = str(metrics.get("content_prefix", ""))
        content = str(metrics.get("content", ""))
        if not content:
            content = self._extract_result_field(
                source_node_id=str(metrics.get("source_node_id", "")),
                field=str(metrics.get("result_field", "output")),
                context=context,
            )
        content = prefix + content
        target = sandbox.write_text(relative_path, content, area=area)
        artifact = {
            "kind": "file_artifact",
            "label": str(node.get("title", relative_path)),
            "status": "completed",
            "path": str(target),
            "summary": f"wrote {relative_path}",
            "content_type": "text/markdown" if target.suffix.lower() in {".md", ".markdown"} else "text/plain",
        }
        return {
            "node_id": str(node.get("node_id", "")),
            "status": "completed",
            "artifact": artifact,
            "result": {
                "path": str(target),
                "relative_path": relative_path,
                "area": area,
                "bytes_written": target.stat().st_size,
                "output": content,
            },
        }

    def _execute_workspace_action(
        self,
        *,
        sandbox: ThreadSandbox,
        execution_id: str,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        action_kind = str(metrics.get("action_kind", "")).strip()
        prompt = str(metrics.get("prompt", context.get("query", ""))).strip()
        workspace_summary = dict(metrics.get("workspace_summary", {})) if isinstance(metrics.get("workspace_summary", {}), dict) else {}
        source_text = self._collect_source_text(metrics=metrics, context=context)
        descriptor = self._resolve_workspace_action_descriptor(action_kind=action_kind, metrics=metrics)
        if descriptor is None or str(descriptor.get("kind", "")).strip() == "validation_execution":
            raise ValueError(f"unsupported workspace_action: {action_kind}")
        render_context = dict(context)
        render_context["_graph"] = graph
        render_context["_action_metrics"] = metrics
        local = self._render_workspace_action_local(
            action_kind=action_kind,
            prompt=prompt,
            source_text=source_text,
            context=render_context,
            workspace_summary=workspace_summary,
            descriptor=descriptor,
        )
        relative_path = str(metrics.get("relative_path", descriptor.get("default_relative_path", ""))).strip() or str(
            descriptor.get("default_relative_path", f"artifacts/{action_kind}.txt")
        )
        content_type = str(metrics.get("content_type", descriptor.get("content_type", "text/markdown"))).strip() or "text/markdown"
        result_field = str(descriptor.get("result_field", workspace_action_result_field(action_kind, content_type=content_type))).strip() or "output"
        body = {
            "node_id": node.get("node_id", ""),
            "action_kind": action_kind,
            "path": relative_path,
            "graph_id": graph.get("graph_id", ""),
            "artifact_contract": dict(descriptor.get("artifact_contract", {})) if isinstance(descriptor.get("artifact_contract", {}), dict) else {},
        }
        if content_type == "application/json":
            payload = dict(local.get("payload", {})) if isinstance(local.get("payload", {}), dict) else {}
            content = json.dumps(payload, indent=2, default=str)
            body[result_field] = payload
        else:
            content = str(local.get("content", ""))
            body[result_field] = content

        live_generation = self._live_workspace_action_generation(
            action_kind=action_kind,
            prompt=prompt,
            source_text=source_text,
            workspace_summary=workspace_summary,
            local_relative_path=relative_path,
            local_body=body,
            local_content=content,
            content_type=content_type,
            context=context,
        )
        generation_source = "local"
        if live_generation:
            generation_source = str(live_generation.get("source", "live_model"))
            relative_path = str(live_generation.get("relative_path", relative_path)).strip() or relative_path
            content_type = str(live_generation.get("content_type", content_type)).strip() or content_type
            if content_type == "application/json":
                payload = dict(live_generation.get("content_json", {})) if isinstance(live_generation.get("content_json", {}), dict) else {}
                if payload:
                    content = json.dumps(payload, indent=2, default=str)
                    field = self._workspace_action_result_field(action_kind)
                    body[field] = payload
            else:
                text = str(live_generation.get("content_text", "")).strip()
                if text:
                    content = text
                    field = self._workspace_action_result_field(action_kind)
                    body[field] = text
        body["path"] = relative_path
        body["generation_source"] = generation_source
        if live_generation and isinstance(live_generation.get("rationale", []), list):
            body["generation_rationale"] = [str(item) for item in live_generation.get("rationale", []) if str(item)]

        target = sandbox.write_text(relative_path, content, area="workspace")
        artifact = {
            "kind": "workspace_action_artifact",
            "label": str(node.get("title", action_kind)),
            "status": "completed",
            "path": str(target),
            "summary": f"workspace action {action_kind} generated via {generation_source}",
            "content_type": content_type,
        }
        return {
            "node_id": str(node.get("node_id", "")),
            "status": "completed",
            "artifact": artifact,
            "result": body,
        }

    def _resolve_workspace_action_descriptor(
        self,
        *,
        action_kind: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        spec = default_workspace_action_specs().get(action_kind)
        if spec is not None:
            return spec.to_dict()
        if not action_kind.startswith("custom:"):
            return None
        artifact_contract = dict(metrics.get("artifact_contract", {})) if isinstance(metrics.get("artifact_contract", {}), dict) else {}
        format_hint = str(metrics.get("format_hint", artifact_contract.get("format_hint", ""))).strip()
        content_type = str(metrics.get("content_type", "")).strip()
        if not content_type:
            if format_hint == "json":
                content_type = "application/json"
            else:
                content_type = "text/markdown" if format_hint in {"", "markdown"} else "text/plain"
        return {
            "kind": action_kind,
            "title": str(metrics.get("title", artifact_contract.get("title", action_kind.replace("custom:", "").replace("_", " ").title()))).strip(),
            "default_relative_path": str(metrics.get("relative_path", artifact_contract.get("relative_path", f"artifacts/{action_kind.replace(':', '-')}.md"))).strip(),
            "content_type": content_type,
            "result_field": workspace_action_result_field(action_kind, content_type=content_type),
            "format_hint": format_hint or ("json" if content_type == "application/json" else "markdown"),
            "artifact_contract": artifact_contract,
        }

    def _render_workspace_action_local(
        self,
        *,
        action_kind: str,
        prompt: str,
        source_text: str,
        context: dict[str, Any],
        workspace_summary: dict[str, Any],
        descriptor: dict[str, Any],
    ) -> dict[str, Any]:
        if action_kind.startswith("custom:"):
            payload = self._build_custom_document_artifact(
                prompt=prompt,
                source_text=source_text,
                workspace_summary=workspace_summary,
                descriptor=descriptor,
            )
        else:
            builder = self._workspace_action_builders().get(action_kind)
            if builder:
                payload = builder(
                    prompt=prompt,
                    source_text=source_text,
                    context=context,
                    workspace_summary=workspace_summary,
                )
            else:
                payload = self._build_generic_workspace_action(
                    prompt=prompt,
                    source_text=source_text,
                    context=context,
                    workspace_summary=workspace_summary,
                    descriptor=descriptor,
                )
        if str(descriptor.get("content_type", "")).strip() == "application/json":
            return {"payload": dict(payload) if isinstance(payload, dict) else {"content": str(payload)}}
        return {"content": str(payload)}

    def _workspace_action_builders(self) -> dict[str, Any]:
        return {
            "patch_scaffold": lambda **kwargs: self._build_patch_scaffold(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
                context=kwargs["context"],
                workspace_summary=kwargs["workspace_summary"],
            ),
            "patch_draft": lambda **kwargs: self._build_patch_draft(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
                workspace_summary=kwargs["workspace_summary"],
            ),
            "completion_packet": lambda **kwargs: self._build_completion_packet(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
                context=kwargs["context"],
                workspace_summary=kwargs["workspace_summary"],
            ),
            "delivery_bundle": lambda **kwargs: self._build_delivery_bundle(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
                context=kwargs["context"],
                workspace_summary=kwargs["workspace_summary"],
            ),
            "benchmark_run_config": lambda **kwargs: self._build_benchmark_run_config(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
                workspace_summary=kwargs["workspace_summary"],
            ),
            "benchmark_manifest": lambda **kwargs: self._build_benchmark_manifest(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
                workspace_summary=kwargs["workspace_summary"],
            ),
            "dataset_pull_spec": lambda **kwargs: self._build_dataset_pull_spec(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
                context=kwargs["context"],
            ),
            "dataset_loader_template": lambda **kwargs: self._build_dataset_loader_template(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
                context=kwargs["context"],
            ),
            "webpage_blueprint": lambda **kwargs: self._build_webpage_blueprint(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
            ),
            "slide_deck_plan": lambda **kwargs: self._build_slide_deck_plan(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
            ),
            "chart_pack_spec": lambda **kwargs: self._build_chart_pack_spec(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
            ),
            "podcast_episode_plan": lambda **kwargs: self._build_podcast_episode_plan(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
            ),
            "video_storyboard": lambda **kwargs: self._build_video_storyboard(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
            ),
            "image_prompt_pack": lambda **kwargs: self._build_image_prompt_pack(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
            ),
            "data_analysis_spec": lambda **kwargs: self._build_data_analysis_spec(
                prompt=kwargs["prompt"],
                source_text=kwargs["source_text"],
            ),
        }

    @staticmethod
    def _build_generic_workspace_action(
        *,
        prompt: str,
        source_text: str,
        context: dict[str, Any],
        workspace_summary: dict[str, Any],
        descriptor: dict[str, Any],
    ) -> dict[str, Any] | str:
        del context
        content_type = str(descriptor.get("content_type", "")).strip()
        title = str(descriptor.get("title", descriptor.get("kind", "Artifact"))).strip()
        format_hint = str(descriptor.get("format_hint", "")).strip()
        contract = dict(descriptor.get("artifact_contract", {})) if isinstance(descriptor.get("artifact_contract", {}), dict) else {}
        note = TaskGraphActionMapper._single_line(source_text)[:240]
        if content_type == "application/json":
            return {
                "title": title,
                "kind": str(descriptor.get("kind", "")),
                "objective": prompt,
                "format_hint": format_hint or "json",
                "workspace_languages": list(workspace_summary.get("languages", [])),
                "workspace_frameworks": list(workspace_summary.get("frameworks", [])),
                "contract": contract,
                "notes": note,
            }
        return (
            f"# {title}\n\n"
            f"- Kind: {descriptor.get('kind', '')}\n"
            f"- Objective: {prompt}\n"
            f"- Format: {format_hint or 'markdown'}\n"
            f"- Workspace languages: {', '.join(workspace_summary.get('languages', [])) or 'unknown'}\n\n"
            "## Grounding\n\n"
            f"{note}\n\n"
            "## Contract\n\n"
            f"{json.dumps(contract, indent=2, ensure_ascii=False) if contract else 'No explicit contract provided.'}\n"
        )

    @classmethod
    def _build_custom_document_artifact(
        cls,
        *,
        prompt: str,
        source_text: str,
        workspace_summary: dict[str, Any],
        descriptor: dict[str, Any],
    ) -> str:
        contract = dict(descriptor.get("artifact_contract", {})) if isinstance(descriptor.get("artifact_contract", {}), dict) else {}
        kind = str(descriptor.get("kind", contract.get("kind", "custom:document"))).strip()
        title = str(descriptor.get("title", contract.get("title", "Document"))).strip()
        sections = [
            str(item).strip()
            for item in contract.get("sections", cls._default_custom_sections(kind=kind, title=title))
            if str(item).strip()
        ]
        grounding = cls._grounding_snapshot(source_text)
        summary = grounding["summary"][0] if grounding["summary"] else f"This document addresses {prompt}."
        lines = [f"# {title}", "", f"Objective: {prompt}", ""]

        if kind == "custom:checklist":
            lines.extend(["## Checklist", ""])
            for item in cls._section_points(section="Checklist", prompt=prompt, grounding=grounding):
                lines.append(f"- [ ] {item}")
            lines.append("")
        elif kind == "custom:faq":
            lines.extend(["## FAQ", ""])
            faq_rows = cls._section_points(section="FAQ", prompt=prompt, grounding=grounding)
            for index, item in enumerate(faq_rows, start=1):
                lines.append(f"### Q{index}")
                lines.append(f"What should stakeholders know about {item.rstrip('.')}?")
                lines.append("")
                lines.append(f"A{index}. {item}")
                lines.append("")
        else:
            lines.extend(["## Summary", "", summary, ""])
            for section in sections:
                lines.extend([f"## {section}", ""])
                for item in cls._section_points(section=section, prompt=prompt, grounding=grounding):
                    lines.append(f"- {item}")
                lines.append("")

        if grounding["evidence"]:
            lines.extend(["## Grounding Signals", ""])
            for item in grounding["evidence"][:5]:
                lines.append(f"- {item}")
            lines.append("")
        if workspace_summary.get("languages") or workspace_summary.get("frameworks"):
            lines.extend(["## Workspace Context", ""])
            if workspace_summary.get("languages"):
                lines.append(f"- Languages: {', '.join(str(item) for item in workspace_summary.get('languages', []))}")
            if workspace_summary.get("frameworks"):
                lines.append(f"- Frameworks: {', '.join(str(item) for item in workspace_summary.get('frameworks', []))}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _default_custom_sections(*, kind: str, title: str) -> list[str]:
        mapping = {
            "custom:decision_memo": ["Decision", "Evidence", "Tradeoffs", "Next Step"],
            "custom:executive_memo": ["Context", "Recommendation", "Risks", "Next Step"],
            "custom:launch_memo": ["Launch Goal", "Audience and Promise", "Execution Plan", "Risks and Controls", "Immediate Ask"],
            "custom:memo": ["Context", "Recommendation", "Implications", "Next Step"],
            "custom:brief": ["Question", "Key Findings", "Implications", "Open Questions"],
            "custom:one_pager": ["Headline", "Why It Matters", "Proof", "Next Step"],
            "custom:checklist": ["Checklist"],
            "custom:faq": ["FAQ"],
        }
        return mapping.get(kind, ["Summary", "Evidence", "Next Step"]) or [title]

    @classmethod
    def _section_points(cls, *, section: str, prompt: str, grounding: dict[str, list[str]]) -> list[str]:
        name = section.lower()
        summary = grounding["summary"]
        evidence = grounding["evidence"]
        risks = grounding["risks"]
        actions = grounding["actions"]
        prompt_focus = cls._single_line(prompt)
        if "decision" in name or "recommendation" in name:
            return [
                summary[0] if summary else f"Proceed with a bounded first release for {prompt_focus}.",
                actions[0] if actions else "Keep the first iteration narrow enough to validate quickly.",
            ]
        if "context" in name or "question" in name or "headline" in name or "launch goal" in name:
            return [
                f"Primary objective: {prompt_focus}.",
                summary[0] if summary else "Use the first artifact to reduce ambiguity before expanding scope.",
            ]
        if "evidence" in name or "proof" in name or "findings" in name:
            return evidence[:3] or summary[:2] or ["Ground the recommendation in inspectable artifacts and citations."]
        if "risk" in name or "tradeoff" in name:
            return risks[:3] or [
                "Main risk is acting on unvalidated assumptions.",
                "Mitigate by keeping a clear owner, evidence trail, and rollback path.",
            ]
        if "next" in name or "ask" in name or "execution" in name or "plan" in name:
            return actions[:3] or [
                "Assign one owner to the first deliverable and the validation step.",
                "Validate the strongest claim with a concrete artifact before rollout.",
            ]
        if "implication" in name or "why it matters" in name:
            return summary[:2] or ["The output should unlock a concrete decision instead of another abstract plan."]
        if "open question" in name:
            return [
                "Which assumption still lacks direct evidence?",
                "What would block a reviewer from approving the next step?",
            ]
        if "faq" in name or "checklist" in name:
            return actions[:2] + risks[:2] or ["Confirm scope, evidence, and approval criteria."]
        return summary[:2] or evidence[:2] or [f"Address {prompt_focus} with a reviewable artifact."]

    @classmethod
    def _grounding_snapshot(cls, source_text: str) -> dict[str, list[str]]:
        summary: list[str] = []
        evidence: list[str] = []
        risks: list[str] = []
        actions: list[str] = []

        for payload in cls._json_objects_from_text(source_text):
            output = payload.get("output")
            if isinstance(output, str):
                summary.extend(cls._meaningful_lines(output, limit=4))
            if isinstance(output, dict):
                record_count = int(output.get("record_count", output.get("count", 0)) or 0)
                if record_count > 0:
                    evidence.append(f"Collected {record_count} evidence records for the task.")
                for record in output.get("records", [])[:3] if isinstance(output.get("records", []), list) else []:
                    if isinstance(record, dict) and str(record.get("title", "")).strip():
                        evidence.append(str(record.get("title", "")).strip())
                risk_matrix = output.get("risk_matrix", [])
                for row in risk_matrix[:3] if isinstance(risk_matrix, list) else []:
                    if not isinstance(row, dict):
                        continue
                    dimension = str(row.get("dimension", "risk")).strip() or "risk"
                    level = str(row.get("level", "unknown")).strip() or "unknown"
                    controls = ", ".join(str(item) for item in row.get("controls", [])[:2]) if isinstance(row.get("controls", []), list) else ""
                    risks.append(f"{dimension.title()} risk is {level}" + (f"; controls: {controls}." if controls else "."))
                results = output.get("results", [])
                for item in results[:3] if isinstance(results, list) else []:
                    if not isinstance(item, dict):
                        continue
                    command = cls._single_line(item.get("command", "")) or "command"
                    exit_code = item.get("exit_code", 0)
                    actions.append(f"Validation command `{command}` exited with code {exit_code}.")
            tool_name = str(payload.get("tool_name", "")).strip()
            if tool_name == "policy_risk_matrix" and not risks:
                risks.append("Use the risk matrix to make controls explicit before rollout.")
            if tool_name in {"external_resource_hub", "evidence_dossier_builder"} and not evidence:
                evidence.append(f"Evidence was collected through {tool_name}.")

        summary.extend(cls._meaningful_lines(source_text, limit=4))
        deduped_summary = cls._dedupe_lines(summary)[:5]
        deduped_evidence = cls._dedupe_lines(evidence)[:5]
        deduped_risks = cls._dedupe_lines(risks)[:5]
        deduped_actions = cls._dedupe_lines(actions)[:5]
        if not deduped_actions:
            deduped_actions = [
                "Package the strongest evidence into the first deliverable.",
                "Keep the next validation step inspectable and bounded.",
            ]
        return {
            "summary": deduped_summary,
            "evidence": deduped_evidence,
            "risks": deduped_risks,
            "actions": deduped_actions,
        }

    @staticmethod
    def _dedupe_lines(lines: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in lines:
            value = TaskGraphActionMapper._single_line(item).strip(" -")
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def _meaningful_lines(text: str, *, limit: int = 5) -> list[str]:
        rows: list[str] = []
        for line in re.split(r"\r?\n+", str(text or "")):
            value = TaskGraphActionMapper._single_line(line).strip(" -")
            if not value:
                continue
            lowered = value.lower()
            if lowered in {"{", "}", "[", "]"}:
                continue
            if lowered.startswith("--- (skill:"):
                continue
            if value.endswith(":") and len(value.split()) <= 4:
                continue
            if re.fullmatch(r"[\{\}\[\],:\"]+", value):
                continue
            if value not in rows:
                rows.append(value)
            if len(rows) >= limit:
                break
        return rows

    @staticmethod
    def _json_objects_from_text(text: str, *, limit: int = 6) -> list[dict[str, Any]]:
        payload = str(text or "")
        decoder = json.JSONDecoder()
        objects: list[dict[str, Any]] = []
        index = 0
        while index < len(payload) and len(objects) < limit:
            start = payload.find("{", index)
            if start < 0:
                break
            try:
                value, offset = decoder.raw_decode(payload[start:])
            except Exception:
                index = start + 1
                continue
            if isinstance(value, dict):
                objects.append(value)
            index = start + max(offset, 1)
        return objects

    def _execute_subagent_node(
        self,
        *,
        sandbox: ThreadSandbox,
        execution_id: str,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        subagent_kind = str(metrics.get("subagent_kind", "general_probe")).strip()
        objective = str(metrics.get("objective", metrics.get("prompt", context.get("query", "")))).strip()
        source_text = self._collect_source_text(metrics=metrics, context=context)

        plan = self._resolve_subagent_plan(
            subagent_kind=subagent_kind,
            objective=objective,
            source_text=source_text,
            sandbox=sandbox,
            context=context,
        )
        trace: list[dict[str, Any]] = []
        collected: list[str] = [objective, source_text]
        for call in self._tool_calls_from_plan(plan, sandbox=sandbox):
            result = self.tools.call(call)
            trace.append(
                {
                    "tool": call.name,
                    "success": bool(result.success),
                    "output": result.output,
                    "error": result.error,
                }
            )
            if result.output:
                collected.append(json.dumps(result.output, indent=2, default=str))

        skill_name = str(plan.get("skill_name", "artifact_synthesis")).strip() or "artifact_synthesis"
        output = execute_skill(skill_name, "\n\n".join(item for item in collected if item))
        body = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "node_type": node.get("node_type", ""),
            "subagent_kind": subagent_kind,
            "objective": objective,
            "plan": plan,
            "trace": trace,
            "output": output,
            "graph_id": graph.get("graph_id", ""),
        }
        return self._write_json_result(
            sandbox=sandbox,
            execution_id=execution_id,
            node=node,
            body=body,
            summary=f"subagent {subagent_kind} completed",
            kind="subagent_result",
        )

    def _execute_command_node(
        self,
        *,
        sandbox: ThreadSandbox,
        execution_id: str,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        area = str(metrics.get("area", "workspace"))
        timeout_seconds = max(1, int(metrics.get("timeout_seconds", 30)))
        commands = list(node.get("commands", [])) if isinstance(node.get("commands", []), list) else []
        if not commands and metrics.get("command"):
            commands = [str(metrics.get("command", ""))]
        results = [sandbox.execute_command(command, area=area, timeout_seconds=timeout_seconds).to_dict() for command in commands]
        body = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "node_type": node.get("node_type", ""),
            "area": area,
            "results": results,
            "graph_id": graph.get("graph_id", ""),
            "context_keys": sorted(context.keys()),
        }
        return self._write_json_result(
            sandbox=sandbox,
            execution_id=execution_id,
            node=node,
            body=body,
            summary=f"executed {len(results)} command(s)",
            kind="command_result",
        )

    def _resolve_subagent_plan(
        self,
        *,
        subagent_kind: str,
        objective: str,
        source_text: str,
        sandbox: ThreadSandbox,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        local = self._default_subagent_plan(
            subagent_kind=subagent_kind,
            objective=objective,
            sandbox=sandbox,
        )
        live = self._live_subagent_plan(
            subagent_kind=subagent_kind,
            objective=objective,
            source_text=source_text,
            local_plan=local,
            context=context,
        )
        return live or local

    def _default_subagent_plan(
        self,
        *,
        subagent_kind: str,
        objective: str,
        sandbox: ThreadSandbox,
    ) -> dict[str, Any]:
        workspace_root = sandbox.workspace_paths().get("workspace", "")
        if subagent_kind == "repair_probe":
            return {
                "source": "local",
                "skill_name": "artifact_synthesis",
                "tool_calls": [
                    {"name": "tool_search", "args": {"query": objective, "limit": 5}},
                    {
                        "name": "workspace_file_search",
                        "args": {
                            "query": self._compact_query(objective),
                            "glob": "*",
                            "limit": 6,
                            "workspace_root": workspace_root,
                        },
                    },
                ],
                "rationale": ["repair probe defaults to capability discovery plus workspace scan"],
            }
        if subagent_kind == "research_probe":
            return {
                "source": "local",
                "skill_name": "research_brief",
                "tool_calls": [
                    {"name": "external_resource_hub", "args": {"query": objective, "limit": 5}},
                    {"name": "evidence_dossier_builder", "args": {"query": objective, "limit": 4}},
                ],
                "rationale": ["research probe defaults to external resources and evidence collection"],
            }
        if subagent_kind == "benchmark_probe":
            return {
                "source": "local",
                "skill_name": "benchmark_ablation",
                "tool_calls": [
                    {"name": "code_experiment_design", "args": {"query": objective, "max_experiments": 4}},
                    {"name": "tool_search", "args": {"query": "benchmark evaluation runner", "limit": 5}},
                ],
                "rationale": ["benchmark probe defaults to experiment design and runner discovery"],
            }
        return {
            "source": "local",
            "skill_name": "artifact_synthesis",
            "tool_calls": [{"name": "tool_search", "args": {"query": objective, "limit": 5}}],
            "rationale": ["general probe falls back to tool discovery"],
        }

    def _live_workspace_action_generation(
        self,
        *,
        action_kind: str,
        prompt: str,
        source_text: str,
        workspace_summary: dict[str, Any],
        local_relative_path: str,
        local_body: dict[str, Any],
        local_content: str,
        content_type: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        overrides = self._resolve_live_model_overrides(context)
        if not overrides:
            return None
        try:
            from app.harness.live_agent import CallBudget, LiveModelConfig, LiveModelGateway

            config = LiveModelConfig.resolve(overrides)
            if not config:
                return None
            gateway = LiveModelGateway(config)
            mode = "json" if content_type == "application/json" else "text"
            payload = {
                "action_kind": action_kind,
                "prompt": prompt,
                "source_text": source_text[:4000],
                "workspace_summary": workspace_summary,
                "local_relative_path": local_relative_path,
                "local_result": local_body,
                "local_content_preview": local_content[:3000],
                "expected_mode": mode,
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are generating the concrete artifact content for one workspace_action inside a general agent runtime. "
                        "Return strict JSON with keys relative_path, content_text, content_json, rationale. "
                        "Only set content_json for JSON artifacts. "
                        "Only set content_text for text, diff, markdown, or python artifacts. "
                        "Keep the output executable and grounded in the provided task, source text, and workspace summary. "
                        "Do not change artifact type."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ]
            text, meta = gateway.chat(
                messages=messages,
                budget=CallBudget(max_calls=1),
                temperature=0.1,
                require_json=True,
            )
            parsed = self._parse_json_dict(text)
            relative_path = str(parsed.get("relative_path", local_relative_path)).strip() or local_relative_path
            rationale = [str(item).strip() for item in parsed.get("rationale", []) if str(item).strip()] if isinstance(parsed.get("rationale", []), list) else []
            if mode == "json":
                content_json = parsed.get("content_json", {})
                if not isinstance(content_json, dict) or not content_json:
                    return None
                return {
                    "source": "live_model",
                    "model": str(meta.get("model", "")),
                    "relative_path": relative_path,
                    "content_json": content_json,
                    "content_type": "application/json",
                    "rationale": rationale,
                }
            content_text = str(parsed.get("content_text", "")).strip()
            if not content_text:
                return None
            return {
                "source": "live_model",
                "model": str(meta.get("model", "")),
                "relative_path": relative_path,
                "content_text": content_text,
                "content_type": content_type,
                "rationale": rationale,
            }
        except Exception:
            return None

    def _live_subagent_plan(
        self,
        *,
        subagent_kind: str,
        objective: str,
        source_text: str,
        local_plan: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        overrides = self._resolve_live_model_overrides(context)
        if not overrides:
            return None
        try:
            from app.harness.live_agent import CallBudget, LiveModelConfig, LiveModelGateway

            config = LiveModelConfig.resolve(overrides)
            if not config:
                return None
            gateway = LiveModelGateway(config)
            payload = {
                "subagent_kind": subagent_kind,
                "objective": objective,
                "source_text": source_text[:2500],
                "local_plan": local_plan,
                "allowed_tools": sorted(self._allowed_subagent_tools()),
                "allowed_skills": sorted(self._allowed_subagent_skills()),
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are generating a mini-plan for a subagent inside a general agent runtime. "
                        "Return strict JSON with keys skill_name, tool_calls, rationale. "
                        "tool_calls must be an array of {name, args}. "
                        "Only use allowed tool names and allowed skill names."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ]
            text, _meta = gateway.chat(
                messages=messages,
                budget=CallBudget(max_calls=1),
                temperature=0.0,
                require_json=True,
            )
            parsed = self._parse_json_dict(text)
            skill_name = str(parsed.get("skill_name", "")).strip()
            if skill_name not in self._allowed_subagent_skills():
                return None
            tool_calls: list[dict[str, Any]] = []
            for item in parsed.get("tool_calls", []) if isinstance(parsed.get("tool_calls", []), list) else []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                args = dict(item.get("args", {})) if isinstance(item.get("args", {}), dict) else {}
                if name not in self._allowed_subagent_tools():
                    continue
                tool_calls.append({"name": name, "args": args})
            if not tool_calls:
                return None
            rationale = list(local_plan.get("rationale", []))
            rationale.append("live model refined subagent mini-plan")
            for item in parsed.get("rationale", []) if isinstance(parsed.get("rationale", []), list) else []:
                text_item = str(item).strip()
                if text_item:
                    rationale.append(text_item)
            return {
                "source": "live_model",
                "skill_name": skill_name,
                "tool_calls": tool_calls,
                "rationale": rationale[:8],
            }
        except Exception:
            return None

    def _tool_calls_from_plan(self, plan: dict[str, Any], *, sandbox: ThreadSandbox) -> list[Any]:
        from app.harness.models import ToolCall

        tool_calls: list[Any] = []
        workspace_root = sandbox.workspace_paths().get("workspace", "")
        for item in plan.get("tool_calls", []) if isinstance(plan.get("tool_calls", []), list) else []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name not in self._allowed_subagent_tools():
                continue
            args = dict(item.get("args", {})) if isinstance(item.get("args", {}), dict) else {}
            if name.startswith("workspace_"):
                args.setdefault("workspace_root", workspace_root)
            tool_calls.append(ToolCall(name=name, tool_type=self.tools.infer_tool_type(name), args=args))
        return tool_calls

    def _execute_graph_replan(
        self,
        *,
        sandbox: ThreadSandbox,
        execution_id: str,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        task_spec = self._coerce_task_spec(
            metrics=metrics,
            graph=graph,
            context=context,
        )
        world_state = build_world_state(graph=graph, context=context)
        state_gap = compute_state_gap(task_spec=task_spec, world_state=world_state)
        completion_packet_preview = self._compile_completion_packet_payload(
            prompt=str(metrics.get("prompt", context.get("query", graph.get("query", "")))),
            source_text=self._collect_source_text(metrics=metrics, context=context),
            context=context,
            workspace_summary=dict(metrics.get("workspace_summary", {})) if isinstance(metrics.get("workspace_summary", {}), dict) else {},
            task_spec=task_spec,
            world_state=world_state,
            state_gap=state_gap,
        )
        failure_policy = self._classify_failure_policy(context=context, completion_packet=completion_packet_preview)
        capability_replan = plan_capability_path(
            task_spec=task_spec,
            registry=default_capability_registry(),
            world_state=world_state,
        )
        additions = self._recompile_nodes_from_state_gap(
            graph=graph,
            metrics=metrics,
            state_gap=state_gap.to_dict(),
            capability_replan=capability_replan,
        )
        if not additions:
            additions = self._propose_replan_nodes(
            node=node,
            graph=graph,
            context=context,
            metrics=metrics,
            failure_policy=failure_policy,
            completion_packet=completion_packet_preview,
            )
        added_node_ids: list[str] = []
        if additions:
            synthesis = next((item for item in graph.get("nodes", []) if str(item.get("node_id", "")) == "synthesis"), None)
            completion_packet = next((item for item in graph.get("nodes", []) if str(item.get("node_id", "")) == "completion_packet"), None)
            for addition in additions:
                graph.setdefault("nodes", []).append(addition)
                added_node_ids.append(str(addition.get("node_id", "")))
                if isinstance(synthesis, dict):
                    deps = list(synthesis.get("depends_on", [])) if isinstance(synthesis.get("depends_on", []), list) else []
                    if addition["node_id"] not in deps:
                        deps.append(addition["node_id"])
                    synthesis["depends_on"] = deps
                if isinstance(completion_packet, dict):
                    deps = list(completion_packet.get("depends_on", [])) if isinstance(completion_packet.get("depends_on", []), list) else []
                    if addition["node_id"] not in deps:
                        deps.append(addition["node_id"])
                    completion_packet["depends_on"] = deps
                    completion_packet["status"] = "ready"
                    completion_packet["artifacts"] = []
                    context.get("node_results", {}).pop("completion_packet", None)
        body = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "node_type": node.get("node_type", ""),
            "failure_policy": failure_policy,
            "task_spec": task_spec.to_dict(),
            "world_state": world_state.to_dict(),
            "state_gap": state_gap.to_dict(),
            "completion_packet_preview": completion_packet_preview,
            "capability_replan": capability_replan,
            "added_node_ids": added_node_ids,
            "added_nodes": additions,
            "graph_id": graph.get("graph_id", ""),
        }
        return self._write_json_result(
            sandbox=sandbox,
            execution_id=execution_id,
            node=node,
            body=body,
            summary=f"replanned graph with {len(added_node_ids)} new node(s)",
            kind="replan_result",
        )

    @staticmethod
    def _coerce_task_spec(*, metrics: dict[str, Any], graph: dict[str, Any], context: dict[str, Any]):
        payload = metrics.get("task_spec", {})
        if isinstance(payload, dict) and payload.get("artifact_contracts"):
            artifact_contracts = []
            for item in payload.get("artifact_contracts", []):
                if not isinstance(item, dict):
                    continue
                from app.core.tasking import ArtifactContract, TaskSpec

                artifact_contracts.append(
                    ArtifactContract(
                        kind=str(item.get("kind", "")),
                        title=str(item.get("title", "")),
                        format_hint=str(item.get("format_hint", "")),
                        required=bool(item.get("required", True)),
                    )
                )
            from app.core.tasking import TaskSpec

            return TaskSpec(
                query=str(payload.get("query", metrics.get("prompt", graph.get("query", "")))),
                goal=str(payload.get("goal", metrics.get("prompt", graph.get("query", "")))),
                target=str(payload.get("target", "general")),
                domains=[str(item) for item in payload.get("domains", []) if str(item).strip()],
                constraints=[str(item) for item in payload.get("constraints", []) if str(item).strip()],
                success_criteria=[str(item) for item in payload.get("success_criteria", []) if str(item).strip()],
                required_channels=[str(item) for item in payload.get("required_channels", []) if str(item).strip()],
                artifact_contracts=artifact_contracts,
                risk_policy=str(payload.get("risk_policy", "balanced")),
                needs_validation=bool(payload.get("needs_validation", False)),
                needs_command_execution=bool(payload.get("needs_command_execution", False)),
            )
        return infer_task_spec(
            query=str(metrics.get("prompt", graph.get("query", context.get("query", "")))),
            output_mode=str(metrics.get("output_mode", "artifact")),
            needs_validation=True,
        )

    def _recompile_nodes_from_state_gap(
        self,
        *,
        graph: dict[str, Any],
        metrics: dict[str, Any],
        state_gap: dict[str, Any],
        capability_replan: dict[str, Any],
    ) -> list[dict[str, Any]]:
        existing_ids = {
            str(item.get("node_id", ""))
            for item in graph.get("nodes", [])
            if isinstance(item, dict)
        }
        existing_workspace_actions = {
            str(item.get("metrics", {}).get("action_kind", "")).strip()
            for item in graph.get("nodes", [])
            if isinstance(item, dict)
            and str(item.get("node_type", "")).strip() == "workspace_action"
            and isinstance(item.get("metrics", {}), dict)
        }
        existing_tool_names = {
            str(item.get("metrics", {}).get("tool_name", "")).strip()
            for item in graph.get("nodes", [])
            if isinstance(item, dict)
            and str(item.get("node_type", "")).strip() == "tool_call"
            and isinstance(item.get("metrics", {}), dict)
        }
        existing_subagent_kinds = {
            str(item.get("metrics", {}).get("subagent_kind", "")).strip()
            for item in graph.get("nodes", [])
            if isinstance(item, dict)
            and str(item.get("node_type", "")).strip() == "subagent"
            and isinstance(item.get("metrics", {}), dict)
        }
        additions: list[dict[str, Any]] = []
        prompt = str(metrics.get("prompt", graph.get("query", "")))
        workspace_summary = dict(metrics.get("workspace_summary", {})) if isinstance(metrics.get("workspace_summary", {}), dict) else {}
        for step in capability_replan.get("steps", []) if isinstance(capability_replan.get("steps", []), list) else []:
            if not isinstance(step, dict):
                continue
            node_type = str(step.get("node_type", "")).strip()
            if node_type not in {"tool_call", "workspace_action", "subagent"}:
                continue
            if node_type == "tool_call":
                spec = {
                    "node_type": node_type,
                    "tool_name": str(step.get("ref", "")).strip(),
                    "tool_args": dict(step.get("default_args", {})) if isinstance(step.get("default_args", {}), dict) else {},
                    "title": str(step.get("title", "")),
                    "reason": str(step.get("reason", "")),
                }
                key = spec["tool_name"]
                if key in existing_tool_names:
                    continue
            elif node_type == "workspace_action":
                spec = {
                    "node_type": node_type,
                    "kind": str(step.get("ref", "")).strip(),
                    "title": str(step.get("title", "")),
                    "reason": str(step.get("reason", "")),
                    "source_node_ids": ["analysis", "execution"],
                }
                key = spec["kind"]
                if key in existing_workspace_actions:
                    continue
            else:
                spec = {
                    "node_type": node_type,
                    "subagent_kind": str(step.get("ref", "")).strip() or "repair_probe",
                    "objective": str(step.get("reason", prompt)),
                    "title": str(step.get("title", "")),
                    "reason": str(step.get("reason", "")),
                    "source_node_ids": ["analysis", "execution"],
                }
                key = spec["subagent_kind"]
                if key in existing_subagent_kinds:
                    continue
            node_id = self._replan_node_id(node_type=node_type, key=key)
            if not key or node_id in existing_ids:
                continue
            additions.append(
                self._build_replanned_node(
                    node_id=node_id,
                    node_type=node_type,
                    spec=spec,
                    prompt=prompt,
                    workspace_summary=workspace_summary,
                )
            )
            existing_ids.add(node_id)
            if node_type == "workspace_action":
                existing_workspace_actions.add(key)
            elif node_type == "tool_call":
                existing_tool_names.add(key)
            else:
                existing_subagent_kinds.add(key)
        if additions or not state_gap:
            return additions
        return []

    def _execute_static_node(
        self,
        *,
        sandbox: ThreadSandbox,
        execution_id: str,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        body = self._render_static_payload(node=node, graph=graph, context=context)
        return self._write_json_result(
            sandbox=sandbox,
            execution_id=execution_id,
            node=node,
            body=body,
            summary=str(body.get("summary", "")),
            kind=f"{node.get('node_type', 'artifact')}_artifact",
        )

    def _write_json_result(
        self,
        *,
        sandbox: ThreadSandbox,
        execution_id: str,
        node: dict[str, Any],
        body: dict[str, Any],
        summary: str,
        kind: str,
    ) -> dict[str, Any]:
        node_id = str(node.get("node_id", "node"))
        relative_path = f"executions/{execution_id}/{node_id}.json"
        target = sandbox.write_text(relative_path, json.dumps(body, indent=2, default=str), area="outputs")
        artifact = {
            "kind": kind,
            "label": str(node.get("title", node_id)),
            "status": "completed",
            "path": str(target),
            "summary": summary,
            "content_type": "application/json",
        }
        return {
            "node_id": node_id,
            "status": "completed",
            "artifact": artifact,
            "result": body,
        }

    @staticmethod
    def _collect_source_text(*, metrics: dict[str, Any], context: dict[str, Any]) -> str:
        source_ids = metrics.get("source_node_ids", [])
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        parts: list[str] = []
        for source_id in source_ids if isinstance(source_ids, list) else []:
            value = TaskGraphActionMapper._extract_result_field(str(source_id), "", context)
            if value:
                parts.append(value)
        return "\n\n".join(parts)

    @staticmethod
    def _extract_result_field(source_node_id: str, field: str, context: dict[str, Any]) -> str:
        if not source_node_id:
            return ""
        node_results = context.get("node_results", {})
        source = node_results.get(source_node_id, {}) if isinstance(node_results, dict) else {}
        if not isinstance(source, dict) or not source:
            return ""
        result = source.get("result", {}) if isinstance(source, dict) else {}
        if result in ({}, None, ""):
            return ""
        if not isinstance(result, dict):
            return str(result)
        if field:
            value = result.get(field, "")
            if isinstance(value, str):
                return value
            return json.dumps(value, indent=2, default=str)
        return json.dumps(result, indent=2, default=str)

    @staticmethod
    def _render_static_payload(*, node: dict[str, Any], graph: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        node_type = str(node.get("node_type", "artifact"))
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        title = str(node.get("title", ""))
        notes = list(node.get("notes", [])) if isinstance(node.get("notes", []), list) else []
        commands = list(node.get("commands", [])) if isinstance(node.get("commands", []), list) else []
        summary = title
        payload: dict[str, Any] = {
            "node_id": node.get("node_id", ""),
            "title": title,
            "node_type": node_type,
            "notes": notes,
            "commands": commands,
            "metrics": metrics,
            "context_keys": sorted(context.keys()),
            "graph_id": graph.get("graph_id", ""),
        }
        if node_type in {"routing", "framing"}:
            summary = f"routing framed with {metrics or 'default metrics'}"
        elif node_type == "evidence":
            summary = f"evidence staged with {metrics.get('record_count', 0)} records"
        elif node_type in {"synthesis", "packaging"}:
            summary = f"packaged {title.lower()}"
        elif node_type in {"evaluation", "review"}:
            summary = f"evaluation or review completed for {title.lower()}"
        elif node_type in {"execution_plan", "validation_plan"}:
            summary = f"execution plan contains {len(commands)} commands"
        payload["summary"] = summary
        return payload

    @staticmethod
    def _build_patch_scaffold(
        *,
        prompt: str,
        source_text: str,
        context: dict[str, Any],
        workspace_summary: dict[str, Any],
    ) -> str:
        files = workspace_summary.get("sample_files", []) if isinstance(workspace_summary.get("sample_files", []), list) else []
        focus = ", ".join(str(item) for item in files[:5]) or "unknown targets"
        validation = TaskGraphActionMapper._extract_result_field("validation", "output", context)
        return (
            "# Patch Scaffold\n\n"
            f"- Task: {prompt}\n"
            f"- Candidate files: {focus}\n"
            "- Proposed edits:\n"
            "  1. isolate the failing path\n"
            "  2. implement the smallest safe patch\n"
            "  3. add or tighten regression coverage\n\n"
            "## Grounding\n\n"
            f"{source_text[:3000]}\n\n"
            "## Validation Hooks\n\n"
            f"{validation[:2000] if validation else 'Use targeted tests plus full regression run.'}\n"
        )

    @staticmethod
    def _build_benchmark_run_config(
        *,
        prompt: str,
        source_text: str,
        workspace_summary: dict[str, Any],
    ) -> dict[str, Any]:
        commands = workspace_summary.get("suggested_commands", []) if isinstance(workspace_summary.get("suggested_commands", []), list) else []
        return {
            "objective": prompt,
            "runner": "agent-harness-benchmark",
            "suite_candidates": ["GAIA", "SWE-bench", "WebArena", "tau-bench"],
            "workspace_languages": list(workspace_summary.get("languages", [])),
            "workspace_frameworks": list(workspace_summary.get("frameworks", [])),
            "validation_commands": commands[:3],
            "artifacts": ["scoreboard.json", "failure-clusters.json", "ablation-report.md"],
            "notes": source_text[:1500],
        }

    @staticmethod
    def _build_dataset_pull_spec(
        *,
        prompt: str,
        source_text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        evidence = TaskGraphActionMapper._extract_result_field("evidence", "", context)
        resources = TaskGraphActionMapper._extract_result_field("external_resources", "", context)
        return {
            "topic": prompt,
            "collection_mode": "curated_external_pull",
            "required_fields": ["title", "url", "source", "date", "snippet", "relevance_score"],
            "preferred_sources": ["official docs", "benchmarks", "papers", "trusted datasets"],
            "resource_seed": resources[:1500],
            "evidence_seed": evidence[:1500],
            "notes": source_text[:1200],
        }

    def _propose_replan_nodes(
        self,
        *,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any],
        metrics: dict[str, Any],
        failure_policy: dict[str, Any],
        completion_packet: dict[str, Any],
    ) -> list[dict[str, Any]]:
        existing_ids = {
            str(item.get("node_id", ""))
            for item in graph.get("nodes", [])
            if isinstance(item, dict)
        }
        proposed = self._live_replan_suggestions(
            node=node,
            graph=graph,
            context=context,
            metrics=metrics,
            failure_policy=failure_policy,
            completion_packet=completion_packet,
        )
        if not proposed:
            proposed = self._local_replan_suggestions(context=context, failure_policy=failure_policy, completion_packet=completion_packet)

        additions: list[dict[str, Any]] = []
        for item in proposed:
            node_type = str(item.get("node_type", "workspace_action")).strip() or "workspace_action"
            key = str(
                item.get("kind", "")
                or item.get("tool_name", "")
                or item.get("subagent_kind", "")
            ).strip()
            node_id = self._replan_node_id(node_type=node_type, key=key)
            if not key or node_id in existing_ids:
                continue
            addition = self._build_replanned_node(
                node_id=node_id,
                node_type=node_type,
                spec=item,
                prompt=str(metrics.get("prompt", context.get("query", ""))),
                workspace_summary=dict(metrics.get("workspace_summary", {})),
            )
            additions.append(addition)
        return additions

    @staticmethod
    def _local_replan_suggestions(*, context: dict[str, Any], failure_policy: dict[str, Any], completion_packet: dict[str, Any]) -> list[dict[str, Any]]:
        policy = str(failure_policy.get("policy", "none"))
        summary = str(failure_policy.get("summary", ""))
        state_gap = completion_packet.get("state_gap", {}) if isinstance(completion_packet.get("state_gap", {}), dict) else {}
        missing_artifacts = [str(item) for item in state_gap.get("missing_artifacts", []) if str(item).strip()] if isinstance(state_gap.get("missing_artifacts", []), list) else []
        missing_channels = [str(item) for item in state_gap.get("missing_channels", []) if str(item).strip()] if isinstance(state_gap.get("missing_channels", []), list) else []
        if policy == "none":
            return []
        if policy == "validation_gap":
            return [
                {
                    "node_type": "tool_call",
                    "tool_name": "workspace_file_search",
                    "tool_args": {"query": "test validation regression", "glob": "*", "limit": 6},
                    "title": "Inspect Validation Targets",
                    "reason": summary or "completion packet shows validation is still open",
                },
                {
                    "node_type": "subagent",
                    "subagent_kind": "repair_probe",
                    "objective": "Investigate open validation gaps and propose a bounded repair path",
                    "title": "Run Validation Repair Probe",
                    "reason": summary or "completion packet indicates unresolved validation gap",
                    "source_node_ids": ["analysis", "completion_packet"],
                },
            ]
        if policy == "workspace_gap":
            return [
                {
                    "node_type": "tool_call",
                    "tool_name": "workspace_file_search",
                    "tool_args": {"query": "repo workspace relevant files", "glob": "*", "limit": 8},
                    "title": "Inspect Missing Workspace Context",
                    "reason": summary or "completion packet shows missing workspace grounding",
                }
            ]
        if policy == "web_gap":
            return [
                {
                    "node_type": "tool_call",
                    "tool_name": "external_resource_hub",
                    "tool_args": {"query": "collect external evidence for unresolved task", "limit": 6},
                    "title": "Collect Missing Web Evidence",
                    "reason": summary or "completion packet shows missing external evidence",
                },
                {
                    "node_type": "subagent",
                    "subagent_kind": "research_probe",
                    "objective": "Close the missing external evidence gap with targeted research actions",
                    "title": "Run Web Evidence Probe",
                    "reason": summary or "delegate evidence gap closure",
                    "source_node_ids": ["analysis", "completion_packet"],
                },
            ]
        if policy == "artifact_gap":
            actions: list[dict[str, Any]] = []
            if any(item in {"patch_plan", "patch_draft"} for item in missing_artifacts):
                actions.append(
                    {
                        "node_type": "workspace_action",
                        "kind": "patch_draft",
                        "title": "Generate Missing Patch Draft",
                        "reason": summary or "completion packet shows patch artifact gap",
                        "source_node_ids": ["analysis", "completion_packet"],
                    }
                )
            if any(item in {"benchmark_manifest", "benchmark_run_config"} for item in missing_artifacts):
                actions.append(
                    {
                        "node_type": "workspace_action",
                        "kind": "benchmark_manifest",
                        "title": "Generate Missing Benchmark Manifest",
                        "reason": summary or "completion packet shows benchmark artifact gap",
                        "source_node_ids": ["analysis", "completion_packet"],
                    }
                )
            if "evidence_bundle" in missing_artifacts or "web" in missing_channels:
                actions.append(
                    {
                        "node_type": "tool_call",
                        "tool_name": "external_resource_hub",
                        "tool_args": {"query": "collect evidence for missing deliverables", "limit": 5},
                        "title": "Collect Evidence For Missing Artifacts",
                        "reason": summary or "artifact gap requires stronger evidence inputs",
                    }
                )
            actions.append(
                {
                    "node_type": "subagent",
                    "subagent_kind": "repair_probe",
                    "objective": "Resolve the remaining artifact gaps reported in the completion packet",
                    "title": "Run Artifact Gap Repair Probe",
                    "reason": summary or "completion packet lists unresolved artifact gaps",
                    "source_node_ids": ["analysis", "completion_packet"],
                }
            )
            return actions
        if policy == "missing_dependency":
            return [
                {
                    "node_type": "tool_call",
                    "tool_name": "workspace_file_search",
                    "tool_args": {"query": "import requirements dependency", "glob": "*", "limit": 6},
                    "title": "Search Dependency Signals",
                    "reason": summary or "dependency failure detected in validation output",
                },
                {
                    "node_type": "tool_call",
                    "tool_name": "tool_search",
                    "tool_args": {"query": "dependency install remediation", "limit": 5},
                    "title": "Discover Dependency Repair Tools",
                    "reason": summary or "need dependency-aware remediation tools",
                },
                {
                    "node_type": "subagent",
                    "subagent_kind": "repair_probe",
                    "objective": "Investigate missing dependency failures and suggest minimal remediation",
                    "title": "Run Dependency Repair Probe",
                    "reason": summary or "delegate dependency-focused repair analysis",
                    "source_node_ids": ["analysis", "execution"],
                },
            ]
        if policy == "timeout":
            return [
                {
                    "node_type": "workspace_action",
                    "kind": "benchmark_manifest",
                    "title": "Generate Timeout-Oriented Benchmark Manifest",
                    "reason": summary or "timeout suggests benchmark or runtime budgeting issue",
                    "source_node_ids": ["analysis", "execution"],
                },
                {
                    "node_type": "tool_call",
                    "tool_name": "code_experiment_design",
                    "tool_args": {"query": "timeout mitigation and runtime ablation", "max_experiments": 4},
                    "title": "Design Timeout Mitigation Experiments",
                    "reason": summary or "timeout should trigger runtime ablation design",
                },
                {
                    "node_type": "subagent",
                    "subagent_kind": "benchmark_probe",
                    "objective": "Investigate timeout bottlenecks and propose benchmark/runtime adjustments",
                    "title": "Run Timeout Benchmark Probe",
                    "reason": summary or "delegate timeout diagnosis to benchmark probe",
                    "source_node_ids": ["analysis", "execution"],
                },
            ]
        if policy == "evidence_gap":
            return [
                {
                    "node_type": "tool_call",
                    "tool_name": "external_resource_hub",
                    "tool_args": {"query": "collect missing evidence for current task", "limit": 5},
                    "title": "Collect Missing External Evidence",
                    "reason": summary or "evidence gap detected",
                },
                {
                    "node_type": "workspace_action",
                    "kind": "dataset_pull_spec",
                    "title": "Generate Missing-Evidence Dataset Pull Spec",
                    "reason": summary or "need reproducible data collection plan",
                    "source_node_ids": ["analysis"],
                },
                {
                    "node_type": "subagent",
                    "subagent_kind": "research_probe",
                    "objective": "Investigate evidence gaps and propose the next evidence collection steps",
                    "title": "Run Evidence Gap Probe",
                    "reason": summary or "delegate evidence diagnosis",
                    "source_node_ids": ["analysis"],
                },
            ]
        if policy == "tool_failure":
            return [
                {
                    "node_type": "tool_call",
                    "tool_name": "tool_search",
                    "tool_args": {"query": "tool failure fallback remediation", "limit": 5},
                    "title": "Discover Fallback Tools",
                    "reason": summary or "tool failure detected",
                },
                {
                    "node_type": "subagent",
                    "subagent_kind": "repair_probe",
                    "objective": "Investigate tool failure and propose fallback execution path",
                    "title": "Run Tool Failure Probe",
                    "reason": summary or "delegate tool-failure recovery analysis",
                    "source_node_ids": ["analysis"],
                },
            ]
        if policy == "assertion_failure":
            return [
                {
                    "node_type": "workspace_action",
                    "kind": "patch_draft",
                    "title": "Generate Assertion-Failure Patch Draft",
                    "reason": summary or "assertion failure suggests targeted code change",
                    "source_node_ids": ["analysis", "execution"],
                },
                {
                    "node_type": "tool_call",
                    "tool_name": "workspace_file_search",
                    "tool_args": {"query": "assert failed regression bug", "glob": "*", "limit": 6},
                    "title": "Search Assertion Failure Hotspots",
                    "reason": summary or "search workspace for regression hotspots",
                },
                {
                    "node_type": "subagent",
                    "subagent_kind": "repair_probe",
                    "objective": "Investigate assertion failures and suggest the smallest corrective patch",
                    "title": "Run Assertion Repair Probe",
                    "reason": summary or "delegate regression-focused repair analysis",
                    "source_node_ids": ["analysis", "execution"],
                },
            ]
        return [
            {
                "node_type": "workspace_action",
                "kind": "patch_scaffold",
                "title": "Generate Remediation Patch Scaffold",
                "reason": summary or "generic validation failure detected",
                "source_node_ids": ["analysis", "execution"],
            },
            {
                "node_type": "workspace_action",
                "kind": "patch_draft",
                "title": "Generate Remediation Patch Draft",
                "reason": summary or "generic validation failure detected",
                "source_node_ids": ["analysis", "execution"],
            },
            {
                "node_type": "tool_call",
                "tool_name": "tool_search",
                "tool_args": {"query": "repair validation failure", "limit": 5},
                "title": "Discover Remediation Tools",
                "reason": summary or "generic remediation search",
            },
            {
                "node_type": "subagent",
                "subagent_kind": "repair_probe",
                "objective": "Investigate the validation failure and suggest next remediation steps",
                "title": "Run Repair Probe",
                "reason": summary or "generic validation failure detected",
                "source_node_ids": ["analysis", "execution"],
            },
        ]

    def _live_replan_suggestions(
        self,
        *,
        node: dict[str, Any],
        graph: dict[str, Any],
        context: dict[str, Any],
        metrics: dict[str, Any],
        failure_policy: dict[str, Any],
        completion_packet: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            from app.harness.live_agent import CallBudget, LiveModelConfig, LiveModelGateway

            config = LiveModelConfig.resolve(self._resolve_live_model_overrides(context))
            if not config:
                return []
            gateway = LiveModelGateway(config)
            payload = {
                "query": str(metrics.get("prompt", context.get("query", ""))),
                "graph_id": graph.get("graph_id", ""),
                "replan_focus": metrics.get("replan_focus", []),
                "failure_policy": failure_policy,
                "completion_packet": completion_packet,
                "node_results": context.get("node_results", {}),
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are replanning a general agent task graph after execution feedback. "
                        "Return strict JSON with key actions. "
                        "Each action must include node_type from workspace_action, tool_call, subagent. "
                        "Use the completion_packet gaps to choose the smallest closure repair. "
                        f"Allowed workspace_action kinds: {', '.join(sorted(allowed_workspace_action_kinds(include_internal=False)))}. "
                        "You may also emit kind starting with custom: when you include relative_path plus content_type or artifact_contract. "
                        "Allowed tool_call names: tool_search, workspace_file_search, external_resource_hub, code_experiment_design. "
                        "Allowed subagent kinds: repair_probe, research_probe, benchmark_probe."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ]
            text, _meta = gateway.chat(
                messages=messages,
                budget=CallBudget(max_calls=1),
                temperature=0.0,
                require_json=True,
            )
            parsed = self._parse_json_dict(text)
            actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
            filtered: list[dict[str, Any]] = []
            for item in actions if isinstance(actions, list) else []:
                if not isinstance(item, dict):
                    continue
                node_type = str(item.get("node_type", "")).strip()
                if node_type not in {"workspace_action", "tool_call", "subagent"}:
                    continue
                filtered.append(item)
            return filtered
        except Exception:
            return []

    @staticmethod
    def _build_patch_draft(*, prompt: str, source_text: str, workspace_summary: dict[str, Any]) -> str:
        targets = workspace_summary.get("sample_files", []) if isinstance(workspace_summary.get("sample_files", []), list) else []
        preferred = [
            str(item)
            for item in targets
            if isinstance(item, str)
            and Path(item).suffix.lower() in {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java"}
            and "test" not in Path(item).name.lower()
        ]
        fallback_targets = [
            str(item)
            for item in targets
            if isinstance(item, str) and Path(item).name.lower() not in {"notes.md", "readme.md"}
        ]
        target = preferred[0] if preferred else (fallback_targets[0] if fallback_targets else (str(targets[0]) if targets else "src/module.py"))
        target_name = Path(target).name.lower()
        lowered_prompt = prompt.lower()
        if target_name.startswith("router") and any(marker in lowered_prompt for marker in {"route", "routing", "parser"}):
            return (
                f"diff --git a/{target} b/{target}\n"
                f"--- a/{target}\n"
                f"+++ b/{target}\n"
                "@@\n"
                "-def route(query):\n"
                "-    return \"research\" if \"report\" in query else \"general\"\n"
                "+def route(query):\n"
                "+    normalized = (query or \"\").strip().lower()\n"
                "+    if any(token in normalized for token in (\"report\", \"research\", \"benchmark\", \"paper\")):\n"
                "+        return \"research\"\n"
                "+    if any(token in normalized for token in (\"patch\", \"test\", \"repo\", \"workspace\", \"code\", \"parser\", \"routing\")):\n"
                "+        return \"code\"\n"
                "+    return \"general\"\n"
            )
        return (
            f"diff --git a/{target} b/{target}\n"
            f"--- a/{target}\n"
            f"+++ b/{target}\n"
            "@@\n"
            f"-# TODO: existing behavior tied to {prompt[:60]}\n"
            f"+# Draft patch for: {prompt[:80]}\n"
            "+# Narrow change surface and keep behavior testable.\n"
            "+\n"
            f"+# Grounding: {TaskGraphActionMapper._single_line(source_text)[:120]}\n"
        )

    @staticmethod
    def _build_benchmark_manifest(*, prompt: str, source_text: str, workspace_summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": "agent-harness-benchmark-manifest",
            "objective": prompt,
            "tracks": ["baseline", "ablation_a", "ablation_b"],
            "workspace_languages": list(workspace_summary.get("languages", [])),
            "workspace_frameworks": list(workspace_summary.get("frameworks", [])),
            "artifacts": {
                "scoreboard": "reports/scoreboard.json",
                "failures": "reports/failure-clusters.json",
                "report": "reports/benchmark-report.md",
            },
            "notes": TaskGraphActionMapper._single_line(source_text)[:240],
        }

    def _build_completion_packet(
        self,
        *,
        prompt: str,
        source_text: str,
        context: dict[str, Any],
        workspace_summary: dict[str, Any],
    ) -> dict[str, Any]:
        graph = dict(context.get("_graph", {})) if isinstance(context.get("_graph", {}), dict) else {}
        metrics = dict(context.get("_action_metrics", {})) if isinstance(context.get("_action_metrics", {}), dict) else {}
        task_spec = self._coerce_task_spec(metrics=metrics, graph=graph, context=context)
        world_state = build_world_state(graph=graph, context=context)
        state_gap = compute_state_gap(task_spec=task_spec, world_state=world_state)
        return self._compile_completion_packet_payload(
            prompt=prompt,
            source_text=source_text,
            context=context,
            workspace_summary=workspace_summary,
            task_spec=task_spec,
            world_state=world_state,
            state_gap=state_gap,
        )

    def _compile_completion_packet_payload(
        self,
        *,
        prompt: str,
        source_text: str,
        context: dict[str, Any],
        workspace_summary: dict[str, Any],
        task_spec: Any,
        world_state: Any,
        state_gap: Any,
    ) -> dict[str, Any]:
        packet_state_gap = self._filter_packet_state_gap(state_gap)
        node_results = context.get("node_results", {}) if isinstance(context.get("node_results", {}), dict) else {}

        delivered_artifacts: list[dict[str, Any]] = []
        for node_id, payload in node_results.items():
            if not isinstance(payload, dict):
                continue
            artifact = payload.get("artifact", {}) if isinstance(payload.get("artifact", {}), dict) else {}
            path = str(artifact.get("path", "")).strip()
            if not path:
                continue
            delivered_artifacts.append(
                {
                    "node_id": str(node_id),
                    "label": str(artifact.get("label", node_id)),
                    "kind": str(artifact.get("kind", "")),
                    "path": path,
                    "summary": str(artifact.get("summary", "")),
                    "content_type": str(artifact.get("content_type", "")),
                }
            )
        delivered_artifacts.sort(key=lambda item: (str(item.get("path", "")), str(item.get("node_id", ""))))

        evidence_output = self._node_output(node_results, "evidence")
        evidence_records = evidence_output.get("records", []) if isinstance(evidence_output.get("records", []), list) else []
        evidence_citations = evidence_output.get("citations", []) if isinstance(evidence_output.get("citations", []), list) else []
        evidence_summary = {
            "record_count": int(evidence_output.get("record_count", len(evidence_records)) or len(evidence_records)),
            "citation_count": len(evidence_citations),
            "citations": [str(item) for item in evidence_citations[:6] if str(item).strip()],
            "highlights": [str(item.get("title", item.get("summary", ""))) for item in evidence_records[:4] if isinstance(item, dict)],
        }

        risk_output = self._node_output(node_results, "risk")
        risk_items = risk_output.get("risks", []) if isinstance(risk_output.get("risks", []), list) else []
        risk_summary = {
            "count": len(risk_items),
            "top_risks": [
                {
                    "title": str(item.get("title", item.get("risk", ""))),
                    "severity": str(item.get("severity", "")),
                    "mitigation": str(item.get("mitigation", "")),
                }
                for item in risk_items[:5]
                if isinstance(item, dict)
            ],
        }

        validation_output = self._node_output(node_results, "validation")
        execution_output = self._node_output(node_results, "execution")
        execution_results = execution_output.get("results", []) if isinstance(execution_output.get("results", []), list) else []
        validation_status = "not_requested"
        if execution_results:
            validation_status = "passed" if world_state.validation_ok else "failed"
        elif validation_output:
            validation_status = "planned"
        validation_summary = {
            "status": validation_status,
            "validation_plan_present": bool(validation_output),
            "execution_count": len(execution_results),
            "commands": [
                {
                    "command": str(item.get("command", "")),
                    "exit_code": int(item.get("exit_code", 0)),
                }
                for item in execution_results[:6]
                if isinstance(item, dict)
            ],
        }

        open_gaps = (
            len(packet_state_gap["missing_channels"])
            + len(packet_state_gap["missing_artifacts"])
            + len(packet_state_gap["failure_types"])
            + (1 if packet_state_gap["missing_validation"] else 0)
        )
        next_steps: list[str] = []
        for channel in packet_state_gap["missing_channels"]:
            next_steps.append(f"Add or rerun a node that satisfies the {channel} channel.")
        for artifact in packet_state_gap["missing_artifacts"]:
            next_steps.append(f"Materialize the missing artifact: {artifact}.")
        if packet_state_gap["missing_validation"]:
            next_steps.append("Run or repair validation before treating the task as closed.")
        for failure in packet_state_gap["failure_types"]:
            next_steps.append(f"Repair execution failure classified as {failure}.")
        if not next_steps:
            next_steps.append("No blocking gaps detected; review the packet and promote the delivered artifacts.")

        return {
            "schema": "agent-harness-completion-packet/v1",
            "query": prompt,
            "goal": task_spec.goal,
            "task_spec": task_spec.to_dict(),
            "summary": {
                "artifact_count": len(delivered_artifacts),
                "channel_count": len(world_state.channels),
                "completed_capability_count": len(world_state.completed_capabilities),
                "open_gap_count": open_gaps,
                "validation_ok": bool(world_state.validation_ok),
            },
            "workspace_summary": {
                "languages": list(workspace_summary.get("languages", [])),
                "frameworks": list(workspace_summary.get("frameworks", [])),
                "sample_files": list(workspace_summary.get("sample_files", []))[:8],
            },
            "world_state": world_state.to_dict(),
            "state_gap": packet_state_gap,
            "delivered_artifacts": delivered_artifacts,
            "evidence": evidence_summary,
            "validation": validation_summary,
            "risk": risk_summary,
            "source_digest": self._single_line(source_text)[:300],
            "next_steps": next_steps,
        }

    def _build_delivery_bundle(
        self,
        *,
        prompt: str,
        source_text: str,
        context: dict[str, Any],
        workspace_summary: dict[str, Any],
    ) -> dict[str, Any]:
        del source_text
        node_results = context.get("node_results", {}) if isinstance(context.get("node_results", {}), dict) else {}
        completion_result = node_results.get("completion_packet", {}) if isinstance(node_results.get("completion_packet", {}), dict) else {}
        completion_result_body = completion_result.get("result", {}) if isinstance(completion_result.get("result", {}), dict) else {}
        completion_packet = completion_result_body.get("packet", {}) if isinstance(completion_result_body.get("packet", {}), dict) else {}
        if not completion_packet:
            completion_packet = self._node_output(node_results, "completion_packet")
        task_spec = completion_packet.get("task_spec", {}) if isinstance(completion_packet.get("task_spec", {}), dict) else {}
        delivered_artifacts = completion_packet.get("delivered_artifacts", []) if isinstance(completion_packet.get("delivered_artifacts", []), list) else []
        if not delivered_artifacts:
            delivered_artifacts = self._delivery_manifest_rows(node_results=node_results)
        validation = completion_packet.get("validation", {}) if isinstance(completion_packet.get("validation", {}), dict) else {}
        evidence = completion_packet.get("evidence", {}) if isinstance(completion_packet.get("evidence", {}), dict) else {}
        risk = completion_packet.get("risk", {}) if isinstance(completion_packet.get("risk", {}), dict) else {}
        summary = completion_packet.get("summary", {}) if isinstance(completion_packet.get("summary", {}), dict) else {}

        manifest: list[dict[str, Any]] = []
        for item in delivered_artifacts:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).replace("\\", "/")
            family = self._artifact_family_from_path(path)
            manifest.append(
                {
                    "node_id": str(item.get("node_id", "")),
                    "label": str(item.get("label", "")),
                    "kind": str(item.get("kind", "")),
                    "family": family,
                    "path": path,
                    "content_type": str(item.get("content_type", "")),
                    "summary": str(item.get("summary", "")),
                }
            )

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in manifest:
            grouped.setdefault(str(item.get("family", "misc")), []).append(item)
        deliverable_index = [
            {
                "family": family,
                "count": len(items),
                "paths": [str(item.get("path", "")) for item in items[:8]],
                "labels": [str(item.get("label", "")) for item in items[:8]],
            }
            for family, items in sorted(grouped.items())
        ]

        reviewer_checklist = [
            f"Review primary report and packet for task: {prompt}",
            f"Validation status is {str(validation.get('status', 'unknown')).replace('_', ' ')}.",
            f"Evidence records available: {int(evidence.get('record_count', 0))}.",
            f"Risk items captured: {int(risk.get('count', 0))}.",
        ]

        return {
            "schema": "agent-harness-delivery-bundle/v1",
            "query": prompt,
            "task_spec": task_spec,
            "workspace_summary": {
                "languages": list(workspace_summary.get("languages", [])),
                "frameworks": list(workspace_summary.get("frameworks", [])),
            },
            "bundle_summary": {
                "artifact_count": int(summary.get("artifact_count", len(manifest))),
                "family_count": len(deliverable_index),
                "validation_status": str(validation.get("status", "unknown")),
                "evidence_count": int(evidence.get("record_count", 0)),
                "risk_count": int(risk.get("count", 0)),
            },
            "deliverable_index": deliverable_index,
            "artifact_manifest": manifest,
            "completion_packet_ref": self._node_artifact_path(node_results, "completion_packet"),
            "report_ref": self._node_artifact_path(node_results, "report"),
            "reviewer_checklist": reviewer_checklist,
            "handoff_order": [str(item.get("path", "")) for item in manifest[:12]],
        }

    @staticmethod
    def _build_dataset_loader_template(*, prompt: str, source_text: str, context: dict[str, Any]) -> str:
        del context
        note = TaskGraphActionMapper._single_line(source_text)[:120]
        return (
            '"""Dataset loader template generated by agent-harness."""\n\n'
            "from __future__ import annotations\n\n"
            "import json\n"
            "from pathlib import Path\n\n\n"
            "def load_records(path: str | Path) -> list[dict[str, object]]:\n"
            "    payload = json.loads(Path(path).read_text(encoding='utf-8'))\n"
            "    rows = payload.get('records', []) if isinstance(payload, dict) else []\n"
            "    return [row for row in rows if isinstance(row, dict)]\n\n\n"
            "if __name__ == '__main__':\n"
            f"    print('loader ready for: {prompt[:80]}')\n"
            f"    print('grounding: {note}')\n"
        )

    @staticmethod
    def _build_webpage_blueprint(*, prompt: str, source_text: str) -> str:
        focus = TaskGraphActionMapper._compact_query(prompt, limit=6)
        note = TaskGraphActionMapper._single_line(source_text)[:220]
        return (
            "# Landing Page Blueprint\n\n"
            f"## Theme\n- Mission: {prompt}\n- Focus terms: {focus}\n\n"
            "## First Screen\n"
            "- Headline: make the promise concrete in one sentence.\n"
            "- Proof point: show one measurable signal or artifact.\n"
            "- CTA: primary action plus a lower-risk secondary action.\n\n"
            "## Section Architecture\n"
            "1. Problem and audience fit\n"
            "2. How the system works\n"
            "3. Evidence, safety, and governance\n"
            "4. Artifact gallery or case study\n"
            "5. Final conversion block\n\n"
            "## Interaction Notes\n"
            "- Prioritize scannability over jargon.\n"
            "- Put concrete artifacts above framework internals.\n\n"
            f"## Grounding\n{note}\n"
        )

    @staticmethod
    def _build_slide_deck_plan(*, prompt: str, source_text: str) -> str:
        note = TaskGraphActionMapper._single_line(source_text)[:220]
        return (
            "# Slide Deck Plan\n\n"
            f"- Presentation goal: {prompt}\n"
            "- Slide 1: opening tension and business context\n"
            "- Slide 2: what exists today and where it breaks\n"
            "- Slide 3: solution or system mechanism\n"
            "- Slide 4: evidence, benchmark, or artifact proof\n"
            "- Slide 5: rollout plan and ownership\n"
            "- Slide 6: risks, asks, and next decision\n\n"
            f"Grounding: {note}\n"
        )

    @staticmethod
    def _build_chart_pack_spec(*, prompt: str, source_text: str) -> dict[str, Any]:
        note = TaskGraphActionMapper._single_line(source_text)[:220]
        return {
            "objective": prompt,
            "charts": [
                {
                    "id": "headline_comparison",
                    "type": "bar",
                    "question": "Which option or segment leads on the main metric?",
                    "fields": ["category", "metric_value", "annotation"],
                },
                {
                    "id": "trend_view",
                    "type": "line",
                    "question": "How does the main signal change over time?",
                    "fields": ["date", "metric_value", "segment"],
                },
                {
                    "id": "risk_pocket",
                    "type": "scatter",
                    "question": "Where are the outliers, failures, or governance pockets?",
                    "fields": ["x_metric", "y_metric", "cluster", "label"],
                },
            ],
            "render_targets": ["dashboard", "briefing deck", "report appendix"],
            "notes": note,
        }

    @staticmethod
    def _build_podcast_episode_plan(*, prompt: str, source_text: str) -> str:
        note = TaskGraphActionMapper._single_line(source_text)[:220]
        return (
            "# Podcast Episode Plan\n\n"
            f"- Episode brief: {prompt}\n"
            "- Cold open: 15-second tension statement\n"
            "- Segment 1: background and why it matters now\n"
            "- Segment 2: explain the mechanism with one concrete example\n"
            "- Segment 3: discuss tradeoffs, risks, and disagreement\n"
            "- Closing: three takeaways and one open question\n\n"
            f"Grounding: {note}\n"
        )

    @staticmethod
    def _build_video_storyboard(*, prompt: str, source_text: str) -> str:
        note = TaskGraphActionMapper._single_line(source_text)[:220]
        return (
            "# Video Storyboard\n\n"
            f"- Brief: {prompt}\n"
            "- Scene 1: visual hook and core claim\n"
            "- Scene 2: product or system in action\n"
            "- Scene 3: charts, artifacts, or evidence montage\n"
            "- Scene 4: risks and contrast against alternatives\n"
            "- Final frame: CTA plus one-sentence takeaway\n\n"
            f"Grounding: {note}\n"
        )

    @staticmethod
    def _build_image_prompt_pack(*, prompt: str, source_text: str) -> str:
        note = TaskGraphActionMapper._single_line(source_text)[:220]
        return (
            "# Image Prompt Pack\n\n"
            f"- Subject: {prompt}\n"
            "- Prompt A: editorial hero visual with strong focal point\n"
            "- Prompt B: technical schematic with labeled modules\n"
            "- Prompt C: campaign poster with bold type hierarchy\n"
            "- Prompt D: product render showing user interaction\n"
            "- Shared constraints: aspect ratio, color palette, negative prompts, required labels\n\n"
            f"Grounding: {note}\n"
        )

    @staticmethod
    def _build_data_analysis_spec(*, prompt: str, source_text: str) -> dict[str, Any]:
        note = TaskGraphActionMapper._single_line(source_text)[:220]
        return {
            "objective": prompt,
            "analysis_questions": [
                "What is the north-star outcome?",
                "Which segments or cohorts explain the variance?",
                "What guardrail metrics should block rollout?",
            ],
            "required_tables": ["fact_table", "dimension_table", "quality_log"],
            "metrics": {
                "north_star": "primary_outcome",
                "diagnostics": ["conversion", "latency"],
                "guardrails": ["risk_rate"],
            },
            "outputs": ["analysis_report.md", "chart_pack.json", "dashboard_spec.md"],
            "notes": note,
        }

    @staticmethod
    def _classify_failure_policy(*, context: dict[str, Any], completion_packet: dict[str, Any] | None = None) -> dict[str, Any]:
        execution_json = TaskGraphActionMapper._extract_result_field("execution", "", context)
        if execution_json:
            try:
                parsed = json.loads(execution_json)
            except Exception:
                parsed = {}
            results = parsed.get("results", []) if isinstance(parsed, dict) else []
            combined = " ".join(
                TaskGraphActionMapper._single_line(str(item.get("stdout", "")) + " " + str(item.get("stderr", "")))
                for item in results
                if isinstance(item, dict)
            ).lower()
            failed = any(int(item.get("exit_code", 0)) != 0 for item in results if isinstance(item, dict))
            if failed:
                if any(marker in combined for marker in ["no module named", "modulenotfounderror", "importerror", "cannot find module", "command not found"]):
                    return {"policy": "missing_dependency", "summary": "missing dependency or unavailable command detected"}
                if any(marker in combined for marker in ["timeout", "timed out"]):
                    return {"policy": "timeout", "summary": "execution timed out"}
                if any(marker in combined for marker in ["assert", "failed", "failure", "traceback"]):
                    return {"policy": "assertion_failure", "summary": "tests or validation assertions failed"}
                return {"policy": "execution_failure", "summary": "execution failed without a more specific classification"}

        evidence_json = TaskGraphActionMapper._extract_result_field("evidence", "", context)
        if evidence_json:
            try:
                parsed = json.loads(evidence_json)
            except Exception:
                parsed = {}
            output = parsed.get("output", {}) if isinstance(parsed, dict) else {}
            count = 0
            if isinstance(output, dict):
                count = int(output.get("record_count", output.get("count", 0)) or 0)
                if count <= 0 and isinstance(output.get("records", []), list):
                    count = len(output.get("records", []))
            if count <= 0:
                return {"policy": "evidence_gap", "summary": "evidence collection returned no records"}

        tool_failures = [
            item
            for item in (context.get("node_results", {}) or {}).values()
            if isinstance(item, dict) and isinstance(item.get("result", {}), dict) and item["result"].get("success") is False
        ]
        if tool_failures:
            return {"policy": "tool_failure", "summary": f"{len(tool_failures)} tool node(s) failed"}
        packet = dict(completion_packet or {})
        state_gap = packet.get("state_gap", {}) if isinstance(packet.get("state_gap", {}), dict) else {}
        missing_channels = [str(item) for item in state_gap.get("missing_channels", []) if str(item).strip()] if isinstance(state_gap.get("missing_channels", []), list) else []
        missing_artifacts = [str(item) for item in state_gap.get("missing_artifacts", []) if str(item).strip()] if isinstance(state_gap.get("missing_artifacts", []), list) else []
        failure_types = [str(item) for item in state_gap.get("failure_types", []) if str(item).strip()] if isinstance(state_gap.get("failure_types", []), list) else []
        actionable_missing_artifacts = [
            item
            for item in missing_artifacts
            if item not in {"completion_packet", "delivery_bundle", "deliverable_report"}
        ]
        if failure_types:
            return {"policy": "artifact_gap", "summary": f"completion packet still reports execution failure types: {', '.join(failure_types[:3])}", "source": "completion_packet"}
        if bool(state_gap.get("missing_validation", False)):
            return {"policy": "validation_gap", "summary": "completion packet shows validation remains unresolved", "source": "completion_packet"}
        if "workspace" in missing_channels:
            return {"policy": "workspace_gap", "summary": "completion packet shows workspace grounding is still missing", "source": "completion_packet"}
        if "web" in missing_channels:
            return {"policy": "web_gap", "summary": "completion packet shows external evidence is still missing", "source": "completion_packet"}
        if actionable_missing_artifacts:
            return {"policy": "artifact_gap", "summary": f"completion packet shows missing artifacts: {', '.join(actionable_missing_artifacts[:3])}", "source": "completion_packet"}
        return {"policy": "none", "summary": "no repair policy triggered"}

    @staticmethod
    def _node_output(node_results: dict[str, Any], node_id: str) -> dict[str, Any]:
        payload = node_results.get(node_id, {}) if isinstance(node_results, dict) else {}
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        if isinstance(result, dict):
            output = result.get("output", {})
            if isinstance(output, dict):
                return output
            for field in ("packet", "bundle", "manifest", "config", "spec"):
                value = result.get(field, {})
                if isinstance(value, dict):
                    return value
            return result
        return {}

    @staticmethod
    def _node_artifact_path(node_results: dict[str, Any], node_id: str) -> str:
        payload = node_results.get(node_id, {}) if isinstance(node_results, dict) else {}
        artifact = payload.get("artifact", {}) if isinstance(payload, dict) else {}
        return str(artifact.get("path", ""))

    @staticmethod
    def _artifact_family_from_path(path: str) -> str:
        lowered = str(path or "").replace("\\", "/").lower()
        mapping = [
            ("report", ["/report", "report.md"]),
            ("packet", ["packets/"]),
            ("bundle", ["bundles/"]),
            ("code", ["patches/", "plans/"]),
            ("benchmark", ["benchmarks/"]),
            ("dataset", ["datasets/"]),
            ("web", ["web/"]),
            ("slides", ["slides/"]),
            ("charts", ["charts/"]),
            ("podcast", ["podcast/"]),
            ("video", ["video/"]),
            ("images", ["images/"]),
            ("analysis", ["analysis/"]),
            ("execution", ["executions/", "execution-trace"]),
            ("briefs", ["briefs/"]),
            ("artifacts", ["artifacts/"]),
        ]
        for family, patterns in mapping:
            if any(pattern in lowered for pattern in patterns):
                return family
        return "misc"

    def _delivery_manifest_rows(self, *, node_results: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node_id, payload in node_results.items():
            if not isinstance(payload, dict):
                continue
            artifact = payload.get("artifact", {}) if isinstance(payload.get("artifact", {}), dict) else {}
            path = str(artifact.get("path", "")).strip()
            if not path or path.replace("\\", "/").lower().endswith("bundles/delivery-bundle.json"):
                continue
            rows.append(
                {
                    "node_id": str(node_id),
                    "label": str(artifact.get("label", node_id)),
                    "kind": str(artifact.get("kind", "")),
                    "path": path,
                    "summary": str(artifact.get("summary", "")),
                    "content_type": str(artifact.get("content_type", "")),
                }
            )
        rows.sort(key=lambda item: (str(item.get("path", "")), str(item.get("node_id", ""))))
        return rows

    @staticmethod
    def _filter_packet_state_gap(state_gap: Any) -> dict[str, Any]:
        missing_channels = list(getattr(state_gap, "missing_channels", []))
        missing_artifacts = [
            str(item)
            for item in list(getattr(state_gap, "missing_artifacts", []))
            if str(item) not in {"completion_packet", "deliverable_report", "delivery_bundle"}
        ]
        failure_types = list(getattr(state_gap, "failure_types", []))
        return {
            "missing_channels": missing_channels,
            "missing_artifacts": missing_artifacts,
            "missing_validation": bool(getattr(state_gap, "missing_validation", False)),
            "failure_types": failure_types,
        }

    @staticmethod
    def _resolve_live_model_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
        payload = context.get("live_model", {})
        return dict(payload) if isinstance(payload, dict) and payload else None

    @staticmethod
    def _workspace_action_result_field(action_kind: str) -> str:
        return workspace_action_result_field(action_kind)

    @staticmethod
    def _allowed_subagent_tools() -> set[str]:
        return {
            "tool_search",
            "workspace_file_search",
            "workspace_file_read",
            "external_resource_hub",
            "evidence_dossier_builder",
            "code_experiment_design",
            "policy_risk_matrix",
        }

    @staticmethod
    def _allowed_subagent_skills() -> set[str]:
        return {
            "artifact_synthesis",
            "research_brief",
            "benchmark_ablation",
            "validation_planner",
            "codebase_triage",
            "ops_runbook",
            "webpage_blueprint",
            "slide_deck_designer",
            "chart_storyboard",
            "podcast_episode_plan",
            "video_storyboard",
            "image_prompt_pack",
            "data_analysis_plan",
        }

    @staticmethod
    def _parse_json_dict(text: str) -> dict[str, Any]:
        if not text:
            return {}
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return {}
            try:
                payload = json.loads(match.group(0))
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}

    @staticmethod
    def _build_replanned_node(
        *,
        node_id: str,
        node_type: str,
        spec: dict[str, Any],
        prompt: str,
        workspace_summary: dict[str, Any],
    ) -> dict[str, Any]:
        title_default = str(
            spec.get("title", spec.get("kind", spec.get("tool_name", spec.get("subagent_kind", "replan"))))
        ).replace("_", " ").title()
        base = {
            "node_id": node_id,
            "title": str(spec.get("title", title_default)),
            "node_type": node_type,
            "status": "ready",
            "depends_on": ["replan"],
            "commands": [],
            "notes": [str(spec.get("reason", "replan addition"))],
            "artifacts": [],
        }
        if node_type == "workspace_action":
            base["metrics"] = {
                "action_kind": str(spec.get("kind", "")),
                "prompt": prompt,
                "source_node_ids": list(spec.get("source_node_ids", ["analysis"])),
                "workspace_summary": workspace_summary,
            }
        elif node_type == "tool_call":
            base["metrics"] = {
                "tool_name": str(spec.get("tool_name", "tool_search")),
                "tool_args": dict(spec.get("tool_args", {"query": prompt, "limit": 5})),
            }
        elif node_type == "subagent":
            base["metrics"] = {
                "subagent_kind": str(spec.get("subagent_kind", "general_probe")),
                "objective": str(spec.get("objective", prompt)),
                "prompt": prompt,
                "source_node_ids": list(spec.get("source_node_ids", ["analysis"])),
                "workspace_summary": workspace_summary,
            }
        else:
            base["metrics"] = {}
        return base

    @staticmethod
    def _replan_node_id(*, node_type: str, key: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_") or "node"
        prefix = {"workspace_action": "action", "tool_call": "tool", "subagent": "subagent"}.get(node_type, "node")
        return f"replan_{prefix}_{normalized}"

    @staticmethod
    def _compact_query(text: str, limit: int = 5) -> str:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(text or ""))
        return " ".join(tokens[:limit]) or str(text or "")[:40]

    @staticmethod
    def _single_line(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()
