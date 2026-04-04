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
        live_generation = self._live_skill_generation(
            node_id=str(node.get("node_id", "")),
            skill_name=skill_name,
            prompt=prompt,
            source_text=source_text,
            local_output=output,
            context=context,
        )
        generation_source = "local"
        generation_model = ""
        if live_generation:
            output = str(live_generation.get("content_text", output))
            generation_source = str(live_generation.get("source", "live_model"))
            generation_model = str(live_generation.get("model", ""))
        body = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "node_type": node.get("node_type", ""),
            "skill_name": skill_name,
            "input": input_text,
            "output": output,
            "generation_source": generation_source,
            "generation_model": generation_model,
            "graph_id": graph.get("graph_id", ""),
        }
        return self._write_json_result(
            sandbox=sandbox,
            execution_id=execution_id,
            node=node,
            body=body,
            summary=f"skill {skill_name} executed via {generation_source}",
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
        if kind == "custom:source_matrix":
            return cls._build_source_matrix_document(prompt=prompt, source_text=source_text, title=title)
        if kind == "custom:research_outline":
            return cls._build_research_outline_document(prompt=prompt, source_text=source_text, title=title)
        if kind == "custom:direct_answer_baseline":
            return cls._build_direct_baseline_document(prompt=prompt, source_text=source_text, title=title)
        if kind in {"custom:memo", "custom:executive_memo", "custom:decision_memo", "custom:launch_memo", "custom:brief", "custom:one_pager"}:
            return cls._build_research_style_document(
                prompt=prompt,
                source_text=source_text,
                workspace_summary=workspace_summary,
                title=title,
                kind=kind,
                sections=sections,
            )
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

    @classmethod
    def _build_research_style_document(
        cls,
        *,
        prompt: str,
        source_text: str,
        workspace_summary: dict[str, Any],
        title: str,
        kind: str,
        sections: list[str],
    ) -> str:
        signal_map = cls._research_signal_map(prompt=prompt, source_text=source_text)
        lines = [f"# {title}", "", f"Objective: {prompt}", ""]
        lead = signal_map["lead"]
        if lead:
            lines.extend(["## Summary", "", lead, ""])
            for paragraph in cls._research_summary_paragraphs(prompt=prompt, signal_map=signal_map):
                lines.extend([paragraph, ""])
        for section in sections:
            paragraphs, bullets = cls._render_research_section(
                section=section,
                prompt=prompt,
                signal_map=signal_map,
                kind=kind,
            )
            if not paragraphs and not bullets:
                continue
            lines.extend([f"## {section}", ""])
            for paragraph in paragraphs:
                lines.extend([paragraph, ""])
            for item in bullets:
                lines.append(f"- {item}")
            lines.append("")
        if signal_map["citations"]:
            lines.extend(["## Evidence References", ""])
            for item in signal_map["citations"][:6]:
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
            "custom:memo": ["Context", "Evidence", "Recommendation", "Implications", "Next Step"],
            "custom:brief": ["Question", "Key Findings", "Implications", "Open Questions"],
            "custom:one_pager": ["Headline", "Why It Matters", "Proof", "Next Step"],
            "custom:source_matrix": ["Question", "Source", "Usefulness", "Open Gaps"],
            "custom:research_outline": ["Core Thesis", "Sections", "Evidence Coverage", "Missing Proof"],
            "custom:direct_answer_baseline": ["Baseline Answer", "What It Misses", "What Harness Must Add"],
            "custom:checklist": ["Checklist"],
            "custom:faq": ["FAQ"],
        }
        return mapping.get(kind, ["Summary", "Evidence", "Next Step"]) or [title]

    @classmethod
    def _research_signal_map(cls, *, prompt: str, source_text: str) -> dict[str, Any]:
        grounding = cls._grounding_snapshot(source_text)
        evidence_payload = cls._research_evidence_payload(source_text)
        evidence_records = evidence_payload["records"]
        citations = evidence_payload["citations"]
        baseline_lines = evidence_payload["baseline_lines"]
        source_lines = evidence_payload["source_lines"]
        benchmark_focus = evidence_payload["benchmark_focus"]
        source_rows = evidence_payload["source_rows"]
        evidence_titles = cls._dedupe_lines(
            [
                record["title"] + (f": {record['summary']}" if record.get("summary") else "")
                for record in evidence_records
                if record.get("title")
            ]
            + benchmark_focus
            + grounding.get("evidence", [])
        )[:8]

        direct_prompt = "direct model answer" in prompt.lower() or "direct answer" in prompt.lower()
        lead = grounding["summary"][0] if grounding["summary"] else f"This memo addresses {prompt}."
        if direct_prompt:
            lead = (
                "General agent frameworks usually lose to a direct model answer when orchestration adds latency, weak intermediate artifacts, "
                "and low-signal planning overhead without producing stronger evidence or a better final deliverable."
            )
        benchmark_clause = cls._join_natural_list(benchmark_focus[:3])
        evidence_clause = cls._join_natural_list([record["title"] for record in evidence_records[:3] if record.get("title")])
        gap_candidates = [
            "Orchestration overhead often creates more intermediate metadata than end-user value.",
            "Evidence collection is frequently disconnected from the final answer, so frameworks do more work without producing better synthesis.",
            "Closure artifacts are often generated before the true primary deliverable is strong enough, which buries the real result.",
            "Benchmark and validation surfaces are often scaffold-like rather than executable enough to prove advantage.",
        ]
        if evidence_titles:
            gap_candidates.insert(0, f"Current evidence points to repeated emphasis on {evidence_titles[0]}.")
        if benchmark_clause:
            gap_candidates.append(
                f"The benchmark-facing evidence centers on {benchmark_clause}, so the framework has to prove value on verifiable external tasks rather than internal scoring."
            )
        recommendation_candidates = [
            "Make the primary deliverable the center of the runtime, with packet and bundle generated only after the result is strong.",
            "Use external evidence and source matrices to improve the final synthesis directly, not as detached side artifacts.",
            "Suppress benchmark, dataset, or validation scaffolds unless the user explicitly asks for them.",
            "Reserve long task graphs for cases that end in verifiable artifacts, inspectable evidence, or executable outputs.",
        ]
        if citations:
            recommendation_candidates.append(f"Keep the strongest evidence references visible in the final deliverable, starting with {citations[0]}.")
        if source_rows:
            recommendation_candidates.append(
                "Force every major claim to map to a source row, an implication, and an unresolved uncertainty before treating the result as publishable."
            )
        open_questions = [
            "Which parts of the improvement story are still narrative rather than benchmark-backed?",
            "Which runtime surfaces create genuine closure value and which are only orchestration ceremony?",
            "What evidence would convince a skeptical reviewer that the framework beats a direct model answer on a real task?",
        ]
        if evidence_clause:
            open_questions.insert(
                0,
                f"How much of the current argument is already supported by {evidence_clause}, and where is the proof still indirect?",
            )
        return {
            "lead": lead,
            "grounding": grounding,
            "records": evidence_records,
            "source_rows": source_rows,
            "evidence_titles": evidence_titles,
            "citations": citations,
            "baseline_lines": baseline_lines,
            "source_lines": source_lines,
            "benchmark_focus": benchmark_focus,
            "gaps": gap_candidates,
            "recommendations": recommendation_candidates,
            "open_questions": open_questions,
        }

    @classmethod
    def _research_section_points(
        cls,
        *,
        section: str,
        prompt: str,
        signal_map: dict[str, Any],
        kind: str,
    ) -> list[str]:
        name = section.lower()
        evidence_titles = signal_map.get("evidence_titles", [])
        citations = signal_map.get("citations", [])
        baseline_lines = signal_map.get("baseline_lines", [])
        source_lines = signal_map.get("source_lines", [])
        grounding = signal_map.get("grounding", {})
        if "context" in name or "launch goal" in name or "headline" in name or "question" in name:
            return [
                signal_map.get("lead", f"This document addresses {prompt}."),
                f"The task is to turn the request into a result that is more valuable than a direct single-model answer for: {cls._single_line(prompt)}",
            ]
        if "decision" in name or "recommendation" in name:
            return signal_map.get("recommendations", [])[:3]
        if "evidence" in name or "proof" in name or "findings" in name:
            rows = evidence_titles[:4] or grounding.get("evidence", [])[:4]
            if citations:
                rows = rows + [f"Evidence references include {citations[0]}."]
            return rows[:4]
        if "tradeoff" in name or "risk" in name or "implication" in name:
            rows = baseline_lines[:2]
            rows.extend(signal_map.get("gaps", [])[:3])
            return cls._dedupe_lines(rows)[:4]
        if "next" in name or "execution" in name or "ask" in name or "plan" in name:
            rows = signal_map.get("recommendations", [])[:3]
            if source_lines:
                rows.append(f"Use a source matrix to keep claims grounded: {source_lines[0]}")
            return cls._dedupe_lines(rows)[:4]
        if "why it matters" in name:
            return signal_map.get("gaps", [])[:3]
        return grounding.get("summary", [])[:2] or signal_map.get("recommendations", [])[:2]

    @classmethod
    def _build_source_matrix_document(cls, *, prompt: str, source_text: str, title: str) -> str:
        signal_map = cls._research_signal_map(prompt=prompt, source_text=source_text)
        rows = signal_map.get("source_rows", [])
        citations = signal_map.get("citations", [])
        question = cls._single_line(prompt)[:110]
        lines = [
            f"# {title}",
            "",
            f"Objective: {prompt}",
            "",
            "| Question | Source | What It Proves | Why It Matters | Remaining Uncertainty |",
            "| --- | --- | --- | --- | --- |",
        ]
        rows_added = 0
        for row in rows[:6]:
            if not isinstance(row, dict):
                continue
            source = cls._markdown_table_cell(row.get("source", "Unknown source"))
            proves = cls._markdown_table_cell(row.get("claim", "Provides partial evidence for the main question."))
            matters = cls._markdown_table_cell(row.get("usefulness", "Helps separate framework substance from orchestration overhead."))
            gap = cls._markdown_table_cell(row.get("gap", "Needs stronger linkage to a real delivered artifact."))
            lines.append(f"| {cls._markdown_table_cell(question)} | {source} | {proves} | {matters} | {gap} |")
            rows_added += 1
        for citation in citations[: max(0, 4 - rows_added)]:
            source = cls._markdown_table_cell(citation)
            lines.append(
                f"| {cls._markdown_table_cell(question)} | {source} | External reference exists but has not yet been distilled into a precise claim. | Useful as an independent anchor for the memo. | Needs source-specific interpretation and linkage to the recommendation. |"
            )
            rows_added += 1
        if rows_added == 0:
            lines.append(
                f"| {cls._markdown_table_cell(question)} | No source captured | Missing evidence | Collect stronger external evidence before promoting the result. | The framework cannot justify a differentiated recommendation yet. |"
            )
        lines.extend(["", "## Reading Notes", ""])
        for item in signal_map.get("gaps", [])[:3]:
            lines.append(f"- {item}")
        return "\n".join(lines).strip() + "\n"

    @classmethod
    def _build_research_outline_document(cls, *, prompt: str, source_text: str, title: str) -> str:
        signal_map = cls._research_signal_map(prompt=prompt, source_text=source_text)
        lines = [
            f"# {title}",
            "",
            f"Objective: {prompt}",
            "",
            "## Core Thesis",
            "",
            f"- {signal_map.get('lead', prompt)}",
            "",
            "## Sections",
            "",
            "- 1. Where direct prompting currently wins",
            "- 2. What frameworks are supposed to add but often fail to deliver",
            "- 3. Evidence and benchmark signals that matter",
            "- 4. Runtime and product architecture changes with the highest leverage",
            "",
            "## Evidence Coverage",
            "",
        ]
        for item in signal_map.get("evidence_titles", [])[:4]:
            lines.append(f"- {item}")
        if not signal_map.get("evidence_titles", []):
            lines.append("- Evidence still needs to be strengthened before final publication.")
        lines.extend(["", "## Missing Proof", ""])
        for item in signal_map.get("gaps", [])[:4]:
            lines.append(f"- {item}")
        return "\n".join(lines).strip() + "\n"

    @classmethod
    def _build_direct_baseline_document(cls, *, prompt: str, source_text: str, title: str) -> str:
        signal_map = cls._research_signal_map(prompt=prompt, source_text=source_text)
        evidence_titles = signal_map.get("evidence_titles", [])
        benchmark_focus = cls._join_natural_list(signal_map.get("benchmark_focus", [])[:3])
        lines = [
            f"# {title}",
            "",
            f"Objective: {prompt}",
            "",
            "## Baseline Answer",
            "",
            (
                "A strong direct model answer would likely argue that general agent frameworks underperform when orchestration overhead, tool-selection noise, and "
                "intermediate artifact churn consume more budget than the final synthesis gains back."
            ),
            "",
            (
                "It would probably recommend simplifying the stack, improving planning, and tightening evaluation, but it would still tend to stop at narrative advice "
                "rather than leaving an inspectable evidence rail or executable follow-up assets."
            ),
            "",
            "## What It Misses",
            "",
            (
                "A direct answer usually does not leave a source matrix, evidence bundle, or artifact index that a reviewer can inspect after the text is written. "
                "That means the answer may sound plausible while still hiding which claims are strongly supported and which are extrapolated."
            ),
            "",
            (
                "It also tends to blend diagnosis and prescription. The answer says to simplify, benchmark, or improve planning, but it rarely shows which parts of the "
                "framework are pure ceremony and which runtime surfaces are genuinely worth keeping because they create closure, validation, or reuse."
            ),
            "",
            "## What Harness Must Add",
            "",
            (
                "Harness must promote one primary deliverable instead of a cloud of intermediate metadata, and every supporting artifact must strengthen that deliverable "
                "rather than compete with it for attention."
            ),
            "",
            (
                "Harness must also link recommendations to concrete supporting evidence and openable artifacts. The differentiator is not just 'more steps' but traceable "
                "proof, clearer synthesis, and reusable execution surfaces."
            ),
            "",
            (
                "Finally, Harness should keep task-graph complexity only when it yields verifiable files, reproducible evidence, or executable follow-up assets. If a step "
                "cannot improve the answer, support validation, or create a reusable artifact, it should probably disappear."
            ),
        ]
        if benchmark_focus:
            lines.extend(
                [
                    "",
                    "## Benchmark Lens",
                    "",
                    f"The most relevant benchmark surfaces in the current evidence are {benchmark_focus}. A direct answer would mention them; Harness has to connect them to concrete runtime changes and delivery quality.",
                ]
            )
        if evidence_titles:
            lines.extend(["", "## Supporting Signals", ""])
            for item in evidence_titles[:4]:
                lines.append(f"- {item}")
        return "\n".join(lines).strip() + "\n"

    @classmethod
    def _research_summary_paragraphs(cls, *, prompt: str, signal_map: dict[str, Any]) -> list[str]:
        benchmark_focus = cls._join_natural_list(signal_map.get("benchmark_focus", [])[:3])
        evidence_titles = cls._join_natural_list(signal_map.get("evidence_titles", [])[:3])
        paragraphs = [
            (
                "The core question is not whether frameworks can orchestrate more steps than a direct model answer, but whether those extra steps create a result that is "
                "measurably more useful. In most failures, the framework spends its budget on planning, routing, packaging, and meta-artifacts without meaningfully improving "
                "the final answer, the evidence quality, or the ability to take the next action."
            )
        ]
        if benchmark_focus or evidence_titles:
            paragraphs.append(
                (
                    f"The current evidence surface is anchored by {benchmark_focus or evidence_titles}. That matters because these sources evaluate whether an agent can "
                    "translate reasoning into externally verifiable outcomes, not just plausible prose. A strong framework therefore has to win on grounded synthesis, artifact "
                    "quality, and task closure rather than on the number of internal components it activates."
                )
            )
        return paragraphs

    @classmethod
    def _render_research_section(
        cls,
        *,
        section: str,
        prompt: str,
        signal_map: dict[str, Any],
        kind: str,
    ) -> tuple[list[str], list[str]]:
        del kind
        name = section.lower()
        records = signal_map.get("records", [])
        source_rows = signal_map.get("source_rows", [])
        evidence_titles = signal_map.get("evidence_titles", [])
        recommendations = signal_map.get("recommendations", [])
        gaps = signal_map.get("gaps", [])
        baseline_lines = signal_map.get("baseline_lines", [])
        benchmark_focus = cls._join_natural_list(signal_map.get("benchmark_focus", [])[:3])
        evidence_focus = cls._join_natural_list([record.get("title", "") for record in records[:3] if isinstance(record, dict)])
        paragraphs: list[str] = []
        bullets: list[str] = []

        if "context" in name or "launch goal" in name or "headline" in name or "question" in name:
            paragraphs.append(
                (
                    f"This document addresses {cls._single_line(prompt)}. The practical objective is to turn the request into a result that is more valuable than a direct "
                    "single-model answer, which means the framework must contribute either stronger evidence, better synthesis, clearer traceability, or a reusable executable artifact."
                )
            )
            if benchmark_focus or evidence_focus:
                paragraphs.append(
                    (
                        f"The strongest available support currently comes from {benchmark_focus or evidence_focus}. Those signals should shape the argument because they test whether "
                        "the framework can actually close work in realistic environments rather than merely produce a polished narrative."
                    )
                )
            bullets = evidence_titles[:3]
            return paragraphs, bullets

        if "evidence" in name or "proof" in name or "findings" in name:
            if source_rows:
                paragraphs.append(
                    (
                        "The evidence does not support the idea that more orchestration is automatically better. Instead, the pattern across the captured sources is that framework "
                        "value appears only when the extra runtime structure improves grounding, creates a verifiable artifact, or reduces execution risk."
                    )
                )
                paragraphs.append(
                    (
                        "Where the framework merely adds routing logic, packaging layers, or synthetic intermediate files, it often loses to a direct answer because the user gets "
                        "more process and not more usable substance."
                    )
                )
                bullets = [
                    (
                        f"{row.get('source', 'Source')}"
                        + (f" (trust {float(row.get('trust_score', 0.0) or 0.0):.2f})" if float(row.get("trust_score", 0.0) or 0.0) > 0 else "")
                        + f": {row.get('claim', row.get('usefulness', 'Relevant evidence captured.'))}"
                    )
                    for row in source_rows[:4]
                    if isinstance(row, dict)
                ]
            else:
                paragraphs.append(
                    "The available evidence is still thin, but even the current signal suggests that benchmark-grounded, externally inspectable results matter more than elaborate internal orchestration."
                )
                bullets = evidence_titles[:4]
            return paragraphs, cls._dedupe_lines(bullets)[:4]

        if "decision" in name or "recommendation" in name:
            paragraphs.append(
                (
                    "The highest-leverage move is to center the runtime on one primary deliverable and demote every supporting artifact to a clearly subordinate role. That forces "
                    "the framework to justify its existence at the user-facing output surface instead of behind the scenes."
                )
            )
            paragraphs.append(
                (
                    "The second move is to make evidence enter the final synthesis directly. Source collection, benchmark references, and workspace findings should not sit in parallel "
                    "tracks; they should materially change the wording, confidence, and recommended next actions of the main deliverable."
                )
            )
            return paragraphs, recommendations[:4]

        if "tradeoff" in name or "risk" in name or "implication" in name:
            paragraphs.append(
                (
                    "The main tradeoff is between generality and useful closure. A framework that keeps every possible branch alive usually pays for that flexibility with slower execution, "
                    "weaker prioritization, and lower final-answer density."
                )
            )
            paragraphs.append(
                (
                    "That does not mean the runtime should collapse into a fixed workflow. It means the runtime needs stronger criteria for when a step is worth keeping: it should either "
                    "improve evidence, improve the main deliverable, or produce a reusable executable asset."
                )
            )
            bullets = cls._dedupe_lines((baseline_lines[:2] if baseline_lines else []) + gaps[:3])[:4]
            return paragraphs, bullets

        if "next" in name or "execution" in name or "ask" in name or "plan" in name:
            paragraphs.append(
                (
                    "The next iteration should remove low-value ceremony and invest more compute in high-value synthesis. In practice, that means shorter default graphs for report-like tasks, "
                    "stronger live-model revision on the final deliverable, and tighter coupling between evidence rows and recommendations."
                )
            )
            if source_rows:
                paragraphs.append(
                    "A useful implementation rule is that every major recommendation must point to a specific source row and a remaining uncertainty. That preserves rigor without dragging the user through internal runtime mechanics."
                )
            bullets = recommendations[:3]
            if signal_map.get("source_lines", []):
                bullets.append(f"Keep claims grounded through a source matrix: {signal_map.get('source_lines', [''])[0]}")
            return paragraphs, cls._dedupe_lines(bullets)[:4]

        if "why it matters" in name:
            paragraphs.append(
                "This matters because users do not buy frameworks for beautiful orchestration diagrams; they buy them for outcomes that beat what a strong direct model call can already deliver. The framework therefore needs a clearer theory of advantage at the result surface."
            )
            return paragraphs, gaps[:3]

        if "open question" in name:
            paragraphs.append(
                "The remaining uncertainty is not whether the framework can execute more steps, but which of those steps create durable value across tasks and which ones should be stripped out."
            )
            return paragraphs, signal_map.get("open_questions", [])[:4]

        bullets = cls._research_section_points(section=section, prompt=prompt, signal_map=signal_map, kind="")
        return paragraphs, bullets[:4]

    @classmethod
    def _research_evidence_payload(cls, source_text: str) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        citations: list[str] = []
        baseline_lines: list[str] = []
        source_lines: list[str] = []
        benchmark_focus: list[str] = []
        source_rows: list[dict[str, str]] = []

        for payload in cls._json_objects_from_text(source_text, limit=10):
            output = payload.get("output", {})
            if (not isinstance(output, dict)) or (not output and any(key in payload for key in ["records", "resources", "citations"])):
                output = payload if any(key in payload for key in ["records", "resources", "citations"]) else {}
            if isinstance(output, dict):
                for record in output.get("records", [])[:8] if isinstance(output.get("records", []), list) else []:
                    if not isinstance(record, dict):
                        continue
                    title = cls._single_line(record.get("title", ""))
                    summary = cls._single_line(record.get("summary", ""))
                    url = cls._single_line(record.get("url", record.get("link", "")))
                    if title:
                        records.append(
                            {
                                "title": title,
                                "summary": summary,
                                "url": url,
                                "source_id": cls._single_line(record.get("source_id", "")),
                                "trust_score": float(record.get("trust_score", 0.0) or 0.0),
                                "freshness_hint": cls._single_line(record.get("freshness_hint", "")),
                            }
                        )
                for resource in output.get("resources", [])[:8] if isinstance(output.get("resources", []), list) else []:
                    if not isinstance(resource, dict):
                        continue
                    title = cls._single_line(resource.get("title", ""))
                    summary = cls._single_line(resource.get("summary", ""))
                    url = cls._single_line(resource.get("url", ""))
                    if title:
                        records.append(
                            {
                                "title": title,
                                "summary": summary,
                                "url": url,
                                "source_id": cls._single_line(resource.get("source_id", "")),
                                "trust_score": float(resource.get("trust_score", 0.0) or 0.0),
                                "freshness_hint": cls._single_line(resource.get("freshness_hint", "")),
                            }
                        )
                for item in output.get("citations", [])[:10] if isinstance(output.get("citations", []), list) else []:
                    text = cls._single_line(item)
                    if text:
                        citations.append(text)
            metadata = payload.get("__tool_metadata__", {}) if isinstance(payload.get("__tool_metadata__", {}), dict) else {}
            for record in metadata.get("evidence_records", [])[:8] if isinstance(metadata.get("evidence_records", []), list) else []:
                if not isinstance(record, dict):
                    continue
                title = cls._single_line(record.get("title", ""))
                summary = cls._single_line(record.get("summary", ""))
                url = cls._single_line(record.get("url", record.get("path", "")))
                if title:
                    records.append(
                        {
                            "title": title,
                            "summary": summary,
                            "url": url,
                            "source_id": cls._single_line(record.get("source_id", "")),
                            "trust_score": float(record.get("trust_score", 0.0) or 0.0),
                            "freshness_hint": cls._single_line(record.get("freshness_hint", "")),
                        }
                    )
            for item in metadata.get("evidence_citations", [])[:10] if isinstance(metadata.get("evidence_citations", []), list) else []:
                text = cls._single_line(item)
                if text:
                    citations.append(text)
            text_output = str(payload.get("output", "")).strip()
            lowered = text_output.lower()
            if "baseline answer" in lowered or "what it misses" in lowered:
                baseline_lines.extend(cls._meaningful_lines(text_output, limit=8))
            if "| question |" in lowered and "| source |" in lowered:
                source_rows.extend(cls._parse_source_matrix_rows(text_output))
            if "source matrix" in lowered or ("question" in lowered and "source" in lowered):
                source_lines.extend(cls._meaningful_lines(text_output, limit=8))

        for line in cls._meaningful_lines(source_text, limit=36):
            lowered = line.lower()
            stripped = line.lstrip()
            if stripped.startswith("{") or stripped.startswith("[{"):
                continue
            if line.startswith("http://") or line.startswith("https://") or line.startswith("internal://"):
                citations.append(line)
                continue
            if lowered.startswith("citation:"):
                citations.append(cls._single_line(line.split(":", 1)[1]))
                continue
            if any(marker in lowered for marker in ["tau-bench", "swe-bench", "model context protocol", "webarena", "gaia benchmark", "gaia ", "mcp"]):
                benchmark_focus.append(line)
            if "baseline answer" in lowered or "what it misses" in lowered:
                baseline_lines.append(line)
            if ("question:" in lowered or "source:" in lowered or "usefulness:" in lowered or "what it proves" in lowered) and line not in source_lines:
                source_lines.append(line)

        if not source_rows:
            for record in records[:6]:
                title = record.get("title", "")
                trust_score = float(record.get("trust_score", 0.0) or 0.0)
                source_id = cls._single_line(record.get("source_id", ""))
                freshness = cls._single_line(record.get("freshness_hint", ""))
                source_bundle = cls._evidence_record_claim_bundle(record)
                provenance_bits = []
                if source_id:
                    provenance_bits.append(f"provider={source_id}")
                if trust_score > 0:
                    provenance_bits.append(f"trust={trust_score:.2f}")
                if freshness:
                    provenance_bits.append(f"freshness={freshness}")
                source_rows.append(
                    {
                        "source": title or "Captured source",
                        "claim": source_bundle["claim"],
                        "usefulness": source_bundle["usefulness"]
                        + (f" ({', '.join(provenance_bits)})" if provenance_bits else ""),
                        "gap": source_bundle["gap"],
                        "trust_score": trust_score,
                    }
                )

        deduped_records: list[dict[str, Any]] = []
        seen_record_keys: set[str] = set()
        for record in records:
            key = f"{record.get('title', '')}|{record.get('url', '')}"
            if not record.get("title") or key in seen_record_keys:
                continue
            seen_record_keys.add(key)
            deduped_records.append(record)
            if len(deduped_records) >= 8:
                break

        deduped_rows: list[dict[str, str]] = []
        seen_sources: set[str] = set()
        for row in source_rows:
            source = cls._single_line(row.get("source", ""))
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)
            deduped_rows.append(
                {
                    "source": source,
                    "claim": cls._single_line(row.get("claim", "")),
                    "usefulness": cls._single_line(row.get("usefulness", "")),
                    "gap": cls._single_line(row.get("gap", "")),
                    "trust_score": float(row.get("trust_score", 0.0) or 0.0),
                }
            )
            if len(deduped_rows) >= 8:
                break

        return {
            "records": deduped_records,
            "citations": cls._dedupe_lines(citations)[:10],
            "baseline_lines": cls._dedupe_lines(baseline_lines)[:8],
            "source_lines": cls._dedupe_lines(source_lines)[:8],
            "benchmark_focus": cls._dedupe_lines(benchmark_focus)[:6],
            "source_rows": deduped_rows,
        }

    @classmethod
    def _evidence_record_claim_bundle(cls, record: dict[str, Any]) -> dict[str, str]:
        title = cls._single_line(record.get("title", ""))
        summary = cls._single_line(record.get("summary", ""))
        lowered = f"{title} {summary}".lower()
        source_id = cls._single_line(record.get("source_id", ""))
        base_claim = summary or f"{title or 'This source'} provides partial evidence for the main question."
        claim = base_claim
        usefulness = "Helps separate framework substance from orchestration overhead."
        gap = "Needs explicit linkage to a concrete recommendation or shipped artifact."

        if "model context protocol" in lowered or title.lower() in {"mcp", "model context protocol architecture"}:
            claim = (
                "Defines a standard for interoperable agent-tool integration, making tool-boundary reliability and capability composition a first-class architectural concern."
            )
            usefulness = "Relevant because many framework failures come from ad hoc tool interfaces rather than from reasoning quality alone."
            gap = "It supports the interoperability argument, but it does not prove that protocol adoption improves latency, accuracy, or user value."
        elif "tau-bench" in lowered:
            claim = "Shows that realistic long-horizon tool-using tasks need externally verifiable task closure, not just plausible reasoning traces."
            usefulness = "Useful because it anchors evaluation on enterprise-style workflows where orchestration quality must survive realistic execution chains."
            gap = "It is a benchmark surface, not direct evidence of cost, latency, or broad user preference across all task categories."
        elif "swe-bench" in lowered:
            claim = "Measures whether agents can finish verifiable software tasks with concrete code changes, making closure quality inspectable rather than rhetorical."
            usefulness = "Important because it tests whether a framework can produce auditable task completion on engineering work instead of only polished analysis."
            gap = "It is strong evidence for code-task closure, but it does not generalize by itself to research, ops, or non-engineering tasks."
        elif "promotion criteria" in lowered or "reproducibility" in lowered:
            claim = "Argues that research candidates should only be promoted when reproducibility, benchmark stability, and operating constraints are jointly satisfied."
            usefulness = "Useful as an operating gate: it turns framework evaluation from narrative optimism into explicit promotion criteria."
            gap = "It is a decision policy, not an external benchmark result showing that a specific framework already wins in production."
        elif "langgraph" in lowered or "durability" in lowered:
            claim = "Describes stateful orchestration and durability patterns that matter for long-running agent execution."
            usefulness = "Relevant because it shows which runtime mechanisms support recovery and persistence when multi-step execution is genuinely necessary."
            gap = "It explains orchestration patterns, but it is not direct evidence that more orchestration improves end-user outcomes."
        elif "benchmark" in lowered:
            usefulness = "Helps anchor the memo in an externally inspectable evaluation surface rather than purely internal framework claims."
            gap = "Needs linkage to the specific failure mode and the task class the benchmark actually covers."
        elif "architecture" in lowered or "protocol" in lowered:
            usefulness = "Helps connect high-level runtime design choices to concrete integration constraints."
            gap = "Supports the architectural framing, but not quantitative claims about performance advantage."

        if not claim:
            claim = base_claim
        return {"claim": claim, "usefulness": usefulness, "gap": gap}

    @classmethod
    def _parse_source_matrix_rows(cls, text: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for line in re.split(r"\r?\n+", str(text or "")):
            raw = line.strip()
            if not raw.startswith("|") or raw.count("|") < 5:
                continue
            if "---" in raw:
                continue
            cells = [cls._single_line(cell) for cell in raw.strip("|").split("|")]
            if len(cells) < 4 or cells[0].lower() == "question":
                continue
            question = cells[0]
            rows.append(
                {
                    "source": cells[1] if len(cells) > 1 else "",
                    "claim": cells[2] if len(cells) > 2 else question,
                    "usefulness": cells[3] if len(cells) > 3 else "",
                    "gap": cells[4] if len(cells) > 4 else "",
                }
            )
        return rows

    @staticmethod
    def _join_natural_list(items: list[str]) -> str:
        clean = [TaskGraphActionMapper._single_line(item) for item in items if TaskGraphActionMapper._single_line(item)]
        if not clean:
            return ""
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]} and {clean[1]}"
        return f"{', '.join(clean[:-1])}, and {clean[-1]}"

    @staticmethod
    def _markdown_table_cell(value: Any) -> str:
        return str(value or "").replace("|", "\\|").replace("\n", " ").strip() or "-"

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
            if value.startswith("|"):
                continue
            if value.startswith("#") and len(value.split()) <= 6:
                continue
            if lowered.startswith("--- (skill:"):
                continue
            if re.match(r"^\[[a-z0-9_-]+\]\s+[a-z ]+$", lowered):
                continue
            if lowered.startswith("tool used:"):
                continue
            if lowered.startswith("objective:") and len(value) < 80:
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

    def _live_high_value_document_generation(
        self,
        *,
        surface_kind: str,
        prompt: str,
        source_text: str,
        local_output: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        overrides = self._resolve_live_model_overrides(context)
        if not overrides or not self._is_high_value_surface(surface_kind):
            return None
        try:
            from app.harness.live_agent import LiveAgentOrchestrator

            evidence = self._research_evidence_payload(source_text)
            plan = self._high_value_document_plan(surface_kind=surface_kind, prompt=prompt, evidence=evidence)
            strategy = self._preferred_live_document_strategy(surface_kind=surface_kind, prompt=prompt)
            mode = "deep" if strategy in {"research_analyst", "systems_architect"} else "balanced"
            discovery = [{"name": "evidence_digest", "score": 0.9}] if evidence.get("records") or evidence.get("citations") else []
            result = LiveAgentOrchestrator().enhance(
                query=prompt,
                mode=mode,
                base_answer=local_output,
                plan=plan,
                steps=[],
                discovery=discovery,
                evidence=evidence,
                max_calls=6,
                live_model_overrides=overrides,
                strategy=strategy,
                surface_guidance=self._surface_guidance(surface_kind=surface_kind),
            )
            refined = self._ground_live_document_output(
                surface_kind=surface_kind,
                prompt=prompt,
                source_text=source_text,
                local_output=local_output,
                live_output=str(result.enhanced_answer or ""),
            )
            if not result.success or len(refined.split()) < max(24, len(str(local_output or "").split()) // 3):
                return None
            rationale = [
                f"live revision loop used strategy {strategy}",
                "final deliverable refined through analysis, critique, and revision",
            ]
            return {
                "source": "live_model",
                "model": result.model,
                "content_text": refined,
                "rationale": rationale,
            }
        except Exception:
            return None

    @staticmethod
    def _is_high_value_surface(surface_kind: str) -> bool:
        high_value = {
            "artifact_synthesis",
            "research_brief",
            "benchmark_ablation",
            "ops_runbook",
            "custom:memo",
            "custom:brief",
            "custom:executive_memo",
            "custom:decision_memo",
            "custom:launch_memo",
            "custom:one_pager",
        }
        return str(surface_kind or "").strip() in high_value

    @staticmethod
    def _preferred_live_document_strategy(*, surface_kind: str, prompt: str) -> str:
        lowered = str(prompt or "").lower()
        if surface_kind in {"custom:memo", "custom:brief", "research_brief"} or any(
            marker in lowered for marker in ["research", "benchmark", "investigate", "report", "memo", "compare"]
        ):
            return "research_analyst"
        if any(marker in lowered for marker in ["patch", "fix", "repo", "repository", "test", "bug", "router", "workspace"]):
            return "systems_architect"
        if any(marker in lowered for marker in ["architecture", "system", "framework", "runtime", "migration"]):
            return "systems_architect"
        if any(marker in lowered for marker in ["risk", "governance", "policy", "security", "compliance"]):
            return "risk_sentinel"
        return "balanced_orchestrator"

    @staticmethod
    def _high_value_document_plan(*, surface_kind: str, prompt: str, evidence: dict[str, Any]) -> list[str]:
        records = evidence.get("records", []) if isinstance(evidence.get("records", []), list) else []
        citations = evidence.get("citations", []) if isinstance(evidence.get("citations", []), list) else []
        plan = [
            f"Establish the main thesis for {prompt}",
            "Extract the strongest supporting evidence and turn it into explicit claims",
            "Revise the deliverable so the final answer beats a direct-response baseline",
        ]
        if records:
            plan.insert(1, f"Ground the answer in {len(records)} captured evidence records")
        if citations:
            plan.append("Expose the strongest source references inside the final deliverable")
        if surface_kind in {"artifact_synthesis", "research_brief", "custom:memo", "custom:brief", "custom:executive_memo", "custom:decision_memo"}:
            plan.insert(1, "Write the primary deliverable itself instead of summarizing framework internals")
            plan.append("Make the result publishable in one read, with dense paragraphs, concrete recommendations, and no metric theater")
        if surface_kind.startswith("custom:"):
            plan.append("Respect the requested document shape while keeping paragraphs dense and reviewable")
        return plan[:5]

    @staticmethod
    def _surface_guidance(*, surface_kind: str) -> str:
        mapping = {
            "custom:memo": (
                "Return markdown for a serious research memo. Keep the result decision-relevant. "
                "Use clear memo sections such as Summary, Context, Evidence, Recommendation, Implications, and Next Step. "
                "Do not invent roadmap phases or benchmark percentages unless the evidence explicitly contains them."
            ),
            "custom:brief": (
                "Return a compact research brief with high-density paragraphs and a short findings list. "
                "Do not pad the answer with generic framework theater."
            ),
            "custom:decision_memo": (
                "Return a decision memo with a hard recommendation, explicit tradeoffs, and the next action. "
                "Avoid speculative quantitative claims unless the evidence supports them."
            ),
            "artifact_synthesis": (
                "Write the main deliverable the user actually asked for. "
                "Lead with the answer, not with the framework. "
                "Use explicit sections such as Executive Summary, Key Findings, Evidence, Recommendations, and Next Steps when they fit. "
                "Do not waste space on internal runtime names, stage labels, scorecards, or bundle narration unless the user explicitly asked for them."
            ),
            "research_brief": (
                "Write a serious research brief with concrete findings, grounded claims, and a recommendation section. "
                "Avoid generic observations that could have been produced without the evidence."
            ),
        }
        return mapping.get(str(surface_kind or "").strip(), "Prefer grounded claims over rhetorical flourish.")

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
        if self._prefer_local_workspace_action(action_kind):
            return None
        if content_type != "application/json":
            revised = self._live_high_value_document_generation(
                surface_kind=action_kind,
                prompt=prompt,
                source_text=source_text,
                local_output=local_content,
                context=context,
            )
            if revised:
                return {
                    "source": str(revised.get("source", "live_model")),
                    "model": str(revised.get("model", "")),
                    "relative_path": local_relative_path,
                    "content_text": str(revised.get("content_text", "")),
                    "content_type": content_type,
                    "rationale": list(revised.get("rationale", [])) if isinstance(revised.get("rationale", []), list) else [],
                }
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
                "quality_brief": self._workspace_action_quality_brief(action_kind),
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
                        "Do not change artifact type. "
                        "Prefer a result that a reviewer could directly use, not a scaffold that merely names sections. "
                        "Avoid mentioning the runtime, framework internals, or generic filler unless the task explicitly asks for them."
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

    @staticmethod
    def _prefer_local_workspace_action(action_kind: str) -> bool:
        local_only = {
            "custom:source_matrix",
            "custom:research_outline",
            "custom:direct_answer_baseline",
            "completion_packet",
            "delivery_bundle",
        }
        return str(action_kind or "").strip() in local_only

    @classmethod
    def _ground_live_document_output(
        cls,
        *,
        surface_kind: str,
        prompt: str,
        source_text: str,
        local_output: str,
        live_output: str,
    ) -> str:
        del prompt
        surface = str(surface_kind or "").strip()
        candidate = str(live_output or "").strip()
        fallback = str(local_output or "").strip()
        if not candidate:
            return fallback
        evidence = cls._research_evidence_payload(source_text)
        supported_numeric = cls._supported_numeric_claims(evidence)
        evidence_text = cls._grounding_evidence_text(evidence)
        normalized_evidence = evidence_text.lower()
        allowed_sources = cls._allowed_source_markers(evidence)
        if cls._document_contradicts_available_evidence(candidate, evidence=evidence):
            candidate = fallback
        removed_claims = 0
        cleaned_lines: list[str] = []
        for raw_line in re.split(r"\r?\n", candidate):
            line = raw_line.rstrip()
            if cls._line_contains_unsupported_quant_claim(line, supported_numeric):
                removed_claims += 1
                continue
            if cls._line_contradicts_available_evidence(line, evidence=evidence):
                removed_claims += 1
                continue
            if cls._line_mentions_unsupported_source(line, allowed_sources):
                removed_claims += 1
                continue
            if cls._line_mentions_unseen_benchmark(line, normalized_evidence):
                removed_claims += 1
                continue
            cleaned_lines.append(line)
        grounded = cls._collapse_markdown_spacing(cleaned_lines).strip()
        if removed_claims and surface in {"custom:memo", "custom:brief", "custom:executive_memo", "custom:decision_memo", "custom:launch_memo", "research_brief"}:
            grounded = cls._append_evidence_limits_note(grounded or fallback, evidence=evidence)
        if len(grounded.split()) < cls._minimum_grounded_word_count(surface_kind=surface, fallback=fallback):
            grounded = cls._append_evidence_limits_note(fallback, evidence=evidence) if removed_claims else fallback
        if surface in {"custom:memo", "custom:executive_memo", "custom:decision_memo", "custom:launch_memo", "research_brief", "artifact_synthesis"}:
            grounded = cls._append_grounded_sources_section(grounded, evidence=evidence)
        return grounded.strip() or fallback

    @classmethod
    def _document_contradicts_available_evidence(cls, text: str, *, evidence: dict[str, Any]) -> bool:
        rows = [cls._single_line(line) for line in re.split(r"\r?\n", str(text or "")) if cls._single_line(line)]
        return any(cls._line_contradicts_available_evidence(line, evidence=evidence) for line in rows[:40])

    @staticmethod
    def _line_contradicts_available_evidence(line: str, *, evidence: dict[str, Any]) -> bool:
        value = str(line or "").lower()
        if not value:
            return False
        record_count = len(evidence.get("records", [])) if isinstance(evidence.get("records", []), list) else 0
        citation_count = len(evidence.get("citations", [])) if isinstance(evidence.get("citations", []), list) else 0
        has_sources = record_count > 0 or citation_count > 0
        if not has_sources:
            return False
        contradiction_markers = [
            "zero records",
            "0 records",
            "no records",
            "no citations",
            "zero citations",
            "0 citations",
            "no source materials",
            "no source material",
            "no sources were available",
            "absence of evidence",
            "evidence base contains zero",
            "our evidence base contains zero",
            "no empirical evidence",
        ]
        return any(marker in value for marker in contradiction_markers)

    @staticmethod
    def _minimum_grounded_word_count(*, surface_kind: str, fallback: str) -> int:
        surface = str(surface_kind or "").strip()
        fallback_words = len(str(fallback or "").split())
        if surface in {"custom:memo", "custom:executive_memo", "custom:decision_memo", "custom:launch_memo"}:
            return max(80, int(fallback_words * 0.55))
        if surface in {"custom:brief", "research_brief"}:
            return 36
        if surface == "artifact_synthesis":
            return max(48, int(fallback_words * 0.4))
        return max(24, int(fallback_words * 0.4))

    @classmethod
    def _grounding_evidence_text(cls, evidence: dict[str, Any]) -> str:
        parts: list[str] = []
        for record in evidence.get("records", []) if isinstance(evidence.get("records", []), list) else []:
            if not isinstance(record, dict):
                continue
            parts.extend(
                [
                    cls._single_line(record.get("title", "")),
                    cls._single_line(record.get("summary", "")),
                    cls._single_line(record.get("content", "")),
                    cls._single_line(record.get("url", "")),
                ]
            )
        for key in ["citations", "baseline_lines", "source_lines", "benchmark_focus"]:
            values = evidence.get(key, [])
            if isinstance(values, list):
                parts.extend(cls._single_line(item) for item in values[:12])
        for row in evidence.get("source_rows", []) if isinstance(evidence.get("source_rows", []), list) else []:
            if not isinstance(row, dict):
                continue
            parts.extend(
                [
                    cls._single_line(row.get("source", "")),
                    cls._single_line(row.get("claim", "")),
                    cls._single_line(row.get("usefulness", "")),
                    cls._single_line(row.get("gap", "")),
                ]
            )
        return "\n".join(part for part in parts if part)

    @classmethod
    def _supported_numeric_claims(cls, evidence: dict[str, Any]) -> set[str]:
        return cls._extract_numeric_claim_tokens(cls._grounding_evidence_text(evidence))

    @staticmethod
    def _extract_numeric_claim_tokens(text: str) -> set[str]:
        payload = str(text or "")
        patterns = [
            r"\b\d+(?:\.\d+)?\s*(?:-|to)\s*\d+(?:\.\d+)?\s*(?:%|percent|x|times|ms|milliseconds?|s|sec|seconds?|m|minutes?|h|hours?|days?|weeks?|months?|years?)\b",
            r"\b\d+(?:\.\d+)?\s*(?:%|percent|x|times|ms|milliseconds?|s|sec|seconds?|m|minutes?|h|hours?|days?|weeks?|months?|years?)\b",
        ]
        tokens: set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, payload, flags=re.IGNORECASE):
                normalized = re.sub(r"\s+", "", str(match).lower())
                if normalized:
                    tokens.add(normalized)
        return tokens

    @classmethod
    def _line_contains_unsupported_quant_claim(cls, line: str, supported_numeric: set[str]) -> bool:
        value = cls._single_line(line)
        if not value:
            return False
        candidates = cls._extract_numeric_claim_tokens(value)
        if not candidates:
            return False
        if not supported_numeric:
            return True
        return any(token not in supported_numeric for token in candidates)

    @classmethod
    def _allowed_source_markers(cls, evidence: dict[str, Any]) -> set[str]:
        markers: set[str] = set()
        for record in evidence.get("records", []) if isinstance(evidence.get("records", []), list) else []:
            if not isinstance(record, dict):
                continue
            for key in ["title", "url"]:
                marker = cls._normalize_marker(record.get(key, ""))
                if marker:
                    markers.add(marker)
        for citation in evidence.get("citations", []) if isinstance(evidence.get("citations", []), list) else []:
            marker = cls._normalize_marker(citation)
            if marker:
                markers.add(marker)
        for row in evidence.get("source_rows", []) if isinstance(evidence.get("source_rows", []), list) else []:
            if not isinstance(row, dict):
                continue
            marker = cls._normalize_marker(row.get("source", ""))
            if marker:
                markers.add(marker)
        return markers

    @staticmethod
    def _normalize_marker(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    @classmethod
    def _line_mentions_unsupported_source(cls, line: str, allowed_sources: set[str]) -> bool:
        value = cls._single_line(line)
        lowered = value.lower()
        if not value or not allowed_sources:
            return False
        if "sources" in lowered and len(value.split()) <= 4:
            return False
        source_like = (
            ("http://" in lowered or "https://" in lowered)
            or " et al." in lowered
            or ("(" in value and ")" in value and any(ch.isdigit() for ch in value))
            or '"' in value
        )
        if not source_like:
            return False
        normalized = cls._normalize_marker(value)
        return not any(marker and marker in normalized for marker in allowed_sources)

    @staticmethod
    def _line_mentions_unseen_benchmark(line: str, evidence_text: str) -> bool:
        known = ["swe-bench", "webarena", "gaia", "tau-bench", "mcp", "toolformer", "react"]
        lowered = str(line or "").lower()
        for marker in known:
            if marker in lowered and marker not in evidence_text:
                return True
        return False

    @staticmethod
    def _collapse_markdown_spacing(lines: list[str]) -> str:
        collapsed: list[str] = []
        blank_pending = False
        for item in lines:
            value = item.rstrip()
            if not value:
                if collapsed:
                    blank_pending = True
                continue
            if blank_pending and collapsed:
                collapsed.append("")
            collapsed.append(value)
            blank_pending = False
        return "\n".join(collapsed)

    @classmethod
    def _append_evidence_limits_note(cls, text: str, *, evidence: dict[str, Any]) -> str:
        body = str(text or "").strip()
        if not body:
            return body
        lowered = body.lower()
        if "## evidence limits" in lowered or "### evidence limits" in lowered:
            return body
        support_note = (
            "The evidence captured in this run is primarily qualitative. Unsupported quantitative claims and ungrounded source references were removed after synthesis."
        )
        if cls._supported_numeric_claims(evidence):
            support_note = (
                "Quantitative claims were kept only where the captured evidence contained matching numeric support. Any unsupported metrics were removed after synthesis."
            )
        return f"{body}\n\n## Evidence Limits\n\n{support_note}\n"

    @classmethod
    def _append_grounded_sources_section(cls, text: str, *, evidence: dict[str, Any]) -> str:
        body = str(text or "").strip()
        if not body:
            return body
        references: list[str] = []
        seen: set[str] = set()
        for record in evidence.get("records", []) if isinstance(evidence.get("records", []), list) else []:
            if not isinstance(record, dict):
                continue
            title = cls._single_line(record.get("title", ""))
            summary = cls._single_line(record.get("summary", ""))
            url = cls._single_line(record.get("url", ""))
            if not title:
                continue
            row = title
            if summary:
                row += f" - {summary}"
            if url:
                row += f" ({url})"
            if row not in seen:
                seen.add(row)
                references.append(row)
            if len(references) >= 4:
                break
        if not references:
            for row in evidence.get("source_rows", []) if isinstance(evidence.get("source_rows", []), list) else []:
                if not isinstance(row, dict):
                    continue
                source = cls._single_line(row.get("source", ""))
                claim = cls._single_line(row.get("claim", ""))
                if not source:
                    continue
                ref = source + (f" - {claim}" if claim else "")
                if ref and ref not in seen:
                    seen.add(ref)
                    references.append(ref)
                if len(references) >= 4:
                    break
        if not references:
            for item in evidence.get("benchmark_focus", []) if isinstance(evidence.get("benchmark_focus", []), list) else []:
                row = cls._single_line(item)
                if row and row not in seen:
                    seen.add(row)
                    references.append(row)
                if len(references) >= 4:
                    break
        if not references:
            for citation in evidence.get("citations", []) if isinstance(evidence.get("citations", []), list) else []:
                row = cls._single_line(citation)
                if row and row not in seen:
                    seen.add(row)
                    references.append(row)
                if len(references) >= 4:
                    break
        if not references:
            return body
        lowered = body.lower()
        if "## sources" in lowered or "### sources" in lowered:
            existing_markers: list[str] = []
            for record in evidence.get("records", []) if isinstance(evidence.get("records", []), list) else []:
                if isinstance(record, dict):
                    title_marker = cls._normalize_marker(record.get("title", ""))
                    if title_marker:
                        existing_markers.append(title_marker)
            for row in evidence.get("source_rows", []) if isinstance(evidence.get("source_rows", []), list) else []:
                if isinstance(row, dict):
                    source_marker = cls._normalize_marker(row.get("source", ""))
                    if source_marker:
                        existing_markers.append(source_marker)
            normalized_body = cls._normalize_marker(body)
            if sum(1 for marker in existing_markers if marker and marker in normalized_body) >= 2:
                return body
        lines = [body, "", "## Sources", ""]
        for row in references:
            lines.append(f"- {row}")
        lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _live_skill_generation(
        self,
        *,
        node_id: str,
        skill_name: str,
        prompt: str,
        source_text: str,
        local_output: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        overrides = self._resolve_live_model_overrides(context)
        if not overrides:
            return None
        if not self._skip_heavy_live_document_generation(surface_kind=skill_name, node_id=node_id, context=context):
            revised = self._live_high_value_document_generation(
                surface_kind=skill_name,
                prompt=prompt,
                source_text=source_text,
                local_output=local_output,
                context=context,
            )
            if revised:
                return revised
        try:
            from app.harness.live_agent import CallBudget, LiveModelConfig, LiveModelGateway

            config = LiveModelConfig.resolve(overrides)
            if not config:
                return None
            gateway = LiveModelGateway(config)
            payload = {
                "skill_name": skill_name,
                "prompt": prompt,
                "source_text": source_text[:7000],
                "local_output": str(local_output)[:5000],
                "quality_brief": self._skill_quality_brief(skill_name),
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are executing one skill inside a general-purpose agent runtime. "
                        "Return strict JSON with keys content_text and rationale. "
                        "Produce the final skill output only, not commentary about the runtime. "
                        "Make the result materially stronger than the local_output while staying grounded in source_text. "
                        "If the skill is research_brief, artifact_synthesis, benchmark_ablation, or ops_runbook, prefer concrete, structured output over generic prose. "
                        "Do not echo raw planning metadata or node names. "
                        "If source_text contains a direct baseline, outperform it by adding missing evidence, sharper synthesis, and a more actionable final recommendation."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ]
            text, meta = gateway.chat(
                messages=messages,
                budget=CallBudget(max_calls=1),
                temperature=0.15,
                require_json=True,
            )
            parsed = self._parse_json_dict(text)
            content_text = str(parsed.get("content_text", "")).strip()
            if not content_text:
                return None
            return {
                "source": "live_model",
                "model": str(meta.get("model", "")),
                "content_text": content_text,
                "rationale": [str(item).strip() for item in parsed.get("rationale", []) if str(item).strip()]
                if isinstance(parsed.get("rationale", []), list)
                else [],
            }
        except Exception:
            return None

    @staticmethod
    def _workspace_action_quality_brief(action_kind: str) -> str:
        mapping = {
            "benchmark_manifest": "Return a real benchmark plan with measurable tracks, datasets or suites, success metrics, and output files.",
            "benchmark_run_config": "Return an executable-style config with suites, evaluation steps, artifacts, and run controls.",
            "custom:memo": "Write a substantive memo with paragraph-grade analysis, concrete evidence, and a clear recommendation.",
            "custom:brief": "Write a compact but high-density brief with findings, evidence, and next actions.",
            "custom:decision_memo": "Write a decision memo that makes a hard recommendation, names tradeoffs, and leaves no ambiguity about the next step.",
            "custom:executive_memo": "Write an executive memo that is concise but concrete, with evidence-backed recommendations and explicit risks.",
            "custom:source_matrix": "Return a research source matrix with concrete rows, specific sources, what each source proves, and unresolved gaps.",
            "custom:research_outline": "Return a serious report outline with thesis, sections, proof obligations, and where evidence should enter.",
            "custom:direct_answer_baseline": "Return the kind of direct answer a strong single model would give without the framework, then state what that answer still misses.",
            "custom:upgrade_roadmap": "Return a concrete roadmap with milestones, owners, dependencies, measurable exit criteria, and sequencing logic.",
            "delivery_bundle": "Summarize the final shipment surface so a reviewer can open the main deliverable first and then inspect supporting artifacts.",
            "completion_packet": "Summarize task closure in reviewer language, highlighting what was delivered, what evidence supports it, and what still blocks closure.",
        }
        return mapping.get(action_kind, "Return a concrete, reviewable artifact with high information density and minimal filler.")

    @staticmethod
    def _skill_quality_brief(skill_name: str) -> str:
        mapping = {
            "artifact_synthesis": "Write the final deliverable, not a meta-summary. Lead with the main thesis, then evidence, then concrete next actions.",
            "research_brief": "Produce a substantive research brief with question, findings, evidence, disagreements, and recommended follow-up.",
            "benchmark_ablation": "Produce a real benchmark and ablation plan with variants, metrics, failure modes, and decision thresholds.",
            "ops_runbook": "Produce a runbook operators could execute under pressure, with triggers, steps, rollback, and escalation points.",
            "codebase_triage": "Write an engineering handoff with root cause, touched files, patch intent, test plan, and validation notes grounded in workspace evidence.",
        }
        return mapping.get(skill_name, "Return a high-density final output that is directly useful to a reviewer.")

    @staticmethod
    def _skip_heavy_live_document_generation(*, surface_kind: str, node_id: str, context: dict[str, Any]) -> bool:
        normalized_node = str(node_id or "").strip().lower()
        if normalized_node in {"analysis", "validation"}:
            return True
        if str(surface_kind or "").strip() == "artifact_synthesis":
            node_results = context.get("node_results", {}) if isinstance(context.get("node_results", {}), dict) else {}
            for key in ["action_custom-memo", "action_custom-brief", "action_custom-executive_memo", "action_custom-decision_memo", "action_custom-launch_memo"]:
                if key in node_results:
                    return True
        return False

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

    @classmethod
    def _collect_source_text(cls, *, metrics: dict[str, Any], context: dict[str, Any]) -> str:
        source_ids = metrics.get("source_node_ids", [])
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        parts: list[str] = []
        for source_id in source_ids if isinstance(source_ids, list) else []:
            block = cls._source_prompt_block(str(source_id), context)
            if block:
                parts.append(block)
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

    @classmethod
    def _source_prompt_block(cls, source_node_id: str, context: dict[str, Any]) -> str:
        node_results = context.get("node_results", {}) if isinstance(context.get("node_results", {}), dict) else {}
        source = node_results.get(source_node_id, {}) if isinstance(node_results, dict) else {}
        if not isinstance(source, dict) or not source:
            return ""
        result = source.get("result", {}) if isinstance(source.get("result", {}), dict) else {}
        artifact = source.get("artifact", {}) if isinstance(source.get("artifact", {}), dict) else {}
        title = str(result.get("title", source.get("title", source_node_id))).strip() or source_node_id
        lines = [f"[{source_node_id}] {title}"]

        output = result.get("output")
        output_lines: list[str] = []
        if isinstance(output, str):
            output_lines.extend(cls._meaningful_lines(output, limit=6))
        elif isinstance(output, dict):
            output_lines.extend(cls._dict_signal_lines(output, limit=6))
        elif output not in (None, "", {}):
            output_lines.append(cls._single_line(output))
        if output_lines:
            lines.append("Output Highlights:")
            lines.extend(f"- {item}" for item in output_lines[:6])
        structured_output = cls._compact_structured_output(output if isinstance(output, dict) else result)
        if structured_output:
            lines.append("Structured Output JSON:")
            lines.append(json.dumps(structured_output, ensure_ascii=False))

        for key in ["manifest", "config", "packet", "bundle", "spec"]:
            value = result.get(key)
            if isinstance(value, dict):
                signal_lines = cls._dict_signal_lines(value, limit=6)
                if signal_lines:
                    lines.append(f"{key.title()} Signals:")
                    lines.extend(f"- {item}" for item in signal_lines[:6])

        preview = cls._artifact_preview_text(str(artifact.get("path", "")))
        if preview:
            lines.append("Artifact Preview:")
            lines.append(preview)

        return "\n".join(lines).strip()

    @classmethod
    def _compact_structured_output(cls, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        compact: dict[str, Any] = {}
        for key in ["summary", "objective", "question", "goal", "decision", "status", "headline", "thesis"]:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                compact[key] = cls._single_line(value)
        citations = payload.get("citations", [])
        if isinstance(citations, list) and citations:
            compact["citations"] = [cls._single_line(item) for item in citations[:6] if cls._single_line(item)]
        for key in ["records", "resources"]:
            rows = payload.get(key, [])
            compact_rows: list[dict[str, Any]] = []
            if isinstance(rows, list):
                for row in rows[:4]:
                    if not isinstance(row, dict):
                        continue
                    compact_row: dict[str, Any] = {}
                    for field in ["title", "summary", "url", "source_id", "trust_score", "freshness_hint"]:
                        value = row.get(field)
                        if value not in (None, "", []):
                            compact_row[field] = value
                    if compact_row:
                        compact_rows.append(compact_row)
            if compact_rows:
                compact[key] = compact_rows
        return compact

    @classmethod
    def _dict_signal_lines(cls, payload: dict[str, Any], *, limit: int = 6) -> list[str]:
        rows: list[str] = []
        for key in ["summary", "objective", "question", "goal", "decision", "status", "headline", "thesis"]:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                rows.append(f"{key}: {cls._single_line(value)}")
        records = payload.get("records", [])
        if isinstance(records, list):
            for record in records[:4]:
                if not isinstance(record, dict):
                    continue
                title = cls._single_line(record.get("title", ""))
                summary = cls._single_line(record.get("summary", ""))
                if title:
                    rows.append(title + (f": {summary}" if summary else ""))
        resources = payload.get("resources", [])
        if isinstance(resources, list):
            for resource in resources[:4]:
                if not isinstance(resource, dict):
                    continue
                title = cls._single_line(resource.get("title", ""))
                summary = cls._single_line(resource.get("summary", ""))
                if title:
                    rows.append(title + (f": {summary}" if summary else ""))
        for key in ["highlights", "citations", "next_steps", "missing_artifacts", "deliverables"]:
            value = payload.get(key)
            if isinstance(value, list):
                for item in value[:3]:
                    text = cls._single_line(item)
                    if text:
                        rows.append(f"{key[:-1] if key.endswith('s') else key}: {text}")
        if isinstance(payload.get("bundle_summary", {}), dict):
            summary = payload.get("bundle_summary", {})
            rows.append(
                "bundle_summary: "
                f"artifacts={int(summary.get('artifact_count', 0))}, "
                f"families={int(summary.get('family_count', 0))}, "
                f"validation={summary.get('validation_status', 'unknown')}"
            )
        if isinstance(payload.get("primary_deliverable", {}), dict):
            primary = payload.get("primary_deliverable", {})
            excerpt = cls._single_line(primary.get("excerpt", ""))
            path = cls._single_line(primary.get("path", ""))
            if excerpt or path:
                rows.append(f"primary_deliverable: {excerpt or path}")
        if not rows:
            rows.extend(cls._meaningful_lines(json.dumps(payload, ensure_ascii=False, indent=2), limit=limit))
        deduped: list[str] = []
        seen: set[str] = set()
        for item in rows:
            clean = cls._single_line(item).strip(" -")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)
            if len(deduped) >= limit:
                break
        return deduped

    @classmethod
    def _artifact_preview_text(cls, path: str, *, limit: int = 1200) -> str:
        artifact_path = Path(str(path or "").strip())
        if not artifact_path.exists() or artifact_path.suffix.lower() not in {".md", ".txt", ".diff", ".json", ".py"}:
            return ""
        try:
            content = artifact_path.read_text(encoding="utf-8")
        except Exception:
            return ""
        if artifact_path.suffix.lower() == ".json":
            try:
                payload = json.loads(content)
                lines = cls._dict_signal_lines(payload if isinstance(payload, dict) else {"content": payload}, limit=6)
                if lines:
                    return "\n".join(f"- {item}" for item in lines)
            except Exception:
                pass
        meaningful = cls._meaningful_lines(content, limit=8)
        if meaningful:
            return "\n".join(f"- {item}" for item in meaningful)[:limit]
        return cls._single_line(content)[:limit]

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
        task_spec_dict = task_spec.to_dict()

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
        delivered_kinds = {
            kind
            for item in delivered_artifacts
            for kind in [self._artifact_kind_from_path(str(item.get("path", "")))]
            if kind
        }

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
            validation_status = "review_ready"
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
        if validation_status in {"review_ready", "passed"}:
            packet_state_gap["missing_validation"] = False
        packet_state_gap["missing_artifacts"] = [
            str(item)
            for item in packet_state_gap.get("missing_artifacts", [])
            if str(item) not in delivered_kinds
        ]

        primary_deliverable = self._primary_deliverable_from_results(node_results=node_results, delivered_artifacts=delivered_artifacts)
        baseline_comparison = self._baseline_comparison(
            node_results=node_results,
            primary_deliverable=primary_deliverable,
            evidence_summary=evidence_summary,
            delivered_artifacts=delivered_artifacts,
        )

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
            "task_spec": task_spec_dict,
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
            "primary_deliverable": primary_deliverable,
            "baseline_comparison": baseline_comparison,
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
        primary_deliverable = completion_packet.get("primary_deliverable", {}) if isinstance(completion_packet.get("primary_deliverable", {}), dict) else {}
        baseline_comparison = completion_packet.get("baseline_comparison", {}) if isinstance(completion_packet.get("baseline_comparison", {}), dict) else {}

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
            f"Open the primary deliverable first: {str(primary_deliverable.get('path', self._node_artifact_path(node_results, 'report')))}",
            f"Review task closure against the request: {prompt}",
            f"Validation status is {str(validation.get('status', 'unknown')).replace('_', ' ')}.",
            f"Evidence records available: {int(evidence.get('record_count', 0))}.",
            f"Risk items captured: {int(risk.get('count', 0))}.",
        ]

        handoff_order = []
        for path in [
            str(primary_deliverable.get("path", "")),
            self._node_artifact_path(node_results, "source_matrix"),
            self._node_artifact_path(node_results, "report_outline"),
            self._node_artifact_path(node_results, "direct_baseline"),
            self._node_artifact_path(node_results, "completion_packet"),
        ]:
            if path and path not in handoff_order:
                handoff_order.append(path)
        for item in manifest[:12]:
            path = str(item.get("path", ""))
            if path and path not in handoff_order:
                handoff_order.append(path)

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
                "primary_path": str(primary_deliverable.get("path", "")),
            },
            "deliverable_index": deliverable_index,
            "artifact_manifest": manifest,
            "completion_packet_ref": self._node_artifact_path(node_results, "completion_packet"),
            "report_ref": self._node_artifact_path(node_results, "report"),
            "primary_deliverable": primary_deliverable,
            "baseline_comparison": baseline_comparison,
            "reviewer_checklist": reviewer_checklist,
            "handoff_order": handoff_order,
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

    @staticmethod
    def _artifact_kind_from_path(path: str) -> str:
        lowered = str(path or "").replace("\\", "/").lower()
        mapping = [
            ("deliverable_report", ["/report", "report.md"]),
            ("completion_packet", ["packets/completion-packet"]),
            ("delivery_bundle", ["bundles/delivery-bundle"]),
            ("benchmark_manifest", ["benchmarks/manifest"]),
            ("benchmark_run_config", ["benchmarks/run-config"]),
            ("data_analysis_spec", ["analysis/data-analysis-spec"]),
            ("dataset_pull_spec", ["datasets/pull-spec"]),
            ("dataset_loader_template", ["datasets/loader_template"]),
            ("patch_plan", ["patch-scaffold"]),
            ("patch_draft", ["patch-draft"]),
            ("webpage_blueprint", ["web/"]),
            ("slide_deck_plan", ["slides/"]),
            ("chart_pack_spec", ["charts/"]),
            ("podcast_episode_plan", ["podcast/"]),
            ("video_storyboard", ["video/"]),
            ("image_prompt_pack", ["images/"]),
        ]
        for kind, patterns in mapping:
            if any(pattern in lowered for pattern in patterns):
                return kind
        return ""

    @classmethod
    def _read_artifact_excerpt(cls, path: str, *, limit: int = 1800) -> str:
        artifact_path = Path(str(path or "").strip())
        if not artifact_path.exists() or artifact_path.suffix.lower() not in {".md", ".txt", ".diff", ".json"}:
            return ""
        try:
            content = artifact_path.read_text(encoding="utf-8")
        except Exception:
            return ""
        if artifact_path.suffix.lower() == ".json":
            try:
                payload = json.loads(content)
                if isinstance(payload, dict):
                    return "\n".join(cls._dict_signal_lines(payload, limit=8))[:limit]
            except Exception:
                pass
        return content[:limit].strip()

    @classmethod
    def _primary_deliverable_from_results(
        cls,
        *,
        node_results: dict[str, Any],
        delivered_artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        preferred_paths: list[str] = []
        for node_id in ["action_custom-memo", "action_custom-executive_memo", "action_custom-decision_memo", "action_custom-launch_memo", "report"]:
            path = cls._node_artifact_path(node_results, node_id)
            if path:
                preferred_paths.append(path)
        for item in delivered_artifacts:
            path = str(item.get("path", ""))
            if not path:
                continue
            normalized = path.replace("\\", "/").lower()
            artifact_kind = cls._artifact_kind_from_path(path)
            if artifact_kind.startswith("custom:") and "memo" in artifact_kind:
                preferred_paths.append(path)
                continue
            if "/briefs/" in normalized or normalized.endswith("research_memo.md") or ("/research/" in normalized and "memo" in Path(path).name.lower()):
                preferred_paths.append(path)
        if not preferred_paths:
            for item in delivered_artifacts:
                path = str(item.get("path", ""))
                if cls._artifact_family_from_path(path) == "report":
                    preferred_paths.append(path)
                    break
        primary_path = next((path for path in preferred_paths if path), "")
        excerpt = cls._read_artifact_excerpt(primary_path, limit=2400) if primary_path else ""
        return {
            "path": primary_path,
            "title": Path(primary_path).name if primary_path else "",
            "excerpt": excerpt[:2000],
        }

    @classmethod
    def _baseline_comparison(
        cls,
        *,
        node_results: dict[str, Any],
        primary_deliverable: dict[str, Any],
        evidence_summary: dict[str, Any],
        delivered_artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        baseline_path = cls._node_artifact_path(node_results, "direct_baseline")
        baseline_excerpt = cls._read_artifact_excerpt(baseline_path, limit=1200) if baseline_path else ""
        additions: list[str] = []
        if int(evidence_summary.get("record_count", 0)) > 0:
            additions.append(f"Adds {int(evidence_summary.get('record_count', 0))} evidence records and {int(evidence_summary.get('citation_count', 0))} citations beyond a plain direct answer.")
        delivered_kinds = {
            cls._artifact_kind_from_path(str(item.get("path", "")))
            for item in delivered_artifacts
            if cls._artifact_kind_from_path(str(item.get("path", "")))
        }
        if {"benchmark_manifest", "benchmark_run_config"} & delivered_kinds:
            additions.append("Turns the answer into a runnable evaluation package with benchmark manifest and run configuration.")
        if cls._node_artifact_path(node_results, "source_matrix"):
            additions.append("Adds a source matrix so claims can be traced back to concrete evidence surfaces.")
        if primary_deliverable.get("path"):
            additions.append("Promotes one primary deliverable instead of leaving the user with disconnected intermediate files.")
        return {
            "baseline_path": baseline_path,
            "baseline_excerpt": baseline_excerpt[:1000],
            "harness_additions": additions[:4],
        }

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
