"""Task-graph node to runtime-action mapper."""

from __future__ import annotations

import fnmatch
import json
import re
import time
from pathlib import Path
from typing import Any

from app.agents.sandbox import ThreadSandbox
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
        body: dict[str, Any]
        relative_path: str
        content: str
        content_type = "text/markdown"

        if action_kind == "patch_scaffold":
            relative_path = str(metrics.get("relative_path", "plans/patch-scaffold.md"))
            content = self._build_patch_scaffold(prompt=prompt, source_text=source_text, context=context, workspace_summary=workspace_summary)
            body = {
                "node_id": node.get("node_id", ""),
                "action_kind": action_kind,
                "path": relative_path,
                "output": content,
                "graph_id": graph.get("graph_id", ""),
            }
        elif action_kind == "patch_draft":
            relative_path = str(metrics.get("relative_path", "patches/patch-draft.diff"))
            content = self._build_patch_draft(prompt=prompt, source_text=source_text, workspace_summary=workspace_summary)
            body = {
                "node_id": node.get("node_id", ""),
                "action_kind": action_kind,
                "path": relative_path,
                "output": content,
                "graph_id": graph.get("graph_id", ""),
            }
        elif action_kind == "benchmark_run_config":
            relative_path = str(metrics.get("relative_path", "benchmarks/run-config.json"))
            payload = self._build_benchmark_run_config(prompt=prompt, source_text=source_text, workspace_summary=workspace_summary)
            content = json.dumps(payload, indent=2, default=str)
            content_type = "application/json"
            body = {
                "node_id": node.get("node_id", ""),
                "action_kind": action_kind,
                "path": relative_path,
                "config": payload,
                "graph_id": graph.get("graph_id", ""),
            }
        elif action_kind == "benchmark_manifest":
            relative_path = str(metrics.get("relative_path", "benchmarks/manifest.json"))
            payload = self._build_benchmark_manifest(prompt=prompt, source_text=source_text, workspace_summary=workspace_summary)
            content = json.dumps(payload, indent=2, default=str)
            content_type = "application/json"
            body = {
                "node_id": node.get("node_id", ""),
                "action_kind": action_kind,
                "path": relative_path,
                "manifest": payload,
                "graph_id": graph.get("graph_id", ""),
            }
        elif action_kind == "dataset_pull_spec":
            relative_path = str(metrics.get("relative_path", "datasets/pull-spec.json"))
            payload = self._build_dataset_pull_spec(prompt=prompt, source_text=source_text, context=context)
            content = json.dumps(payload, indent=2, default=str)
            content_type = "application/json"
            body = {
                "node_id": node.get("node_id", ""),
                "action_kind": action_kind,
                "path": relative_path,
                "spec": payload,
                "graph_id": graph.get("graph_id", ""),
            }
        elif action_kind == "dataset_loader_template":
            relative_path = str(metrics.get("relative_path", "datasets/loader_template.py"))
            content = self._build_dataset_loader_template(prompt=prompt, source_text=source_text, context=context)
            body = {
                "node_id": node.get("node_id", ""),
                "action_kind": action_kind,
                "path": relative_path,
                "output": content,
                "graph_id": graph.get("graph_id", ""),
            }
        else:
            raise ValueError(f"unsupported workspace_action: {action_kind}")

        target = sandbox.write_text(relative_path, content, area="workspace")
        artifact = {
            "kind": "workspace_action_artifact",
            "label": str(node.get("title", action_kind)),
            "status": "completed",
            "path": str(target),
            "summary": f"workspace action {action_kind} generated",
            "content_type": content_type,
        }
        return {
            "node_id": str(node.get("node_id", "")),
            "status": "completed",
            "artifact": artifact,
            "result": body,
        }

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

            config = LiveModelConfig.from_overrides(overrides)
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
        failure_policy = self._classify_failure_policy(context=context)
        additions = self._propose_replan_nodes(
            node=node,
            graph=graph,
            context=context,
            metrics=metrics,
            failure_policy=failure_policy,
        )
        added_node_ids: list[str] = []
        if additions:
            synthesis = next((item for item in graph.get("nodes", []) if str(item.get("node_id", "")) == "synthesis"), None)
            for addition in additions:
                graph.setdefault("nodes", []).append(addition)
                added_node_ids.append(str(addition.get("node_id", "")))
                if isinstance(synthesis, dict):
                    deps = list(synthesis.get("depends_on", [])) if isinstance(synthesis.get("depends_on", []), list) else []
                    if addition["node_id"] not in deps:
                        deps.append(addition["node_id"])
                    synthesis["depends_on"] = deps
        body = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "node_type": node.get("node_type", ""),
            "failure_policy": failure_policy,
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
        result = source.get("result", {}) if isinstance(source, dict) else {}
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
        )
        if not proposed:
            proposed = self._local_replan_suggestions(context=context, failure_policy=failure_policy)

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
    def _local_replan_suggestions(*, context: dict[str, Any], failure_policy: dict[str, Any]) -> list[dict[str, Any]]:
        policy = str(failure_policy.get("policy", "none"))
        summary = str(failure_policy.get("summary", ""))
        if policy == "none":
            return []
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
    ) -> list[dict[str, Any]]:
        try:
            from app.harness.live_agent import CallBudget, LiveModelConfig, LiveModelGateway

            config = LiveModelConfig.from_overrides(self._resolve_live_model_overrides(context))
            if not config:
                return []
            gateway = LiveModelGateway(config)
            payload = {
                "query": str(metrics.get("prompt", context.get("query", ""))),
                "graph_id": graph.get("graph_id", ""),
                "replan_focus": metrics.get("replan_focus", []),
                "failure_policy": failure_policy,
                "node_results": context.get("node_results", {}),
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are replanning a general agent task graph after execution feedback. "
                        "Return strict JSON with key actions. "
                        "Each action must include node_type from workspace_action, tool_call, subagent. "
                        "Allowed workspace_action kinds: patch_scaffold, patch_draft, benchmark_run_config, "
                        "benchmark_manifest, dataset_pull_spec, dataset_loader_template. "
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
        target = str(targets[0]) if targets else "src/module.py"
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
    def _classify_failure_policy(*, context: dict[str, Any]) -> dict[str, Any]:
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
            count = int(output.get("count", 0)) if isinstance(output, dict) else 0
            if count <= 0:
                return {"policy": "evidence_gap", "summary": "evidence collection returned no records"}

        tool_failures = [
            item
            for item in (context.get("node_results", {}) or {}).values()
            if isinstance(item, dict) and isinstance(item.get("result", {}), dict) and item["result"].get("success") is False
        ]
        if tool_failures:
            return {"policy": "tool_failure", "summary": f"{len(tool_failures)} tool node(s) failed"}
        return {"policy": "none", "summary": "no repair policy triggered"}

    @staticmethod
    def _resolve_live_model_overrides(context: dict[str, Any]) -> dict[str, Any] | None:
        payload = context.get("live_model", {})
        return dict(payload) if isinstance(payload, dict) and payload else None

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
