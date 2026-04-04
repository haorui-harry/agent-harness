"""Generic agent thread runtime with workspace, artifacts, task execution, and control state."""

from __future__ import annotations

import copy
import json
import re
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
import time
from typing import Any

from app.agents.sandbox import LocalThreadSandboxProvider, ThreadSandboxProvider
from app.agents.task_actions import TaskGraphActionMapper

THREADS_DIR = Path(__file__).resolve().parents[2] / "data" / "agent_threads"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return text or "artifact"


def _as_message_blocks(content: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": str(content)}]


class AgentThreadRuntime:
    """Persistent runtime for generic agent threads and their artifacts."""

    def __init__(
        self,
        root_dir: Path | None = None,
        *,
        sandbox_provider: ThreadSandboxProvider | None = None,
        action_mapper: TaskGraphActionMapper | None = None,
    ) -> None:
        self.root = root_dir or THREADS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.sandbox_provider = sandbox_provider or LocalThreadSandboxProvider()
        self.action_mapper = action_mapper or TaskGraphActionMapper()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-thread-runtime")
        self._futures: dict[str, Future[Any]] = {}
        self._futures_lock = Lock()
        self._io_lock = Lock()

    def create_thread(
        self,
        *,
        title: str = "",
        agent_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        thread_id = uuid.uuid4().hex[:16]
        thread_dir = self.root / thread_id
        sandbox = self.sandbox_provider.get(thread_dir)
        payload = {
            "thread_id": thread_id,
            "title": title.strip() or "Untitled Thread",
            "agent_name": agent_name.strip(),
            "status": "idle",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "message_count": 0,
            "artifact_count": 0,
            "latest_query": "",
            "latest_summary": "",
            "workspace": sandbox.workspace_paths(),
            "messages": [],
            "artifacts": [],
            "runs": [],
            "executions": [],
            "events": [],
            "next_event_id": 1,
            "control": {
                "interrupt_requested": False,
                "interrupt_reason": "",
                "active_execution_id": "",
            },
            "metadata": dict(metadata or {}),
        }
        self._save_thread_payload(thread_id, payload)
        return payload

    def ensure_thread(
        self,
        thread_id: str,
        *,
        title: str = "",
        agent_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.load_thread(thread_id)
        if current:
            return current
        sandbox = self._sandbox(thread_id)
        payload = {
            "thread_id": thread_id,
            "title": title.strip() or "Untitled Thread",
            "agent_name": agent_name.strip(),
            "status": "idle",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "message_count": 0,
            "artifact_count": 0,
            "latest_query": "",
            "latest_summary": "",
            "workspace": sandbox.workspace_paths(),
            "messages": [],
            "artifacts": [],
            "runs": [],
            "executions": [],
            "events": [],
            "next_event_id": 1,
            "control": {
                "interrupt_requested": False,
                "interrupt_reason": "",
                "active_execution_id": "",
            },
            "metadata": dict(metadata or {}),
        }
        self._save_thread_payload(thread_id, payload)
        return payload

    def list_threads(self, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*/thread.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(
                {
                    "thread_id": payload.get("thread_id", path.parent.name),
                    "title": payload.get("title", ""),
                    "agent_name": payload.get("agent_name", ""),
                    "status": payload.get("status", ""),
                    "updated_at": payload.get("updated_at", ""),
                    "message_count": int(payload.get("message_count", 0)),
                    "artifact_count": int(payload.get("artifact_count", 0)),
                    "latest_query": payload.get("latest_query", ""),
                    "interrupt_requested": bool(payload.get("control", {}).get("interrupt_requested", False)),
                }
            )
        return rows[:limit]

    def load_thread(self, thread_id: str) -> dict[str, Any] | None:
        path = self.root / thread_id / "thread.json"
        if not path.exists():
            return None
        for attempt in range(3):
            try:
                with self._io_lock:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                payload, changed = self._reconcile_thread_payload(payload)
                if changed:
                    self._save_thread_payload(thread_id, payload)
                return payload
            except json.JSONDecodeError:
                if attempt >= 2:
                    raise
                time.sleep(0.01)
        return None

    def _reconcile_thread_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if not isinstance(payload, dict):
            return payload, False
        executions = payload.get("executions", []) if isinstance(payload.get("executions", []), list) else []
        if not executions:
            return payload, False
        latest = executions[-1] if isinstance(executions[-1], dict) else {}
        execution_id = str(latest.get("execution_id", ""))
        future_active = False
        if execution_id:
            with self._futures_lock:
                future = self._futures.get(execution_id)
            future_active = bool(future and not future.done())
        if future_active:
            return payload, False

        changed = False
        control = payload.setdefault("control", {})
        graph_summary = latest.get("graph", {}).get("summary", {}) if isinstance(latest.get("graph", {}), dict) else {}
        node_count = int(graph_summary.get("node_count", 0) or 0)
        completed_nodes = int(graph_summary.get("completed_nodes", 0) or 0)
        runnable_nodes = graph_summary.get("runnable_nodes", []) if isinstance(graph_summary.get("runnable_nodes", []), list) else []
        execution_status = str(latest.get("status", ""))
        thread_status = str(payload.get("status", ""))

        desired_status = thread_status
        desired_execution_status = execution_status
        if execution_status == "running":
            if node_count and completed_nodes >= node_count and not runnable_nodes:
                desired_execution_status = "completed"
                desired_status = "completed"
            elif runnable_nodes or completed_nodes < node_count:
                desired_execution_status = "paused"
                desired_status = "paused"
        elif execution_status in {"completed", "failed", "paused", "interrupted"}:
            desired_status = execution_status

        if desired_execution_status and desired_execution_status != execution_status:
            latest["status"] = desired_execution_status
            latest["updated_at"] = _utc_now()
            changed = True
        if desired_status and desired_status != thread_status:
            payload["status"] = desired_status
            payload["updated_at"] = _utc_now()
            changed = True
        if desired_status in {"completed", "failed", "paused", "interrupted"} and str(control.get("active_execution_id", "")):
            control["active_execution_id"] = ""
            changed = True
        return payload, changed

    def get_runtime_context(self, thread_id: str, limit: int = 8) -> dict[str, Any]:
        payload = self.load_thread(thread_id) or self.ensure_thread(thread_id)
        return {
            "thread_id": thread_id,
            "title": payload.get("title", ""),
            "messages": list(payload.get("messages", []))[-limit:],
            "artifacts": list(payload.get("artifacts", []))[-limit:],
            "runs": list(payload.get("runs", []))[-limit:],
            "executions": list(payload.get("executions", []))[-limit:],
            "workspace": payload.get("workspace", {}),
            "control": payload.get("control", {}),
        }

    def export_frontend_thread_snapshot(self, thread_id: str) -> dict[str, Any]:
        """Export one thread in a DeerFlow-like frontend snapshot contract."""

        payload = self.load_thread(thread_id) or self.ensure_thread(thread_id)
        workspace = payload.get("workspace", {}) if isinstance(payload.get("workspace", {}), dict) else {}
        artifacts = payload.get("artifacts", []) if isinstance(payload.get("artifacts", []), list) else []
        executions = payload.get("executions", []) if isinstance(payload.get("executions", []), list) else []
        events = payload.get("events", []) if isinstance(payload.get("events", []), list) else []
        active_execution = executions[-1] if executions else {}
        runnable = active_execution.get("graph", {}).get("summary", {}).get("runnable_nodes", []) if isinstance(active_execution.get("graph", {}), dict) else []
        uploads = self._sandbox(thread_id).list_files("uploads")
        output_virtuals = []
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("relative_path", "")).replace("\\", "/").strip("/")
            if rel:
                output_virtuals.append(f"/mnt/user-data/{rel}")
        return {
            "values": {
                "messages": [self._serialize_message(item) for item in payload.get("messages", []) if isinstance(item, dict)],
                "thread_data": {
                    "workspace_path": str(workspace.get("workspace", "")),
                    "uploads_path": str(workspace.get("uploads", "")),
                    "outputs_path": str(workspace.get("outputs", "")),
                },
                "uploaded_files": [f"/mnt/user-data/uploads/{item}" for item in uploads],
                "title": payload.get("title", ""),
                "artifacts": output_virtuals,
                "events": self._serialize_frontend_events(events),
            },
            "next": [str(item) for item in runnable],
            "tasks": self._serialize_task_events(events),
            "metadata": {
                "thread_id": payload.get("thread_id", thread_id),
                "source": "agent-harness-thread-runtime",
                "event_contract": "agent-harness-thread-events/v2",
                "status": payload.get("status", ""),
                "agent_name": payload.get("agent_name", ""),
                "step": len(payload.get("events", [])),
                "active_execution_id": payload.get("control", {}).get("active_execution_id", ""),
            },
            "created_at": payload.get("created_at", ""),
            "checkpoint": {
                "checkpoint_id": active_execution.get("execution_id", ""),
                "thread_id": payload.get("thread_id", thread_id),
                "checkpoint_ns": "",
            },
            "parent_checkpoint": {
                "checkpoint_id": active_execution.get("parent_execution_id", ""),
                "thread_id": payload.get("thread_id", thread_id),
                "checkpoint_ns": "",
            },
            "interrupts": payload.get("control", {}) if payload.get("control") else [],
            "checkpoint_id": active_execution.get("execution_id", ""),
            "parent_checkpoint_id": active_execution.get("parent_execution_id", ""),
        }

    def append_message(self, thread_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        payload = self.ensure_thread(thread_id)
        payload.setdefault("messages", []).append(
            {
                "role": role,
                "content": str(content),
                "timestamp": _utc_now(),
                "metadata": dict(metadata or {}),
            }
        )
        payload["message_count"] = len(payload.get("messages", []))
        payload["updated_at"] = _utc_now()
        self._save_thread_payload(thread_id, payload)

    def append_event(self, thread_id: str, event: dict[str, Any]) -> None:
        payload = self.ensure_thread(thread_id)
        payload.setdefault("events", [])
        next_event_id = int(payload.get("next_event_id", len(payload.get("events", [])) + 1))
        payload["events"].append({**dict(event), "timestamp": _utc_now(), "event_id": next_event_id})
        payload["next_event_id"] = next_event_id + 1
        payload["updated_at"] = _utc_now()
        self._save_thread_payload(thread_id, payload)

    def list_events(self, thread_id: str, *, after: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        payload = self.ensure_thread(thread_id)
        events = payload.get("events", []) if isinstance(payload.get("events", []), list) else []
        rows = [
            dict(item)
            for item in events
            if isinstance(item, dict) and int(item.get("event_id", 0)) > max(0, int(after))
        ]
        return rows[: max(1, min(int(limit), 500))]

    def wait_for_events(
        self,
        thread_id: str,
        *,
        after: int = 0,
        limit: int = 100,
        timeout_seconds: float = 15.0,
        poll_interval_seconds: float = 0.1,
    ) -> list[dict[str, Any]]:
        deadline = time.time() + max(0.1, float(timeout_seconds))
        while time.time() <= deadline:
            rows = self.list_events(thread_id, after=after, limit=limit)
            if rows:
                return rows
            time.sleep(max(0.02, float(poll_interval_seconds)))
        return []

    def write_artifact(
        self,
        thread_id: str,
        *,
        name: str,
        content: str,
        kind: str,
        content_type: str,
        directory: str = "outputs",
        summary: str = "",
    ) -> dict[str, Any]:
        payload = self.ensure_thread(thread_id)
        thread_root = (self.root / thread_id).resolve()
        safe_name = name.strip() or f"{kind}.txt"
        target = self._sandbox(thread_id).write_text(safe_name, content, area=directory)
        artifact = {
            "artifact_id": uuid.uuid4().hex[:12],
            "name": safe_name,
            "kind": kind,
            "content_type": content_type,
            "path": str(target),
            "relative_path": target.resolve().relative_to(thread_root).as_posix(),
            "summary": summary,
            "created_at": _utc_now(),
            "size_bytes": target.stat().st_size,
        }
        payload.setdefault("artifacts", []).append(artifact)
        payload["artifact_count"] = len(payload.get("artifacts", []))
        payload["updated_at"] = _utc_now()
        self._save_thread_payload(thread_id, payload)
        return artifact

    def request_interrupt(self, thread_id: str, reason: str = "manual") -> dict[str, Any]:
        payload = self.ensure_thread(thread_id)
        payload.setdefault("control", {})
        payload["control"]["interrupt_requested"] = True
        payload["control"]["interrupt_reason"] = reason
        if payload.get("status") == "running":
            payload["status"] = "interrupt_requested"
        self._save_thread_payload(thread_id, payload)
        self.append_event(thread_id, {"event": "interrupt_requested", "reason": reason})
        return self.load_thread(thread_id) or payload

    def clear_interrupt(self, thread_id: str) -> dict[str, Any]:
        payload = self.ensure_thread(thread_id)
        payload.setdefault("control", {})
        payload["control"]["interrupt_requested"] = False
        payload["control"]["interrupt_reason"] = ""
        if payload.get("status") == "interrupt_requested":
            payload["status"] = "idle"
        self._save_thread_payload(thread_id, payload)
        self.append_event(thread_id, {"event": "interrupt_cleared"})
        return self.load_thread(thread_id) or payload

    def execute_task_graph(
        self,
        thread_id: str,
        *,
        graph: dict[str, Any],
        execution_label: str = "",
        context: dict[str, Any] | None = None,
        max_nodes: int = 0,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        payload = self.ensure_thread(thread_id)
        execution = self._get_execution(payload, execution_id)
        if execution is None:
            execution = self._create_execution(graph=graph, execution_label=execution_label, context=context)
            payload.setdefault("executions", []).append(execution)
        elif context:
            execution["context"] = {**dict(execution.get("context", {})), **dict(context)}
        execution["status"] = "running"
        execution["updated_at"] = _utc_now()
        payload["status"] = "running"
        payload.setdefault("control", {})
        payload["control"]["active_execution_id"] = execution["execution_id"]
        self._save_execution(thread_id, payload, execution)

        sandbox = self._sandbox(thread_id)
        execution_context = dict(execution.get("context", {}) if isinstance(execution.get("context", {}), dict) else {})
        execution_context.setdefault("thread_id", thread_id)
        execution_context.setdefault("workspace", payload.get("workspace", sandbox.workspace_paths()))
        execution_context.setdefault("node_results", {})
        execution["context"] = execution_context
        processed = 0

        while True:
            if bool(payload.get("control", {}).get("interrupt_requested", False)):
                execution["status"] = "interrupted"
                execution["updated_at"] = _utc_now()
                execution["interrupt_reason"] = str(payload["control"].get("interrupt_reason", "manual"))
                payload["status"] = "interrupted"
                self._save_execution(thread_id, payload, execution)
                self.append_event(
                    thread_id,
                    {
                        "event": "execution_interrupted",
                        "execution_id": execution["execution_id"],
                        "reason": execution["interrupt_reason"],
                    },
                )
                return execution

            node = self._next_runnable_node(execution["graph"])
            if node is None:
                break
            if max_nodes and processed >= max_nodes:
                execution["status"] = "paused"
                execution["updated_at"] = _utc_now()
                self._refresh_graph_summary(execution["graph"])
                payload["status"] = "paused"
                self._save_execution(thread_id, payload, execution)
                self.append_event(
                    thread_id,
                    {
                        "event": "execution_paused",
                        "execution_id": execution["execution_id"],
                        "processed_nodes": processed,
                    },
                )
                return execution

            node["status"] = "running"
            self._save_execution(thread_id, payload, execution)
            self.append_event(
                thread_id,
                self._build_node_event_payload(
                    event="node_started",
                    phase="started",
                    execution_id=execution["execution_id"],
                    node=node,
                ),
            )
            result = self.action_mapper.execute_node(
                sandbox=sandbox,
                execution_id=execution["execution_id"],
                node=node,
                graph=execution["graph"],
                context=execution_context,
            )
            node["status"] = str(result.get("status", "completed"))
            execution_context.setdefault("node_results", {})[str(node.get("node_id", ""))] = result
            execution_context["latest_result"] = result
            execution["context"] = execution_context
            if result.get("artifact"):
                node.setdefault("artifacts", []).append(dict(result["artifact"]))
                thread_root = (self.root / thread_id).resolve()
                artifact_path = Path(str(result["artifact"].get("path", ""))).resolve()
                artifact_record = {
                    "artifact_id": uuid.uuid4().hex[:12],
                    "name": Path(str(result["artifact"].get("path", ""))).name,
                    "kind": result["artifact"].get("kind", "node_artifact"),
                    "content_type": str(result["artifact"].get("content_type", "application/json")),
                    "path": str(result["artifact"].get("path", "")),
                    "relative_path": artifact_path.relative_to(thread_root).as_posix(),
                    "summary": str(result["artifact"].get("summary", "")),
                    "created_at": _utc_now(),
                    "size_bytes": artifact_path.stat().st_size,
                }
                payload.setdefault("artifacts", []).append(artifact_record)
                payload["artifact_count"] = len(payload.get("artifacts", []))
                self.append_event(
                    thread_id,
                    self._build_node_event_payload(
                        event="artifact_written",
                        phase="artifact",
                        execution_id=execution["execution_id"],
                        node=node,
                        result=result,
                        artifact=artifact_record,
                    ),
                )
            execution.setdefault("node_results", []).append(result)
            execution["updated_at"] = _utc_now()
            self._refresh_graph_summary(execution["graph"])
            processed += 1
            self._save_execution(thread_id, payload, execution)
            self.append_event(
                thread_id,
                self._build_node_event_payload(
                    event="node_result",
                    phase="result",
                    execution_id=execution["execution_id"],
                    node=node,
                    result=result,
                ),
            )
            self.append_event(
                thread_id,
                self._build_node_event_payload(
                    event="node_completed",
                    phase="completed",
                    execution_id=execution["execution_id"],
                    node=node,
                    result=result,
                ),
            )
            payload = self.load_thread(thread_id) or payload

        execution["status"] = "completed"
        execution["updated_at"] = _utc_now()
        self._refresh_graph_summary(execution["graph"])
        payload["status"] = "completed"
        payload["control"]["active_execution_id"] = ""
        self._save_execution(thread_id, payload, execution)
        self.append_event(
            thread_id,
            {
                "event": "execution_completed",
                "execution_id": execution["execution_id"],
                "node_count": len(execution.get("graph", {}).get("nodes", [])),
            },
        )
        return execution

    def resume_execution(self, thread_id: str, execution_id: str) -> dict[str, Any]:
        self.clear_interrupt(thread_id)
        payload = self.ensure_thread(thread_id)
        execution = self._get_execution(payload, execution_id)
        if execution is None:
            raise ValueError(f"unknown execution: {execution_id}")
        return self.execute_task_graph(
            thread_id,
            graph=execution["graph"],
            execution_label=str(execution.get("label", "")),
            context=execution.get("context", {}) if isinstance(execution.get("context", {}), dict) else None,
            execution_id=execution_id,
        )

    def start_task_graph_async(
        self,
        thread_id: str,
        *,
        graph: dict[str, Any],
        execution_label: str = "",
        context: dict[str, Any] | None = None,
        max_nodes: int = 0,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        payload = self.ensure_thread(thread_id)
        execution = self._get_execution(payload, execution_id)
        if execution is None:
            execution = self._create_execution(graph=graph, execution_label=execution_label, context=context)
            execution["status"] = "queued"
            current = self.load_thread(thread_id) or payload
            current.setdefault("executions", [])
            if not any(str(item.get("execution_id", "")) == execution["execution_id"] for item in current.get("executions", [])):
                current["executions"].append(execution)
            payload = current
            self._save_thread_payload(thread_id, payload)
        future = self._executor.submit(
            self.execute_task_graph,
            thread_id,
            graph=execution["graph"],
            execution_label=str(execution.get("label", execution_label)),
            context=context,
            max_nodes=max_nodes,
            execution_id=execution["execution_id"],
        )
        with self._futures_lock:
            self._futures[execution["execution_id"]] = future
        self.append_event(
            thread_id,
            {
                "event": "execution_queued",
                "execution_id": execution["execution_id"],
                "label": execution.get("label", execution_label),
            },
        )
        response = self.load_thread(thread_id) or payload
        response["execution"] = copy.deepcopy(execution)
        return response

    def wait_for_execution(
        self,
        thread_id: str,
        execution_id: str,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        future: Future[Any] | None
        with self._futures_lock:
            future = self._futures.get(execution_id)
        if future is None:
            payload = self.ensure_thread(thread_id)
            execution = self._get_execution(payload, execution_id)
            if execution is None:
                raise ValueError(f"unknown execution: {execution_id}")
            return execution
        try:
            result = future.result(timeout=max(0.1, timeout_seconds))
        except FuturesTimeoutError:
            payload = self.ensure_thread(thread_id)
            execution = self._get_execution(payload, execution_id)
            return execution or {"execution_id": execution_id, "status": "waiting"}
        finally:
            if future.done():
                with self._futures_lock:
                    self._futures.pop(execution_id, None)
        return result

    def retry_execution(self, thread_id: str, execution_id: str, from_node_id: str = "") -> dict[str, Any]:
        payload = self.ensure_thread(thread_id)
        execution = self._get_execution(payload, execution_id)
        if execution is None:
            raise ValueError(f"unknown execution: {execution_id}")
        graph = copy.deepcopy(execution["graph"])
        reset = not from_node_id
        for node in graph.get("nodes", []):
            node_id = str(node.get("node_id", ""))
            if from_node_id and node_id == from_node_id:
                reset = True
            if reset:
                node["status"] = "ready" if node.get("depends_on") else "completed"
                node["artifacts"] = []
        self._refresh_graph_summary(graph)
        retry_execution = self._create_execution(
            graph=graph,
            execution_label=f"retry:{execution.get('label', execution_id)}",
            parent_execution_id=execution_id,
            context=execution.get("context", {}) if isinstance(execution.get("context", {}), dict) else None,
        )
        payload.setdefault("executions", []).append(retry_execution)
        self._save_thread_payload(thread_id, payload)
        self.append_event(
            thread_id,
            {
                "event": "execution_retried",
                "execution_id": retry_execution["execution_id"],
                "parent_execution_id": execution_id,
                "from_node_id": from_node_id,
            },
        )
        return self.execute_task_graph(
            thread_id,
            graph=retry_execution["graph"],
            execution_label=str(retry_execution.get("label", "")),
            execution_id=retry_execution["execution_id"],
        )

    def record_harness_run(
        self,
        thread_id: str,
        *,
        query: str,
        run: Any,
        mission: dict[str, Any],
        report_json: dict[str, Any],
        report_markdown: str,
    ) -> dict[str, Any]:
        payload = self.ensure_thread(
            thread_id,
            title=str(getattr(run, "query", "") or query)[:80],
            agent_name=str(getattr(run, "metadata", {}).get("selected_agent", "")),
        )
        self.append_message(
            thread_id,
            "user",
            query,
            metadata={"source": "thread_runtime"},
        )
        self.append_message(
            thread_id,
            "assistant",
            str(getattr(run, "final_answer", "")),
            metadata={
                "source": "harness_run",
                "completed": bool(getattr(run, "completed", False)),
            },
        )

        stem = f"{_utc_now()[:19].replace(':', '').replace('-', '')}-{_slugify(query)[:32]}"
        mission_artifact = self.write_artifact(
            thread_id,
            name=f"{stem}-mission.json",
            content=json.dumps(mission, indent=2, default=str),
            kind="mission_pack",
            content_type="application/json",
            summary=str(mission.get("primary_deliverable", "")),
        )
        report_artifact = self.write_artifact(
            thread_id,
            name=f"{stem}-report.md",
            content=report_markdown,
            kind="run_report",
            content_type="text/markdown",
            summary=str(report_json.get("final_answer_preview", ""))[:160],
        )
        summary_artifact = self.write_artifact(
            thread_id,
            name=f"{stem}-summary.json",
            content=json.dumps(report_json, indent=2, default=str),
            kind="run_summary",
            content_type="application/json",
            summary=f"completed={bool(getattr(run, 'completed', False))}",
        )

        payload = self.load_thread(thread_id) or payload
        payload["status"] = "completed" if bool(getattr(run, "completed", False)) else "blocked"
        payload["agent_name"] = str(getattr(run, "metadata", {}).get("selected_agent", ""))
        payload["latest_query"] = query
        payload["latest_summary"] = str(getattr(run, "final_answer", ""))[:240]
        payload["updated_at"] = _utc_now()
        payload.setdefault("runs", []).append(
            {
                "run_id": uuid.uuid4().hex[:12],
                "timestamp": _utc_now(),
                "query": query,
                "completed": bool(getattr(run, "completed", False)),
                "selected_agent": str(getattr(run, "metadata", {}).get("selected_agent", "")),
                "selected_skills": list(getattr(run, "metadata", {}).get("selected_skills", []))[:8],
                "mission_type": str(mission.get("name", "")),
                "value_index": float(getattr(run, "metadata", {}).get("value_card", {}).get("value_index", 0.0)),
                "artifacts": [
                    mission_artifact["relative_path"],
                    report_artifact["relative_path"],
                    summary_artifact["relative_path"],
                ],
            }
        )
        self._save_thread_payload(thread_id, payload)
        self.append_event(
            thread_id,
            {
                "event": "harness_run_recorded",
                "query": query,
                "mission_type": mission.get("name", ""),
                "artifacts": [mission_artifact["name"], report_artifact["name"], summary_artifact["name"]],
            },
        )
        return self.load_thread(thread_id) or payload

    def _create_execution(
        self,
        *,
        graph: dict[str, Any],
        execution_label: str = "",
        parent_execution_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        graph_copy = self._normalize_graph_for_execution(copy.deepcopy(graph))
        return {
            "execution_id": uuid.uuid4().hex[:12],
            "label": execution_label or str(graph_copy.get("graph_id", "task_graph")),
            "status": "ready",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "parent_execution_id": parent_execution_id,
            "interrupt_reason": "",
            "graph": graph_copy,
            "node_results": [],
            "context": dict(context or {}),
        }

    @staticmethod
    def _normalize_graph_for_execution(graph: dict[str, Any]) -> dict[str, Any]:
        nodes = graph.get("nodes", []) if isinstance(graph.get("nodes", []), list) else []
        if nodes and all(str(node.get("status", "")) == "completed" for node in nodes):
            for node in nodes:
                node.setdefault("metrics", {})
                node["metrics"]["source_status"] = "completed"
                node["status"] = "ready"
                node["artifacts"] = []
        AgentThreadRuntime._refresh_graph_summary(graph)
        return graph

    @staticmethod
    def _next_runnable_node(graph: dict[str, Any]) -> dict[str, Any] | None:
        nodes = graph.get("nodes", []) if isinstance(graph.get("nodes", []), list) else []
        by_id = {str(node.get("node_id", "")): node for node in nodes}
        for node in nodes:
            status = str(node.get("status", ""))
            if status not in {"ready", "planned"}:
                continue
            parents = [by_id[parent_id] for parent_id in node.get("depends_on", []) if parent_id in by_id]
            if all(str(parent.get("status", "")) == "completed" for parent in parents):
                return node
        return None

    @staticmethod
    def _refresh_graph_summary(graph: dict[str, Any]) -> None:
        nodes = graph.get("nodes", []) if isinstance(graph.get("nodes", []), list) else []
        edges = graph.get("edges", []) if isinstance(graph.get("edges", []), list) else []
        completed = sum(1 for node in nodes if str(node.get("status", "")) == "completed")
        by_id = {str(node.get("node_id", "")): node for node in nodes}
        runnable: list[str] = []
        for node in nodes:
            if str(node.get("status", "")) not in {"ready", "planned"}:
                continue
            if all(str(by_id[parent].get("status", "")) == "completed" for parent in node.get("depends_on", []) if parent in by_id):
                runnable.append(str(node.get("node_id", "")))
        graph["summary"] = {
            "node_count": len(nodes),
            "completed_nodes": completed,
            "completion_ratio": round(completed / max(len(nodes), 1), 4),
            "runnable_nodes": runnable,
            "critical_path": graph.get("summary", {}).get("critical_path", []),
            "edge_count": len(edges),
        }

    @staticmethod
    def _get_execution(payload: dict[str, Any], execution_id: str | None) -> dict[str, Any] | None:
        if not execution_id:
            return None
        for execution in payload.get("executions", []):
            if str(execution.get("execution_id", "")) == execution_id:
                return execution
        return None

    def _save_execution(self, thread_id: str, payload: dict[str, Any], execution: dict[str, Any]) -> None:
        current = self.load_thread(thread_id)
        if current:
            current_control = current.get("control", {})
            if current_control:
                payload["control"] = dict(current_control)
            if len(current.get("events", [])) > len(payload.get("events", [])):
                payload["events"] = list(current.get("events", []))
            if len(current.get("messages", [])) > len(payload.get("messages", [])):
                payload["messages"] = list(current.get("messages", []))
            payload.setdefault("executions", [])
            known_ids = {
                str(item.get("execution_id", ""))
                for item in payload.get("executions", [])
                if isinstance(item, dict)
            }
            for item in current.get("executions", []):
                if not isinstance(item, dict):
                    continue
                execution_id = str(item.get("execution_id", ""))
                if execution_id and execution_id not in known_ids:
                    payload["executions"].append(item)
                    known_ids.add(execution_id)
        for index, current in enumerate(payload.get("executions", [])):
            if str(current.get("execution_id", "")) == str(execution.get("execution_id", "")):
                payload["executions"][index] = execution
                break
        else:
            payload.setdefault("executions", []).append(execution)
        payload["artifact_count"] = len(payload.get("artifacts", []))
        payload["updated_at"] = _utc_now()
        self._save_thread_payload(thread_id, payload)

    def _sandbox(self, thread_id: str):
        return self.sandbox_provider.get(self.root / thread_id)

    @staticmethod
    def _serialize_message(message: dict[str, Any]) -> dict[str, Any]:
        role = str(message.get("role", "assistant"))
        content = str(message.get("content", ""))
        msg_type = {"user": "human", "assistant": "ai"}.get(role, role)
        return {
            "content": _as_message_blocks(content) if msg_type == "human" else content,
            "additional_kwargs": {},
            "response_metadata": {},
            "type": msg_type,
            "name": None,
            "id": str(message.get("timestamp", "")) or uuid.uuid4().hex[:12],
        }

    @staticmethod
    def _serialize_task_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tasks: dict[str, dict[str, Any]] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            name = str(event.get("task_name", "")).strip()
            kind = str(event.get("task_kind", "")).strip()
            execution_id = str(event.get("execution_id", "")).strip()
            if not name and not execution_id:
                continue
            key = execution_id or f"{kind}:{name}"
            event_name = str(event.get("event", ""))
            if event_name in {
                "task_started",
                "task_running",
                "task_completed",
                "task_failed",
                "task_interrupted",
                "task_updated",
            }:
                tasks[key] = {
                    "id": key,
                    "name": name or execution_id,
                    "kind": kind or "task",
                    "status": str(event.get("status", "")) or event_name.replace("task_", ""),
                    "event_id": int(event.get("event_id", 0)),
                    "updated_at": str(event.get("timestamp", "")),
                    "execution_id": execution_id,
                    "completed_nodes": int(event.get("completed_nodes", 0)),
                    "node_count": int(event.get("node_count", 0)),
                }
                continue
            if event_name not in {"node_started", "node_result", "artifact_written", "node_completed"}:
                continue
            node_id = str(event.get("node_id", "")).strip()
            node_key = f"{execution_id}:{node_id}" if execution_id and node_id else key or node_id
            if not node_key:
                continue
            tasks[node_key] = {
                "id": node_key,
                "name": str(event.get("node_title", "")) or node_id or execution_id,
                "kind": str(event.get("node_type", "")) or "node",
                "status": str(event.get("status", "")) or str(event.get("phase", "")) or event_name,
                "event_id": int(event.get("event_id", 0)),
                "updated_at": str(event.get("timestamp", "")),
                "execution_id": execution_id,
                "node_id": node_id,
                "node_type": str(event.get("node_type", "")),
                "phase": str(event.get("phase", "")),
                "summary": str(event.get("summary", "")),
                "artifact_relative_path": str(event.get("artifact_relative_path", "")),
                "artifact_kind": str(event.get("artifact_kind", "")),
                "preview": str(event.get("preview", "")),
                "completed_nodes": int(event.get("completed_nodes", 0)),
                "node_count": int(event.get("node_count", 0)),
            }
        return list(tasks.values())[-20:]

    @staticmethod
    def _serialize_frontend_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in events[-50:]:
            if not isinstance(event, dict):
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
                    "artifact_kind": str(event.get("artifact_kind", "")),
                    "preview": str(event.get("preview", "")),
                }
            )
        return rows

    @staticmethod
    def _build_node_event_payload(
        *,
        event: str,
        phase: str,
        execution_id: str,
        node: dict[str, Any],
        result: dict[str, Any] | None = None,
        artifact: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        result_dict = result if isinstance(result, dict) else {}
        result_payload = result_dict.get("result", {}) if isinstance(result_dict.get("result", {}), dict) else {}
        artifact_payload = artifact or {}
        return {
            "event": event,
            "event_contract": "agent-harness-thread-events/v2",
            "phase": phase,
            "execution_id": execution_id,
            "node_id": str(node.get("node_id", "")),
            "node_title": str(node.get("title", "")),
            "node_type": str(node.get("node_type", "")),
            "depends_on": list(node.get("depends_on", [])) if isinstance(node.get("depends_on", []), list) else [],
            "status": str(result_dict.get("status", node.get("status", phase))),
            "tool_name": str(metrics.get("tool_name", "")),
            "skill_name": str(metrics.get("skill_name", "")),
            "action_kind": str(metrics.get("action_kind", "")),
            "subagent_kind": str(metrics.get("subagent_kind", "")),
            "summary": AgentThreadRuntime._result_summary(result_payload, artifact_payload),
            "preview": AgentThreadRuntime._result_preview(result_payload),
            "artifact_relative_path": str(artifact_payload.get("relative_path", "")),
            "artifact_kind": str(artifact_payload.get("kind", "")),
        }

    @staticmethod
    def _result_summary(result_payload: dict[str, Any], artifact_payload: dict[str, Any]) -> str:
        if artifact_payload.get("summary"):
            return str(artifact_payload.get("summary", ""))
        for key in ("summary", "action_kind", "tool_name", "skill_name", "subagent_kind", "node_type"):
            value = str(result_payload.get(key, "")).strip()
            if value:
                return value
        if isinstance(result_payload.get("results", []), list) and result_payload.get("results", []):
            return f"produced {len(result_payload.get('results', []))} command result(s)"
        return "node updated"

    @staticmethod
    def _result_preview(result_payload: dict[str, Any]) -> str:
        for key in ("output", "notes", "objective", "path"):
            value = result_payload.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()[:240]
        for key in ("config", "manifest", "spec"):
            value = result_payload.get(key, {})
            if isinstance(value, dict) and value:
                return json.dumps(value, ensure_ascii=False, default=str)[:240]
        return ""

    def _save_thread_payload(self, thread_id: str, payload: dict[str, Any]) -> None:
        thread_dir = self.root / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)
        payload.setdefault("workspace", self._sandbox(thread_id).workspace_paths())
        payload.setdefault("executions", [])
        payload.setdefault("events", [])
        payload.setdefault("next_event_id", len(payload.get("events", [])) + 1)
        payload.setdefault(
            "control",
            {
                "interrupt_requested": False,
                "interrupt_reason": "",
                "active_execution_id": "",
            },
        )
        payload["message_count"] = len(payload.get("messages", []))
        payload["artifact_count"] = len(payload.get("artifacts", []))
        payload["updated_at"] = _utc_now()
        content = json.dumps(payload, indent=2, default=str)
        target = thread_dir / "thread.json"
        temp = thread_dir / "thread.json.tmp"
        with self._io_lock:
            temp.write_text(content, encoding="utf-8")
            temp.replace(target)
