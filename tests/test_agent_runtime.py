"""Tests for generic agent thread runtime and harness integration."""

from __future__ import annotations

import json
import time
from io import BytesIO
from pathlib import Path
from urllib import request

from app.agents.runtime import AgentThreadRuntime
from app.agents.sandbox import LocalThreadSandboxProvider, RemoteSandboxConfig, RemoteThreadSandboxProvider
from app.agents.scheduler import AgentExecutionScheduler
from app.agents.task_actions import TaskGraphActionMapper
from app.agents.workspace_view import ThreadWorkspaceStreamBuilder
from app.harness.engine import HarnessEngine
from app.harness.models import HarnessConstraints
from app.harness.state import HarnessMemoryStore


def test_agent_thread_runtime_creates_workspace_and_artifact(tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="General Agent Thread", agent_name="ResearchAgent")

    assert thread["thread_id"]
    assert Path(thread["workspace"]["workspace"]).exists()
    assert Path(thread["workspace"]["uploads"]).exists()
    assert Path(thread["workspace"]["outputs"]).exists()

    runtime.append_message(thread["thread_id"], "user", "Investigate rollout blockers.")
    artifact = runtime.write_artifact(
        thread["thread_id"],
        name="summary.md",
        content="# Summary\n",
        kind="report",
        content_type="text/markdown",
        summary="unit test report",
    )
    loaded = runtime.load_thread(thread["thread_id"])

    assert artifact["relative_path"] == "outputs/summary.md"
    assert loaded
    assert loaded["message_count"] == 1
    assert loaded["artifact_count"] == 1


def test_agent_thread_runtime_executes_task_graph_with_pause_resume_and_retry(tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="Graph Runtime Thread")
    graph = {
        "graph_id": "unit-graph",
        "nodes": [
            {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
            {"node_id": "evidence", "title": "Evidence", "node_type": "evidence", "status": "ready", "depends_on": ["scope"], "commands": [], "notes": [], "artifacts": [], "metrics": {"record_count": 2}},
            {"node_id": "review", "title": "Review", "node_type": "review", "status": "ready", "depends_on": ["evidence"], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
        ],
    }

    paused = runtime.execute_task_graph(thread["thread_id"], graph=graph, execution_label="unit", max_nodes=1)
    assert paused["status"] == "paused"
    assert paused["graph"]["nodes"][0]["status"] == "completed"

    resumed = runtime.resume_execution(thread["thread_id"], paused["execution_id"])
    assert resumed["status"] == "completed"
    assert all(node["status"] == "completed" for node in resumed["graph"]["nodes"])

    retried = runtime.retry_execution(thread["thread_id"], resumed["execution_id"], from_node_id="evidence")
    assert retried["status"] == "completed"
    assert retried["parent_execution_id"] == resumed["execution_id"]


def test_agent_thread_runtime_interrupts_execution_and_uses_thread_workspace(tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads", sandbox_provider=LocalThreadSandboxProvider())
    thread = runtime.create_thread(title="Interruptible Thread")
    graph = {
        "graph_id": "interrupt-graph",
        "nodes": [
            {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
            {"node_id": "package", "title": "Package", "node_type": "packaging", "status": "ready", "depends_on": ["scope"], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
        ],
    }

    runtime.request_interrupt(thread["thread_id"], reason="manual-test")
    interrupted = runtime.execute_task_graph(thread["thread_id"], graph=graph, execution_label="interrupt")
    persisted = runtime.load_thread(thread["thread_id"])

    assert interrupted["status"] == "interrupted"
    assert persisted
    assert persisted["status"] == "interrupted"
    assert persisted["control"]["interrupt_requested"] is True

    runtime.clear_interrupt(thread["thread_id"])
    resumed = runtime.resume_execution(thread["thread_id"], interrupted["execution_id"])
    sandbox = runtime.sandbox_provider.get(tmp_path / "threads" / thread["thread_id"])
    output_files = sandbox.list_files("outputs")

    assert resumed["status"] == "completed"
    assert any(path.endswith("scope.json") for path in output_files)
    assert any(path.endswith("package.json") for path in output_files)


def test_harness_run_can_persist_into_thread_runtime(tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")

    thread = engine.create_thread(title="Persistent Execution Thread")
    run = engine.run(
        query="Design an implementation roadmap with migration risks and validation gates.",
        constraints=HarnessConstraints(max_steps=3, max_tool_calls=3),
        thread_id=thread["thread_id"],
    )
    persisted = engine.get_thread(thread["thread_id"])

    assert run.metadata.get("thread", {}).get("thread_id") == thread["thread_id"]
    assert persisted
    assert persisted["latest_query"] == run.query
    assert persisted["message_count"] >= 2
    assert persisted["artifact_count"] >= 3
    assert persisted["runs"][-1]["mission_type"] in {"general", "research", "implementation"}
    assert any(item["kind"] == "mission_pack" for item in persisted["artifacts"])


def test_harness_engine_can_execute_mission_graph_inside_thread(tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")

    thread = engine.create_thread(title="Mission Execution Thread")
    run = engine.run(
        query="Create a practical execution plan with risks and measurable checkpoints.",
        constraints=HarnessConstraints(max_steps=3, max_tool_calls=3),
        thread_id=thread["thread_id"],
    )
    execution = engine.execute_thread_task_graph(
        thread["thread_id"],
        run.mission["task_graph"],
        execution_label=run.mission["name"],
        context={"mission": run.mission},
    )
    persisted = engine.get_thread(thread["thread_id"])

    assert execution["status"] == "completed"
    assert persisted
    assert persisted["executions"][-1]["execution_id"] == execution["execution_id"]
    assert any(item["kind"].endswith("_artifact") for item in persisted["artifacts"])


def test_agent_thread_runtime_supports_async_wait_and_interrupt(tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="Async Runtime Thread")
    graph = {
        "graph_id": "async-graph",
        "nodes": [
            {
                "node_id": "slow_scope",
                "title": "Slow Scope",
                "node_type": "routing",
                "status": "ready",
                "depends_on": [],
                "commands": [],
                "notes": [],
                "artifacts": [],
                "metrics": {"delay_ms": 250},
            },
            {
                "node_id": "finalize",
                "title": "Finalize",
                "node_type": "review",
                "status": "ready",
                "depends_on": ["slow_scope"],
                "commands": [],
                "notes": [],
                "artifacts": [],
                "metrics": {"delay_ms": 250},
            },
        ],
    }

    queued = runtime.start_task_graph_async(thread["thread_id"], graph=graph, execution_label="async")
    execution_id = queued["executions"][-1]["execution_id"]
    waiting = runtime.wait_for_execution(thread["thread_id"], execution_id, timeout_seconds=0.01)
    runtime.request_interrupt(thread["thread_id"], reason="test-interrupt")
    interrupted = runtime.wait_for_execution(thread["thread_id"], execution_id, timeout_seconds=2.0)

    assert waiting["status"] in {"queued", "running", "waiting"}
    assert interrupted["status"] == "interrupted"
    assert interrupted["interrupt_reason"] == "test-interrupt"


def test_parallel_subagents_scheduler_and_workspace_view(tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")
    engine.scheduler.runtime = engine.thread_runtime
    engine.subagents.runtime = engine.thread_runtime

    thread = engine.create_thread(title="Parallel Subagents Thread")
    run = engine.run(
        query="Create a cross-functional launch plan with evidence and execution tracks.",
        constraints=HarnessConstraints(max_steps=3, max_tool_calls=3),
        thread_id=thread["thread_id"],
    )
    subagents = [
        {
            "name": "ops",
            "graph": {
                "graph_id": "ops",
                "nodes": [
                    {"node_id": "scope_ops", "title": "Scope Ops", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
                    {"node_id": "report_ops", "title": "Report Ops", "node_type": "review", "status": "ready", "depends_on": ["scope_ops"], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
                ],
            },
            "context": {"mission": run.mission},
        },
        {
            "name": "risk",
            "graph": {
                "graph_id": "risk",
                "nodes": [
                    {"node_id": "scope_risk", "title": "Scope Risk", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
                    {"node_id": "report_risk", "title": "Report Risk", "node_type": "review", "status": "ready", "depends_on": ["scope_risk"], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
                ],
            },
            "context": {"mission": run.mission},
        },
    ]

    parallel = engine.run_parallel_subagents(thread["thread_id"], subagents, wait_timeout_seconds=5.0)
    stream = engine.build_thread_workspace_stream(thread["thread_id"])
    html = engine.render_thread_workspace_html(thread["thread_id"])

    assert parallel["summary"]["count"] == 2
    assert parallel["summary"]["completed"] == 2
    assert stream["schema"] == "agent-harness-workspace-stream/v1"
    assert len(stream["executions"]) >= 2
    assert "<html" in html.lower()


def test_scheduler_recovers_interrupted_execution(tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    scheduler = AgentExecutionScheduler(runtime)
    thread = runtime.create_thread(title="Recovery Thread")
    graph = {
        "graph_id": "recoverable",
        "nodes": [
            {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {"delay_ms": 200}},
            {"node_id": "report", "title": "Report", "node_type": "review", "status": "ready", "depends_on": ["scope"], "commands": [], "notes": [], "artifacts": [], "metrics": {"delay_ms": 200}},
        ],
    }

    queued = runtime.start_task_graph_async(thread["thread_id"], graph=graph, execution_label="recoverable")
    execution_id = queued["executions"][-1]["execution_id"]
    time.sleep(0.05)
    runtime.request_interrupt(thread["thread_id"], reason="recover-me")
    runtime.wait_for_execution(thread["thread_id"], execution_id, timeout_seconds=5.0)
    recoverable = scheduler.list_recoverable()
    recovered = scheduler.recover_execution(thread["thread_id"], execution_id, async_mode=False)

    assert any(item["execution_id"] == execution_id for item in recoverable)
    assert recovered["status"] == "completed"


def test_ground_live_document_output_removes_unsupported_quant_claims() -> None:
    mapper = TaskGraphActionMapper()
    local_output = "General agent frameworks need stronger evidence wiring and better final deliverables."
    live_output = """
# Memo

## Summary

General agent frameworks often lose to direct answers because they add orchestration layers without reliably improving the final user-facing result. The biggest gap is not the existence of planning or tool calls, but whether the runtime can turn those steps into a denser, more grounded final artifact that a reviewer would actually prefer to use.

## Evidence

The captured evidence in this run is qualitative and focuses on workflow overhead, weak closure artifacts, and the need for stronger synthesis. Benchmarks and external references matter when they are directly tied to the delivered result rather than left as detached support material. Framework quality rises when evidence enters the main answer instead of being trapped in side artifacts.

Benchmarks show 3-10x latency overhead and 200-500ms direct response times.

## Recommendation

Promote one primary deliverable, keep support artifacts subordinate, and treat evidence linkage as part of answer writing rather than as a parallel reporting lane.
"""
    grounded = mapper._ground_live_document_output(
        surface_kind="custom:memo",
        prompt="Write a memo.",
        source_text="Citation: https://example.com/qualitative-study\nFramework value appears only when evidence improves the final answer.",
        local_output=local_output,
        live_output=live_output,
    )

    assert "3-10x latency overhead" not in grounded
    assert "200-500ms direct response times" not in grounded
    assert "## Evidence Limits" in grounded
    assert "Promote one primary deliverable" in grounded


def test_live_workspace_action_generation_prefers_local_for_support_artifacts() -> None:
    mapper = TaskGraphActionMapper()

    result = mapper._live_workspace_action_generation(
        action_kind="custom:source_matrix",
        prompt="Write a source matrix.",
        source_text="Citation: https://example.com/source",
        workspace_summary={},
        local_relative_path="research/source-matrix.md",
        local_body={"node_id": "source-matrix"},
        local_content="# Local Source Matrix\n",
        content_type="text/markdown",
        context={"live_model": {"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"}},
    )

    assert result is None


def test_source_matrix_uses_evidence_records_before_keyword_heuristics() -> None:
    mapper = TaskGraphActionMapper()
    source_text = json.dumps(
        {
            "output": {
                "records": [
                    {
                        "title": "SWE-bench",
                        "summary": "Benchmark suite for measuring whether agents can resolve real GitHub issues with verifiable code changes.",
                        "url": "https://github.com/SWE-bench/SWE-bench",
                        "source_id": "built_in_catalog",
                        "trust_score": 0.92,
                    }
                ],
                "citations": ["https://github.com/SWE-bench/SWE-bench"],
            }
        }
    ) + "\n\nBenchmark-Reality Gap: Many frameworks optimize for synthetic benchmarks."

    document = mapper._build_source_matrix_document(
        prompt="Write a deep research memo.",
        source_text=source_text,
        title="Build Source Matrix",
    )

    assert "| SWE-bench |" in document
    assert "| **Benchmark-Reality Gap** |" not in document


def test_source_prompt_block_embeds_structured_evidence_payload() -> None:
    context = {
        "node_results": {
            "evidence": {
                "result": {
                    "title": "Build Evidence Dossier",
                    "output": {
                        "records": [
                            {
                                "title": "SWE-bench",
                                "summary": "Benchmark suite for measuring whether agents can resolve real GitHub issues with verifiable code changes.",
                                "url": "https://github.com/SWE-bench/SWE-bench",
                                "source_id": "built_in_catalog",
                                "trust_score": 0.92,
                            }
                        ],
                        "citations": ["https://github.com/SWE-bench/SWE-bench"],
                    },
                },
                "artifact": {},
            }
        }
    }

    block = TaskGraphActionMapper._source_prompt_block("evidence", context)

    assert "Structured Output JSON:" in block
    assert '"records"' in block
    assert "SWE-bench" in block


def test_research_evidence_payload_reads_direct_structured_json_records() -> None:
    source_text = """
[evidence] Build Evidence Dossier
Structured Output JSON:
{"records":[{"title":"SWE-bench","summary":"Benchmark suite for measuring whether agents can resolve real GitHub issues with verifiable code changes.","url":"https://github.com/SWE-bench/SWE-bench","source_id":"built_in_catalog","trust_score":0.92}],"citations":["https://github.com/SWE-bench/SWE-bench"]}
"""
    payload = TaskGraphActionMapper._research_evidence_payload(source_text)

    assert payload["records"]
    assert payload["records"][0]["title"] == "SWE-bench"
    assert "https://github.com/SWE-bench/SWE-bench" in payload["citations"]


def test_source_rows_use_source_specific_claim_bundles() -> None:
    source_text = json.dumps(
        {
            "output": {
                "records": [
                    {
                        "title": "Model Context Protocol Architecture",
                        "summary": "Reference for interoperable agent-tool integration and external capability composition.",
                        "url": "https://modelcontextprotocol.io/specification/2025-06-18/architecture/index",
                        "source_id": "built_in_catalog",
                        "trust_score": 0.88,
                    },
                    {
                        "title": "SWE-bench",
                        "summary": "Benchmark suite for measuring whether agents can resolve real GitHub issues with verifiable code changes.",
                        "url": "https://github.com/SWE-bench/SWE-bench",
                        "source_id": "built_in_catalog",
                        "trust_score": 0.92,
                    },
                ]
            }
        }
    )
    payload = TaskGraphActionMapper._research_evidence_payload(source_text)
    row_map = {row["source"]: row for row in payload["source_rows"]}

    assert "tool integration" in row_map["Model Context Protocol Architecture"]["claim"].lower()
    assert "code-task closure" in row_map["SWE-bench"]["gap"].lower()


def test_append_grounded_sources_section_falls_back_to_source_rows() -> None:
    text = TaskGraphActionMapper._append_grounded_sources_section(
        "# Memo\n\nMain body.\n",
        evidence={
            "records": [],
            "citations": [],
            "source_rows": [
                {
                    "source": "SWE-bench",
                    "claim": "Benchmark suite for measuring whether agents can resolve real GitHub issues with verifiable code changes.",
                }
            ],
            "benchmark_focus": [],
        },
    )

    assert "## Sources" in text
    assert "SWE-bench" in text


def test_append_grounded_sources_section_does_not_duplicate_existing_sources_block() -> None:
    body = (
        "# Memo\n\n"
        "## Sources\n\n"
        "- SWE-bench: benchmark suite for measuring whether agents can resolve real GitHub issues with verifiable code changes\n"
        "- tau-bench: enterprise-oriented benchmark for realistic tool-using agent tasks\n"
    )
    text = TaskGraphActionMapper._append_grounded_sources_section(
        body,
        evidence={
            "records": [
                {"title": "SWE-bench", "summary": "Benchmark suite", "url": "https://github.com/SWE-bench/SWE-bench"},
                {"title": "tau-bench", "summary": "Enterprise benchmark", "url": "https://github.com/sierra-research/tau-bench"},
            ],
            "citations": [],
            "source_rows": [],
            "benchmark_focus": [],
        },
    )

    assert text.count("## Sources") == 1


def test_ground_live_document_output_rejects_false_zero_evidence_claims() -> None:
    mapper = TaskGraphActionMapper()
    local_output = (
        "# Memo\n\n"
        "## Summary\n\n"
        "The available evidence is qualitative but non-empty, with benchmark and architecture references that constrain what can be claimed.\n"
    )
    live_output = (
        "# Memo\n\n"
        "## Summary\n\n"
        "Our evidence base contains zero records and no citations, so no analysis is possible.\n"
    )
    source_text = json.dumps(
        {
            "output": {
                "records": [
                    {
                        "title": "SWE-bench",
                        "summary": "Benchmark suite for measuring whether agents can resolve real GitHub issues with verifiable code changes.",
                        "url": "https://github.com/SWE-bench/SWE-bench",
                        "source_id": "built_in_catalog",
                        "trust_score": 0.92,
                    }
                ],
                "citations": ["https://github.com/SWE-bench/SWE-bench"],
            }
        }
    )

    grounded = mapper._ground_live_document_output(
        surface_kind="custom:memo",
        prompt="Write a memo.",
        source_text=source_text,
        local_output=local_output,
        live_output=live_output,
    )

    assert "zero records" not in grounded.lower()
    assert "no citations" not in grounded.lower()
    assert "SWE-bench" in grounded


def test_remote_thread_sandbox_provider_uses_http_contract(monkeypatch, tmp_path: Path) -> None:
    config = RemoteSandboxConfig(base_url="https://sandbox.example.com", api_key="secret", timeout_seconds=5)
    provider = RemoteThreadSandboxProvider(config)
    sandbox = provider.get(tmp_path / "threads" / "thread123")
    calls: list[tuple[str, str, dict[str, str]]] = []

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        body = req.data.decode("utf-8") if req.data else "{}"
        calls.append((req.method, req.full_url, dict(req.headers)))
        if req.full_url.endswith("/workspace"):
            return _FakeResponse({"root": "/remote/root", "workspace": "/remote/workspace", "uploads": "/remote/uploads", "outputs": "/remote/outputs"})
        if req.full_url.endswith("/write_text"):
            payload = json.loads(body)
            return _FakeResponse({"path": f"/remote/{payload['area']}/{payload['relative_path']}"})
        if req.full_url.endswith("/read_text"):
            return _FakeResponse({"content": "remote text"})
        if "/list_files" in req.full_url:
            return _FakeResponse({"files": ["a.txt", "b.txt"]})
        return _FakeResponse({"command": "echo hi", "exit_code": 0, "stdout": "hi", "stderr": "", "duration_ms": 1.0})

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)

    assert sandbox.workspace_paths()["workspace"] == "/remote/workspace"
    assert sandbox.write_text("demo.txt", "hello", area="outputs").as_posix() == "/remote/outputs/demo.txt"
    assert sandbox.read_text("demo.txt", area="outputs") == "remote text"
    assert sandbox.list_files("workspace") == ["a.txt", "b.txt"]
    assert sandbox.execute_command("echo hi").stdout == "hi"
    assert any("Authorization" in headers for _, _, headers in calls)


def test_workspace_stream_builder_compacts_thread_payload() -> None:
    payload = {
        "thread_id": "thread-1",
        "title": "Workspace",
        "status": "completed",
        "agent_name": "ResearchAgent",
        "latest_query": "hello",
        "workspace": {"workspace": "/tmp/workspace"},
        "messages": [{"role": "user", "content": "hello"}],
        "artifacts": [{"name": "summary.json", "kind": "run_summary"}],
        "executions": [{"execution_id": "exec1", "label": "mission", "status": "completed", "graph": {"summary": {"completed_nodes": 3, "node_count": 3, "runnable_nodes": []}}}],
        "events": [{"event": "execution_completed", "timestamp": "now"}],
    }
    stream = ThreadWorkspaceStreamBuilder().build(payload)

    assert stream["metrics"]["artifact_count"] == 1
    assert stream["executions"][0]["completed_nodes"] == 3


def test_engine_executes_generic_task_graph_inside_thread_workspace(tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")
    engine.scheduler.runtime = engine.thread_runtime
    engine.subagents.runtime = engine.thread_runtime

    thread = engine.create_thread(title="Generic Task Thread")
    sandbox = engine.thread_runtime.sandbox_provider.get(tmp_path / "threads" / thread["thread_id"])
    sandbox.write_text("notes.md", "Fix parser bug, add tests, and package the result.", area="workspace")
    sandbox.write_text("test_demo.py", "def test_demo():\n    assert True\n", area="workspace")
    sandbox.write_text("tests/test_demo.py", "def test_demo():\n    assert True\n", area="workspace")

    payload = engine.execute_thread_generic_task(
        thread["thread_id"],
        "Inspect the workspace, prepare a concrete engineering task brief, and run validation tests.",
        target="code",
    )
    persisted = engine.get_thread(thread["thread_id"])
    stream = engine.build_thread_workspace_stream(thread["thread_id"])
    html = engine.render_thread_workspace_html(thread["thread_id"])

    assert payload["execution"]["status"] == "completed"
    assert payload["graph"]["summary"]["node_count"] >= 5
    assert persisted is not None
    assert any(item["relative_path"].endswith("report.md") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("execution-trace.json") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("patch-scaffold.md") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("patch-draft.diff") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("completion-packet.json") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("delivery-bundle.json") for item in persisted["artifacts"])

    packet_path = Path(thread["workspace"]["workspace"]) / "packets" / "completion-packet.json"
    bundle_path = Path(thread["workspace"]["workspace"]) / "bundles" / "delivery-bundle.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert packet["schema"] == "agent-harness-completion-packet/v1"
    assert packet["summary"]["artifact_count"] >= 3
    assert any("patch-draft.diff" in str(item.get("path", "")) for item in packet["delivered_artifacts"])
    assert any(item.get("kind") == "completion_packet" for item in packet["task_spec"]["artifact_contracts"])
    assert packet["task_spec"]["primary_artifact_kind"] == "patch_draft"
    assert str(packet["primary_deliverable"]["path"]).endswith("patch-draft.diff")
    assert "delivery_bundle" not in packet["state_gap"]["missing_artifacts"]
    assert bundle["schema"] == "agent-harness-delivery-bundle/v1"
    assert bundle["artifact_manifest"]
    assert bundle["deliverable_index"]
    assert str(bundle["primary_deliverable"]["path"]).endswith("patch-draft.diff")
    assert any("patch-draft.diff" in str(item.get("path", "")) for item in bundle["artifact_manifest"])
    assert stream["completion_packet"]["schema"] == "agent-harness-completion-packet/v1"
    assert stream["delivery_bundle"]["schema"] == "agent-harness-delivery-bundle/v1"
    assert stream["showcase"]["primary_artifact"]["kind"] == "patch_draft"
    assert "delivery bundle" in stream["showcase"]["summary"].lower()
    assert stream["delivery_bundle"]["deliverable_index"]
    assert "Deliverable Index" in html
    assert "Openable Artifact Manifest" in html


def test_engine_executes_dataset_actions_inside_thread_workspace(tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")
    engine.scheduler.runtime = engine.thread_runtime
    engine.subagents.runtime = engine.thread_runtime

    thread = engine.create_thread(title="Data Task Thread")
    sandbox = engine.thread_runtime.sandbox_provider.get(tmp_path / "threads" / thread["thread_id"])
    sandbox.write_text("tests/test_demo.py", "def test_demo():\n    assert True\n", area="workspace")

    payload = engine.execute_thread_generic_task(
        thread["thread_id"],
        "Design an external evidence collection plan and prepare dataset pull spec plus loader template.",
        target="general",
    )
    persisted = engine.get_thread(thread["thread_id"])

    assert payload["execution"]["status"] == "completed"
    assert persisted is not None
    assert any(item["relative_path"].endswith("pull-spec.json") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("loader_template.py") for item in persisted["artifacts"])


def test_engine_executes_creative_artifact_actions_inside_thread_workspace(tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")
    engine.scheduler.runtime = engine.thread_runtime
    engine.subagents.runtime = engine.thread_runtime

    thread = engine.create_thread(title="Creative Task Thread")

    payload = engine.execute_thread_generic_task(
        thread["thread_id"],
        "Create a landing page, keynote deck, chart pack, podcast episode, video storyboard, and image prompt pack for agent-harness.",
        target="general",
    )
    persisted = engine.get_thread(thread["thread_id"])
    stream = engine.build_thread_workspace_stream(thread["thread_id"])

    assert payload["execution"]["status"] == "completed"
    assert persisted is not None
    assert any(item["relative_path"].endswith("landing-page-blueprint.md") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("deck-plan.md") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("chart-pack.json") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("episode-plan.md") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("storyboard.md") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("prompt-pack.md") for item in persisted["artifacts"])
    assert any("landing page" in item.lower() or "slide deck" in item.lower() for item in stream["showcase"]["deliverables"])
    assert stream["showcase"]["primary_artifact"]["kind"] == "webpage_blueprint"
    assert any(str(item.get("family", "")) in {"web", "slides"} for item in stream["delivery_bundle"]["deliverable_index"])


def test_engine_executes_custom_document_artifacts_for_business_query(tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")
    engine.scheduler.runtime = engine.thread_runtime
    engine.subagents.runtime = engine.thread_runtime

    thread = engine.create_thread(title="Business Memo Thread")

    payload = engine.execute_thread_generic_task(
        thread["thread_id"],
        "Prepare an executive memo, a one-pager brief, and a slide-deck plan for the rollout.",
        target="general",
    )
    persisted = engine.get_thread(thread["thread_id"])

    assert payload["execution"]["status"] == "completed"
    assert persisted is not None
    assert any(item["relative_path"].endswith("briefs/executive-memo.md") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("briefs/one-pager.md") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("slides/deck-plan.md") for item in persisted["artifacts"])


def test_graph_replan_can_add_tool_and_subagent_nodes_on_failure(tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")
    engine.scheduler.runtime = engine.thread_runtime
    engine.subagents.runtime = engine.thread_runtime

    thread = engine.create_thread(title="Replan Failure Thread")
    sandbox = engine.thread_runtime.sandbox_provider.get(tmp_path / "threads" / thread["thread_id"])
    sandbox.write_text("tests/test_fail.py", "def test_fail():\n    assert False\n", area="workspace")

    payload = engine.execute_thread_generic_task(
        thread["thread_id"],
        "Inspect the workspace, run validation tests, and produce a repair plan.",
        target="code",
    )
    persisted = engine.get_thread(thread["thread_id"])

    assert payload["execution"]["status"] == "completed"
    assert persisted is not None
    replan_result = payload["execution"]["context"]["node_results"]["replan"]["result"]
    assert replan_result["failure_policy"]["policy"] == "assertion_failure"
    assert replan_result["state_gap"]["missing_validation"] is True
    assert replan_result["capability_replan"]["steps"]
    assert any(item["relative_path"].endswith("replan_tool_workspace_file_search.json") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("replan_subagent_repair_probe.json") for item in persisted["artifacts"])


def test_live_model_initial_graph_expansion_can_add_tool_and_subagent_nodes(monkeypatch, tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")
    engine.scheduler.runtime = engine.thread_runtime
    engine.subagents.runtime = engine.thread_runtime

    thread = engine.create_thread(title="Live Expansion Thread")

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        body = json.loads(req.data.decode("utf-8")) if req.data else {}
        messages = body.get("messages", []) if isinstance(body, dict) else []
        system_text = str(messages[0].get("content", "")) if messages and isinstance(messages[0], dict) else ""
        if "expanding a general agent task graph" in system_text:
            content = {
                "nodes": [
                    {
                        "node_type": "tool_call",
                        "tool_name": "external_resource_hub",
                        "tool_args": {"query": "agent-harness task runtime patterns", "limit": 3},
                        "title": "Collect Runtime References",
                        "depends_on": ["analysis"],
                        "reason": "collect external references in the first pass",
                    },
                    {
                        "node_type": "subagent",
                        "subagent_kind": "research_probe",
                        "objective": "Synthesize frontier runtime and planner patterns",
                        "title": "Run Runtime Research Probe",
                        "depends_on": ["analysis"],
                        "reason": "delegate synthesis of external findings",
                    },
                    {
                        "node_type": "workspace_action",
                        "kind": "slide_deck_plan",
                        "title": "Generate Slide Deck Plan",
                        "depends_on": ["analysis"],
                        "reason": "materialize a presentation artifact",
                    },
                ],
                "replan_enabled": True,
                "replan_focus": ["evidence", "artifacts"],
                "rationale": ["add external evidence and a delegated synthesis pass before final synthesis"],
            }
            return _FakeResponse(
                {
                    "model": "demo-model",
                    "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
                }
            )
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [{"message": {"content": json.dumps({"selected_channels": ["discovery", "web"], "rationale": ["research presentation should inspect external evidence first"], "channel_scores": {"discovery": 0.9, "web": 0.8, "workspace": 0.2, "risk": 0.1}})}, "finish_reason": "stop"}],
            }
        )

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)

    payload = engine.execute_thread_generic_task(
        thread["thread_id"],
        "Research general agent runtime patterns and prepare a presentation-ready deliverable.",
        target="general",
        live_model={"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"},
    )
    persisted = engine.get_thread(thread["thread_id"])
    node_types = {str(item.get("node_type", "")) for item in payload["graph"]["nodes"]}

    assert payload["execution"]["status"] == "completed"
    assert "tool_call" in node_types
    assert "subagent" in node_types
    assert persisted is not None
    assert any(item["relative_path"].endswith("tool_external_resource_hub.json") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("subagent_research_probe.json") for item in persisted["artifacts"])
    assert any(item["relative_path"].endswith("deck-plan.md") for item in persisted["artifacts"])


def test_workspace_action_can_use_live_model_generated_content(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="Live Workspace Action")

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        content = {
            "relative_path": "web/live-landing-page.md",
            "content_text": "# Live Landing Page\n\nHero claim from live model.\n",
            "rationale": ["upgrade the first-screen specificity and CTA wording"],
        }
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
            }
        )

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    graph = {
        "graph_id": "live-action",
        "nodes": [
            {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
            {
                "node_id": "page",
                "title": "Generate Page",
                "node_type": "workspace_action",
                "status": "ready",
                "depends_on": ["scope"],
                "commands": [],
                "notes": [],
                "artifacts": [],
                "metrics": {"action_kind": "webpage_blueprint", "prompt": "Create a landing page for the agent runtime"},
            },
        ],
    }

    execution = runtime.execute_task_graph(
        thread["thread_id"],
        graph=graph,
        execution_label="live-action",
        context={"query": "Create a landing page for the agent runtime", "live_model": {"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"}},
    )

    result = execution["context"]["node_results"]["page"]["result"]
    output_path = Path(thread["workspace"]["workspace"]) / "web" / "live-landing-page.md"

    assert execution["status"] == "completed"
    assert result["generation_source"] == "live_model"
    assert output_path.exists()
    assert "Hero claim from live model." in output_path.read_text(encoding="utf-8")


def test_workspace_action_live_path_runs_before_local_fallback(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="Live First Workspace Action")

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        content = {
            "relative_path": "web/live-first-page.md",
            "content_text": "# Live First Page\n\nDirect live generation should win before local fallback.\n",
            "rationale": ["live path produced a usable primary artifact immediately"],
        }
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
            }
        )

    def fail_if_local(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("local workspace-action builder should not run when live generation succeeds")

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    monkeypatch.setattr(TaskGraphActionMapper, "_render_workspace_action_local", fail_if_local)

    graph = {
        "graph_id": "live-first-action",
        "nodes": [
            {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
            {
                "node_id": "page",
                "title": "Generate Page",
                "node_type": "workspace_action",
                "status": "ready",
                "depends_on": ["scope"],
                "commands": [],
                "notes": [],
                "artifacts": [],
                "metrics": {"action_kind": "webpage_blueprint", "prompt": "Create a landing page for the agent runtime"},
            },
        ],
    }

    execution = runtime.execute_task_graph(
        thread["thread_id"],
        graph=graph,
        execution_label="live-first-action",
        context={"query": "Create a landing page for the agent runtime", "live_model": {"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"}},
    )

    result = execution["context"]["node_results"]["page"]["result"]
    output_path = Path(thread["workspace"]["workspace"]) / "web" / "live-first-page.md"
    assert execution["status"] == "completed"
    assert result["generation_source"] == "live_model"
    assert output_path.exists()


def test_skill_call_can_use_live_model_generated_content(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="Live Skill Thread")

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        content = {
            "content_text": (
                "Live-generated research brief with cited benchmark direction. The result explains that agent quality improves only when evidence is "
                "connected directly to the final answer, when benchmark references are explicit, and when the runtime suppresses support artifacts that "
                "do not strengthen the deliverable. It also recommends a primary-output-first policy so users can review the memo before any packet, "
                "bundle, or other orchestration metadata."
            ),
            "rationale": ["upgrade the skill output from local heuristic to model-backed synthesis"],
        }
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
            }
        )

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    graph = {
        "graph_id": "live-skill",
        "nodes": [
            {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
            {
                "node_id": "synthesis",
                "title": "Research Brief",
                "node_type": "skill_call",
                "status": "ready",
                "depends_on": ["scope"],
                "commands": [],
                "notes": [],
                "artifacts": [],
                "metrics": {"skill_name": "research_brief", "prompt": "Write a benchmark-focused research brief", "source_node_ids": ["scope"]},
            },
        ],
    }

    execution = runtime.execute_task_graph(
        thread["thread_id"],
        graph=graph,
        execution_label="live-skill",
        context={"query": "Write a benchmark-focused research brief", "live_model": {"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"}},
    )

    result = execution["context"]["node_results"]["synthesis"]["result"]

    assert execution["status"] == "completed"
    assert result["generation_source"] == "live_model"
    assert "Live-generated research brief" in result["output"]


def test_custom_workspace_action_contract_executes_without_builtin_kind(tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="Custom Artifact Thread")

    graph = {
        "graph_id": "custom-artifact",
        "nodes": [
            {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
            {
                "node_id": "memo",
                "title": "Decision Memo",
                "node_type": "workspace_action",
                "status": "ready",
                "depends_on": ["scope"],
                "commands": [],
                "notes": [],
                "artifacts": [],
                "metrics": {
                    "action_kind": "custom:decision_memo",
                    "prompt": "Summarize the tradeoff and the recommended next action",
                    "relative_path": "briefs/decision-memo.md",
                    "content_type": "text/markdown",
                    "format_hint": "markdown",
                    "artifact_contract": {"title": "Decision Memo", "sections": ["Decision", "Tradeoffs", "Next Step"]},
                },
            },
        ],
    }

    execution = runtime.execute_task_graph(
        thread["thread_id"],
        graph=graph,
        execution_label="custom-artifact",
        context={"query": "Summarize the tradeoff and the recommended next action"},
    )

    result = execution["context"]["node_results"]["memo"]["result"]
    output_path = Path(thread["workspace"]["workspace"]) / "briefs" / "decision-memo.md"

    assert execution["status"] == "completed"
    assert result["action_kind"] == "custom:decision_memo"
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "Decision Memo" in text
    assert "## Decision" in text
    assert "## Next Step" in text


def test_research_document_local_fallback_is_paragraph_grade() -> None:
    source_text = """
{"output": {"records": [
  {"title": "SWE-bench", "summary": "Measures whether agents resolve real GitHub issues with verifiable patches.", "url": "https://github.com/SWE-bench/SWE-bench"},
  {"title": "GAIA Benchmark", "summary": "Evaluates general assistants on multi-step reasoning tasks.", "url": "https://gaia-benchmark.example"}
], "citations": ["https://github.com/SWE-bench/SWE-bench"]}}
"""
    text = TaskGraphActionMapper._build_research_style_document(
        prompt="Write a deep research memo on why agent frameworks often fail to beat a direct model answer.",
        source_text=source_text,
        workspace_summary={"languages": ["python"], "frameworks": ["pytest"]},
        title="Memo",
        kind="custom:memo",
        sections=["Context", "Evidence", "Recommendation", "Implications", "Next Step"],
    )

    assert "The core question is not whether frameworks can orchestrate more steps" in text
    assert "SWE-bench" in text
    assert "## Recommendation" in text
    assert "## Workspace Context" in text


def test_source_matrix_document_uses_richer_columns() -> None:
    source_text = """
{"output": {"records": [
  {"title": "tau-bench", "summary": "Enterprise-oriented benchmark for realistic tool-using agents.", "url": "https://github.com/sierra-research/tau-bench"}
], "citations": ["https://github.com/sierra-research/tau-bench"]}}
"""
    text = TaskGraphActionMapper._build_source_matrix_document(
        prompt="Prepare a research memo on what makes an agent framework valuable in real tasks.",
        source_text=source_text,
        title="Build Source Matrix",
    )

    assert "| What It Proves | Why It Matters | Remaining Uncertainty |" in text
    assert "tau-bench" in text
    assert "## Reading Notes" in text


def test_meaningful_lines_filters_markdown_tables_and_short_headings() -> None:
    lines = TaskGraphActionMapper._meaningful_lines(
        "## Baseline Answer\n| Question | Source |\n| --- | --- |\nUseful sentence about evidence.\n",
        limit=5,
    )

    assert "## Baseline Answer" not in lines
    assert not any(line.startswith("|") for line in lines)
    assert "Useful sentence about evidence." in lines


def test_custom_workspace_action_can_use_live_revision_loop(monkeypatch, tmp_path: Path) -> None:
    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    responses = [
        {"thesis": "Frameworks only win when they improve grounded closure.", "findings": ["Evidence must shape the final answer."]},
        "Draft memo that is still too generic.",
        {"confidence": 0.61, "blind_spots": ["needs more evidence"], "red_flags": ["too generic"], "improve": ["cite benchmarks and sharpen the recommendation"]},
        (
            "## Decision\n\n"
            "Adopt a primary-deliverable-first runtime. Ground the memo in SWE-bench and tau-bench evidence, then remove orchestration steps that do not improve "
            "the final result. The framework should collect benchmark context, connect that evidence directly to the recommendation, and keep support artifacts in "
            "a secondary role so the reviewer can inspect the main memo first. This makes the runtime more defensible than a direct answer because the evidence and "
            "the conclusion stay coupled instead of drifting apart across intermediate steps.\n"
        ),
    ]

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        payload = responses.pop(0)
        content = json.dumps(payload) if isinstance(payload, dict) else str(payload)
        return _FakeResponse({"model": "demo-model", "choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)
    mapper = TaskGraphActionMapper()
    result = mapper._live_workspace_action_generation(
        action_kind="custom:decision_memo",
        prompt="Write a decision memo on why the framework must beat direct model answers on real tasks.",
        source_text="SWE-bench highlights verifiable software-engineering evaluation. tau-bench focuses on task execution reliability.",
        workspace_summary={},
        local_relative_path="briefs/decision-memo.md",
        local_body={"node_id": "memo"},
        local_content="# Decision Memo\n\nLocal draft.\n",
        content_type="text/markdown",
        context={"live_model": {"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"}},
    )

    assert result is not None
    assert "primary-deliverable-first runtime" in str(result.get("content_text", ""))
    assert result.get("source") == "live_model"


def test_research_graph_includes_research_artifact_nodes(tmp_path: Path) -> None:
    engine = HarnessEngine()
    payload = engine.compile_generic_task_payload(
        query="Generate a deep research memo about benchmark strategy and evidence standards.",
        target="research",
        workspace_root=str(tmp_path),
    )
    node_ids = {str(item.get("node_id", "")) for item in payload["graph"]["nodes"]}
    node_map = {str(item.get("node_id", "")): item for item in payload["graph"]["nodes"]}

    assert "source_matrix" in node_ids
    assert "report_outline" in node_ids
    assert "direct_baseline" in node_ids
    assert "external_resources" in node_map["source_matrix"]["metrics"]["source_node_ids"]
    assert "evidence" in node_map["source_matrix"]["metrics"]["source_node_ids"]


def test_patch_draft_prefers_code_file_for_routing_task(tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="Patch Draft Thread")

    graph = {
        "graph_id": "patch-draft",
        "nodes": [
            {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
            {
                "node_id": "draft",
                "title": "Patch Draft",
                "node_type": "workspace_action",
                "status": "ready",
                "depends_on": ["scope"],
                "commands": [],
                "notes": [],
                "artifacts": [],
                "metrics": {
                    "action_kind": "patch_draft",
                    "prompt": "Inspect routing logic, fix parser weakness, and draft the patch.",
                    "workspace_summary": {
                        "sample_files": ["notes.md", "router.py", "tests/test_smoke.py"],
                        "languages": ["python"],
                        "frameworks": ["pytest"],
                    },
                },
            },
        ],
    }

    execution = runtime.execute_task_graph(
        thread["thread_id"],
        graph=graph,
        execution_label="patch-draft",
        context={"query": "Inspect routing logic, fix parser weakness, and draft the patch."},
    )

    output_path = Path(thread["workspace"]["workspace"]) / "patches" / "patch-draft.diff"
    text = output_path.read_text(encoding="utf-8")

    assert execution["status"] == "completed"
    assert "a/router.py" in text
    assert "return \"code\"" in text


def test_evidence_failure_policy_uses_record_count_not_legacy_count() -> None:
    policy = TaskGraphActionMapper._classify_failure_policy(
        context={
            "node_results": {
                "evidence": {
                    "result": {
                        "output": {
                            "record_count": 2,
                            "records": [{"title": "Agent Runtime Patterns"}, {"title": "Observability Study"}],
                        }
                    }
                }
            }
        }
    )

    assert policy["policy"] == "none"


def test_completion_packet_can_drive_gap_repair_policy() -> None:
    policy = TaskGraphActionMapper._classify_failure_policy(
        context={},
        completion_packet={
            "state_gap": {
                "missing_channels": [],
                "missing_artifacts": ["patch_draft"],
                "missing_validation": True,
                "failure_types": [],
            }
        },
    )

    assert policy["policy"] == "validation_gap"
    assert policy["source"] == "completion_packet"


def test_thread_event_contract_exposes_node_timeline_and_frontend_snapshot(tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="Event Contract Thread")
    graph = {
        "graph_id": "event-contract",
        "nodes": [
            {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
            {
                "node_id": "pack",
                "title": "Pack Evidence",
                "node_type": "workspace_action",
                "status": "ready",
                "depends_on": ["scope"],
                "commands": [],
                "notes": [],
                "artifacts": [],
                "metrics": {"action_kind": "slide_deck_plan", "prompt": "Prepare an executive presentation"},
            },
        ],
    }

    execution = runtime.execute_task_graph(thread["thread_id"], graph=graph, execution_label="event-contract", context={"query": "Prepare an executive presentation"})
    events = runtime.list_events(thread["thread_id"], after=0, limit=50)
    snapshot = runtime.export_frontend_thread_snapshot(thread["thread_id"])
    stream = ThreadWorkspaceStreamBuilder().build(runtime.load_thread(thread["thread_id"]) or {})

    assert execution["status"] == "completed"
    assert any(item["event"] == "node_started" and item["node_type"] == "workspace_action" for item in events)
    assert any(item["event"] == "node_result" and item["summary"] for item in events)
    assert any(item["event"] == "artifact_written" and item["artifact_relative_path"].endswith("slides/deck-plan.md") for item in events)
    assert any(item["event"] == "node_completed" and item["phase"] == "completed" for item in events)
    assert snapshot["metadata"]["event_contract"] == "agent-harness-thread-events/v3"
    assert snapshot["metadata"]["current_loop_phase"] == "deliver"
    assert snapshot["values"]["execution_loop"]["schema"] == "agent-harness-generic-loop/v1"
    assert any(item.get("node_type") == "workspace_action" for item in snapshot["tasks"])
    assert any(item.get("node_type") == "workspace_action" for item in snapshot["values"]["events"])
    assert any(item.get("artifact_relative_path", "").endswith("slides/deck-plan.md") for item in stream["timeline"])


def test_live_model_initial_graph_expansion_accepts_custom_workspace_action(monkeypatch, tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")
    engine.scheduler.runtime = engine.thread_runtime
    engine.subagents.runtime = engine.thread_runtime

    thread = engine.create_thread(title="Custom Expansion Thread")

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        body = json.loads(req.data.decode("utf-8")) if req.data else {}
        messages = body.get("messages", []) if isinstance(body, dict) else []
        system_text = str(messages[0].get("content", "")) if messages and isinstance(messages[0], dict) else ""
        if "expanding a general agent task graph" in system_text:
            content = {
                "nodes": [
                    {
                        "node_type": "workspace_action",
                        "kind": "custom:decision_memo",
                        "title": "Generate Decision Memo",
                        "depends_on": ["analysis"],
                        "reason": "this task needs a memo-shaped artifact",
                        "relative_path": "briefs/frontier-decision-memo.md",
                        "content_type": "text/markdown",
                        "format_hint": "markdown",
                        "artifact_contract": {"title": "Frontier Decision Memo", "sections": ["Decision", "Why now", "Risks"]},
                    }
                ],
                "replan_enabled": True,
                "replan_focus": ["artifacts"],
                "rationale": ["add a custom memo artifact for the final deliverable"],
            }
            return _FakeResponse(
                {
                    "model": "demo-model",
                    "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
                }
            )
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [{"message": {"content": json.dumps({"selected_channels": ["discovery"], "rationale": ["start broad before committing"], "channel_scores": {"discovery": 0.9, "web": 0.2, "workspace": 0.2, "risk": 0.1}})}, "finish_reason": "stop"}],
            }
        )

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    payload = engine.execute_thread_generic_task(
        thread["thread_id"],
        "Prepare a high-level decision memo about the general agent direction.",
        target="general",
        live_model={"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"},
    )
    persisted = engine.get_thread(thread["thread_id"])

    assert payload["execution"]["status"] == "completed"
    assert any(str(item.get("metrics", {}).get("action_kind", "")) == "custom:decision_memo" for item in payload["graph"]["nodes"])
    assert persisted is not None
    assert any(item["relative_path"].endswith("briefs/frontier-decision-memo.md") for item in persisted["artifacts"])


def test_graph_replan_selects_missing_dependency_policy(tmp_path: Path) -> None:
    engine = HarnessEngine()
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    engine.memory = HarnessMemoryStore(tmp_path / "memory.json")
    engine.scheduler.runtime = engine.thread_runtime
    engine.subagents.runtime = engine.thread_runtime

    thread = engine.create_thread(title="Missing Dependency Thread")
    sandbox = engine.thread_runtime.sandbox_provider.get(tmp_path / "threads" / thread["thread_id"])
    sandbox.write_text("tests/test_import.py", "import definitely_missing_module\n\ndef test_demo():\n    assert True\n", area="workspace")

    payload = engine.execute_thread_generic_task(
        thread["thread_id"],
        "Inspect the workspace, run validation tests, and recover from dependency failures.",
        target="code",
    )
    persisted = engine.get_thread(thread["thread_id"])

    assert payload["execution"]["status"] == "completed"
    assert persisted is not None
    replan_result = payload["execution"]["context"]["node_results"]["replan"]["result"]
    assert replan_result["failure_policy"]["policy"] == "missing_dependency"
    assert any(item["relative_path"].endswith("replan_tool_workspace_file_search.json") for item in persisted["artifacts"])


def test_graph_replan_live_model_receives_execution_loop(monkeypatch, tmp_path: Path) -> None:
    captured_requests: list[dict[str, object]] = []

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        captured_requests.append(json.loads(req.data.decode("utf-8")))
        content = {
            "actions": [
                {
                    "node_type": "workspace_action",
                    "kind": "patch_scaffold",
                    "title": "Generate Repair Scaffold",
                    "reason": "advance act phase with a concrete repair artifact",
                }
            ]
        }
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
            }
        )

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)

    graph = {
        "graph_id": "live-replan",
        "metadata": {
            "execution_loop": {
                "schema": "agent-harness-generic-loop/v1",
                "phases": [
                    {"phase": "observe", "goal": "gather missing context"},
                    {"phase": "decide", "goal": "pick the smallest repair"},
                    {"phase": "act", "goal": "materialize the missing artifact"},
                    {"phase": "deliver", "goal": "close the task"},
                ],
            }
        },
        "nodes": [],
    }

    mapper = TaskGraphActionMapper()
    replan_node = {
        "node_id": "replan",
        "title": "Replan",
        "node_type": "graph_replan",
        "status": "ready",
        "depends_on": ["completion_packet"],
        "commands": [],
        "notes": [],
        "artifacts": [],
        "metrics": {
            "prompt": "Inspect the workspace and fix the failing path",
            "replan_focus": ["artifacts", "validation"],
            "execution_loop": {
                "schema": "agent-harness-generic-loop/v1",
                "phases": [
                    {"phase": "observe", "goal": "gather missing context"},
                    {"phase": "decide", "goal": "pick the smallest repair"},
                    {"phase": "act", "goal": "materialize the missing artifact"},
                    {"phase": "deliver", "goal": "close the task"},
                ],
            },
            "loop_phase": "decide",
        },
    }
    mapper._live_replan_suggestions(
        node=replan_node,
        graph=graph,
        context={
            "query": "Inspect the workspace and fix the failing path",
            "current_loop_phase": "decide",
            "live_model": {"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"},
            "node_results": {},
        },
        metrics=replan_node["metrics"],
        failure_policy={"policy": "artifact_gap", "summary": "patch draft is still missing"},
        completion_packet={"state_gap": {"missing_artifacts": ["patch_draft"], "missing_validation": True}},
        state_gap={"missing_artifacts": ["patch_draft"], "missing_validation": True},
        capability_replan={"steps": []},
        fallback_seed=[{"node_type": "workspace_action", "kind": "patch_draft", "title": "Generate Missing Patch Draft"}],
    )

    assert captured_requests
    payload = json.loads(str(captured_requests[0]["messages"][1]["content"]))
    assert payload["execution_loop"]["schema"] == "agent-harness-generic-loop/v1"
    assert payload["current_loop_phase"] == "decide"
    assert payload["state_gap"]["missing_artifacts"] == ["patch_draft"]
    assert payload["fallback_seed"]
    assert payload["capability_replan"] == {"steps": []}


def test_live_subagent_plan_receives_execution_loop(monkeypatch) -> None:
    mapper = TaskGraphActionMapper()
    captured_requests: list[dict[str, object]] = []

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        captured_requests.append(json.loads(req.data.decode("utf-8")))
        content = {
            "skill_name": "research_brief",
            "tool_calls": [
                {"name": "external_resource_hub", "args": {"query": "collect current external references", "limit": 4}}
            ],
            "rationale": ["current phase is observe, so gather the smallest missing evidence first"],
        }
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
            }
        )

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)

    payload = mapper._live_subagent_plan(
        subagent_kind="research_probe",
        objective="Investigate current external evidence for the runtime",
        source_text="Need stronger evidence before synthesis.",
        local_plan={"skill_name": "research_brief", "tool_calls": [{"name": "external_resource_hub", "args": {"query": "Investigate current external evidence for the runtime", "limit": 5}}], "rationale": ["local plan"]},
        graph={
            "graph_id": "subagent-live-plan",
            "metadata": {
                "execution_loop": {
                    "schema": "agent-harness-generic-loop/v1",
                    "phases": [
                        {"phase": "observe", "goal": "gather missing context"},
                        {"phase": "decide", "goal": "choose the minimal next move"},
                        {"phase": "act", "goal": "materialize requested work"},
                        {"phase": "deliver", "goal": "ship the primary result"},
                    ],
                }
            },
            "summary": {"phase_summary": [{"phase": "observe", "completed_nodes": 0, "node_count": 2}]},
        },
        context={"current_loop_phase": "observe", "live_model": {"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"}},
    )

    assert payload is not None
    assert captured_requests
    request_payload = json.loads(str(captured_requests[0]["messages"][1]["content"]))
    assert request_payload["execution_loop"]["schema"] == "agent-harness-generic-loop/v1"
    assert request_payload["current_loop_phase"] == "observe"


def test_subagent_can_use_live_model_generated_mini_plan(monkeypatch, tmp_path: Path) -> None:
    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="Live Subagent Thread")
    sandbox = runtime.sandbox_provider.get(tmp_path / "threads" / thread["thread_id"])
    sandbox.write_text("notes.md", "repair parser issue and inspect tests", area="workspace")

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self.payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "skill_name": "codebase_triage",
                                    "tool_calls": [
                                        {"name": "tool_search", "args": {"query": "repair parser", "limit": 4}},
                                        {"name": "workspace_file_search", "args": {"query": "parser", "glob": "*", "limit": 4}},
                                    ],
                                    "rationale": ["use workspace plus discovery for repair work"],
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    graph = {
        "graph_id": "subagent-live",
        "nodes": [
            {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {}},
            {
                "node_id": "subagent",
                "title": "Subagent",
                "node_type": "subagent",
                "status": "ready",
                "depends_on": ["scope"],
                "commands": [],
                "notes": [],
                "artifacts": [],
                "metrics": {"subagent_kind": "repair_probe", "objective": "repair parser failure", "source_node_ids": ["scope"]},
            },
        ],
    }

    execution = runtime.execute_task_graph(
        thread["thread_id"],
        graph=graph,
        execution_label="live-subagent",
        context={"query": "repair parser failure", "live_model": {"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"}},
    )

    result = execution["context"]["node_results"]["subagent"]["result"]
    assert execution["status"] == "completed"
    assert result["plan"]["source"] == "live_model"
    assert result["plan"]["skill_name"] == "codebase_triage"
