"""Tests for research lab and newly added harness tools."""

from __future__ import annotations

import json
from urllib import request

from app.harness.engine import HarnessEngine
from app.harness.models import HarnessConstraints, ToolCall, ToolType
from app.harness.tools import ToolRegistry


def test_research_lab_lists_scenarios_and_presets() -> None:
    engine = HarnessEngine()
    scenarios = engine.list_research_scenarios()
    presets = engine.list_research_presets()

    assert len(scenarios) >= 6
    assert any(item.get("category") == "daily" for item in scenarios)
    assert any(item.get("category") == "research" for item in scenarios)
    assert any(item.get("category") == "safety" for item in scenarios)
    assert any(item.get("category") == "creative" for item in scenarios)
    assert any(item.get("category") == "enterprise" for item in scenarios)
    assert any(item.get("name") == "core" for item in presets)
    assert any(item.get("name") == "broad" for item in presets)


def test_research_lab_run_returns_ranked_leaderboard() -> None:
    engine = HarnessEngine()
    payload = engine.run_research_lab(
        preset="daily",
        repeats=1,
        scenario_ids=["daily-001", "daily-002"],
        constraints=HarnessConstraints(max_steps=4, max_tool_calls=4),
    )

    assert payload["scenario_count"] == 2
    assert payload["candidate_count"] >= 2
    assert len(payload["leaderboard"]) >= 2
    assert "best" in payload
    assert "composite_score" in payload["leaderboard"][0]
    assert "ci95_value_index" in payload["leaderboard"][0]
    assert "competition" in payload
    assert "pareto_frontier" in payload["competition"]
    assert "release_decision" in payload
    assert payload["release_decision"]["decision"] in {"go", "caution", "block"}
    repro = payload.get("reproducibility", {})
    assert repro.get("isolate_memory") is True
    assert repro.get("fresh_memory_per_candidate") is True


def test_tool_registry_new_tools_output_shape() -> None:
    tools = ToolRegistry()

    portfolio = tools.call(
        ToolCall(
            name="api_skill_portfolio_optimizer",
            tool_type=ToolType.API,
            args={"query": "daily planning and vendor comparison", "limit": 3},
        )
    )
    assert portfolio.success is True
    assert "portfolio" in portfolio.output
    assert "portfolio_summary" in portfolio.output

    design = tools.call(
        ToolCall(
            name="code_experiment_design",
            tool_type=ToolType.CODE,
            args={"query": "ablation benchmark for multi-agent routing", "max_experiments": 4},
        )
    )
    assert design.success is True
    assert "experiment_matrix" in design.output
    assert len(design.output["experiment_matrix"]) == 4


def test_tool_registry_supports_tool_search_workspace_and_task_graph(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.md").write_text("Fix parser bug and add regression tests.", encoding="utf-8")

    tools = ToolRegistry()
    catalog = tools.call(
        ToolCall(
            name="tool_search",
            tool_type=ToolType.CODE,
            args={"query": "workspace", "limit": 5},
        )
    )
    assert catalog.success is True
    assert any(item["name"] == "workspace_file_search" for item in catalog.output["matches"])

    search = tools.call(
        ToolCall(
            name="workspace_file_search",
            tool_type=ToolType.CODE,
            args={"workspace_root": str(workspace), "query": "parser", "glob": "*.md", "limit": 5},
        )
    )
    assert search.success is True
    assert search.output["count"] == 1

    graph = tools.call(
        ToolCall(
            name="task_graph_builder",
            tool_type=ToolType.CODE,
            args={"query": "Inspect workspace and prepare a code task brief", "target": "code", "workspace_root": str(workspace)},
        )
    )
    assert graph.success is True
    assert graph.output["profile"]["requires_workspace"] is True
    assert "workspace" in graph.output["profile"]["selected_channels"]
    assert graph.output["graph"]["summary"]["node_count"] >= 5


def test_task_graph_builder_infers_web_research_without_repo_context(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.md").write_text("Research topic with evidence and benchmarks.", encoding="utf-8")

    tools = ToolRegistry()
    graph = tools.call(
        ToolCall(
            name="task_graph_builder",
            tool_type=ToolType.CODE,
            args={"query": "Generate a deep research report with evidence", "target": "research", "workspace_root": str(workspace)},
        )
    )

    assert graph.success is True
    assert graph.output["profile"]["evidence_strategy"] == "web"
    assert "web" in graph.output["profile"]["selected_channels"]
    nodes = graph.output["graph"]["nodes"]
    node_ids = {item["node_id"] for item in nodes}
    assert {"external_resources", "evidence", "analysis", "validation", "synthesis", "report"}.issubset(node_ids)
    assert "workspace_scan" not in node_ids
    assert graph.output["graph"]["summary"]["node_count"] >= 6


def test_task_graph_builder_infers_workspace_grounded_repo_analysis(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "service.py").write_text("def broken_parser():\n    return None\n", encoding="utf-8")

    tools = ToolRegistry()
    graph = tools.call(
        ToolCall(
            name="task_graph_builder",
            tool_type=ToolType.CODE,
            args={"query": "Analyze my repository, propose fixes, and design tests", "workspace_root": str(workspace)},
        )
    )

    assert graph.success is True
    assert graph.output["profile"]["evidence_strategy"] == "workspace"
    assert "workspace" in graph.output["profile"]["selected_channels"]
    nodes = graph.output["graph"]["nodes"]
    node_ids = {item["node_id"] for item in nodes}
    assert {"workspace_scan", "workspace_focus", "analysis", "validation", "report"}.issubset(node_ids)
    assert {"action_patch_scaffold", "replan"}.issubset(node_ids)
    assert "completion_packet" in node_ids
    synthesis = next(item for item in nodes if item["node_id"] == "synthesis")
    assert "completion_packet" in synthesis["depends_on"]
    assert "external_resources" not in node_ids


def test_task_graph_builder_supports_hybrid_task_context(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "README.md").write_text("Compare our agent harness with external frameworks.", encoding="utf-8")

    tools = ToolRegistry()
    graph = tools.call(
        ToolCall(
            name="task_graph_builder",
            tool_type=ToolType.CODE,
            args={
                "query": "Compare my repository with external agent frameworks and produce a research improvement report",
                "workspace_root": str(workspace),
            },
        )
    )

    assert graph.success is True
    assert graph.output["profile"]["evidence_strategy"] == "hybrid"
    assert {"workspace", "web"}.issubset(set(graph.output["profile"]["selected_channels"]))
    node_ids = {item["node_id"] for item in graph.output["graph"]["nodes"]}
    assert "workspace_scan" in node_ids
    assert "external_resources" in node_ids


def test_task_graph_builder_expands_benchmark_graph_with_executable_actions(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "tests").mkdir(parents=True, exist_ok=True)
    (workspace / "tests" / "test_demo.py").write_text("def test_demo():\n    assert True\n", encoding="utf-8")

    tools = ToolRegistry()
    graph = tools.call(
        ToolCall(
            name="task_graph_builder",
            tool_type=ToolType.CODE,
            args={
                "query": "Build a benchmark and ablation plan, produce run config, and pull external evaluation data",
                "workspace_root": str(workspace),
            },
        )
    )

    assert graph.success is True
    assert graph.output["profile"]["graph_expansion"]["replan_enabled"] is True
    node_ids = {item["node_id"] for item in graph.output["graph"]["nodes"]}
    assert {"action_benchmark_run_config", "action_dataset_pull_spec", "replan"}.issubset(node_ids)


def test_task_graph_builder_can_use_live_model_for_channel_selection(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.md").write_text("general notes", encoding="utf-8")

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
                                    "selected_channels": ["discovery", "web"],
                                    "rationale": ["external evidence is necessary", "discovery should stay enabled"],
                                    "channel_scores": {
                                        "workspace": 0.2,
                                        "web": 0.91,
                                        "discovery": 0.88,
                                        "risk": 0.1,
                                    },
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)

    tools = ToolRegistry()
    graph = tools.call(
        ToolCall(
            name="task_graph_builder",
            tool_type=ToolType.CODE,
            args={
                "query": "Investigate frontier agent trends and prepare a research memo",
                "workspace_root": str(workspace),
                "live_model": {
                    "base_url": "https://example.com/v1",
                    "api_key": "secret",
                    "model_name": "demo-model",
                },
            },
        )
    )

    assert graph.success is True
    selected = set(graph.output["profile"]["selected_channels"])
    assert {"discovery", "web"}.issubset(selected)
    assert "live model refined channel selection" in graph.output["profile"]["deliberation"]["rationale"]


def test_task_graph_builder_can_use_live_model_for_graph_expansion(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

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
        {
            "model": "demo-model",
            "choices": [
                {
                    "message": {"content": json.dumps({"selected_channels": ["discovery", "workspace"], "rationale": ["local repo likely matters"], "channel_scores": {"workspace": 0.9, "web": 0.2, "discovery": 0.8, "risk": 0.1}})},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "model": "demo-model",
            "choices": [
                {
                    "message": {"content": json.dumps({"actions": [{"kind": "patch_scaffold", "title": "Generate Patch Scaffold", "reason": "need an actionable code change surface"}], "replan_enabled": True, "replan_focus": ["execution"], "rationale": ["patch artifact is essential"]})},
                    "finish_reason": "stop",
                }
            ],
        },
    ]

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        payload = responses.pop(0)
        return _FakeResponse(payload)

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)

    tools = ToolRegistry()
    graph = tools.call(
        ToolCall(
            name="task_graph_builder",
            tool_type=ToolType.CODE,
            args={
                "query": "Analyze my repository and prepare an executable patch plan",
                "workspace_root": str(workspace),
                "live_model": {
                    "base_url": "https://example.com/v1",
                    "api_key": "secret",
                    "model_name": "demo-model",
                },
            },
        )
    )

    assert graph.success is True
    assert graph.output["profile"]["graph_expansion"]["source"] == "live_model"
    assert any(item["kind"] == "patch_scaffold" for item in graph.output["profile"]["graph_expansion"]["actions"])
