"""Tests for deep research report generation."""

from __future__ import annotations

from pathlib import Path

from app.harness.deep_research import HarnessDeepResearchBuilder


def test_deep_research_builder_writes_bundle(tmp_path: Path) -> None:
    subject = tmp_path / "subject"
    competitor = tmp_path / "competitor"
    for path in [
        subject / "app" / "agents",
        subject / "app" / "harness",
        subject / "app" / "skills",
        subject / "app" / "benchmark",
        subject / "tests",
        competitor / "skills" / "public" / "deep-research",
        competitor / "skills" / "public" / "consulting-analysis",
        competitor / "frontend",
        competitor / "backend" / "tests",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    (subject / "app" / "agents" / "runtime.py").write_text(
        "class AgentThreadRuntime:\n"
        "    def resume_execution(self): ...\n"
        "    def retry_execution(self): ...\n"
        "    def request_interrupt(self): ...\n",
        encoding="utf-8",
    )
    (subject / "app" / "agents" / "scheduler.py").write_text(
        "def recover_execution(): ...\n"
        "def recover_all(): ...\n",
        encoding="utf-8",
    )
    (subject / "app" / "agents" / "workspace_view.py").write_text("# workspace view\n", encoding="utf-8")
    (subject / "app" / "harness" / "engine.py").write_text("def execute_thread_generic_task(): ...\n", encoding="utf-8")
    (subject / "app" / "harness" / "tools.py").write_text(
        "workspace_file_search = 1\nworkspace_file_read = 1\nworkspace_file_write = 1\ntask_graph_builder = 1\n",
        encoding="utf-8",
    )
    (subject / "app" / "harness" / "evidence.py").write_text("# evidence\n", encoding="utf-8")
    (subject / "app" / "harness" / "deep_research.py").write_text("# self marker\n", encoding="utf-8")
    (subject / "app" / "skills" / "interop.py").write_text("# interop\n", encoding="utf-8")
    (subject / "app" / "benchmark" / "adapters.py").write_text("# adapters\n", encoding="utf-8")
    (subject / "app" / "main.py").write_text('@app.command("demo")\n', encoding="utf-8")
    (subject / "tests" / "test_demo.py").write_text("def test_demo():\n    assert True\n", encoding="utf-8")

    (competitor / "README.md").write_text(
        "Super agent harness with sub-agent, memory, sandbox, MCP, official website and install skill flows.\n",
        encoding="utf-8",
    )
    (competitor / "frontend" / "CLAUDE.md").write_text("workspace artifacts memory skills\n", encoding="utf-8")
    (competitor / "backend" / "CLAUDE.md").write_text("sandbox mcp memory subagents\n", encoding="utf-8")
    (competitor / "backend" / "tests" / "test_demo.py").write_text("def test_demo():\n    assert True\n", encoding="utf-8")

    builder = HarnessDeepResearchBuilder()
    payload = builder.build(
        topic="Generate a deep research and improvement report for AI agent harness.",
        subject_root=subject,
        competitor_root=competitor,
        subject_name="agent-harness",
        competitor_name="deer-flow",
    )
    paths = builder.write_bundle(payload, output_dir=tmp_path / "reports")

    assert payload["schema"] == "agent-harness-deep-research/v1"
    assert "DeerFlow" in payload["report_markdown"]
    assert len(payload["dimensions"]) >= 5
    assert Path(paths["framework"]).exists()
    assert Path(paths["report"]).exists()
    assert Path(paths["bundle"]).exists()
    assert Path(paths["html"]).exists()
