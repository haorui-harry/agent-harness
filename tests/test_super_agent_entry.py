"""Tests for thread-first super-agent entry and skill package catalog."""

from __future__ import annotations

from pathlib import Path

from app.harness.engine import HarnessEngine
from app.harness.runtime_settings import HarnessRuntimeSettings
from app.harness.task_profile import analyze_task_request
from app.harness.tools import ToolRegistry
from app.harness.models import ToolCall, ToolType
from app.skills.packages import SkillPackageCatalog


def test_skill_package_catalog_loads_disk_and_registry_packages() -> None:
    catalog = SkillPackageCatalog()

    deep_research = catalog.get_package("deep-research")
    code_mission = catalog.get_package("code-mission")
    builtin = catalog.get_package("identify_risks")
    suggested = catalog.suggest("research a topic with evidence and web sources", target="research", limit=3)

    assert deep_research is not None
    assert deep_research.source == "public"
    assert "external_resource_hub" in deep_research.tool_refs
    assert code_mission is not None
    assert "patch" in " ".join(code_mission.artifact_kinds)
    assert builtin is not None
    assert builtin.source == "builtin"
    assert any(item.name == "deep-research" for item in suggested)


def test_skill_package_catalog_falls_back_to_general_packages_for_vague_queries() -> None:
    catalog = SkillPackageCatalog()

    suggested = catalog.suggest("help me figure out the best way to approach this task", target="general", limit=4)
    names = [item.name for item in suggested]

    assert suggested
    assert "general-purpose" in names
    assert "find-skills" in names


def test_code_skill_search_merges_package_and_builtin_results() -> None:
    registry = ToolRegistry()

    result = registry.call(
        ToolCall(
            name="code_skill_search",
            tool_type=ToolType.CODE,
            args={"query": "research evidence report", "target": "research", "limit": 8},
        )
    )

    assert result.success is True
    skills = result.output.get("skills", [])
    assert any(item.get("tier") == "package" for item in skills)
    assert any(item.get("source") == "builtin" for item in skills)


def test_thread_first_super_agent_runs_and_records_packages(tmp_path: Path) -> None:
    settings = HarnessRuntimeSettings(
        threads_root=tmp_path / "threads",
        memory_path=tmp_path / "memory.json",
    )
    engine = HarnessEngine(settings=settings)

    thread = engine.create_thread(title="Thread First")
    sandbox = engine.thread_runtime.sandbox_provider.get((tmp_path / "threads" / thread["thread_id"]).resolve())
    sandbox.write_text("notes.md", "inspect code, draft patch, and validate", area="workspace")
    sandbox.write_text("tests/test_demo.py", "def test_demo():\n    assert True\n", area="workspace")

    payload = engine.run_thread_first(
        thread["thread_id"],
        "Inspect the workspace, create a patch draft, and validate the result.",
        target="auto",
    )

    assert payload["schema"] == "agent-harness-thread-super-agent/v1"
    assert payload["route"]["kind"] == "task_graph"
    assert payload["execution"]["status"] == "completed"
    assert any(item["name"] == "code-mission" for item in payload["packages"])

    persisted = engine.get_thread(thread["thread_id"])
    assert persisted is not None
    assert persisted["messages"][-1]["metadata"]["entrypoint"] == "thread-first-super-agent"
    assert any(event["event"] == "super_agent_planned" for event in persisted["events"])


def test_thread_first_super_agent_keeps_general_target_for_ambiguous_requests(tmp_path: Path) -> None:
    settings = HarnessRuntimeSettings(
        threads_root=tmp_path / "threads",
        memory_path=tmp_path / "memory.json",
    )
    engine = HarnessEngine(settings=settings)

    thread = engine.create_thread(title="General Routing")
    payload = engine.run_thread_first(
        thread["thread_id"],
        "Investigate launch options, discover relevant skills, and prepare a polished deliverable for stakeholders.",
        target="auto",
    )

    assert payload["route"]["target"] == "general"
    assert any(item["name"] == "general-purpose" for item in payload["packages"])


def test_analyze_task_request_keeps_discovery_and_allows_mixed_channels() -> None:
    profile = analyze_task_request(
        "Inspect my repository, compare it with the latest public benchmarks, and prepare a recommendation deck."
    )

    assert profile.requires_discovery is True
    assert "discovery" in profile.deliberation.selected
    assert "workspace" in profile.deliberation.selected
    assert "web" in profile.deliberation.selected


def test_runtime_settings_centralize_thread_memory_and_sandbox_paths(tmp_path: Path) -> None:
    settings = HarnessRuntimeSettings(
        threads_root=tmp_path / "central_threads",
        memory_path=tmp_path / "state" / "memory.json",
    )
    engine = HarnessEngine(settings=settings)

    thread = engine.create_thread(title="Centralized Runtime")

    assert engine.thread_runtime.root == (tmp_path / "central_threads")
    assert engine.memory._file == (tmp_path / "state" / "memory.json")
    assert Path(thread["workspace"]["workspace"]).exists()
