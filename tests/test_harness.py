"""Tests for harness layer (planner/tools/memory/guardrails/eval)."""

from __future__ import annotations

from pathlib import Path

from app.harness.engine import HarnessEngine
from app.harness.guardrails import GuardrailEngine
from app.harness.models import HarnessConstraints, ToolCall, ToolType
from app.harness.planner import HarnessPlanner
from app.harness.state import HarnessMemoryStore


def test_harness_planner_builds_plan() -> None:
    planner = HarnessPlanner()
    plan = planner.build_plan("Analyze my repository and give me a fix plan plus tests")
    assert len(plan) >= 3
    assert any("workspace" in item for item in plan)
    assert any("validate" in item for item in plan)


def test_harness_planner_switches_tools_by_task_context() -> None:
    planner = HarnessPlanner()

    repo_tool = planner.next_tool_call(
        query="Analyze my repository and propose fixes",
        step=1,
        plan=planner.build_plan("Analyze my repository and propose fixes"),
    )
    research_tool = planner.next_tool_call(
        query="Write a deep research report about AI agent frameworks",
        step=1,
        plan=planner.build_plan("Write a deep research report about AI agent frameworks"),
    )

    assert repo_tool is not None
    assert repo_tool.name == "workspace_file_search"
    assert research_tool is not None
    assert research_tool.name == "external_resource_hub"


def test_harness_planner_advances_along_capability_sequence() -> None:
    planner = HarnessPlanner()
    query = "Write a deep research report about AI agent frameworks with evidence and citations"

    first_tool = planner.next_tool_call(
        query=query,
        step=1,
        plan=planner.build_plan(query),
    )
    second_tool = planner.next_tool_call(
        query=query,
        step=2,
        plan=planner.build_plan(query),
        used_tools={first_tool.name} if first_tool else set(),
        session_events=[{"tool": first_tool.name}] if first_tool else [],
    )

    assert first_tool is not None
    assert first_tool.name == "external_resource_hub"
    assert second_tool is not None
    assert second_tool.name == "evidence_dossier_builder"


def test_guardrail_blocks_known_tool() -> None:
    guardrails = GuardrailEngine()
    notes = guardrails.check_tool_call(
        tool_call=ToolCall(name="delete_path", tool_type=ToolType.CODE, args={}),
        constraints=HarnessConstraints(),
        step=1,
        high_risk=False,
    )
    assert any(note.startswith("BLOCK") for note in notes)


def test_memory_store_roundtrip(tmp_path: Path) -> None:
    store = HarnessMemoryStore(file_path=tmp_path / "harness_memory_test.json")
    store.append_event("session-a", {"event": "x"})
    recent = store.read_recent("session-a")
    assert len(recent) == 1
    assert recent[0]["event"] == "x"

    snapshot = store.snapshot()
    assert "sessions" in snapshot
    store.clear()
    assert store.read_recent("session-a") == []
    store.restore(snapshot)
    restored = store.read_recent("session-a")
    assert len(restored) == 1


def test_harness_engine_run_outputs_eval() -> None:
    engine = HarnessEngine()
    run = engine.run(
        query="Summarize this report and identify risks",
        constraints=HarnessConstraints(max_steps=3, max_tool_calls=2),
    )
    assert run.completed is True
    assert len(run.steps) <= 3
    assert "tool_success_rate" in run.eval_metrics
    assert "## Direct Answer" in run.final_answer
    assert "## Recommended Next Actions" in run.final_answer


def test_research_run_avoids_half_baked_research_brief_sections() -> None:
    engine = HarnessEngine()
    run = engine.run(
        query="Generate a deep research memo on how a general agent runtime should beat direct model answers on real tasks, including failure modes, design principles, and concrete runtime improvements.",
        constraints=HarnessConstraints(max_steps=4, max_tool_calls=4),
    )
    assert "## Improvement Path" in run.final_answer
    assert "## Working Thesis" not in run.final_answer
    assert "## Open Gaps" not in run.final_answer
    assert "Ensemble Synthesis:" not in run.final_answer
    assert "Evidence still needs to be deepened beyond initial notes." not in run.final_answer


def test_harness_eval_suite() -> None:
    engine = HarnessEngine()
    result = engine.eval_suite(["Summarize this report", "Compare options safely"])
    assert result["count"] == 2
    assert "tool_success_rate" in result["avg"]


def test_optimizer_generates_profile_driven_candidates() -> None:
    engine = HarnessEngine()
    payload = engine.optimize_query(
        query="Create a practical execution plan with risks and measurable checkpoints.",
        constraints=HarnessConstraints(max_steps=2, max_tool_calls=2),
    )

    candidates = payload.get("candidates", [])
    assert any(item.get("auto_recipe") is False and item.get("recipe") == "" for item in candidates)
    assert any(item.get("mode") == "safety_critical" and item.get("auto_recipe") is True for item in candidates)
    assert "leaderboard" in payload


def test_visual_payload_and_showcase_emphasize_delivery_state() -> None:
    engine = HarnessEngine()
    run = engine.run(
        query="Create a practical execution plan with risks and measurable checkpoints.",
        constraints=HarnessConstraints(max_steps=3, max_tool_calls=3),
    )

    visual = engine.build_visual_payload(run)
    showcase = engine.run_showcase(pack_name="security-first")

    assert visual["delivery"]["primary_deliverable"]
    assert visual["delivery"]["ready"] is True
    assert visual["first_screen_blueprint"]["hero"]["title"]
    assert showcase["comparison"]["summary"]["deliverables_ready"] >= 1
    assert "deliverables" in showcase["hero_story"][0].lower()
