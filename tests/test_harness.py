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
    plan = planner.build_plan("Audit this compliance plan and compare options")
    assert len(plan) >= 3
    assert "synthesize and evaluate" in plan


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


def test_harness_engine_run_outputs_eval() -> None:
    engine = HarnessEngine()
    run = engine.run(
        query="Summarize this report and identify risks",
        constraints=HarnessConstraints(max_steps=3, max_tool_calls=2),
    )
    assert run.completed is True
    assert len(run.steps) <= 3
    assert "tool_success_rate" in run.eval_metrics


def test_harness_eval_suite() -> None:
    engine = HarnessEngine()
    result = engine.eval_suite(["Summarize this report", "Compare options safely"])
    assert result["count"] == 2
    assert "tool_success_rate" in result["avg"]
