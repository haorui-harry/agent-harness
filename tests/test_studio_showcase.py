"""Tests for flagship studio showcase pipeline and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from urllib import error

from app.demo import PRESS_DEMO_QUERY
from app.harness import HarnessConstraints
from app.harness.engine import HarnessEngine
from app.studio.flagship import FLAGSHIP_ONE_LINER, StudioShowcaseBuilder


def test_build_showcase_payload_shape() -> None:
    builder = StudioShowcaseBuilder(harness=HarnessEngine())
    payload = builder.build_showcase(
        query="Create a practical execution plan with risks and measurable checkpoints.",
        mode="balanced",
        lab_preset="daily",
        lab_repeats=1,
        scenario_ids=["daily-001"],
        include_marketplace=False,
        include_external=False,
        include_harness_tools=False,
        include_interop_catalog=False,
    )

    assert payload["identity"]["one_liner"] == FLAGSHIP_ONE_LINER
    assert payload["schema"] == "agent-studio/v1"
    assert "story" in payload
    assert "mission" in payload
    assert "proposal" in payload
    assert "agent_comparison" in payload
    assert "theme" in payload["story"]
    assert "release_need" in payload["story"]
    assert payload["mission"]["primary_deliverable"]
    assert len(payload["mission"].get("deliverables", [])) >= 1
    assert "benchmark_targets" not in payload["mission"]
    assert payload["mission"].get("task_graph", {}).get("schema") == "agent-harness-executable-task-graph/v1"
    assert payload["mission"].get("task_graph", {}).get("summary", {}).get("node_count", 0) >= 5
    assert len(payload["story"].get("strategy_plan", [])) >= 3
    assert "harness" in payload and "plan" in payload["harness"]
    assert "final_answer_excerpt" in payload["harness"]
    assert "generation" in payload["harness"]
    assert "frontier" in payload
    assert payload["frontier"]["score"] >= 0.0
    assert payload["frontier"]["score"] <= 1.0
    assert "comparison" in payload
    assert len(payload["comparison"]["archetypes"]) >= 0
    assert "lab" in payload and "leaderboard" in payload["lab"]
    assert payload["proposal"]["scenario_name"]
    assert len(payload["proposal"].get("phases", [])) >= 3
    assert len(payload["proposal"].get("expected_impact", [])) >= 3
    assert len(payload["proposal"].get("critical_risks", [])) >= 1
    assert payload["harness"].get("delivery_brief_excerpt", "")
    assert "scenario" in payload
    assert payload["scenario"]["name"]
    assert payload["harness"]["run_summary"].get("evidence", {}).get("record_count", 0) >= 1


def test_write_showcase_with_interop_bundle(tmp_path: Path) -> None:
    builder = StudioShowcaseBuilder(harness=HarnessEngine())
    payload = builder.build_showcase(
        query="Evaluation strategy options and produce a governance-ready recommendation.",
        mode="balanced",
        lab_preset="daily",
        lab_repeats=1,
        scenario_ids=["daily-001"],
        include_marketplace=False,
        include_external=False,
        include_harness_tools=False,
        include_interop_catalog=True,
    )
    paths = builder.write_showcase(
        payload=payload,
        output_dir=str(tmp_path),
        tag="unit",
        export_interop=True,
    )

    json_path = Path(paths["json"])
    html_path = Path(paths["html"])
    deliverable_path = Path(paths["deliverable"])
    assert json_path.exists()
    assert html_path.exists()
    assert deliverable_path.exists()

    interop = paths.get("interop", {})
    assert isinstance(interop, dict)
    assert Path(str(interop.get("index", ""))).exists()

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert "catalog" not in written.get("interop", {})
    assert written.get("handoff", {}).get("primary_artifact", {}).get("path", "").endswith(".md")
    html_content = html_path.read_text(encoding="utf-8")
    assert "Agent Studio" in html_content
    assert "Primary Deliverable" in html_content
    assert "Evidence And Runtime" in html_content
    assert "Deliverable Package" in html_content
    assert "Openable Files" in html_content
    assert "Appendix" in html_content
    assert "Primary Deliverable Raw Text" in html_content
    assert "Delivery Boundary And Review" in html_content
    assert "Benchmark Fit And Boundary" not in html_content
    assert "Research Lab Leaderboard" in html_content


def test_fintech_demo_query_maps_to_scenario_with_evidence() -> None:
    builder = StudioShowcaseBuilder(harness=HarnessEngine())
    payload = builder.build_showcase(
        query=PRESS_DEMO_QUERY,
        mode="deep",
        lab_preset="daily",
        lab_repeats=1,
        scenario_ids=["daily-001"],
        include_marketplace=False,
        include_external=False,
        include_harness_tools=False,
        include_interop_catalog=False,
    )

    assert payload["scenario"]["name"]
    assert payload["proposal"]["headline"]
    assert payload["mission"]["name"]
    assert payload["harness"]["run_summary"]["evidence"]["record_count"] >= 1
    assert payload["mission"].get("task_graph", {}).get("nodes")
    brief = str(payload["harness"].get("delivery_brief_excerpt", ""))
    assert "artifact gap detected" not in brief


def test_enterprise_query_maps_to_scenario() -> None:
    builder = StudioShowcaseBuilder(harness=HarnessEngine())
    payload = builder.build_showcase(
        query="Create an enterprise workflow platform rollout and deployment plan for an internal AI operating layer with security checkpoints and business operations adoption.",
        mode="deep",
        lab_preset="daily",
        lab_repeats=1,
        scenario_ids=["enterprise-001"],
        include_marketplace=False,
        include_external=False,
        include_harness_tools=False,
        include_interop_catalog=False,
    )

    assert payload["scenario"]["name"]
    assert payload["proposal"]["headline"]


def test_research_query_maps_to_scenario() -> None:
    builder = StudioShowcaseBuilder(harness=HarnessEngine())
    payload = builder.build_showcase(
        query="Build an applied research lab operating plan with controlled experiments, paper-grade evidence, and release promotion criteria.",
        mode="deep",
        lab_preset="research",
        lab_repeats=1,
        scenario_ids=["research-001"],
        include_marketplace=False,
        include_external=False,
        include_harness_tools=False,
        include_interop_catalog=False,
    )

    assert payload["scenario"]["name"]
    assert payload["proposal"]["headline"]
    brief = str(payload["harness"].get("delivery_brief_excerpt", ""))
    assert "Deliverable Package:" in brief


def test_research_improvement_report_prefers_research_scenario() -> None:
    builder = StudioShowcaseBuilder(harness=HarnessEngine())
    payload = builder.build_showcase(
        query="Generate a deep research and improvement report for an applied research platform, covering evidence standards, system gaps, and a 12-week upgrade roadmap.",
        mode="deep",
        lab_preset="research",
        lab_repeats=1,
        scenario_ids=["research-001"],
        include_marketplace=False,
        include_external=False,
        include_harness_tools=False,
        include_interop_catalog=False,
    )

    assert payload["scenario"]["name"]


def test_live_showcase_sanitizes_endpoint_details(monkeypatch) -> None:
    builder = StudioShowcaseBuilder(harness=HarnessEngine())

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        raise error.URLError("temporary ssl eof")

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)

    payload = builder.build_showcase(
        query="Create a launch plan with live model refinement.",
        mode="deep",
        lab_preset="daily",
        lab_repeats=1,
        scenario_ids=["daily-001"],
        include_marketplace=False,
        include_external=False,
        include_harness_tools=False,
        include_interop_catalog=False,
        constraints=HarnessConstraints(enable_live_agent=True, max_live_agent_calls=4),
        live_model={"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"},
    )

    serialized = json.dumps(payload, ensure_ascii=False)
    assert "example.com" not in serialized
    assert "/chat/completions" not in serialized
