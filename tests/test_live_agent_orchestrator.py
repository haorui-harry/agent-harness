"""Tests for live-agent orchestration quality loop."""

from __future__ import annotations

import json

from app.harness.live_agent import LiveAgentOrchestrator


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_live_agent_uses_evidence_and_revision(monkeypatch) -> None:
    orchestrator = LiveAgentOrchestrator()
    seen_payloads: list[dict[str, object]] = []
    call_count = {"value": 0}

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        call_count["value"] += 1
        body = json.loads(req.data.decode("utf-8"))
        seen_payloads.append(body)
        if call_count["value"] == 1:
            content = {
                "thesis": "base thesis",
                "findings": ["need stronger benchmark protocol"],
                "improvement_roadmap": ["standardize manifests"],
            }
            return _FakeResponse({"model": "demo-model", "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}]})
        if call_count["value"] == 2:
            return _FakeResponse(
                {
                    "model": "demo-model",
                    "choices": [{"message": {"content": "Draft answer without enough evidence."}, "finish_reason": "stop"}],
                }
            )
        if call_count["value"] == 3:
            critique = {
                "confidence": 0.58,
                "blind_spots": ["missing evidence grounding"],
                "red_flags": ["too generic"],
                "improve": ["cite concrete benchmark sources"],
            }
            return _FakeResponse({"model": "demo-model", "choices": [{"message": {"content": json.dumps(critique)}, "finish_reason": "stop"}]})
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Revised answer with benchmark specificity.\n\n"
                                "Sources\n"
                                "- https://github.com/SWE-bench/SWE-bench"
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)

    result = orchestrator.enhance(
        query="Write a deep research report for agent-harness benchmark strategy.",
        mode="deep",
        base_answer="base answer",
        plan=["collect evidence", "write report"],
        steps=[{"step": 1, "tool_call": {"name": "external_resource_hub"}, "tool_result": {"success": True}}],
        discovery=[{"name": "external_resource_hub", "score": 0.9}],
        evidence={
            "record_count": 1,
            "citation_count": 1,
            "citations": ["https://github.com/SWE-bench/SWE-bench"],
            "records": [
                {
                    "title": "SWE-bench",
                    "summary": "Benchmark for software engineering agents.",
                    "content": "Verifiable GitHub issue benchmark.",
                    "source_id": "built_in_catalog",
                    "url": "https://github.com/SWE-bench/SWE-bench",
                }
            ],
        },
        max_calls=6,
        live_model_overrides={"base_url": "https://example.com/v1", "api_key": "secret", "model_name": "demo-model"},
    )

    assert result.success is True
    assert result.calls_used == 4
    assert "Revised answer with benchmark specificity." in result.enhanced_answer
    revision_prompt = json.dumps(seen_payloads[-1], ensure_ascii=False)
    assert "SWE-bench" in revision_prompt
    assert "evidence_digest" in revision_prompt
