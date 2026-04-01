"""Tests for structured response contract and dissent integration."""

from app.core.state import GraphState
from app.graph import build_graph


def test_response_contract_exists() -> None:
    graph = build_graph()
    result = graph.invoke(GraphState(query="Summarize this report and highlight main risks."))
    contract = result.get("response_contract", {})
    assert "user" in contract
    assert "debug" in contract
    assert "evaluation" in contract


def test_dissent_field_present() -> None:
    graph = build_graph()
    result = graph.invoke(
        GraphState(
            query="Review this critical compliance recommendation and challenge weak assumptions.",
            system_mode="safety_critical",
        )
    )
    assert "disagreement_triggered" in result
    assert "verification_findings" in result
