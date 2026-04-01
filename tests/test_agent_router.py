"""Tests for the agent router."""

from app.core.state import GraphState
from app.routing.agent_router import route_to_agent


class TestAgentRouter:
    def test_routes_risk_query_to_analysis(self) -> None:
        state = GraphState(query="Analyze the risks in this quarterly report")
        result = route_to_agent(state)
        assert result.agent_name == "AnalysisAgent"

    def test_routes_summary_query(self) -> None:
        state = GraphState(query="Give me a brief summary of the document")
        result = route_to_agent(state)
        assert result.agent_name == "SummaryAgent"

    def test_routes_creative_query(self) -> None:
        state = GraphState(query="Brainstorm innovative ideas for the product")
        result = route_to_agent(state)
        assert result.agent_name == "CreativeAgent"

    def test_routes_advice_query(self) -> None:
        state = GraphState(query="What strategy should we recommend for Q3?")
        result = route_to_agent(state)
        assert result.agent_name == "AdvisorAgent"

    def test_trace_is_populated(self) -> None:
        state = GraphState(query="summarize this")
        result = route_to_agent(state)
        assert "agent_decision" in result.routing_trace
        assert "selected" in result.routing_trace["agent_decision"]
        assert "intent_signals" in result.routing_trace["agent_decision"]

    def test_routes_research_query(self) -> None:
        state = GraphState(query="Investigate the evidence behind this claim")
        result = route_to_agent(state)
        assert result.agent_name == "ResearchAgent"

    def test_query_complexity_estimation(self) -> None:
        state = GraphState(
            query="Analyze risks and compare options and recommend a plan and build a timeline"
        )
        result = route_to_agent(state)
        assert result.query_complexity in ("complex", "expert")

    def test_collaboration_detection(self) -> None:
        state = GraphState(query="Analyze the data and brainstorm solutions and plan the implementation")
        result = route_to_agent(state)
        assert len(result.detected_intents) >= 2
