"""Tests for end-to-end graph execution."""

from app.core.state import GraphState
from app.graph import build_graph


class TestGraphE2E:
    def setup_method(self) -> None:
        self.graph = build_graph()

    def test_full_pipeline(self) -> None:
        result = self.graph.invoke(GraphState(query="Summarize and highlight risks"))
        assert result["agent_name"] != ""
        assert len(result["selected_skills"]) >= 1
        assert result["final_output"] != ""

    def test_trace_complete(self) -> None:
        result = self.graph.invoke(GraphState(query="Compare options A and B"))
        trace = result["routing_trace"]
        assert "agent_decision" in trace
        assert "skill_decision" in trace

    def test_different_queries_different_agents(self) -> None:
        r1 = self.graph.invoke(GraphState(query="Brainstorm marketing ideas"))
        r2 = self.graph.invoke(GraphState(query="Analyze the risk exposure"))
        assert r1["agent_name"] != r2["agent_name"]

    def test_personality_is_populated(self) -> None:
        result = self.graph.invoke(GraphState(query="Analyze the risk exposure"))
        assert "personality" in result and result.get("personality")

    def test_conflicts_detected_field_exists(self) -> None:
        result = self.graph.invoke(GraphState(query="Summarize and highlight risks"))
        assert "conflicts_detected" in result

    def test_consensus_result_exists(self) -> None:
        result = self.graph.invoke(GraphState(query="Compare options A and B"))
        assert "consensus_result" in result
        assert "strength" in result["consensus_result"]

    def test_trace_has_breakdown_fields(self) -> None:
        result = self.graph.invoke(
            GraphState(
                query="Audit this high-risk recommendation and challenge weak assumptions.",
                system_mode="safety_critical",
            )
        )
        trace = result["routing_trace"]
        for key in [
            "agent_candidates",
            "skill_candidates",
            "execution_timeline",
            "cost_breakdown",
            "latency_breakdown",
            "final_confidence_breakdown",
        ]:
            assert key in trace

    def test_selected_and_rejected_do_not_overlap(self) -> None:
        result = self.graph.invoke(
            GraphState(
                query="Review this critical compliance recommendation and challenge weak assumptions.",
                system_mode="safety_critical",
            )
        )
        decision = result["routing_trace"]["skill_decision"]
        selected = set(decision.get("selected", []))
        rejected = set(decision.get("rejected", []))
        assert selected.isdisjoint(rejected)
