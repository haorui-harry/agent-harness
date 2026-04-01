"""LangGraph workflow definition for the skill-router system."""

from __future__ import annotations

from typing import Callable

try:
    from langgraph.graph import END, START, StateGraph

    HAS_LANGGRAPH = True
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight envs
    END = "__end__"
    START = "__start__"
    StateGraph = None
    HAS_LANGGRAPH = False

from app.coordination.dissent import DissentEngine
from app.coordination.conflicts import ConflictDetector
from app.coordination.consensus import ConsensusBuilder
from app.core.state import GraphState, QueryComplexity, personality_from_dict
from app.policy.center import infer_risk_level, normalize_mode
from app.personality.adaptation import PersonalityAdaptor
from app.routing.agent_router import route_to_agent
from app.routing.executor import aggregate_outputs, execute_skills
from app.routing.skill_router import route_to_skills


def _query_understanding_node(state: GraphState) -> GraphState:
    """LangGraph node: perform lightweight query understanding pre-pass."""

    mode = normalize_mode(state.system_mode)
    risk = infer_risk_level(state.query)
    state.system_mode = mode.value
    state.risk_level = risk.value

    if not state.query_state:
        state.query_state = {
            "query_raw": state.query,
            "domain_tags": [],
            "risk_level": state.risk_level,
            "routing_goals": ["accuracy", "traceability", "complementarity"],
            "approval_policy": "auto",
        }

    if mode.value == "fast":
        state.max_skills = min(state.max_skills, 2)

    state.reasoning_path.append(
        {
            "step": len(state.reasoning_path) + 1,
            "event": "query_received",
            "elapsed_ms": 0.2,
            "description": "Query understanding initialized",
            "data": {
                "mode": state.system_mode,
                "risk_level": state.risk_level,
                "max_skills": state.max_skills,
            },
        }
    )
    state.routing_trace.setdefault("state_snapshots", []).append(
        {
            "phase": "query_understanding",
            "mode": state.system_mode,
            "risk_level": state.risk_level,
            "max_skills": state.max_skills,
        }
    )
    return state


def _adapt_personality_node(state: GraphState) -> GraphState:
    """LangGraph node: adapt personality according to query complexity."""

    if not state.personality:
        return state

    adaptor = PersonalityAdaptor()
    original = personality_from_dict(state.personality)
    complexity = QueryComplexity(state.query_complexity)

    adapted = adaptor.adapt_for_complexity(original, complexity)

    state.personality = {
        "risk_tolerance": adapted.risk_tolerance,
        "creativity_bias": adapted.creativity_bias,
        "diversity_preference": adapted.diversity_preference,
        "confidence_threshold": adapted.confidence_threshold,
        "collaboration_tendency": adapted.collaboration_tendency,
        "depth_vs_breadth": adapted.depth_vs_breadth,
    }
    state.personality_adjustments = {
        "adaptations": adaptor.describe_adaptations(original, adapted),
        "reason": f"adapted for {complexity.value} complexity",
    }
    state.reasoning_path.append(
        {
            "step": len(state.reasoning_path) + 1,
            "event": "personality_adapted",
            "elapsed_ms": 2.0,
            "description": "Personality adapted for query complexity",
            "data": state.personality_adjustments,
        }
    )
    return state


def _detect_conflicts_node(state: GraphState) -> GraphState:
    """LangGraph node: detect conflicts across skill outputs."""

    detector = ConflictDetector()
    state.conflicts_detected = detector.detect(state.skill_outputs)
    state.reasoning_path.append(
        {
            "step": len(state.reasoning_path) + 1,
            "event": "conflict_detected",
            "elapsed_ms": 9.0,
            "description": "Conflict scan finished",
            "data": {"count": len(state.conflicts_detected)},
        }
    )
    return state


def _build_consensus_node(state: GraphState) -> GraphState:
    """LangGraph node: build consensus report from skill outputs."""

    builder = ConsensusBuilder()
    state.consensus_result = builder.build(state.skill_outputs, state.conflicts_detected)
    state.reasoning_path.append(
        {
            "step": len(state.reasoning_path) + 1,
            "event": "consensus_built",
            "elapsed_ms": 10.0,
            "description": "Consensus report generated",
            "data": {
                "strength": state.consensus_result.get("strength", "unknown"),
                "agreement_ratio": state.consensus_result.get("agreement_ratio", 0.0),
            },
        }
    )
    return state


def _dissent_node(state: GraphState) -> GraphState:
    """LangGraph node: run structured dissent check when needed."""

    engine = DissentEngine()
    result = engine.evaluate(state)
    state.disagreement_triggered = bool(result.get("triggered", False))
    state.verification_findings = [result]
    if result.get("needs_human_review"):
        state.approval_required = True
    state.reasoning_path.append(
        {
            "step": len(state.reasoning_path) + 1,
            "event": "dissent_evaluated",
            "elapsed_ms": 11.0,
            "description": "Structured dissent evaluation completed",
            "data": {
                "triggered": state.disagreement_triggered,
                "reasons": result.get("reasons", []),
                "needs_human_review": result.get("needs_human_review", False),
            },
        }
    )
    state.routing_trace["verifier_findings"] = list(state.verification_findings)
    if state.approval_required:
        state.routing_trace.setdefault("interrupt_events", []).append(
            {
                "type": "approval_required",
                "reason": "dissent_or_risk_policy",
            }
        )
    return state


class _FallbackCompiledGraph:
    """Sequential graph fallback used when langgraph is unavailable."""

    def __init__(self, nodes: list[Callable[[GraphState], GraphState]]) -> None:
        self._nodes = nodes

    def invoke(self, state: GraphState | dict) -> dict:
        current = state if isinstance(state, GraphState) else GraphState(**state)
        for node in self._nodes:
            current = node(current)
        return current.model_dump()


def build_graph():
    """Construct and compile the upgraded routing graph."""

    if HAS_LANGGRAPH:
        graph = StateGraph(GraphState)

        graph.add_node("query_understanding", _query_understanding_node)
        graph.add_node("route_agent", route_to_agent)
        graph.add_node("adapt_personality", _adapt_personality_node)
        graph.add_node("route_skills", route_to_skills)
        graph.add_node("execute", execute_skills)
        graph.add_node("detect_conflicts", _detect_conflicts_node)
        graph.add_node("build_consensus", _build_consensus_node)
        graph.add_node("dissent", _dissent_node)
        graph.add_node("aggregate", aggregate_outputs)

        graph.add_edge(START, "query_understanding")
        graph.add_edge("query_understanding", "route_agent")
        graph.add_edge("route_agent", "adapt_personality")
        graph.add_edge("adapt_personality", "route_skills")
        graph.add_edge("route_skills", "execute")
        graph.add_edge("execute", "detect_conflicts")
        graph.add_edge("detect_conflicts", "build_consensus")
        graph.add_edge("build_consensus", "dissent")
        graph.add_edge("dissent", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()

    return _FallbackCompiledGraph(
        [
            _query_understanding_node,
            route_to_agent,
            _adapt_personality_node,
            route_to_skills,
            execute_skills,
            _detect_conflicts_node,
            _build_consensus_node,
            _dissent_node,
            aggregate_outputs,
        ]
    )
