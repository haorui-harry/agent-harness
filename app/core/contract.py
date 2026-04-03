"""Response contract builder for structured output packages."""

from __future__ import annotations

from typing import Any

from app.core.mission import MissionRegistry
from app.core.state import GraphState


def _extract_key_findings(state: GraphState) -> list[str]:
    findings: list[str] = []
    for name, text in state.skill_outputs.items():
        for line in text.splitlines():
            lowered = line.lower()
            if any(k in lowered for k in ["risk", "critical", "recommend", "evidence", "claim"]):
                findings.append(f"[{name}] {line.strip()}")
            if len(findings) >= 6:
                return findings
    return findings


def _build_confidence_components(state: GraphState) -> dict[str, float]:
    redundancy = float(state.routing_metrics.get("redundancy", 0.0))
    coverage = float(state.routing_metrics.get("coverage", 0.0))
    avg_quality = float(state.routing_metrics.get("avg_quality_score", 0.0))
    agreement = float(state.consensus_result.get("agreement_ratio", 0.0))
    worst_case = float(state.routing_metrics.get("robust_worst_case_utility", 0.0))
    avg_uncertainty = float(state.routing_metrics.get("avg_uncertainty", 0.0))
    conflict_count = float(state.routing_metrics.get("conflict_count", 0.0))
    routing_signal = max(
        0.0,
        min(
            1.0,
            0.35 * max(0.0, 1.0 - redundancy)
            + 0.20 * coverage
            + 0.15 * avg_quality
            + 0.15 * agreement
            + 0.15 * max(0.0, min(1.0, worst_case * (1.0 - avg_uncertainty))),
        ),
    )
    return {
        "routing_confidence": routing_signal,
        "portfolio_confidence": max(0.0, min(1.0, 0.55 * coverage + 0.45 * agreement)),
        "evidence_confidence": agreement,
        "consistency_confidence": max(0.0, 1.0 - conflict_count / 3.0),
        "verification_confidence": float(state.consensus_result.get("strength_score", 0.0)),
        "calibration_adjusted_confidence": max(0.0, min(1.0, 0.65 * avg_quality + 0.35 * agreement)),
        "robustness_confidence": max(
            0.0,
            min(
                1.0,
                worst_case
                * (1.0 - avg_uncertainty),
            ),
        ),
    }


def build_response_contract(state: GraphState) -> dict[str, Any]:
    """Build structured response contract from execution state."""

    components = _build_confidence_components(state)
    overall_conf = sum(components.values()) / max(len(components), 1)
    mission = MissionRegistry().infer(state.query).to_dict()

    agent_trace = state.routing_trace.get("agent_decision", {})
    agent_scores = agent_trace.get("scores", {})
    ranked = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
    route_regret = 0.0
    if len(ranked) >= 2:
        route_regret = float(ranked[0][1]) - float(ranked[1][1])

    key_findings = _extract_key_findings(state)
    route_signature = (
        f"{state.system_mode}:{state.agent_name}:{len(state.selected_skills)}skills:"
        f"{'dissent' if state.disagreement_triggered else 'direct'}"
    )
    cost_breakdown = state.routing_trace.get("cost_breakdown", {})
    latency_breakdown = state.routing_trace.get("latency_breakdown", {})

    contract = {
        "user": {
            "final_answer": state.final_output,
            "executive_summary": state.final_output[:480],
            "confidence_summary": {
                "overall": round(overall_conf, 3),
                "strength": state.consensus_result.get("strength", "unknown"),
                "requires_human_review": state.approval_required,
            },
            "key_risks_or_findings": key_findings,
            "next_steps": [
                "Review dissent findings if present.",
                "Validate high-impact conclusions with additional evidence.",
                "Escalate to human review for critical decisions.",
            ],
            "delivery_contract": {
                "mission_type": mission.get("title", ""),
                "primary_deliverable": mission.get("primary_deliverable", ""),
                "output_views": mission.get("output_views", []),
            },
            "trace_summary": {
                "trace_id": state.trace_id,
                "events": len(state.reasoning_path),
                "disagreement_triggered": state.disagreement_triggered,
                "route_signature": route_signature,
            },
        },
        "debug": {
            "selected_agent": state.agent_name,
            "agent_candidates": state.routing_trace.get("agent_scores", agent_trace.get("scores", {})),
            "selected_skills": state.selected_skills,
            "rejected_skills": state.routing_trace.get("skill_decision", {}).get("rejected", []),
            "portfolio_plan": state.portfolio_plan,
            "verification_findings": state.verification_findings,
            "cost_latency_profile": {
                "cost_breakdown": cost_breakdown,
                "latency_breakdown": latency_breakdown,
                "total_budget_used": cost_breakdown.get("total", state.routing_metrics.get("total_budget_used", 0.0)),
                "total_execution_ms": latency_breakdown.get("total", state.routing_metrics.get("total_execution_ms", 0.0)),
            },
            "state_timeline": state.reasoning_path,
            "full_trace_id": state.trace_id,
            "route_signature": route_signature,
        },
        "evaluation": {
            "route_regret_estimate": round(route_regret, 4),
            "coverage_score": float(state.routing_metrics.get("coverage", 0.0)),
            "redundancy_score": float(state.routing_metrics.get("redundancy", 0.0)),
            "robust_expected_utility": float(state.routing_metrics.get("robust_expected_utility", 0.0)),
            "robust_worst_case_utility": float(state.routing_metrics.get("robust_worst_case_utility", 0.0)),
            "avg_uncertainty": float(state.routing_metrics.get("avg_uncertainty", 0.0)),
            "disagreement_triggered": state.disagreement_triggered,
            "approval_required": state.approval_required,
            "confidence_components": components,
        },
    }

    return contract
