"""Skill executor with runtime contexts, retries, and synthesis metrics."""

from __future__ import annotations

import time
from typing import Any

from app.coordination.conflicts import ConflictDetector
from app.coordination.consensus import ConsensusBuilder
from app.core.contract import build_response_contract
from app.core.state import GraphState
from app.ecosystem.reputation import record_marketplace_outcome
from app.memory.learning import record_run
from app.skills.registry import execute_skill, get_skill_metadata


def execute_skills(state: GraphState) -> GraphState:
    """LangGraph node: execute selected skills and store execution contexts."""

    outputs: dict[str, str] = {}
    contexts: dict[str, dict[str, Any]] = {}

    order = state.execution_order or state.selected_skills
    for skill_name in order:
        context = _execute_single_skill(skill_name, state.query)
        outputs[skill_name] = str(context.get("output", ""))
        contexts[skill_name] = context

        if not context.get("success", False) and context.get("retry_count", 0) < 1:
            retry_context = _execute_single_skill(skill_name, state.query, retry=True)
            if retry_context.get("success", False):
                outputs[skill_name] = str(retry_context.get("output", ""))
                contexts[skill_name] = retry_context

        record_marketplace_outcome(
            skill_name=skill_name,
            success=bool(contexts[skill_name].get("success", False)),
        )

    state.skill_outputs = outputs
    state.execution_contexts = contexts
    timeline = _build_execution_timeline(contexts)
    state.routing_trace["execution_timeline"] = timeline
    state.routing_trace["latency_breakdown"] = _latency_breakdown(contexts)
    state.routing_trace.setdefault("state_snapshots", []).append(
        {
            "phase": "executor",
            "executed_skills": list(outputs.keys()),
            "timeline_points": len(timeline),
        }
    )
    state.reasoning_path.append(
        {
            "step": len(state.reasoning_path) + 1,
            "event": "execution_completed",
            "elapsed_ms": 8.0,
            "description": "Skill execution completed",
            "data": {
                "skills": list(outputs.keys()),
                "durations_ms": {k: round(v.get("duration_ms", 0.0), 2) for k, v in contexts.items()},
            },
        }
    )
    return state


def _execute_single_skill(skill_name: str, query: str, retry: bool = False) -> dict[str, Any]:
    """Execute one skill and return execution context payload."""

    start = time.time()
    try:
        output = execute_skill(skill_name, query)
        success = not output.startswith("[ERROR]")
        quality = _estimate_output_quality(output)
        error_message = ""
    except Exception as exc:  # pragma: no cover - defensive
        output = f"[ERROR] {skill_name}: {exc}"
        success = False
        quality = 0.0
        error_message = str(exc)

    end = time.time()
    return {
        "skill_name": skill_name,
        "output": output,
        "success": success,
        "start_time": start,
        "end_time": end,
        "duration_ms": (end - start) * 1000.0,
        "retry_count": 1 if retry else 0,
        "output_length": len(output),
        "quality_score": quality,
        "error_message": error_message,
    }


def _estimate_output_quality(output: str) -> float:
    """Heuristic output quality estimation (0-1)."""

    if not output or output.startswith("[ERROR]"):
        return 0.0

    length = len(output)
    if length < 20:
        length_score = 0.2
    elif length < 50:
        length_score = 0.5
    elif length <= 500:
        length_score = 1.0
    elif length <= 1000:
        length_score = 0.8
    else:
        length_score = 0.6

    indicators = [":\n", "- ", "1.", "2.", "| ", "Claim", "Evidence", "Priority", "Timeline"]
    structure_hits = sum(1 for marker in indicators if marker in output)
    structure_score = min(structure_hits / 3.0, 1.0)

    return 0.6 * length_score + 0.4 * structure_score


def aggregate_outputs(state: GraphState) -> GraphState:
    """LangGraph node: merge all skill outputs into final output text."""

    if not state.skill_outputs:
        state.final_output = "[No skills were selected - unable to produce output]"
        return state

    sections: list[str] = []
    for skill_name, output in state.skill_outputs.items():
        sections.append(f"[{skill_name}]\n{output}")

    conflicts = state.conflicts_detected
    if not conflicts:
        conflicts = ConflictDetector().detect(state.skill_outputs)
        state.conflicts_detected = conflicts

    consensus = state.consensus_result
    if not consensus:
        consensus = ConsensusBuilder().build(state.skill_outputs, conflicts)
        state.consensus_result = consensus

    synthesis = _enhanced_ensemble_reasoning(
        query=state.query,
        outputs=state.skill_outputs,
        contexts=state.execution_contexts,
        conflicts=conflicts,
        consensus=consensus,
        dissent=state.verification_findings,
    )
    sections.append(synthesis)

    state.routing_metrics = _compute_routing_metrics(state)
    if state.disagreement_triggered and state.verification_findings:
        sections.append(
            "DISSENT Findings:\n"
            f"- Reasons: {', '.join(state.verification_findings[0].get('reasons', []))}\n"
            f"- Human review required: {state.verification_findings[0].get('needs_human_review', False)}"
        )
    state.final_output = "\n\n".join(sections)
    state.response_contract = build_response_contract(state)
    state.routing_trace["verifier_findings"] = list(state.verification_findings)
    state.routing_trace["cost_breakdown"] = _cost_breakdown(state.selected_skills)
    state.routing_trace["latency_breakdown"] = _latency_breakdown(state.execution_contexts)
    state.routing_trace["final_confidence_breakdown"] = (
        state.response_contract.get("evaluation", {}).get("confidence_components", {})
    )
    if state.approval_required or state.disagreement_triggered:
        state.routing_trace.setdefault("interrupt_events", []).append(
            {
                "type": "human_review_gate" if state.approval_required else "dissent_review_gate",
                "approval_required": state.approval_required,
                "disagreement_triggered": state.disagreement_triggered,
            }
        )
    state.routing_trace.setdefault("state_snapshots", []).append(
        {
            "phase": "aggregate",
            "conflicts": len(state.conflicts_detected),
            "consensus": state.consensus_result.get("strength", "unknown"),
            "approval_required": state.approval_required,
        }
    )
    record_run(state)
    state.reasoning_path.append(
        {
            "step": len(state.reasoning_path) + 1,
            "event": "ensemble_synthesized",
            "elapsed_ms": 12.0,
            "description": "Final ensemble synthesis built",
            "data": {
                "conflicts": len(conflicts),
                "consensus": consensus.get("strength", "unknown"),
                "metrics": state.routing_metrics,
            },
        }
    )
    return state


def _enhanced_ensemble_reasoning(
    query: str,
    outputs: dict[str, str],
    contexts: dict[str, dict[str, Any]],
    conflicts: list[dict[str, Any]],
    consensus: dict[str, Any],
    dissent: list[dict[str, Any]],
) -> str:
    """Build synthesis section with quality, conflict, and consensus details."""

    skill_names = ", ".join(outputs.keys())

    quality_scores = [float(ctx.get("quality_score", 0.0)) for ctx in contexts.values()]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    conflict_summary = f"{len(conflicts)} conflicts detected" if conflicts else "no conflicts"
    consensus_strength = consensus.get("strength", "unknown")

    total_ms = sum(float(ctx.get("duration_ms", 0.0)) for ctx in contexts.values())
    risk_signals = sum(1 for text in outputs.values() if "risk" in text.lower())
    dissent_triggered = bool(dissent and dissent[0].get("triggered"))
    confidence = (
        "high"
        if len(outputs) >= 3 and risk_signals >= 1 and not conflicts and not dissent_triggered
        else "medium"
    )

    return (
        "Ensemble Synthesis:\n"
        f"- Query: {query}\n"
        f"- Contributing skills: {skill_names}\n"
        f"- Cross-skill agreement signals: {risk_signals}\n"
        f"- Conflicts: {conflict_summary}\n"
        f"- Consensus strength: {consensus_strength}\n"
        f"- DISSENT triggered: {dissent_triggered}\n"
        f"- Average output quality: {avg_quality:.2f}\n"
        f"- Total execution time: {total_ms:.1f}ms\n"
        f"- Ensemble confidence: {confidence}"
    )


def _compute_routing_metrics(state: GraphState) -> dict[str, float]:
    """Compute aggregate routing quality metrics from trace + execution."""

    metrics: dict[str, float] = {}

    skill_trace = state.routing_trace.get("skill_decision", {})
    comp_metrics = skill_trace.get("complementarity_metrics", {})
    for key, value in comp_metrics.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    contexts = state.execution_contexts
    if contexts:
        total_ms = sum(float(ctx.get("duration_ms", 0.0)) for ctx in contexts.values())
        avg_quality = sum(float(ctx.get("quality_score", 0.0)) for ctx in contexts.values()) / len(contexts)
        success_rate = sum(1 for ctx in contexts.values() if ctx.get("success")) / len(contexts)
        metrics["total_execution_ms"] = total_ms
        metrics["avg_quality_score"] = avg_quality
        metrics["skill_success_rate"] = success_rate

    metrics["conflict_count"] = float(len(state.conflicts_detected))
    metrics["consensus_strength"] = float(state.consensus_result.get("strength_score", 0.0))
    metrics["disagreement_triggered"] = 1.0 if state.disagreement_triggered else 0.0
    return metrics


def _build_execution_timeline(contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build execution timeline entries sorted by start time."""

    timeline = []
    for name, context in contexts.items():
        timeline.append(
            {
                "skill": name,
                "start_time": float(context.get("start_time", 0.0)),
                "end_time": float(context.get("end_time", 0.0)),
                "duration_ms": float(context.get("duration_ms", 0.0)),
                "success": bool(context.get("success", False)),
            }
        )
    timeline.sort(key=lambda row: row["start_time"])
    return timeline


def _cost_breakdown(selected_skills: list[str]) -> dict[str, float]:
    """Estimate cost profile from skill metadata compute costs."""

    per_skill: dict[str, float] = {}
    for name in selected_skills:
        meta = get_skill_metadata(name)
        per_skill[name] = float(meta.compute_cost) if meta else 1.0
    total = sum(per_skill.values())
    per_skill["total"] = total
    return per_skill


def _latency_breakdown(contexts: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Extract latency profile from execution contexts."""

    per_skill = {name: float(ctx.get("duration_ms", 0.0)) for name, ctx in contexts.items()}
    per_skill["total"] = sum(per_skill.values())
    return per_skill
