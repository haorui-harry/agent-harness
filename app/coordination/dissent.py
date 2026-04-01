"""DISSENT engine for structured disagreement injection."""

from __future__ import annotations

from app.core.state import GraphState


class DissentEngine:
    """Trigger and construct dissent findings for risky/uncertain runs."""

    dissent_keywords = {
        "challenge",
        "counter",
        "weakness",
        "漏洞",
        "反驳",
        "质疑",
        "审稿",
        "review",
    }

    def evaluate(self, state: GraphState) -> dict:
        reasons: list[str] = []

        low_consensus = float(state.consensus_result.get("strength_score", 0.0)) < 0.55
        has_conflict = len(state.conflicts_detected) > 0
        high_risk = state.risk_level in {"high", "critical"}
        query_requires_dissent = any(k in state.query.lower() for k in self.dissent_keywords)
        weak_quality = float(state.routing_metrics.get("avg_quality_score", 0.0)) < 0.55

        if low_consensus:
            reasons.append("low_consensus")
        if has_conflict:
            reasons.append("conflicts_detected")
        if high_risk:
            reasons.append("high_risk_task")
        if query_requires_dissent:
            reasons.append("query_requests_critical_review")
        if weak_quality:
            reasons.append("weak_output_quality")

        triggered = len(reasons) > 0

        challenged_claims: list[dict[str, str]] = []
        for name, text in state.skill_outputs.items():
            lowered = text.lower()
            if "risk" in lowered or "recommend" in lowered or "claim" in lowered:
                challenged_claims.append(
                    {
                        "skill": name,
                        "claim": text.splitlines()[0][:180],
                        "challenge": "Need additional evidence or cross-check before acceptance.",
                    }
                )

        alternatives = []
        if "validate_claims" not in state.selected_skills:
            alternatives.append("Add validate_claims for stronger evidence verification")
        if "detect_anomalies" not in state.selected_skills:
            alternatives.append("Add detect_anomalies to probe contradictions")
        if "synthesize_perspectives" not in state.selected_skills:
            alternatives.append("Add synthesize_perspectives to test alternative interpretations")

        needs_human = high_risk and (low_consensus or has_conflict)

        return {
            "triggered": triggered,
            "reasons": reasons,
            "challenged_claims": challenged_claims[:5],
            "alternative_paths": alternatives,
            "needs_human_review": needs_human,
            "summary": (
                "DISSENT branch triggered for structured challenge."
                if triggered
                else "No dissent branch required."
            ),
        }
