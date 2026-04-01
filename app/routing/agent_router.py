"""Agent router with complexity estimation and collaboration detection."""

from __future__ import annotations

from app.agents.definitions import AgentProfile, get_agent, list_agents
from app.core.state import GraphState, QueryComplexity, QueryState, RoutingDecision, personality_to_dict
from app.policy.center import RiskLevel, infer_risk_level, normalize_mode, policy_for_mode


def _intent_signals(query: str) -> dict[str, float]:
    """Extract soft intent signals from the query."""

    q = query.lower()
    signals: dict[str, float] = {}

    intent_map = {
        "analysis": ["analyze", "assess", "evaluate", "examine", "investigate", "audit"],
        "summary": ["summarize", "summary", "overview", "brief", "digest", "condense"],
        "creative": ["brainstorm", "idea", "creative", "imagine", "innovate", "what if"],
        "advice": ["recommend", "suggest", "advise", "should", "strategy", "plan", "decision"],
        "risk": ["risk", "threat", "danger", "vulnerability", "concern", "issue"],
        "comparison": ["compare", "versus", "vs", "difference", "trade-off", "pros and cons"],
        "extraction": ["extract", "find", "list", "identify", "pull out", "highlight"],
        "research": ["evidence", "claim", "source", "verify", "study", "literature"],
        "debug": ["bug", "error", "debug", "anomaly", "root cause", "failure"],
        "planning": ["roadmap", "timeline", "milestone", "phase", "schedule", "dependency"],
        "critical": ["critique", "challenge", "weakness", "flaw", "assumption", "stress test"],
    }

    for intent, keywords in intent_map.items():
        hits = sum(1 for keyword in keywords if keyword in q)
        if hits > 0:
            signals[intent] = min(hits / len(keywords), 1.0)

    return signals


def _estimate_query_complexity(query: str, signals: dict[str, float]) -> QueryComplexity:
    """Estimate query complexity using lexical and intent features."""

    words = query.lower().split()
    word_count = len(words)
    intent_count = len(signals)
    multi_step_markers = sum(
        1
        for word in words
        if word in {"and", "then", "also", "additionally", "moreover", "plus"}
    )

    score = (
        min(word_count / 30.0, 1.0) * 0.3
        + min(intent_count / 4.0, 1.0) * 0.4
        + min(multi_step_markers / 3.0, 1.0) * 0.3
    )

    if score < 0.25:
        return QueryComplexity.SIMPLE
    if score < 0.50:
        return QueryComplexity.MODERATE
    if score < 0.75:
        return QueryComplexity.COMPLEX
    return QueryComplexity.EXPERT


def _detect_collaboration_need(
    query: str,
    best_agent: AgentProfile,
    scored: list[tuple[AgentProfile, float, dict[str, float]]],
    signals: dict[str, float],
) -> tuple[list[str], str]:
    """Detect whether additional collaborators should be suggested."""

    _ = query
    collaborators: list[str] = []
    reason = ""

    if len(scored) < 2:
        return collaborators, reason

    score_gap = scored[0][1] - scored[1][1]
    intent_count = len(signals)

    if score_gap < 0.15 and scored[1][1] > 0.2:
        collaborators.append(scored[1][0].name)
        reason = f"close score gap ({score_gap:.3f})"
    elif intent_count >= 3 or best_agent.personality.collaboration_tendency > 0.6:
        for partner_name in best_agent.collaboration_partners:
            partner = get_agent(partner_name)
            if partner and partner.name != best_agent.name:
                collaborators.append(partner.name)
                reason = f"multi-intent query ({intent_count} intents)"
                break

    return collaborators, reason


def _score_agent(agent: AgentProfile, query: str, signals: dict[str, float], risk_level: RiskLevel) -> tuple[float, dict[str, float]]:
    """Score how suitable an agent is for this query (0-1) with AURORA-like components."""

    lowered_query = query.lower()
    query_tokens = set(lowered_query.split())

    domain_hits = sum(1 for domain in agent.domains if domain in lowered_query)
    domain_score = min(domain_hits / max(len(agent.domains), 1), 1.0)

    intent_overlap = 0.0
    for intent, strength in signals.items():
        if intent in agent.domains or any(intent in domain for domain in agent.domains):
            intent_overlap += strength
    intent_score = min(intent_overlap / max(len(signals), 1), 1.0) if signals else 0.0

    desc_tokens = set(agent.description.lower().split())
    desc_overlap = len(query_tokens & desc_tokens) / max(len(query_tokens), 1)

    anti_hits = sum(1 for pattern in agent.anti_patterns if pattern.lower() in lowered_query)
    anti_penalty = min(anti_hits * 0.3, 0.5)

    intent_fit = intent_score
    domain_affinity = domain_score
    output_style_fit = desc_overlap
    skill_access_fit = min(len(agent.preferred_skills) / 6.0, 1.0)
    memory_affinity = 0.5

    if risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        risk_alignment = 0.9 if agent.style.value == "cautious" else 0.45
    elif risk_level == RiskLevel.MEDIUM:
        risk_alignment = 0.7
    else:
        risk_alignment = 0.55 if agent.style.value == "aggressive" else 0.65

    long_context = len(query_tokens) > 20
    if long_context and ("summary" in " ".join(agent.domains).lower() or "analysis" in " ".join(agent.domains).lower()):
        context_compression_gain = 0.8
    elif long_context:
        context_compression_gain = 0.55
    else:
        context_compression_gain = 0.45

    cost_penalty = 0.25 if agent.style.value == "cautious" else 0.10
    latency_penalty = 0.20 if agent.style.value == "cautious" else 0.08
    overconfidence_risk = 0.35 if (agent.style.value == "aggressive" and risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}) else 0.08

    raw = (
        0.22 * intent_fit
        + 0.18 * domain_affinity
        + 0.10 * output_style_fit
        + 0.09 * skill_access_fit
        + 0.05 * memory_affinity
        + 0.16 * risk_alignment
        + 0.08 * context_compression_gain
        - 0.04 * cost_penalty
        - 0.03 * latency_penalty
        - 0.05 * overconfidence_risk
        - 0.10 * anti_penalty
    )
    score = max(0.0, min(raw, 1.0))
    breakdown = {
        "intent_fit": round(intent_fit, 4),
        "domain_affinity": round(domain_affinity, 4),
        "output_style_fit": round(output_style_fit, 4),
        "skill_access_fit": round(skill_access_fit, 4),
        "memory_affinity": round(memory_affinity, 4),
        "risk_alignment": round(risk_alignment, 4),
        "context_compression_gain": round(context_compression_gain, 4),
        "cost_penalty": round(cost_penalty, 4),
        "latency_penalty": round(latency_penalty, 4),
        "overconfidence_risk": round(overconfidence_risk, 4),
        "anti_pattern_penalty": round(anti_penalty, 4),
    }
    return score, breakdown


def _complexity_skill_count(complexity: QueryComplexity) -> int:
    mapping = {
        QueryComplexity.SIMPLE: 1,
        QueryComplexity.MODERATE: 2,
        QueryComplexity.COMPLEX: 3,
        QueryComplexity.EXPERT: 4,
    }
    return mapping.get(complexity, 2)


def route_to_agent(state: GraphState) -> GraphState:
    """LangGraph node: choose the best agent for the query."""

    mode = normalize_mode(state.system_mode)
    policy = state.policy or policy_for_mode(mode).to_dict()
    state.policy = policy
    state.system_mode = mode.value

    risk_level = infer_risk_level(state.query)
    state.risk_level = risk_level.value

    agents = list_agents()
    signals = _intent_signals(state.query)

    scored = []
    for agent in agents:
        score, breakdown = _score_agent(agent, state.query, signals, risk_level)
        scored.append((agent, score, breakdown))
    scored.sort(key=lambda item: item[1], reverse=True)

    best_agent, best_score, _ = scored[0]

    complexity = _estimate_query_complexity(state.query, signals)
    state.query_complexity = complexity.value
    state.detected_intents = list(signals.keys())
    state.estimated_skill_count = _complexity_skill_count(complexity)

    collaborators, collaboration_reason = _detect_collaboration_need(
        state.query,
        best_agent,
        scored,
        signals,
    )
    allow_secondary = bool(policy.get("budget", {}).get("allow_secondary_agent", True))
    if not allow_secondary:
        collaborators = []
        collaboration_reason = "secondary agent disabled by policy"

    state.collaborating_agents = collaborators
    state.collaboration_reason = collaboration_reason

    state.personality = personality_to_dict(best_agent.personality)
    state.approval_required = bool(policy.get("risk", {}).get("require_human_approval", False)) and (
        risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    )

    query_state = QueryState(
        query_raw=state.query,
        intent_vector=signals,
        domain_tags=list(signals.keys())[:5],
        risk_level=risk_level.value,
        output_mode="text",
        budget_constraints={"max_total_cost": float(policy.get("budget", {}).get("max_total_cost", 5.0))},
        latency_constraints={"max_parallelism": float(policy.get("budget", {}).get("max_parallelism", 3))},
        evidence_required=bool(policy.get("risk", {}).get("require_verifier", False)),
        uncertainty_estimate=max(0.0, 1.0 - best_score),
        routing_goals=["complementarity", "traceability", "risk-aware decision"],
        approval_policy="required" if state.approval_required else "auto",
    )
    state.query_state = query_state.__dict__

    selected_names = [best_agent.name]
    rejected_names = [agent.name for agent, _, _ in scored[1:]]
    reasons: dict[str, str] = {
        best_agent.name: (
            f"selected (score={best_score:.3f}, complexity={complexity.value}): "
            "best domain+intent match"
        )
    }

    for agent, score, _ in scored[1:]:
        reasons[agent.name] = f"rejected (score={score:.3f}): lower relevance"

    decision = RoutingDecision(
        selected=selected_names,
        rejected=rejected_names,
        reasons=reasons,
    )

    state.routing_trace["agent_decision"] = {
        "selected": decision.selected,
        "rejected": decision.rejected,
        "reasons": decision.reasons,
        "scores": {agent.name: round(score, 4) for agent, score, _ in scored},
        "score_breakdown": {agent.name: detail for agent, _, detail in scored},
        "intent_signals": signals,
        "query_complexity": complexity.value,
        "system_mode": state.system_mode,
        "risk_level": state.risk_level,
        "collaboration": {
            "agents": collaborators,
            "reason": collaboration_reason,
        },
        "personality": state.personality,
        "policy": {
            "allow_secondary_agent": allow_secondary,
            "require_verifier": bool(policy.get("risk", {}).get("require_verifier", False)),
            "force_dissent_branch": bool(policy.get("diversity", {}).get("force_dissent_branch", False)),
        },
    }
    state.routing_trace["agent_candidates"] = [agent.name for agent, _, _ in scored]
    state.routing_trace["agent_scores"] = {agent.name: round(score, 4) for agent, score, _ in scored}
    state.routing_trace["agent_selection_reason"] = reasons.get(best_agent.name, "")
    state.routing_trace.setdefault("rejection_reasons", {}).update(
        {agent.name: reasons.get(agent.name, "") for agent, _, _ in scored if agent.name != best_agent.name}
    )
    state.routing_trace.setdefault("state_snapshots", []).append(
        {
            "phase": "agent_router",
            "agent": best_agent.name,
            "mode": state.system_mode,
            "risk_level": state.risk_level,
            "complexity": complexity.value,
            "intents": list(signals.keys()),
        }
    )

    state.reasoning_path.append(
        {
            "step": len(state.reasoning_path) + 1,
            "event": "agent_selected",
            "elapsed_ms": 1.0,
            "description": "Agent selected",
            "data": {
                "agent": best_agent.name,
                "score": round(best_score, 3),
                "complexity": complexity.value,
                "collaborators": collaborators,
            },
        }
    )

    state.agent_name = best_agent.name
    state.agent_style = state.forced_style or best_agent.style
    state.available_skills = best_agent.preferred_skills

    return state
