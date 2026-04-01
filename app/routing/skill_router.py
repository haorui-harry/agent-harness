"""Skill router with budget-aware, personality-aware multi-round selection."""

from __future__ import annotations

from app.core.state import (
    AgentPersonality,
    AgentStyle,
    GraphState,
    QueryComplexity,
    RoutingStrategy,
    SkillPortfolio,
    SkillMetadata,
    personality_from_dict,
)
from app.ecosystem.marketplace import discover_for_query
from app.personality.engine import PersonalityEngine
from app.policy.center import SystemMode, normalize_mode
from app.routing.complementarity import ComplementarityEngine
from app.skills.registry import get_skill_dependencies, get_skill_metadata, list_all_skills


def _adjust_max_skills(base: int, complexity: QueryComplexity) -> int:
    """Adjust max skill count by query complexity."""

    multiplier = {
        QueryComplexity.SIMPLE: 0.5,
        QueryComplexity.MODERATE: 1.0,
        QueryComplexity.COMPLEX: 1.5,
        QueryComplexity.EXPERT: 2.0,
    }
    return max(1, int(base * multiplier.get(complexity, 1.0)))


def _build_candidate_pool(state: GraphState) -> list[SkillMetadata]:
    """Build candidate pool from preferred and global skill registries."""

    candidates: list[SkillMetadata] = []

    for name in state.available_skills:
        meta = get_skill_metadata(name)
        if meta:
            candidates.append(meta)

    all_skills = list_all_skills()
    seen = {meta.name for meta in candidates}
    for meta in all_skills:
        if meta.name not in seen:
            candidates.append(meta)
            seen.add(meta.name)

    return candidates


def _reconstruct_personality(data: dict[str, float]) -> AgentPersonality:
    """Reconstruct AgentPersonality from serialized dict."""

    return personality_from_dict(data)


def _resolve_execution_order(skill_names: list[str]) -> list[str]:
    """Resolve execution order using dependency-enhancement topological sorting."""

    graph: dict[str, list[str]] = {name: [] for name in skill_names}

    for name in skill_names:
        dependency = get_skill_dependencies(name)
        if not dependency:
            continue
        for enhancer in dependency.enhances:
            if enhancer in graph:
                graph[name].append(enhancer)

    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            return
        visiting.add(node)
        for dep in graph.get(node, []):
            dfs(dep)
        visiting.remove(node)
        visited.add(node)
        order.append(node)

    for name in skill_names:
        dfs(name)

    return order


def _role_of(meta: SkillMetadata) -> str:
    if meta.category.value in {"recall", "extraction"}:
        return "evidence"
    if meta.category.value in {"reasoning", "analysis"}:
        return "reasoning"
    if meta.category.value in {"communication"}:
        return "communication"
    if "validate" in meta.name or "anomal" in meta.name or "crit" in meta.name:
        return "verification"
    return "generation"


def _build_portfolio_plan(
    selected: list[str],
    all_scored: list,
    execution_order: list[str],
    budget_used: float,
) -> dict:
    role_coverage: dict[str, int] = {}
    for scored in all_scored:
        if scored.metadata.name not in selected:
            continue
        role = _role_of(scored.metadata)
        role_coverage[role] = role_coverage.get(role, 0) + 1

    parallel_groups: list[list[str]] = []
    if len(selected) >= 3:
        parallel_groups = [selected[:2], selected[2:]]
    elif selected:
        parallel_groups = [selected]

    verification_chain = [name for name in selected if "validate" in name or "anomal" in name or "critic" in name]
    selected_scored = [s for s in all_scored if s.metadata.name in selected]
    avg_score = sum(s.composite_score for s in selected_scored) / max(len(selected_scored), 1)
    latency_estimate = sum(s.budget_cost * 100.0 for s in selected_scored)

    portfolio = SkillPortfolio(
        selected_skills=selected,
        role_coverage=role_coverage,
        execution_order=execution_order,
        parallel_groups=parallel_groups,
        verification_chain=verification_chain,
        budget_estimate=budget_used,
        latency_estimate=latency_estimate,
        portfolio_score=avg_score,
        portfolio_rationale="Role-aware complementary portfolio optimized under budget and diversity constraints.",
        rejected_candidates=[s.metadata.name for s in all_scored if s.metadata.name not in selected],
        fallback_portfolios=[selected[: max(1, len(selected) - 1)]],
    )
    return portfolio.__dict__


def _required_roles(state: GraphState, require_verifier: bool) -> list[str]:
    """Infer role slots required for this query/policy."""

    intents = {intent.lower() for intent in state.detected_intents}
    query = state.query.lower()
    roles = ["evidence", "reasoning"]
    if (
        "summary" in intents
        or "compare" in query
        or "recommend" in query
        or "brief" in query
        or "report" in query
    ):
        roles.append("communication")
    if require_verifier or state.risk_level in {"high", "critical"}:
        roles.append("verification")
    deduped: list[str] = []
    for role in roles:
        if role not in deduped:
            deduped.append(role)
    return deduped


def _enforce_role_slots(
    selected_names: list[str],
    all_scored: list,
    required_roles: list[str],
    budget_limit: float,
    effective_max: int,
    reasons: dict[str, str] | None = None,
) -> tuple[list[str], float, dict[str, str]]:
    """Ensure minimum role coverage with low-overhead slot completion."""

    selected = list(dict.fromkeys(selected_names))
    by_name = {item.metadata.name: item for item in all_scored}
    used_budget = 0.0
    for name in selected:
        if name in by_name:
            used_budget += by_name[name].budget_cost
        else:
            meta = get_skill_metadata(name)
            used_budget += float(meta.compute_cost) if meta else 1.0

    max_allowed = max(effective_max, len(required_roles), len(selected))
    if max_allowed < effective_max + 1:
        max_allowed = effective_max + 1

    def _covered_roles(items: list[str]) -> set[str]:
        covered: set[str] = set()
        for skill_name in items:
            meta = get_skill_metadata(skill_name)
            if meta:
                covered.add(_role_of(meta))
        return covered

    slot_reasons: dict[str, str] = dict(reasons or {})

    for role in required_roles:
        covered = _covered_roles(selected)
        if role in covered:
            continue

        candidates = [
            item
            for item in sorted(all_scored, key=lambda row: row.composite_score, reverse=True)
            if item.metadata.name not in selected and _role_of(item.metadata) == role
        ]
        for candidate in candidates:
            if len(selected) >= max_allowed:
                break
            projected = used_budget + candidate.budget_cost
            if projected > budget_limit:
                continue
            selected.append(candidate.metadata.name)
            used_budget = projected
            slot_reasons[candidate.metadata.name] = (
                f"selected (role-slot={role}, relevance={candidate.relevance:.3f}, "
                f"cost={candidate.budget_cost:.2f})"
            )
            break

    return selected, used_budget, slot_reasons


def route_to_skills(state: GraphState) -> GraphState:
    """LangGraph node: select skills via V2 complementarity engine."""

    mode = normalize_mode(state.system_mode)
    policy = state.policy or {}

    complexity = QueryComplexity(state.query_complexity)
    effective_max = _adjust_max_skills(state.max_skills, complexity)
    effective_max = max(effective_max, state.estimated_skill_count)
    budget_policy = policy.get("budget", {})
    if budget_policy.get("max_skills"):
        effective_max = min(effective_max, int(budget_policy["max_skills"]))

    candidates = _build_candidate_pool(state)
    banned = set(policy.get("risk", {}).get("banned_high_risk_skills", []))
    if banned and state.risk_level in {"high", "critical"}:
        candidates = [meta for meta in candidates if meta.name not in banned]
    if mode == SystemMode.FAST:
        candidates = candidates[: min(len(candidates), 8)]

    market_limit_map = {
        SystemMode.FAST: 2,
        SystemMode.BALANCED: 5,
        SystemMode.DEEP: 8,
        SystemMode.SAFETY_CRITICAL: 6,
    }
    marketplace_hits = discover_for_query(state.query, limit=market_limit_map.get(mode, 5))

    personality = _reconstruct_personality(state.personality) if state.personality else None

    refinement = 0 if mode == SystemMode.FAST else 1 if mode == SystemMode.BALANCED else 2
    engine = ComplementarityEngine(
        max_skills=effective_max,
        redundancy_threshold=0.7,
        budget_limit=float(budget_policy.get("max_total_cost", state.skill_budget)),
        enable_synergy=True,
        enable_conflict_avoidance=True,
        refinement_rounds=refinement,
    )

    result = engine.select(
        skills=candidates,
        query=state.query,
        style=AgentStyle(state.agent_style),
        personality=personality,
    )

    all_scored = result.selected + result.rejected
    selected_names = [item.metadata.name for item in result.selected]
    require_verifier = bool(policy.get("risk", {}).get("require_verifier", False))
    budget_limit = float(budget_policy.get("max_total_cost", state.skill_budget))

    role_slots = _required_roles(state, require_verifier=require_verifier)
    selected_names, slot_budget_used, slot_reasons = _enforce_role_slots(
        selected_names=selected_names,
        all_scored=all_scored,
        required_roles=role_slots,
        budget_limit=budget_limit,
        effective_max=effective_max,
    )

    reasons: dict[str, str] = {}
    for selected in result.selected:
        reasons[selected.metadata.name] = (
            f"selected (relevance={selected.relevance:.3f}, diversity={selected.diversity_bonus:.3f}, "
            f"synergy={selected.synergy_bonus:.3f}, cost={selected.budget_cost:.2f}, "
            f"composite={selected.composite_score:.3f})"
        )

    show_reject_reasons = bool(policy.get("trace", {}).get("show_reject_reasons", True))
    if show_reject_reasons:
        for rejected in result.rejected:
            reasons[rejected.metadata.name] = (
                f"rejected (relevance={rejected.relevance:.3f}, redundancy={rejected.redundancy_penalty:.3f}, "
                f"cost={rejected.budget_cost:.2f})"
            )
    reasons.update(slot_reasons)

    selected_set = set(selected_names)
    rejected_names = [item.metadata.name for item in result.rejected if item.metadata.name not in selected_set]
    for selected_name in selected_names:
        if selected_name in reasons and reasons[selected_name].startswith("rejected"):
            reasons[selected_name] = f"selected (policy override for role/risk coverage)"

    if require_verifier and "validate_claims" in selected_set:
        reasons["validate_claims"] = reasons.get(
            "validate_claims",
            "selected (policy-required verifier slot)",
        )

    state.selected_skills = selected_names
    state.execution_order = _resolve_execution_order(selected_names)

    if personality:
        strategy = PersonalityEngine().suggest_routing_strategy(personality)
        state.routing_strategy = strategy.value
    else:
        state.routing_strategy = RoutingStrategy.COMPLEMENTARY.value

    score_map: dict[str, float] = {}
    for item in all_scored:
        score_map[item.metadata.name] = round(item.composite_score, 4)
    for name in selected_names:
        score_map.setdefault(name, 0.0)

    total_budget_used = slot_budget_used if slot_budget_used > 0 else result.total_budget_used

    state.portfolio_plan = _build_portfolio_plan(
        selected=selected_names,
        all_scored=all_scored,
        execution_order=state.execution_order,
        budget_used=total_budget_used,
    )

    state.routing_trace["skill_decision"] = {
        "skill_candidates": [meta.name for meta in candidates],
        "selected": selected_names,
        "rejected": rejected_names,
        "skill_scores": score_map,
        "reasons": reasons,
        "complementarity_metrics": {
            "coverage": result.total_coverage,
            "redundancy": result.total_redundancy,
            "diversity_shannon": result.diversity_shannon,
            "diversity_simpson": result.diversity_simpson,
            "ensemble_coherence": result.ensemble_coherence,
            "total_synergy": result.total_synergy,
            "total_budget_used": total_budget_used,
            "selection_rounds": result.selection_rounds,
        },
        "pairwise_scores": result.pairwise_scores,
        "complementarity_matrix": result.synergy_matrix,
        "synergy_matrix": result.synergy_matrix,
        "marketplace_discovery": marketplace_hits,
        "execution_order": state.execution_order,
        "effective_max_skills": effective_max,
        "strategy": state.routing_strategy,
        "portfolio_plan": state.portfolio_plan,
        "required_role_slots": role_slots,
    }
    state.routing_trace["skill_candidates"] = [meta.name for meta in candidates]
    state.routing_trace["skill_scores"] = score_map
    state.routing_trace["complementarity_matrix"] = result.synergy_matrix
    state.routing_trace.setdefault("rejection_reasons", {}).update(
        {name: reasons.get(name, "") for name in rejected_names}
    )
    state.routing_trace.setdefault("state_snapshots", []).append(
        {
            "phase": "skill_router",
            "selected": selected_names,
            "rejected": rejected_names,
            "required_role_slots": role_slots,
            "effective_max_skills": effective_max,
            "budget_used": round(total_budget_used, 4),
        }
    )

    state.reasoning_path.append(
        {
            "step": len(state.reasoning_path) + 1,
            "event": "skill_selected",
            "elapsed_ms": 3.0,
            "description": "Skills selected via complementarity engine",
            "data": {
                "selected": selected_names,
                "execution_order": state.execution_order,
                "strategy": state.routing_strategy,
            },
        }
    )

    return state
