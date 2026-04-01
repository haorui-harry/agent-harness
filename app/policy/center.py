"""Policy center implementing system modes and routing governance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class SystemMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"
    SAFETY_CRITICAL = "safety_critical"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BudgetPolicy:
    max_skills: int
    max_total_cost: float
    max_parallelism: int
    allow_deep_execution: bool
    allow_secondary_agent: bool


@dataclass
class RiskPolicy:
    require_verifier: bool
    banned_high_risk_skills: list[str]
    require_human_approval: bool
    require_uncertainty_notice: bool


@dataclass
class DiversityPolicy:
    encourage_rare_skill_combos: bool
    complementarity_weight: float
    redundancy_penalty_weight: float
    force_dissent_branch: bool


@dataclass
class TracePolicy:
    trace_granularity: str
    show_reject_reasons: bool
    record_state_diff: bool
    output_user_trace_summary: bool


@dataclass
class GovernancePolicy:
    interrupt_for_external_actions: bool
    redact_trace: bool
    record_evidence_chain: bool
    allow_write_skills: bool


@dataclass
class PolicyBundle:
    mode: SystemMode
    budget: BudgetPolicy
    risk: RiskPolicy
    diversity: DiversityPolicy
    trace: TracePolicy
    governance: GovernancePolicy

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


def infer_risk_level(query: str) -> RiskLevel:
    lowered = query.lower()
    critical_markers = [
        "legal",
        "compliance",
        "audit",
        "medical",
        "financial",
        "safety",
        "regulatory",
    ]
    high_markers = ["risk", "critical", "security", "vulnerability", "incident", "breach"]

    if any(marker in lowered for marker in critical_markers):
        return RiskLevel.CRITICAL
    if any(marker in lowered for marker in high_markers):
        return RiskLevel.HIGH
    if len(lowered.split()) > 18:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def policy_for_mode(mode: SystemMode) -> PolicyBundle:
    if mode == SystemMode.FAST:
        return PolicyBundle(
            mode=mode,
            budget=BudgetPolicy(
                max_skills=2,
                max_total_cost=2.5,
                max_parallelism=2,
                allow_deep_execution=False,
                allow_secondary_agent=False,
            ),
            risk=RiskPolicy(
                require_verifier=False,
                banned_high_risk_skills=["scenario_generator"],
                require_human_approval=False,
                require_uncertainty_notice=False,
            ),
            diversity=DiversityPolicy(
                encourage_rare_skill_combos=False,
                complementarity_weight=0.8,
                redundancy_penalty_weight=1.2,
                force_dissent_branch=False,
            ),
            trace=TracePolicy(
                trace_granularity="compact",
                show_reject_reasons=False,
                record_state_diff=False,
                output_user_trace_summary=True,
            ),
            governance=GovernancePolicy(
                interrupt_for_external_actions=False,
                redact_trace=False,
                record_evidence_chain=False,
                allow_write_skills=False,
            ),
        )

    if mode == SystemMode.DEEP:
        return PolicyBundle(
            mode=mode,
            budget=BudgetPolicy(
                max_skills=5,
                max_total_cost=8.0,
                max_parallelism=4,
                allow_deep_execution=True,
                allow_secondary_agent=True,
            ),
            risk=RiskPolicy(
                require_verifier=True,
                banned_high_risk_skills=[],
                require_human_approval=False,
                require_uncertainty_notice=True,
            ),
            diversity=DiversityPolicy(
                encourage_rare_skill_combos=True,
                complementarity_weight=1.2,
                redundancy_penalty_weight=1.0,
                force_dissent_branch=True,
            ),
            trace=TracePolicy(
                trace_granularity="full",
                show_reject_reasons=True,
                record_state_diff=True,
                output_user_trace_summary=True,
            ),
            governance=GovernancePolicy(
                interrupt_for_external_actions=True,
                redact_trace=False,
                record_evidence_chain=True,
                allow_write_skills=False,
            ),
        )

    if mode == SystemMode.SAFETY_CRITICAL:
        return PolicyBundle(
            mode=mode,
            budget=BudgetPolicy(
                max_skills=6,
                max_total_cost=10.0,
                max_parallelism=3,
                allow_deep_execution=True,
                allow_secondary_agent=True,
            ),
            risk=RiskPolicy(
                require_verifier=True,
                banned_high_risk_skills=["brainstorm_ideas", "generate_analogies"],
                require_human_approval=True,
                require_uncertainty_notice=True,
            ),
            diversity=DiversityPolicy(
                encourage_rare_skill_combos=False,
                complementarity_weight=1.0,
                redundancy_penalty_weight=1.4,
                force_dissent_branch=True,
            ),
            trace=TracePolicy(
                trace_granularity="forensic",
                show_reject_reasons=True,
                record_state_diff=True,
                output_user_trace_summary=True,
            ),
            governance=GovernancePolicy(
                interrupt_for_external_actions=True,
                redact_trace=True,
                record_evidence_chain=True,
                allow_write_skills=False,
            ),
        )

    return PolicyBundle(
        mode=SystemMode.BALANCED,
        budget=BudgetPolicy(
            max_skills=3,
            max_total_cost=5.0,
            max_parallelism=3,
            allow_deep_execution=False,
            allow_secondary_agent=True,
        ),
        risk=RiskPolicy(
            require_verifier=False,
            banned_high_risk_skills=[],
            require_human_approval=False,
            require_uncertainty_notice=False,
        ),
        diversity=DiversityPolicy(
            encourage_rare_skill_combos=False,
            complementarity_weight=1.0,
            redundancy_penalty_weight=1.0,
            force_dissent_branch=False,
        ),
        trace=TracePolicy(
            trace_granularity="standard",
            show_reject_reasons=True,
            record_state_diff=False,
            output_user_trace_summary=True,
        ),
        governance=GovernancePolicy(
            interrupt_for_external_actions=False,
            redact_trace=False,
            record_evidence_chain=False,
            allow_write_skills=False,
        ),
    )


def normalize_mode(mode: str | None) -> SystemMode:
    if not mode:
        return SystemMode.BALANCED
    try:
        return SystemMode(mode)
    except ValueError:
        return SystemMode.BALANCED
