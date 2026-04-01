"""Core state definitions for the LangGraph Skill Router."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SkillCategory(str, Enum):
    """Categories that describe what a skill does."""

    RECALL = "recall"
    REASONING = "reasoning"
    COMMUNICATION = "communication"
    EXTRACTION = "extraction"
    GENERATION = "generation"
    ANALYSIS = "analysis"


class AgentStyle(str, Enum):
    """Decision-making style of an agent."""

    AGGRESSIVE = "aggressive"  # picks the strongest single skill
    CAUTIOUS = "cautious"  # picks multiple skills for cross-validation
    CREATIVE = "creative"  # picks uncommon combinations
    BALANCED = "balanced"  # default: complementary selection


class SkillTier(str, Enum):
    """Capability tiers for skills."""

    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"
    LEGENDARY = "legendary"


class RoutingStrategy(str, Enum):
    """Routing strategy families."""

    GREEDY = "greedy"
    COMPLEMENTARY = "complementary"
    ENSEMBLE = "ensemble"
    BUDGET_AWARE = "budget_aware"
    DIVERSITY_FIRST = "diversity_first"


class ConflictPolicy(str, Enum):
    """How to resolve conflicting skill outputs."""

    VOTING = "voting"
    WEIGHTED = "weighted"
    HIERARCHICAL = "hierarchical"
    DEBATE = "debate"
    HUMAN_IN_LOOP = "human_in_loop"


class OutputFormat(str, Enum):
    """Output format classes used by skills."""

    TEXT = "text"
    STRUCTURED = "structured"
    LIST = "list"
    SCORE = "score"
    MATRIX = "matrix"
    TIMELINE = "timeline"
    GRAPH = "graph"


class QueryComplexity(str, Enum):
    """Estimated query complexity level."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class SkillMetadata:
    """Metadata that each skill exposes for routing decisions."""

    name: str
    description: str
    strengths: list[str]
    weaknesses: list[str]
    category: SkillCategory
    output_type: str  # e.g. "text", "structured", "list", "score"
    skill_id: str = ""
    owner: str = "core"
    version: str = "1.0.0"
    summary: str = ""
    applicable_tasks: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    optional_inputs: list[str] = field(default_factory=list)
    output_schema: dict[str, Any] = field(default_factory=dict)
    evidence_style: str = "heuristic"
    reasoning_role: str = "general"
    latency_profile: str = "medium"
    cost_profile: str = "medium"
    risk_profile: str = "medium"
    interpretability_score: float = 0.7
    calibration_score: float = 0.6
    failure_modes: list[str] = field(default_factory=list)
    recovery_suggestions: list[str] = field(default_factory=list)
    compatible_with: list[str] = field(default_factory=list)
    complements: list[str] = field(default_factory=list)
    redundant_with: list[str] = field(default_factory=list)
    ideal_position_in_pipeline: str = "middle"
    supports_parallelism: bool = True
    success_rate_by_domain: dict[str, float] = field(default_factory=dict)
    success_rate_by_agent: dict[str, float] = field(default_factory=dict)
    pairwise_synergy_scores: dict[str, float] = field(default_factory=dict)
    pairwise_redundancy_scores: dict[str, float] = field(default_factory=dict)
    drift_flag: bool = False
    reputation_score: float = 0.5
    confidence_keywords: list[str] = field(default_factory=list)
    tier: SkillTier = SkillTier.BASIC
    compute_cost: float = 1.0
    synergies: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    min_context_length: int = 10
    max_context_length: int = 10000

    def to_prompt_description(self) -> str:
        return (
            f"[{self.name}] ({self.category.value}/{self.tier.value})\n"
            f"  Good at: {', '.join(self.strengths)}\n"
            f"  Bad at: {', '.join(self.weaknesses)}\n"
            f"  Output: {self.output_type}; Cost: {self.compute_cost:.1f}"
        )

    def to_skill_card(self) -> dict[str, Any]:
        """Return a richer SkillCard-like payload for inspection/trace/debug."""

        skill_id = self.skill_id or self.name
        return {
            "skill_id": skill_id,
            "name": self.name,
            "summary": self.summary or self.description,
            "owner": self.owner,
            "version": self.version,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "applicable_tasks": self.applicable_tasks,
            "contraindications": self.contraindications,
            "required_inputs": self.required_inputs,
            "optional_inputs": self.optional_inputs,
            "output_type": self.output_type,
            "output_schema": self.output_schema,
            "evidence_style": self.evidence_style,
            "reasoning_role": self.reasoning_role,
            "latency_profile": self.latency_profile,
            "cost_profile": self.cost_profile,
            "risk_profile": self.risk_profile,
            "interpretability_score": self.interpretability_score,
            "calibration_score": self.calibration_score,
            "failure_modes": self.failure_modes,
            "recovery_suggestions": self.recovery_suggestions,
            "compatible_with": self.compatible_with or self.synergies,
            "conflicts_with": self.conflicts,
            "complements": self.complements or self.synergies,
            "redundant_with": self.redundant_with,
            "ideal_position_in_pipeline": self.ideal_position_in_pipeline,
            "supports_parallelism": self.supports_parallelism,
            "success_rate_by_domain": self.success_rate_by_domain,
            "success_rate_by_agent": self.success_rate_by_agent,
            "pairwise_synergy_scores": self.pairwise_synergy_scores,
            "pairwise_redundancy_scores": self.pairwise_redundancy_scores,
            "drift_flag": self.drift_flag,
            "reputation_score": self.reputation_score,
        }


@dataclass
class RoutingDecision:
    """A record of a routing decision at agent or skill level."""

    selected: list[str]
    rejected: list[str]
    reasons: dict[str, str]


@dataclass
class RoutingTrace:
    """Full trace of all routing decisions for a single query."""

    query: str
    agent_decision: RoutingDecision | None = None
    skill_decision: RoutingDecision | None = None
    complementarity_scores: dict[str, float] = field(default_factory=dict)
    agent_candidates: list[str] = field(default_factory=list)
    agent_scores: dict[str, float] = field(default_factory=dict)
    agent_selection_reason: str = ""
    skill_candidates: list[str] = field(default_factory=list)
    skill_scores: dict[str, float] = field(default_factory=dict)
    complementarity_matrix: dict[str, float] = field(default_factory=dict)
    rejection_reasons: dict[str, str] = field(default_factory=dict)
    state_snapshots: list[dict[str, Any]] = field(default_factory=list)
    execution_timeline: list[dict[str, Any]] = field(default_factory=list)
    interrupt_events: list[dict[str, Any]] = field(default_factory=list)
    verifier_findings: list[dict[str, Any]] = field(default_factory=list)
    final_confidence_breakdown: dict[str, float] = field(default_factory=dict)
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    latency_breakdown: dict[str, float] = field(default_factory=dict)
    query_complexity: str = QueryComplexity.MODERATE.value
    collaboration: dict[str, Any] = field(default_factory=dict)
    personality: dict[str, float] = field(default_factory=dict)
    routing_strategy: str = RoutingStrategy.COMPLEMENTARY.value

    def summary(self) -> str:
        lines = [f"Query: {self.query}", f"Complexity: {self.query_complexity}", ""]
        if self.agent_decision:
            lines.append("Agent Routing:")
            lines.append(f"  Selected: {', '.join(self.agent_decision.selected)}")
            for name, reason in self.agent_decision.reasons.items():
                lines.append(f"    {name}: {reason}")
            lines.append("")
        if self.skill_decision:
            lines.append("Skill Routing:")
            lines.append(f"  Selected: {', '.join(self.skill_decision.selected)}")
            lines.append(f"  Rejected: {', '.join(self.skill_decision.rejected)}")
            for name, reason in self.skill_decision.reasons.items():
                lines.append(f"    {name}: {reason}")
            lines.append("")
        if self.complementarity_scores:
            lines.append("Complementarity Scores:")
            for combo, score in self.complementarity_scores.items():
                lines.append(f"  {combo}: {score:.3f}")
        return "\n".join(lines)


@dataclass
class AgentPersonality:
    """Decision personality for skill routing behavior."""

    risk_tolerance: float
    creativity_bias: float
    diversity_preference: float
    confidence_threshold: float
    collaboration_tendency: float
    depth_vs_breadth: float

    def describe(self) -> str:
        parts: list[str] = []

        if self.risk_tolerance >= 0.7:
            parts.append("high risk tolerance")
        elif self.risk_tolerance <= 0.3:
            parts.append("risk-averse")

        if self.creativity_bias >= 0.7:
            parts.append("creative explorer")
        elif self.creativity_bias <= 0.3:
            parts.append("conservative selector")

        if self.diversity_preference >= 0.7:
            parts.append("diversity-seeking")
        elif self.diversity_preference <= 0.3:
            parts.append("focuses on narrow top skills")

        if self.depth_vs_breadth <= 0.3:
            parts.append("depth-first")
        elif self.depth_vs_breadth >= 0.7:
            parts.append("breadth-first")

        if not parts:
            return "Balanced decision profile"
        return "Profile: " + ", ".join(parts)


@dataclass
class SkillExecutionContext:
    """Runtime context for one skill execution."""

    skill_name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    error_message: str = ""
    retry_count: int = 0
    output_length: int = 0
    quality_score: float = 0.0

    @property
    def is_slow(self) -> bool:
        return self.duration_ms > 500.0


@dataclass
class EnsembleConfig:
    """Configuration for ensemble synthesis."""

    min_agreement_ratio: float = 0.6
    conflict_policy: ConflictPolicy = ConflictPolicy.WEIGHTED
    weight_by_relevance: bool = True
    weight_by_reputation: bool = True
    include_minority_report: bool = True
    max_synthesis_length: int = 2000


@dataclass
class RoutingMetrics:
    """Full routing quality metric set."""

    coverage: float = 0.0
    redundancy: float = 0.0
    diversity_shannon: float = 0.0
    diversity_simpson: float = 0.0
    ensemble_coherence: float = 0.0

    total_execution_ms: float = 0.0
    avg_quality_score: float = 0.0
    skill_success_rate: float = 0.0

    synergy_score: float = 0.0
    conflict_count: int = 0
    consensus_strength: float = 0.0

    def to_dict(self) -> dict[str, float]:
        data = asdict(self)
        out: dict[str, float] = {}
        for key, value in data.items():
            if isinstance(value, bool):
                out[key] = float(value)
            elif isinstance(value, (int, float)):
                out[key] = float(value)
        return out


@dataclass
class SkillDependency:
    """Dependency relationships across skills."""

    skill_name: str
    depends_on: list[str]
    enhances: list[str]
    conflicts_with: list[str]


@dataclass
class SkillBudget:
    """Abstract computation budget model for skills."""

    compute_cost: float = 1.0
    latency_estimate_ms: float = 100.0
    context_requirement: float = 0.5


@dataclass
class QueryState:
    """Structured request state used by routing policy and analytics."""

    query_raw: str
    intent_vector: dict[str, float] = field(default_factory=dict)
    domain_tags: list[str] = field(default_factory=list)
    risk_level: str = "medium"
    output_mode: str = "text"
    budget_constraints: dict[str, float] = field(default_factory=dict)
    latency_constraints: dict[str, float] = field(default_factory=dict)
    evidence_required: bool = False
    conversation_context: list[str] = field(default_factory=list)
    memory_handles: list[str] = field(default_factory=list)
    uncertainty_estimate: float = 0.0
    routing_goals: list[str] = field(default_factory=list)
    approval_policy: str = "auto"


@dataclass
class SkillPortfolio:
    """A structured skill portfolio plan, not only a selected list."""

    selected_skills: list[str] = field(default_factory=list)
    role_coverage: dict[str, int] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)
    verification_chain: list[str] = field(default_factory=list)
    budget_estimate: float = 0.0
    latency_estimate: float = 0.0
    portfolio_score: float = 0.0
    portfolio_rationale: str = ""
    rejected_candidates: list[str] = field(default_factory=list)
    fallback_portfolios: list[list[str]] = field(default_factory=list)


class GraphState(BaseModel):
    """The shared state that flows through the LangGraph graph."""

    query: str
    agent_name: str = ""
    agent_style: AgentStyle = AgentStyle.BALANCED
    forced_style: Optional[AgentStyle] = None
    max_skills: int = 3
    available_skills: list[str] = Field(default_factory=list)
    selected_skills: list[str] = Field(default_factory=list)
    skill_outputs: dict[str, Any] = Field(default_factory=dict)
    final_output: str = ""
    routing_trace: dict[str, Any] = Field(default_factory=dict)

    personality: dict[str, float] = Field(default_factory=dict)
    personality_adjustments: dict[str, Any] = Field(default_factory=dict)

    query_complexity: str = QueryComplexity.MODERATE.value
    detected_intents: list[str] = Field(default_factory=list)
    estimated_skill_count: int = 2

    collaborating_agents: list[str] = Field(default_factory=list)
    collaboration_reason: str = ""

    routing_strategy: str = RoutingStrategy.COMPLEMENTARY.value
    skill_budget: float = 5.0

    execution_contexts: dict[str, Any] = Field(default_factory=dict)
    execution_order: list[str] = Field(default_factory=list)

    conflicts_detected: list[dict[str, Any]] = Field(default_factory=list)
    consensus_result: dict[str, Any] = Field(default_factory=dict)

    reasoning_path: list[dict[str, Any]] = Field(default_factory=list)
    routing_metrics: dict[str, float] = Field(default_factory=dict)

    system_mode: str = "balanced"
    risk_level: str = "medium"
    query_state: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    portfolio_plan: dict[str, Any] = Field(default_factory=dict)
    verification_findings: list[dict[str, Any]] = Field(default_factory=list)
    disagreement_triggered: bool = False
    approval_required: bool = False
    trace_id: str = ""
    response_contract: dict[str, Any] = Field(default_factory=dict)


def personality_from_dict(data: dict[str, float]) -> AgentPersonality:
    """Rebuild AgentPersonality from serialized dict."""

    return AgentPersonality(
        risk_tolerance=float(data.get("risk_tolerance", 0.5)),
        creativity_bias=float(data.get("creativity_bias", 0.5)),
        diversity_preference=float(data.get("diversity_preference", 0.5)),
        confidence_threshold=float(data.get("confidence_threshold", 0.2)),
        collaboration_tendency=float(data.get("collaboration_tendency", 0.5)),
        depth_vs_breadth=float(data.get("depth_vs_breadth", 0.5)),
    )


def personality_to_dict(personality: AgentPersonality) -> dict[str, float]:
    """Serialize AgentPersonality to dict for GraphState."""

    return {
        "risk_tolerance": personality.risk_tolerance,
        "creativity_bias": personality.creativity_bias,
        "diversity_preference": personality.diversity_preference,
        "confidence_threshold": personality.confidence_threshold,
        "collaboration_tendency": personality.collaboration_tendency,
        "depth_vs_breadth": personality.depth_vs_breadth,
    }
