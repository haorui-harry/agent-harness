"""Built-in and extensible skills for the LangGraph Skill Router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from app.core.state import (
    SkillBudget,
    SkillCategory,
    SkillDependency,
    SkillMetadata,
    SkillTier,
)
from app.ecosystem.marketplace import list_marketplace_skill_metadata


def identify_risks(text: str) -> str:
    """Extract and list key risks from the given text."""

    return (
        "Key Risks Identified:\n"
        "1. [Operational risk extracted from input text]\n"
        "2. [Financial or delivery risk extracted from input text]\n"
        "3. [Mitigation gap that requires follow-up]\n"
        "--- (skill: identify_risks)"
    )


def executive_summary(text: str) -> str:
    """Produce a concise executive summary."""

    return (
        "Executive Summary:\n"
        "- Context: [what this text is about]\n"
        "- Core finding: [most important conclusion]\n"
        "- Decision implication: [what to do next]\n"
        "--- (skill: executive_summary)"
    )


def compare_options(text: str) -> str:
    """Compare options or alternatives mentioned in the text."""

    return (
        "Option Comparison:\n"
        "| Option | Strengths | Risks |\n"
        "|--------|-----------|-------|\n"
        "| A      | ...       | ...   |\n"
        "| B      | ...       | ...   |\n"
        "Recommendation Basis: [trade-offs + context]\n"
        "--- (skill: compare_options)"
    )


def extract_facts(text: str) -> str:
    """Extract factual statements from the text."""

    return (
        "Extracted Facts:\n"
        "- [Fact 1]\n"
        "- [Fact 2]\n"
        "- [Fact 3]\n"
        "Evidence quality: medium\n"
        "--- (skill: extract_facts)"
    )


def generate_recommendations(text: str) -> str:
    """Generate actionable recommendations based on analysis."""

    return (
        "Recommendations:\n"
        "1. [Action item 1 with owner and timeline]\n"
        "2. [Action item 2 with owner and timeline]\n"
        "3. [Action item 3 with expected impact]\n"
        "--- (skill: generate_recommendations)"
    )


def brainstorm_ideas(text: str) -> str:
    """Generate creative ideas and angles related to the topic."""

    return (
        "Brainstormed Ideas:\n"
        "- [Creative angle 1]\n"
        "- [Creative angle 2]\n"
        "- [Unconventional approach]\n"
        "- [High-risk/high-upside experiment]\n"
        "--- (skill: brainstorm_ideas)"
    )


def detect_anomalies(text: str) -> str:
    """Detect anomalies, contradictions, and inconsistency patterns."""

    return (
        "Anomaly Detection Report:\n"
        "1. [Inconsistency detected between section A and B]\n"
        "2. [Unusual pattern: metric X deviates from expected range]\n"
        "3. [Contradiction between claim and table evidence]\n"
        "--- (skill: detect_anomalies)"
    )


def build_timeline(text: str) -> str:
    """Extract events and build a timeline."""

    return (
        "Timeline:\n"
        "  [T1] [Event A - earliest mention]\n"
        "  [T2] [Event B - subsequent development]\n"
        "  [T3] [Event C - most recent]\n"
        "  Dependencies: T2 depends on T1, T3 depends on T2\n"
        "--- (skill: build_timeline)"
    )


def synthesize_perspectives(text: str) -> str:
    """Integrate viewpoints to find consensus and divergence."""

    return (
        "Perspective Synthesis:\n"
        "  Viewpoint A: [perspective summary]\n"
        "  Viewpoint B: [perspective summary]\n"
        "  Common Ground: [shared conclusions]\n"
        "  Key Disagreements: [where perspectives diverge]\n"
        "  Synthesis: [integrated view acknowledging trade-offs]\n"
        "--- (skill: synthesize_perspectives)"
    )


def validate_claims(text: str) -> str:
    """Validate claims and estimate evidence confidence."""

    return (
        "Claim Validation:\n"
        "  Claim 1: [claim text]\n"
        "    Evidence: [supporting/contradicting evidence]\n"
        "    Confidence: HIGH / MEDIUM / LOW\n"
        "    Verdict: SUPPORTED / UNSUPPORTED / INSUFFICIENT_EVIDENCE\n"
        "  Claim 2: [claim text]\n"
        "    Evidence: [supporting/contradicting evidence]\n"
        "    Confidence: MEDIUM\n"
        "    Verdict: PARTIALLY_SUPPORTED\n"
        "--- (skill: validate_claims)"
    )


def generate_analogies(text: str) -> str:
    """Generate analogies for complex concepts."""

    return (
        "Analogies:\n"
        "  Concept: [complex concept from text]\n"
        "  Analogy 1: [relatable analogy from everyday experience]\n"
        "  Analogy 2: [cross-domain analogy for deeper insight]\n"
        "  Why it works: [structural mapping between source and target]\n"
        "--- (skill: generate_analogies)"
    )


def prioritize_items(text: str) -> str:
    """Rank and prioritize items by urgency and impact."""

    return (
        "Priority Ranking:\n"
        "  [P0 - Critical] [item description] - Impact: HIGH, Urgency: HIGH\n"
        "  [P1 - High]     [item description] - Impact: HIGH, Urgency: MEDIUM\n"
        "  [P2 - Medium]   [item description] - Impact: MEDIUM, Urgency: MEDIUM\n"
        "  [P3 - Low]      [item description] - Impact: LOW, Urgency: LOW\n"
        "  Rationale: [why this ordering]\n"
        "--- (skill: prioritize_items)"
    )


SKILL_REGISTRY: dict[str, dict[str, Any]] = {
    "identify_risks": {
        "fn": identify_risks,
        "metadata": SkillMetadata(
            name="identify_risks",
            description="Extract and categorize risks from text",
            strengths=["risk detection", "threat analysis", "negative outcome patterns"],
            weaknesses=["may over-flag benign items", "less useful for purely positive content"],
            category=SkillCategory.EXTRACTION,
            output_type="list",
            confidence_keywords=["risk", "threat", "danger", "concern", "issue", "problem"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.2,
            synergies=["prioritize_items", "generate_recommendations", "detect_anomalies"],
            conflicts=[],
        ),
    },
    "executive_summary": {
        "fn": executive_summary,
        "metadata": SkillMetadata(
            name="executive_summary",
            description="Produce a concise executive summary of the content",
            strengths=["brevity", "clarity", "distilling key points"],
            weaknesses=["loses nuance", "not suitable for deep analysis"],
            category=SkillCategory.COMMUNICATION,
            output_type="text",
            confidence_keywords=["summarize", "summary", "overview", "brief", "tldr"],
            tier=SkillTier.BASIC,
            compute_cost=0.8,
            synergies=["extract_facts", "identify_risks"],
            conflicts=[],
        ),
    },
    "compare_options": {
        "fn": compare_options,
        "metadata": SkillMetadata(
            name="compare_options",
            description="Compare alternatives or options mentioned in the content",
            strengths=["structured comparison", "trade-off analysis"],
            weaknesses=["needs clearly defined options", "less useful for single-topic content"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=["compare", "versus", "vs", "alternative", "option", "trade-off"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.2,
            synergies=["generate_recommendations", "extract_facts"],
            conflicts=[],
        ),
    },
    "extract_facts": {
        "fn": extract_facts,
        "metadata": SkillMetadata(
            name="extract_facts",
            description="Pull out factual statements and data points",
            strengths=["precision", "fact isolation", "data extraction"],
            weaknesses=["ignores opinions and context", "can be overly literal"],
            category=SkillCategory.RECALL,
            output_type="list",
            confidence_keywords=["fact", "data", "number", "statistic", "evidence"],
            tier=SkillTier.BASIC,
            compute_cost=0.9,
            synergies=["validate_claims", "build_timeline", "synthesize_perspectives"],
            conflicts=["generate_analogies"],
        ),
    },
    "generate_recommendations": {
        "fn": generate_recommendations,
        "metadata": SkillMetadata(
            name="generate_recommendations",
            description="Generate actionable recommendations",
            strengths=["actionability", "decision support", "prioritization"],
            weaknesses=["may assume context not present", "can be generic"],
            category=SkillCategory.GENERATION,
            output_type="list",
            confidence_keywords=["recommend", "suggest", "advise", "action", "should", "next step"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.1,
            synergies=["identify_risks", "prioritize_items", "compare_options"],
            conflicts=[],
        ),
    },
    "brainstorm_ideas": {
        "fn": brainstorm_ideas,
        "metadata": SkillMetadata(
            name="brainstorm_ideas",
            description="Generate creative ideas and unconventional angles",
            strengths=["creativity", "lateral thinking", "diverse perspectives"],
            weaknesses=["may lack focus", "not all ideas are practical"],
            category=SkillCategory.GENERATION,
            output_type="list",
            confidence_keywords=["idea", "creative", "brainstorm", "innovate", "explore", "what if"],
            tier=SkillTier.BASIC,
            compute_cost=0.9,
            synergies=["generate_analogies", "synthesize_perspectives"],
            conflicts=["validate_claims", "detect_anomalies"],
        ),
    },
    "detect_anomalies": {
        "fn": detect_anomalies,
        "metadata": SkillMetadata(
            name="detect_anomalies",
            description="Detect anomalies, contradictions, and inconsistencies in text",
            strengths=[
                "contradiction detection",
                "pattern deviation",
                "outlier identification",
                "consistency checking",
            ],
            weaknesses=["may flag intentional contrasts", "less useful for subjective content"],
            category=SkillCategory.ANALYSIS,
            output_type="list",
            confidence_keywords=[
                "anomaly",
                "inconsistent",
                "contradiction",
                "unusual",
                "outlier",
                "deviation",
                "bug",
                "error",
            ],
            tier=SkillTier.ADVANCED,
            compute_cost=1.5,
            synergies=["identify_risks", "validate_claims"],
            conflicts=["brainstorm_ideas"],
        ),
    },
    "build_timeline": {
        "fn": build_timeline,
        "metadata": SkillMetadata(
            name="build_timeline",
            description="Extract events and build a chronological timeline with dependencies",
            strengths=["temporal ordering", "dependency mapping", "sequence reconstruction"],
            weaknesses=["requires temporal markers", "less useful for non-sequential content"],
            category=SkillCategory.EXTRACTION,
            output_type="timeline",
            confidence_keywords=["timeline", "sequence", "when", "before", "after", "history", "phase", "schedule"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.3,
            synergies=["extract_facts", "prioritize_items"],
            conflicts=[],
        ),
    },
    "synthesize_perspectives": {
        "fn": synthesize_perspectives,
        "metadata": SkillMetadata(
            name="synthesize_perspectives",
            description="Identify, compare, and synthesize multiple viewpoints into an integrated analysis",
            strengths=["multi-perspective integration", "consensus detection", "nuance preservation"],
            weaknesses=["may create false balance", "requires diverse sources"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=[
                "perspective",
                "viewpoint",
                "opinion",
                "stakeholder",
                "debate",
                "synthesis",
                "integrate",
            ],
            tier=SkillTier.EXPERT,
            compute_cost=2.0,
            synergies=["extract_facts", "validate_claims"],
            conflicts=[],
        ),
    },
    "validate_claims": {
        "fn": validate_claims,
        "metadata": SkillMetadata(
            name="validate_claims",
            description="Evaluate claims against available evidence and assign confidence levels",
            strengths=["evidence evaluation", "claim verification", "confidence assessment", "bias detection"],
            weaknesses=["needs factual content", "cannot verify against external sources"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=["claim", "verify", "evidence", "true", "false", "proof", "validate", "fact check"],
            tier=SkillTier.EXPERT,
            compute_cost=2.0,
            synergies=["extract_facts", "detect_anomalies"],
            conflicts=["brainstorm_ideas"],
        ),
    },
    "generate_analogies": {
        "fn": generate_analogies,
        "metadata": SkillMetadata(
            name="generate_analogies",
            description="Create illuminating analogies to explain complex concepts",
            strengths=["concept mapping", "simplification", "cross-domain transfer", "intuition building"],
            weaknesses=["analogies can mislead if taken literally", "not suitable for strict specs"],
            category=SkillCategory.GENERATION,
            output_type="text",
            confidence_keywords=["analogy", "like", "similar to", "metaphor", "explain", "simplify", "intuition"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.2,
            synergies=["brainstorm_ideas", "synthesize_perspectives"],
            conflicts=["extract_facts"],
        ),
    },
    "prioritize_items": {
        "fn": prioritize_items,
        "metadata": SkillMetadata(
            name="prioritize_items",
            description="Rank and prioritize items by impact, urgency, and dependencies",
            strengths=["impact assessment", "urgency classification", "dependency-aware ordering"],
            weaknesses=["subjective without clear criteria", "hard on equal-priority cases"],
            category=SkillCategory.ANALYSIS,
            output_type="list",
            confidence_keywords=["priority", "rank", "important", "urgent", "first", "order", "triage", "critical"],
            tier=SkillTier.BASIC,
            compute_cost=0.8,
            synergies=["identify_risks", "generate_recommendations"],
            conflicts=[],
        ),
    },
}

# Third-party skill adapters (runtime-registered).
EXTERNAL_SKILL_REGISTRY: dict[str, dict[str, Any]] = {}


def get_skill_metadata(name: str) -> SkillMetadata | None:
    entry = SKILL_REGISTRY.get(name)
    if entry:
        return entry["metadata"]
    ext = EXTERNAL_SKILL_REGISTRY.get(name)
    if ext:
        return ext["metadata"]
    return None


def list_all_skills() -> list[SkillMetadata]:
    built_in = [entry["metadata"] for entry in SKILL_REGISTRY.values()]
    market = list_marketplace_skill_metadata()
    external = [entry["metadata"] for entry in EXTERNAL_SKILL_REGISTRY.values()]

    combined: list[SkillMetadata] = []
    seen: set[str] = set()
    for meta in built_in + market + external:
        if meta.name not in seen:
            combined.append(meta)
            seen.add(meta.name)
    return combined


def list_builtin_skills() -> list[SkillMetadata]:
    return [entry["metadata"] for entry in SKILL_REGISTRY.values()]


def list_external_skills() -> list[SkillMetadata]:
    return [entry["metadata"] for entry in EXTERNAL_SKILL_REGISTRY.values()]


def register_external_skill(
    name: str,
    fn: Callable[[str], str],
    metadata: SkillMetadata,
    source: str = "third_party",
) -> None:
    """Register an external skill so it can be routed like built-ins."""

    if not metadata.skill_id:
        metadata.skill_id = name
    if not metadata.summary:
        metadata.summary = metadata.description
    EXTERNAL_SKILL_REGISTRY[name] = {
        "fn": fn,
        "metadata": metadata,
        "source": source,
    }


def load_external_skills_from_file(path: str) -> int:
    """Load external skill specs from a local JSON file.

    Expected schema:
    {
      "skills": [
        {
          "name": "third_party_skill",
          "description": "...",
          "category": "analysis",
          "output_type": "structured",
          "confidence_keywords": ["..."],
          "tier": "advanced",
          "compute_cost": 1.1,
          "synergies": ["identify_risks"],
          "conflicts": [],
          "template": "Result for {query}"
        }
      ]
    }
    """

    spec_path = Path(path)
    if not spec_path.exists():
        return 0

    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    skills = payload.get("skills", [])
    loaded = 0
    for raw in skills:
        name = str(raw.get("name", "")).strip()
        if not name:
            continue

        template = str(raw.get("template", "Third-party output for: {query}"))

        def _make_fn(tpl: str, skill_name: str) -> Callable[[str], str]:
            def _fn(query: str) -> str:
                return f"{tpl.format(query=query)}\n--- (skill: {skill_name})"

            return _fn

        metadata = SkillMetadata(
            name=name,
            description=str(raw.get("description", f"External skill {name}")),
            strengths=list(raw.get("strengths", ["external capability"])),
            weaknesses=list(raw.get("weaknesses", ["unknown reliability"])),
            category=SkillCategory(str(raw.get("category", "analysis"))),
            output_type=str(raw.get("output_type", "text")),
            skill_id=str(raw.get("skill_id", name)),
            owner=str(raw.get("owner", raw.get("source", "third_party"))),
            version=str(raw.get("version", "1.0.0")),
            summary=str(raw.get("summary", raw.get("description", f"External skill {name}"))),
            applicable_tasks=list(raw.get("applicable_tasks", [])),
            contraindications=list(raw.get("contraindications", [])),
            required_inputs=list(raw.get("required_inputs", [])),
            optional_inputs=list(raw.get("optional_inputs", [])),
            output_schema=dict(raw.get("output_schema", {})),
            evidence_style=str(raw.get("evidence_style", "heuristic")),
            reasoning_role=str(raw.get("reasoning_role", "general")),
            latency_profile=str(raw.get("latency_profile", "medium")),
            cost_profile=str(raw.get("cost_profile", "medium")),
            risk_profile=str(raw.get("risk_profile", "medium")),
            interpretability_score=float(raw.get("interpretability_score", 0.65)),
            calibration_score=float(raw.get("calibration_score", 0.55)),
            failure_modes=list(raw.get("failure_modes", [])),
            recovery_suggestions=list(raw.get("recovery_suggestions", [])),
            compatible_with=list(raw.get("compatible_with", [])),
            complements=list(raw.get("complements", [])),
            redundant_with=list(raw.get("redundant_with", [])),
            ideal_position_in_pipeline=str(raw.get("ideal_position_in_pipeline", "middle")),
            supports_parallelism=bool(raw.get("supports_parallelism", True)),
            success_rate_by_domain=dict(raw.get("success_rate_by_domain", {})),
            success_rate_by_agent=dict(raw.get("success_rate_by_agent", {})),
            pairwise_synergy_scores=dict(raw.get("pairwise_synergy_scores", {})),
            pairwise_redundancy_scores=dict(raw.get("pairwise_redundancy_scores", {})),
            drift_flag=bool(raw.get("drift_flag", False)),
            reputation_score=float(raw.get("reputation_score", 0.5)),
            confidence_keywords=list(raw.get("confidence_keywords", [])),
            tier=SkillTier(str(raw.get("tier", "basic"))),
            compute_cost=float(raw.get("compute_cost", 1.0)),
            synergies=list(raw.get("synergies", [])),
            conflicts=list(raw.get("conflicts", [])),
        )
        register_external_skill(
            name=name,
            fn=_make_fn(template, name),
            metadata=metadata,
            source=str(raw.get("source", "third_party")),
        )
        loaded += 1

    return loaded


def get_skill_dependencies(name: str) -> SkillDependency | None:
    """Get skill dependency and enhancement info."""

    meta = get_skill_metadata(name)
    if not meta:
        return None
    return SkillDependency(
        skill_name=name,
        depends_on=[],
        enhances=list(meta.synergies),
        conflicts_with=list(meta.conflicts),
    )


def get_skill_budget(name: str) -> SkillBudget:
    """Get abstract budget for a skill."""

    meta = get_skill_metadata(name)
    cost = meta.compute_cost if meta else 1.0
    return SkillBudget(
        compute_cost=cost,
        latency_estimate_ms=cost * 100.0,
        context_requirement=0.5,
    )


def validate_skill_metadata(metadata: SkillMetadata) -> list[str]:
    """Validate metadata quality for registry/lifecycle checks."""

    issues: list[str] = []
    if not metadata.name.strip():
        issues.append("missing_name")
    if not metadata.description.strip():
        issues.append("missing_description")
    if not metadata.strengths:
        issues.append("missing_strengths")
    if not metadata.confidence_keywords:
        issues.append("missing_confidence_keywords")
    if metadata.compute_cost <= 0:
        issues.append("invalid_compute_cost")
    if not (metadata.summary or metadata.description):
        issues.append("missing_summary")
    if metadata.interpretability_score < 0.3:
        issues.append("low_interpretability")
    return issues


def get_skill_card(name: str) -> dict[str, Any] | None:
    """Return rich SkillCard + lifecycle signals for one skill."""

    meta = get_skill_metadata(name)
    if not meta:
        return None

    from app.memory.learning import get_skill_reliability  # local import to avoid circular startup cost

    card = meta.to_skill_card()
    card["validation_issues"] = validate_skill_metadata(meta)
    card["runtime_reliability"] = round(get_skill_reliability(name), 4)
    card["lifecycle_stage"] = "active" if not card["validation_issues"] else "needs_review"
    return card


def get_skill_lifecycle_status(name: str) -> dict[str, Any]:
    """Return coarse lifecycle status used by demos/inspection."""

    card = get_skill_card(name)
    if not card:
        return {"skill": name, "status": "unknown"}
    stage = card.get("lifecycle_stage", "active")
    drift = bool(card.get("drift_flag", False))
    reliability = float(card.get("runtime_reliability", 0.5))
    if drift:
        stage = "drift_alert"
    elif reliability < 0.4:
        stage = "observation"
    return {
        "skill": name,
        "status": stage,
        "reliability": reliability,
        "validation_issues": card.get("validation_issues", []),
    }


def list_skills_by_category(category: SkillCategory) -> list[SkillMetadata]:
    """List skills by category."""

    return [meta for meta in list_all_skills() if meta.category == category]


def list_skills_by_tier(tier: SkillTier) -> list[SkillMetadata]:
    """List skills by tier."""

    return [meta for meta in list_all_skills() if meta.tier == tier]


def get_synergy_pairs() -> list[tuple[str, str, float]]:
    """Return (skill_a, skill_b, strength) tuples for known synergies."""

    pairs: list[tuple[str, str, float]] = []
    for entry in SKILL_REGISTRY.values():
        meta = entry["metadata"]
        for partner in meta.synergies:
            pairs.append((meta.name, partner, 0.8))
    for entry in EXTERNAL_SKILL_REGISTRY.values():
        meta = entry["metadata"]
        for partner in meta.synergies:
            pairs.append((meta.name, partner, 0.8))
    return pairs


def _fallback_marketplace_skill(name: str, text: str) -> str:
    return (
        f"Marketplace Skill `{name}` Output:\n"
        f"- Query interpreted: {text}\n"
        "- Evidence: synthetic local execution (replace with real tool backend)\n"
        "- Confidence: medium"
    )


def execute_skill(name: str, text: str) -> str:
    """Execute built-in, external, or marketplace-backed skills."""

    entry = SKILL_REGISTRY.get(name)
    if entry:
        return entry["fn"](text)

    ext = EXTERNAL_SKILL_REGISTRY.get(name)
    if ext:
        try:
            return ext["fn"](text)
        except Exception as exc:  # pragma: no cover - defensive
            return f"[ERROR] External skill `{name}` failed: {exc}"

    market = {skill.name for skill in list_marketplace_skill_metadata()}
    if name in market:
        return _fallback_marketplace_skill(name, text)

    return f"[ERROR] Unknown skill: {name}"
