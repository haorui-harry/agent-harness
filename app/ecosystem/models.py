"""Skill ecosystem domain models."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.state import SkillCategory, SkillMetadata, SkillTier


@dataclass
class ProviderProfile:
    """Information about a marketplace provider."""

    provider_id: str
    name: str
    description: str
    trust_score: float = 0.5
    skills_published: int = 0
    total_installs: int = 0


@dataclass
class SkillVersion:
    """Version record for a skill."""

    version: str
    changelog: str
    published_date: str
    breaking_changes: bool = False


@dataclass
class MarketplaceSkill:
    """Skill published in the local marketplace."""

    skill_id: str
    provider: str
    version: str
    metadata: SkillMetadata
    tags: list[str] = field(default_factory=list)
    installs: int = 0
    uses: int = 0
    success: int = 0
    failures: int = 0
    avg_rating: float = 0.0
    rating_count: int = 0

    version_history: list[dict] = field(default_factory=list)
    compatibility: list[str] = field(default_factory=list)
    last_updated: str = ""
    trending_score: float = 0.0

    def reputation_score(self) -> float:
        """Unified reputation score for ranking and discovery."""

        usage_term = min(self.uses / 100.0, 1.0)
        reliability = self.success / max(self.success + self.failures, 1)
        rating_term = self.avg_rating / 5.0
        install_term = min(self.installs / 200.0, 1.0)
        trending_term = min(self.trending_score / 0.5, 1.0)
        return (
            0.30 * reliability
            + 0.25 * rating_term
            + 0.20 * usage_term
            + 0.10 * install_term
            + 0.15 * trending_term
        )


def from_dict(payload: dict) -> MarketplaceSkill:
    """Deserialize MarketplaceSkill from dict payload."""

    metadata_raw = payload["metadata"]
    metadata = SkillMetadata(
        name=metadata_raw["name"],
        description=metadata_raw["description"],
        strengths=metadata_raw["strengths"],
        weaknesses=metadata_raw["weaknesses"],
        category=SkillCategory(metadata_raw["category"]),
        output_type=metadata_raw["output_type"],
        skill_id=str(metadata_raw.get("skill_id", payload.get("skill_id", metadata_raw["name"]))),
        owner=str(metadata_raw.get("owner", payload.get("provider", "marketplace"))),
        version=str(metadata_raw.get("version", payload.get("version", "1.0.0"))),
        summary=str(metadata_raw.get("summary", metadata_raw.get("description", ""))),
        applicable_tasks=list(metadata_raw.get("applicable_tasks", [])),
        contraindications=list(metadata_raw.get("contraindications", [])),
        required_inputs=list(metadata_raw.get("required_inputs", [])),
        optional_inputs=list(metadata_raw.get("optional_inputs", [])),
        output_schema=dict(metadata_raw.get("output_schema", {})),
        evidence_style=str(metadata_raw.get("evidence_style", "heuristic")),
        reasoning_role=str(metadata_raw.get("reasoning_role", "general")),
        latency_profile=str(metadata_raw.get("latency_profile", "medium")),
        cost_profile=str(metadata_raw.get("cost_profile", "medium")),
        risk_profile=str(metadata_raw.get("risk_profile", "medium")),
        interpretability_score=float(metadata_raw.get("interpretability_score", 0.7)),
        calibration_score=float(metadata_raw.get("calibration_score", 0.6)),
        failure_modes=list(metadata_raw.get("failure_modes", [])),
        recovery_suggestions=list(metadata_raw.get("recovery_suggestions", [])),
        compatible_with=list(metadata_raw.get("compatible_with", [])),
        complements=list(metadata_raw.get("complements", [])),
        redundant_with=list(metadata_raw.get("redundant_with", [])),
        ideal_position_in_pipeline=str(metadata_raw.get("ideal_position_in_pipeline", "middle")),
        supports_parallelism=bool(metadata_raw.get("supports_parallelism", True)),
        success_rate_by_domain=dict(metadata_raw.get("success_rate_by_domain", {})),
        success_rate_by_agent=dict(metadata_raw.get("success_rate_by_agent", {})),
        pairwise_synergy_scores=dict(metadata_raw.get("pairwise_synergy_scores", {})),
        pairwise_redundancy_scores=dict(metadata_raw.get("pairwise_redundancy_scores", {})),
        drift_flag=bool(metadata_raw.get("drift_flag", False)),
        reputation_score=float(metadata_raw.get("reputation_score", 0.5)),
        confidence_keywords=metadata_raw.get("confidence_keywords", []),
        tier=SkillTier(metadata_raw.get("tier", "basic")),
        compute_cost=float(metadata_raw.get("compute_cost", 1.0)),
        synergies=list(metadata_raw.get("synergies", [])),
        conflicts=list(metadata_raw.get("conflicts", [])),
        min_context_length=int(metadata_raw.get("min_context_length", 10)),
        max_context_length=int(metadata_raw.get("max_context_length", 10000)),
    )

    return MarketplaceSkill(
        skill_id=payload["skill_id"],
        provider=payload["provider"],
        version=payload["version"],
        metadata=metadata,
        tags=payload.get("tags", []),
        installs=int(payload.get("installs", 0)),
        uses=int(payload.get("uses", 0)),
        success=int(payload.get("success", 0)),
        failures=int(payload.get("failures", 0)),
        avg_rating=float(payload.get("avg_rating", 0.0)),
        rating_count=int(payload.get("rating_count", 0)),
        version_history=list(payload.get("version_history", [])),
        compatibility=list(payload.get("compatibility", [])),
        last_updated=str(payload.get("last_updated", "")),
        trending_score=float(payload.get("trending_score", 0.0)),
    )


def to_dict(skill: MarketplaceSkill) -> dict:
    """Serialize MarketplaceSkill into a JSON-safe dict."""

    return {
        "skill_id": skill.skill_id,
        "provider": skill.provider,
        "version": skill.version,
        "tags": skill.tags,
        "installs": skill.installs,
        "uses": skill.uses,
        "success": skill.success,
        "failures": skill.failures,
        "avg_rating": skill.avg_rating,
        "rating_count": skill.rating_count,
        "version_history": skill.version_history,
        "compatibility": skill.compatibility,
        "last_updated": skill.last_updated,
        "trending_score": skill.trending_score,
        "metadata": {
            "name": skill.metadata.name,
            "description": skill.metadata.description,
            "strengths": skill.metadata.strengths,
            "weaknesses": skill.metadata.weaknesses,
            "category": skill.metadata.category.value,
            "output_type": skill.metadata.output_type,
            "skill_id": skill.metadata.skill_id or skill.skill_id,
            "owner": skill.metadata.owner,
            "version": skill.metadata.version or skill.version,
            "summary": skill.metadata.summary,
            "applicable_tasks": skill.metadata.applicable_tasks,
            "contraindications": skill.metadata.contraindications,
            "required_inputs": skill.metadata.required_inputs,
            "optional_inputs": skill.metadata.optional_inputs,
            "output_schema": skill.metadata.output_schema,
            "evidence_style": skill.metadata.evidence_style,
            "reasoning_role": skill.metadata.reasoning_role,
            "latency_profile": skill.metadata.latency_profile,
            "cost_profile": skill.metadata.cost_profile,
            "risk_profile": skill.metadata.risk_profile,
            "interpretability_score": skill.metadata.interpretability_score,
            "calibration_score": skill.metadata.calibration_score,
            "failure_modes": skill.metadata.failure_modes,
            "recovery_suggestions": skill.metadata.recovery_suggestions,
            "compatible_with": skill.metadata.compatible_with,
            "complements": skill.metadata.complements,
            "redundant_with": skill.metadata.redundant_with,
            "ideal_position_in_pipeline": skill.metadata.ideal_position_in_pipeline,
            "supports_parallelism": skill.metadata.supports_parallelism,
            "success_rate_by_domain": skill.metadata.success_rate_by_domain,
            "success_rate_by_agent": skill.metadata.success_rate_by_agent,
            "pairwise_synergy_scores": skill.metadata.pairwise_synergy_scores,
            "pairwise_redundancy_scores": skill.metadata.pairwise_redundancy_scores,
            "drift_flag": skill.metadata.drift_flag,
            "reputation_score": skill.metadata.reputation_score,
            "confidence_keywords": skill.metadata.confidence_keywords,
            "tier": skill.metadata.tier.value,
            "compute_cost": skill.metadata.compute_cost,
            "synergies": skill.metadata.synergies,
            "conflicts": skill.metadata.conflicts,
            "min_context_length": skill.metadata.min_context_length,
            "max_context_length": skill.metadata.max_context_length,
        },
    }
