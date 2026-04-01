"""Predefined personality archetypes for routing behavior."""

from __future__ import annotations

from app.core.state import AgentPersonality

BERSERKER = AgentPersonality(
    risk_tolerance=1.0,
    creativity_bias=0.8,
    diversity_preference=0.2,
    confidence_threshold=0.0,
    collaboration_tendency=0.1,
    depth_vs_breadth=0.3,
)

SCHOLAR = AgentPersonality(
    risk_tolerance=0.1,
    creativity_bias=0.2,
    diversity_preference=0.9,
    confidence_threshold=0.4,
    collaboration_tendency=0.8,
    depth_vs_breadth=0.1,
)

EXPLORER = AgentPersonality(
    risk_tolerance=0.9,
    creativity_bias=0.95,
    diversity_preference=0.95,
    confidence_threshold=0.05,
    collaboration_tendency=0.6,
    depth_vs_breadth=0.9,
)

DIPLOMAT = AgentPersonality(
    risk_tolerance=0.5,
    creativity_bias=0.5,
    diversity_preference=0.7,
    confidence_threshold=0.25,
    collaboration_tendency=0.95,
    depth_vs_breadth=0.5,
)

SURGEON = AgentPersonality(
    risk_tolerance=0.2,
    creativity_bias=0.1,
    diversity_preference=0.3,
    confidence_threshold=0.5,
    collaboration_tendency=0.3,
    depth_vs_breadth=0.05,
)

ENSEMBLE_MASTER = AgentPersonality(
    risk_tolerance=0.5,
    creativity_bias=0.4,
    diversity_preference=0.8,
    confidence_threshold=0.2,
    collaboration_tendency=0.7,
    depth_vs_breadth=0.6,
)

PERSONALITY_PROFILES: dict[str, AgentPersonality] = {
    "berserker": BERSERKER,
    "scholar": SCHOLAR,
    "explorer": EXPLORER,
    "diplomat": DIPLOMAT,
    "surgeon": SURGEON,
    "ensemble_master": ENSEMBLE_MASTER,
}


def get_profile(name: str) -> AgentPersonality | None:
    """Get a personality profile by name."""

    return PERSONALITY_PROFILES.get(name.lower())


def list_profiles() -> list[str]:
    """List available personality profile names."""

    return list(PERSONALITY_PROFILES.keys())


def blend_profiles(profiles: list[tuple[str, float]]) -> AgentPersonality:
    """Blend multiple personality profiles using weighted average."""

    total_weight = sum(weight for _, weight in profiles)
    if total_weight <= 0:
        return ENSEMBLE_MASTER

    risk = creativity = diversity = confidence = collaboration = depth = 0.0

    for name, weight in profiles:
        profile = get_profile(name)
        if not profile:
            continue
        ratio = weight / total_weight
        risk += profile.risk_tolerance * ratio
        creativity += profile.creativity_bias * ratio
        diversity += profile.diversity_preference * ratio
        confidence += profile.confidence_threshold * ratio
        collaboration += profile.collaboration_tendency * ratio
        depth += profile.depth_vs_breadth * ratio

    return AgentPersonality(
        risk_tolerance=risk,
        creativity_bias=creativity,
        diversity_preference=diversity,
        confidence_threshold=confidence,
        collaboration_tendency=collaboration,
        depth_vs_breadth=depth,
    )
