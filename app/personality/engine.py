"""Personality engine turning personality vectors into routing strategy hints."""

from __future__ import annotations

from app.core.state import AgentPersonality, RoutingStrategy


class PersonalityEngine:
    """Translate personality dimensions into concrete routing preferences."""

    def suggest_routing_strategy(self, personality: AgentPersonality) -> RoutingStrategy:
        """Recommend a routing strategy from personality profile."""

        if personality.risk_tolerance > 0.8 and personality.diversity_preference < 0.3:
            return RoutingStrategy.GREEDY
        if personality.diversity_preference > 0.7:
            return RoutingStrategy.DIVERSITY_FIRST
        if personality.collaboration_tendency > 0.7:
            return RoutingStrategy.ENSEMBLE
        if personality.depth_vs_breadth < 0.3:
            return RoutingStrategy.BUDGET_AWARE
        return RoutingStrategy.COMPLEMENTARY

    def adjust_max_skills(self, base: int, personality: AgentPersonality) -> int:
        """Adjust max skills from depth-vs-breadth preference."""

        if personality.depth_vs_breadth < 0.3:
            return max(1, base - 1)
        if personality.depth_vs_breadth > 0.7:
            return base + 1
        return base

    def adjust_redundancy_threshold(self, base: float, personality: AgentPersonality) -> float:
        """Adjust redundancy threshold based on diversity preference."""

        adjustment = (personality.diversity_preference - 0.5) * 0.3
        return max(0.3, min(0.9, base - adjustment))

    def should_include_minority_report(self, personality: AgentPersonality) -> bool:
        """Whether minority report should be included in synthesis."""

        return personality.risk_tolerance > 0.5 or personality.collaboration_tendency > 0.6

    def describe_strategy(self, personality: AgentPersonality) -> str:
        """Describe strategy implied by personality in plain language."""

        parts: list[str] = []

        if personality.risk_tolerance > 0.7:
            parts.append("willing to take risks with unproven skills")
        elif personality.risk_tolerance < 0.3:
            parts.append("prefers well-established skills only")

        if personality.creativity_bias > 0.7:
            parts.append("seeks unconventional combinations")
        elif personality.creativity_bias < 0.3:
            parts.append("sticks to standard approaches")

        if personality.diversity_preference > 0.7:
            parts.append("maximizes skill diversity")
        elif personality.diversity_preference < 0.3:
            parts.append("focuses on a few highly relevant skills")

        if personality.depth_vs_breadth < 0.3:
            parts.append("goes deep with fewer skills")
        elif personality.depth_vs_breadth > 0.7:
            parts.append("casts a wide net across many skills")

        if not parts:
            return "Strategy: balanced approach"
        return "Strategy: " + "; ".join(parts)
