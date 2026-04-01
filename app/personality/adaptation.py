"""Dynamic personality adaptation based on runtime signals."""

from __future__ import annotations

import copy

from app.core.state import AgentPersonality, QueryComplexity


class PersonalityAdaptor:
    """Adapt personality parameters using query complexity and feedback."""

    def adapt_for_complexity(
        self,
        personality: AgentPersonality,
        complexity: QueryComplexity,
    ) -> AgentPersonality:
        """Adjust personality knobs for query complexity."""

        adapted = copy.copy(personality)

        if complexity == QueryComplexity.SIMPLE:
            adapted.diversity_preference = max(0.1, adapted.diversity_preference - 0.2)
            adapted.confidence_threshold = min(0.8, adapted.confidence_threshold + 0.15)
            adapted.depth_vs_breadth = max(0.0, adapted.depth_vs_breadth - 0.2)
        elif complexity in (QueryComplexity.COMPLEX, QueryComplexity.EXPERT):
            adapted.diversity_preference = min(1.0, adapted.diversity_preference + 0.15)
            adapted.confidence_threshold = max(0.05, adapted.confidence_threshold - 0.1)
            adapted.depth_vs_breadth = min(1.0, adapted.depth_vs_breadth + 0.1)
            adapted.collaboration_tendency = min(1.0, adapted.collaboration_tendency + 0.1)

        return adapted

    def adapt_for_feedback(
        self,
        personality: AgentPersonality,
        success_rate: float,
        conflict_rate: float,
    ) -> AgentPersonality:
        """Adjust personality from historical outcome signals."""

        adapted = copy.copy(personality)

        if success_rate < 0.5:
            adapted.risk_tolerance = max(0.0, adapted.risk_tolerance - 0.15)
            adapted.confidence_threshold = min(0.8, adapted.confidence_threshold + 0.1)

        if conflict_rate > 0.3:
            adapted.diversity_preference = max(0.1, adapted.diversity_preference - 0.1)
            adapted.creativity_bias = max(0.1, adapted.creativity_bias - 0.1)

        return adapted

    def describe_adaptations(
        self,
        original: AgentPersonality,
        adapted: AgentPersonality,
    ) -> list[str]:
        """Return a human-readable list of adaptation deltas."""

        changes: list[str] = []
        fields = [
            ("risk_tolerance", "Risk tolerance"),
            ("creativity_bias", "Creativity bias"),
            ("diversity_preference", "Diversity preference"),
            ("confidence_threshold", "Confidence threshold"),
            ("collaboration_tendency", "Collaboration tendency"),
            ("depth_vs_breadth", "Depth vs breadth"),
        ]

        for attr, label in fields:
            old = getattr(original, attr)
            new = getattr(adapted, attr)
            if abs(old - new) > 0.01:
                direction = "increased" if new > old else "decreased"
                changes.append(f"{label}: {old:.2f} -> {new:.2f} ({direction})")

        return changes
