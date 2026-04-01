"""Tests for the complementarity engine."""

from __future__ import annotations

from typing import Optional

from app.core.state import AgentStyle, SkillCategory, SkillMetadata, SkillTier
from app.routing.complementarity import ComplementarityEngine


def _make_skill(
    name: str,
    category: SkillCategory,
    keywords: Optional[list[str]] = None,
    strengths: Optional[list[str]] = None,
    output_type: str = "text",
    tier: SkillTier = SkillTier.BASIC,
    compute_cost: float = 1.0,
    synergies: Optional[list[str]] = None,
    conflicts: Optional[list[str]] = None,
) -> SkillMetadata:
    return SkillMetadata(
        name=name,
        description=f"Test skill: {name}",
        strengths=strengths or [name],
        weaknesses=["test"],
        category=category,
        output_type=output_type,
        confidence_keywords=keywords or [],
        tier=tier,
        compute_cost=compute_cost,
        synergies=synergies or [],
        conflicts=conflicts or [],
    )


class TestComplementarityEngine:
    def setup_method(self) -> None:
        self.engine = ComplementarityEngine(max_skills=3)
        self.skills = [
            _make_skill("risk_finder", SkillCategory.EXTRACTION, ["risk", "threat"]),
            _make_skill("summarizer", SkillCategory.COMMUNICATION, ["summarize", "overview"], output_type="text"),
            _make_skill("comparer", SkillCategory.REASONING, ["compare", "versus"], output_type="structured"),
            _make_skill("fact_puller", SkillCategory.RECALL, ["fact", "data"], output_type="list"),
            _make_skill("advisor", SkillCategory.GENERATION, ["recommend", "suggest"], output_type="list"),
        ]

    def test_selects_relevant_skills(self) -> None:
        result = self.engine.select(self.skills, "summarize and find risks")
        names = [s.metadata.name for s in result.selected]
        assert "risk_finder" in names or "summarizer" in names
        assert len(result.selected) >= 1

    def test_diversity_over_redundancy(self) -> None:
        skills = [
            _make_skill("extract_a", SkillCategory.EXTRACTION, ["risk"], strengths=["risk detection"]),
            _make_skill("extract_b", SkillCategory.EXTRACTION, ["risk", "threat"], strengths=["risk analysis"]),
            _make_skill("summarizer", SkillCategory.COMMUNICATION, ["risk", "summarize"]),
        ]
        result = self.engine.select(skills, "risk analysis", style=AgentStyle.BALANCED)
        categories = {s.metadata.category for s in result.selected}
        assert len(categories) >= min(2, len(result.selected))

    def test_aggressive_style_picks_one(self) -> None:
        result = self.engine.select(self.skills, "summarize this", style=AgentStyle.AGGRESSIVE)
        assert len(result.selected) == 1

    def test_coverage_metric(self) -> None:
        result = self.engine.select(self.skills, "summarize and compare the facts")
        assert 0.0 <= result.total_coverage <= 1.0
        assert 0.0 <= result.diversity_index

    def test_empty_skills(self) -> None:
        result = self.engine.select([], "anything")
        assert result.selected == []
        assert result.total_coverage == 0.0

    def test_pairwise_scores_computed(self) -> None:
        result = self.engine.select(self.skills, "compare options and risks")
        assert len(result.pairwise_scores) > 0
        for score in result.pairwise_scores.values():
            assert 0.0 <= score <= 1.0

    def test_synergy_bonus(self) -> None:
        skills = [
            _make_skill(
                "risk_finder",
                SkillCategory.EXTRACTION,
                ["risk"],
                synergies=["risk_recommender"],
            ),
            _make_skill(
                "risk_recommender",
                SkillCategory.GENERATION,
                ["recommend", "risk"],
                synergies=["risk_finder"],
            ),
            _make_skill("neutral", SkillCategory.COMMUNICATION, ["summary"]),
        ]
        result = self.engine.select(skills, "identify risks and recommend actions")
        selected = [s.metadata.name for s in result.selected]
        assert "risk_finder" in selected
        assert "risk_recommender" in selected

    def test_conflict_avoidance(self) -> None:
        skills = [
            _make_skill("creative", SkillCategory.GENERATION, ["idea"], conflicts=["validator"]),
            _make_skill("validator", SkillCategory.REASONING, ["validate"], conflicts=["creative"]),
            _make_skill("safe", SkillCategory.EXTRACTION, ["extract"]),
        ]
        result = self.engine.select(skills, "generate ideas and validate assumptions")
        selected = [s.metadata.name for s in result.selected]
        assert not ("creative" in selected and "validator" in selected)

    def test_budget_limit(self) -> None:
        skills = [
            _make_skill("expensive_a", SkillCategory.REASONING, ["a"], compute_cost=1.4),
            _make_skill("expensive_b", SkillCategory.ANALYSIS, ["b"], compute_cost=1.4),
            _make_skill("cheap", SkillCategory.COMMUNICATION, ["summary"], compute_cost=0.5),
        ]
        engine = ComplementarityEngine(max_skills=10, budget_limit=2.0)
        result = engine.select(skills, "analyze and summarize")
        assert result.total_budget_used <= 2.0 + 1e-9

    def test_refinement_improves_selection(self) -> None:
        engine_no_refine = ComplementarityEngine(max_skills=3, refinement_rounds=0)
        base = engine_no_refine.select(self.skills, "compare options and summarize")
        base_score = sum(item.composite_score for item in base.selected)

        engine_refine = ComplementarityEngine(max_skills=3, refinement_rounds=2)
        refined = engine_refine.select(self.skills, "compare options and summarize")
        refined_score = sum(item.composite_score for item in refined.selected)

        assert refined_score >= base_score - 0.05
