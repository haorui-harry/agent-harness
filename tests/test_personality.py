"""Tests for personality profiles and adaptation."""

from app.core.state import QueryComplexity, RoutingStrategy
from app.personality.adaptation import PersonalityAdaptor
from app.personality.engine import PersonalityEngine
from app.personality.profiles import blend_profiles, get_profile, list_profiles


class TestPersonalityProfiles:
    def test_get_known_profile(self) -> None:
        profile = get_profile("berserker")
        assert profile is not None
        assert profile.risk_tolerance == 1.0

    def test_blend_profiles(self) -> None:
        blended = blend_profiles([("scholar", 0.7), ("explorer", 0.3)])
        assert 0.1 < blended.risk_tolerance < 0.9

    def test_list_profiles(self) -> None:
        profiles = list_profiles()
        assert len(profiles) >= 6


class TestPersonalityEngine:
    def test_suggest_greedy_for_berserker(self) -> None:
        engine = PersonalityEngine()
        profile = get_profile("berserker")
        strategy = engine.suggest_routing_strategy(profile)
        assert strategy == RoutingStrategy.GREEDY

    def test_suggest_diversity_for_explorer(self) -> None:
        engine = PersonalityEngine()
        profile = get_profile("explorer")
        strategy = engine.suggest_routing_strategy(profile)
        assert strategy == RoutingStrategy.DIVERSITY_FIRST


class TestPersonalityAdaptation:
    def test_simple_query_increases_confidence_threshold(self) -> None:
        adaptor = PersonalityAdaptor()
        original = get_profile("ensemble_master")
        adapted = adaptor.adapt_for_complexity(original, QueryComplexity.SIMPLE)
        assert adapted.confidence_threshold > original.confidence_threshold
