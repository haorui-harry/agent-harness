"""Tests for ecosystem marketplace/reputation/search features."""

from app.ecosystem.marketplace import discover_for_query, get_trending_skills
from app.ecosystem.reputation import compute_provider_trust, decay_reputation
from app.ecosystem.search import search_marketplace_skills
from app.ecosystem.store import load_marketplace


class TestEcosystem:
    def test_marketplace_has_6_skills(self) -> None:
        skills = load_marketplace()
        assert len(skills) >= 6

    def test_search_returns_ranked_results(self) -> None:
        skills = load_marketplace()
        results = search_marketplace_skills("risk analysis", skills, limit=3)
        assert len(results) <= 3
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_discover_for_query(self) -> None:
        results = discover_for_query("evidence and claims", limit=3)
        assert len(results) >= 1
        assert "name" in results[0]
        assert "score" in results[0]

    def test_provider_trust(self) -> None:
        trust = compute_provider_trust("core-labs")
        assert 0.0 <= trust <= 1.0

    def test_reputation_decay(self) -> None:
        original = 0.8
        decayed = decay_reputation(days_since_last_use=30, current_score=original)
        assert decayed < original

    def test_trending_skills(self) -> None:
        trending = get_trending_skills(limit=3)
        assert len(trending) <= 3
        if len(trending) >= 2:
            assert trending[0]["trending_score"] >= trending[1]["trending_score"]
