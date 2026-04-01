"""Marketplace reputation updates and trust scoring."""

from __future__ import annotations

import math

from app.ecosystem.store import load_marketplace, save_marketplace


def record_marketplace_outcome(skill_name: str, success: bool) -> None:
    """Record one runtime outcome for a marketplace skill if present."""

    skills = load_marketplace()
    changed = False

    for item in skills:
        if item.metadata.name == skill_name:
            item.uses += 1
            if success:
                item.success += 1
                item.trending_score = min(1.0, item.trending_score + 0.01)
            else:
                item.failures += 1
                item.trending_score = max(0.0, item.trending_score - 0.01)
            changed = True
            break

    if changed:
        save_marketplace(skills)


def submit_marketplace_rating(skill_name: str, rating: float) -> bool:
    """Submit a 1-5 rating for a marketplace skill."""

    clipped = max(1.0, min(5.0, rating))
    skills = load_marketplace()

    for item in skills:
        if item.metadata.name == skill_name:
            total = item.avg_rating * item.rating_count + clipped
            item.rating_count += 1
            item.avg_rating = total / item.rating_count
            save_marketplace(skills)
            return True

    return False


def compute_provider_trust(provider_name: str) -> float:
    """Compute provider trust score from all skills of the provider."""

    skills = [item for item in load_marketplace() if item.provider == provider_name]
    if not skills:
        return 0.0

    weights = [max(item.uses, 1) for item in skills]
    weighted_scores = [item.reputation_score() * weight for item, weight in zip(skills, weights)]
    return max(0.0, min(1.0, sum(weighted_scores) / max(sum(weights), 1)))


def decay_reputation(days_since_last_use: int, current_score: float) -> float:
    """Apply exponential decay to stale reputation values."""

    decay_rate = 0.01
    return current_score * math.exp(-decay_rate * days_since_last_use)
