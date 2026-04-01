"""Marketplace services for ecosystem discovery and analytics."""

from __future__ import annotations

from app.core.state import SkillMetadata
from app.ecosystem.search import search_marketplace_skills
from app.ecosystem.store import load_marketplace


def list_marketplace_skill_metadata() -> list[SkillMetadata]:
    """Return marketplace skill metadata list."""

    return [item.metadata for item in load_marketplace()]


def discover_for_query(query: str, limit: int = 5, method: str = "hybrid") -> list[dict]:
    """Discover relevant marketplace skills for a query."""

    matches = search_marketplace_skills(query=query, skills=load_marketplace(), limit=limit, method=method)
    out: list[dict] = []
    for skill, score in matches:
        out.append(
            {
                "name": skill.metadata.name,
                "provider": skill.provider,
                "version": skill.version,
                "score": round(score, 4),
                "reputation": round(skill.reputation_score(), 4),
                "trending_score": round(skill.trending_score, 4),
                "tags": skill.tags,
            }
        )
    return out


def get_trending_skills(limit: int = 5) -> list[dict]:
    """Get top trending marketplace skills."""

    skills = sorted(load_marketplace(), key=lambda item: item.trending_score, reverse=True)
    out: list[dict] = []
    for skill in skills[:limit]:
        out.append(
            {
                "name": skill.metadata.name,
                "provider": skill.provider,
                "trending_score": round(skill.trending_score, 4),
                "installs": skill.installs,
                "uses": skill.uses,
                "rating": round(skill.avg_rating, 2),
                "tags": skill.tags,
            }
        )
    return out


def get_provider_stats(provider_name: str) -> dict:
    """Aggregate provider-level statistics across all published skills."""

    skills = [item for item in load_marketplace() if item.provider == provider_name]
    if not skills:
        return {
            "provider": provider_name,
            "count": 0,
            "installs": 0,
            "uses": 0,
            "avg_rating": 0.0,
            "avg_reputation": 0.0,
        }

    installs = sum(item.installs for item in skills)
    uses = sum(item.uses for item in skills)
    weighted_rating_total = sum(item.avg_rating * item.rating_count for item in skills)
    weighted_rating_count = sum(item.rating_count for item in skills)
    avg_rating = weighted_rating_total / max(weighted_rating_count, 1)
    avg_reputation = sum(item.reputation_score() for item in skills) / len(skills)

    return {
        "provider": provider_name,
        "count": len(skills),
        "installs": installs,
        "uses": uses,
        "avg_rating": round(avg_rating, 3),
        "avg_reputation": round(avg_reputation, 3),
        "skills": [item.metadata.name for item in skills],
    }


def list_skills_by_tag(tag: str) -> list[dict]:
    """List marketplace skills containing the specified tag."""

    if not tag:
        return []

    lowered = tag.lower()
    out: list[dict] = []
    for skill in load_marketplace():
        tags = [item.lower() for item in skill.tags]
        if lowered in tags:
            out.append(
                {
                    "name": skill.metadata.name,
                    "provider": skill.provider,
                    "version": skill.version,
                    "tags": skill.tags,
                    "reputation": round(skill.reputation_score(), 4),
                }
            )
    return out


def list_marketplace_skills() -> list[dict]:
    """Return all marketplace skills sorted by reputation."""

    skills = sorted(load_marketplace(), key=lambda item: item.reputation_score(), reverse=True)
    return [
        {
            "name": skill.metadata.name,
            "provider": skill.provider,
            "version": skill.version,
            "rating": round(skill.avg_rating, 2),
            "reputation": round(skill.reputation_score(), 4),
            "installs": skill.installs,
            "uses": skill.uses,
            "tags": skill.tags,
        }
        for skill in skills
    ]
