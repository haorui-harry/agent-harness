"""Routing benchmark strategies: greedy, random, complementary."""

from __future__ import annotations

import random

from app.core.state import AgentStyle
from app.routing.complementarity import ComplementarityEngine
from app.skills.registry import list_all_skills


def greedy_strategy(query: str, k: int = 3) -> list[str]:
    skills = list_all_skills()
    engine = ComplementarityEngine(max_skills=max(1, len(skills)))
    scored = engine.select(skills=skills, query=query, style=AgentStyle.BALANCED)
    ranked = sorted(scored.selected + scored.rejected, key=lambda x: x.relevance, reverse=True)
    return [item.metadata.name for item in ranked[:k]]


def random_strategy(query: str, k: int = 3, seed: int = 7) -> list[str]:
    _ = query
    pool = [s.name for s in list_all_skills()]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:k]


def complementary_strategy(query: str, k: int = 3) -> list[str]:
    skills = list_all_skills()
    engine = ComplementarityEngine(max_skills=k)
    result = engine.select(skills=skills, query=query, style=AgentStyle.BALANCED)
    return [item.metadata.name for item in result.selected]

