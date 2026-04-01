"""Evaluation runner for routing benchmark."""

from __future__ import annotations

from app.benchmark.dataset import load_dataset
from app.benchmark.strategies import complementary_strategy, greedy_strategy, random_strategy


def jaccard(pred: list[str], gold: list[str]) -> float:
    a = set(pred)
    b = set(gold)
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def run_benchmark() -> dict:
    items = load_dataset()
    if not items:
        return {"greedy": 0.0, "random": 0.0, "complementary": 0.0, "count": 0}

    scores = {"greedy": 0.0, "random": 0.0, "complementary": 0.0}
    details = []
    for row in items:
        query = row["query"]
        gold = row["optimal_skills"]
        greedy = greedy_strategy(query)
        random_pick = random_strategy(query)
        comp = complementary_strategy(query)
        g = jaccard(greedy, gold)
        r = jaccard(random_pick, gold)
        c = jaccard(comp, gold)
        scores["greedy"] += g
        scores["random"] += r
        scores["complementary"] += c
        details.append(
            {
                "query": query,
                "gold": gold,
                "greedy": greedy,
                "random": random_pick,
                "complementary": comp,
                "scores": {"greedy": g, "random": r, "complementary": c},
            }
        )

    count = len(items)
    return {
        "count": count,
        "greedy": scores["greedy"] / count,
        "random": scores["random"] / count,
        "complementary": scores["complementary"] / count,
        "details": details,
    }

