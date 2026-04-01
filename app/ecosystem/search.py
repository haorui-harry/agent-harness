"""Hybrid search over marketplace skills (cosine + BM25 + reputation)."""

from __future__ import annotations

import math
import re

from app.ecosystem.models import MarketplaceSkill


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _term_freq(tokens: list[str]) -> dict[str, float]:
    total = max(len(tokens), 1)
    out: dict[str, float] = {}
    for token in tokens:
        out[token] = out.get(token, 0.0) + 1.0 / total
    return out


def _vectorize_skill(skill: MarketplaceSkill) -> dict[str, float]:
    text = " ".join(
        [
            skill.metadata.name,
            skill.metadata.description,
            " ".join(skill.metadata.strengths),
            " ".join(skill.metadata.confidence_keywords),
            " ".join(skill.tags),
            skill.provider,
        ]
    )
    return _term_freq(_tokenize(text))


def _vectorize_query(query: str) -> dict[str, float]:
    return _term_freq(_tokenize(query))


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_doc_len: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Simplified BM25 score."""

    doc_len = len(doc_tokens)
    tf: dict[str, int] = {}
    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1

    score = 0.0
    for qt in query_tokens:
        if qt in tf:
            f = tf[qt]
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1.0))
            score += numerator / denominator
    return score


def search_marketplace_skills(
    query: str,
    skills: list[MarketplaceSkill],
    limit: int = 5,
    method: str = "hybrid",
) -> list[tuple[MarketplaceSkill, float]]:
    """Search marketplace skills by cosine, BM25, or hybrid method."""

    if not skills:
        return []

    query_tokens = _tokenize(query)
    qv = _vectorize_query(query)

    documents: dict[str, list[str]] = {}
    for skill in skills:
        docs = _tokenize(
            " ".join(
                [
                    skill.metadata.name,
                    skill.metadata.description,
                    " ".join(skill.metadata.strengths),
                    " ".join(skill.metadata.confidence_keywords),
                    " ".join(skill.tags),
                ]
            )
        )
        documents[skill.metadata.name] = docs

    avg_doc_len = sum(len(tokens) for tokens in documents.values()) / max(len(documents), 1)

    scored: list[tuple[MarketplaceSkill, float]] = []
    for skill in skills:
        cosine = _cosine(qv, _vectorize_skill(skill))
        bm25 = _bm25_score(query_tokens, documents[skill.metadata.name], avg_doc_len)
        reputation = skill.reputation_score()

        if method == "cosine":
            score = 0.80 * cosine + 0.20 * reputation
        elif method == "bm25":
            normalized_bm25 = bm25 / max(len(query_tokens), 1)
            score = 0.75 * normalized_bm25 + 0.25 * reputation
        else:
            normalized_bm25 = bm25 / max(len(query_tokens), 1)
            score = 0.45 * cosine + 0.35 * normalized_bm25 + 0.20 * reputation

        scored.append((skill, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]
