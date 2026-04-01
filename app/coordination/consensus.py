"""Consensus builder for synthesizing multi-skill outputs."""

from __future__ import annotations

import re


class ConsensusBuilder:
    """Extract shared themes and consensus strength from skill outputs."""

    def build(self, outputs: dict[str, str], conflicts: list[dict]) -> dict:
        """Build consensus report from outputs + conflict list."""

        if not outputs:
            return {
                "shared_themes": [],
                "unique_contributions": {},
                "agreement_ratio": 0.0,
                "strength": "weak",
                "strength_score": 0.0,
            }

        keyword_sets: dict[str, set[str]] = {}
        for name, text in outputs.items():
            keyword_sets[name] = self._extract_keywords(text)

        all_keywords: dict[str, int] = {}
        for kw_set in keyword_sets.values():
            for keyword in kw_set:
                all_keywords[keyword] = all_keywords.get(keyword, 0) + 1

        threshold = max(len(outputs) * 0.5, 1)
        shared_themes = [keyword for keyword, count in all_keywords.items() if count >= threshold]
        shared_themes.sort(key=lambda keyword: all_keywords[keyword], reverse=True)

        unique_contributions: dict[str, list[str]] = {}
        for name, kw_set in keyword_sets.items():
            unique_kws = [keyword for keyword in kw_set if all_keywords.get(keyword, 0) == 1]
            unique_contributions[name] = unique_kws[:5]

        if len(outputs) <= 1:
            agreement = 1.0
        else:
            max_possible_conflicts = len(outputs) * (len(outputs) - 1) / 2
            conflict_ratio = len(conflicts) / max(max_possible_conflicts, 1)
            agreement = 1.0 - conflict_ratio

        theme_strength = min(len(shared_themes) / 5.0, 1.0)
        strength_score = 0.6 * agreement + 0.4 * theme_strength

        if strength_score > 0.7:
            strength = "strong"
        elif strength_score > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        return {
            "shared_themes": shared_themes[:10],
            "unique_contributions": unique_contributions,
            "agreement_ratio": round(agreement, 3),
            "strength": strength,
            "strength_score": round(strength_score, 3),
        }

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract coarse keywords via lightweight token frequency."""

        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "of",
            "in",
            "to",
            "for",
            "with",
            "on",
            "at",
            "from",
            "by",
            "and",
            "or",
            "but",
            "not",
            "no",
            "this",
            "that",
            "it",
            "its",
            "skill",
            "query",
            "output",
            "text",
        }

        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        filtered = [word for word in words if word not in stop_words]
        freq: dict[str, int] = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        top_words = sorted(freq.keys(), key=lambda word: freq[word], reverse=True)
        return set(top_words[:15])
