"""Conflict detection across multiple skill outputs."""

from __future__ import annotations

import re
from itertools import combinations


class ConflictDetector:
    """Detect semantic conflicts between skill outputs."""

    OPPOSITION_PAIRS: list[tuple[list[str], list[str]]] = [
        (["high risk", "critical", "severe", "danger"], ["low risk", "minimal", "safe", "stable"]),
        (["recommend", "should", "must"], ["avoid", "should not", "must not", "do not"]),
        (["increasing", "growing", "rising"], ["decreasing", "declining", "falling"]),
        (["strong", "robust", "solid"], ["weak", "fragile", "vulnerable"]),
        (["positive", "optimistic", "favorable"], ["negative", "pessimistic", "unfavorable"]),
    ]

    def detect(self, outputs: dict[str, str]) -> list[dict]:
        """Detect conflicts across all output pairs."""

        conflicts: list[dict] = []
        names = list(outputs.keys())

        for name_a, name_b in combinations(names, 2):
            text_a = outputs[name_a].lower()
            text_b = outputs[name_b].lower()

            for pos_group, neg_group in self.OPPOSITION_PAIRS:
                a_has_pos = any(signal in text_a for signal in pos_group)
                a_has_neg = any(signal in text_a for signal in neg_group)
                b_has_pos = any(signal in text_b for signal in pos_group)
                b_has_neg = any(signal in text_b for signal in neg_group)

                if (a_has_pos and b_has_neg) or (a_has_neg and b_has_pos):
                    signal_a = next((s for s in pos_group + neg_group if s in text_a), "")
                    signal_b = next((s for s in pos_group + neg_group if s in text_b), "")
                    conflicts.append(
                        {
                            "type": "CONTRADICTORY",
                            "skill_a": name_a,
                            "skill_b": name_b,
                            "signal_a": signal_a,
                            "signal_b": signal_b,
                            "severity": 0.8,
                        }
                    )

            sentiment_a = self._simple_sentiment(text_a)
            sentiment_b = self._simple_sentiment(text_b)
            if abs(sentiment_a - sentiment_b) > 0.6:
                conflicts.append(
                    {
                        "type": "EMPHASIS_DIVERGENCE",
                        "skill_a": name_a,
                        "skill_b": name_b,
                        "signal_a": f"sentiment={sentiment_a:.2f}",
                        "signal_b": f"sentiment={sentiment_b:.2f}",
                        "severity": 0.5,
                    }
                )

        return conflicts

    def _simple_sentiment(self, text: str) -> float:
        """Tiny lexicon sentiment score in [0,1]."""

        positive_words = {"good", "great", "strong", "positive", "success", "safe", "stable", "improve"}
        negative_words = {"bad", "poor", "weak", "negative", "failure", "risk", "danger", "decline"}
        words = set(re.findall(r"\w+", text))
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        total = pos + neg
        if total == 0:
            return 0.5
        return pos / total
