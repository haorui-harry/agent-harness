"""Post-run routing quality analyzer."""

from __future__ import annotations


class RoutingAnalyzer:
    """Analyze quality and efficiency from reasoning path and metrics."""

    def analyze(self, path: list[dict], metrics: dict[str, float]) -> dict:
        """Return a full analysis bundle."""

        return {
            "efficiency": self._analyze_efficiency(path),
            "quality": self._analyze_quality(metrics),
            "recommendations": self._generate_recommendations(path, metrics),
        }

    def _analyze_efficiency(self, path: list[dict]) -> dict:
        total_ms = path[-1]["elapsed_ms"] if path else 0.0
        event_count = len(path)
        decision_count = sum(
            1
            for item in path
            if "selected" in str(item.get("event", "")) or "rejected" in str(item.get("event", ""))
        )
        return {
            "total_time_ms": total_ms,
            "event_count": event_count,
            "decision_count": decision_count,
            "avg_time_per_decision_ms": total_ms / max(decision_count, 1),
            "rating": "fast" if total_ms < 50 else "moderate" if total_ms < 200 else "slow",
        }

    def _analyze_quality(self, metrics: dict[str, float]) -> dict:
        coverage = metrics.get("coverage", 0.0)
        redundancy = metrics.get("redundancy", 0.0)
        diversity = metrics.get("diversity_shannon", 0.0)
        coherence = metrics.get("ensemble_coherence", 0.0)
        quality_score = metrics.get("avg_quality_score", 0.0)

        overall = (
            0.25 * coverage
            + 0.20 * (1 - redundancy)
            + 0.20 * min(diversity / 1.5, 1.0)
            + 0.20 * coherence
            + 0.15 * quality_score
        )

        return {
            "coverage": coverage,
            "redundancy": redundancy,
            "diversity": diversity,
            "coherence": coherence,
            "output_quality": quality_score,
            "overall_score": round(overall, 3),
            "grade": self._score_to_grade(overall),
        }

    def _generate_recommendations(self, path: list[dict], metrics: dict[str, float]) -> list[str]:
        _ = path
        recommendations: list[str] = []

        coverage = metrics.get("coverage", 0.0)
        redundancy = metrics.get("redundancy", 0.0)
        conflict_count = metrics.get("conflict_count", 0.0)

        if coverage < 0.5:
            recommendations.append(
                "Low category coverage - consider increasing max_skills or diversity_first strategy"
            )
        if redundancy > 0.5:
            recommendations.append(
                "High redundancy - lower redundancy threshold or use a more diverse style"
            )
        if conflict_count > 1:
            recommendations.append(
                f"{int(conflict_count)} conflicts detected - enable stronger conflict avoidance"
            )

        if not recommendations:
            recommendations.append("Routing quality looks good - no immediate improvements suggested")

        return recommendations

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 0.85:
            return "A"
        if score >= 0.70:
            return "B"
        if score >= 0.55:
            return "C"
        if score >= 0.40:
            return "D"
        return "F"
