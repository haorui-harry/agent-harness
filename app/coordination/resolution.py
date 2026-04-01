"""Conflict resolution strategies for multi-skill outputs."""

from __future__ import annotations

from app.core.state import ConflictPolicy


class ResolutionStrategy:
    """Resolve conflicts according to configured policy."""

    def resolve(
        self,
        conflicts: list[dict],
        outputs: dict[str, str],
        policy: ConflictPolicy,
        relevance_scores: dict[str, float] | None = None,
    ) -> dict:
        """Resolve conflicts and return selected winning perspectives."""

        if not conflicts:
            return {
                "resolved_conflicts": [],
                "resolution_method": "none_needed",
                "winning_skills": list(outputs.keys()),
                "minority_report": "",
            }

        if policy == ConflictPolicy.VOTING:
            return self._resolve_by_voting(conflicts, outputs)
        if policy == ConflictPolicy.WEIGHTED:
            return self._resolve_by_weight(conflicts, outputs, relevance_scores or {})
        if policy == ConflictPolicy.HIERARCHICAL:
            return self._resolve_by_hierarchy(conflicts, outputs, relevance_scores or {})
        if policy == ConflictPolicy.DEBATE:
            return self._resolve_by_debate(conflicts, outputs)
        return self._resolve_by_weight(conflicts, outputs, relevance_scores or {})

    def _resolve_by_voting(self, conflicts: list[dict], outputs: dict[str, str]) -> dict:
        vote_count: dict[str, int] = {name: 0 for name in outputs}
        for conflict in conflicts:
            vote_count[conflict["skill_a"]] = vote_count.get(conflict["skill_a"], 0) + 1
            vote_count[conflict["skill_b"]] = vote_count.get(conflict["skill_b"], 0) + 1

        winners = sorted(vote_count.keys(), key=lambda name: vote_count[name], reverse=True)
        losers = [name for name in winners if vote_count[name] == min(vote_count.values())]

        return {
            "resolved_conflicts": conflicts,
            "resolution_method": "voting",
            "winning_skills": winners,
            "minority_report": f"Skills with fewer votes: {', '.join(losers)}" if losers else "",
        }

    def _resolve_by_weight(
        self,
        conflicts: list[dict],
        outputs: dict[str, str],
        relevance_scores: dict[str, float],
    ) -> dict:
        winners = sorted(outputs.keys(), key=lambda name: relevance_scores.get(name, 0.5), reverse=True)
        losers = winners[len(winners) // 2 :]
        return {
            "resolved_conflicts": conflicts,
            "resolution_method": "weighted",
            "winning_skills": winners,
            "minority_report": f"Lower-weight perspectives from: {', '.join(losers)}",
        }

    def _resolve_by_hierarchy(
        self,
        conflicts: list[dict],
        outputs: dict[str, str],
        relevance_scores: dict[str, float],
    ) -> dict:
        if not outputs:
            return {
                "resolved_conflicts": conflicts,
                "resolution_method": "hierarchical",
                "winning_skills": [],
                "minority_report": "",
            }

        top_skill = max(outputs.keys(), key=lambda name: relevance_scores.get(name, 0.0))
        others = [name for name in outputs if name != top_skill]
        return {
            "resolved_conflicts": conflicts,
            "resolution_method": "hierarchical",
            "winning_skills": [top_skill],
            "minority_report": f"Overridden by {top_skill}: {', '.join(others)}",
        }

    def _resolve_by_debate(self, conflicts: list[dict], outputs: dict[str, str]) -> dict:
        debate_lines: list[str] = []
        for conflict in conflicts:
            debate_lines.append(
                f"  [{conflict['skill_a']}] says: {conflict['signal_a']} vs "
                f"[{conflict['skill_b']}] says: {conflict['signal_b']} "
                f"(severity: {conflict['severity']:.1f})"
            )

        return {
            "resolved_conflicts": conflicts,
            "resolution_method": "debate",
            "winning_skills": list(outputs.keys()),
            "minority_report": "Debate record:\n" + "\n".join(debate_lines),
        }
