"""Tests for conflict detection, consensus, and resolution."""

from app.coordination.conflicts import ConflictDetector
from app.coordination.consensus import ConsensusBuilder
from app.coordination.resolution import ResolutionStrategy
from app.core.state import ConflictPolicy


class TestConflictDetector:
    def test_detects_contradictory_signals(self) -> None:
        detector = ConflictDetector()
        outputs = {
            "skill_a": "The situation is high risk and critical.",
            "skill_b": "The situation is safe and stable.",
        }
        conflicts = detector.detect(outputs)
        assert len(conflicts) >= 1
        assert conflicts[0]["type"] == "CONTRADICTORY"

    def test_no_conflicts_when_aligned(self) -> None:
        detector = ConflictDetector()
        outputs = {
            "skill_a": "Key risks identified in the data.",
            "skill_b": "Risk factors extracted from the report.",
        }
        conflicts = detector.detect(outputs)
        contradictory = [c for c in conflicts if c["type"] == "CONTRADICTORY"]
        assert len(contradictory) == 0


class TestConsensusBuilder:
    def test_strong_consensus_when_aligned(self) -> None:
        builder = ConsensusBuilder()
        outputs = {
            "skill_a": "Risk assessment shows critical vulnerabilities.",
            "skill_b": "Risk analysis reveals significant vulnerabilities.",
        }
        result = builder.build(outputs, conflicts=[])
        assert result["strength"] in ("strong", "moderate")

    def test_shared_themes_extracted(self) -> None:
        builder = ConsensusBuilder()
        outputs = {
            "skill_a": "Revenue growth and market expansion are strong.",
            "skill_b": "Revenue trends show growth in key markets.",
        }
        result = builder.build(outputs, conflicts=[])
        themes = result["shared_themes"]
        assert any("revenue" in t or "growth" in t or "market" in t for t in themes)


class TestResolutionStrategy:
    def test_debate_preserves_all_perspectives(self) -> None:
        strategy = ResolutionStrategy()
        conflicts = [
            {
                "type": "CONTRADICTORY",
                "skill_a": "a",
                "skill_b": "b",
                "signal_a": "high risk",
                "signal_b": "low risk",
                "severity": 0.8,
            }
        ]
        result = strategy.resolve(conflicts, {"a": "...", "b": "..."}, ConflictPolicy.DEBATE)
        assert len(result["winning_skills"]) == 2
