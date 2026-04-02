"""Strategy profiles for live-agent prompt orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LiveStrategyProfile:
    """Prompt strategy profile for analyze/synthesize/critique stages."""

    name: str
    title: str
    summary: str
    analysis_system: str
    synthesis_system: str
    critique_system: str
    temperature_bias: float = 0.0
    max_tokens_bias: int = 0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "temperature_bias": round(self.temperature_bias, 4),
            "max_tokens_bias": self.max_tokens_bias,
            "tags": list(self.tags),
        }


class LiveStrategyRegistry:
    """Built-in strategy profile registry and selection policy."""

    def __init__(self) -> None:
        self._profiles = {item.name: item for item in self._defaults()}

    def get(self, name: str) -> LiveStrategyProfile | None:
        return self._profiles.get(name.strip()) if name else None

    def list_cards(self) -> list[dict[str, Any]]:
        cards = [item.to_dict() for item in self._profiles.values()]
        cards.sort(key=lambda item: str(item.get("name", "")))
        return cards

    def resolve(
        self,
        query: str,
        mode: str,
        preferred: str = "",
        champion: dict[str, Any] | None = None,
    ) -> tuple[LiveStrategyProfile, str, str]:
        if preferred:
            profile = self.get(preferred)
            if profile:
                return profile, "manual", "explicit CLI/runtime override"

        champion_name = str((champion or {}).get("name", "")).strip()
        if champion_name:
            champion_profile = self.get(champion_name)
            if champion_profile:
                return champion_profile, "champion", f"selected from stored champion ({champion_name})"

        heuristic_name = self._heuristic_profile_name(query=query, mode=mode)
        profile = self.get(heuristic_name) or self.get("balanced_orchestrator")
        assert profile is not None  # pragma: no cover - defaults ensure this
        return profile, "heuristic", f"auto-selected by mode={mode}"

    def _heuristic_profile_name(self, query: str, mode: str) -> str:
        text = query.lower().strip()
        if mode == "safety_critical":
            return "risk_sentinel"
        if mode == "deep":
            return "systems_architect"
        if mode == "fast":
            return "decision_theater"

        rule_map = [
            ("redteam_mirror", r"(jailbreak|prompt injection|exploit|attack|adversarial)"),
            ("risk_sentinel", r"(risk|audit|compliance|policy|governance|security|control)"),
            ("systems_architect", r"(architecture|blueprint|system design|refactor|roadmap)"),
            ("decision_theater", r"(compare|option|tradeoff|decision|priorit)"),
        ]
        for name, pattern in rule_map:
            if re.search(pattern, text):
                return name
        return "balanced_orchestrator"

    @staticmethod
    def _defaults() -> list[LiveStrategyProfile]:
        return [
            LiveStrategyProfile(
                name="balanced_orchestrator",
                title="Balanced Orchestrator",
                summary="General-purpose strategy balancing depth, reliability, and execution speed.",
                analysis_system=(
                    "You are a reliability-first agent architect. "
                    "Return compact JSON only with keys: thesis, key_risks, missing_evidence, "
                    "best_moves, expected_value."
                ),
                synthesis_system=(
                    "You are the final synthesis agent in Agent Harness. "
                    "Produce a high-signal answer with sections: Executive Take, Decisions, Risks, "
                    "Action Plan, Confidence & Limits."
                ),
                critique_system=(
                    "You are a strict reviewer. Return JSON with keys: confidence (0-1), "
                    "blind_spots (list), red_flags (list), improve (list)."
                ),
                tags=["balanced", "general", "production"],
            ),
            LiveStrategyProfile(
                name="risk_sentinel",
                title="Risk Sentinel",
                summary="Safety-heavy strategy emphasizing controls, compliance, and failure containment.",
                analysis_system=(
                    "You are an enterprise risk committee in one agent. "
                    "Return JSON only with keys: thesis, key_risks, missing_evidence, controls, expected_value."
                ),
                synthesis_system=(
                    "Generate an executive memo with sections: Decision, Control Matrix, Residual Risk, "
                    "Escalation Gates, 72h Action Plan, Confidence."
                ),
                critique_system=(
                    "Audit the memo harshly. Return JSON: confidence, blind_spots, red_flags, improve, "
                    "policy_gaps, legal_gaps."
                ),
                temperature_bias=-0.04,
                tags=["safety", "compliance", "enterprise"],
            ),
            LiveStrategyProfile(
                name="systems_architect",
                title="Systems Architect",
                summary="Design-heavy strategy for architecture, integration, and evolution planning.",
                analysis_system=(
                    "You are a principal architect. Return JSON with: thesis, bottlenecks, missing_evidence, "
                    "target_architecture, migration_path, expected_value."
                ),
                synthesis_system=(
                    "Produce a technical architecture blueprint with sections: Target State, Phased Migration, "
                    "Dependency Map, Failure Modes, Guardrails, Next 2 Iterations."
                ),
                critique_system=(
                    "Act as architecture review board. Return JSON: confidence, blind_spots, red_flags, "
                    "improve, scalability_risks, operability_risks."
                ),
                temperature_bias=0.02,
                max_tokens_bias=180,
                tags=["architecture", "deep", "roadmap"],
            ),
            LiveStrategyProfile(
                name="decision_theater",
                title="Decision Theater",
                summary="Decision-centric strategy focused on scenario tradeoffs and recommendation clarity.",
                analysis_system=(
                    "You are a strategy decision analyst. Return JSON with: thesis, options, key_risks, "
                    "missing_evidence, winner_hypothesis, expected_value."
                ),
                synthesis_system=(
                    "Write a boardroom decision brief with sections: Options Compared, Winner, Why Not Others, "
                    "Risk Hedge, Metrics To Watch."
                ),
                critique_system=(
                    "Return JSON only: confidence, blind_spots, red_flags, improve, bias_checks."
                ),
                temperature_bias=0.03,
                tags=["decision", "tradeoff", "exec"],
            ),
            LiveStrategyProfile(
                name="redteam_mirror",
                title="Red Team Mirror",
                summary="Adversarial strategy that stress-tests reasoning and exposes hidden vulnerabilities.",
                analysis_system=(
                    "You are an adversarial red-team analyst. Return JSON only with: thesis, attack_vectors, "
                    "key_risks, missing_evidence, exploit_paths, expected_value."
                ),
                synthesis_system=(
                    "Produce a defense-oriented answer with sections: Core Position, Attack Surface, "
                    "Countermeasures, Detection Signals, Hardening Plan."
                ),
                critique_system=(
                    "Perform final red-team review. Return JSON: confidence, blind_spots, red_flags, improve, "
                    "remaining_attack_vectors."
                ),
                temperature_bias=-0.02,
                tags=["adversarial", "security", "robustness"],
            ),
        ]


def compute_efficiency(delta_value_index: float, avg_calls: float) -> float:
    """Value gain per model call used."""

    if avg_calls <= 0:
        return 0.0
    return round(delta_value_index / avg_calls, 4)
