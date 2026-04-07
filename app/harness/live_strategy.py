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
        rule_map = [
            ("redteam_mirror", r"(jailbreak|prompt injection|exploit|attack|adversarial)"),
            ("research_analyst", r"(deep research|research report|gap analysis|comparative analysis|literature|survey|investigate|improvement report)"),
            ("risk_sentinel", r"(risk|audit|compliance|policy|governance|security|control)"),
            ("decision_theater", r"(launch|rollout|strategy|operating plan|board|decision|tradeoff|option|priorit)"),
            ("systems_architect", r"(architecture|blueprint|system design|refactor|roadmap|migration|integration)"),
        ]
        for name, pattern in rule_map:
            if re.search(pattern, text):
                return name

        if mode == "safety_critical":
            return "risk_sentinel"
        if mode == "deep":
            return "systems_architect"
        if mode == "fast":
            return "decision_theater"
        return "balanced_orchestrator"

    @staticmethod
    def _defaults() -> list[LiveStrategyProfile]:
        return [
            LiveStrategyProfile(
                name="research_analyst",
                title="Research Analyst",
                summary="Long-form research and gap-analysis strategy for deep reports, evidence review, and improvement studies.",
                analysis_system=(
                    "You are a principal research analyst. Return JSON only with keys: thesis, key_questions, findings, "
                    "competitor_map, system_gaps, delivery_targets, evidence_needs, improvement_roadmap."
                ),
                synthesis_system=(
                    "Write a deep research deliverable between 900 and 1600 words. "
                    "Adapt the document structure to match the user's request — if they ask for a memo, brief, comparison, or report, follow that format. "
                    "When no format is specified, use sections like: Summary, Current State, Analysis, Gaps, Recommendations, Open Questions. "
                    "Be concrete, comparative, technically specific, and explicit about evidence limits."
                ),
                critique_system=(
                    "Act as a harsh research review committee. Return JSON only with keys: confidence, blind_spots, "
                    "red_flags, improve, missing_controls, delivery_gaps."
                ),
                temperature_bias=0.01,
                max_tokens_bias=320,
                tags=["research", "delivery", "report"],
            ),
            LiveStrategyProfile(
                name="balanced_orchestrator",
                title="Balanced Orchestrator",
                summary="General-purpose strategy balancing depth, reliability, and execution speed.",
                analysis_system=(
                    "You are a reliability-first analyst. "
                    "Return compact JSON only with keys: thesis, key_risks, missing_evidence, "
                    "best_moves, expected_value."
                ),
                synthesis_system=(
                    "You are a task completion agent. "
                    "Produce a high-quality answer that directly addresses the user's request. "
                    "Choose the output format that best fits the task (report, comparison table, action plan, analysis, etc.). "
                    "Do NOT force a fixed template — adapt the structure to the question. "
                    "Be concrete, cite evidence when available, and state limitations honestly."
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
                    "Generate a clear executive memo that directly answers the user's question. "
                    "Include sections for: Decision, Key Risks, Controls, Action Plan, and Confidence Level. "
                    "Adapt the depth and scope to the actual question asked."
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
                    "Produce a technical architecture deliverable that addresses the user's request. "
                    "Use sections appropriate to the task (e.g., Target State, Migration Path, Dependencies, Risk Mitigation). "
                    "Be specific about implementation details, not generic phases."
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
                    "You are a launch strategist writing for a board, product lead, and risk owner at the same time. "
                    "Return JSON with: thesis, target_users, options, key_risks, missing_evidence, winner_hypothesis, "
                    "launch_phases, proof_points, controls, expected_value."
                ),
                synthesis_system=(
                    "Write a decision-oriented deliverable that helps the reader choose and act. "
                    "Structure around: What's the decision, What are the options with tradeoffs, "
                    "What evidence supports the recommendation, What should happen next. "
                    "Adapt the format to the user's actual question."
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
