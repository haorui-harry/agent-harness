"""Value scoring model for harness runs (demo-impact oriented)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.harness.manifest import ToolManifestRegistry
from app.harness.models import HarnessRun


@dataclass
class ValueDimension:
    """One value dimension used for presentation and ranking."""

    name: str
    score: float
    weight: float
    evidence: list[str] = field(default_factory=list)
    visual_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "weight": round(self.weight, 4),
            "weighted_score": round(self.score * self.weight, 4),
            "evidence": self.evidence,
            "visual_hint": self.visual_hint,
        }


class HarnessValueScorer:
    """Convert harness execution signals into a presentation-ready value card."""

    _WEIGHTS: dict[str, float] = {
        "reliability": 0.25,
        "observability": 0.20,
        "adaptability": 0.20,
        "safety": 0.20,
        "innovation": 0.15,
    }

    def score_run(self, run: HarnessRun, manifests: ToolManifestRegistry | None = None) -> dict[str, Any]:
        """Return a high-signal value card that a front-end can visualize directly."""

        dimensions = [
            self._score_reliability(run),
            self._score_observability(run),
            self._score_adaptability(run),
            self._score_safety(run),
            self._score_innovation(run, manifests),
        ]

        total = sum(item.score * item.weight for item in dimensions)
        value_index = round(total * 100.0, 2)
        band = self._band(value_index)
        hooks = self._visual_hooks(dimensions)
        narrative = self._narrative(run, value_index, band, dimensions)

        return {
            "value_index": value_index,
            "band": band,
            "dimensions": [item.to_dict() for item in dimensions],
            "narrative": narrative,
            "visual_hooks": hooks,
        }

    def _score_reliability(self, run: HarnessRun) -> ValueDimension:
        metrics = run.eval_metrics
        tool_success = float(metrics.get("tool_success_rate", 0.0))
        completion = float(metrics.get("completion_score", 0.0))
        context_reuse = float(metrics.get("context_reuse_score", 0.0))
        tool_calls = float(metrics.get("tool_calls", 0.0))
        block_count = float(metrics.get("guardrail_block_count", 0.0))
        live_success = float(metrics.get("live_agent_success", 0.0))

        block_penalty = min(0.35, block_count / max(tool_calls, 1.0) * 0.25)
        score = (
            0.55 * tool_success
            + 0.25 * completion
            + 0.15 * context_reuse
            + 0.04 * max(0.0, 1.0 - block_penalty)
            + 0.01 * live_success
        )
        score = max(0.0, min(1.0, score))

        return ValueDimension(
            name="reliability",
            score=score,
            weight=self._WEIGHTS["reliability"],
            evidence=[
                f"tool_success_rate={tool_success:.2f}",
                f"completion_score={completion:.2f}",
                f"context_reuse_score={context_reuse:.2f}",
                f"live_agent_success={live_success:.2f}",
            ],
            visual_hint="kpi_card",
        )

    def _score_observability(self, run: HarnessRun) -> ValueDimension:
        metrics = run.eval_metrics
        discovery_count = float(metrics.get("discovery_count", 0.0))
        discovery_norm = min(1.0, discovery_count / 8.0)

        metadata = run.metadata if isinstance(run.metadata, dict) else {}
        has_security = 1.0 if metadata.get("security") else 0.0
        has_recipe = 1.0 if metadata.get("recipe") else 0.0

        traced_steps = 0
        for step in run.steps:
            if step.guardrail_notes or step.discovery_notes or step.security:
                traced_steps += 1
        step_trace_ratio = traced_steps / max(len(run.steps), 1)

        score = 0.35 * discovery_norm + 0.25 * has_security + 0.20 * has_recipe + 0.20 * step_trace_ratio
        score = max(0.0, min(1.0, score))

        return ValueDimension(
            name="observability",
            score=score,
            weight=self._WEIGHTS["observability"],
            evidence=[
                f"discovery_count={discovery_count:.1f}",
                f"step_trace_ratio={step_trace_ratio:.2f}",
                f"security_trace={'yes' if has_security else 'no'}",
            ],
            visual_hint="timeline_with_annotations",
        )

    def _score_adaptability(self, run: HarnessRun) -> ValueDimension:
        metrics = run.eval_metrics
        discovery_utilization = float(metrics.get("discovery_utilization", 0.0))
        recipe_completion = float(metrics.get("recipe_completion", 0.0))

        sources = {
            step.tool_call.source
            for step in run.steps
            if step.tool_call and isinstance(step.tool_call.source, str) and step.tool_call.source
        }
        source_norm = min(1.0, len(sources) / 3.0)

        tool_types = {
            step.tool_call.tool_type.value
            for step in run.steps
            if step.tool_call and step.tool_call.tool_type
        }
        type_norm = min(1.0, len(tool_types) / 3.0)

        score = 0.35 * discovery_utilization + 0.25 * recipe_completion + 0.20 * source_norm + 0.20 * type_norm
        score = max(0.0, min(1.0, score))

        return ValueDimension(
            name="adaptability",
            score=score,
            weight=self._WEIGHTS["adaptability"],
            evidence=[
                f"discovery_utilization={discovery_utilization:.2f}",
                f"recipe_completion={recipe_completion:.2f}",
                f"source_variants={len(sources)}",
            ],
            visual_hint="strategy_heatmap",
        )

    def _score_safety(self, run: HarnessRun) -> ValueDimension:
        metrics = run.eval_metrics
        metadata = run.metadata if isinstance(run.metadata, dict) else {}
        security = metadata.get("security", {}) if isinstance(metadata.get("security", {}), dict) else {}
        action = str(security.get("preflight_action", "allow"))

        action_base = {"allow": 0.72, "challenge": 0.78, "block": 0.90}.get(action, 0.70)
        security_blocks = float(metrics.get("security_block_count", 0.0))
        security_challenges = float(metrics.get("security_challenge_count", 0.0))
        guardrail_blocks = float(metrics.get("guardrail_block_count", 0.0))

        hard_penalty = min(0.30, guardrail_blocks * 0.06)
        challenge_bonus = min(0.12, security_challenges * 0.02)
        score = action_base + challenge_bonus - hard_penalty + min(0.08, security_blocks * 0.015)
        score = max(0.0, min(1.0, score))

        findings = security.get("preflight_findings", [])
        finding_count = len(findings) if isinstance(findings, list) else 0

        return ValueDimension(
            name="safety",
            score=score,
            weight=self._WEIGHTS["safety"],
            evidence=[
                f"preflight_action={action}",
                f"security_findings={finding_count}",
                f"guardrail_block_count={guardrail_blocks:.1f}",
            ],
            visual_hint="risk_shield",
        )

    def _score_innovation(self, run: HarnessRun, manifests: ToolManifestRegistry | None = None) -> ValueDimension:
        used_tools = [step.tool_call.name for step in run.steps if step.tool_call]
        unique_tools = sorted(set(used_tools))
        tool_variety = min(1.0, len(unique_tools) / 4.0)
        live_calls = float(run.eval_metrics.get("live_agent_calls", 0.0))
        live_bonus = min(1.0, live_calls / 3.0) if live_calls > 0 else 0.0

        novelty_scores: list[float] = []
        if manifests:
            for name in unique_tools:
                manifest = manifests.get(name)
                if manifest:
                    novelty_scores.append(float(manifest.novelty_score))
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.5

        external_tool_bonus = 1.0 if "external_resource_hub" in unique_tools else 0.0

        tool_types = {
            step.tool_call.tool_type.value
            for step in run.steps
            if step.tool_call and step.tool_call.tool_type
        }
        cross_type = min(1.0, len(tool_types) / 3.0)

        score = (
            0.40 * avg_novelty
            + 0.22 * tool_variety
            + 0.18 * cross_type
            + 0.10 * external_tool_bonus
            + 0.10 * live_bonus
        )
        score = max(0.0, min(1.0, score))

        return ValueDimension(
            name="innovation",
            score=score,
            weight=self._WEIGHTS["innovation"],
            evidence=[
                f"avg_novelty={avg_novelty:.2f}",
                f"unique_tools={len(unique_tools)}",
                f"cross_type_coverage={len(tool_types)}",
                f"live_agent_calls={live_calls:.1f}",
            ],
            visual_hint="novelty_radar",
        )

    @staticmethod
    def _band(value_index: float) -> str:
        if value_index >= 85.0:
            return "platinum"
        if value_index >= 72.0:
            return "gold"
        if value_index >= 58.0:
            return "silver"
        return "bronze"

    @staticmethod
    def _visual_hooks(dimensions: list[ValueDimension]) -> list[dict[str, str]]:
        hooks: list[dict[str, str]] = []
        for item in sorted(dimensions, key=lambda dim: dim.score, reverse=True):
            hooks.append(
                {
                    "dimension": item.name,
                    "recommended_visual": item.visual_hint,
                    "headline": f"{item.name.title()} {item.score * 100:.0f}%",
                }
            )
        return hooks

    @staticmethod
    def _narrative(
        run: HarnessRun,
        value_index: float,
        band: str,
        dimensions: list[ValueDimension],
    ) -> str:
        top_two = sorted(dimensions, key=lambda item: item.score, reverse=True)[:2]
        leaders = ", ".join(item.name for item in top_two) if top_two else "none"
        return (
            f"Value Index {value_index:.1f} ({band}) with strengths in {leaders}. "
            f"Completed={run.completed}, steps={len(run.steps)}, "
            f"tool_calls={int(run.eval_metrics.get('tool_calls', 0.0))}."
        )
