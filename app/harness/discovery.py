"""Dynamic tool discovery and scoring for the harness."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.harness.manifest import HarnessToolManifest, ToolManifestRegistry
from app.harness.models import HarnessConstraints, ToolCall


@dataclass
class DiscoveredTool:
    """Scored discovery result for a tool candidate."""

    name: str
    score: float
    reasons: list[str] = field(default_factory=list)
    manifest: HarnessToolManifest | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "manifest": self.manifest.to_dict() if self.manifest else {},
        }


class ToolDiscoveryEngine:
    """Rank tools by intent fit, risk fit, and operational constraints."""

    _INTENT_KEYWORDS: dict[str, list[str]] = {
        "risk": ["risk", "safety", "audit", "threat", "danger", "compliance", "governance"],
        "comparison": ["compare", "alternative", "trade-off", "versus", "option"],
        "research": ["research", "trend", "landscape", "latest", "ecosystem", "hot"],
        "implementation": ["implement", "code", "write", "refactor", "integration", "patch", "test", "fix"],
        "planning": ["plan", "schedule", "roadmap", "milestone", "sequence"],
        "context": ["memory", "history", "context", "state", "previous", "workspace", "artifact", "file"],
        "design": ["design", "architecture", "blueprint", "pattern"],
        "operations": ["daily", "routine", "meeting", "todo", "task", "workflow", "productivity"],
        "evaluation": ["benchmark", "evaluate", "experiment", "ablation", "reproducible", "study", "paper"],
        "creative": ["creative", "design", "ui", "ux", "brand", "visual", "campaign", "presentation"],
        "enterprise": ["enterprise", "stakeholder", "board", "communication", "governance", "compliance"],
    }

    def __init__(self, manifests: ToolManifestRegistry) -> None:
        self._manifests = manifests

    def infer_intents(self, query: str) -> list[str]:
        lowered = query.lower()
        intents: list[str] = []
        for intent, keywords in self._INTENT_KEYWORDS.items():
            if any(token in lowered for token in keywords):
                intents.append(intent)
        if not intents:
            intents = ["research", "planning"]
        return intents

    def discover(
        self,
        query: str,
        constraints: HarnessConstraints,
        mode: str = "balanced",
        limit: int = 8,
        available_tools: set[str] | None = None,
    ) -> list[DiscoveredTool]:
        """Return ranked tool candidates for this query."""

        intents = self.infer_intents(query)
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        available = available_tools or set()
        out: list[DiscoveredTool] = []

        for manifest in self._manifests.list_all():
            if available and manifest.name not in available:
                continue

            reasons: list[str] = []
            score = 0.0

            score += manifest.reliability_score * 0.25
            score += manifest.novelty_score * 0.15
            score += (2.0 - manifest.latency_score) * 0.10
            score += (2.0 - manifest.cost_score) * 0.08

            matched_intents = [intent for intent in intents if intent in manifest.intents]
            if matched_intents:
                score += 0.18 * len(matched_intents)
                reasons.append(f"intent_match:{','.join(matched_intents)}")

            searchable = set(item.lower() for item in (manifest.tags + manifest.capabilities))
            overlap = sorted(list(tokens & searchable))
            if overlap:
                score += min(0.35, 0.06 * len(overlap))
                reasons.append(f"token_overlap:{','.join(overlap[:4])}")

            if manifest.name in constraints.blocked_tools:
                score -= 2.0
                reasons.append("blocked_by_constraints")

            if manifest.write_actions and not constraints.allow_write_actions:
                score -= 1.2
                reasons.append("write_disallowed")

            if manifest.network_actions and not constraints.allow_network_actions:
                score -= 1.0
                reasons.append("network_disallowed")

            if manifest.code_execution and not constraints.allow_code_execution:
                score -= 0.8
                reasons.append("code_exec_disallowed")

            if manifest.tool_type.value == "browser" and not constraints.allow_browser_actions:
                score -= 0.8
                reasons.append("browser_disallowed")

            if mode == "safety_critical" and "risk" in manifest.tags:
                score += 0.28
                reasons.append("mode_boost:safety_critical")
            elif mode == "deep" and "analysis" in manifest.tags:
                score += 0.18
                reasons.append("mode_boost:deep_analysis")
            elif mode == "fast" and manifest.latency_score <= 0.8:
                score += 0.12
                reasons.append("mode_boost:fast_path")

            out.append(
                DiscoveredTool(
                    name=manifest.name,
                    score=score,
                    reasons=reasons,
                    manifest=manifest,
                )
            )

        out.sort(key=lambda item: item.score, reverse=True)
        return out[:limit]

    def build_tool_call(
        self,
        tool_name: str,
        query: str,
        step: int,
        args_override: dict | None = None,
        source: str = "discovery",
        score: float = 0.0,
    ) -> ToolCall | None:
        """Build a typed ToolCall from manifest defaults + overrides."""

        manifest = self._manifests.get(tool_name)
        if not manifest:
            return None

        args = dict(manifest.default_args)
        if args_override:
            args.update(args_override)

        if "query" not in args:
            args["query"] = query
        args.setdefault("step", step)

        return ToolCall(
            name=manifest.name,
            tool_type=manifest.tool_type,
            args=args,
            source=source,
            score=score,
        )

    def recommend_for_step(
        self,
        query: str,
        step: int,
        discovered: list[DiscoveredTool],
        used_tools: set[str],
        plan: list[str] | None = None,
    ) -> ToolCall | None:
        """Choose one candidate for the current loop step."""

        if not discovered:
            return None

        step_bias = ""
        if step == 1:
            step_bias = "discovery"
        elif step == 2:
            step_bias = "analysis"
        elif step >= 3:
            step_bias = "design"

        for item in discovered:
            if item.name in used_tools:
                continue
            manifest = item.manifest
            if not manifest:
                continue
            if step_bias and (step_bias not in manifest.tags and step_bias not in manifest.intents):
                continue
            return self.build_tool_call(
                tool_name=item.name,
                query=query,
                step=step,
                source="dynamic_discovery",
                score=item.score,
            )

        for item in discovered:
            if item.name in used_tools:
                continue
            return self.build_tool_call(
                tool_name=item.name,
                query=query,
                step=step,
                source="dynamic_discovery",
                score=item.score,
            )

        return None
