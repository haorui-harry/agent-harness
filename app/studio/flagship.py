"""Flagship studio pipeline that unifies routing, evaluation, ecosystem, and interop."""

from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.state import GraphState
from app.core.mission import MissionRegistry
from app.ecosystem.marketplace import discover_for_query, get_provider_stats, get_trending_skills
from app.graph import build_graph
from app.harness import HarnessConstraints
from app.harness.engine import HarnessEngine
from app.policy.center import normalize_mode, policy_for_mode
from app.skills.interop import export_interop_all, write_interop_bundle
from app.studio.proposals import ProposalRegistry, ProposalScenario
from app.tracing.analyzer import RoutingAnalyzer
from app.tracing.visualizer import render_trace_views

FLAGSHIP_ONE_LINER = (
    "Agent Harness Studio turns one user request into an auditable, benchmarked, "
    "and ecosystem-portable agent product."
)
FLAGSHIP_DIFF = (
    "Single pipeline with runtime routing evidence, research-grade release gating, "
    "and OpenAI/Anthropic skill export compatibility."
)
STUDIO_SCHEMA = "agent-harness-studio/v1"
DEFAULT_STUDIO_SCENARIOS = [
    "daily-001",
    "research-001",
    "creative-001",
    "enterprise-001",
    "safety-001",
]

FRAMEWORK_ARCHETYPES: list[dict[str, Any]] = [
    {
        "name": "data-flow",
        "description": "Strong flow orchestration and deterministic pipelines; weaker portability.",
        "vector": {
            "orchestration_quality": 0.82,
            "research_rigor": 0.58,
            "ecosystem_leverage": 0.52,
            "governance_safety": 0.62,
            "product_readiness": 0.66,
            "interoperability": 0.50,
        },
    },
    {
        "name": "deep-research",
        "description": "Strong research loop and analysis depth; weaker productization and onboarding.",
        "vector": {
            "orchestration_quality": 0.72,
            "research_rigor": 0.88,
            "ecosystem_leverage": 0.40,
            "governance_safety": 0.66,
            "product_readiness": 0.48,
            "interoperability": 0.55,
        },
    },
    {
        "name": "skill-hub",
        "description": "Strong ecosystem integration; weaker rigorous evaluation gates.",
        "vector": {
            "orchestration_quality": 0.62,
            "research_rigor": 0.50,
            "ecosystem_leverage": 0.90,
            "governance_safety": 0.45,
            "product_readiness": 0.72,
            "interoperability": 0.86,
        },
    },
]

TOOL_DISPLAY_ALIASES: dict[str, str] = {
    "api_skill_portfolio_optimizer": "portfolio optimizer",
    "api_skill_dependency_graph": "dependency graph",
    "policy_risk_matrix": "policy risk matrix",
    "memory_context_digest": "memory context layer",
    "code_experiment_design": "experiment design workbench",
    "external_resource_hub": "external validation hub",
    "code_router_blueprint": "architecture blueprint engine",
    "identify_risks": "risk identification",
    "risk_heatmap": "risk heatmap",
    "synthesize_perspectives": "perspective synthesis",
    "executive_summary": "executive summary",
    "extract_facts": "fact extraction",
    "build_timeline": "timeline planning",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _slug_tag(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", raw.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned[:64]


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _sanitize_business_text(text: object) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    for raw, alias in TOOL_DISPLAY_ALIASES.items():
        value = value.replace(raw, alias)
    return value


def _safe_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in ("", None):
        return []
    return [value]


class StudioShowcaseBuilder:
    """Generate concentrated showcase artifacts from a single product pipeline."""

    def __init__(self, harness: HarnessEngine | None = None) -> None:
        self.harness = harness or HarnessEngine()
        self.missions = MissionRegistry()
        self.proposals = ProposalRegistry()

    def build_showcase(
        self,
        query: str,
        mode: str = "balanced",
        lab_preset: str = "broad",
        lab_repeats: int = 1,
        scenario_ids: list[str] | None = None,
        include_marketplace: bool = True,
        include_external: bool = True,
        include_harness_tools: bool = True,
        include_interop_catalog: bool = False,
        constraints: HarnessConstraints | None = None,
        live_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build flagship payload by orchestrating all major project modules."""

        parsed_mode = normalize_mode(mode)
        router_payload = self._run_router_query(query=query, mode=parsed_mode.value)
        analyzer = RoutingAnalyzer()
        router_analysis = analyzer.analyze(
            router_payload.get("reasoning_path", []),
            router_payload.get("routing_metrics", {}),
        )
        router_views = render_trace_views(router_payload.get("routing_trace", {}))

        run = self.harness.run(
            query=query,
            mode=parsed_mode.value,
            constraints=constraints,
            live_model=live_model,
        )
        run_summary = self.harness.reporter.summary(run)
        value_card = self.harness.build_value_card(run)
        visual_payload = self.harness.build_visual_payload(run, value_card=value_card)
        scenario = self.proposals.infer(query)
        mission_profile = self.missions.infer(query)

        active_scenarios = scenario_ids or list(DEFAULT_STUDIO_SCENARIOS)
        lab_payload = self.harness.run_research_lab(
            preset=lab_preset,
            repeats=max(1, int(lab_repeats)),
            scenario_ids=active_scenarios,
            include_runs=False,
            isolate_memory=True,
            fresh_memory_per_candidate=True,
        )
        lab_story = self.harness.build_lab_product_bundle(lab_payload=lab_payload, tag="studio-preview")

        discovery = discover_for_query(query=query, limit=6)
        trending = get_trending_skills(limit=6)
        providers = self._provider_snapshot(discovery=discovery, trending=trending)

        interop_catalog = export_interop_all(
            include_marketplace=include_marketplace,
            include_external=include_external,
            include_harness_tools=include_harness_tools,
        )
        interop_summary = self._interop_summary(interop_catalog)

        capability = self._capability_vector(
            router_analysis=router_analysis,
            run_summary=run_summary,
            value_card=value_card,
            lab_payload=lab_payload,
            interop_summary=interop_summary,
            discovery=discovery,
            trending=trending,
        )
        frontier = self._frontier_score(capability)
        comparison = self._compare_archetypes(capability=capability, frontier_score=frontier["score"])
        story = self._story_frame(
            query=query,
            mode=parsed_mode.value,
            router_payload=router_payload,
            run_summary=run_summary,
            value_card=value_card,
            lab_payload=lab_payload,
            interop_summary=interop_summary,
            frontier=frontier,
            scenario=scenario,
        )
        proposal = self._proposal_frame(
            run=run,
            run_summary=run_summary,
            story=story,
            lab_payload=lab_payload,
            scenario=scenario,
        )
        agent_comparison = self._agent_comparison(router_payload)
        mission_pack = self.missions.build_release_pack(
            query=query,
            run=run,
            scenario=scenario.to_dict(),
            story=story,
            proposal=proposal,
            run_summary=run_summary,
            lab_payload=lab_payload,
            agent_comparison=agent_comparison,
            profile=mission_profile,
        )
        delivery_brief_excerpt = self._delivery_brief(
            query=query,
            story=story,
            proposal=proposal,
            run_summary=run_summary,
        )

        payload: dict[str, Any] = {
            "schema": STUDIO_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario.to_dict(),
            "identity": {
                "name": "Agent Harness Studio",
                "one_liner": FLAGSHIP_ONE_LINER,
                "differentiator": FLAGSHIP_DIFF,
            },
            "query": {
                "text": query,
                "mode": parsed_mode.value,
                "selected_agent": router_payload.get("agent_name", ""),
                "selected_skills": router_payload.get("selected_skills", []),
                "complexity": router_payload.get("query_complexity", "unknown"),
                "risk_level": router_payload.get("risk_level", "unknown"),
            },
            "router": {
                "analysis": router_analysis,
                "trace_views": router_views,
                "trace": router_payload.get("routing_trace", {}),
            },
            "harness": {
                "run_summary": run_summary,
                "plan": list(run.plan),
                "final_answer": run.final_answer,
                "final_answer_excerpt": run.final_answer[:1600],
                "delivery_brief_excerpt": delivery_brief_excerpt,
                "generation": self._generation_summary(run),
                "value_card": value_card,
                "visual_kpis": visual_payload.get("kpis", {}),
                "first_screen_blueprint": visual_payload.get("first_screen_blueprint", {}),
            },
            "lab": {
                "preset": lab_preset,
                "repeats": max(1, int(lab_repeats)),
                "scenario_ids": active_scenarios,
                "release_decision": lab_payload.get("release_decision", {}),
                "best": lab_payload.get("best", {}),
                "leaderboard": lab_payload.get("leaderboard", []),
                "competition": lab_payload.get("competition", {}),
                "story_summary": {
                    "summary": lab_story.get("summary", {}),
                    "applause_points": lab_story.get("applause_points", []),
                    "trend": lab_story.get("trend", {}),
                    "champion_streak": lab_story.get("champion_streak", {}),
                },
            },
            "ecosystem": {
                "query_discovery": discovery,
                "trending": trending,
                "providers": providers,
            },
            "interop": {
                "config": {
                    "include_marketplace": include_marketplace,
                    "include_external": include_external,
                    "include_harness_tools": include_harness_tools,
                },
                "summary": interop_summary,
            },
            "story": story,
            "mission": mission_pack,
            "proposal": proposal,
            "agent_comparison": agent_comparison,
            "capability_vector": capability,
            "frontier": frontier,
            "comparison": comparison,
            "score_provenance": self._score_provenance(
                run_summary=run_summary,
                value_card=value_card,
                frontier=frontier,
                comparison=comparison,
                generation=self._generation_summary(run),
            ),
            "why_use_this": self._why_use(capability=capability, frontier=frontier, comparison=comparison),
        }
        if include_interop_catalog:
            payload["interop"]["catalog"] = interop_catalog
        return payload

    def write_showcase(
        self,
        payload: dict[str, Any],
        output_dir: str = "reports/studio",
        tag: str = "",
        export_interop: bool = True,
    ) -> dict[str, Any]:
        """Write studio JSON/HTML artifacts and optional interop bundle."""

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        run_tag = _slug_tag(tag) if tag.strip() else _now_tag()
        json_path = root / f"studio_showcase_{run_tag}.json"
        html_path = root / f"studio_showcase_{run_tag}.html"
        brief_path = root / f"studio_press_brief_{run_tag}.md"
        deliverable_path = root / f"studio_deliverable_{run_tag}.md"
        manifest_path = root / f"studio_bundle_manifest_{run_tag}.json"

        serialized = json.loads(json.dumps(payload, default=str))
        interop = serialized.get("interop", {})
        catalog = interop.pop("catalog", None) if isinstance(interop, dict) else None
        serialized["handoff"] = self._handoff_manifest(run_tag=run_tag)
        deliverable_path.write_text(self.render_primary_deliverable_markdown(serialized), encoding="utf-8")
        json_path.write_text(json.dumps(serialized, indent=2, default=str), encoding="utf-8")
        html_path.write_text(self.render_showcase_html(serialized), encoding="utf-8")

        result: dict[str, Any] = {
            "run_tag": run_tag,
            "json": str(json_path),
            "html": str(html_path),
            "brief": str(brief_path),
            "deliverable": str(deliverable_path),
            "manifest": str(manifest_path),
        }
        if export_interop:
            effective_catalog = catalog if isinstance(catalog, dict) else self._regen_interop_catalog(serialized)
            result["interop"] = write_interop_bundle(effective_catalog, root / f"studio_interop_{run_tag}")
        brief_path.write_text(self.render_press_brief_markdown(serialized, result), encoding="utf-8")
        manifest_path.write_text(json.dumps(self._bundle_manifest(serialized, result), indent=2, default=str), encoding="utf-8")
        return result

    @staticmethod
    def _run_router_query(query: str, mode: str) -> dict[str, Any]:
        parsed = normalize_mode(mode)
        state = GraphState(query=query, system_mode=parsed.value, policy=policy_for_mode(parsed).to_dict())
        result = build_graph().invoke(state)
        payload = result if isinstance(result, dict) else result.model_dump()
        payload.setdefault("reasoning_path", StudioShowcaseBuilder._build_reasoning_path(payload))
        return payload

    @staticmethod
    def _build_reasoning_path(payload: dict[str, Any]) -> list[dict[str, Any]]:
        if payload.get("reasoning_path"):
            return payload["reasoning_path"]
        path: list[dict[str, Any]] = []
        agent = payload.get("routing_trace", {}).get("agent_decision", {})
        if agent:
            path.append({"step": 1, "event": "agent_selected", "elapsed_ms": 1.0, "data": {"selected": agent.get("selected", [])}})
        skill = payload.get("routing_trace", {}).get("skill_decision", {})
        if skill:
            path.append(
                {
                    "step": len(path) + 1,
                    "event": "skill_selected",
                    "elapsed_ms": 3.0,
                    "data": {"selected": skill.get("selected", []), "execution_order": skill.get("execution_order", [])},
                }
            )
        path.append({"step": len(path) + 1, "event": "execution_completed", "elapsed_ms": 8.0, "data": {"conflicts": len(payload.get("conflicts_detected", []))}})
        return path

    @staticmethod
    def _provider_snapshot(discovery: list[dict[str, Any]], trending: list[dict[str, Any]]) -> list[dict[str, Any]]:
        providers: list[str] = []
        for row in discovery + trending:
            provider = row.get("provider")
            if isinstance(provider, str) and provider and provider not in providers:
                providers.append(provider)
        return [get_provider_stats(name) for name in providers[:4]]

    @staticmethod
    def _interop_summary(catalog: dict[str, Any]) -> dict[str, Any]:
        frameworks = catalog.get("frameworks", {}) if isinstance(catalog, dict) else {}
        rows = [
            {"framework": name, "skill_count": int(payload.get("skill_count", 0))}
            for name, payload in frameworks.items()
            if isinstance(payload, dict)
        ]
        active = sum(1 for item in rows if item["skill_count"] > 0)
        return {
            "framework_count": len(rows),
            "coverage_ratio": round(active / max(len(rows), 1), 4),
            "total_skill_entries": sum(item["skill_count"] for item in rows),
            "frameworks": rows,
        }

    @staticmethod
    def _value_dims(value_card: dict[str, Any]) -> dict[str, float]:
        dims: dict[str, float] = {}
        for row in value_card.get("dimensions", []):
            if isinstance(row, dict) and row.get("name"):
                dims[str(row["name"]).strip().lower()] = _safe_float(row.get("score", 0.0))
        return dims

    def _capability_vector(
        self,
        router_analysis: dict[str, Any],
        run_summary: dict[str, Any],
        value_card: dict[str, Any],
        lab_payload: dict[str, Any],
        interop_summary: dict[str, Any],
        discovery: list[dict[str, Any]],
        trending: list[dict[str, Any]],
    ) -> dict[str, float]:
        quality = _safe_float(router_analysis.get("quality", {}).get("overall_score", 0.0))
        robust_worst_case = _safe_float(router_analysis.get("quality", {}).get("robust_worst_case_utility", 0.0))
        avg_uncertainty_metric = _safe_float(router_analysis.get("quality", {}).get("avg_uncertainty", 0.0))
        efficiency = {"fast": 1.0, "moderate": 0.75, "slow": 0.45}.get(
            str(router_analysis.get("efficiency", {}).get("rating", "moderate")),
            0.65,
        )
        orchestration_quality = _clamp01(
            0.55 * quality + 0.20 * efficiency + 0.20 * min(max(robust_worst_case, 0.0) / 1.2, 1.0) - 0.05 * avg_uncertainty_metric
        )

        release = {"go": 1.0, "caution": 0.78, "block": 0.52}.get(
            str(lab_payload.get("release_decision", {}).get("decision", "block")),
            0.52,
        )
        best = _safe_float(lab_payload.get("best", {}).get("composite_score", 0.0))
        categories = {str(item.get("category", "")).strip().lower() for item in lab_payload.get("scenarios", []) if isinstance(item, dict)}
        research_rigor = _clamp01(0.60 * best + 0.25 * release + 0.15 * _clamp01(len([x for x in categories if x]) / 5.0))

        avg_discovery = _clamp01(len(discovery) / 6.0)
        avg_trending = _clamp01((sum(_safe_float(item.get("trending_score", 0.0)) for item in trending) / max(len(trending), 1)) / 0.35)
        ecosystem_leverage = _clamp01(0.45 * avg_discovery + 0.25 * avg_trending + 0.30 * _safe_float(interop_summary.get("coverage_ratio", 0.0)))

        dims = self._value_dims(value_card)
        governance_safety = _clamp01(0.65 * dims.get("safety", 0.0) + 0.35 * _safe_float(lab_payload.get("best", {}).get("avg_security_alignment", 0.0)))

        metrics = run_summary.get("metrics", {}) if isinstance(run_summary, dict) else {}
        product_readiness = _clamp01(
            0.33 * dims.get("reliability", 0.0)
            + 0.27 * dims.get("observability", 0.0)
            + 0.20 * dims.get("adaptability", 0.0)
            + 0.20 * _safe_float(metrics.get("completion_score", 0.0))
        )

        return {
            "orchestration_quality": round(orchestration_quality, 4),
            "research_rigor": round(research_rigor, 4),
            "ecosystem_leverage": round(ecosystem_leverage, 4),
            "governance_safety": round(governance_safety, 4),
            "product_readiness": round(product_readiness, 4),
            "interoperability": round(_clamp01(_safe_float(interop_summary.get("coverage_ratio", 0.0))), 4),
        }

    @staticmethod
    def _frontier_score(vector: dict[str, float]) -> dict[str, Any]:
        values = [_clamp01(v) for v in vector.values()]
        mean_score = sum(values) / max(len(values), 1)
        min_score = min(values) if values else 0.0
        geo = math.exp(sum(math.log(max(v, 1e-6)) for v in values) / max(len(values), 1)) if values else 0.0
        score = _clamp01(0.35 * mean_score + 0.40 * min_score + 0.25 * geo)
        axis, axis_score = sorted(vector.items(), key=lambda item: item[1])[0] if vector else ("", 0.0)
        return {
            "score": round(score, 4),
            "mean": round(mean_score, 4),
            "minimum_axis": round(min_score, 4),
            "geometric": round(geo, 4),
            "kind": "internal_heuristic",
            "benchmark_validated": False,
            "bottleneck": {"axis": axis, "score": round(axis_score, 4)},
        }

    def _compare_archetypes(self, capability: dict[str, float], frontier_score: float) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for item in FRAMEWORK_ARCHETYPES:
            vector = item["vector"]
            baseline = self._frontier_score(vector)["score"]
            deltas = {key: round(capability.get(key, 0.0) - float(vector.get(key, 0.0)), 4) for key in capability}
            wins = [key for key, delta in deltas.items() if delta > 0]
            losses = [key for key, delta in deltas.items() if delta < 0]
            rows.append(
                {
                    "name": item["name"],
                    "description": item["description"],
                    "baseline_frontier": round(baseline, 4),
                    "frontier_gap": round(frontier_score - baseline, 4),
                    "wins": wins,
                    "losses": losses,
                    "deltas": deltas,
                    "advantage_ratio": round(len(wins) / max(len(deltas), 1), 4),
                }
            )
        rows.sort(key=lambda row: float(row.get("frontier_gap", 0.0)), reverse=True)
        strongest = rows[0] if rows else {}
        headline = (
            f"Ahead of built-in {strongest.get('name', 'baseline')} archetype by {strongest.get('frontier_gap', 0.0):+.3f} internal frontier."
            if strongest
            else "No comparison available."
        )
        return {
            "archetypes": rows,
            "positioning": {
                "headline": headline,
                "best_vs_name": strongest.get("name", ""),
                "best_vs_gap": strongest.get("frontier_gap", 0.0),
                "benchmark_validated": False,
                "comparison_basis": "built_in_archetype_vectors",
            },
        }

    @staticmethod
    def _story_frame(
        query: str,
        mode: str,
        router_payload: dict[str, Any],
        run_summary: dict[str, Any],
        value_card: dict[str, Any],
        lab_payload: dict[str, Any],
        interop_summary: dict[str, Any],
        frontier: dict[str, Any],
        scenario: ProposalScenario,
    ) -> dict[str, Any]:
        """Build a plain-language narrative so the showcase is immediately understandable."""

        plan = [str(item) for item in run_summary.get("plan", [])]
        selected_skills = [str(item) for item in router_payload.get("selected_skills", [])]
        release = lab_payload.get("release_decision", {})
        bottleneck = frontier.get("bottleneck", {})
        dims = StudioShowcaseBuilder._value_dims(value_card)
        evidence = run_summary.get("evidence", {}) if isinstance(run_summary, dict) else {}
        routing_confidence = _safe_float(
            router_payload.get("routing_trace", {}).get("final_confidence_breakdown", {}).get("routing_confidence", 0.0)
        )

        return {
            "theme": scenario.theme,
            "release_need": scenario.release_need,
            "strategy_plan": list(scenario.strategy_plan),
            "execution_recipe": plan,
            "selected_agent": str(router_payload.get("agent_name", "")),
            "selected_skills": selected_skills,
            "evidence_bundle": [
                f"Release decision: {release.get('decision', 'block')} ({release.get('reason', '-')})",
                f"Evidence packet: {int(evidence.get('record_count', 0))} records / {int(evidence.get('citation_count', 0))} citations",
                f"Interop export: {int(interop_summary.get('framework_count', 0))} frameworks / {int(interop_summary.get('total_skill_entries', 0))} skill entries",
                f"Routing confidence (internal): {routing_confidence:.3f}",
                f"Safety / reliability signals (internal): {_safe_float(dims.get('safety', 0.0)):.2f} / {_safe_float(dims.get('reliability', 0.0)):.2f}",
                f"Frontier estimate (internal): {_safe_float(frontier.get('score', 0.0)):.3f}; bottleneck axis: {bottleneck.get('axis', '-')}",
                f"Value heuristic (internal): {_safe_float(value_card.get('value_index', 0.0)):.2f} ({value_card.get('band', '-')})",
            ],
            "audience_takeaway": scenario.audience_takeaway,
            "mode": mode,
            "query": query,
            "scenario_name": scenario.name,
        }

    @staticmethod
    def _generation_summary(run: Any) -> dict[str, Any]:
        """Summarize whether the showcase content was generated by a live model or baseline pipeline."""

        metadata = run.metadata if hasattr(run, "metadata") and isinstance(run.metadata, dict) else {}
        live = metadata.get("live_agent", {}) if isinstance(metadata, dict) else {}
        live_enabled = bool(live.get("enabled", False))
        live_configured = bool(live.get("configured", False))
        live_success = bool(live.get("success", False))
        mode = "baseline"
        if live_enabled and live_configured and live_success:
            mode = "live_api"
        elif live_enabled:
            mode = "live_fallback"

        return {
            "mode": mode,
            "live_agent_enabled": live_enabled,
            "live_agent_configured": live_configured,
            "live_agent_success": live_success,
            "model": str(live.get("model", "")),
            "calls_used": int(live.get("calls_used", 0)),
            "call_budget": int(live.get("call_budget", 0)),
            "notes": list(live.get("notes", [])) if isinstance(live.get("notes", []), list) else [],
            "errors": [
                _sanitize_business_text(item) for item in live.get("errors", [])
            ]
            if isinstance(live.get("errors", []), list)
            else [],
        }

    @staticmethod
    def _proposal_frame(
        run: Any,
        run_summary: dict[str, Any],
        story: dict[str, Any],
        lab_payload: dict[str, Any],
        scenario: ProposalScenario,
    ) -> dict[str, Any]:
        """Build a business-first proposal view for the showcase front page."""

        metadata = run.metadata if hasattr(run, "metadata") and isinstance(run.metadata, dict) else {}
        live = metadata.get("live_agent", {}) if isinstance(metadata, dict) else {}
        analysis = live.get("analysis", {}) if isinstance(live, dict) else {}
        critique = live.get("critique", {}) if isinstance(live, dict) else {}
        release = lab_payload.get("release_decision", {})
        expected = analysis.get("expected_value", {}) if isinstance(analysis, dict) else {}
        migration = analysis.get("launch_phases", []) if isinstance(analysis, dict) else []
        if not migration:
            migration = analysis.get("migration_path", []) if isinstance(analysis, dict) else []
        architecture = analysis.get("target_architecture", {}) if isinstance(analysis, dict) else {}
        proof_points = _safe_list(analysis.get("proof_points", [])) if isinstance(analysis, dict) else []
        target_users = _safe_list(analysis.get("target_users", [])) if isinstance(analysis, dict) else []
        controls = _safe_list(analysis.get("controls", [])) if isinstance(analysis, dict) else []
        winner_hypothesis = _sanitize_business_text(analysis.get("winner_hypothesis", "")) if isinstance(analysis, dict) else ""
        metrics = run_summary.get("metrics", {}) if isinstance(run_summary, dict) else {}
        value_card = run_summary.get("value_card", {}) if isinstance(run_summary, dict) else {}
        security = run_summary.get("security", {}) if isinstance(run_summary, dict) else {}
        evidence = run_summary.get("evidence", {}) if isinstance(run_summary, dict) else {}

        pillars: list[dict[str, Any]] = []
        for blueprint in scenario.pillars:
            pillar = architecture.get(blueprint.live_key, {}) if isinstance(architecture, dict) and blueprint.live_key else {}
            capabilities = pillar.get("capabilities", []) if isinstance(pillar, dict) else []
            integrations = pillar.get("integration_points", []) if isinstance(pillar, dict) else []
            summary = _sanitize_business_text(", ".join(capabilities[:3])) or blueprint.summary
            integration = _sanitize_business_text(", ".join(integrations[:2])) or blueprint.integration
            pillars.append(
                {
                    "title": blueprint.title,
                    "summary": summary,
                    "integration": integration,
                }
            )

        phases: list[dict[str, Any]] = []
        for item in _safe_list(migration)[:3]:
            if not isinstance(item, dict):
                continue
            phases.append(
                {
                    "phase": _sanitize_business_text(item.get("phase", "")),
                    "actions": [_sanitize_business_text(x) for x in item.get("actions", [])[:3]],
                    "success_metrics": [_sanitize_business_text(x) for x in item.get("success_metrics", [])[:3]],
                }
            )
        if not phases:
            plan_chunks = StudioShowcaseBuilder._chunk_plan(run_summary.get("plan", []), max(len(scenario.phases), 1))
            for index, blueprint in enumerate(scenario.phases):
                fallback_actions = [_sanitize_business_text(x) for x in plan_chunks[index]] if index < len(plan_chunks) else []
                merged_actions: list[str] = []
                for action in list(blueprint.actions[:2]) + fallback_actions[:2]:
                    clean = _sanitize_business_text(action)
                    if clean and clean not in merged_actions:
                        merged_actions.append(clean)
                phases.append(
                    {
                        "phase": blueprint.phase,
                        "actions": merged_actions[:3] or [_sanitize_business_text(x) for x in blueprint.actions[:3]],
                        "success_metrics": [_sanitize_business_text(x) for x in blueprint.success_metrics[:3]],
                    }
                )

        expected_impact = StudioShowcaseBuilder._impact_frame(
            scenario=scenario,
            expected=expected,
            metrics=metrics,
            value_card=value_card,
            release=release,
            target_users=target_users,
        )
        critical_risks = [_sanitize_business_text(x) for x in _safe_list(critique.get("red_flags", []))[:3]] if isinstance(critique, dict) else []
        if not critical_risks:
            critical_risks = [_sanitize_business_text(x) for x in _safe_list(analysis.get("bottlenecks", []))[:3]] if isinstance(analysis, dict) else []
        if not critical_risks:
            security_findings = security.get("preflight_findings", []) if isinstance(security, dict) else []
            critical_risks = [
                _sanitize_business_text(item.get("title", ""))
                for item in security_findings[:2]
                if isinstance(item, dict) and item.get("title")
            ]
        if not critical_risks:
            critical_risks = [_sanitize_business_text(x) for x in scenario.critical_risks[:3]]

        thesis = _sanitize_business_text(analysis.get("thesis", "")) if isinstance(analysis, dict) else ""
        if not thesis:
            thesis = _sanitize_business_text(story.get("audience_takeaway", ""))
        if winner_hypothesis:
            thesis = f"{thesis} Winning move: {winner_hypothesis}".strip()

        business_summary = list(scenario.business_summary)
        if proof_points:
            business_summary.append(f"Proof points: {', '.join(_sanitize_business_text(x) for x in proof_points[:3])}.")
        if controls:
            business_summary.append(f"Control focus: {', '.join(_sanitize_business_text(x) for x in controls[:3])}.")
        if int(evidence.get("record_count", 0)) > 0:
            business_summary.append(
                f"Evidence base: {int(evidence.get('record_count', 0))} records and {int(evidence.get('citation_count', 0))} citations were injected into the launch packet."
            )
        business_summary = [item for item in business_summary if item][:4]

        return {
            "headline": scenario.headline,
            "subheadline": thesis,
            "decision": {
                "status": release.get("decision", "block"),
                "reason": _sanitize_business_text(release.get("reason", "")),
                "selected_candidate": release.get("selected_candidate", ""),
            },
            "pillars": pillars,
            "phases": phases,
            "expected_impact": expected_impact,
            "critical_risks": critical_risks,
            "business_summary": business_summary,
            "execution_plan": [_sanitize_business_text(item) for item in run_summary.get("plan", [])],
            "target_users": [_sanitize_business_text(x) for x in target_users[:4]],
            "proof_points": [_sanitize_business_text(x) for x in proof_points[:4]],
            "scenario_name": scenario.name,
        }

    @staticmethod
    def _chunk_plan(items: list[Any], bucket_count: int) -> list[list[str]]:
        rows = [str(item).strip() for item in items if str(item).strip()]
        if bucket_count <= 0:
            return [rows]
        chunks: list[list[str]] = [[] for _ in range(bucket_count)]
        for index, item in enumerate(rows):
            chunks[index % bucket_count].append(item)
        return chunks

    @staticmethod
    def _impact_frame(
        scenario: ProposalScenario,
        expected: dict[str, Any] | Any,
        metrics: dict[str, Any],
        value_card: dict[str, Any],
        release: dict[str, Any],
        target_users: list[Any],
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        if isinstance(expected, dict):
            for key, value in list(expected.items())[:5]:
                rows.append(
                    {
                        "label": _sanitize_business_text(str(key).replace("_", " ").title()),
                        "value": _sanitize_business_text(value),
                    }
                )
        elif _sanitize_business_text(expected):
            rows.append(
                {
                    "label": scenario.impact_labels[0] if scenario.impact_labels else "Expected Impact",
                    "value": _sanitize_business_text(expected),
                }
            )
        if rows:
            return rows
        fallback_values = [
            f"release decision {release.get('decision', 'block')} with reason {release.get('reason', '-')}",
            f"completion score {_safe_float(metrics.get('completion_score', 0.0)):.2f} and tool success {_safe_float(metrics.get('tool_success_rate', 0.0)):.2f}",
            f"internal value heuristic {_safe_float(value_card.get('value_index', 0.0)):.1f} and band {value_card.get('band', '-')}",
            f"target users {', '.join(_sanitize_business_text(x) for x in target_users[:3])}" if target_users else "stakeholder packet ready for product, operations, and governance review",
        ]
        for label, value in zip(scenario.impact_labels, fallback_values):
            rows.append({"label": label, "value": _sanitize_business_text(value)})
        return rows[:4]

    @staticmethod
    def _delivery_brief(
        query: str,
        story: dict[str, Any],
        proposal: dict[str, Any],
        run_summary: dict[str, Any],
    ) -> str:
        phases = proposal.get("phases", [])
        impact = proposal.get("expected_impact", [])
        risks = proposal.get("critical_risks", [])
        evidence = run_summary.get("evidence", {}) if isinstance(run_summary, dict) else {}
        lines = [
            proposal.get("headline", "Launch Plan"),
            "",
            f"Scenario: {story.get('theme', '')}",
            f"Decision: {proposal.get('decision', {}).get('status', 'block')} ({proposal.get('decision', {}).get('reason', '-')})",
            f"Request: {query}",
            "",
            "Operating Thesis:",
            f"- {proposal.get('subheadline', '')}",
            "",
            "Business Summary:",
        ]
        for item in proposal.get("business_summary", [])[:4]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Phased Rollout:")
        for phase in phases[:3]:
            lines.append(f"- {phase.get('phase', 'Phase')}: {', '.join(str(x) for x in phase.get('actions', [])[:3])}")
        lines.append("")
        lines.append("Expected Impact:")
        for row in impact[:4]:
            lines.append(f"- {row.get('label', 'Impact')}: {row.get('value', '')}")
        lines.append("")
        lines.append("Critical Risks:")
        for risk in risks[:4]:
            lines.append(f"- {risk}")
        if int(evidence.get("citation_count", 0)) > 0:
            lines.append("")
            lines.append("Evidence Citations:")
            for item in evidence.get("citations", [])[:4]:
                lines.append(f"- {item}")
        lines.append("")
        lines.append("Execution Backbone:")
        for item in run_summary.get("plan", [])[:6]:
            lines.append(f"- {item}")
        return "\n".join(str(item).rstrip() for item in lines if str(item).strip()).strip()

    @staticmethod
    def _agent_comparison(router_payload: dict[str, Any]) -> dict[str, Any]:
        """Summarize agent-vs-agent scoring from the router."""

        decision = router_payload.get("routing_trace", {}).get("agent_decision", {})
        scores = decision.get("scores", {}) if isinstance(decision, dict) else {}
        breakdown = decision.get("score_breakdown", {}) if isinstance(decision, dict) else {}
        reasons = decision.get("reasons", {}) if isinstance(decision, dict) else {}
        rows: list[dict[str, Any]] = []
        for name, score in scores.items():
            row = {
                "agent": str(name),
                "score": _safe_float(score),
                "reason": _sanitize_business_text(reasons.get(name, "")),
                "breakdown": breakdown.get(name, {}),
            }
            rows.append(row)
        rows.sort(key=lambda item: item["score"], reverse=True)
        winner = rows[0] if rows else {}
        runner_up = rows[1] if len(rows) > 1 else {}
        gap = _safe_float(winner.get("score", 0.0)) - _safe_float(runner_up.get("score", 0.0))
        return {
            "winner": winner.get("agent", ""),
            "runner_up": runner_up.get("agent", ""),
            "score_gap": round(gap, 4),
            "rows": rows[:6],
        }

    @staticmethod
    def _why_use(capability: dict[str, float], frontier: dict[str, Any], comparison: dict[str, Any]) -> list[str]:
        top = sorted(capability.items(), key=lambda item: item[1], reverse=True)[:3]
        top_text = ", ".join(f"{k}={v:.2f}" for k, v in top)
        bottleneck = frontier.get("bottleneck", {})
        return [
            f"Concentrated value axis: {top_text}.",
            f"Internal frontier estimate={frontier.get('score', 0.0):.3f} with bottleneck `{bottleneck.get('axis', '')}`.",
            str(comparison.get("positioning", {}).get("headline", "")),
            "Method edge: routing balances deliverable fit, evidence need, and execution risk instead of forcing one fixed workflow.",
            "Same command emits narrative report, quantitative leaderboard, and OpenAI/Anthropic skill bundle.",
        ]

    @staticmethod
    def _score_provenance(
        *,
        run_summary: dict[str, Any],
        value_card: dict[str, Any],
        frontier: dict[str, Any],
        comparison: dict[str, Any],
        generation: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = run_summary.get("metrics", {}) if isinstance(run_summary, dict) else {}
        evidence = run_summary.get("evidence", {}) if isinstance(run_summary, dict) else {}
        facts = [
            {
                "name": "tool_success_rate",
                "value": round(_safe_float(metrics.get("tool_success_rate", 0.0)), 4),
                "basis": "measured_run_execution",
            },
            {
                "name": "completion_score",
                "value": round(_safe_float(metrics.get("completion_score", 0.0)), 4),
                "basis": "measured_run_completion",
            },
            {
                "name": "evidence_records",
                "value": int(evidence.get("record_count", 0)),
                "basis": "counted_evidence_bundle",
            },
            {
                "name": "evidence_citations",
                "value": int(evidence.get("citation_count", 0)),
                "basis": "counted_citations",
            },
            {
                "name": "live_agent_success",
                "value": bool(generation.get("live_agent_success", False)),
                "basis": "measured_api_run",
            },
        ]
        heuristics = [
            {
                "name": "value_index",
                "value": round(_safe_float(value_card.get("value_index", 0.0)), 2),
                "basis": "internal_weighted_heuristic",
            },
            {
                "name": "frontier_score",
                "value": round(_safe_float(frontier.get("score", 0.0)), 4),
                "basis": "internal_bottleneck_aware_heuristic",
            },
            {
                "name": "archetype_gap",
                "value": round(_safe_float(comparison.get("positioning", {}).get("best_vs_gap", 0.0)), 4),
                "basis": "built_in_archetype_comparison",
            },
        ]
        warnings = [
            "Value index, frontier score, and archetype gap are internal heuristics, not public benchmark results.",
            "Archetype comparison uses built-in baseline vectors and should not be presented as a measured win over external repositories.",
        ]
        if any("length" in str(item).lower() for item in generation.get("notes", []) if isinstance(item, str)):
            warnings.append("At least one live-model stage ended on length, so answer quality may still be token-bounded.")
        if int(evidence.get("record_count", 0)) <= 0:
            warnings.append("Evidence packet is empty; any narrative score should be treated as low-confidence.")
        return {"facts": facts, "heuristics": heuristics, "warnings": warnings}

    @staticmethod
    def _regen_interop_catalog(payload: dict[str, Any]) -> dict[str, Any]:
        config = payload.get("interop", {}).get("config", {})
        if not isinstance(config, dict):
            config = {}
        return export_interop_all(
            include_marketplace=bool(config.get("include_marketplace", True)),
            include_external=bool(config.get("include_external", True)),
            include_harness_tools=bool(config.get("include_harness_tools", True)),
        )

    @staticmethod
    def _handoff_manifest(run_tag: str) -> dict[str, Any]:
        return {
            "primary_artifact": {
                "label": "Primary Deliverable",
                "path": f"studio_deliverable_{run_tag}.md",
                "kind": "deliverable_markdown",
            },
            "artifacts": [
                {"label": "Primary Deliverable", "path": f"studio_deliverable_{run_tag}.md", "kind": "deliverable_markdown"},
                {"label": "Showcase HTML", "path": f"studio_showcase_{run_tag}.html", "kind": "showcase_html"},
                {"label": "Showcase JSON", "path": f"studio_showcase_{run_tag}.json", "kind": "showcase_json"},
                {"label": "Press Brief", "path": f"studio_press_brief_{run_tag}.md", "kind": "press_brief"},
                {"label": "Bundle Manifest", "path": f"studio_bundle_manifest_{run_tag}.json", "kind": "bundle_manifest"},
                {"label": "Interop Bundle", "path": f"studio_interop_{run_tag}/index.json", "kind": "interop_bundle"},
            ],
        }

    @staticmethod
    def _primary_output_text(payload: dict[str, Any]) -> str:
        delivery = payload.get("harness", {}) if isinstance(payload.get("harness", {}), dict) else {}
        proposal = payload.get("proposal", {}) if isinstance(payload.get("proposal", {}), dict) else {}
        final_answer = str(delivery.get("final_answer", "")).strip()
        if final_answer:
            return final_answer
        brief = str(delivery.get("delivery_brief_excerpt", "")).strip()
        if brief:
            return brief
        subheadline = str(proposal.get("subheadline", "")).strip()
        if subheadline:
            return subheadline
        return "No primary deliverable content was generated."

    @staticmethod
    def _preview_text(text: str, limit: int = 3200) -> str:
        value = str(text or "").strip()
        if len(value) <= limit:
            return value
        cutoff = value[:limit].rstrip()
        return cutoff + "\n\n[truncated for preview]"

    @staticmethod
    def _key_lines(text: str, limit: int = 6) -> list[str]:
        rows: list[str] = []
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("#", "-", "*")):
                cleaned = line.lstrip("#*- ").strip()
                if cleaned:
                    rows.append(cleaned)
            elif ":" in line and len(line) <= 140:
                rows.append(line)
            if len(rows) >= limit:
                break
        if rows:
            return rows
        paragraphs = [chunk.strip() for chunk in str(text or "").split("\n\n") if chunk.strip()]
        return paragraphs[:limit]

    def render_primary_deliverable_markdown(self, payload: dict[str, Any]) -> str:
        identity = payload.get("identity", {})
        query = payload.get("query", {})
        mission = payload.get("mission", {})
        story = payload.get("story", {})
        delivery = payload.get("harness", {}) if isinstance(payload.get("harness", {}), dict) else {}
        evidence = delivery.get("run_summary", {}).get("evidence", {}) if isinstance(delivery.get("run_summary", {}), dict) else {}
        handoff = payload.get("handoff", {}) if isinstance(payload.get("handoff", {}), dict) else {}
        primary_text = self._primary_output_text(payload)
        artifact_rows = handoff.get("artifacts", []) if isinstance(handoff.get("artifacts", []), list) else []
        cleaned_primary = str(primary_text).strip()
        structured_markers = ("#", "## ", "**Executive Summary**", "**Decision**", "**MEMORANDUM**")
        if cleaned_primary.startswith(structured_markers) or len(cleaned_primary) >= 1600:
            lines = [cleaned_primary]
        else:
            title = str(mission.get("primary_deliverable", "")) or str(payload.get("proposal", {}).get("headline", "")) or "Primary Deliverable"
            lines = [
                f"# {title}",
                "",
                f"_Generated by {identity.get('name', 'Agent Harness Studio')}_",
                "",
                "## Task",
                "",
                str(query.get("text", "")),
                "",
                "## Context",
                "",
                str(story.get("release_need", "")),
                "",
                "## Main Deliverable",
                "",
                cleaned_primary,
            ]
        citations = evidence.get("citations", []) if isinstance(evidence.get("citations", []), list) else []
        if citations:
            lines.extend(
                [
                    "",
                    "## Evidence References",
                    "",
                    *(f"- {item}" for item in citations[:8]),
                ]
            )
        if artifact_rows:
            lines.extend(
                [
                    "",
                    "## Openable Files",
                    "",
                    *(f"- {item.get('label', '')}: {item.get('path', '')}" for item in artifact_rows),
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def render_showcase_html(self, payload: dict[str, Any]) -> str:
        """Render standalone HTML for showcase delivery."""

        identity = payload.get("identity", {})
        query = payload.get("query", {})
        mission = payload.get("mission", {})
        lab = payload.get("lab", {})
        comparison = payload.get("comparison", {})
        score_provenance = payload.get("score_provenance", {})
        why_use = payload.get("why_use_this", [])
        interop = payload.get("interop", {}).get("summary", {})
        story = payload.get("story", {})
        proposal = payload.get("proposal", {})
        agent_comparison = payload.get("agent_comparison", {})
        delivery = payload.get("harness", {})
        generation = delivery.get("generation", {}) if isinstance(delivery, dict) else {}
        evidence = delivery.get("run_summary", {}).get("evidence", {}) if isinstance(delivery, dict) else {}
        run_summary = delivery.get("run_summary", {}) if isinstance(delivery, dict) else {}
        handoff = payload.get("handoff", {}) if isinstance(payload.get("handoff", {}), dict) else {}
        primary_output = self._primary_output_text(payload)
        primary_preview = self._preview_text(primary_output, limit=3600)
        primary_points = self._key_lines(primary_output, limit=6)
        handoff_rows = handoff.get("artifacts", []) if isinstance(handoff.get("artifacts", []), list) else []
        release = lab.get("release_decision", {})

        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Agent Harness Studio</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;600&display=swap');
:root{{--bg:#07111d;--bg2:#10263b;--ink:#122233;--muted:#587086;--card:rgba(248,251,255,.94);--line:rgba(10,30,48,.12);--accent:#0ea5a6;--accent2:#f59e0b;--accent3:#38bdf8;--shadow:0 18px 50px rgba(4,18,30,.24);}}
*{{box-sizing:border-box}}body{{margin:0;font-family:'IBM Plex Sans','Segoe UI',sans-serif;color:var(--ink);background:
radial-gradient(circle at 10% 10%, rgba(14,165,166,.30), transparent 28%),
radial-gradient(circle at 90% 8%, rgba(245,158,11,.20), transparent 24%),
radial-gradient(circle at 78% 78%, rgba(56,189,248,.18), transparent 28%),
linear-gradient(135deg,var(--bg),var(--bg2));padding:24px}}
.wrap{{max-width:1240px;margin:0 auto;display:grid;gap:14px}}
.hero{{position:relative;overflow:hidden;background:linear-gradient(135deg,rgba(7,17,29,.88),rgba(15,53,76,.82));color:#edf7ff;border:1px solid rgba(255,255,255,.12);border-radius:28px;padding:28px;box-shadow:var(--shadow)}}
.hero:before{{content:'';position:absolute;inset:auto -60px -80px auto;width:280px;height:280px;background:radial-gradient(circle,rgba(245,158,11,.26),transparent 68%)}} 
.hero-grid{{display:grid;grid-template-columns:1.35fr .95fr;gap:18px}}@media(max-width:980px){{.hero-grid{{grid-template-columns:1fr}}}}
.hero h1{{font-family:'Space Grotesk','IBM Plex Sans',sans-serif;font-size:44px;line-height:1.05;margin:0 0 10px}}
.hero p{{color:rgba(237,247,255,.84);margin:6px 0}}
.hero-panel{{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);border-radius:20px;padding:16px}}
.card{{background:var(--card);border:1px solid rgba(255,255,255,.55);border-radius:20px;padding:18px;box-shadow:var(--shadow)}}
.glass{{background:linear-gradient(180deg,rgba(255,255,255,.92),rgba(245,249,252,.86))}}
h2,h3{{font-family:'Space Grotesk','IBM Plex Sans',sans-serif;margin:0 0 10px}}p{{margin:5px 0;color:var(--muted)}}ul{{margin:8px 0 0;padding-left:18px}}li{{margin:6px 0}}
.grid{{display:grid;grid-template-columns:1.2fr 1fr;gap:14px}}@media(max-width:980px){{.grid{{grid-template-columns:1fr}}}}
.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}@media(max-width:980px){{.grid3{{grid-template-columns:1fr}}}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}@media(max-width:980px){{.grid4{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:700px){{.grid4{{grid-template-columns:1fr}}}}
.badge{{display:inline-block;padding:6px 11px;border-radius:999px;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.16);font-size:12px;margin:0 8px 8px 0;color:#edf7ff}}
.signal{{padding:14px;border-radius:18px;background:linear-gradient(180deg,rgba(255,255,255,.72),rgba(255,255,255,.55));border:1px solid var(--line)}}
.signal .label{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}.signal .value{{font-size:28px;font-family:'Space Grotesk';margin-top:6px}}
.kicker{{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px 6px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}
pre{{margin:0;font-size:12px;white-space:pre-wrap;background:rgba(6,24,38,.96);color:#e7f7ff;border:1px solid rgba(255,255,255,.12);border-radius:14px;padding:12px;max-height:320px;overflow:auto}}
.report{{white-space:pre-wrap;line-height:1.7;font-size:14px;background:rgba(6,24,38,.96);color:#eef8ff;border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:16px;max-height:560px;overflow:auto}}
details{{border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.7);padding:14px}}summary{{cursor:pointer;font-weight:600}}
.muted{{color:var(--muted)}}
</style></head><body><div class="wrap">
<section class="hero">
  <div class="hero-grid">
    <div>
      <div class="kicker">Task Deliverable Demo</div>
      <h1>{html.escape(str(mission.get("primary_deliverable", proposal.get("headline", "Agent Deliverable"))))}</h1>
      <p><strong>{html.escape(str(story.get("theme", "")))}</strong></p>
      <p>{html.escape(str(story.get("release_need", "")) or str(mission.get("summary", "")))}</p>
      <div style="margin-top:14px">
        <span class="badge">Release {html.escape(str(release.get("decision", "block"))).upper()}</span>
        <span class="badge">Generation {html.escape(str(generation.get("mode", "baseline"))).upper()}</span>
        <span class="badge">Model {html.escape(str(generation.get("model", "")) or "-")}</span>
        <span class="badge">Primary Artifact {html.escape(str(handoff.get("primary_artifact", {}).get("path", "studio_deliverable.md")))}</span>
      </div>
      <div style="margin-top:14px" class="hero-panel">
        <div class="kicker">Task</div>
        <p>{html.escape(str(query.get("text", "")))}</p>
      </div>
    </div>
    <div class="hero-panel">
      <div class="kicker">What Comes Out</div>
      <div class="signal"><div class="label">Primary Output</div><div class="value">{html.escape(str(generation.get("mode", "baseline"))).upper()}</div><p>{html.escape(str(proposal.get("subheadline", "")) or str(identity.get("differentiator", "")))}</p></div>
      <div class="grid4" style="margin-top:12px">
        <div class="signal"><div class="label">Files</div><div class="value">{len(handoff_rows)}</div></div>
        <div class="signal"><div class="label">Citations</div><div class="value">{int(evidence.get("citation_count", 0))}</div></div>
        <div class="signal"><div class="label">Live Calls</div><div class="value">{int(generation.get("calls_used", 0))}</div></div>
        <div class="signal"><div class="label">Decision</div><div class="value">{html.escape(str(proposal.get("decision", {}).get("status", "block"))).upper()}</div></div>
      </div>
    </div>
  </div>
</section>
<section class="card glass">
  <div class="kicker">Primary Deliverable</div>
  <div class="grid">
    <article>
      <h2>Direct Preview</h2>
      <div class="report">{html.escape(primary_preview)}</div>
    </article>
    <article>
      <h2>Key Takeaways</h2>
      <ul>{"".join(f"<li>{html.escape(str(item))}</li>" for item in primary_points)}</ul>
      <h3 style="margin-top:14px">Openable Files</h3>
      <table><thead><tr><th>Artifact</th><th>Kind</th><th>Path</th></tr></thead><tbody>{"".join(f"<tr><td>{html.escape(str(item.get('label','')))}</td><td>{html.escape(str(item.get('kind','')))}</td><td><code>{html.escape(str(item.get('path','')))}</code></td></tr>" for item in handoff_rows) or "<tr><td colspan='3'>No handoff artifacts.</td></tr>"}</tbody></table>
    </article>
  </div>
</section>
<section class="card glass">
  <div class="kicker">Evidence And Runtime</div>
  <div class="grid">
    <article>
      <h2>Task Context</h2>
      <p><strong>Why now:</strong> {html.escape(str(story.get("release_need", "")))}</p>
      <p><strong>Audience:</strong> {html.escape(str(story.get("audience_takeaway", "")))}</p>
      <h3 style="margin-top:14px">Target Users</h3>
      <div>{self._chip_cloud(mission.get("target_users", []), tone="bright")}</div>
      <h3 style="margin-top:14px">Evidence References</h3>
      <ul>{"".join(f"<li>{html.escape(str(item))}</li>" for item in evidence.get("citations", [])[:6]) or "<li>No injected citations.</li>"}</ul>
    </article>
    <article>
      <h2>Runtime Surface</h2>
      <div class="grid4">
        {self._runtime_signal_cards(
            run_summary=run_summary,
            generation=generation,
            interop=interop,
            evidence=evidence,
        )}
      </div>
      <h3 style="margin-top:14px">Execution Plan</h3>
      <ul>{"".join(f"<li>{html.escape(str(item))}</li>" for item in delivery.get("plan", []))}</ul>
    </article>
  </div>
</section>
<section class="card glass">
  <div class="kicker">Deliverable Package</div>
  <div class="grid4">{self._deliverable_cards(mission.get("deliverables", []))}</div>
</section>
<section class="card glass">
  <div class="kicker">Execution Tracks</div>
  <div class="grid3">{self._execution_track_cards(mission.get("execution_tracks", []))}</div>
</section>
<section class="grid">
  <article class="card glass">
    <div class="kicker">Expected Impact</div>
    {self._impact_rows(proposal.get("expected_impact", []))}
  </article>
  <article class="card glass">
    <div class="kicker">Critical Risks</div>
    <ul>{"".join(f"<li>{html.escape(str(item))}</li>" for item in proposal.get("critical_risks", []))}</ul>
  </article>
</section>
<section class="card glass">
  <div class="kicker">Why This System Wins</div>
  <ul>{"".join(f"<li>{html.escape(str(x))}</li>" for x in why_use)}</ul>
</section>
<section class="card glass">
  <div class="kicker">Appendix</div>
  <details>
    <summary>Primary Deliverable Raw Text</summary>
    <pre>{html.escape(primary_output)}</pre>
  </details>
  <details style="margin-top:12px">
    <summary>Scenario Scaffolding</summary>
    <div class="grid3">{self._pillar_cards(proposal.get("pillars", []))}</div>
    <div class="grid3" style="margin-top:12px">{self._phase_cards(proposal.get("phases", []))}</div>
  </details>
  <details style="margin-top:12px">
    <summary>Benchmark Fit And Boundary</summary>
    <table><thead><tr><th>Benchmark</th><th>Fit</th><th>Strength</th><th>Gap</th></tr></thead><tbody>{self._benchmark_rows(mission.get("benchmark_targets", []))}</tbody></table>
    <p style="margin-top:12px">{html.escape(str(mission.get("honest_boundary", "")))}</p>
    <ul>{"".join(f"<li>{html.escape(str(item))}</li>" for item in mission.get("review_questions", []))}</ul>
  </details>
  <details style="margin-top:12px">
    <summary>Comparisons And Leaderboard</summary>
    <h3>Built-in Archetype Comparison</h3>
    <table><thead><tr><th>Archetype</th><th>Gap</th><th>Advantage</th><th>Wins</th></tr></thead><tbody>{self._compare_rows(comparison.get("archetypes", []), compact=True)}</tbody></table>
    <h3 style="margin-top:12px">Agent Comparison</h3>
    <table><thead><tr><th>Agent</th><th>Score</th><th>Reason</th></tr></thead><tbody>{self._agent_rows(agent_comparison.get("rows", []))}</tbody></table>
    <h3 style="margin-top:12px">Research Lab Leaderboard</h3>
    <table><thead><tr><th>Candidate</th><th>Composite</th><th>Value</th><th>Pass</th><th>Safety</th><th>Pareto</th></tr></thead><tbody>{self._leaderboard_rows(lab.get("leaderboard", []))}</tbody></table>
  </details>
  <details style="margin-top:12px">
    <summary>Score Provenance</summary>
    <div class="grid">
      <article>
        <h3>Measured Signals</h3>
        <table><thead><tr><th>Name</th><th>Value</th><th>Basis</th></tr></thead><tbody>{self._score_rows(score_provenance.get("facts", []))}</tbody></table>
      </article>
      <article>
        <h3>Internal Heuristics</h3>
        <table><thead><tr><th>Name</th><th>Value</th><th>Basis</th></tr></thead><tbody>{self._score_rows(score_provenance.get("heuristics", []))}</tbody></table>
        <ul>{"".join(f"<li>{html.escape(str(item))}</li>" for item in score_provenance.get("warnings", []))}</ul>
      </article>
    </div>
  </details>
  <details style="margin-top:12px">
    <summary>Mission Pack JSON</summary>
    <pre>{html.escape(json.dumps(mission, indent=2, ensure_ascii=False))}</pre>
  </details>
  <details style="margin-top:12px">
    <summary>Internal Skills And Tools</summary>
    <table><thead><tr><th>Type</th><th>Name</th><th>Purpose</th></tr></thead><tbody>{self._appendix_rows(query.get("selected_skills", []), delivery.get("run_summary", {}).get("top_discovery", []), delivery.get("run_summary", {}).get("steps", []))}</tbody></table>
  </details>
  <details style="margin-top:12px">
    <summary>Trace Snapshot</summary>
    <pre>{html.escape(str(payload.get("router", {}).get("trace_views", "")))}</pre>
  </details>
</section>
<section class="card"><small>Generated at {html.escape(str(payload.get("generated_at", "")))} | schema {html.escape(str(payload.get("schema", STUDIO_SCHEMA)))} | query {html.escape(str(query.get("text", "")))}</small></section>
</div></body></html>"""

    @staticmethod
    def render_press_brief_markdown(payload: dict[str, Any], paths: dict[str, Any]) -> str:
        """Render a launch-style markdown brief for humans."""

        identity = payload.get("identity", {})
        query = payload.get("query", {})
        mission = payload.get("mission", {})
        frontier = payload.get("frontier", {})
        comparison = payload.get("comparison", {}).get("positioning", {})
        proposal = payload.get("proposal", {})
        agent_comparison = payload.get("agent_comparison", {})
        lab = payload.get("lab", {})
        release = lab.get("release_decision", {})
        router_quality = payload.get("router", {}).get("analysis", {}).get("quality", {})
        interop = payload.get("interop", {}).get("summary", {})
        story = payload.get("story", {})
        delivery = payload.get("harness", {})
        generation = delivery.get("generation", {}) if isinstance(delivery, dict) else {}
        evidence = delivery.get("run_summary", {}).get("evidence", {}) if isinstance(delivery, dict) else {}
        primary_output = StudioShowcaseBuilder._primary_output_text(payload)
        why_use = payload.get("why_use_this", [])
        score_provenance = payload.get("score_provenance", {})
        return "\n".join(
            [
                f"# {identity.get('name', 'Agent Harness Studio')}",
                "",
                str(identity.get("one_liner", "")),
                "",
                "## Task",
                "",
                str(query.get("text", "")),
                "",
                "## Primary Deliverable",
                "",
                primary_output,
                "",
                "## Task Context",
                "",
                str(story.get("release_need", "")),
                "",
                "## Evidence References",
                "",
                *(f"- {item}" for item in evidence.get("citations", [])[:8]),
                "",
                "## Deliverable Package",
                "",
                f"- Type: {mission.get('title', '')}",
                f"- Primary deliverable: {mission.get('primary_deliverable', '')}",
                *(f"- Deliverable: {item.get('title', '')} -> {item.get('description', '')}" for item in mission.get("deliverables", [])[:6]),
                "",
                "## Runtime Notes",
                "",
                f"- Mode: {generation.get('mode', 'baseline')}",
                f"- Live agent success: {generation.get('live_agent_success', False)}",
                f"- Model: {generation.get('model', '') or '-'}",
                f"- Calls used: {generation.get('calls_used', 0)}",
                "",
                "## Demo Snapshot",
                "",
                f"- Scenario: {payload.get('scenario', {}).get('name', '-')}",
                f"- Selected agent: {query.get('selected_agent', '-')}",
                f"- Skills: {', '.join(query.get('selected_skills', [])) or '-'}",
                f"- Internal frontier estimate: {float(frontier.get('score', 0.0)):.3f}",
                f"- Bottleneck axis: {frontier.get('bottleneck', {}).get('axis', '-')}",
                f"- Release decision: {release.get('decision', 'block')} ({release.get('reason', '-')})",
                f"- Robust expected utility: {_safe_float(router_quality.get('robust_expected_utility', 0.0)):.3f}",
                f"- Robust worst case: {_safe_float(router_quality.get('robust_worst_case_utility', 0.0)):.3f}",
                f"- Avg uncertainty: {_safe_float(router_quality.get('avg_uncertainty', 0.0)):.3f}",
                f"- Interop frameworks: {int(interop.get('framework_count', 0))}",
                f"- Exported skill entries: {int(interop.get('total_skill_entries', 0))}",
                "",
                "## Why This Is Different",
                "",
                *(f"- {item}" for item in why_use),
                "",
                "## Artifact Bundle",
                "",
                f"- Deliverable: {paths.get('deliverable', '')}",
                f"- JSON payload: {paths.get('json', '')}",
                f"- HTML showcase: {paths.get('html', '')}",
                f"- Press brief: {paths.get('brief', '')}",
                f"- Bundle manifest: {paths.get('manifest', '')}",
                f"- Interop bundle index: {paths.get('interop', {}).get('index', '-') if isinstance(paths.get('interop'), dict) else '-'}",
                "",
                "## Appendix",
                "",
                f"- Agent comparison winner: {agent_comparison.get('winner', '-')}",
                f"- Agent score gap: {_safe_float(agent_comparison.get('score_gap', 0.0)):.4f}",
                f"- Built-in positioning: {comparison.get('headline', '')}",
                *(f"- Fact: {item.get('name', '')}={item.get('value', '')} ({item.get('basis', '')})" for item in score_provenance.get("facts", [])[:6]),
                *(f"- Heuristic: {item.get('name', '')}={item.get('value', '')} ({item.get('basis', '')})" for item in score_provenance.get("heuristics", [])[:6]),
                f"_Generated at {payload.get('generated_at', '')}_",
                "",
            ]
        )

    @staticmethod
    def _bundle_manifest(payload: dict[str, Any], paths: dict[str, Any]) -> dict[str, Any]:
        """Build a compact manifest for downstream consumers."""

        comparison = payload.get("comparison", {}).get("positioning", {})
        release = payload.get("lab", {}).get("release_decision", {})
        return {
            "schema": "agent-harness-studio-bundle/v1",
            "generated_at": payload.get("generated_at", ""),
            "identity": payload.get("identity", {}),
            "mission": payload.get("mission", {}),
            "headline": comparison.get("headline", ""),
            "release_decision": release,
            "frontier": payload.get("frontier", {}),
            "query": payload.get("query", {}),
            "artifacts": paths,
        }

    @staticmethod
    def _metric_rows(capability: dict[str, float]) -> str:
        blocks: list[str] = []
        for name, value in capability.items():
            pct = round(_clamp01(value) * 100.0, 1)
            label = html.escape(name.replace("_", " ").title())
            blocks.append(
                "<div class='metric'>"
                f"<div class='row'><span>{label}</span><strong>{pct:.1f}%</strong></div>"
                f"<div class='bar'><div class='fill' style='width:{pct:.1f}%'></div></div>"
                "</div>"
            )
        return "".join(blocks)

    @staticmethod
    def _score_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "<tr><td colspan='3'>No score provenance data.</td></tr>"
        parts: list[str] = []
        for row in rows[:8]:
            value = row.get("value", "")
            if isinstance(value, float):
                value_text = f"{value:.4f}"
            else:
                value_text = str(value)
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('name', '')))}</td>"
                f"<td>{html.escape(value_text)}</td>"
                f"<td>{html.escape(str(row.get('basis', '')))}</td>"
                "</tr>"
            )
        return "".join(parts)

    @staticmethod
    def _leaderboard_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "<tr><td colspan='6'>No leaderboard data.</td></tr>"
        parts: list[str] = []
        for row in rows[:8]:
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('candidate', '')))}</td>"
                f"<td>{_safe_float(row.get('composite_score', 0.0)):.4f}</td>"
                f"<td>{_safe_float(row.get('avg_value_index', 0.0)):.2f}</td>"
                f"<td>{_safe_float(row.get('pass_rate', 0.0)):.3f}</td>"
                f"<td>{_safe_float(row.get('avg_security_alignment', 0.0)):.3f}</td>"
                f"<td>{'yes' if bool(row.get('pareto_frontier', False)) else 'no'}</td>"
                "</tr>"
            )
        return "".join(parts)

    @staticmethod
    def _compare_rows(rows: list[dict[str, Any]], compact: bool = False) -> str:
        if not rows:
            return "<tr><td colspan='5'>No comparison data.</td></tr>"
        parts: list[str] = []
        for row in rows:
            wins = html.escape(", ".join(row.get("wins", [])) or "-")
            losses = html.escape(", ".join(row.get("losses", [])) or "-")
            if compact:
                parts.append(
                    "<tr>"
                    f"<td>{html.escape(str(row.get('name', '')))}</td>"
                    f"<td>{_safe_float(row.get('frontier_gap', 0.0)):+.4f}</td>"
                    f"<td>{_safe_float(row.get('advantage_ratio', 0.0)):.2f}</td>"
                    f"<td>{wins}</td>"
                    "</tr>"
                )
            else:
                parts.append(
                    "<tr>"
                    f"<td>{html.escape(str(row.get('name', '')))}</td>"
                    f"<td>{_safe_float(row.get('frontier_gap', 0.0)):+.4f}</td>"
                    f"<td>{_safe_float(row.get('advantage_ratio', 0.0)):.2f}</td>"
                    f"<td>{wins}</td>"
                    f"<td>{losses}</td>"
                    "</tr>"
                )
        return "".join(parts)

    @staticmethod
    def _pillar_cards(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        parts: list[str] = []
        for row in rows:
            parts.append(
                "<article class='card glass'>"
                f"<div class='kicker'>{html.escape(str(row.get('title', 'Pillar')))}</div>"
                f"<h2>{html.escape(str(row.get('title', '')))}</h2>"
                f"<p>{html.escape(str(row.get('summary', '')))}</p>"
                f"<p><strong>Integration:</strong> {html.escape(str(row.get('integration', '')))}</p>"
                "</article>"
            )
        return "".join(parts)

    @staticmethod
    def _deliverable_cards(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        parts: list[str] = []
        for row in rows[:4]:
            parts.append(
                "<article class='card glass'>"
                f"<div class='kicker'>{html.escape(str(row.get('audience', 'deliverable')))}</div>"
                f"<h2>{html.escape(str(row.get('title', 'Deliverable')))}</h2>"
                f"<p>{html.escape(str(row.get('description', '')))}</p>"
                f"<p><strong>Status:</strong> {html.escape(str(row.get('status', 'draft')))}</p>"
                f"<p><strong>Signal:</strong> {html.escape(str(row.get('evidence_hint', '-')))}</p>"
                "</article>"
            )
        return "".join(parts)

    @staticmethod
    def _phase_cards(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        parts: list[str] = []
        for row in rows:
            parts.append(
                "<article class='phase'>"
                f"<div class='kicker'>{html.escape(str(row.get('phase', 'Phase')))}</div>"
                f"<h3>{html.escape(str(row.get('phase', '')))}</h3>"
                f"<ul>{''.join(f'<li>{html.escape(str(item))}</li>' for item in row.get('actions', []))}</ul>"
                f"<p><strong>Success:</strong> {html.escape(', '.join(row.get('success_metrics', [])) or '-')}</p>"
                "</article>"
            )
        return "".join(parts)

    @staticmethod
    def _execution_track_cards(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        parts: list[str] = []
        for row in rows[:4]:
            parts.append(
                "<article class='phase'>"
                f"<div class='kicker'>{html.escape(str(row.get('name', 'Track')))}</div>"
                f"<h3>{html.escape(str(row.get('name', 'Track')))}</h3>"
                f"<p>{html.escape(str(row.get('focus', '')))}</p>"
                f"<p><strong>Success:</strong> {html.escape(str(row.get('success', '-')))}</p>"
                "</article>"
            )
        return "".join(parts)

    @staticmethod
    def _impact_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "<p>No expected impact recorded.</p>"
        parts: list[str] = []
        for row in rows:
            parts.append(
                "<div class='metric'>"
                f"<div class='row'><span>{html.escape(str(row.get('label', 'Impact')))}</span></div>"
                f"<strong>{html.escape(str(row.get('value', '')))}</strong>"
                "</div>"
            )
        return "".join(parts)

    @staticmethod
    def _chip_cloud(items: list[Any], tone: str = "muted") -> str:
        values = [str(item).strip() for item in items if str(item).strip()]
        if not values:
            return "<span class='badge'>No items</span>"
        background = "rgba(255,255,255,.10)" if tone == "bright" else "rgba(10,30,48,.06)"
        color = "#edf7ff" if tone == "bright" else "#24445f"
        border = "rgba(255,255,255,.16)" if tone == "bright" else "rgba(10,30,48,.10)"
        return "".join(
            f"<span class='badge' style='background:{background};color:{color};border-color:{border}'>{html.escape(value)}</span>"
            for value in values[:12]
        )

    @staticmethod
    def _runtime_signal_cards(
        *,
        run_summary: dict[str, Any],
        generation: dict[str, Any],
        interop: dict[str, Any],
        evidence: dict[str, Any],
    ) -> str:
        metrics = run_summary.get("metrics", {}) if isinstance(run_summary, dict) else {}
        cards = [
            ("Model", str(generation.get("model", "")) or "baseline"),
            ("Live Calls", str(int(generation.get("calls_used", 0)))),
            ("Evidence", str(int(evidence.get("record_count", 0)))),
            ("Interop", str(int(interop.get("framework_count", 0)))),
            ("Tool Success", f"{_safe_float(metrics.get('tool_success_rate', 0.0)):.2f}"),
            ("Completion", f"{_safe_float(metrics.get('completion_score', 0.0)):.2f}"),
        ]
        return "".join(
            "<div class='signal'>"
            f"<div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{html.escape(value)}</div>"
            "</div>"
            for label, value in cards
        )

    @staticmethod
    def _artifact_family_rows(mission: dict[str, Any], interop: dict[str, Any]) -> str:
        deliverables = mission.get("deliverables", []) if isinstance(mission.get("deliverables", []), list) else []
        rows: list[str] = []
        for item in deliverables[:6]:
            if not isinstance(item, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('title', 'Deliverable')))}</td>"
                f"<td>{html.escape(str(item.get('description', '')))}</td>"
                "</tr>"
            )
        output_views = mission.get("output_views", []) if isinstance(mission.get("output_views", []), list) else []
        for view in output_views[:4]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(view))}</td>"
                "<td>User-facing output format in the generated bundle.</td>"
                "</tr>"
            )
        rows.append(
            "<tr>"
            "<td>OpenAI / Anthropic Skill Export</td>"
            f"<td>{int(interop.get('framework_count', 0))} framework bundle(s) and {int(interop.get('total_skill_entries', 0))} exported entries.</td>"
            "</tr>"
        )
        if not rows:
            return "<tr><td colspan='2'>No artifact families.</td></tr>"
        return "".join(rows)

    @staticmethod
    def _benchmark_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "<tr><td colspan='4'>No benchmark mapping.</td></tr>"
        parts: list[str] = []
        for row in rows[:6]:
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('name', '')))}</td>"
                f"<td>{html.escape(str(row.get('fit', '')))}</td>"
                f"<td>{html.escape(str(row.get('strength', '')))}</td>"
                f"<td>{html.escape(str(row.get('gap', '')))}</td>"
                "</tr>"
            )
        return "".join(parts)

    @staticmethod
    def _agent_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "<tr><td colspan='3'>No agent comparison data.</td></tr>"
        parts: list[str] = []
        for row in rows:
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('agent', '')))}</td>"
                f"<td>{_safe_float(row.get('score', 0.0)):.4f}</td>"
                f"<td>{html.escape(str(row.get('reason', '')))}</td>"
                "</tr>"
            )
        return "".join(parts)

    @staticmethod
    def _appendix_rows(
        selected_skills: list[Any],
        discovery: list[dict[str, Any]],
        steps: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        for name in selected_skills[:6]:
            raw = str(name)
            parts.append(
                "<tr>"
                "<td>Skill</td>"
                f"<td>{html.escape(raw)}</td>"
                f"<td>{html.escape(TOOL_DISPLAY_ALIASES.get(raw, _sanitize_business_text(raw.replace('_', ' '))))}</td>"
                "</tr>"
            )
        for item in discovery[:5]:
            raw = str(item.get("name", ""))
            parts.append(
                "<tr>"
                "<td>Discovery</td>"
                f"<td>{html.escape(raw)}</td>"
                f"<td>{html.escape(', '.join(item.get('reasons', [])) or '-')}</td>"
                "</tr>"
            )
        for row in steps[:6]:
            raw = str(row.get("tool", ""))
            parts.append(
                "<tr>"
                "<td>Executed Tool</td>"
                f"<td>{html.escape(raw)}</td>"
                f"<td>{html.escape(TOOL_DISPLAY_ALIASES.get(raw, _sanitize_business_text(raw.replace('_', ' '))))}</td>"
                "</tr>"
            )
        if not parts:
            return "<tr><td colspan='3'>No appendix rows.</td></tr>"
        return "".join(parts)
