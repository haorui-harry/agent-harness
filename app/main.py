"""CLI entry point for the LangGraph Skill Router."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from app.benchmark.adapters import BenchmarkAdapterRunner
from app.benchmark.evaluate import run_benchmark
from app.core.state import AgentStyle, GraphState
from app.core.mission import MissionRegistry
from app.demo import (
    demo_benchmark,
    demo_conflict_resolution,
    demo_full_trace,
    demo_marketplace,
    demo_personality_comparison,
    demo_press_launch,
    run_all_demos,
)
from app.ecosystem.marketplace import (
    discover_for_query,
    get_provider_stats,
    get_trending_skills,
    list_marketplace_skills,
    list_skills_by_tag,
)
from app.ecosystem.reputation import submit_marketplace_rating
from app.ecosystem.store import import_marketplace_from_file
from app.graph import build_graph
from app.harness import HarnessEngine, HarnessConstraints
from app.harness.live_agent import LiveModelConfig
from app.policy.center import SystemMode, normalize_mode, policy_for_mode
from app.personality.profiles import blend_profiles, get_profile, list_profiles
from app.skills.registry import (
    get_skill_card,
    get_skill_lifecycle_status,
    list_external_skills,
    load_external_skills_from_file,
)
from app.skills.interop import export_interop_all, export_interop_catalog, write_interop_bundle
from app.studio.flagship import StudioShowcaseBuilder
from app.studio.proposals import ProposalRegistry
from app.tracing.analyzer import RoutingAnalyzer
from app.tracing.store import list_recent_traces, load_trace, save_trace
from app.tracing.visualizer import render_trace_views
from app.utils.console import Console
from app.utils.display import (
    print_benchmark_results,
    print_conflict_report,
    print_contract,
    print_execution_timeline,
    print_final_output,
    print_marketplace_browser,
    print_personality,
    print_reasoning_path,
    print_routing_quality,
    print_routing_trace,
)

console = Console()
app = typer.Typer(
    name="skill-router",
    help="Multi-agent system with dynamic, complementary skill selection.",
    add_completion=False,
)
HARNESS = HarnessEngine()
PROPOSALS = ProposalRegistry()
MISSIONS = MissionRegistry()
STUDIO = StudioShowcaseBuilder(harness=HARNESS)
BENCHMARK_ADAPTERS = BenchmarkAdapterRunner()


def _parse_style(style: str) -> AgentStyle | None:
    if style == "auto":
        return None
    try:
        return AgentStyle(style)
    except ValueError as exc:
        valid = ", ".join([item.value for item in AgentStyle] + ["auto"])
        raise typer.BadParameter(f"Invalid style '{style}'. Valid values: {valid}") from exc


def _parse_mode(mode: str) -> SystemMode:
    parsed = normalize_mode(mode)
    if parsed.value != mode and mode != "":
        valid = ", ".join(item.value for item in SystemMode)
        raise typer.BadParameter(f"Invalid mode '{mode}'. Valid values: {valid}")
    return parsed


def _extract_result_payload(result: dict | GraphState) -> dict:
    if isinstance(result, dict):
        return result
    return result.model_dump()


def _mask_secret(secret: str) -> str:
    value = secret.strip()
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}***{value[-4:]}"


def _build_live_model_overrides(
    model_base_url: str,
    model_api_key: str,
    model_name: str,
    timeout_seconds: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    if model_base_url.strip():
        payload["base_url"] = model_base_url.strip()
    if model_api_key.strip():
        payload["api_key"] = model_api_key.strip()
    if model_name.strip():
        payload["model_name"] = model_name.strip()
    if timeout_seconds > 0:
        payload["timeout_seconds"] = timeout_seconds
    payload["temperature"] = max(0.0, min(1.5, float(temperature)))
    payload["max_tokens"] = max(128, min(8192, int(max_tokens)))
    return payload


def _default_live_experiment_queries() -> list[str]:
    return [
        "Audit this critical launch strategy and enumerate governance controls.",
        "Compare three rollout options and produce a risk-weighted decision memo.",
        "Design a high-velocity but safe execution plan with dependency checkpoints.",
        "Map ecosystem opportunities and recommend complementary external skills.",
    ]


def _build_subagent_graphs_from_mission(mission: dict[str, Any]) -> list[dict[str, Any]]:
    tracks = mission.get("execution_tracks", []) if isinstance(mission.get("execution_tracks", []), list) else []
    if not tracks:
        tracks = [
            {"name": "Track 1", "focus": mission.get("primary_deliverable", "mission"), "success": mission.get("summary", "")}
        ]
    graphs: list[dict[str, Any]] = []
    for index, item in enumerate(tracks[:3], start=1):
        label = str(item.get("name", f"Subagent {index}"))
        focus = str(item.get("focus", mission.get("primary_deliverable", "")))
        success = str(item.get("success", ""))
        graphs.append(
            {
                "name": label,
                "graph": {
                    "graph_id": f"subagent-{index}",
                    "nodes": [
                        {
                            "node_id": f"scope_{index}",
                            "title": f"{label} Scope",
                            "node_type": "routing",
                            "status": "ready",
                            "depends_on": [],
                            "commands": [],
                            "notes": [focus],
                            "artifacts": [],
                            "metrics": {},
                        },
                        {
                            "node_id": f"execute_{index}",
                            "title": f"{label} Execute",
                            "node_type": "execution_plan",
                            "status": "ready",
                            "depends_on": [f"scope_{index}"],
                            "commands": [focus],
                            "notes": [success],
                            "artifacts": [],
                            "metrics": {},
                        },
                        {
                            "node_id": f"report_{index}",
                            "title": f"{label} Report",
                            "node_type": "review",
                            "status": "ready",
                            "depends_on": [f"execute_{index}"],
                            "commands": [],
                            "notes": [success],
                            "artifacts": [],
                            "metrics": {},
                        },
                    ],
                },
                "context": {"mission_name": mission.get("name", ""), "track": item},
            }
        )
    return graphs


def _build_reasoning_path(payload: dict) -> list[dict]:
    """Build a lightweight reasoning path for trace/analyze commands."""

    if payload.get("reasoning_path"):
        return payload["reasoning_path"]

    path: list[dict] = []

    agent = payload.get("routing_trace", {}).get("agent_decision", {})
    if agent:
        path.append(
            {
                "step": 1,
                "event": "agent_selected",
                "elapsed_ms": 1.0,
                "description": "Agent selected",
                "data": {
                    "selected": agent.get("selected", []),
                    "complexity": agent.get("query_complexity", ""),
                    "collaboration": agent.get("collaboration", {}),
                },
            }
        )

    skill = payload.get("routing_trace", {}).get("skill_decision", {})
    if skill:
        path.append(
            {
                "step": len(path) + 1,
                "event": "skill_selected",
                "elapsed_ms": 3.0,
                "description": "Skills selected",
                "data": {
                    "selected": skill.get("selected", []),
                    "execution_order": skill.get("execution_order", []),
                },
            }
        )

    path.append(
        {
            "step": len(path) + 1,
            "event": "execution_completed",
            "elapsed_ms": 8.0,
            "description": "Execution completed",
            "data": {
                "contexts": {
                    name: round(ctx.get("duration_ms", 0.0), 2)
                    for name, ctx in payload.get("execution_contexts", {}).items()
                },
                "conflict_count": len(payload.get("conflicts_detected", [])),
                "consensus": payload.get("consensus_result", {}).get("strength", "unknown"),
            },
        }
    )

    return path


def run_query(
    query: str,
    style: AgentStyle | None = None,
    max_skills: int = 3,
    mode: SystemMode = SystemMode.BALANCED,
) -> dict:
    graph = build_graph()
    state = GraphState(
        query=query,
        forced_style=style,
        max_skills=max_skills,
        system_mode=mode.value,
        policy=policy_for_mode(mode).to_dict(),
    )
    result = graph.invoke(state)
    payload = _extract_result_payload(result)
    payload.setdefault("reasoning_path", _build_reasoning_path(payload))
    trace_id = save_trace(payload)
    payload["trace_id"] = trace_id
    payload.setdefault("routing_trace", {})["trace_id"] = trace_id
    if payload.get("response_contract"):
        payload["response_contract"]["user"]["trace_summary"]["trace_id"] = trace_id
        payload["response_contract"]["debug"]["full_trace_id"] = trace_id
    return payload


@app.command()
def run(
    query: str = typer.Argument(..., help="The query to process"),
    style: str = typer.Option(
        "auto",
        "--style",
        "-s",
        help="Force an agent style: aggressive, cautious, creative, balanced, or auto",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full routing trace"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output full payload as JSON"),
    max_skills: int = typer.Option(3, "--max-skills", "-k", help="Max skills to select"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode: fast|balanced|deep|safety_critical"),
    show_contract: bool = typer.Option(False, "--contract", help="Show structured response contract"),
) -> None:
    """Process a query through the hierarchical routing system."""

    console.print(f"\n[bold]Processing:[/] {query}\n")
    payload = run_query(
        query=query,
        style=_parse_style(style),
        max_skills=max_skills,
        mode=_parse_mode(mode),
    )

    if json_output:
        console.print_json(json.dumps(payload, indent=2, default=str))
        return

    print_routing_trace(payload.get("routing_trace", {}), query)
    print_personality(payload.get("personality", {}), payload.get("agent_name", ""))
    print_execution_timeline(payload.get("execution_contexts", {}))
    print_conflict_report(payload.get("conflicts_detected", []), payload.get("consensus_result", {}))
    print_routing_quality(payload.get("routing_metrics", {}))
    print_final_output(payload.get("final_output", ""))
    if show_contract:
        print_contract(payload.get("response_contract", {}))

    if verbose:
        console.print("\n[dim]Full trace payload:[/]")
        console.print_json(json.dumps(payload.get("routing_trace", {}), indent=2, default=str))


@app.command("benchmark")
def benchmark_command() -> None:
    """Run routing benchmark: greedy vs random vs complementary."""

    result = run_benchmark()
    print_benchmark_results(result)


@app.command("benchmark-adapters")
def benchmark_adapters_command() -> None:
    """List unified benchmark adapters."""

    console.print_json(json.dumps({"adapters": BENCHMARK_ADAPTERS.list_adapters()}, indent=2, default=str))


@app.command("benchmark-suite")
def benchmark_suite_command(
    adapters: str = typer.Option("", "--adapters", help="Comma-separated adapter names"),
    repeats: int = typer.Option(1, "--repeats", help="Repeat count for harness-lab adapters"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run unified benchmark suite across internal adapters."""

    selected = [item.strip() for item in adapters.split(",") if item.strip()] if adapters else None
    payload = BENCHMARK_ADAPTERS.run_suite(engine=HARNESS, adapters=selected, repeats=max(1, repeats))
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Benchmark suite written:[/] {path}")
        return
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("benchmark-ablation")
def benchmark_ablation_command(
    repeats: int = typer.Option(1, "--repeats", help="Repeat count for ablation study"),
    scenarios: str = typer.Option("", "--scenarios", help="Comma-separated scenario ids"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run standardized ablation study and failure clustering."""

    scenario_ids = [item.strip() for item in scenarios.split(",") if item.strip()] if scenarios else None
    payload = BENCHMARK_ADAPTERS.run_ablation(
        engine=HARNESS,
        repeats=max(1, repeats),
        scenario_ids=scenario_ids,
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Benchmark ablation written:[/] {path}")
        return
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("market-search")
def market_search_command(
    query: str = typer.Argument(..., help="Search query for marketplace skills"),
    limit: int = typer.Option(5, "--limit", "-n", help="Max results"),
) -> None:
    """Search marketplace skills with hybrid retrieval + reputation."""

    result = discover_for_query(query=query, limit=limit)
    print_marketplace_browser(result)


@app.command("rate-skill")
def rate_skill_command(
    skill_name: str = typer.Argument(..., help="Marketplace skill name"),
    rating: float = typer.Argument(..., help="Rating from 1 to 5"),
) -> None:
    """Submit a rating for a marketplace skill."""

    ok = submit_marketplace_rating(skill_name=skill_name, rating=rating)
    if not ok:
        raise typer.BadParameter(f"Skill not found: {skill_name}")
    console.print(f"[green]Rating saved:[/] {skill_name} -> {rating}")


@app.command("personality")
def personality_command(
    name: str = typer.Argument(
        "",
        help="Profile name (berserker, scholar, explorer, diplomat, surgeon, ensemble_master)",
    ),
    list_all: bool = typer.Option(False, "--list", "-l", help="List all profiles"),
    blend: str = typer.Option("", "--blend", help="Blend profiles, e.g. 'scholar:0.6,explorer:0.4'"),
) -> None:
    """Explore and blend agent personality profiles."""

    if list_all:
        console.print("\n".join(list_profiles()))
        return

    if blend:
        parts = [item.strip() for item in blend.split(",") if item.strip()]
        parsed: list[tuple[str, float]] = []
        for part in parts:
            if ":" not in part:
                raise typer.BadParameter("Blend format must be profile:weight,profile:weight")
            profile_name, weight_str = part.split(":", 1)
            parsed.append((profile_name.strip(), float(weight_str.strip())))
        mixed = blend_profiles(parsed)
        print_personality(
            {
                "risk_tolerance": mixed.risk_tolerance,
                "creativity_bias": mixed.creativity_bias,
                "diversity_preference": mixed.diversity_preference,
                "confidence_threshold": mixed.confidence_threshold,
                "collaboration_tendency": mixed.collaboration_tendency,
                "depth_vs_breadth": mixed.depth_vs_breadth,
            },
            "blended",
        )
        return

    if not name:
        raise typer.BadParameter("Provide a profile name, or use --list / --blend")

    profile = get_profile(name)
    if not profile:
        raise typer.BadParameter(f"Unknown profile: {name}")

    print_personality(
        {
            "risk_tolerance": profile.risk_tolerance,
            "creativity_bias": profile.creativity_bias,
            "diversity_preference": profile.diversity_preference,
            "confidence_threshold": profile.confidence_threshold,
            "collaboration_tendency": profile.collaboration_tendency,
            "depth_vs_breadth": profile.depth_vs_breadth,
        },
        name,
    )


@app.command("trace")
def trace_command(
    query: str = typer.Argument(..., help="Query to trace"),
    decisions_only: bool = typer.Option(False, "--decisions", "-d", help="Show only decision points"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    views: bool = typer.Option(False, "--views", help="Render multi-view trace dashboard"),
) -> None:
    """Run a query and show reasoning path."""

    payload = run_query(query, mode=_parse_mode(mode))
    path = payload.get("reasoning_path", [])

    if decisions_only:
        condensed = [
            item
            for item in path
            if "selected" in item.get("event", "") or "conflict" in item.get("event", "")
        ]
        print_reasoning_path(condensed)
    else:
        print_reasoning_path(path)

    if views:
        console.print(render_trace_views(payload.get("routing_trace", {})))


@app.command("ecosystem")
def ecosystem_command(
    action: str = typer.Argument("browse", help="browse | trending | providers | tags"),
    tag: str = typer.Option("", "--tag", "-t", help="Filter by tag"),
    provider: str = typer.Option("", "--provider", "-p", help="Provider name"),
) -> None:
    """Browse the skill marketplace ecosystem."""

    if action == "browse":
        print_marketplace_browser(list_marketplace_skills())
        return

    if action == "trending":
        print_marketplace_browser(get_trending_skills())
        return

    if action == "providers":
        if not provider:
            raise typer.BadParameter("Use --provider for provider stats")
        console.print_json(json.dumps(get_provider_stats(provider), indent=2, default=str))
        return

    if action == "tags":
        if not tag:
            raise typer.BadParameter("Use --tag with action=tags")
        print_marketplace_browser(list_skills_by_tag(tag))
        return

    raise typer.BadParameter("action must be one of: browse | trending | providers | tags")


@app.command("import-marketplace")
def import_marketplace_command(
    path: str = typer.Argument(..., help="Path to a marketplace JSON file containing skills[]"),
) -> None:
    """Import third-party marketplace skills from a local JSON file."""

    imported = import_marketplace_from_file(path)
    console.print(f"[green]Imported {imported} marketplace skill(s).[/]")


@app.command("import-external-skills")
def import_external_skills_command(
    path: str = typer.Argument(..., help="Path to external skill JSON spec"),
) -> None:
    """Load runtime external skills into in-process registry."""

    imported = load_external_skills_from_file(path)
    skills = [meta.name for meta in list_external_skills()]
    console.print(f"[green]Imported {imported} external skill(s).[/]")
    console.print(f"Active external skills: {skills}")


@app.command("skill-card")
def skill_card_command(
    name: str = typer.Argument(..., help="Skill name"),
) -> None:
    """Inspect a skill's SkillCard metadata and lifecycle status."""

    card = get_skill_card(name)
    if not card:
        raise typer.BadParameter(f"Unknown skill: {name}")
    card["lifecycle"] = get_skill_lifecycle_status(name)
    console.print_json(json.dumps(card, indent=2, default=str))


@app.command("skill-packages")
def skill_packages_command(
    enabled_only: bool = typer.Option(False, "--enabled-only", help="Only show enabled packages"),
) -> None:
    """List DeerFlow-style skill packages."""

    console.print_json(json.dumps({"packages": HARNESS.list_skill_packages(enabled_only=enabled_only)}, indent=2, default=str))


@app.command("skill-package")
def skill_package_command(
    name: str = typer.Argument(..., help="Skill package name"),
) -> None:
    """Inspect one DeerFlow-style skill package."""

    payload = HARNESS.get_skill_package(name)
    if payload is None:
        raise typer.BadParameter(f"Unknown skill package: {name}")
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("skill-package-update")
def skill_package_update_command(
    name: str = typer.Argument(..., help="Skill package name"),
    enabled: bool = typer.Option(..., "--enabled/--disabled", help="Enable or disable the package"),
) -> None:
    """Enable or disable one skill package."""

    payload = HARNESS.update_skill_package(name, enabled=enabled)
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("skill-package-install")
def skill_package_install_command(
    archive_path: str = typer.Argument(..., help="Path to a .skill archive"),
) -> None:
    """Install a DeerFlow-style .skill archive into skills/custom."""

    payload = HARNESS.install_skill_package_archive(archive_path)
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("skills-interop-export")
def skills_interop_export_command(
    framework: str = typer.Option("all", "--framework", "-f", help="Target: all|openai|anthropic"),
    output_dir: str = typer.Option("reports/skills_interop", "--output-dir", "-o", help="Output directory"),
    include_marketplace: bool = typer.Option(True, "--marketplace/--no-marketplace", help="Include marketplace skills"),
    include_external: bool = typer.Option(True, "--external/--no-external", help="Include external runtime skills"),
    include_harness_tools: bool = typer.Option(
        True,
        "--harness-tools/--no-harness-tools",
        help="Suggest harness tools in exported skill metadata",
    ),
) -> None:
    """Export compatibility bundles for OpenAI/Anthropic skill ecosystems."""

    target = framework.strip().lower()
    if target == "all":
        payload = export_interop_all(
            include_marketplace=include_marketplace,
            include_external=include_external,
            include_harness_tools=include_harness_tools,
        )
    elif target in {"openai", "anthropic"}:
        payload = export_interop_catalog(
            framework=target,
            include_marketplace=include_marketplace,
            include_external=include_external,
            include_harness_tools=include_harness_tools,
        )
    else:
        raise typer.BadParameter("framework must be one of: all|openai|anthropic")

    written = write_interop_bundle(payload, output_dir=output_dir)
    result = {
        "framework": target,
        "output_dir": output_dir,
        "include_marketplace": include_marketplace,
        "include_external": include_external,
        "include_harness_tools": include_harness_tools,
        "written": written,
    }
    console.print_json(json.dumps(result, indent=2, default=str))


@app.command("analyze")
def analyze_command(query: str = typer.Argument(..., help="Query to analyze")) -> None:
    """Run a query and produce routing quality analysis."""

    payload = run_query(query)
    analyzer = RoutingAnalyzer()
    analysis = analyzer.analyze(payload.get("reasoning_path", []), payload.get("routing_metrics", {}))
    console.print_json(json.dumps(analysis, indent=2, default=str))


@app.command("policy")
def policy_command(
    mode: str = typer.Argument("balanced", help="Mode: fast|balanced|deep|safety_critical"),
) -> None:
    """Show policy bundle for a system mode."""

    bundle = policy_for_mode(_parse_mode(mode))
    console.print_json(json.dumps(bundle.to_dict(), indent=2, default=str))


@app.command("replay")
def replay_command(
    trace_id: str = typer.Argument(..., help="Trace ID to replay"),
) -> None:
    """Replay a stored trace payload."""

    payload = load_trace(trace_id)
    if not payload:
        raise typer.BadParameter(f"Trace not found: {trace_id}")
    query = payload.get("query", "")
    print_routing_trace(payload.get("routing_trace", {}), query)
    print_reasoning_path(payload.get("reasoning_path", []))
    print_final_output(payload.get("final_output", ""))
    print_contract(payload.get("response_contract", {}))


@app.command("traces")
def traces_command(limit: int = typer.Option(10, "--limit", "-n", help="Max traces to list")) -> None:
    """List recent trace IDs for replay/audit."""

    console.print_json(json.dumps(list_recent_traces(limit=limit), indent=2, default=str))


@app.command("mode-compare")
def mode_compare_command(
    query: str = typer.Argument(..., help="Query to compare across modes"),
) -> None:
    """Run the same query in fast/balanced/deep/safety_critical and compare routes."""

    rows = []
    for mode in [SystemMode.FAST, SystemMode.BALANCED, SystemMode.DEEP, SystemMode.SAFETY_CRITICAL]:
        payload = run_query(query=query, mode=mode)
        rows.append(
            {
                "mode": mode.value,
                "agent": payload.get("agent_name", ""),
                "skills": payload.get("selected_skills", []),
                "coverage": round(float(payload.get("routing_metrics", {}).get("coverage", 0.0)), 3),
                "redundancy": round(float(payload.get("routing_metrics", {}).get("redundancy", 0.0)), 3),
                "conflicts": int(payload.get("routing_metrics", {}).get("conflict_count", 0.0)),
                "trace_id": payload.get("trace_id", ""),
            }
        )
    console.print_json(json.dumps(rows, indent=2, default=str))


@app.command("demo")
def demo_command(
    scenario: str = typer.Argument(
        "all",
        help="Scenario: all, personality, conflict, benchmark, marketplace, trace, launch",
    ),
) -> None:
    """Run demo scenarios showcasing system capabilities."""

    if scenario == "all":
        run_all_demos()
        return
    if scenario == "personality":
        demo_personality_comparison()
        return
    if scenario == "conflict":
        demo_conflict_resolution()
        return
    if scenario == "benchmark":
        demo_benchmark()
        return
    if scenario == "marketplace":
        demo_marketplace()
        return
    if scenario == "trace":
        demo_full_trace()
        return
    if scenario == "launch":
        demo_press_launch()
        return

    raise typer.BadParameter(
        "Scenario must be one of: all, personality, conflict, benchmark, marketplace, trace, launch"
    )


@app.command("launch-demo")
def launch_demo_command(
    output_dir: str = typer.Option("reports/launch_demo", "--output-dir", "-o", help="Output directory"),
    tag: str = typer.Option("press", "--tag", help="Output tag"),
    query: str = typer.Option("", "--query", help="Override the launch demo query/scenario"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model generation for the launch story"),
    max_model_calls: int = typer.Option(8, "--max-model-calls", help="Live model call budget per run (<=50)"),
) -> None:
    """Generate launch-ready showcase assets for external presentation."""

    payload = demo_press_launch(
        output_dir=output_dir,
        tag=tag,
        live_agent=live_agent,
        max_model_calls=max_model_calls,
        query=query,
    )
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("proposal-scenarios")
def proposal_scenarios_command() -> None:
    """List built-in proposal scenarios for studio/demo generation."""

    console.print_json(json.dumps({"scenarios": PROPOSALS.list_cards()}, indent=2, default=str))


@app.command("mission-profiles")
def mission_profiles_command() -> None:
    """List generalized mission-pack profiles for studio outputs."""

    console.print_json(json.dumps({"missions": MISSIONS.list_cards()}, indent=2, default=str))


@app.command("agent-thread-create")
def agent_thread_create_command(
    title: str = typer.Argument("", help="Optional thread title"),
    agent_name: str = typer.Option("", "--agent", help="Optional preferred agent name"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Create a persistent generic agent thread with workspace and outputs."""

    payload = HARNESS.create_thread(title=title, agent_name=agent_name)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Thread written:[/] {path}")
        return
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-threads")
def agent_threads_command(
    limit: int = typer.Option(20, "--limit", "-n", help="Max threads to list"),
) -> None:
    """List persistent generic agent threads."""

    console.print_json(json.dumps({"threads": HARNESS.list_threads(limit=limit)}, indent=2, default=str))


@app.command("agent-thread-show")
def agent_thread_show_command(
    thread_id: str = typer.Argument(..., help="Thread id"),
) -> None:
    """Show one persistent generic agent thread."""

    payload = HARNESS.get_thread(thread_id)
    if not payload:
        raise typer.BadParameter(f"Unknown thread id: {thread_id}")
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-interrupt")
def agent_thread_interrupt_command(
    thread_id: str = typer.Argument(..., help="Thread id"),
    reason: str = typer.Option("manual", "--reason", help="Interrupt reason"),
) -> None:
    """Request interrupt for one persistent generic agent thread."""

    payload = HARNESS.request_thread_interrupt(thread_id, reason=reason)
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness")
def harness_command(
    query: str = typer.Argument(..., help="Task query to run through harness"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    max_steps: int = typer.Option(4, "--max-steps", help="Harness planner max steps"),
    max_tool_calls: int = typer.Option(4, "--max-tool-calls", help="Harness max tool calls"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to JSON/YAML recipe file"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model agent enhancement"),
    max_model_calls: int = typer.Option(8, "--max-model-calls", help="Live model call budget per run (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Override model base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Override model API key"),
    model_name: str = typer.Option("", "--model-name", help="Override model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Live model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Live model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Live model max tokens"),
    json_output: bool = typer.Option(False, "--json", help="Render harness payload as JSON"),
) -> None:
    """Run harness loop: planner + tools + memory + guardrails + eval."""

    live_overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    run = HARNESS.run(
        query=query,
        constraints=HarnessConstraints(
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            enable_live_agent=live_agent,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        ),
        mode=_parse_mode(mode).value,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        live_model=live_overrides if live_overrides else None,
    )
    payload = HARNESS.run_to_dict(run)
    if json_output:
        console.print_json(json.dumps(payload, indent=2, default=str))
        return

    console.print(f"[bold]Harness Query:[/] {query}")
    console.print(f"[bold]Plan:[/] {payload.get('plan', [])}")
    console.print(f"[bold]Eval:[/] {payload.get('eval_metrics', {})}")
    live_meta = payload.get("metadata", {}).get("live_agent", {})
    if live_meta:
        masked = _mask_secret(model_api_key)
        console.print(
            "[bold]Live Agent:[/] "
            f"enabled={live_meta.get('enabled', False)} "
            f"configured={live_meta.get('configured', False)} "
            f"calls={live_meta.get('calls_used', 0)}/{live_meta.get('call_budget', 0)} "
            f"model={live_meta.get('model', '')}"
            + (f" api_key={masked}" if masked else "")
        )
    console.print(f"[bold]Final Answer:[/]\n{payload.get('final_answer', '')}")


@app.command("harness-live")
def harness_live_command(
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to recipe file"),
    max_steps: int = typer.Option(6, "--max-steps", help="Harness max steps"),
    max_tool_calls: int = typer.Option(6, "--max-tool-calls", help="Harness max tool calls"),
    max_model_calls: int = typer.Option(10, "--max-model-calls", help="Live model call budget per run (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
    json_output: bool = typer.Option(False, "--json", help="Render payload as JSON"),
) -> None:
    """Run harness with real-model agent enhancement enabled."""

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    constraints = HarnessConstraints(
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        enable_live_agent=True,
        max_live_agent_calls=max(1, min(max_model_calls, 50)),
        live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
        live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
    )
    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        constraints=constraints,
        live_model=overrides if overrides else None,
    )
    payload = HARNESS.run_to_dict(run)
    if json_output:
        console.print_json(json.dumps(payload, indent=2, default=str))
        return

    live_meta = payload.get("metadata", {}).get("live_agent", {})
    console.print(f"[bold]Query:[/] {query}")
    console.print(f"[bold]Mode:[/] {_parse_mode(mode).value}")
    console.print(f"[bold]Eval:[/] {payload.get('eval_metrics', {})}")
    console.print(
        "[bold]Live Agent:[/] "
        f"configured={live_meta.get('configured', False)} "
        f"success={live_meta.get('success', False)} "
        f"calls={live_meta.get('calls_used', 0)}/{live_meta.get('call_budget', 0)} "
        f"model={live_meta.get('model', '')} "
        f"api_key={_mask_secret(model_api_key)}"
    )
    if live_meta.get("errors"):
        console.print(f"[yellow]Live Errors:[/] {live_meta.get('errors')}")
    console.print(f"[bold]Final Answer:[/]\n{payload.get('final_answer', '')}")


@app.command("harness-live-experiment")
def harness_live_experiment_command(
    queries_file: str = typer.Option("", "--queries-file", help="Optional text file, one query per line"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    max_total_calls: int = typer.Option(30, "--max-total-calls", help="Total live API call budget (<=50)"),
    max_calls_per_query: int = typer.Option(8, "--max-calls-per-query", help="Live API calls per query (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run baseline vs live-agent A/B experiment with strict call limits."""

    if max_total_calls > 50:
        raise typer.BadParameter("max-total-calls must be <= 50")
    if max_calls_per_query > 50:
        raise typer.BadParameter("max-calls-per-query must be <= 50")

    queries: list[str]
    if queries_file:
        path = Path(queries_file)
        if not path.exists():
            raise typer.BadParameter(f"queries file not found: {queries_file}")
        queries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        queries = _default_live_experiment_queries()

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    constraints = HarnessConstraints(enable_live_agent=True, max_live_agent_calls=max_calls_per_query)
    payload = HARNESS.run_live_experiment(
        queries=queries,
        mode=_parse_mode(mode).value,
        recipe=recipe,
        live_model=overrides if overrides else None,
        max_total_calls=max_total_calls,
        max_calls_per_query=max_calls_per_query,
        constraints=constraints,
    )
    payload.setdefault("model", {}).update(
        {
            "base_url": model_base_url,
            "model_name": model_name,
            "api_key_masked": _mask_secret(model_api_key),
        }
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Live experiment written:[/] {path}")
        return

    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness-live-history")
def harness_live_history_command(
    limit: int = typer.Option(10, "--limit", "-n", help="Max history items"),
) -> None:
    """Show recent live-agent experiment history."""

    payload = {"history": HARNESS.list_live_experiment_history(limit=limit)}
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness-live-config")
def harness_live_config_command(
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
) -> None:
    """Show masked live model configuration (CLI override + env fallback)."""

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=45,
        temperature=0.15,
        max_tokens=1400,
    )
    config = LiveModelConfig.resolve(overrides)
    if config:
        payload = config.masked()
    else:
        payload = {
            "configured": False,
            "note": "No complete model config found in CLI args or env.",
        }
    console.print_json(json.dumps({"live_model": payload}, indent=2, default=str))


@app.command("harness-tools")
def harness_tools_command(
    query: str = typer.Argument("", help="Optional query for dynamic tool discovery"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    limit: int = typer.Option(8, "--limit", "-n", help="Max tools to return"),
) -> None:
    """Inspect harness tool catalog or discover tools for a query."""

    if query:
        discovered = HARNESS.discover_tools(
            query=query,
            mode=_parse_mode(mode).value,
            limit=limit,
        )
        console.print_json(json.dumps({"query": query, "discovered": discovered}, indent=2, default=str))
        return

    console.print_json(json.dumps({"catalog": HARNESS.list_tool_catalog()}, indent=2, default=str))


@app.command("harness-recipes")
def harness_recipes_command() -> None:
    """List built-in harness recipes."""

    console.print_json(json.dumps({"recipes": HARNESS.list_recipes()}, indent=2, default=str))


@app.command("harness-recipe")
def harness_recipe_command(
    query: str = typer.Argument(..., help="Task query"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to JSON/YAML recipe file"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    max_steps: int = typer.Option(6, "--max-steps", help="Harness max steps"),
    max_tool_calls: int = typer.Option(6, "--max-tool-calls", help="Harness max tool calls"),
    json_output: bool = typer.Option(False, "--json", help="Render payload as JSON"),
) -> None:
    """Run harness with a built-in or file-based recipe."""

    run = HARNESS.run_recipe(
        query=query,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        mode=_parse_mode(mode).value,
        constraints=HarnessConstraints(max_steps=max_steps, max_tool_calls=max_tool_calls),
    )
    payload = HARNESS.run_to_dict(run)
    if json_output:
        console.print_json(json.dumps(payload, indent=2, default=str))
        return

    recipe_meta = payload.get("metadata", {}).get("recipe", {})
    console.print(f"[bold]Harness Recipe:[/] {recipe_meta.get('name', '(auto)')}")
    console.print(f"[bold]Plan:[/] {payload.get('plan', [])}")
    console.print(f"[bold]Eval:[/] {payload.get('eval_metrics', {})}")
    console.print(f"[bold]Final Answer:[/]\n{payload.get('final_answer', '')}")


@app.command("harness-redteam")
def harness_redteam_command(
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    strict: bool = typer.Option(True, "--strict/--relaxed", help="Strict security profile"),
    include_runs: bool = typer.Option(False, "--include-runs", help="Include full run payloads"),
) -> None:
    """Run harness red-team suite for safety/reliability checks."""

    constraints = HarnessConstraints(
        max_steps=3,
        max_tool_calls=3,
        allow_write_actions=False,
        allow_network_actions=not strict,
        allow_browser_actions=not strict,
        security_strictness="strict" if strict else "balanced",
    )
    result = HARNESS.run_redteam(
        mode=_parse_mode(mode).value,
        constraints=constraints,
        include_runs=include_runs,
    )
    console.print_json(json.dumps(result, indent=2, default=str))


@app.command("harness-report")
def harness_report_command(
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to recipe file"),
    max_steps: int = typer.Option(6, "--max-steps", help="Harness max steps"),
    max_tool_calls: int = typer.Option(6, "--max-tool-calls", help="Harness max tool calls"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model agent enhancement"),
    max_model_calls: int = typer.Option(8, "--max-model-calls", help="Live model calls per run (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
    report_format: str = typer.Option("markdown", "--format", "-f", help="Report format: markdown|json"),
    output: str = typer.Option("", "--output", "-o", help="Optional output path"),
) -> None:
    """Run harness and render a shareable report."""

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        constraints=HarnessConstraints(
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            enable_live_agent=live_agent,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        ),
        live_model=overrides if overrides else None,
    )
    fmt = report_format.lower().strip()
    if fmt not in {"markdown", "json"}:
        raise typer.BadParameter("format must be markdown or json")

    report = HARNESS.build_report(run, fmt=fmt)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        else:
            output_path.write_text(str(report), encoding="utf-8")
        console.print(f"[green]Report written:[/] {output_path}")
        return

    if fmt == "json":
        console.print_json(json.dumps(report, indent=2, default=str))
    else:
        console.print(str(report))


@app.command("harness-value")
def harness_value_command(
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to recipe file"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model agent enhancement"),
    max_model_calls: int = typer.Option(8, "--max-model-calls", help="Live model calls per run (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
    json_output: bool = typer.Option(False, "--json", help="Render payload as JSON"),
) -> None:
    """Run harness and emit a value card for demo storytelling."""

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        constraints=HarnessConstraints(
            enable_live_agent=live_agent,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        ),
        live_model=overrides if overrides else None,
    )
    card = HARNESS.build_value_card(run)
    card.setdefault("model", {}).update(
        {
            "base_url": model_base_url,
            "model_name": model_name,
            "api_key_masked": _mask_secret(model_api_key),
        }
    )
    if json_output:
        console.print_json(json.dumps(card, indent=2, default=str))
        return

    console.print(f"[bold]Value Index:[/] {card.get('value_index')} ({card.get('band')})")
    console.print(f"[bold]Narrative:[/] {card.get('narrative', '')}")
    for item in card.get("dimensions", []):
        console.print(f"- {item.get('name')}: {round(float(item.get('score', 0.0)) * 100, 1)}%")


@app.command("harness-mission")
def harness_mission_command(
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to recipe file"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model agent enhancement"),
    max_model_calls: int = typer.Option(8, "--max-model-calls", help="Live model calls per run (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run harness and emit the shared mission-pack artifact."""

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        constraints=HarnessConstraints(
            enable_live_agent=live_agent,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        ),
        live_model=overrides if overrides else None,
    )
    payload = HARNESS.build_mission_pack(run)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Mission pack written:[/] {path}")
        return
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-harness-run")
def agent_thread_harness_run_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to recipe file"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run harness inside a persistent generic agent thread."""

    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        thread_id=thread_id,
    )
    payload = {
        "thread": HARNESS.get_thread(thread_id),
        "mission": HARNESS.build_mission_pack(run),
        "report": HARNESS.build_report(run, fmt="json"),
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Thread run written:[/] {path}")
        return
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-exec-mission")
def agent_thread_exec_mission_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    max_nodes: int = typer.Option(0, "--max-nodes", help="Optional execution slice size"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run harness, then execute the mission task graph inside the persistent thread runtime."""

    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        thread_id=thread_id,
    )
    mission = HARNESS.build_mission_pack(run)
    execution = HARNESS.execute_thread_task_graph(
        thread_id,
        mission.get("task_graph", {}),
        execution_label=mission.get("name", "mission"),
        context={"mission": mission, "query": query},
        max_nodes=max(0, max_nodes),
    )
    payload = {
        "thread": HARNESS.get_thread(thread_id),
        "mission": mission,
        "execution": execution,
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Thread execution written:[/] {path}")
        return
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-exec-mission-async")
def agent_thread_exec_mission_async_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    max_nodes: int = typer.Option(0, "--max-nodes", help="Optional execution slice size"),
) -> None:
    """Run harness, then queue the mission task graph for background execution."""

    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        thread_id=thread_id,
    )
    mission = HARNESS.build_mission_pack(run)
    payload = HARNESS.start_thread_task_graph_async(
        thread_id,
        mission.get("task_graph", {}),
        execution_label=mission.get("name", "mission"),
        context={"mission": mission, "query": query},
        max_nodes=max(0, max_nodes),
    )
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-exec-task")
def agent_thread_exec_task_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    query: str = typer.Argument(..., help="General task query"),
    target: str = typer.Option("general", "--target", help="general | code | research | ops"),
    max_nodes: int = typer.Option(0, "--max-nodes", help="Optional execution slice size"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Build and execute a generic executable task graph inside the persistent thread runtime."""

    payload = HARNESS.execute_thread_generic_task(
        thread_id,
        query,
        target=target,
        max_nodes=max(0, max_nodes),
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Thread task execution written:[/] {path}")
        return
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-run")
def agent_thread_run_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    query: str = typer.Argument(..., help="General task query"),
    target: str = typer.Option("auto", "--target", help="auto | general | code | research | ops | benchmark"),
    max_nodes: int = typer.Option(0, "--max-nodes", help="Optional execution slice size"),
    async_mode: bool = typer.Option(False, "--async/--sync", help="Queue in background or execute immediately"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Single thread-first super-agent entrypoint."""

    payload = HARNESS.run_thread_first(
        thread_id,
        query,
        target=target,
        max_nodes=max(0, max_nodes),
        async_mode=async_mode,
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Thread super-agent payload written:[/] {path}")
        return
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("deep-research-report")
def deep_research_report_command(
    topic: str = typer.Argument(..., help="Research topic"),
    subject_root: str = typer.Option(".", "--subject-root", help="Primary repository root"),
    competitor_root: str = typer.Option("", "--competitor-root", help="Optional comparator repository root"),
    subject_name: str = typer.Option("agent-harness", "--subject-name", help="Primary repository label"),
    competitor_name: str = typer.Option("deer-flow", "--competitor-name", help="Comparator repository label"),
    output_dir: str = typer.Option("reports", "--output-dir", "-o", help="Output directory"),
) -> None:
    """Generate a deep research framework/report/html bundle for repository comparison."""

    competitor_value = competitor_root.strip()
    if not competitor_value:
        default_ref = (Path(subject_root).resolve().parent / "deer-flow-ref").resolve()
        if default_ref.exists():
            competitor_value = str(default_ref)
    payload = HARNESS.generate_deep_research_report(
        topic,
        subject_root=subject_root,
        competitor_root=competitor_value or None,
        subject_name=subject_name,
        competitor_name=competitor_name,
        output_dir=output_dir,
    )
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-resume")
def agent_thread_resume_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    execution_id: str = typer.Argument(..., help="Execution id"),
) -> None:
    """Resume a paused or interrupted execution inside a persistent thread."""

    payload = HARNESS.resume_thread_execution(thread_id, execution_id)
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-wait")
def agent_thread_wait_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    execution_id: str = typer.Argument(..., help="Execution id"),
    timeout_seconds: float = typer.Option(30.0, "--timeout", help="Wait timeout seconds"),
) -> None:
    """Wait for a background execution to finish or return its latest state."""

    payload = HARNESS.wait_for_thread_execution(thread_id, execution_id, timeout_seconds=timeout_seconds)
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-recoverables")
def agent_thread_recoverables_command(
    limit: int = typer.Option(50, "--limit", "-n", help="Max executions to list"),
) -> None:
    """List recoverable thread executions."""

    console.print_json(json.dumps({"recoverable": HARNESS.list_recoverable_thread_executions(limit=limit)}, indent=2, default=str))


@app.command("agent-thread-recover")
def agent_thread_recover_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    execution_id: str = typer.Argument(..., help="Execution id"),
    async_mode: bool = typer.Option(True, "--async/--sync", help="Recover in background or foreground"),
) -> None:
    """Recover one incomplete execution."""

    payload = HARNESS.recover_thread_execution(thread_id, execution_id, async_mode=async_mode)
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-recover-all")
def agent_thread_recover_all_command(
    async_mode: bool = typer.Option(True, "--async/--sync", help="Recover in background or foreground"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max recoveries"),
) -> None:
    """Recover all incomplete executions."""

    payload = HARNESS.recover_all_thread_executions(async_mode=async_mode, limit=limit)
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-workspace-view")
def agent_thread_workspace_view_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    html_output: str = typer.Option("", "--html-output", help="Optional HTML output file"),
    json_output: str = typer.Option("", "--json-output", help="Optional JSON output file"),
) -> None:
    """Build workspace stream payload and optional HTML snapshot for one thread."""

    stream_payload = HARNESS.build_thread_workspace_stream(thread_id)
    if json_output:
        path = Path(json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stream_payload, indent=2, default=str), encoding="utf-8")
    if html_output:
        path = Path(html_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HARNESS.render_thread_workspace_html(thread_id), encoding="utf-8")
    if html_output or json_output:
        console.print_json(
            json.dumps(
                {
                    "thread_id": thread_id,
                    "json_output": json_output,
                    "html_output": html_output,
                },
                indent=2,
                default=str,
            )
        )
        return
    console.print_json(json.dumps(stream_payload, indent=2, default=str))


@app.command("agent-thread-export")
def agent_thread_export_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Export a DeerFlow-like thread snapshot contract for frontend use."""

    payload = HARNESS.export_thread_frontend_snapshot(thread_id)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Thread export written:[/] {path}")
        return
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-subagents")
def agent_thread_subagents_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    wait_timeout: float = typer.Option(30.0, "--wait-timeout", help="Wait timeout seconds"),
) -> None:
    """Run parallel subagents derived from the mission execution tracks."""

    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        thread_id=thread_id,
    )
    mission = HARNESS.build_mission_pack(run)
    payload = HARNESS.run_parallel_subagents(
        thread_id,
        _build_subagent_graphs_from_mission(mission),
        wait_timeout_seconds=max(0.1, wait_timeout),
    )
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("agent-thread-retry")
def agent_thread_retry_command(
    thread_id: str = typer.Argument(..., help="Persistent thread id"),
    execution_id: str = typer.Argument(..., help="Execution id"),
    from_node_id: str = typer.Option("", "--from-node", help="Optional node id to restart from"),
) -> None:
    """Retry an execution inside a persistent thread."""

    payload = HARNESS.retry_thread_execution(thread_id, execution_id, from_node_id=from_node_id)
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness-code-pack")
def harness_code_pack_command(
    query: str = typer.Argument(..., help="Implementation or code task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to recipe file"),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Workspace root for code artifact discovery"),
    execute_validation: bool = typer.Option(False, "--execute-validation", help="Run inferred validation commands"),
    validation_timeout: int = typer.Option(180, "--validation-timeout", help="Per-command validation timeout seconds"),
    max_validation_commands: int = typer.Option(3, "--max-validation-commands", help="Max validation commands to execute"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model agent enhancement"),
    max_model_calls: int = typer.Option(8, "--max-model-calls", help="Live model calls per run (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run harness and emit a code mission pack with patch/test/trace artifacts."""

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        constraints=HarnessConstraints(
            enable_live_agent=live_agent,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        ),
        live_model=overrides if overrides else None,
    )
    payload = HARNESS.build_code_mission_pack(
        run,
        workspace=workspace,
        execute_validation=execute_validation,
        validation_timeout_seconds=max(5, min(validation_timeout, 1800)),
        max_validation_commands=max(1, min(max_validation_commands, 8)),
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Code mission pack written:[/] {path}")
        return
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness-visual")
def harness_visual_command(
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to recipe file"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model agent enhancement"),
    max_model_calls: int = typer.Option(8, "--max-model-calls", help="Live model calls per run (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run harness and export front-end ready visualization payload."""

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        constraints=HarnessConstraints(
            enable_live_agent=live_agent,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        ),
        live_model=overrides if overrides else None,
    )
    card = HARNESS.build_value_card(run)
    payload = HARNESS.build_visual_payload(run, value_card=card)
    payload.setdefault("model", {}).update(
        {
            "base_url": model_base_url,
            "model_name": model_name,
            "api_key_masked": _mask_secret(model_api_key),
        }
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Visual payload written:[/] {path}")
        return

    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness-showcase-packs")
def harness_showcase_packs_command() -> None:
    """List built-in showcase packs."""

    console.print_json(json.dumps({"packs": HARNESS.list_showcase_packs()}, indent=2, default=str))


@app.command("harness-evidence-sources")
def harness_evidence_sources_command() -> None:
    """List configured evidence sources backing evidence-aware harness tools."""

    console.print_json(json.dumps({"sources": HARNESS.list_evidence_sources()}, indent=2, default=str))


@app.command("harness-showcase")
def harness_showcase_command(
    pack: str = typer.Option("impact-lens", "--pack", "-p", help="Showcase pack name"),
    mode_override: str = typer.Option("", "--mode-override", help="Override mode for all scenarios"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
    strict: bool = typer.Option(False, "--strict", help="Use strict safety constraints"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model agent in showcase scenarios"),
    max_model_calls: int = typer.Option(6, "--max-model-calls", help="Live model calls per scenario (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
) -> None:
    """Run a scenario pack and export comparative visual payloads."""

    constraints = None
    if strict:
        constraints = HarnessConstraints(
            max_steps=4,
            max_tool_calls=4,
            allow_write_actions=False,
            allow_network_actions=False,
            allow_browser_actions=False,
            security_strictness="strict",
            enable_live_agent=live_agent,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        )
    elif live_agent:
        constraints = HarnessConstraints(
            enable_live_agent=True,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        )

    if mode_override:
        _ = _parse_mode(mode_override)

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    payload = HARNESS.run_showcase(
        pack_name=pack,
        mode_override=mode_override,
        constraints=constraints,
        live_model=overrides if overrides else None,
    )
    payload.setdefault("model", {}).update(
        {
            "base_url": model_base_url,
            "model_name": model_name,
            "api_key_masked": _mask_secret(model_api_key),
        }
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Showcase payload written:[/] {path}")
        return

    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness-blueprint")
def harness_blueprint_command(
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to recipe file"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model agent enhancement"),
    max_model_calls: int = typer.Option(8, "--max-model-calls", help="Live model calls per run (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run harness and export first-screen dashboard blueprint."""

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        constraints=HarnessConstraints(
            enable_live_agent=live_agent,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        ),
        live_model=overrides if overrides else None,
    )
    card = HARNESS.build_value_card(run)
    visual = HARNESS.build_visual_payload(run, value_card=card)
    blueprint = HARNESS.build_first_screen_blueprint(visual)
    blueprint.setdefault("model", {}).update(
        {
            "base_url": model_base_url,
            "model_name": model_name,
            "api_key_masked": _mask_secret(model_api_key),
        }
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blueprint, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Blueprint written:[/] {path}")
        return

    console.print_json(json.dumps(blueprint, indent=2, default=str))


@app.command("harness-stream")
def harness_stream_command(
    query: str = typer.Argument(..., help="Task query"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    recipe: str = typer.Option("", "--recipe", "-r", help="Built-in recipe name"),
    recipe_path: str = typer.Option("", "--recipe-path", help="Path to recipe file"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model agent enhancement"),
    max_model_calls: int = typer.Option(8, "--max-model-calls", help="Live model calls per run (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run harness and export replay event stream."""

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    run = HARNESS.run(
        query=query,
        mode=_parse_mode(mode).value,
        recipe=recipe or None,
        recipe_path=recipe_path or None,
        constraints=HarnessConstraints(
            enable_live_agent=live_agent,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        ),
        live_model=overrides if overrides else None,
    )
    card = HARNESS.build_value_card(run)
    visual = HARNESS.build_visual_payload(run, value_card=card)
    stream = visual.get("event_stream", [])
    payload = {"query": query, "count": len(stream), "events": stream}
    payload.setdefault("model", {}).update(
        {
            "base_url": model_base_url,
            "model_name": model_name,
            "api_key_masked": _mask_secret(model_api_key),
        }
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Event stream written:[/] {path}")
        return

    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness-optimize")
def harness_optimize_command(
    query: str = typer.Argument(..., help="Task query"),
    strict: bool = typer.Option(False, "--strict", help="Use stricter safety constraints"),
    live_agent: bool = typer.Option(False, "--live-agent", help="Enable real-model agent in each candidate"),
    max_model_calls: int = typer.Option(6, "--max-model-calls", help="Live model calls per candidate (<=50)"),
    model_base_url: str = typer.Option("", "--model-base-url", help="Model API base URL"),
    model_api_key: str = typer.Option("", "--model-api-key", help="Model API key"),
    model_name: str = typer.Option("", "--model-name", help="Model name"),
    model_timeout: int = typer.Option(45, "--model-timeout", help="Model timeout seconds"),
    model_temperature: float = typer.Option(0.15, "--model-temperature", help="Model temperature"),
    model_max_tokens: int = typer.Option(1400, "--model-max-tokens", help="Model max tokens"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Auto-tune mode + recipe candidates and return best configuration."""

    overrides = _build_live_model_overrides(
        model_base_url=model_base_url,
        model_api_key=model_api_key,
        model_name=model_name,
        timeout_seconds=model_timeout,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
    )
    constraints = None
    if strict:
        constraints = HarnessConstraints(
            max_steps=4,
            max_tool_calls=4,
            allow_write_actions=False,
            allow_network_actions=False,
            allow_browser_actions=False,
            security_strictness="strict",
            enable_live_agent=live_agent,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        )
    elif live_agent:
        constraints = HarnessConstraints(
            enable_live_agent=True,
            max_live_agent_calls=max(1, min(max_model_calls, 50)),
            live_agent_temperature=max(0.0, min(model_temperature, 1.5)),
            live_agent_timeout_seconds=max(5, min(model_timeout, 300)),
        )
    payload = HARNESS.optimize_query(
        query=query,
        constraints=constraints,
        live_model=overrides if overrides else None,
    )
    payload.setdefault("model", {}).update(
        {
            "base_url": model_base_url,
            "model_name": model_name,
            "api_key_masked": _mask_secret(model_api_key),
        }
    )
    if live_agent:
        payload["note"] = "Optimizer uses real-model enhancement where configured."
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Optimization result written:[/] {path}")
        return

    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness-eval")
def harness_eval_command(
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
) -> None:
    """Run a tiny harness evaluation suite."""

    queries = [
        "Summarize this report and highlight key risks",
        "Compare two options and recommend a safe plan",
        "Audit this compliance proposal and challenge weak assumptions",
    ]
    result = HARNESS.eval_suite(queries=queries, mode=_parse_mode(mode).value)
    console.print_json(json.dumps(result, indent=2, default=str))


@app.command("harness-lab")
def harness_lab_command(
    preset: str = typer.Option("core", "--preset", "-p", help="Preset: core|daily|research|strict|broad"),
    repeats: int = typer.Option(1, "--repeats", "-r", help="Repeat each scenario N times"),
    seed: int = typer.Option(7, "--seed", help="Random seed used for bootstrap CI"),
    scenarios: str = typer.Option(
        "",
        "--scenarios",
        help="Optional comma-separated scenario IDs to run (default: all)",
    ),
    strict: bool = typer.Option(False, "--strict", help="Apply strict safety constraints"),
    include_runs: bool = typer.Option(False, "--include-runs", help="Include full run payloads"),
    isolate_memory: bool = typer.Option(
        True,
        "--isolate-memory/--shared-memory",
        help="Isolate harness memory state for reproducible lab runs",
    ),
    fresh_memory_per_candidate: bool = typer.Option(
        True,
        "--fresh-memory-per-candidate/--carry-memory-between-candidates",
        help="Reset memory before each candidate to avoid cross-candidate contamination",
    ),
    list_scenarios: bool = typer.Option(False, "--list-scenarios", help="List available lab scenarios"),
    list_presets: bool = typer.Option(False, "--list-presets", help="List candidate presets"),
    output: str = typer.Option("", "--output", "-o", help="Optional output JSON file"),
) -> None:
    """Run research-grade reproducible harness experiments."""

    if list_scenarios:
        console.print_json(json.dumps({"scenarios": HARNESS.list_research_scenarios()}, indent=2, default=str))
        return

    if list_presets:
        console.print_json(json.dumps({"presets": HARNESS.list_research_presets()}, indent=2, default=str))
        return

    preset_key = preset.strip().lower()
    allowed = {"core", "daily", "research", "strict", "broad"}
    if preset_key not in allowed:
        raise typer.BadParameter("preset must be one of: core|daily|research|strict|broad")

    scenario_ids = [item.strip() for item in scenarios.split(",") if item.strip()] if scenarios else None

    constraints = None
    if strict:
        constraints = HarnessConstraints(
            max_steps=4,
            max_tool_calls=4,
            allow_write_actions=False,
            allow_network_actions=False,
            allow_browser_actions=False,
            security_strictness="strict",
        )

    payload = HARNESS.run_research_lab(
        preset=preset_key,
        constraints=constraints,
        scenario_ids=scenario_ids,
        repeats=max(1, repeats),
        seed=seed,
        include_runs=include_runs,
        isolate_memory=isolate_memory,
        fresh_memory_per_candidate=fresh_memory_per_candidate,
    )

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Research lab result written:[/] {path}")
        return

    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness-lab-product")
def harness_lab_product_command(
    preset: str = typer.Option("core", "--preset", "-p", help="Preset: core|daily|research|strict|broad"),
    repeats: int = typer.Option(1, "--repeats", "-r", help="Repeat each scenario N times"),
    seed: int = typer.Option(7, "--seed", help="Random seed"),
    scenarios: str = typer.Option(
        "",
        "--scenarios",
        help="Optional comma-separated scenario IDs to run (default: all)",
    ),
    tag: str = typer.Option("", "--tag", help="Optional run tag for output file names"),
    output_dir: str = typer.Option("reports", "--output-dir", help="Output directory for bundle artifacts"),
    strict: bool = typer.Option(False, "--strict", help="Apply strict safety constraints"),
) -> None:
    """Generate productized harness-lab assets: JSON + Markdown + CSV + history."""

    preset_key = preset.strip().lower()
    allowed = {"core", "daily", "research", "strict", "broad"}
    if preset_key not in allowed:
        raise typer.BadParameter("preset must be one of: core|daily|research|strict|broad")

    scenario_ids = [item.strip() for item in scenarios.split(",") if item.strip()] if scenarios else None
    constraints = None
    if strict:
        constraints = HarnessConstraints(
            max_steps=4,
            max_tool_calls=4,
            allow_write_actions=False,
            allow_network_actions=False,
            allow_browser_actions=False,
            security_strictness="strict",
        )

    lab = HARNESS.run_research_lab(
        preset=preset_key,
        constraints=constraints,
        scenario_ids=scenario_ids,
        repeats=max(1, repeats),
        seed=seed,
        include_runs=False,
        isolate_memory=True,
        fresh_memory_per_candidate=True,
    )
    bundle = HARNESS.build_lab_product_bundle(lab_payload=lab, tag=tag)
    paths = HARNESS.write_lab_product_bundle(bundle=bundle, output_dir=output_dir)
    payload = {
        "summary": bundle.get("summary", {}),
        "applause_points": bundle.get("applause_points", []),
        "paths": paths,
    }
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("harness-lab-history")
def harness_lab_history_command(
    limit: int = typer.Option(12, "--limit", "-n", help="Max history items"),
) -> None:
    """Show recent productized harness-lab runs."""

    payload = {"history": HARNESS.list_lab_product_history(limit=max(1, limit))}
    console.print_json(json.dumps(payload, indent=2, default=str))


@app.command("studio-showcase")
def studio_showcase_command(
    query: str = typer.Argument(..., help="Task/query to showcase"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Execution mode"),
    lab_preset: str = typer.Option("broad", "--lab-preset", help="Preset: core|daily|research|strict|broad"),
    lab_repeats: int = typer.Option(1, "--lab-repeats", help="Repeat each scenario N times"),
    scenarios: str = typer.Option(
        "",
        "--scenarios",
        help="Optional comma-separated scenario IDs (default: studio curated set)",
    ),
    output_dir: str = typer.Option("reports/studio", "--output-dir", "-o", help="Output directory"),
    tag: str = typer.Option("", "--tag", help="Optional output tag"),
    export_interop: bool = typer.Option(
        True,
        "--export-interop/--no-export-interop",
        help="Write OpenAI/Anthropic interop bundle",
    ),
    include_marketplace: bool = typer.Option(
        True,
        "--marketplace/--no-marketplace",
        help="Include marketplace skills in interop catalog",
    ),
    include_external: bool = typer.Option(
        True,
        "--external/--no-external",
        help="Include external runtime skills in interop catalog",
    ),
    include_harness_tools: bool = typer.Option(
        True,
        "--harness-tools/--no-harness-tools",
        help="Suggest harness tools in interop skill metadata",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Include full payload in CLI output"),
) -> None:
    """Build flagship showcase artifact that unifies routing, lab, ecosystem, and interop."""

    preset_key = lab_preset.strip().lower()
    allowed = {"core", "daily", "research", "strict", "broad"}
    if preset_key not in allowed:
        raise typer.BadParameter("lab_preset must be one of: core|daily|research|strict|broad")

    parsed_mode = _parse_mode(mode).value
    scenario_ids = [item.strip() for item in scenarios.split(",") if item.strip()] if scenarios else None
    payload = STUDIO.build_showcase(
        query=query,
        mode=parsed_mode,
        lab_preset=preset_key,
        lab_repeats=max(1, lab_repeats),
        scenario_ids=scenario_ids,
        include_marketplace=include_marketplace,
        include_external=include_external,
        include_harness_tools=include_harness_tools,
        include_interop_catalog=export_interop,
    )
    paths = STUDIO.write_showcase(
        payload=payload,
        output_dir=output_dir,
        tag=tag,
        export_interop=export_interop,
    )
    result = {
        "identity": payload.get("identity", {}),
        "query": payload.get("query", {}),
        "frontier": payload.get("frontier", {}),
        "release_decision": payload.get("lab", {}).get("release_decision", {}),
        "positioning": payload.get("comparison", {}).get("positioning", {}),
        "why_use_this": payload.get("why_use_this", []),
        "paths": paths,
    }
    if json_output:
        result["payload"] = payload
    console.print_json(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    app()
