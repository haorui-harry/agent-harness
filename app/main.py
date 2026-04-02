"""CLI entry point for the LangGraph Skill Router."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from app.benchmark.evaluate import run_benchmark
from app.core.state import AgentStyle, GraphState
from app.demo import (
    demo_benchmark,
    demo_conflict_resolution,
    demo_full_trace,
    demo_marketplace,
    demo_personality_comparison,
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
from app.tracing.analyzer import RoutingAnalyzer
from app.tracing.store import list_recent_traces, load_trace, save_trace
from app.tracing.visualizer import render_trace_views
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
    scenario: str = typer.Argument("all", help="Scenario: all, personality, conflict, benchmark, marketplace, trace"),
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

    raise typer.BadParameter("Scenario must be one of: all, personality, conflict, benchmark, marketplace, trace")


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
    config = LiveModelConfig.from_overrides(overrides)
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


if __name__ == "__main__":
    app()
