"""Complete demo script showcasing system capabilities."""

from __future__ import annotations

from rich.console import Console

from app.benchmark.evaluate import run_benchmark
from app.coordination.conflicts import ConflictDetector
from app.coordination.consensus import ConsensusBuilder
from app.core.state import GraphState
from app.ecosystem.marketplace import discover_for_query, get_trending_skills
from app.graph import build_graph
from app.policy.center import SystemMode, policy_for_mode
from app.personality.profiles import get_profile
from app.skills.registry import get_skill_card
from app.tracing.visualizer import render_trace_views
from app.utils.display import (
    print_benchmark_results,
    print_conflict_report,
    print_contract,
    print_marketplace_browser,
    print_reasoning_path,
)

console = Console()


def _invoke(query: str, mode: SystemMode = SystemMode.BALANCED) -> dict:
    graph = build_graph()
    result = graph.invoke(GraphState(query=query, system_mode=mode.value, policy=policy_for_mode(mode).to_dict()))
    if isinstance(result, dict):
        return result
    return result.model_dump()


def demo_basic_routing() -> None:
    """Demo 1: basic routing across varied query intents."""

    queries = [
        "Summarize this report and highlight the main risks.",
        "Brainstorm creative ideas for the product launch.",
        "Compare option A vs option B and recommend one.",
        "Investigate the root cause of the system failure.",
    ]

    for query in queries:
        result = _invoke(query)
        console.print(f"[bold]{query}[/]")
        console.print(
            f"  agent={result.get('agent_name')} "
            f"skills={result.get('selected_skills')} "
            f"complexity={result.get('query_complexity')}"
        )


def demo_personality_comparison() -> None:
    """Demo 2: personality profile comparison."""

    query = "Analyze this data and generate insights."
    profiles = ["berserker", "scholar", "explorer", "surgeon"]

    console.print(f"[bold]Query:[/] {query}")
    for name in profiles:
        profile = get_profile(name)
        if not profile:
            continue
        console.print(
            f"- {name}: risk={profile.risk_tolerance:.2f}, "
            f"creativity={profile.creativity_bias:.2f}, "
            f"diversity={profile.diversity_preference:.2f}, "
            f"threshold={profile.confidence_threshold:.2f}"
        )

    result = _invoke(query)
    console.print(f"System selected agent={result.get('agent_name')} skills={result.get('selected_skills')}")


def demo_conflict_resolution() -> None:
    """Demo 3: conflict detection and consensus building."""

    outputs = {
        "risk_assessor": "The situation is high risk and critical.",
        "stability_checker": "The situation is safe and stable with low risk.",
        "summary_writer": "There are mixed signals and uncertainty.",
    }
    conflicts = ConflictDetector().detect(outputs)
    consensus = ConsensusBuilder().build(outputs, conflicts)
    print_conflict_report(conflicts, consensus)


def demo_benchmark() -> None:
    """Demo 4: benchmark strategy comparison."""

    result = run_benchmark()
    print_benchmark_results(result)


def demo_marketplace() -> None:
    """Demo 5: marketplace search and trending."""

    console.print("[bold]Trending Skills[/]")
    print_marketplace_browser(get_trending_skills(limit=3))

    console.print("[bold]Search: risk analysis[/]")
    print_marketplace_browser(discover_for_query("risk analysis", limit=3))


def demo_full_trace() -> None:
    """Demo 6: full reasoning path from query to output."""

    result = _invoke("Summarize this report and highlight the main risks.")
    path = result.get("reasoning_path") or [
        {
            "step": 1,
            "event": "agent_selected",
            "elapsed_ms": 1.0,
            "description": "agent decision",
            "data": {"agent": result.get("agent_name", "")},
        },
        {
            "step": 2,
            "event": "skill_selected",
            "elapsed_ms": 3.0,
            "description": "skill decision",
            "data": {"skills": result.get("selected_skills", [])},
        },
    ]
    print_reasoning_path(path)
    console.print(render_trace_views(result.get("routing_trace", {})))


def demo_skill_card_lifecycle() -> None:
    """Demo 7: inspect a skill card and lifecycle status."""

    card = get_skill_card("identify_risks")
    if not card:
        console.print("Skill card not found.")
        return
    console.print(
        f"[bold]Skill:[/] {card.get('name')}  "
        f"[bold]Owner:[/] {card.get('owner')}  "
        f"[bold]Lifecycle:[/] {card.get('lifecycle_stage')}"
    )
    console.print(f"Failure Modes: {card.get('failure_modes', [])}")
    console.print(f"Recovery Suggestions: {card.get('recovery_suggestions', [])}")


def demo_mode_comparison() -> None:
    """Demo 8: compare routing behavior across modes."""

    query = "Audit this plan, identify risks, challenge assumptions, and provide safe recommendations."
    console.print(f"[bold]Query:[/] {query}")
    for mode in [SystemMode.FAST, SystemMode.BALANCED, SystemMode.DEEP, SystemMode.SAFETY_CRITICAL]:
        result = _invoke(query, mode=mode)
        console.print(
            f"- {mode.value:<15} agent={result.get('agent_name'):<15} "
            f"skills={len(result.get('selected_skills', []))} "
            f"dissent={result.get('disagreement_triggered')}"
        )


def demo_dissent_rescue() -> None:
    """Demo 9: show structured dissent and contract output."""

    query = "Review this high-risk recommendation and find weaknesses or counterarguments."
    result = _invoke(query, mode=SystemMode.SAFETY_CRITICAL)
    print_reasoning_path(result.get("reasoning_path", []))
    print_contract(result.get("response_contract", {}))


def run_all_demos() -> None:
    """Run all demo scenarios in sequence."""

    demos = [
        ("Basic Routing", demo_basic_routing),
        ("Personality Comparison", demo_personality_comparison),
        ("Conflict Resolution", demo_conflict_resolution),
        ("Benchmark", demo_benchmark),
        ("Marketplace", demo_marketplace),
        ("Full Trace", demo_full_trace),
        ("SkillCard Lifecycle", demo_skill_card_lifecycle),
        ("Mode Comparison", demo_mode_comparison),
        ("DISSENT Rescue", demo_dissent_rescue),
    ]

    for title, fn in demos:
        console.rule(f"[bold cyan]{title}[/]")
        fn()
        console.print()


if __name__ == "__main__":
    run_all_demos()
