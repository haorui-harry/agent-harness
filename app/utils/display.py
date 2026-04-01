"""Rich terminal display helpers for routing traces and analytics."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from app.tracing.visualizer import render_reasoning_tree

console = Console()


def print_routing_trace(trace: dict[str, Any], query: str) -> None:
    """Pretty-print a full routing trace to the terminal."""

    console.print()
    console.print(Panel(f"[bold cyan]{query}[/]", title="Query", border_style="cyan"))

    agent = trace.get("agent_decision", {})
    if agent:
        tree = Tree("[bold yellow]Agent Router[/]")
        selected = agent.get("selected", [])
        rejected = agent.get("rejected", [])
        reasons = agent.get("reasons", {})
        signals = agent.get("intent_signals", {})

        if signals:
            branch = tree.add("[dim]Intent Signals[/]")
            for intent, score in sorted(signals.items(), key=lambda item: -item[1]):
                branch.add(f"{intent}: {score:.2f}")

        complexity = agent.get("query_complexity")
        if complexity:
            tree.add(f"[dim]Complexity:[/] {complexity}")

        collaboration = agent.get("collaboration", {})
        if collaboration:
            tree.add(
                f"[dim]Collaboration:[/] agents={collaboration.get('agents', [])}, "
                f"reason={collaboration.get('reason', '')}"
            )

        for name in selected:
            tree.add(f"[bold green]-> {name}[/]  {reasons.get(name, '')}")
        for name in rejected:
            tree.add(f"[dim red]   {name}[/]  {reasons.get(name, '')}")

        console.print(tree)

    skill = trace.get("skill_decision", {})
    if skill:
        tree = Tree("[bold magenta]Skill Router[/]")
        selected = skill.get("selected", [])
        rejected = skill.get("rejected", [])
        reasons = skill.get("reasons", {})

        for name in selected:
            tree.add(f"[bold green]-> {name}[/]  {reasons.get(name, '')}")
        for name in rejected:
            tree.add(f"[dim red]   {name}[/]  {reasons.get(name, '')}")

        metrics = skill.get("complementarity_metrics", {})
        if metrics:
            branch = tree.add("[dim]Complementarity Metrics[/]")
            for key in [
                "coverage",
                "redundancy",
                "diversity_shannon",
                "diversity_simpson",
                "ensemble_coherence",
                "total_synergy",
                "total_budget_used",
            ]:
                if key in metrics:
                    branch.add(f"{key}: {metrics.get(key):.3f}")

        discovery = skill.get("marketplace_discovery", [])
        if discovery:
            branch = tree.add("[dim]Marketplace Discovery[/]")
            for item in discovery:
                branch.add(
                    f"{item.get('name')} ({item.get('provider')}) "
                    f"score={item.get('score')} rep={item.get('reputation')}"
                )

        required_slots = skill.get("required_role_slots", [])
        if required_slots:
            tree.add(f"[dim]Required role slots:[/] {', '.join(required_slots)}")

        console.print(tree)

    confidence = trace.get("final_confidence_breakdown", {})
    if confidence:
        table = Table(title="Confidence Breakdown")
        table.add_column("Component")
        table.add_column("Value", justify="right")
        for key, value in confidence.items():
            if isinstance(value, (int, float)):
                table.add_row(key, f"{float(value):.3f}")
        console.print(table)

    cost = trace.get("cost_breakdown", {})
    latency = trace.get("latency_breakdown", {})
    if cost or latency:
        table = Table(title="Cost / Latency Profile")
        table.add_column("Skill")
        table.add_column("Cost", justify="right")
        table.add_column("Latency (ms)", justify="right")
        keys = sorted(set(cost.keys()) | set(latency.keys()))
        for key in keys:
            cost_value = float(cost.get(key, 0.0))
            latency_value = float(latency.get(key, 0.0))
            table.add_row(key, f"{cost_value:.2f}", f"{latency_value:.1f}")
        console.print(table)

    console.print()


def print_final_output(output: str) -> None:
    """Print final aggregated output."""

    console.print(Panel(output, title="Output", border_style="green"))


def print_personality(personality: dict[str, float], agent_name: str) -> None:
    """Print personality dimensions as ASCII bar chart."""

    if not personality:
        console.print("[dim]No personality data available.[/]")
        return

    def bar(value: float, width: int = 10) -> str:
        filled = max(0, min(width, int(round(value * width))))
        return "#" * filled + "-" * (width - filled)

    table = Table(title=f"Agent Personality: {agent_name}")
    table.add_column("Dimension")
    table.add_column("Profile")
    table.add_column("Value", justify="right")

    mapping = [
        ("risk_tolerance", "Risk Tolerance"),
        ("creativity_bias", "Creativity Bias"),
        ("diversity_preference", "Diversity Pref"),
        ("confidence_threshold", "Confidence Thr"),
        ("collaboration_tendency", "Collaboration"),
        ("depth_vs_breadth", "Depth vs Breadth"),
    ]

    for key, label in mapping:
        value = float(personality.get(key, 0.0))
        table.add_row(label, bar(value), f"{value:.2f}")

    console.print(table)


def print_conflict_report(conflicts: list[dict], consensus: dict) -> None:
    """Print conflict and consensus summary."""

    if not conflicts:
        console.print(Panel("No conflicts detected.", title="Conflicts", border_style="green"))
    else:
        table = Table(title="Conflicts Detected")
        table.add_column("Type")
        table.add_column("Skill A")
        table.add_column("Skill B")
        table.add_column("Signals")
        table.add_column("Severity", justify="right")
        for conflict in conflicts:
            table.add_row(
                str(conflict.get("type", "")),
                str(conflict.get("skill_a", "")),
                str(conflict.get("skill_b", "")),
                f"{conflict.get('signal_a', '')} | {conflict.get('signal_b', '')}",
                f"{float(conflict.get('severity', 0.0)):.2f}",
            )
        console.print(table)

    if consensus:
        console.print(
            Panel(
                "\n".join(
                    [
                        f"strength: {consensus.get('strength', 'unknown')}",
                        f"agreement_ratio: {consensus.get('agreement_ratio', 0)}",
                        f"shared_themes: {', '.join(consensus.get('shared_themes', [])[:8])}",
                    ]
                ),
                title="Consensus",
                border_style="cyan",
            )
        )


def print_reasoning_path(path: list[dict]) -> None:
    """Print reasoning path in ASCII tree format."""

    console.print(Panel(render_reasoning_tree(path), title="Reasoning Path", border_style="blue"))


def print_benchmark_results(results: dict) -> None:
    """Print benchmark result table with strategy metrics."""

    strategies = results.get("strategies", {})
    if not strategies:
        console.print("[dim]No benchmark strategy results available.[/]")
        return

    table = Table(title="Benchmark Results")
    table.add_column("Strategy")
    table.add_column("Jaccard", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    best_name = None
    best_f1 = -1.0
    for name, metrics in strategies.items():
        f1 = float(metrics.get("f1", 0.0))
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    for name, metrics in strategies.items():
        style = "bold green" if name == best_name else ""
        table.add_row(
            f"[{style}]{name}[/{style}]" if style else name,
            f"{float(metrics.get('jaccard', 0.0)):.3f}",
            f"{float(metrics.get('precision', 0.0)):.3f}",
            f"{float(metrics.get('recall', 0.0)):.3f}",
            f"{float(metrics.get('f1', 0.0)):.3f}",
        )

    console.print(table)


def print_marketplace_browser(skills: list[dict]) -> None:
    """Print marketplace skills as compact panels."""

    if not skills:
        console.print("[dim]No marketplace skills found.[/]")
        return

    for skill in skills:
        body = "\n".join(
            [
                f"Provider: {skill.get('provider', '-')}",
                f"Version: {skill.get('version', '-')}",
                f"Rating: {skill.get('rating', skill.get('reputation', '-'))}",
                f"Installs: {skill.get('installs', '-')}",
                f"Tags: {', '.join(skill.get('tags', []))}",
            ]
        )
        console.print(Panel(body, title=skill.get("name", "unknown"), border_style="magenta"))


def print_execution_timeline(contexts: dict[str, dict]) -> None:
    """Print execution timeline in a simple Gantt-like bar chart."""

    if not contexts:
        console.print("[dim]No execution contexts found.[/]")
        return

    console.print("Execution Timeline:")
    max_duration = max(float(ctx.get("duration_ms", 0.0)) for ctx in contexts.values())
    max_duration = max(max_duration, 1.0)

    for name, ctx in contexts.items():
        duration = float(ctx.get("duration_ms", 0.0))
        success = bool(ctx.get("success", False))
        slots = int(round((duration / max_duration) * 20))
        bar = "#" * max(1, slots) + "-" * max(0, 20 - max(1, slots))
        status = "OK" if success else "ERR"
        console.print(f"{name:<20} |{bar}| {duration:>6.1f}ms  {status}")


def print_routing_quality(metrics: dict[str, float]) -> None:
    """Print routing quality dashboard table."""

    if not metrics:
        console.print("[dim]No routing metrics available.[/]")
        return

    table = Table(title="Routing Quality")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, float):
            table.add_row(key, f"{value:.3f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


def print_contract(contract: dict[str, Any]) -> None:
    """Print structured response contract sections."""

    if not contract:
        console.print("[dim]No response contract available.[/]")
        return

    user = contract.get("user", {})
    debug = contract.get("debug", {})
    evaluation = contract.get("evaluation", {})

    user_panel = Panel(
        "\n".join(
            [
                f"overall confidence: {user.get('confidence_summary', {}).get('overall', '-')}",
                f"requires review: {user.get('confidence_summary', {}).get('requires_human_review', '-')}",
                f"trace_id: {user.get('trace_summary', {}).get('trace_id', '-')}",
                f"findings: {len(user.get('key_risks_or_findings', []))}",
            ]
        ),
        title="Response Contract / User",
        border_style="cyan",
    )
    console.print(user_panel)

    debug_table = Table(title="Response Contract / Debug")
    debug_table.add_column("Field")
    debug_table.add_column("Value")
    debug_table.add_row("selected_agent", str(debug.get("selected_agent", "")))
    debug_table.add_row("selected_skills", ", ".join(debug.get("selected_skills", [])))
    debug_table.add_row("rejected_skills", ", ".join(debug.get("rejected_skills", [])))
    debug_table.add_row("full_trace_id", str(debug.get("full_trace_id", "")))
    console.print(debug_table)

    eval_table = Table(title="Response Contract / Evaluation")
    eval_table.add_column("Metric")
    eval_table.add_column("Value", justify="right")
    for key in [
        "route_regret_estimate",
        "coverage_score",
        "redundancy_score",
        "disagreement_triggered",
        "approval_required",
    ]:
        value = evaluation.get(key, "-")
        if isinstance(value, float):
            eval_table.add_row(key, f"{value:.3f}")
        else:
            eval_table.add_row(key, str(value))
    console.print(eval_table)
