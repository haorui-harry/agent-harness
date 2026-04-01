"""Example demo entrypoint with multiple capability scenarios."""

from __future__ import annotations

from rich.console import Console

from app.demo import (
    demo_basic_routing,
    demo_conflict_resolution,
    demo_full_trace,
    demo_marketplace,
    demo_personality_comparison,
)

console = Console()


if __name__ == "__main__":
    scenarios = [
        ("Basic Routing", demo_basic_routing),
        ("Personality Comparison", demo_personality_comparison),
        ("Conflict Resolution", demo_conflict_resolution),
        ("Marketplace", demo_marketplace),
        ("Full Trace", demo_full_trace),
    ]
    for title, fn in scenarios:
        console.rule(f"[bold cyan]{title}[/]")
        fn()
        console.print()
