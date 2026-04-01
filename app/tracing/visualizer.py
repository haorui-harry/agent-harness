"""ASCII visualization helpers for reasoning traces."""

from __future__ import annotations


def render_reasoning_tree(path: list[dict], max_width: int = 80) -> str:
    """Render reasoning path into a readable ASCII tree."""

    lines: list[str] = []
    for index, step in enumerate(path):
        elapsed = step.get("elapsed_ms", 0.0)
        event = step.get("event", "")
        data = step.get("data", {})

        lines.append(f"[{elapsed:>7.1f}ms] {_format_event_name(event)}")

        is_last_step = index == len(path) - 1
        branch_prefix = "└── " if is_last_step else "├── "

        if data:
            items = list(data.items())
            for item_index, (key, value) in enumerate(items):
                is_last_item = is_last_step and item_index == len(items) - 1
                prefix = "└── " if is_last_item else branch_prefix
                lines.append(f"{prefix}{_format_value(key, value, max_width - len(prefix))}")

        if not is_last_step:
            lines.append("│")

    return "\n".join(lines)


def render_decision_summary(decisions: list[dict]) -> str:
    """Render compact decision summary text."""

    lines = ["Decision Summary", "=" * 40]
    for index, decision in enumerate(decisions, 1):
        lines.append(f"{index}. [{decision['event']}] {decision['description']}")
        for key, value in decision.get("data", {}).items():
            lines.append(f"   {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def render_routing_sankey(trace: dict) -> str:
    """Render a lightweight sankey-style routing path."""

    agent_candidates = trace.get("agent_candidates", [])
    selected_agent = trace.get("agent_decision", {}).get("selected", ["-"])[0]
    skill_candidates = trace.get("skill_candidates", trace.get("skill_decision", {}).get("skill_candidates", []))
    selected_skills = trace.get("skill_decision", {}).get("selected", [])

    lines = ["Routing Sankey", "=" * 40]
    lines.append("Query")
    lines.append("  -> Agent Candidates: " + (", ".join(agent_candidates[:6]) if agent_candidates else "-"))
    lines.append(f"  -> Selected Agent: {selected_agent}")
    lines.append("  -> Skill Candidates: " + (", ".join(skill_candidates[:8]) if skill_candidates else "-"))
    lines.append("  -> Selected Skills: " + (", ".join(selected_skills) if selected_skills else "-"))
    return "\n".join(lines)


def render_skill_matrix(trace: dict) -> str:
    """Render compact complementarity matrix view."""

    matrix = trace.get("complementarity_matrix", {})
    if not matrix:
        return "Skill Matrix\n========================================\n(no matrix data)"

    lines = ["Skill Matrix", "=" * 40]
    top_items = sorted(matrix.items(), key=lambda item: item[1], reverse=True)[:12]
    for pair, score in top_items:
        lines.append(f"{pair}: {score:.3f}")
    return "\n".join(lines)


def render_execution_gantt(trace: dict) -> str:
    """Render simple Gantt-like execution timeline."""

    timeline = trace.get("execution_timeline", [])
    if not timeline:
        return "Execution Gantt\n========================================\n(no execution timeline)"

    max_ms = max(float(item.get("duration_ms", 0.0)) for item in timeline) or 1.0
    lines = ["Execution Gantt", "=" * 40]
    for item in timeline:
        name = item.get("skill", "-")
        duration = float(item.get("duration_ms", 0.0))
        slots = max(1, int(round((duration / max_ms) * 20)))
        bar = "#" * slots + "-" * max(0, 20 - slots)
        lines.append(f"{name:<22} |{bar}| {duration:>7.1f}ms")
    return "\n".join(lines)


def render_confidence_waterfall(trace: dict) -> str:
    """Render confidence component breakdown view."""

    components = trace.get("final_confidence_breakdown", {})
    if not components:
        return "Confidence Breakdown\n========================================\n(no confidence components)"

    lines = ["Confidence Breakdown", "=" * 40]
    for key, value in components.items():
        if not isinstance(value, (int, float)):
            continue
        slots = max(0, min(20, int(round(float(value) * 20))))
        bar = "#" * slots + "-" * (20 - slots)
        lines.append(f"{key:<30} {float(value):>5.2f}  |{bar}|")
    return "\n".join(lines)


def render_trace_views(trace: dict) -> str:
    """Render all core trace views in one payload."""

    sections = [
        render_routing_sankey(trace),
        render_skill_matrix(trace),
        render_execution_gantt(trace),
        render_confidence_waterfall(trace),
    ]
    return "\n\n".join(sections)


def _format_event_name(event: str) -> str:
    return event.replace("_", " ").title()


def _format_value(key: str, value: object, max_len: int) -> str:
    if isinstance(value, float):
        formatted = f"{key}: {value:.3f}"
    elif isinstance(value, list) and len(value) <= 5:
        formatted = f"{key}: [{', '.join(str(v) for v in value)}]"
    elif isinstance(value, dict):
        preview = ", ".join(f"{k}={v}" for k, v in list(value.items())[:3])
        formatted = f"{key}: {{{preview}}}"
    else:
        formatted = f"{key}: {value}"
    return formatted[:max_len]
