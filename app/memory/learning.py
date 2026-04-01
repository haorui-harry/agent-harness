"""Lightweight online learning store for skill and agent performance."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.state import GraphState

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "learning_stats.json"


def _ensure_file() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        DATA_FILE.write_text(
            json.dumps({"skills": {}, "agents": {}, "pairs": {}}, indent=2),
            encoding="utf-8",
        )


def _load() -> dict:
    _ensure_file()
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def _save(payload: dict) -> None:
    DATA_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_run(state: GraphState) -> None:
    """Update simple success/reliability counters from one run."""

    payload = _load()

    agent_key = state.agent_name or "unknown"
    agent_stats = payload["agents"].setdefault(agent_key, {"runs": 0, "success": 0})
    agent_stats["runs"] += 1

    success_count = 0
    for name, ctx in state.execution_contexts.items():
        stat = payload["skills"].setdefault(name, {"runs": 0, "success": 0, "avg_quality": 0.0})
        stat["runs"] += 1
        if ctx.get("success"):
            stat["success"] += 1
            success_count += 1

        quality = float(ctx.get("quality_score", 0.0))
        prev_avg = float(stat.get("avg_quality", 0.0))
        stat["avg_quality"] = prev_avg + (quality - prev_avg) / max(stat["runs"], 1)

    if state.execution_contexts and success_count == len(state.execution_contexts):
        agent_stats["success"] += 1

    selected = list(state.selected_skills)
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            key = f"{selected[i]}|{selected[j]}"
            pair = payload["pairs"].setdefault(key, {"runs": 0, "success": 0})
            pair["runs"] += 1
            if state.consensus_result.get("strength") in {"strong", "moderate"}:
                pair["success"] += 1

    _save(payload)


def get_skill_reliability(skill_name: str) -> float:
    """Return empirical skill reliability in [0,1]."""

    payload = _load()
    stat = payload["skills"].get(skill_name)
    if not stat:
        return 0.5
    return float(stat.get("success", 0)) / max(float(stat.get("runs", 0)), 1.0)
