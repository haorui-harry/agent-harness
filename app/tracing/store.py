"""Trace persistence utilities for replay and audit."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

TRACE_DIR = Path(__file__).resolve().parents[2] / "data" / "traces"


def _ensure() -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)


def save_trace(payload: dict[str, Any]) -> str:
    """Persist one execution payload and return trace_id."""

    _ensure()
    trace_id = payload.get("trace_id") or uuid.uuid4().hex[:16]
    payload = dict(payload)
    payload["trace_id"] = trace_id
    path = TRACE_DIR / f"{trace_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return trace_id


def load_trace(trace_id: str) -> dict[str, Any] | None:
    """Load a persisted trace by id."""

    _ensure()
    path = TRACE_DIR / f"{trace_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_recent_traces(limit: int = 10) -> list[dict[str, Any]]:
    """List metadata for most recent traces."""

    _ensure()
    files = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append(
            {
                "trace_id": payload.get("trace_id", path.stem),
                "query": payload.get("query", ""),
                "agent": payload.get("agent_name", ""),
                "skills": payload.get("selected_skills", []),
                "mode": payload.get("system_mode", "balanced"),
            }
        )
    return out
