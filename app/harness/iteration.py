"""Iteration tracker for live-agent experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "live_experiment_log.json"


class LiveIterationTracker:
    """Persist experiment summaries to support iterative optimization."""

    def __init__(self, file_path: Path | None = None) -> None:
        self._file = file_path or DATA_FILE
        self._ensure()

    def _ensure(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        if not self._file.exists():
            self._file.write_text(json.dumps({"experiments": []}, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        self._ensure()
        return json.loads(self._file.read_text(encoding="utf-8"))

    def _save(self, payload: dict[str, Any]) -> None:
        self._file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def record(self, payload: dict[str, Any], model_info: dict[str, Any] | None = None) -> dict[str, Any]:
        data = self._load()
        experiments = data.setdefault("experiments", [])
        item = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "summary": payload.get("summary", {}),
            "config": payload.get("config", {}),
            "count": payload.get("count", 0),
            "model": model_info or {},
        }
        experiments.append(item)
        self._save(data)
        return item

    def latest(self, limit: int = 10) -> list[dict[str, Any]]:
        payload = self._load()
        items = payload.get("experiments", [])
        if not isinstance(items, list):
            return []
        return items[-max(limit, 1) :]

