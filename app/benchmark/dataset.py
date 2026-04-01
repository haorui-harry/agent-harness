"""Benchmark dataset loader for routing evaluation."""

from __future__ import annotations

import json
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "benchmark_dataset.json"


def ensure_dataset() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return
    seed = {
        "items": [
            {
                "query": "Summarize the report and highlight main risks.",
                "optimal_skills": ["identify_risks", "executive_summary", "extract_facts"],
            },
            {
                "query": "Compare option A and B, and recommend one.",
                "optimal_skills": ["compare_options", "generate_recommendations", "extract_facts"],
            },
            {
                "query": "Create a concise board brief with key decision points.",
                "optimal_skills": ["executive_summary", "board_brief", "generate_recommendations"],
            },
            {
                "query": "Extract evidence and assess uncertainty in claims.",
                "optimal_skills": ["extract_facts", "evidence_matrix", "compare_options"],
            },
        ]
    }
    DATA_FILE.write_text(json.dumps(seed, indent=2), encoding="utf-8")


def load_dataset() -> list[dict]:
    ensure_dataset()
    payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return payload.get("items", [])

