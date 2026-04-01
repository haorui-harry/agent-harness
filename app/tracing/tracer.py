"""Reasoning tracer for recording routing decision events."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class TraceEventType(str, Enum):
    QUERY_RECEIVED = "query_received"
    COMPLEXITY_ESTIMATED = "complexity_estimated"
    AGENT_CANDIDATES_SCORED = "agent_candidates_scored"
    AGENT_SELECTED = "agent_selected"
    COLLABORATION_DETECTED = "collaboration_detected"
    PERSONALITY_LOADED = "personality_loaded"
    PERSONALITY_ADAPTED = "personality_adapted"
    SKILL_CANDIDATES_BUILT = "skill_candidates_built"
    SKILL_RELEVANCE_SCORED = "skill_relevance_scored"
    PAIRWISE_SIMILARITY_COMPUTED = "pairwise_similarity_computed"
    SYNERGY_MATRIX_COMPUTED = "synergy_matrix_computed"
    SKILL_SELECTED = "skill_selected"
    SKILL_REJECTED = "skill_rejected"
    STYLE_ADJUSTMENT_APPLIED = "style_adjustment_applied"
    REFINEMENT_SWAP = "refinement_swap"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_RETRIED = "execution_retried"
    CONFLICT_DETECTED = "conflict_detected"
    CONSENSUS_BUILT = "consensus_built"
    ENSEMBLE_SYNTHESIZED = "ensemble_synthesized"
    METRICS_COMPUTED = "metrics_computed"


@dataclass
class TraceEvent:
    """Single event in reasoning trace."""

    event_type: TraceEventType
    timestamp: float
    data: dict = field(default_factory=dict)
    description: str = ""

    @property
    def elapsed_ms(self) -> float:
        return 0.0


class ReasoningTracer:
    """Collect and expose reasoning events for one query run."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._start_time: float = 0.0
        self._query: str = ""

    def start(self, query: str) -> None:
        """Start tracking a new query."""

        self._start_time = time.time()
        self._query = query
        self.record(TraceEventType.QUERY_RECEIVED, {"query": query})

    def record(
        self,
        event_type: TraceEventType,
        data: dict | None = None,
        description: str = "",
    ) -> None:
        """Record a trace event."""

        event = TraceEvent(
            event_type=event_type,
            timestamp=time.time(),
            data=data or {},
            description=description or event_type.value,
        )
        self.events.append(event)

    def get_path(self) -> list[dict]:
        """Return full path as serializable list."""

        path: list[dict] = []
        for event in self.events:
            elapsed = (event.timestamp - self._start_time) * 1000
            path.append(
                {
                    "step": len(path) + 1,
                    "event": event.event_type.value,
                    "elapsed_ms": round(elapsed, 2),
                    "description": event.description,
                    "data": event.data,
                }
            )
        return path

    def get_summary(self) -> dict:
        """Return trace summary metrics."""

        total_ms = 0.0
        if self.events:
            total_ms = (self.events[-1].timestamp - self._start_time) * 1000

        counts: dict[str, int] = {}
        for event in self.events:
            key = event.event_type.value
            counts[key] = counts.get(key, 0) + 1

        return {
            "query": self._query,
            "total_events": len(self.events),
            "total_time_ms": round(total_ms, 2),
            "event_counts": counts,
        }

    def get_decision_points(self) -> list[dict]:
        """Return only decision-related events."""

        decision_types = {
            TraceEventType.AGENT_SELECTED,
            TraceEventType.SKILL_SELECTED,
            TraceEventType.SKILL_REJECTED,
            TraceEventType.REFINEMENT_SWAP,
            TraceEventType.CONFLICT_DETECTED,
        }
        return [
            {
                "event": event.event_type.value,
                "description": event.description,
                "data": event.data,
            }
            for event in self.events
            if event.event_type in decision_types
        ]
