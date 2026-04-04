"""Shared gateway dependencies."""

from __future__ import annotations

from app.harness.engine import HarnessEngine

_HARNESS: HarnessEngine | None = None


def get_harness() -> HarnessEngine:
    """Return the shared harness engine for gateway services."""

    global _HARNESS
    if _HARNESS is None:
        _HARNESS = HarnessEngine()
    return _HARNESS


def set_harness(engine: HarnessEngine) -> None:
    """Override the shared harness engine, primarily for tests."""

    global _HARNESS
    _HARNESS = engine
