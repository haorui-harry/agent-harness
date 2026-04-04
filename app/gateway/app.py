"""Gateway application assembly."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI
except Exception:  # pragma: no cover - optional dependency
    FastAPI = None

from app.gateway.routers.skills import router as skills_router
from app.gateway.routers.threads import router as threads_router


def build_gateway_app() -> Any:
    """Build the optional FastAPI gateway app if dependencies are installed."""

    if FastAPI is None:
        raise RuntimeError("fastapi is not installed; gateway HTTP app is unavailable in this environment")
    app = FastAPI(title="Agent Harness Gateway", version="0.1.0")
    if skills_router is not None:
        app.include_router(skills_router)
    if threads_router is not None:
        app.include_router(threads_router)
    return app


app = build_gateway_app() if FastAPI is not None else None
