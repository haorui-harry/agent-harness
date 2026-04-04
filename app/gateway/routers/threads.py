"""Gateway thread endpoints and services."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Generator

from pydantic import BaseModel, Field

from app.gateway.deps import get_harness

try:  # pragma: no cover - optional dependency
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import StreamingResponse
except Exception:  # pragma: no cover - optional dependency
    APIRouter = None
    HTTPException = RuntimeError
    StreamingResponse = None


class ThreadCreateRequest(BaseModel):
    title: str = ""
    agent_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThreadResponse(BaseModel):
    thread_id: str
    title: str
    agent_name: str = ""
    status: str = "idle"
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0
    artifact_count: int = 0
    latest_query: str = ""
    latest_summary: str = ""
    workspace: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThreadsListResponse(BaseModel):
    threads: list[ThreadResponse]


class ThreadEventsResponse(BaseModel):
    thread_id: str
    events: list[dict[str, Any]]
    cursor: int = 0


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    executions: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)


def _coerce_thread_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "thread_id": payload.get("thread_id", ""),
        "title": payload.get("title", ""),
        "agent_name": payload.get("agent_name", ""),
        "status": payload.get("status", ""),
        "created_at": payload.get("created_at", ""),
        "updated_at": payload.get("updated_at", ""),
        "message_count": int(payload.get("message_count", 0)),
        "artifact_count": int(payload.get("artifact_count", 0)),
        "latest_query": payload.get("latest_query", ""),
        "latest_summary": payload.get("latest_summary", ""),
        "workspace": payload.get("workspace", {}),
        "metadata": payload.get("metadata", {}),
    }


def create_thread_service(*, title: str = "", agent_name: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    engine = get_harness()
    return _coerce_thread_response(engine.create_thread(title=title, agent_name=agent_name, metadata=metadata))


def list_threads_service(*, limit: int = 20) -> dict[str, Any]:
    engine = get_harness()
    return {"threads": [_coerce_thread_response(item) for item in engine.list_threads(limit=limit)]}


def get_thread_service(thread_id: str) -> dict[str, Any] | None:
    engine = get_harness()
    payload = engine.get_thread(thread_id)
    return _coerce_thread_response(payload) if payload else None


def get_thread_state_service(thread_id: str) -> dict[str, Any]:
    engine = get_harness()
    if engine.get_thread(thread_id) is None:
        raise ValueError(f"unknown thread: {thread_id}")
    return engine.export_thread_frontend_snapshot(thread_id)


def get_thread_history_service(thread_id: str, *, limit: int = 50) -> dict[str, Any]:
    engine = get_harness()
    payload = engine.get_thread(thread_id)
    if not payload:
        raise ValueError(f"unknown thread: {thread_id}")
    return {
        "thread_id": thread_id,
        "messages": list(payload.get("messages", []))[-limit:],
        "artifacts": list(payload.get("artifacts", []))[-limit:],
        "executions": list(payload.get("executions", []))[-limit:],
        "events": list(payload.get("events", []))[-limit:],
    }


def list_thread_events_service(thread_id: str, *, after: int = 0, limit: int = 100) -> dict[str, Any]:
    engine = get_harness()
    if engine.get_thread(thread_id) is None:
        raise ValueError(f"unknown thread: {thread_id}")
    events = engine.thread_runtime.list_events(thread_id, after=after, limit=limit)
    cursor = max((int(item.get("event_id", 0)) for item in events), default=max(0, int(after)))
    return {
        "thread_id": thread_id,
        "events": events,
        "cursor": cursor,
    }


def format_sse_event(payload: dict[str, Any], *, event: str = "thread_event") -> str:
    return f"id: {payload.get('cursor', int(time.time()))}\nevent: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def stream_thread_events_service(
    thread_id: str,
    *,
    after: int = 0,
    limit: int = 100,
    timeout_seconds: float = 15.0,
    heartbeat_seconds: float = 5.0,
    max_batches: int = 0,
) -> Generator[str, None, None]:
    engine = get_harness()
    if engine.get_thread(thread_id) is None:
        raise ValueError(f"unknown thread: {thread_id}")
    cursor = max(0, int(after))
    batches = 0
    last_heartbeat = time.time()
    while True:
        events = engine.thread_runtime.wait_for_events(
            thread_id,
            after=cursor,
            limit=limit,
            timeout_seconds=timeout_seconds,
        )
        if events:
            cursor = max(cursor, max(int(item.get("event_id", 0)) for item in events))
            payload = {"thread_id": thread_id, "events": events, "cursor": cursor}
            yield format_sse_event(payload, event="thread_events")
            batches += 1
            if max_batches and batches >= max_batches:
                break
            continue
        if time.time() - last_heartbeat >= max(1.0, heartbeat_seconds):
            heartbeat = {
                "thread_id": thread_id,
                "events": [],
                "cursor": cursor,
                "heartbeat_id": uuid.uuid4().hex[:8],
            }
            yield format_sse_event(heartbeat, event="heartbeat")
            last_heartbeat = time.time()
            batches += 1
            if max_batches and batches >= max_batches:
                break


router = APIRouter(prefix="/api/threads", tags=["threads"]) if APIRouter else None

if router is not None:  # pragma: no cover - thin FastAPI wrapper

    @router.get("", response_model=ThreadsListResponse)
    async def list_threads(limit: int = 20) -> ThreadsListResponse:
        return ThreadsListResponse(**list_threads_service(limit=limit))


    @router.post("", response_model=ThreadResponse)
    async def create_thread(body: ThreadCreateRequest) -> ThreadResponse:
        return ThreadResponse(**create_thread_service(title=body.title, agent_name=body.agent_name, metadata=body.metadata))


    @router.get("/{thread_id}", response_model=ThreadResponse)
    async def get_thread(thread_id: str) -> ThreadResponse:
        payload = get_thread_service(thread_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
        return ThreadResponse(**payload)


    @router.get("/{thread_id}/state")
    async def get_thread_state(thread_id: str) -> dict[str, Any]:
        try:
            return get_thread_state_service(thread_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.get("/{thread_id}/history", response_model=ThreadHistoryResponse)
    async def get_thread_history(thread_id: str, limit: int = 50) -> ThreadHistoryResponse:
        try:
            return ThreadHistoryResponse(**get_thread_history_service(thread_id, limit=limit))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.get("/{thread_id}/events", response_model=ThreadEventsResponse)
    async def get_thread_events(thread_id: str, after: int = 0, limit: int = 100) -> ThreadEventsResponse:
        try:
            return ThreadEventsResponse(**list_thread_events_service(thread_id, after=after, limit=limit))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.get("/{thread_id}/stream")
    async def stream_thread_events(thread_id: str, after: int = 0, limit: int = 100, timeout_seconds: float = 15.0):
        if StreamingResponse is None:
            raise HTTPException(status_code=500, detail="StreamingResponse is unavailable")
        try:
            return StreamingResponse(
                stream_thread_events_service(thread_id, after=after, limit=limit, timeout_seconds=timeout_seconds),
                media_type="text/event-stream",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
