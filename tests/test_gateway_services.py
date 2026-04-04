"""Tests for gateway service contracts without requiring FastAPI runtime."""

from __future__ import annotations

import zipfile
from pathlib import Path

from app.agents.runtime import AgentThreadRuntime
from app.gateway.deps import set_harness
from app.gateway.routers.skills import (
    get_skill_service,
    install_skill_service,
    list_skills_service,
    update_skill_service,
)
from app.gateway.routers.threads import (
    create_thread_service,
    format_sse_event,
    get_thread_history_service,
    get_thread_service,
    get_thread_state_service,
    list_thread_events_service,
    list_threads_service,
    stream_thread_events_service,
)
from app.harness.engine import HarnessEngine
from app.harness.runtime_settings import HarnessRuntimeSettings
from app.skills.manager import SkillPackageManager


def _build_engine(tmp_path: Path) -> HarnessEngine:
    settings = HarnessRuntimeSettings(
        threads_root=tmp_path / "threads",
        memory_path=tmp_path / "memory.json",
    )
    engine = HarnessEngine(settings=settings)
    engine.skill_manager = SkillPackageManager(
        skills_root=tmp_path / "skills",
        state_file=tmp_path / "skill_packages.json",
    )
    engine.skill_packages = engine.skill_packages.__class__(
        skills_root=tmp_path / "skills",
        state_file=tmp_path / "skill_packages.json",
    )
    set_harness(engine)
    return engine


def test_gateway_skills_services_support_list_update_and_install(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path)
    public = tmp_path / "skills" / "public" / "demo"
    public.mkdir(parents=True)
    (public / "SKILL.md").write_text("---\nname: demo\ndescription: demo skill\n---\n# Demo\n", encoding="utf-8")
    engine.skill_packages = engine.skill_packages.__class__(
        skills_root=tmp_path / "skills",
        state_file=tmp_path / "skill_packages.json",
    )

    listed = list_skills_service()
    assert any(item["name"] == "demo" for item in listed["skills"])

    updated = update_skill_service("demo", enabled=False)
    assert updated["enabled"] is False
    assert get_skill_service("demo") is not None

    archive = tmp_path / "new.skill"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("new-skill/SKILL.md", "---\nname: new-skill\ndescription: installed\n---\n# New\n")
    installed = install_skill_service(str(archive))
    assert installed["success"] is True
    assert get_skill_service("new-skill") is not None


def test_gateway_threads_services_support_export_history_and_event_stream(tmp_path: Path) -> None:
    engine = _build_engine(tmp_path)
    engine.thread_runtime = AgentThreadRuntime(tmp_path / "threads")
    set_harness(engine)

    thread = create_thread_service(title="Gateway Thread")
    thread_id = thread["thread_id"]
    engine.thread_runtime.append_message(thread_id, "user", "hello")
    engine.thread_runtime.append_message(thread_id, "assistant", "world")
    engine.thread_runtime.append_event(thread_id, {"event": "task_started", "task_name": "demo", "task_kind": "subagent"})

    listed = list_threads_service()
    assert any(item["thread_id"] == thread_id for item in listed["threads"])

    current = get_thread_service(thread_id)
    assert current is not None
    assert current["title"] == "Gateway Thread"

    state = get_thread_state_service(thread_id)
    assert state["values"]["title"] == "Gateway Thread"

    history = get_thread_history_service(thread_id)
    assert len(history["messages"]) == 2
    assert len(history["events"]) >= 1

    events = list_thread_events_service(thread_id, after=0, limit=10)
    assert events["events"][0]["event"] == "task_started"
    assert events["cursor"] >= events["events"][0]["event_id"]

    stream = stream_thread_events_service(thread_id, after=0, limit=10, timeout_seconds=0.1, max_batches=1)
    first_chunk = next(stream)
    assert "event: thread_events" in first_chunk
    assert thread_id in first_chunk

    formatted = format_sse_event({"cursor": 5, "events": [], "thread_id": thread_id})
    assert formatted.startswith("id: 5")
