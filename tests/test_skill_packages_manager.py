"""Tests for DeerFlow-style skill package lifecycle management."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from app.skills.manager import SkillPackageManager
from app.skills.packages import SkillPackageCatalog


def test_skill_package_manager_can_toggle_package_state(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    public_root = skills_root / "public" / "demo-skill"
    public_root.mkdir(parents=True)
    (public_root / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\n# Demo\n",
        encoding="utf-8",
    )
    manager = SkillPackageManager(skills_root=skills_root, state_file=tmp_path / "skill_packages.json")

    listed = manager.list_packages()
    assert any(item["name"] == "demo-skill" and item["enabled"] is True for item in listed)

    updated = manager.update_package("demo-skill", enabled=False)
    assert updated["enabled"] is False

    catalog = SkillPackageCatalog(skills_root=skills_root, state_file=tmp_path / "skill_packages.json")
    package = catalog.get_package("demo-skill")
    assert package is not None
    assert package.enabled is False


def test_skill_package_manager_installs_skill_archive(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    manager = SkillPackageManager(skills_root=skills_root, state_file=tmp_path / "skill_packages.json")
    archive = tmp_path / "demo.skill"

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "demo-installed/SKILL.md",
            "---\nname: demo-installed\ndescription: installed skill\n---\n# Demo Installed\n",
        )

    result = manager.install_archive(archive)
    assert result["success"] is True
    assert (skills_root / "custom" / "demo-installed" / "SKILL.md").exists()

    catalog = SkillPackageCatalog(skills_root=skills_root, state_file=tmp_path / "skill_packages.json")
    package = catalog.get_package("demo-installed")
    assert package is not None
    assert package.source == "custom"


def test_thread_runtime_exports_deerflow_like_frontend_snapshot(tmp_path: Path) -> None:
    from app.agents.runtime import AgentThreadRuntime

    runtime = AgentThreadRuntime(tmp_path / "threads")
    thread = runtime.create_thread(title="Exported Thread", agent_name="super-agent")
    runtime.append_message(thread["thread_id"], "user", "Inspect the workspace.")
    runtime.append_message(thread["thread_id"], "assistant", "Generated a result.")
    runtime.write_artifact(
        thread["thread_id"],
        name="index.html",
        content="<html></html>",
        kind="webpage",
        content_type="text/html",
        summary="exported webpage",
    )
    runtime.start_task_graph_async(
        thread["thread_id"],
        graph={
            "graph_id": "demo",
            "nodes": [
                {"node_id": "scope", "title": "Scope", "node_type": "routing", "status": "ready", "depends_on": [], "commands": [], "notes": [], "artifacts": [], "metrics": {"delay_ms": 10}},
            ],
        },
        execution_label="demo",
    )
    payload = runtime.export_frontend_thread_snapshot(thread["thread_id"])

    assert payload["values"]["title"] == "Exported Thread"
    assert payload["values"]["messages"][0]["type"] == "human"
    assert payload["values"]["artifacts"][0].startswith("/mnt/user-data/outputs/")
    assert payload["metadata"]["thread_id"] == thread["thread_id"]
    assert "workspace_path" in payload["values"]["thread_data"]
