"""Skill package lifecycle manager aligned with DeerFlow package operations."""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from app.skills.packages import SkillPackageCatalog

STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "skill_packages.json"


class SkillPackageExistsError(ValueError):
    """Raised when a custom skill package already exists."""


@dataclass(frozen=True)
class SkillPackageState:
    """Persisted lifecycle state for one package."""

    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled}


def _default_state() -> dict[str, Any]:
    return {"skills": {}}


def load_skill_package_state(state_file: Path | None = None) -> dict[str, SkillPackageState]:
    path = state_file or STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(_default_state(), indent=2), encoding="utf-8")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        path.write_text(json.dumps(_default_state(), indent=2), encoding="utf-8")
        return {}
    skills = payload.get("skills", {}) if isinstance(payload, dict) else {}
    if not isinstance(skills, dict):
        return {}
    return {
        str(name): SkillPackageState(enabled=bool(state.get("enabled", True)))
        for name, state in skills.items()
        if isinstance(state, dict)
    }


def save_skill_package_state(
    states: dict[str, SkillPackageState],
    state_file: Path | None = None,
) -> None:
    path = state_file or STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "skills": {
            name: state.to_dict()
            for name, state in sorted(states.items(), key=lambda item: item[0])
        }
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_skill_package_state(
    name: str,
    *,
    enabled: bool,
    state_file: Path | None = None,
) -> SkillPackageState:
    states = load_skill_package_state(state_file=state_file)
    states[str(name)] = SkillPackageState(enabled=bool(enabled))
    save_skill_package_state(states, state_file=state_file)
    return states[str(name)]


def _is_unsafe_zip_member(info: zipfile.ZipInfo) -> bool:
    name = info.filename
    if not name:
        return False
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    path = PurePosixPath(normalized)
    if path.is_absolute():
        return True
    if PureWindowsPath(name).is_absolute():
        return True
    return ".." in path.parts


def _is_symlink_member(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def _extract_safe_archive(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if _is_unsafe_zip_member(info):
                raise ValueError(f"unsafe archive member: {info.filename}")
            if _is_symlink_member(info):
                continue
            normalized = PurePosixPath(info.filename.replace("\\", "/"))
            target = dest.joinpath(*normalized.parts).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError(f"archive entry escapes target: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _resolve_skill_root(extracted_dir: Path) -> Path:
    visible = [item for item in extracted_dir.iterdir() if not item.name.startswith(".") and item.name != "__MACOSX"]
    if not visible:
        raise ValueError("skill archive is empty")
    if len(visible) == 1 and visible[0].is_dir():
        return visible[0]
    return extracted_dir


def install_skill_package_archive(
    archive_path: str | Path,
    *,
    skills_root: Path | None = None,
) -> dict[str, Any]:
    """Install a DeerFlow-style skill archive under skills/custom."""

    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(f"skill archive not found: {archive_path}")
    if path.suffix.lower() != ".skill":
        raise ValueError("skill archive must use .skill extension")

    root = skills_root or (Path(__file__).resolve().parents[2] / "skills")
    custom_root = root / "custom"
    custom_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _extract_safe_archive(path, tmp_path)
        skill_root = _resolve_skill_root(tmp_path)
        skill_file = skill_root / "SKILL.md"
        if not skill_file.exists():
            raise ValueError("skill archive does not contain SKILL.md")
        first_lines = skill_file.read_text(encoding="utf-8").splitlines()
        skill_name = ""
        for line in first_lines[:20]:
            if line.strip().startswith("name:"):
                skill_name = line.split(":", 1)[1].strip()
                break
        skill_name = skill_name or skill_root.name
        if not skill_name or any(token in skill_name for token in ("..", "/", "\\")):
            raise ValueError(f"invalid skill name: {skill_name}")
        target = custom_root / skill_name
        if target.exists():
            raise SkillPackageExistsError(f"skill '{skill_name}' already exists")
        shutil.copytree(skill_root, target)
    return {
        "success": True,
        "skill_name": skill_name,
        "path": str((custom_root / skill_name).resolve()),
        "message": f"skill '{skill_name}' installed successfully",
    }


class SkillPackageManager:
    """Lifecycle operations for package-style skills."""

    def __init__(self, *, skills_root: Path | None = None, state_file: Path | None = None) -> None:
        self.skills_root = skills_root or (Path(__file__).resolve().parents[2] / "skills")
        self.state_file = state_file or STATE_FILE

    def list_packages(self) -> list[dict[str, Any]]:
        catalog = SkillPackageCatalog(skills_root=self.skills_root, state_file=self.state_file)
        return [item.to_dict() for item in catalog.list_packages(enabled_only=False)]

    def get_package(self, name: str) -> dict[str, Any] | None:
        catalog = SkillPackageCatalog(skills_root=self.skills_root, state_file=self.state_file)
        package = catalog.get_package(name)
        return package.to_dict() if package else None

    def update_package(self, name: str, *, enabled: bool) -> dict[str, Any]:
        update_skill_package_state(name, enabled=enabled, state_file=self.state_file)
        package = self.get_package(name)
        if package is None:
            raise ValueError(f"unknown skill package: {name}")
        return package

    def install_archive(self, archive_path: str | Path) -> dict[str, Any]:
        return install_skill_package_archive(archive_path, skills_root=self.skills_root)
