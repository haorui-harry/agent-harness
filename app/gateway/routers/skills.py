"""Gateway skill-package endpoints and services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.gateway.deps import get_harness

try:  # pragma: no cover - optional dependency
    from fastapi import APIRouter, HTTPException
except Exception:  # pragma: no cover - optional dependency
    APIRouter = None
    HTTPException = RuntimeError


class SkillPackageResponse(BaseModel):
    name: str
    description: str
    source: str
    package_path: str = ""
    enabled: bool = True
    category: str = "general"
    owner: str = "core"
    version: str = "1.0.0"
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    tool_refs: list[str] = Field(default_factory=list)
    skill_refs: list[str] = Field(default_factory=list)
    artifact_kinds: list[str] = Field(default_factory=list)
    runtime_requirements: list[str] = Field(default_factory=list)


class SkillPackagesListResponse(BaseModel):
    skills: list[SkillPackageResponse]


class SkillPackageUpdateRequest(BaseModel):
    enabled: bool


class SkillPackageInstallRequest(BaseModel):
    path: str


class SkillPackageInstallResponse(BaseModel):
    success: bool
    skill_name: str
    path: str = ""
    message: str


def list_skills_service(*, enabled_only: bool = False) -> dict[str, Any]:
    engine = get_harness()
    return {"skills": engine.list_skill_packages(enabled_only=enabled_only)}


def get_skill_service(skill_name: str) -> dict[str, Any] | None:
    engine = get_harness()
    return engine.get_skill_package(skill_name)


def update_skill_service(skill_name: str, *, enabled: bool) -> dict[str, Any]:
    engine = get_harness()
    return engine.update_skill_package(skill_name, enabled=enabled)


def install_skill_service(path: str) -> dict[str, Any]:
    engine = get_harness()
    return engine.install_skill_package_archive(Path(path))


router = APIRouter(prefix="/api/skills", tags=["skills"]) if APIRouter else None

if router is not None:  # pragma: no cover - thin FastAPI wrapper

    @router.get("", response_model=SkillPackagesListResponse)
    async def list_skills(enabled_only: bool = False) -> SkillPackagesListResponse:
        return SkillPackagesListResponse(**list_skills_service(enabled_only=enabled_only))


    @router.get("/{skill_name}", response_model=SkillPackageResponse)
    async def get_skill(skill_name: str) -> SkillPackageResponse:
        payload = get_skill_service(skill_name)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")
        return SkillPackageResponse(**payload)


    @router.put("/{skill_name}", response_model=SkillPackageResponse)
    async def update_skill(skill_name: str, body: SkillPackageUpdateRequest) -> SkillPackageResponse:
        try:
            return SkillPackageResponse(**update_skill_service(skill_name, enabled=body.enabled))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @router.post("/install", response_model=SkillPackageInstallResponse)
    async def install_skill(body: SkillPackageInstallRequest) -> SkillPackageInstallResponse:
        try:
            return SkillPackageInstallResponse(**install_skill_service(body.path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
