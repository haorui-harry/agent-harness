"""DeerFlow-style skill/package catalog for built-in and external capabilities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.state import SkillMetadata
from app.ecosystem.marketplace import list_marketplace_skill_metadata
from app.skills.registry import list_builtin_skills, list_external_skills

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


def _split_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,\n]", text) if item.strip()]


def _parse_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw_meta, body = match.groups()
    meta: dict[str, Any] = {}
    for line in raw_meta.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body.strip()


@dataclass(frozen=True)
class SkillPackage:
    """Package-like view over one skill bundle."""

    name: str
    description: str
    source: str
    package_path: str = ""
    enabled: bool = True
    category: str = "general"
    owner: str = "core"
    version: str = "1.0.0"
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    tool_refs: list[str] = field(default_factory=list)
    skill_refs: list[str] = field(default_factory=list)
    artifact_kinds: list[str] = field(default_factory=list)
    runtime_requirements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "package_path": self.package_path,
            "enabled": self.enabled,
            "category": self.category,
            "owner": self.owner,
            "version": self.version,
            "summary": self.summary or self.description,
            "tags": list(self.tags),
            "tool_refs": list(self.tool_refs),
            "skill_refs": list(self.skill_refs),
            "artifact_kinds": list(self.artifact_kinds),
            "runtime_requirements": list(self.runtime_requirements),
        }

    def score_for_query(self, query: str, *, target: str = "general") -> float:
        lowered = str(query or "").lower()
        if not lowered:
            base = 0.12 if self.enabled else 0.0
            if self.category in {"general", "utility", "analysis"}:
                base += 0.08
            return base
        score = 0.0
        haystack = " ".join(
            [
                self.name,
                self.description,
                self.summary,
                " ".join(self.tags),
                " ".join(self.skill_refs),
                " ".join(self.tool_refs),
                " ".join(self.artifact_kinds),
                " ".join(self.runtime_requirements),
            ]
        ).lower()
        if self.name.lower() in lowered:
            score += 0.5
        for word in set(re.findall(r"[a-z0-9_-]+", lowered)):
            if len(word) < 3:
                continue
            if word in haystack:
                score += 0.08
        if target and target != "general":
            target_lower = target.lower()
            if target_lower in haystack or target_lower == self.category.lower():
                score += 0.2
        if self.enabled:
            score += 0.05
        if self.category in {"general", "utility"}:
            score += 0.04
        if any(tag in {"general", "discovery", "workflow", "analysis"} for tag in self.tags):
            score += 0.03
        return score


class SkillPackageCatalog:
    """Unified package catalog across on-disk skills, built-ins, external, and marketplace."""

    def __init__(self, skills_root: Path | None = None, state_file: Path | None = None) -> None:
        self.skills_root = skills_root or (Path(__file__).resolve().parents[2] / "skills")
        self.state_file = state_file
        self._packages = self._load_packages()

    def list_packages(self, enabled_only: bool = False) -> list[SkillPackage]:
        items = list(self._packages.values())
        if enabled_only:
            items = [item for item in items if item.enabled]
        return sorted(items, key=lambda item: (item.source, item.name))

    def get_package(self, name: str) -> SkillPackage | None:
        return self._packages.get(name)

    def suggest(self, query: str, *, target: str = "general", limit: int = 6, enabled_only: bool = True) -> list[SkillPackage]:
        ranked: list[tuple[float, SkillPackage]] = []
        pool = self.list_packages(enabled_only=enabled_only)
        for package in pool:
            score = package.score_for_query(query, target=target)
            if score <= 0:
                continue
            ranked.append((score, package))
        ranked.sort(key=lambda item: (-item[0], item[1].name))
        if ranked:
            return [package for _, package in ranked[: max(1, limit)]]

        fallback_names = [
            "general-purpose",
            "find-skills",
            "consulting-analysis",
            "workspace-operator",
            "deep-research",
        ]
        selected: list[SkillPackage] = []
        seen: set[str] = set()
        by_name = {item.name: item for item in pool}
        for name in fallback_names:
            item = by_name.get(name)
            if item and item.name not in seen:
                selected.append(item)
                seen.add(item.name)
            if len(selected) >= max(1, limit):
                break
        if len(selected) < max(1, limit):
            for item in pool:
                if item.name in seen:
                    continue
                if item.category in {"general", "utility", "analysis"}:
                    selected.append(item)
                    seen.add(item.name)
                if len(selected) >= max(1, limit):
                    break
        return selected[: max(1, limit)]

    def _load_packages(self) -> dict[str, SkillPackage]:
        packages: dict[str, SkillPackage] = {}
        from app.skills.manager import load_skill_package_state

        states = load_skill_package_state(state_file=self.state_file)
        for package in self._load_disk_packages():
            packages[package.name] = package

        for meta in list_builtin_skills():
            packages.setdefault(meta.name, self._package_from_metadata(meta, source="builtin"))
        for meta in list_external_skills():
            packages.setdefault(meta.name, self._package_from_metadata(meta, source="external"))
        for meta in list_marketplace_skill_metadata():
            packages.setdefault(meta.name, self._package_from_metadata(meta, source="marketplace"))
        for name, state in states.items():
            package = packages.get(name)
            if package is None:
                continue
            packages[name] = SkillPackage(
                name=package.name,
                description=package.description,
                source=package.source,
                package_path=package.package_path,
                enabled=state.enabled,
                category=package.category,
                owner=package.owner,
                version=package.version,
                summary=package.summary,
                tags=list(package.tags),
                tool_refs=list(package.tool_refs),
                skill_refs=list(package.skill_refs),
                artifact_kinds=list(package.artifact_kinds),
                runtime_requirements=list(package.runtime_requirements),
            )
        return packages

    def _load_disk_packages(self) -> list[SkillPackage]:
        packages: list[SkillPackage] = []
        for category in ("public", "custom"):
            root = self.skills_root / category
            if not root.exists():
                continue
            for skill_file in sorted(root.rglob("SKILL.md")):
                meta, body = _parse_frontmatter(skill_file)
                name = str(meta.get("name", skill_file.parent.name)).strip()
                if not name:
                    continue
                packages.append(
                    SkillPackage(
                        name=name,
                        description=str(meta.get("description", "")).strip() or body.splitlines()[0].strip() if body else name,
                        source=category,
                        package_path=str(skill_file.parent),
                        enabled=str(meta.get("enabled", "true")).strip().lower() not in {"0", "false", "no"},
                        category=str(meta.get("category", "general")).strip() or "general",
                        owner=str(meta.get("owner", category)).strip() or category,
                        version=str(meta.get("version", "1.0.0")).strip() or "1.0.0",
                        summary=str(meta.get("summary", "")).strip() or str(meta.get("description", "")).strip(),
                        tags=_split_list(meta.get("tags", "")),
                        tool_refs=_split_list(meta.get("tools", "")),
                        skill_refs=_split_list(meta.get("skills", "")),
                        artifact_kinds=_split_list(meta.get("artifacts", "")),
                        runtime_requirements=_split_list(meta.get("runtime_requirements", "")),
                    )
                )
        return packages

    @staticmethod
    def _package_from_metadata(meta: SkillMetadata, *, source: str) -> SkillPackage:
        return SkillPackage(
            name=meta.name,
            description=meta.description,
            source=source,
            package_path="app/skills/registry.py",
            enabled=True,
            category=meta.category.value,
            owner=meta.owner,
            version=meta.version,
            summary=meta.summary or meta.description,
            tags=list(meta.confidence_keywords[:8]),
            tool_refs=[],
            skill_refs=[meta.name],
            artifact_kinds=[meta.output_type],
            runtime_requirements=[meta.reasoning_role or "general"],
        )


def load_skill_package_catalog(skills_root: Path | None = None) -> SkillPackageCatalog:
    """Build the default package catalog."""

    return SkillPackageCatalog(skills_root=skills_root)
