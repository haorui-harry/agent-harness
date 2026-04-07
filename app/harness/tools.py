"""Harness tool adapters (API/browser/code-like capabilities)."""

from __future__ import annotations

import fnmatch
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.task_graph import TaskGraphNode
from app.ecosystem.marketplace import (
    discover_for_query,
    get_provider_stats,
    get_trending_skills,
    list_marketplace_skills,
)
from app.ecosystem.store import load_marketplace
from app.harness.evidence import EvidenceProviderRegistry
from app.harness.manifest import ToolManifestRegistry
from app.harness.models import ToolCall, ToolResult, ToolType
from app.harness.task_profile import analyze_task_request, build_dynamic_task_graph, infer_domains
from app.skills.packages import SkillPackageCatalog
from app.skills.registry import list_all_skills


class ToolRegistry:
    """Registry for harness-level tool calls."""

    def __init__(
        self,
        *,
        evidence_registry: EvidenceProviderRegistry | None = None,
        package_catalog: SkillPackageCatalog | None = None,
        gateway_config: dict[str, Any] | None = None,
    ) -> None:
        self._evidence = evidence_registry or EvidenceProviderRegistry()
        self._packages = package_catalog or SkillPackageCatalog()
        self._gateway = dict(gateway_config or {})
        self._manifests = ToolManifestRegistry()
        self._tool_schemas = self._build_tool_schemas()
        self._tools: dict[str, Callable[[dict[str, Any]], Any]] = {
            "tool_search": self._tool_search,
            "api_market_discover": self._api_market_discover,
            "browser_trending_scan": self._browser_trending_scan,
            "code_skill_search": self._code_skill_search,
            "workspace_file_search": self._workspace_file_search,
            "workspace_file_read": self._workspace_file_read,
            "workspace_file_write": self._workspace_file_write,
            "task_graph_builder": self._task_graph_builder,
            "policy_risk_matrix": self._policy_risk_matrix,
            "memory_context_digest": self._memory_context_digest,
            "ecosystem_provider_radar": self._ecosystem_provider_radar,
            "evidence_dossier_builder": self._evidence_dossier_builder,
            "external_resource_hub": self._external_resource_hub,
            "api_skill_dependency_graph": self._api_skill_dependency_graph,
            "code_router_blueprint": self._code_router_blueprint,
            "api_skill_portfolio_optimizer": self._api_skill_portfolio_optimizer,
            "code_experiment_design": self._code_experiment_design,
        }
        self._tool_types: dict[str, ToolType] = {
            "tool_search": ToolType.CODE,
            "api_market_discover": ToolType.API,
            "browser_trending_scan": ToolType.BROWSER,
            "code_skill_search": ToolType.CODE,
            "workspace_file_search": ToolType.CODE,
            "workspace_file_read": ToolType.CODE,
            "workspace_file_write": ToolType.CODE,
            "task_graph_builder": ToolType.CODE,
            "policy_risk_matrix": ToolType.CODE,
            "memory_context_digest": ToolType.CODE,
            "ecosystem_provider_radar": ToolType.API,
            "evidence_dossier_builder": ToolType.BROWSER,
            "external_resource_hub": ToolType.BROWSER,
            "api_skill_dependency_graph": ToolType.API,
            "code_router_blueprint": ToolType.CODE,
            "api_skill_portfolio_optimizer": ToolType.API,
            "code_experiment_design": ToolType.CODE,
        }

    def available_tools(self) -> list[str]:
        """List all registered tools."""

        return sorted(self._tools.keys())

    def describe_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return schema-like descriptions for searchable tool discovery."""

        selected = set(names or [])
        out: list[dict[str, Any]] = []
        for manifest in self._manifests.list_all():
            if selected and manifest.name not in selected:
                continue
            out.append(
                {
                    "name": manifest.name,
                    "tool_type": manifest.tool_type.value,
                    "summary": manifest.summary,
                    "risk_level": manifest.risk_level,
                    "tags": list(manifest.tags),
                    "intents": list(manifest.intents),
                    "capabilities": list(manifest.capabilities),
                    "schema": self._tool_schemas.get(manifest.name, {}),
                }
            )
        return out

    def list_evidence_sources(self) -> list[dict[str, Any]]:
        """List configured evidence sources backing evidence-aware tools."""

        return self._evidence.list_sources()

    def list_skill_packages(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        """List package-style skills exposed to the planner/runtime."""

        return [item.to_dict() for item in self._packages.list_packages(enabled_only=enabled_only)]

    def suggest_skill_packages(self, query: str, *, target: str = "general", limit: int = 6) -> list[dict[str, Any]]:
        """Suggest package-style skills for a task."""

        return [item.to_dict() for item in self._packages.suggest(query, target=target, limit=limit)]

    def infer_tool_type(self, tool_name: str) -> ToolType:
        """Infer tool type from tool name."""

        return self._tool_types.get(tool_name, ToolType.CODE)

    def call(self, tool_call: ToolCall) -> ToolResult:
        """Execute one tool call and return standardized result."""

        start = time.time()
        fn = self._tools.get(tool_call.name)
        if not fn:
            end = time.time()
            return ToolResult(
                name=tool_call.name,
                success=False,
                output={},
                latency_ms=(end - start) * 1000.0,
                error=f"unknown_tool:{tool_call.name}",
            )

        try:
            output = fn(tool_call.args)
            metadata: dict[str, Any] = {}
            if isinstance(output, dict) and "__tool_metadata__" in output:
                metadata = dict(output.get("__tool_metadata__", {}))
                output = {key: value for key, value in output.items() if key != "__tool_metadata__"}
            success = True
            error = ""
        except Exception as exc:  # pragma: no cover - defensive
            output = {}
            metadata = {}
            success = False
            error = str(exc)
        end = time.time()

        return ToolResult(
            name=tool_call.name,
            success=success,
            output=output,
            latency_ms=(end - start) * 1000.0,
            error=error,
            metadata=metadata,
        )

    def _tool_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        limit = max(1, min(int(args.get("limit", 5)), 20))
        catalog = self.describe_tools()
        if not query:
            matches = catalog[:limit]
        elif query.startswith("select:"):
            wanted = {item.strip() for item in query[7:].split(",") if item.strip()}
            matches = [item for item in catalog if item["name"] in wanted][:limit]
        else:
            pattern = query
            required = ""
            if query.startswith("+"):
                parts = query[1:].split(None, 1)
                required = parts[0].lower()
                pattern = parts[1] if len(parts) > 1 else required
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                regex = re.compile(re.escape(pattern), re.IGNORECASE)

            scored: list[tuple[float, dict[str, Any]]] = []
            for item in catalog:
                searchable = " ".join(
                    [
                        item["name"],
                        item["summary"],
                        " ".join(item.get("tags", [])),
                        " ".join(item.get("capabilities", [])),
                        " ".join(item.get("intents", [])),
                    ]
                )
                if required and required not in item["name"].lower():
                    continue
                if not regex.search(searchable):
                    continue
                score = 0.2
                if regex.search(item["name"]):
                    score += 0.5
                score += 0.08 * len(regex.findall(searchable))
                if required:
                    score += 0.2
                scored.append((score, item))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            matches = [item for _, item in scored[:limit]]

        return {
            "query": query,
            "count": len(matches),
            "matches": matches,
        }

    @staticmethod
    def _api_market_discover(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        limit = int(args.get("limit", 3))
        return {"matches": discover_for_query(query=query, limit=limit)}

    @staticmethod
    def _browser_trending_scan(args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit", 3))
        return {"trending": get_trending_skills(limit=limit)}

    def _code_skill_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).lower()
        query_tokens = set(re.findall(r"[a-z0-9_-]+", query))
        limit = int(args.get("limit", 8))
        target = str(args.get("target", "general")).strip().lower() or "general"

        ranked: list[tuple[float, dict[str, Any]]] = []
        for index, package in enumerate(
            self._packages.suggest(query, target=target, limit=max(limit, 6), enabled_only=False)
        ):
            score = package.score_for_query(query, target=target)
            score += max(0.0, 0.06 - 0.01 * index)
            ranked.append(
                (
                    score,
                    {
                        "name": package.name,
                        "category": package.category,
                        "tier": "package",
                        "cost": float(max(len(package.tool_refs), 1)),
                        "source": package.source,
                        "summary": package.summary or package.description,
                        "skills": list(package.skill_refs),
                        "tools": list(package.tool_refs),
                        "artifacts": list(package.artifact_kinds),
                        "match_score": round(score, 4),
                    },
                )
            )

        for meta in list_all_skills():
            haystack = " ".join([meta.name, meta.description, " ".join(meta.confidence_keywords)]).lower()
            score = 0.0
            if not query:
                score = 0.18
            if query and meta.name.lower() in query:
                score += 0.52
            overlap = [token for token in query_tokens if token in haystack]
            if overlap:
                score += min(0.3, 0.08 * len(overlap[:4]))
            if target != "general" and (target == meta.category.value or target in haystack):
                score += 0.18
            if score <= 0.0:
                continue
            ranked.append(
                (
                    score,
                    {
                        "name": meta.name,
                        "category": meta.category.value,
                        "tier": meta.tier.value,
                        "cost": meta.compute_cost,
                        "source": "builtin",
                        "summary": meta.summary or meta.description,
                        "skills": [meta.name],
                        "tools": [],
                        "artifacts": [meta.output_type],
                        "match_score": round(score, 4),
                    },
                )
            )

        ranked.sort(key=lambda item: (-item[0], item[1]["tier"] != "package", item[1]["name"]))
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for _score, item in ranked:
            key = (str(item.get("name", "")), str(item.get("tier", "")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                break

        if not merged:
            for package in self._packages.list_packages(enabled_only=False)[:limit]:
                merged.append(
                    {
                        "name": package.name,
                        "category": package.category,
                        "tier": "package",
                        "cost": float(max(len(package.tool_refs), 1)),
                        "source": package.source,
                        "summary": package.summary or package.description,
                        "skills": list(package.skill_refs),
                        "tools": list(package.tool_refs),
                        "artifacts": list(package.artifact_kinds),
                    }
                )
        return {"skills": merged[:limit], "gateway": self._gateway}

    @staticmethod
    def _workspace_file_search(args: dict[str, Any]) -> dict[str, Any]:
        workspace_root = ToolRegistry._resolve_workspace_root(args)
        query = str(args.get("query", "")).strip().lower()
        pattern = str(args.get("pattern", "")).strip().lower()
        glob = str(args.get("glob", "*")).strip() or "*"
        limit = max(1, min(int(args.get("limit", 10)), 50))
        matches: list[dict[str, Any]] = []

        for path in sorted(p for p in workspace_root.rglob("*") if p.is_file()):
            rel = path.relative_to(workspace_root).as_posix()
            if glob and not fnmatch.fnmatch(rel, glob) and not fnmatch.fnmatch(path.name, glob):
                continue
            name_hit = pattern in rel.lower() if pattern else False
            content_hit = False
            preview = rel
            if query or pattern:
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    content = ""
                if query and query in content.lower():
                    content_hit = True
                    preview = ToolRegistry._preview_match(content, query)
                elif pattern and pattern in content.lower():
                    content_hit = True
                    preview = ToolRegistry._preview_match(content, pattern)
            if query or pattern:
                if not (name_hit or content_hit):
                    continue
            matches.append(
                {
                    "path": str(path),
                    "relative_path": rel,
                    "name_hit": name_hit,
                    "content_hit": content_hit,
                    "preview": preview,
                }
            )
            if len(matches) >= limit:
                break

        return {
            "workspace_root": str(workspace_root),
            "count": len(matches),
            "matches": matches,
        }

    @staticmethod
    def _workspace_file_read(args: dict[str, Any]) -> dict[str, Any]:
        workspace_root = ToolRegistry._resolve_workspace_root(args)
        target = ToolRegistry._resolve_within_workspace(
            workspace_root,
            str(args.get("path", args.get("relative_path", ""))).strip(),
        )
        start_line = max(1, int(args.get("start_line", 1)))
        default_count = max(1, int(args.get("line_count", 80)))
        end_line = max(start_line, int(args.get("end_line", start_line + default_count - 1)))
        max_chars = max(200, min(int(args.get("max_chars", 4000)), 20000))
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        window = lines[start_line - 1 : end_line]
        excerpt = "\n".join(window)[:max_chars]
        return {
            "path": str(target),
            "relative_path": target.relative_to(workspace_root).as_posix(),
            "start_line": start_line,
            "end_line": min(end_line, len(lines)),
            "line_count": len(lines),
            "excerpt": excerpt,
        }

    @staticmethod
    def _workspace_file_write(args: dict[str, Any]) -> dict[str, Any]:
        workspace_root = ToolRegistry._resolve_workspace_root(args)
        relative_path = str(args.get("path", args.get("relative_path", ""))).strip()
        if not relative_path:
            raise ValueError("path is required")
        target = ToolRegistry._resolve_within_workspace(workspace_root, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = str(args.get("content", ""))
        target.write_text(content, encoding="utf-8")
        return {
            "path": str(target),
            "relative_path": target.relative_to(workspace_root).as_posix(),
            "bytes_written": target.stat().st_size,
        }

    @staticmethod
    def _task_graph_builder(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        target = str(args.get("target", "general")).strip().lower() or "general"
        workspace_root = ToolRegistry._resolve_workspace_root(args)
        live_model = args.get("live_model", {}) if isinstance(args.get("live_model", {}), dict) else None
        profile, graph = build_dynamic_task_graph(
            query=query,
            target=target,
            workspace_root=workspace_root,
            live_model_overrides=live_model,
        )
        return {
            "query": query,
            "target": target,
            "workspace_root": str(workspace_root),
            "profile": profile.to_dict(),
            "graph": graph.to_dict(),
        }

    def _policy_risk_matrix(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).lower()
        evidence_bundle = self._evidence.collect(
            query=query,
            limit=max(1, min(int(args.get("evidence_limit", 4)), 8)),
            domains=["risk", "governance", "compliance"],
        )
        dimensions = {
            "security": ["security", "attack", "credential", "secret", "token", "breach"],
            "compliance": ["compliance", "audit", "regulation", "policy", "governance", "legal"],
            "delivery": ["deadline", "timeline", "release", "milestone", "delivery", "ship"],
            "financial": ["cost", "budget", "roi", "price", "financial", "spend"],
            "reputation": ["brand", "trust", "stakeholder", "public", "reputation"],
        }
        controls = {
            "security": ["least-privilege tools", "input sanitization", "human approval for high risk"],
            "compliance": ["trace logging", "policy checklist", "evidence retention"],
            "delivery": ["milestone gating", "fallback plan", "critical path buffer"],
            "financial": ["cost cap per step", "budget checkpoint", "defer expensive tools"],
            "reputation": ["transparent rationale", "minority report", "stakeholder review"],
        }

        matrix: list[dict[str, Any]] = []
        overall_score = 0.0
        evidence_records = evidence_bundle.get("records", []) if isinstance(evidence_bundle, dict) else []
        for dim, words in dimensions.items():
            hits = sum(1 for token in words if token in query)
            score = min(1.0, 0.2 + 0.2 * hits) if hits > 0 else 0.2
            evidence_hits = 0
            for record in evidence_records:
                if not isinstance(record, dict):
                    continue
                domains = [str(x).lower() for x in record.get("domains", [])]
                tags = [str(x).lower() for x in record.get("tags", [])]
                if dim in domains or dim in tags:
                    evidence_hits += 1
            if evidence_hits:
                score = min(1.0, score + min(0.2, 0.05 * evidence_hits))
            if "critical" in query and dim in {"security", "compliance"}:
                score = min(1.0, score + 0.2)
            level = "low"
            if score >= 0.75:
                level = "high"
            elif score >= 0.45:
                level = "medium"
            matrix.append(
                {
                    "dimension": dim,
                    "score": round(score, 3),
                    "level": level,
                    "controls": controls[dim],
                }
            )
            overall_score += score

        overall_score /= max(len(matrix), 1)
        overall_level = "low"
        if overall_score >= 0.75:
            overall_level = "high"
        elif overall_score >= 0.45:
            overall_level = "medium"

        return {
            "risk_matrix": matrix,
            "overall_score": round(overall_score, 3),
            "overall_level": overall_level,
            "evidence_packet": {
                "count": evidence_bundle.get("count", 0),
                "citations": evidence_bundle.get("citations", []),
                "records": evidence_records[:4],
            },
            "__tool_metadata__": {
                "evidence_records": evidence_records[:4],
                "evidence_citations": evidence_bundle.get("citations", [])[:6],
                "evidence_source_count": len(evidence_bundle.get("providers", [])),
            },
        }

    @staticmethod
    def _memory_context_digest(args: dict[str, Any]) -> dict[str, Any]:
        events = args.get("events", [])
        limit = int(args.get("limit", 8))
        if not isinstance(events, list):
            events = []
        sampled = events[-limit:]

        total = len(sampled)
        success_count = sum(1 for item in sampled if isinstance(item, dict) and item.get("success"))
        avg_latency = 0.0
        latency_values = [
            float(item.get("latency_ms", 0.0))
            for item in sampled
            if isinstance(item, dict) and "latency_ms" in item
        ]
        if latency_values:
            avg_latency = sum(latency_values) / len(latency_values)

        recent_tools: list[str] = []
        for item in reversed(sampled):
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", ""))
            if tool and tool not in recent_tools:
                recent_tools.append(tool)

        return {
            "event_count": total,
            "success_rate": round(success_count / max(total, 1), 3),
            "avg_latency_ms": round(avg_latency, 2),
            "recent_tools": recent_tools[:5],
        }

    @staticmethod
    def _ecosystem_provider_radar(args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit", 5))
        all_skills = list_marketplace_skills()
        providers: list[str] = []
        for item in all_skills:
            provider = str(item.get("provider", ""))
            if provider and provider not in providers:
                providers.append(provider)
            if len(providers) >= limit:
                break

        stats = [get_provider_stats(provider) for provider in providers]
        stats.sort(key=lambda item: item.get("avg_reputation", 0.0), reverse=True)
        return {"providers": stats[:limit]}

    def _evidence_dossier_builder(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        limit = max(1, min(int(args.get("limit", 6)), 12))
        domains = [str(x) for x in args.get("domains", [])] if isinstance(args.get("domains", []), list) else []
        bundle = self._evidence.collect(query=query, limit=limit, domains=domains or self._infer_domains(query))
        records = bundle.get("records", []) if isinstance(bundle, dict) else []
        providers = bundle.get("providers", []) if isinstance(bundle, dict) else []
        return {
            "query": query,
            "record_count": len(records),
            "providers": providers,
            "citations": bundle.get("citations", []),
            "records": records,
            "dossier_summary": {
                "source_count": len(providers),
                "domain_coverage": sorted(
                    {
                        str(domain)
                        for item in records
                        if isinstance(item, dict)
                        for domain in item.get("domains", [])
                    }
                ),
            },
            "__tool_metadata__": {
                "evidence_records": records[:6],
                "evidence_citations": bundle.get("citations", [])[:8],
                "evidence_source_count": len(providers),
            },
        }

    def _external_resource_hub(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).lower()
        limit = int(args.get("limit", 5))

        evidence_bundle = self._evidence.collect(query=query, limit=max(limit, 6), domains=self._infer_domains(query))
        evidence_records = evidence_bundle.get("records", []) if isinstance(evidence_bundle, dict) else []
        scored: list[tuple[dict[str, Any], float]] = []
        for item in evidence_records:
            if not isinstance(item, dict):
                continue
            scored.append(
                (
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", item.get("path", "")),
                        "tags": item.get("tags", []),
                        "summary": item.get("summary", ""),
                        "source_id": item.get("source_id", "evidence"),
                    },
                    float(item.get("score", 0.0)),
                )
            )
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return {
            "resources": [
                {
                    "title": item["title"],
                    "url": item["url"],
                    "score": round(score, 3),
                    "tags": item["tags"],
                    "summary": item.get("summary", ""),
                    "source_id": item.get("source_id", "curated"),
                }
                for item, score in scored[:limit]
            ],
            "__tool_metadata__": {
                "evidence_records": evidence_records[:6],
                "evidence_citations": evidence_bundle.get("citations", [])[:8],
                "evidence_source_count": len(evidence_bundle.get("providers", [])),
            },
        }

    @staticmethod
    def _api_skill_dependency_graph(args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit", 20))
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        all_skills = list_all_skills()
        for meta in all_skills:
            nodes.append(
                {
                    "id": meta.name,
                    "category": meta.category.value,
                    "tier": meta.tier.value,
                    "cost": meta.compute_cost,
                }
            )
            for target in meta.synergies:
                edges.append({"from": meta.name, "to": target, "type": "synergy"})
            for target in meta.conflicts:
                edges.append({"from": meta.name, "to": target, "type": "conflict"})

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes[:limit],
            "edges": edges[:limit],
        }

    @staticmethod
    def _code_router_blueprint(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        return {
            "query": query,
            "blueprint": [
                {
                    "phase": "observe",
                    "objective": "Capture routing, guardrail, and tool telemetry as first-class trace.",
                    "artifacts": ["discovery_trace", "security_decisions", "step_latency"],
                },
                {
                    "phase": "decide",
                    "objective": "Apply policy-conditioned tool selection with explicit rationale.",
                    "artifacts": ["tool_manifest", "recipe_decision", "fallback_chain"],
                },
                {
                    "phase": "adapt",
                    "objective": "Continuously improve selection from eval and red-team outcomes.",
                    "artifacts": ["eval_metrics", "redteam_pass_rate", "failure_clusters"],
                },
            ],
            "design_principles": [
                "Prefer composable tool cards over hard-coded orchestration.",
                "Treat security checks as executable policy, not static docs.",
                "Keep each loop step explainable with source + score + constraints.",
            ],
        }

    @staticmethod
    def _api_skill_portfolio_optimizer(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        limit = max(1, min(int(args.get("limit", 5)), 10))
        tolerance = str(args.get("risk_tolerance", "medium")).strip().lower()
        risk_budget = {"low": 0.25, "medium": 0.5, "high": 0.8}.get(tolerance, 0.5)

        discovered = discover_for_query(query=query, limit=25)
        market_map = {item.metadata.name: item for item in load_marketplace()}

        ranked: list[dict[str, Any]] = []
        for item in discovered:
            name = str(item.get("name", ""))
            market_skill = market_map.get(name)
            if not market_skill:
                continue

            relevance = float(item.get("score", 0.0))
            reputation = float(item.get("reputation", 0.0))
            trending = float(item.get("trending_score", 0.0))
            cost = float(market_skill.metadata.compute_cost or 1.0)
            risk_profile = str(market_skill.metadata.risk_profile).lower()
            risk_score = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(risk_profile, 0.5)

            cost_term = max(0.0, 1.0 - min(cost / 3.0, 1.0))
            risk_penalty = max(0.0, risk_score - risk_budget) * 0.28
            objective = (
                0.46 * relevance
                + 0.26 * reputation
                + 0.13 * trending
                + 0.15 * cost_term
                - risk_penalty
            )

            ranked.append(
                {
                    "name": name,
                    "provider": market_skill.provider,
                    "objective_score": round(objective, 4),
                    "signals": {
                        "relevance": round(relevance, 4),
                        "reputation": round(reputation, 4),
                        "trending": round(trending, 4),
                        "risk_score": round(risk_score, 4),
                        "cost_score": round(cost_term, 4),
                    },
                    "rationale": [
                        f"risk_tolerance={tolerance}",
                        f"risk_profile={risk_profile}",
                        f"compute_cost={cost:.2f}",
                        f"tags={','.join(item.get('tags', [])[:3])}",
                    ],
                }
            )

        ranked.sort(key=lambda item: float(item.get("objective_score", 0.0)), reverse=True)
        selected = ranked[:limit]

        providers = sorted({item.get("provider", "") for item in selected if item.get("provider")})
        avg_objective = sum(float(item.get("objective_score", 0.0)) for item in selected) / max(len(selected), 1)
        avg_risk = (
            sum(float(item.get("signals", {}).get("risk_score", 0.0)) for item in selected)
            / max(len(selected), 1)
        )

        return {
            "query": query,
            "risk_tolerance": tolerance,
            "count": len(selected),
            "portfolio": selected,
            "portfolio_summary": {
                "provider_diversity": len(providers),
                "providers": providers,
                "avg_objective_score": round(avg_objective, 4),
                "avg_risk_score": round(avg_risk, 4),
            },
        }

    @staticmethod
    def _code_experiment_design(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        objective = str(args.get("objective", "maximize value while preserving safety")).strip()
        max_experiments = max(2, min(int(args.get("max_experiments", 6)), 12))
        lowered = query.lower()

        base_metrics = [
            "value_index",
            "tool_success_rate",
            "completion_score",
            "security_block_count",
            "discovery_utilization",
        ]
        if "latency" in lowered or "fast" in lowered:
            base_metrics.append("avg_latency_ms")
        if "innovation" in lowered or "novel" in lowered:
            base_metrics.append("innovation")

        experiments = [
            {
                "experiment_id": "exp-00-baseline",
                "change": "Balanced mode with auto recipe",
                "hypothesis": "Provides stable baseline for value and completion.",
                "controls": ["fixed query set", "fixed random seed", "same tool catalog"],
            },
            {
                "experiment_id": "exp-01-no-discovery",
                "change": "Disable dynamic discovery (planner fallback only)",
                "hypothesis": "Expected to reduce innovation and discovery utilization.",
                "controls": ["same constraints", "same queries"],
            },
            {
                "experiment_id": "exp-02-strict-security",
                "change": "Enable strict security profile with reduced network/browser actions",
                "hypothesis": "Expected to improve safety but may reduce completion.",
                "controls": ["same candidate modes", "same run budget"],
            },
            {
                "experiment_id": "exp-03-research-recipe",
                "change": "Use research-rig recipe for trend + experiment design tasks",
                "hypothesis": "Expected to improve observability and innovation in research category.",
                "controls": ["same metrics", "same evaluation set"],
            },
            {
                "experiment_id": "exp-04-daily-recipe",
                "change": "Use daily-operator recipe for daily operations tasks",
                "hypothesis": "Expected to improve decision quality and tool relevance on daily workloads.",
                "controls": ["same daily scenarios", "same budget limits"],
            },
            {
                "experiment_id": "exp-05-live-agent-on",
                "change": "Enable live agent enhancement with bounded call budget",
                "hypothesis": "Potentially improves answer quality; monitor cost and latency tradeoff.",
                "controls": ["call budget <= 8", "same prompt set"],
            },
        ][:max_experiments]

        threats = [
            "Heuristic labels for expected tools may bias coverage scoring.",
            "Synthetic scenario mix can differ from real production traffic.",
            "Shared memory history across repeated runs may affect independence.",
        ]
        mitigations = [
            "Add held-out scenario set and report by category.",
            "Track confidence interval and effect size, not only mean scores.",
            "Reset memory state for strict reproducibility studies when required.",
        ]

        return {
            "query": query,
            "objective": objective,
            "metrics": base_metrics,
            "experiment_matrix": experiments,
            "analysis_protocol": {
                "primary_metric": "value_index",
                "secondary_metrics": ["completion_score", "tool_success_rate", "security_alignment"],
                "recommended_repeats": 3,
                "statistical_checks": ["bootstrap_95ci", "category-wise comparison", "pass_rate >= 0.67"],
            },
            "threats_to_validity": threats,
            "mitigations": mitigations,
        }

    @staticmethod
    def _infer_domains(query: str) -> list[str]:
        domains = infer_domains(query)
        return domains or ["evidence"]

    @staticmethod
    def _resolve_workspace_root(args: dict[str, Any]) -> Path:
        raw = str(args.get("workspace_root", ".")).strip() or "."
        root = Path(raw).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _resolve_within_workspace(workspace_root: Path, relative_or_absolute: str) -> Path:
        if not relative_or_absolute:
            raise ValueError("path is required")
        raw = Path(relative_or_absolute)
        target = raw.resolve() if raw.is_absolute() else (workspace_root / raw).resolve()
        if target != workspace_root and workspace_root not in target.parents:
            raise ValueError(f"path escapes workspace: {relative_or_absolute}")
        return target

    @staticmethod
    def _preview_match(content: str, needle: str, radius: int = 80) -> str:
        lowered = content.lower()
        index = lowered.find(needle.lower())
        if index < 0:
            return content[: radius * 2].replace("\n", " ")
        start = max(0, index - radius)
        end = min(len(content), index + len(needle) + radius)
        return content[start:end].replace("\n", " ")

    @staticmethod
    def _build_task_graph_nodes(query: str, target: str) -> list[TaskGraphNode]:
        _, graph = build_dynamic_task_graph(query=query, target=target)
        nodes = graph.to_dict().get("nodes", [])
        return [
            TaskGraphNode(
                node_id=str(item.get("node_id", "")),
                title=str(item.get("title", "")),
                node_type=str(item.get("node_type", "")),
                status=str(item.get("status", "")),
                depends_on=list(item.get("depends_on", [])),
                commands=list(item.get("commands", [])),
                notes=list(item.get("notes", [])),
                metrics=dict(item.get("metrics", {})),
            )
            for item in nodes
        ]

    @staticmethod
    def _build_tool_schemas() -> dict[str, dict[str, Any]]:
        return {
            "tool_search": {
                "arguments": {
                    "query": "Search string. Supports select:name1,name2 and +required term forms.",
                    "limit": "Maximum number of tool schemas to return.",
                }
            },
            "api_market_discover": {"arguments": {"query": "Search query.", "limit": "Result limit."}},
            "browser_trending_scan": {"arguments": {"limit": "Trending result limit."}},
            "code_skill_search": {"arguments": {"query": "Capability keyword.", "limit": "Result limit."}},
            "workspace_file_search": {
                "arguments": {
                    "workspace_root": "Workspace root directory.",
                    "query": "Content substring to search for.",
                    "pattern": "Filename or content substring.",
                    "glob": "File glob filter such as *.py or docs/*.",
                    "limit": "Maximum matches.",
                }
            },
            "workspace_file_read": {
                "arguments": {
                    "workspace_root": "Workspace root directory.",
                    "path": "Relative or absolute path inside workspace_root.",
                    "start_line": "1-based start line.",
                    "end_line": "1-based end line.",
                    "max_chars": "Maximum excerpt length.",
                }
            },
            "workspace_file_write": {
                "arguments": {
                    "workspace_root": "Workspace root directory.",
                    "path": "Relative output path within workspace_root.",
                    "content": "Content to write.",
                }
            },
            "task_graph_builder": {
                "arguments": {
                    "query": "Task request to convert into executable task graph.",
                    "target": "general | code | research | ops.",
                    "workspace_root": "Optional workspace root used for downstream graph execution.",
                }
            },
            "policy_risk_matrix": {"arguments": {"query": "Task text.", "evidence_limit": "Evidence item limit."}},
            "memory_context_digest": {"arguments": {"events": "Prior event list.", "limit": "Event window size."}},
            "ecosystem_provider_radar": {"arguments": {"limit": "Provider count."}},
            "evidence_dossier_builder": {
                "arguments": {
                    "query": "Research query.",
                    "limit": "Evidence record limit.",
                    "domains": "Optional domain filters.",
                }
            },
            "external_resource_hub": {"arguments": {"query": "Topic query.", "limit": "Resource limit."}},
            "api_skill_dependency_graph": {"arguments": {"limit": "Node or edge output limit."}},
            "code_router_blueprint": {"arguments": {"query": "Architecture query."}},
            "api_skill_portfolio_optimizer": {
                "arguments": {
                    "query": "Portfolio query.",
                    "limit": "Portfolio size.",
                    "risk_tolerance": "low | medium | high.",
                }
            },
            "code_experiment_design": {
                "arguments": {
                    "query": "Experiment topic.",
                    "objective": "Optimization objective.",
                    "max_experiments": "Maximum experiment rows.",
                }
            },
        }
