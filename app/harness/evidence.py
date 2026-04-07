"""Evidence provider registry and normalized record injection for harness tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import parse, request


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class EvidenceRecord:
    """Normalized evidence record consumed by harness tools and showcase layers."""

    record_id: str
    title: str
    summary: str
    source_id: str
    source_type: str
    evidence_type: str = "reference"
    url: str = ""
    path: str = ""
    tags: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    trust_score: float = 0.7
    freshness_hint: str = "stable"
    content: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "title": self.title,
            "summary": self.summary,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "evidence_type": self.evidence_type,
            "url": self.url,
            "path": self.path,
            "tags": list(self.tags),
            "domains": list(self.domains),
            "trust_score": round(self.trust_score, 4),
            "freshness_hint": self.freshness_hint,
            "content": self.content,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class EvidenceSourceConfig:
    """Config for one evidence provider source."""

    source_id: str
    source_type: str
    enabled: bool = True
    root: str = ""
    url: str = ""
    query_param: str = "query"
    timeout_seconds: int = 8
    tags: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    headers_env: dict[str, str] = field(default_factory=dict)
    title_field: str = "title"
    summary_field: str = "summary"
    url_field: str = "url"
    tags_field: str = "tags"
    domains_field: str = "domains"
    records_key: str = "records"

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "EvidenceSourceConfig":
        return EvidenceSourceConfig(
            source_id=str(payload.get("source_id", payload.get("id", "source"))),
            source_type=str(payload.get("source_type", payload.get("type", "local_dossier"))),
            enabled=bool(payload.get("enabled", True)),
            root=str(payload.get("root", payload.get("path", ""))),
            url=str(payload.get("url", "")),
            query_param=str(payload.get("query_param", "query")),
            timeout_seconds=max(2, min(int(payload.get("timeout_seconds", 8)), 60)),
            tags=[str(x) for x in payload.get("tags", [])],
            domains=[str(x) for x in payload.get("domains", [])],
            headers_env={str(k): str(v) for k, v in payload.get("headers_env", {}).items()},
            title_field=str(payload.get("title_field", "title")),
            summary_field=str(payload.get("summary_field", "summary")),
            url_field=str(payload.get("url_field", "url")),
            tags_field=str(payload.get("tags_field", "tags")),
            domains_field=str(payload.get("domains_field", "domains")),
            records_key=str(payload.get("records_key", "records")),
        )


class EvidenceProviderRegistry:
    """Collect evidence records from local dossiers, static catalogs, and HTTP JSON endpoints."""

    def __init__(self, config_path: str = "", resolved_headers: dict[str, str] | None = None) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.default_dossier_root = self.repo_root / "docs" / "evidence"
        self.resolved_headers = dict(resolved_headers or {})
        self.sources = self._load_sources(config_path=config_path)

    def enable_live_search(self, base_url: str, api_key: str, model_name: str = "gpt-4o") -> None:
        """Dynamically add a live_search evidence source backed by an LLM API."""
        if not base_url or not api_key:
            return
        already = any(s.source_type == "live_search" for s in self.sources)
        if already:
            return
        self.resolved_headers["api_key"] = api_key
        self.sources.insert(0, EvidenceSourceConfig(
            source_id="live_model_search",
            source_type="live_search",
            enabled=True,
            url=base_url,
            root=model_name,
            timeout_seconds=15,
            domains=["general"],
        ))

    def list_sources(self) -> list[dict[str, Any]]:
        return [
            {
                "source_id": item.source_id,
                "source_type": item.source_type,
                "enabled": item.enabled,
                "root": item.root,
                "url": item.url,
                "tags": list(item.tags),
                "domains": list(item.domains),
            }
            for item in self.sources
        ]

    def collect(
        self,
        query: str,
        limit: int = 6,
        domains: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        target_domains = [str(item).strip().lower() for item in (domains or []) if str(item).strip()]
        target_tags = [str(item).strip().lower() for item in (tags or []) if str(item).strip()]
        candidates: list[EvidenceRecord] = []
        source_hits: dict[str, int] = {}

        for source in self.sources:
            if not source.enabled:
                continue
            records = self._load_source_records(source=source, query=query)
            for record in records:
                scored = self._score_record(record=record, query=query, target_domains=target_domains, target_tags=target_tags)
                if scored.score <= 0.01:
                    continue
                candidates.append(scored)
                source_hits[scored.source_id] = source_hits.get(scored.source_id, 0) + 1

        candidates.sort(key=lambda item: item.score, reverse=True)
        selected = candidates[: max(1, limit)]
        citations = [item.url or item.path or item.title for item in selected]
        return {
            "query": query,
            "count": len(selected),
            "records": [item.to_dict() for item in selected],
            "citations": citations,
            "providers": [
                {"source_id": name, "records": count}
                for name, count in sorted(source_hits.items(), key=lambda item: item[1], reverse=True)
            ],
        }

    def _load_sources(self, config_path: str) -> list[EvidenceSourceConfig]:
        sources = [
            EvidenceSourceConfig(
                source_id="repo_dossiers",
                source_type="local_dossier",
                enabled=self.default_dossier_root.exists(),
                root=str(self.default_dossier_root),
                domains=["risk", "governance", "evidence", "research", "enterprise", "fintech"],
            ),
            EvidenceSourceConfig(
                source_id="built_in_catalog",
                source_type="static_catalog",
                enabled=True,
                domains=["risk", "governance", "evidence", "research"],
            ),
        ]

        if config_path:
            path = Path(config_path)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                extra = payload.get("sources", []) if isinstance(payload, dict) else []
                for item in extra:
                    if isinstance(item, dict):
                        sources.append(EvidenceSourceConfig.from_dict(item))
        return sources

    def _load_source_records(self, source: EvidenceSourceConfig, query: str) -> list[EvidenceRecord]:
        if source.source_type == "local_dossier":
            return self._load_local_dossier(source)
        if source.source_type == "http_json":
            return self._load_http_json(source=source, query=query)
        if source.source_type == "static_catalog":
            return self._load_static_catalog(source)
        if source.source_type == "live_search":
            return self._load_live_search(source=source, query=query)
        return []

    def _load_local_dossier(self, source: EvidenceSourceConfig) -> list[EvidenceRecord]:
        root = Path(source.root)
        if not root.exists():
            return []
        records: list[EvidenceRecord] = []
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            if path.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            record = self._record_from_file(source=source, path=path)
            if record:
                records.append(record)
        return records

    def _record_from_file(self, source: EvidenceSourceConfig, path: Path) -> EvidenceRecord | None:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            title = str(payload.get("title", path.stem))
            summary = str(payload.get("summary", payload.get("thesis", "")))
            content = str(payload.get("content", payload.get("notes", "")))
            tags = [str(x) for x in payload.get("tags", [])]
            domains = [str(x) for x in payload.get("domains", payload.get("domain", [] if payload.get("domain") is None else [payload.get("domain")]))]
            return EvidenceRecord(
                record_id=str(payload.get("record_id", path.stem)),
                title=title,
                summary=summary,
                source_id=source.source_id,
                source_type=source.source_type,
                evidence_type=str(payload.get("evidence_type", "dossier")),
                url=str(payload.get("url", "")),
                path=str(path),
                tags=tags,
                domains=domains,
                trust_score=_safe_float(payload.get("trust_score", 0.82), 0.82),
                freshness_hint=str(payload.get("freshness_hint", "stable")),
                content=content,
            )

        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        summary = lines[1] if len(lines) > 1 else path.stem
        return EvidenceRecord(
            record_id=path.stem,
            title=title,
            summary=summary,
            source_id=source.source_id,
            source_type=source.source_type,
            evidence_type="note",
            path=str(path),
            tags=list(source.tags),
            domains=list(source.domains),
            trust_score=0.7,
            freshness_hint="unknown",
            content="\n".join(lines[:8]),
        )

    def _load_http_json(self, source: EvidenceSourceConfig, query: str) -> list[EvidenceRecord]:
        if not source.url:
            return []
        params = parse.urlencode({source.query_param: query}) if source.query_param else ""
        url = source.url
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{params}"
        headers = {name: self.resolved_headers.get(env_key, "") for name, env_key in source.headers_env.items()}
        req = request.Request(url, headers={k: v for k, v in headers.items() if v})
        try:
            with request.urlopen(req, timeout=source.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception:
            return []
        rows = payload.get(source.records_key, []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
        if not isinstance(rows, list):
            return []
        records: list[EvidenceRecord] = []
        for idx, row in enumerate(rows[:40]):
            if not isinstance(row, dict):
                continue
            tags = row.get(source.tags_field, [])
            domains = row.get(source.domains_field, [])
            records.append(
                EvidenceRecord(
                    record_id=str(row.get("record_id", f"{source.source_id}-{idx}")),
                    title=str(row.get(source.title_field, f"{source.source_id}-{idx}")),
                    summary=str(row.get(source.summary_field, "")),
                    source_id=source.source_id,
                    source_type=source.source_type,
                    evidence_type=str(row.get("evidence_type", "remote_record")),
                    url=str(row.get(source.url_field, "")),
                    tags=[str(x) for x in (tags if isinstance(tags, list) else [tags])],
                    domains=[str(x) for x in (domains if isinstance(domains, list) else [domains])],
                    trust_score=_safe_float(row.get("trust_score", 0.74), 0.74),
                    freshness_hint=str(row.get("freshness_hint", "live")),
                    content=str(row.get("content", row.get("snippet", ""))),
                )
            )
        return records

    @staticmethod
    def _load_static_catalog(source: EvidenceSourceConfig) -> list[EvidenceRecord]:
        records = [
            {
                "record_id": "nist-ai-rmf",
                "title": "NIST AI Risk Management Framework",
                "summary": "Baseline framing for govern, map, measure, and manage controls in AI deployments.",
                "url": "https://www.nist.gov/itl/ai-risk-management-framework",
                "tags": ["risk", "governance", "controls", "framework"],
                "domains": ["risk", "governance"],
                "trust_score": 0.95,
            },
            {
                "record_id": "owasp-llm-top10",
                "title": "OWASP Top 10 for LLM Applications",
                "summary": "Practical failure modes and control categories for LLM application launches.",
                "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
                "tags": ["security", "llm", "risk", "controls"],
                "domains": ["risk", "security"],
                "trust_score": 0.9,
            },
            {
                "record_id": "mcp-architecture",
                "title": "Model Context Protocol Architecture",
                "summary": "Reference for interoperable agent-tool integration and external capability composition.",
                "url": "https://modelcontextprotocol.io/specification/2025-06-18/architecture/index",
                "tags": ["interop", "architecture", "tools", "protocol"],
                "domains": ["evidence", "research", "enterprise"],
                "trust_score": 0.88,
            },
            {
                "record_id": "langgraph-overview",
                "title": "LangGraph Overview",
                "summary": "Operational patterns for stateful agent execution, durability, and orchestration loops.",
                "url": "https://docs.langchain.com/oss/python/langgraph/overview",
                "tags": ["agents", "orchestration", "durability"],
                "domains": ["research", "enterprise"],
                "trust_score": 0.82,
            },
            {
                "record_id": "gaia-benchmark",
                "title": "GAIA Benchmark",
                "summary": "Public benchmark for evaluating general AI assistants on multi-step reasoning tasks.",
                "url": "https://huggingface.co/gaia-benchmark",
                "tags": ["benchmark", "research", "reasoning"],
                "domains": ["research", "evidence"],
                "trust_score": 0.9,
            },
            {
                "record_id": "swe-bench",
                "title": "SWE-bench",
                "summary": "Benchmark suite for measuring whether agents can resolve real GitHub issues with verifiable code changes.",
                "url": "https://github.com/SWE-bench/SWE-bench",
                "tags": ["benchmark", "engineering", "evaluation"],
                "domains": ["research", "enterprise"],
                "trust_score": 0.92,
            },
            {
                "record_id": "webarena",
                "title": "WebArena",
                "summary": "Benchmark for evaluating long-horizon web environment interaction by autonomous agents.",
                "url": "https://webarena.dev/",
                "tags": ["benchmark", "web", "agents"],
                "domains": ["research", "enterprise"],
                "trust_score": 0.88,
            },
            {
                "record_id": "tau-bench",
                "title": "tau-bench",
                "summary": "Enterprise-oriented benchmark for realistic tool-using agent tasks and long-horizon workflows.",
                "url": "https://github.com/sierra-research/tau-bench",
                "tags": ["benchmark", "enterprise", "workflow"],
                "domains": ["research", "enterprise"],
                "trust_score": 0.9,
            },
        ]
        return [
            EvidenceRecord(
                record_id=str(row["record_id"]),
                title=str(row["title"]),
                summary=str(row["summary"]),
                source_id=source.source_id,
                source_type=source.source_type,
                evidence_type="reference",
                url=str(row["url"]),
                tags=[str(x) for x in row["tags"]],
                domains=[str(x) for x in row["domains"]],
                trust_score=_safe_float(row.get("trust_score", 0.8), 0.8),
                freshness_hint="stable",
            )
            for row in records
        ]

    def _load_live_search(self, source: EvidenceSourceConfig, query: str) -> list[EvidenceRecord]:
        """Use the live model API to generate relevant evidence via function calling."""
        import os

        base_url = source.url or os.getenv("AGENT_HARNESS_MODEL_BASE_URL", "").strip()
        api_key = self.resolved_headers.get("api_key", "") or os.getenv("AGENT_HARNESS_MODEL_API_KEY", "").strip()
        model_name = source.root or os.getenv("AGENT_HARNESS_MODEL_NAME", "gpt-4o").strip()
        if not base_url or not api_key:
            return []

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Given a query, produce a JSON array of 5-8 relevant reference items. "
                    "Each item must have: title (str), url (str - real URL if known, otherwise empty), "
                    "summary (str - 1-2 sentence description of why this is relevant), "
                    "tags (list of str). Focus on authoritative, specific, and diverse sources. "
                    "Return ONLY the JSON array, no other text."
                ),
            },
            {"role": "user", "content": f"Find relevant references for: {query}"},
        ]

        payload = json.dumps({
            "model": model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1500,
        }).encode("utf-8")

        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/v1/chat/completions" if not endpoint.endswith("/v1") else "/chat/completions"

        req = request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=source.timeout_seconds or 15) as response:
                result = json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception:
            return []

        content = ""
        choices = result.get("choices", [])
        if choices and isinstance(choices[0], dict):
            content = choices[0].get("message", {}).get("content", "")

        # Parse the JSON array from the response
        content = content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        try:
            items = json.loads(content)
        except Exception:
            return []

        if not isinstance(items, list):
            return []

        records: list[EvidenceRecord] = []
        for idx, item in enumerate(items[:8]):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            tags = item.get("tags", [])
            records.append(
                EvidenceRecord(
                    record_id=f"live-search-{idx}",
                    title=title,
                    summary=str(item.get("summary", "")),
                    source_id=source.source_id,
                    source_type="live_search",
                    evidence_type="live_reference",
                    url=str(item.get("url", "")),
                    tags=[str(t) for t in (tags if isinstance(tags, list) else [])],
                    domains=list(source.domains),
                    trust_score=0.75,
                    freshness_hint="live",
                )
            )
        return records

    @staticmethod
    def _score_record(
        record: EvidenceRecord,
        query: str,
        target_domains: list[str],
        target_tags: list[str],
    ) -> EvidenceRecord:
        haystack = " ".join(
            [
                record.title.lower(),
                record.summary.lower(),
                " ".join(tag.lower() for tag in record.tags),
                " ".join(domain.lower() for domain in record.domains),
                record.content.lower(),
            ]
        )
        tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 3]
        overlap = sum(1 for token in tokens if token in haystack)
        domain_bonus = sum(1 for item in target_domains if item in [domain.lower() for domain in record.domains]) * 0.5
        tag_bonus = sum(1 for item in target_tags if item in [tag.lower() for tag in record.tags]) * 0.4
        score = min(5.0, overlap * 0.18 + domain_bonus + tag_bonus + record.trust_score * 0.45)
        return EvidenceRecord(**{**record.__dict__, "score": round(score, 4)})
