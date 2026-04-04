"""Centralized runtime settings for thread-first harness execution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agents.runtime import THREADS_DIR
from app.agents.sandbox import (
    LocalThreadSandboxProvider,
    RemoteSandboxConfig,
    RemoteThreadSandboxProvider,
    ThreadSandboxProvider,
)
from app.harness.evidence import EvidenceProviderRegistry
from app.harness.live_agent import LiveModelConfig
from app.harness.state import DATA_FILE as MEMORY_DATA_FILE, HarnessMemoryStore


@dataclass(frozen=True)
class GatewayConfig:
    """Gateway/provider endpoint configuration used by runtime services."""

    base_url: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "headers": dict(self.headers),
        }


@dataclass(frozen=True)
class HarnessRuntimeSettings:
    """Single source of truth for harness runtime dependencies and endpoints."""

    threads_root: Path = THREADS_DIR
    memory_path: Path = MEMORY_DATA_FILE
    evidence_config_path: str = ""
    evidence_headers: dict[str, str] = field(default_factory=dict)
    sandbox_provider: str = "local"
    remote_sandbox: RemoteSandboxConfig | None = None
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    live_model: LiveModelConfig | None = None

    @staticmethod
    def from_env() -> "HarnessRuntimeSettings":
        """Resolve all environment-backed settings in one place."""

        threads_root = Path(os.getenv("AGENT_HARNESS_THREADS_ROOT", str(THREADS_DIR))).resolve()
        memory_path = Path(os.getenv("AGENT_HARNESS_MEMORY_PATH", str(MEMORY_DATA_FILE))).resolve()
        evidence_config_path = os.getenv("AGENT_HARNESS_EVIDENCE_CONFIG", "").strip()
        gateway_base_url = os.getenv("AGENT_HARNESS_GATEWAY_BASE_URL", "").strip()
        gateway_headers = HarnessRuntimeSettings._resolve_gateway_headers()
        evidence_headers = HarnessRuntimeSettings._resolve_evidence_headers(evidence_config_path)

        sandbox_base_url = os.getenv("AGENT_HARNESS_SANDBOX_BASE_URL", "").strip()
        sandbox_api_key = os.getenv("AGENT_HARNESS_SANDBOX_API_KEY", "").strip()
        remote_sandbox = None
        sandbox_provider = "local"
        if sandbox_base_url:
            sandbox_provider = "remote"
            remote_sandbox = RemoteSandboxConfig(
                base_url=sandbox_base_url,
                api_key=sandbox_api_key,
                timeout_seconds=max(5, min(int(os.getenv("AGENT_HARNESS_SANDBOX_TIMEOUT", "20")), 120)),
                require_https=os.getenv("AGENT_HARNESS_SANDBOX_REQUIRE_HTTPS", "true").strip().lower() not in {"0", "false", "no"},
            )

        return HarnessRuntimeSettings(
            threads_root=threads_root,
            memory_path=memory_path,
            evidence_config_path=evidence_config_path,
            evidence_headers=evidence_headers,
            sandbox_provider=sandbox_provider,
            remote_sandbox=remote_sandbox,
            gateway=GatewayConfig(base_url=gateway_base_url, headers=gateway_headers),
            live_model=LiveModelConfig.from_env(),
        )

    @staticmethod
    def _resolve_gateway_headers() -> dict[str, str]:
        raw = os.getenv("AGENT_HARNESS_GATEWAY_HEADERS", "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): str(value) for key, value in payload.items() if str(value).strip()}

    @staticmethod
    def _resolve_evidence_headers(config_path: str) -> dict[str, str]:
        if not config_path:
            return {}
        path = Path(config_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        sources = payload.get("sources", []) if isinstance(payload, dict) else []
        resolved: dict[str, str] = {}
        for source in sources:
            if not isinstance(source, dict):
                continue
            headers_env = source.get("headers_env", {})
            if not isinstance(headers_env, dict):
                continue
            for env_key in headers_env.values():
                key = str(env_key).strip()
                if key and key not in resolved:
                    resolved[key] = os.getenv(key, "")
        return resolved

    def build_sandbox_provider(self) -> ThreadSandboxProvider:
        """Instantiate the configured sandbox provider."""

        if self.sandbox_provider == "remote" and self.remote_sandbox:
            return RemoteThreadSandboxProvider(self.remote_sandbox)
        return LocalThreadSandboxProvider()

    def build_memory_store(self) -> HarnessMemoryStore:
        """Instantiate the configured memory store."""

        return HarnessMemoryStore(self.memory_path)

    def build_evidence_registry(self) -> EvidenceProviderRegistry:
        """Instantiate the configured evidence registry."""

        return EvidenceProviderRegistry(
            config_path=self.evidence_config_path,
            resolved_headers=self.evidence_headers,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "threads_root": str(self.threads_root),
            "memory_path": str(self.memory_path),
            "evidence_config_path": self.evidence_config_path,
            "sandbox_provider": self.sandbox_provider,
            "remote_sandbox": {
                "base_url": self.remote_sandbox.base_url,
                "timeout_seconds": self.remote_sandbox.timeout_seconds,
                "require_https": self.remote_sandbox.require_https,
            }
            if self.remote_sandbox
            else None,
            "gateway": self.gateway.to_dict(),
            "live_model": self.live_model.masked() if self.live_model else None,
        }
