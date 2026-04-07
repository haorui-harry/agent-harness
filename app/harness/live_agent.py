"""Real-model gateway and orchestration loop for live agent enhancement."""

from __future__ import annotations

import json
import os
import re
import socket
import ssl
import time
from dataclasses import dataclass, field
from typing import Any
from urllib import error, request

from app.harness.live_strategy import LiveStrategyProfile, LiveStrategyRegistry


def _redact_endpoint_text(value: object) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = re.sub(r"https?://[^\s|)]+", "<redacted-endpoint>", text)
    text = re.sub(r"<redacted-endpoint>(?:/v\d+)?/chat/completions", "<redacted-endpoint>", text)
    return text


def _sanitize_transport_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "endpoint":
            continue
        if key == "retry_errors" and isinstance(value, list):
            cleaned[key] = [_redact_endpoint_text(item) for item in value]
            continue
        if isinstance(value, str):
            cleaned[key] = _redact_endpoint_text(value)
            continue
        if isinstance(value, list):
            cleaned[key] = [_redact_endpoint_text(item) if isinstance(item, str) else item for item in value]
            continue
        cleaned[key] = value
    return cleaned


@dataclass
class LiveModelConfig:
    """Runtime configuration for OpenAI-compatible model endpoints."""

    base_url: str
    api_key: str
    model_name: str
    timeout_seconds: int = 45
    temperature: float = 0.15
    # 0 means let the provider default decide instead of imposing a local cap.
    max_tokens: int = 0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 0.75

    @staticmethod
    def from_env() -> LiveModelConfig | None:
        base_url = os.getenv("AGENT_HARNESS_MODEL_BASE_URL", "").strip()
        api_key = os.getenv("AGENT_HARNESS_MODEL_API_KEY", "").strip()
        model_name = os.getenv("AGENT_HARNESS_MODEL_NAME", "").strip()
        if not (base_url and api_key and model_name):
            return None
        return LiveModelConfig(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=_safe_int(os.getenv("AGENT_HARNESS_MODEL_TIMEOUT", "45"), 45, minimum=5, maximum=300),
            temperature=_safe_float(
                os.getenv("AGENT_HARNESS_MODEL_TEMPERATURE", "0.15"),
                0.15,
                minimum=0.0,
                maximum=1.5,
            ),
            max_tokens=_optional_non_negative_int(os.getenv("AGENT_HARNESS_MODEL_MAX_TOKENS", "0"), 0),
            retry_attempts=_safe_int(os.getenv("AGENT_HARNESS_MODEL_RETRY_ATTEMPTS", "3"), 3, minimum=1, maximum=8),
            retry_backoff_seconds=_safe_float(
                os.getenv("AGENT_HARNESS_MODEL_RETRY_BACKOFF", "0.75"),
                0.75,
                minimum=0.1,
                maximum=10.0,
            ),
        )

    @staticmethod
    def from_overrides(
        overrides: dict[str, Any] | None,
        base: LiveModelConfig | None = None,
    ) -> LiveModelConfig | None:
        payload = dict(overrides or {})
        base_config = base
        if base_config:
            merged = {
                "base_url": base_config.base_url,
                "api_key": base_config.api_key,
                "model_name": base_config.model_name,
                "timeout_seconds": base_config.timeout_seconds,
                "temperature": base_config.temperature,
                "max_tokens": base_config.max_tokens,
                "retry_attempts": base_config.retry_attempts,
                "retry_backoff_seconds": base_config.retry_backoff_seconds,
            }
            merged.update({k: v for k, v in payload.items() if v not in ("", None)})
            payload = merged

        base_url = str(payload.get("base_url", "")).strip()
        api_key = str(payload.get("api_key", "")).strip()
        model_name = str(payload.get("model_name", "")).strip()
        if not (base_url and api_key and model_name):
            return None

        return LiveModelConfig(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=_safe_int(payload.get("timeout_seconds", 45), 45, minimum=5, maximum=300),
            temperature=_safe_float(payload.get("temperature", 0.15), 0.15, minimum=0.0, maximum=1.5),
            max_tokens=_optional_non_negative_int(payload.get("max_tokens", 0), 0),
            retry_attempts=_safe_int(payload.get("retry_attempts", 3), 3, minimum=1, maximum=8),
            retry_backoff_seconds=_safe_float(
                payload.get("retry_backoff_seconds", 0.75),
                0.75,
                minimum=0.1,
                maximum=10.0,
            ),
        )

    @staticmethod
    def resolve(
        overrides: dict[str, Any] | None,
        *,
        base: LiveModelConfig | None = None,
    ) -> LiveModelConfig | None:
        """Resolve live-model config from explicit base first, then the environment in one place."""

        return LiveModelConfig.from_overrides(overrides, base=base or LiveModelConfig.from_env())

    def masked(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "retry_attempts": self.retry_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "api_key_masked": _mask_secret(self.api_key),
        }


@dataclass
class CallBudget:
    """Per-run call budget guard."""

    max_calls: int
    used_calls: int = 0

    def consume(self) -> None:
        if self.max_calls > 0 and self.used_calls >= self.max_calls:
            raise RuntimeError(f"live_agent_call_budget_exhausted:{self.max_calls}")
        self.used_calls += 1

    @property
    def remaining(self) -> int:
        if self.max_calls <= 0:
            return 10**9
        return max(self.max_calls - self.used_calls, 0)


@dataclass
class LiveAgentResult:
    """Live-agent enhancement outcome."""

    enabled: bool = False
    configured: bool = False
    calls_used: int = 0
    call_budget: int = 0
    model: str = ""
    base_url: str = ""
    success: bool = False
    latency_ms: float = 0.0
    enhanced_answer: str = ""
    analysis: dict[str, Any] = field(default_factory=dict)
    critique: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    transport: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "calls_used": self.calls_used,
            "call_budget": self.call_budget,
            "model": self.model,
            "success": self.success,
            "latency_ms": round(self.latency_ms, 2),
            "analysis": self.analysis,
            "critique": self.critique,
            "strategy": self.strategy,
            "transport": {
                key: _sanitize_transport_payload(value) if isinstance(value, dict) else value
                for key, value in self.transport.items()
            },
            "notes": self.notes,
            "errors": [_redact_endpoint_text(item) for item in self.errors],
        }


class LiveModelGateway:
    """OpenAI-compatible chat completion gateway."""

    def __init__(self, config: LiveModelConfig) -> None:
        self.config = config

    def chat(
        self,
        messages: list[dict[str, str]],
        budget: CallBudget,
        temperature: float | None = None,
        max_tokens: int | None = None,
        require_json: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        budget.consume()
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else float(temperature),
        }
        effective_max_tokens = self.config.max_tokens if max_tokens is None else int(max_tokens)
        if effective_max_tokens > 0:
            payload["max_tokens"] = effective_max_tokens
        if require_json:
            payload["response_format"] = {"type": "json_object"}

        raw = json.dumps(payload).encode("utf-8")
        start = time.time()
        attempts = max(1, int(self.config.retry_attempts))
        endpoints = self._candidate_endpoints(self.config.base_url)
        retry_errors: list[str] = []
        for attempt in range(1, attempts + 1):
            for endpoint_index, endpoint in enumerate(endpoints):
                req = request.Request(
                    endpoint,
                    data=raw,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                try:
                    with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                        body = response.read().decode("utf-8", errors="replace")
                    latency_ms = (time.time() - start) * 1000.0
                    parsed = json.loads(body)
                    content = self._extract_content(parsed)
                    meta = {
                        "latency_ms": latency_ms,
                        "model": parsed.get("model", self.config.model_name),
                        "finish_reason": self._extract_finish_reason(parsed),
                        "usage": parsed.get("usage", {}),
                        "attempts": attempt,
                        "retries": max(0, attempt - 1),
                    }
                    if retry_errors:
                        meta["retry_errors"] = [_redact_endpoint_text(item) for item in retry_errors]
                    return content, meta
                except error.HTTPError as exc:
                    text = exc.read().decode("utf-8", errors="replace")
                    descriptor = f"http_error:{exc.code}:{_redact_endpoint_text(text[:500])}"
                    if endpoint_index < len(endpoints) - 1:
                        retry_errors.append(descriptor)
                        continue
                    if attempt < attempts and self._is_retryable_http_error(exc.code, text):
                        retry_errors.append(descriptor)
                        time.sleep(self._retry_delay(attempt))
                        continue
                    raise RuntimeError(self._finalize_error(descriptor, retry_errors, attempt)) from exc
                except (error.URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as exc:
                    descriptor = self._format_network_error(exc)
                    if endpoint_index < len(endpoints) - 1:
                        retry_errors.append(descriptor)
                        continue
                    if attempt < attempts and self._is_retryable_network_error(exc):
                        retry_errors.append(descriptor)
                        time.sleep(self._retry_delay(attempt))
                        continue
                    raise RuntimeError(self._finalize_error(descriptor, retry_errors, attempt)) from exc
                except json.JSONDecodeError as exc:
                    descriptor = f"json_decode_error:{str(exc)}"
                    if endpoint_index < len(endpoints) - 1:
                        retry_errors.append(descriptor)
                        continue
                    raise RuntimeError(self._finalize_error(descriptor, retry_errors, attempt)) from exc
                except Exception as exc:  # pragma: no cover - defensive
                    descriptor = f"live_gateway_error:{_redact_endpoint_text(exc)}"
                    if endpoint_index < len(endpoints) - 1:
                        retry_errors.append(descriptor)
                        continue
                    raise RuntimeError(self._finalize_error(descriptor, retry_errors, attempt)) from exc
        raise RuntimeError("live_gateway_error:exhausted_all_endpoints")

    def _retry_delay(self, attempt: int) -> float:
        base = max(0.1, float(self.config.retry_backoff_seconds))
        return min(base * (2 ** max(attempt - 1, 0)), 8.0)

    @staticmethod
    def _candidate_endpoints(base_url: str) -> list[str]:
        root = str(base_url or "").rstrip("/")
        if not root:
            return []
        if root.endswith("/chat/completions"):
            return [root]
        candidates = [root + "/chat/completions"]
        if not root.endswith("/v1"):
            candidates.insert(0, root + "/v1/chat/completions")
        return candidates

    @staticmethod
    def _finalize_error(descriptor: str, retry_errors: list[str], attempt: int) -> str:
        if not retry_errors:
            return descriptor
        prior = " | ".join(retry_errors[-3:])
        return f"{descriptor} after_attempt={attempt} retries={len(retry_errors)} prior={prior}"

    @staticmethod
    def _is_retryable_http_error(status_code: int, body: str) -> bool:
        if int(status_code) in {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}:
            return True
        lowered = body.lower()
        transient_markers = (
            "rate limit",
            "temporarily unavailable",
            "temporary",
            "timeout",
            "timed out",
            "overloaded",
            "try again",
            "upstream",
            "unavailable",
        )
        return any(marker in lowered for marker in transient_markers)

    @staticmethod
    def _is_retryable_network_error(exc: BaseException) -> bool:
        lowered = LiveModelGateway._network_reason_text(exc)
        transient_markers = (
            "timed out",
            "timeout",
            "tempor",
            "temporary",
            "reset",
            "closed",
            "eof",
            "handshake",
            "ssl",
            "refused",
            "unreachable",
            "connection aborted",
            "connection reset",
            "remote end closed",
            "remote disconnected",
        )
        return any(marker in lowered for marker in transient_markers)

    @staticmethod
    def _format_network_error(exc: BaseException) -> str:
        reason = LiveModelGateway._network_reason_text(exc)
        label = "url_error" if isinstance(exc, error.URLError) else "network_error"
        return f"{label}:{reason}"

    @staticmethod
    def _network_reason_text(exc: BaseException) -> str:
        if isinstance(exc, error.URLError):
            reason = exc.reason
            return str(reason)
        return str(exc)

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if choices and isinstance(choices, list):
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    out = []
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            out.append(item["text"])
                    if out:
                        return "\n".join(out)
            if isinstance(first.get("text"), str):
                return str(first["text"])
        if isinstance(payload.get("output_text"), str):
            return str(payload["output_text"])
        return ""

    @staticmethod
    def _extract_finish_reason(payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if choices and isinstance(choices, list) and isinstance(choices[0], dict):
            return str(choices[0].get("finish_reason", ""))
        return ""


class LiveAgentOrchestrator:
    """Three-stage agent loop: analyze -> synthesize -> critique."""

    def __init__(self) -> None:
        self.strategies = LiveStrategyRegistry()

    def enhance(
        self,
        query: str,
        mode: str,
        base_answer: str,
        plan: list[str],
        steps: list[dict[str, Any]],
        discovery: list[dict[str, Any]],
        evidence: dict[str, Any] | None = None,
        max_calls: int = 8,
        temperature: float = 0.15,
        live_model_overrides: dict[str, Any] | None = None,
        strategy: str = "",
        champion: dict[str, Any] | None = None,
        surface_guidance: str = "",
    ) -> LiveAgentResult:
        config = LiveModelConfig.resolve(live_model_overrides)
        profile, source, reason = self.strategies.resolve(
            query=query,
            mode=mode,
            preferred=strategy,
            champion=champion,
        )
        result = LiveAgentResult(
            enabled=True,
            configured=config is not None,
            call_budget=max(0, int(max_calls)),
            model=config.model_name if config else "",
            base_url=config.base_url if config else "",
            strategy={
                **profile.to_dict(),
                "source": source,
                "reason": reason,
            },
        )
        if not config:
            result.errors.append("live_model_not_configured")
            result.notes.append(
                "Set AGENT_HARNESS_MODEL_BASE_URL / AGENT_HARNESS_MODEL_API_KEY / AGENT_HARNESS_MODEL_NAME."
            )
            return result

        budget = CallBudget(max_calls=result.call_budget)
        gateway = LiveModelGateway(config)
        start = time.time()
        try:
            analysis_text, analysis_meta = gateway.chat(
                messages=self._analysis_messages(
                    query,
                    mode,
                    plan,
                    steps,
                    discovery,
                    evidence or {},
                    profile=profile,
                    surface_guidance=surface_guidance,
                ),
                budget=budget,
                temperature=max(0.0, min(1.5, temperature + profile.temperature_bias)),
                require_json=True,
            )
            result.analysis = _coerce_json(analysis_text)
            result.transport["analysis"] = analysis_meta
            if analysis_meta.get("finish_reason"):
                result.notes.append(f"analysis_finish:{analysis_meta['finish_reason']}")
            if int(analysis_meta.get("retries", 0)) > 0:
                result.notes.append(f"analysis_retries:{analysis_meta['retries']}")

            synth_text = base_answer
            if budget.remaining >= 1:
                synth_text, synth_meta = gateway.chat(
                    messages=self._synthesis_messages(
                        query,
                        mode,
                        base_answer,
                        plan,
                        steps,
                        result.analysis,
                        evidence or {},
                        profile=profile,
                        surface_guidance=surface_guidance,
                    ),
                    budget=budget,
                    temperature=max(0.0, min(1.5, temperature + profile.temperature_bias)),
                    require_json=False,
                )
                result.transport["synthesis"] = synth_meta
                if synth_meta.get("finish_reason"):
                    result.notes.append(f"synthesis_finish:{synth_meta['finish_reason']}")
                if int(synth_meta.get("retries", 0)) > 0:
                    result.notes.append(f"synthesis_retries:{synth_meta['retries']}")
            else:
                result.notes.append("synthesis_skipped_budget")

            critique_payload: dict[str, Any] = {}
            if budget.remaining >= 1:
                critique_text, critique_meta = gateway.chat(
                    messages=self._critique_messages(
                        query,
                        synth_text,
                        evidence or {},
                        profile=profile,
                        surface_guidance=surface_guidance,
                    ),
                    budget=budget,
                    temperature=max(0.0, min(1.5, temperature + profile.temperature_bias - 0.05)),
                    require_json=True,
                )
                critique_payload = _coerce_json(critique_text)
                result.transport["critique"] = critique_meta
                if critique_meta.get("finish_reason"):
                    result.notes.append(f"critique_finish:{critique_meta['finish_reason']}")
                if int(critique_meta.get("retries", 0)) > 0:
                    result.notes.append(f"critique_retries:{critique_meta['retries']}")

            revised_text = synth_text
            if budget.remaining >= 1:
                revised_text, revision_meta = gateway.chat(
                    messages=self._revision_messages(
                        query=query,
                        synthesized=synth_text,
                        analysis=result.analysis,
                        critique=critique_payload,
                        evidence=evidence or {},
                        profile=profile,
                        surface_guidance=surface_guidance,
                    ),
                    budget=budget,
                    temperature=max(0.0, min(1.5, temperature + profile.temperature_bias - 0.02)),
                    require_json=False,
                )
                result.transport["revision"] = revision_meta
                if revision_meta.get("finish_reason"):
                    result.notes.append(f"revision_finish:{revision_meta['finish_reason']}")
                if int(revision_meta.get("retries", 0)) > 0:
                    result.notes.append(f"revision_retries:{revision_meta['retries']}")
            else:
                result.notes.append("revision_skipped_budget")

            result.critique = critique_payload
            result.enhanced_answer = revised_text.strip() or synth_text.strip() or base_answer
            result.success = bool(result.enhanced_answer)
        except Exception as exc:
            result.errors.append(str(exc))
            result.enhanced_answer = base_answer
            result.success = False

        result.calls_used = budget.used_calls
        result.latency_ms = (time.time() - start) * 1000.0
        return result

    @staticmethod
    def _analysis_messages(
        query: str,
        mode: str,
        plan: list[str],
        steps: list[dict[str, Any]],
        discovery: list[dict[str, Any]],
        evidence: dict[str, Any],
        profile: LiveStrategyProfile,
        surface_guidance: str = "",
    ) -> list[dict[str, str]]:
        step_digest = []
        for item in steps[:8]:
            step_digest.append(
                {
                    "step": item.get("step", 0),
                    "tool": item.get("tool_call", {}).get("name", ""),
                    "success": item.get("tool_result", {}).get("success", False),
                }
            )
        prompt = {
            "query": query,
            "mode": mode,
            "plan": plan,
            "step_digest": step_digest,
            "top_discovery": discovery[:6],
            "evidence_digest": LiveAgentOrchestrator._evidence_digest(evidence),
        }
        return [
            {
                "role": "system",
                "content": (
                    f"{profile.analysis_system} "
                    "Before producing your analysis, think step by step about: "
                    "1) What exactly is the user asking for? "
                    "2) What evidence from evidence_digest actually supports an answer? "
                    "3) What is missing that you must explicitly flag as a gap? "
                    "Use the evidence_digest as the boundary of what is actually supported. "
                    "If the evidence_digest does not contain a quantitative claim, do not infer one. "
                    + (f"Surface guidance: {surface_guidance}" if surface_guidance else "")
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
        ]

    @staticmethod
    def _synthesis_messages(
        query: str,
        mode: str,
        base_answer: str,
        plan: list[str],
        steps: list[dict[str, Any]],
        analysis: dict[str, Any],
        evidence: dict[str, Any],
        profile: LiveStrategyProfile,
        surface_guidance: str = "",
    ) -> list[dict[str, str]]:
        payload = {
            "query": query,
            "mode": mode,
            "plan": plan,
            "analysis": analysis,
            "tool_steps": steps[:8],
            "evidence_digest": LiveAgentOrchestrator._evidence_digest(evidence),
            "base_answer": base_answer[:6000],
        }
        return [
            {
                "role": "system",
                "content": (
                    f"{profile.synthesis_system} "
                    "IMPORTANT: The 'analysis' field contains structured findings from the first reasoning pass. "
                    "You MUST use these findings as the backbone of your answer — do not ignore them and start from scratch. "
                    "Use the evidence_digest to ground concrete claims. "
                    "Prefer specific details, names, and implementation actions over generic phases. "
                    "When evidence sources are present, end with a short 'Sources' section listing them. "
                    "Do not invent numbers, benchmark scores, or study results unless they appear in evidence_digest or base_answer. "
                    "If the evidence is qualitative, write qualitative claims and explicitly state evidence limits. "
                    + (f"Surface guidance: {surface_guidance}" if surface_guidance else "")
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ]

    @staticmethod
    def _critique_messages(
        query: str,
        synthesized: str,
        evidence: dict[str, Any],
        profile: LiveStrategyProfile,
        surface_guidance: str = "",
    ) -> list[dict[str, str]]:
        payload = {
            "query": query,
            "candidate_answer": synthesized[:5000],
            "evidence_digest": LiveAgentOrchestrator._evidence_digest(evidence),
        }
        return [
            {
                "role": "system",
                "content": (
                    f"{profile.critique_system} "
                    "Flag unsupported quantitative claims, invented benchmark results, and source references not present in evidence_digest. "
                    + (f"Surface guidance: {surface_guidance}" if surface_guidance else "")
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ]

    @staticmethod
    def _revision_messages(
        query: str,
        synthesized: str,
        analysis: dict[str, Any],
        critique: dict[str, Any],
        evidence: dict[str, Any],
        profile: LiveStrategyProfile,
        surface_guidance: str = "",
    ) -> list[dict[str, str]]:
        # Extract specific critique items to inject as mandatory fix targets
        red_flags = critique.get("red_flags", [])
        blind_spots = critique.get("blind_spots", [])
        improve = critique.get("improve", [])
        confidence = critique.get("confidence", 1.0)

        critique_instructions = []
        if red_flags:
            critique_instructions.append(
                "RED FLAGS (must fix): " + "; ".join(str(f) for f in red_flags[:5])
            )
        if blind_spots:
            critique_instructions.append(
                "BLIND SPOTS (must address): " + "; ".join(str(b) for b in blind_spots[:5])
            )
        if improve:
            critique_instructions.append(
                "IMPROVEMENTS (must apply): " + "; ".join(str(i) for i in improve[:5])
            )
        try:
            conf_val = float(confidence)
        except (TypeError, ValueError):
            conf_val = 0.5  # non-numeric confidence = treat as uncertain
        if conf_val < 0.7:
            critique_instructions.append(
                f"Critique confidence is LOW ({confidence}). Significantly strengthen evidence grounding."
            )

        critique_block = "\n".join(critique_instructions) if critique_instructions else "No specific critique issues found."

        payload = {
            "query": query,
            "candidate_answer": synthesized[:9000],
            "analysis": analysis,
            "critique": critique,
            "evidence_digest": LiveAgentOrchestrator._evidence_digest(evidence),
        }
        return [
            {
                "role": "system",
                "content": (
                    f"{profile.synthesis_system} "
                    "You are revising an existing answer after peer review. "
                    "The critique identified SPECIFIC issues that you MUST fix in this revision:\n\n"
                    f"{critique_block}\n\n"
                    "For each red flag and blind spot, either fix it or explicitly explain why it's not applicable. "
                    "Do NOT just rephrase the original — make substantive improvements. "
                    "Preserve strong sections, sharpen weak ones, and ground every claim in the evidence_digest. "
                    "If evidence_digest includes sources, include a concise 'Sources' section. "
                    "Delete any unsupported metrics or benchmark claims not in evidence_digest. "
                    + (f"Surface guidance: {surface_guidance}" if surface_guidance else "")
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ]

    @staticmethod
    def _evidence_digest(evidence: dict[str, Any]) -> dict[str, Any]:
        records = evidence.get("records", []) if isinstance(evidence.get("records", []), list) else []
        citations = evidence.get("citations", []) if isinstance(evidence.get("citations", []), list) else []
        sources = evidence.get("sources", []) if isinstance(evidence.get("sources", []), list) else []
        digest_rows: list[dict[str, Any]] = []
        for item in records[:6]:
            if not isinstance(item, dict):
                continue
            digest_rows.append(
                {
                    "title": str(item.get("title", "")),
                    "summary": str(item.get("summary", "")),
                    "content": str(item.get("content", ""))[:300],
                    "source_id": str(item.get("source_id", "")),
                    "url": str(item.get("url", item.get("path", ""))),
                }
            )
        return {
            "record_count": int(evidence.get("record_count", len(records))) if isinstance(evidence, dict) else len(records),
            "citation_count": int(evidence.get("citation_count", len(citations))) if isinstance(evidence, dict) else len(citations),
            "top_records": digest_rows,
            "source_titles": [str(item.get("title", "")) for item in digest_rows if str(item.get("title", ""))],
            "citations": [str(item) for item in citations[:8]],
            "sources": sources[:6],
        }


def _coerce_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", stripped)
    if match:
        block = match.group(0)
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"raw": stripped}
    return {"raw": stripped}


def _mask_secret(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}***{secret[-4:]}"


def _safe_int(value: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        num = int(value)
    except Exception:
        num = fallback
    return max(minimum, min(maximum, num))


def _optional_non_negative_int(value: Any, fallback: int = 0) -> int:
    try:
        num = int(value)
    except Exception:
        num = fallback
    return max(0, num)


def _safe_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    try:
        num = float(value)
    except Exception:
        num = fallback
    return max(minimum, min(maximum, num))
