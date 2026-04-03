"""Tests for live gateway retry behavior."""

from __future__ import annotations

import io
import json
from urllib import error

from app.harness.live_agent import CallBudget, LiveModelConfig, LiveModelGateway


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_live_gateway_retries_transient_url_errors(monkeypatch) -> None:
    config = LiveModelConfig(
        base_url="https://example.com/v1",
        api_key="secret",
        model_name="demo-model",
        retry_attempts=3,
        retry_backoff_seconds=0.01,
    )
    gateway = LiveModelGateway(config)
    calls = {"count": 0}

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        if calls["count"] == 1:
            raise error.URLError("timed out")
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 12},
            }
        )

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)
    text, meta = gateway.chat(
        messages=[{"role": "user", "content": "hello"}],
        budget=CallBudget(max_calls=2),
    )

    assert text == "ok"
    assert calls["count"] == 2
    assert meta["retries"] == 1
    assert meta["attempts"] == 2
    assert "retry_errors" in meta


def test_live_gateway_does_not_retry_non_retryable_http_errors(monkeypatch) -> None:
    config = LiveModelConfig(
        base_url="https://example.com/v1",
        api_key="secret",
        model_name="demo-model",
        retry_attempts=3,
        retry_backoff_seconds=0.01,
    )
    gateway = LiveModelGateway(config)
    calls = {"count": 0}

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        raise error.HTTPError(
            url="https://example.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"bad api key"}'),
        )

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)

    try:
        gateway.chat(
            messages=[{"role": "user", "content": "hello"}],
            budget=CallBudget(max_calls=2),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")

    assert calls["count"] == 1
    assert "http_error:401" in message


def test_live_gateway_unlimited_budget_and_provider_default_tokens(monkeypatch) -> None:
    config = LiveModelConfig(
        base_url="https://example.com/v1",
        api_key="secret",
        model_name="demo-model",
        max_tokens=0,
    )
    gateway = LiveModelGateway(config)
    seen: dict[str, object] = {}

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "model": "demo-model",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 9},
            }
        )

    monkeypatch.setattr("app.harness.live_agent.request.urlopen", fake_urlopen)

    budget = CallBudget(max_calls=0)
    first, _ = gateway.chat(messages=[{"role": "user", "content": "first"}], budget=budget)
    second, _ = gateway.chat(messages=[{"role": "user", "content": "second"}], budget=budget)

    assert first == "ok"
    assert second == "ok"
    assert budget.used_calls == 2
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert "max_tokens" not in payload
