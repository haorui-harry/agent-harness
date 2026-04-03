"""Data models for harness planning/execution/evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolType(str, Enum):
    """Supported harness tool families."""

    API = "api"
    BROWSER = "browser"
    CODE = "code"


@dataclass
class ToolCall:
    """One planned tool invocation."""

    name: str
    tool_type: ToolType
    args: dict[str, Any] = field(default_factory=dict)
    source: str = "planner"
    score: float = 0.0


@dataclass
class ToolResult:
    """Result payload of a tool call."""

    name: str
    success: bool
    output: Any
    latency_ms: float
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessConstraints:
    """Execution constraints and guardrails for harness loop."""

    max_steps: int = 4
    max_tool_calls: int = 4
    allow_write_actions: bool = False
    allow_network_actions: bool = True
    allow_browser_actions: bool = True
    allow_code_execution: bool = True
    require_approval_on_high_risk: bool = True
    enable_security_scan: bool = True
    enable_dynamic_discovery: bool = True
    auto_recipe: bool = True
    security_strictness: str = "balanced"
    enable_live_agent: bool = False
    max_live_agent_calls: int = 8
    # 0 means unlimited and lets the runtime stop on task quality rather than an arbitrary call cap.
    live_agent_temperature: float = 0.15
    live_agent_timeout_seconds: int = 45
    live_agent_fail_open: bool = True
    blocked_tools: list[str] = field(default_factory=lambda: ["unsafe_write", "delete_path"])


@dataclass
class HarnessStep:
    """One loop step including thought/decision/tool/action."""

    step: int
    thought: str
    decision: str
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    guardrail_notes: list[str] = field(default_factory=list)
    discovery_notes: list[str] = field(default_factory=list)
    security: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessRun:
    """End-to-end harness run artifact."""

    query: str
    plan: list[str]
    steps: list[HarnessStep]
    final_answer: str
    completed: bool
    eval_metrics: dict[str, float]
    memory_snapshot: list[dict[str, Any]] = field(default_factory=list)
    mission: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
