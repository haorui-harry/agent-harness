"""Harness tool adapters (API/browser/code-like capabilities)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from app.ecosystem.marketplace import discover_for_query, get_trending_skills
from app.skills.registry import list_all_skills

from app.harness.models import ToolCall, ToolResult, ToolType


class ToolRegistry:
    """Registry for harness-level tool calls."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[[dict[str, Any]], Any]] = {
            "api_market_discover": self._api_market_discover,
            "browser_trending_scan": self._browser_trending_scan,
            "code_skill_search": self._code_skill_search,
        }
        self._tool_types: dict[str, ToolType] = {
            "api_market_discover": ToolType.API,
            "browser_trending_scan": ToolType.BROWSER,
            "code_skill_search": ToolType.CODE,
        }

    def available_tools(self) -> list[str]:
        """List all registered tools."""

        return sorted(self._tools.keys())

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
            success = True
            error = ""
        except Exception as exc:  # pragma: no cover - defensive
            output = {}
            success = False
            error = str(exc)
        end = time.time()

        return ToolResult(
            name=tool_call.name,
            success=success,
            output=output,
            latency_ms=(end - start) * 1000.0,
            error=error,
        )

    @staticmethod
    def _api_market_discover(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        limit = int(args.get("limit", 3))
        return {"matches": discover_for_query(query=query, limit=limit)}

    @staticmethod
    def _browser_trending_scan(args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit", 3))
        return {"trending": get_trending_skills(limit=limit)}

    @staticmethod
    def _code_skill_search(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).lower()
        limit = int(args.get("limit", 8))
        matches = []
        for meta in list_all_skills():
            haystack = " ".join([meta.name, meta.description, " ".join(meta.confidence_keywords)]).lower()
            if query and query in haystack:
                matches.append(
                    {
                        "name": meta.name,
                        "category": meta.category.value,
                        "tier": meta.tier.value,
                        "cost": meta.compute_cost,
                    }
                )
        if not query:
            matches = [
                {
                    "name": meta.name,
                    "category": meta.category.value,
                    "tier": meta.tier.value,
                    "cost": meta.compute_cost,
                }
                for meta in list_all_skills()[:limit]
            ]
        return {"skills": matches[:limit]}
