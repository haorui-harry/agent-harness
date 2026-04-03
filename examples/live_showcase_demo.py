"""Run a real-model studio showcase and emit a readable summary bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.harness.engine import HarnessEngine
from app.harness.models import HarnessConstraints
from app.studio.flagship import StudioShowcaseBuilder


DEFAULT_QUERY = (
    "Generate a deep research and improvement report for agent-harness. "
    "Diagnose why it still falls short of a truly general agent framework, "
    "propose concrete runtime and architecture upgrades, define benchmark strategy, "
    "and produce a 90-day execution roadmap with measurable milestones."
)


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _resolve_live_model() -> dict[str, object]:
    base_url = _env("AGENT_HARNESS_MODEL_BASE_URL")
    api_key = _env("AGENT_HARNESS_MODEL_API_KEY")
    model_name = _env("AGENT_HARNESS_MODEL_NAME")
    if not (base_url and api_key and model_name):
        raise SystemExit(
            "Missing live model env. Set AGENT_HARNESS_MODEL_BASE_URL, "
            "AGENT_HARNESS_MODEL_API_KEY, and AGENT_HARNESS_MODEL_NAME."
        )
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
        "timeout_seconds": int(_env("AGENT_HARNESS_MODEL_TIMEOUT") or "120"),
        "temperature": float(_env("AGENT_HARNESS_MODEL_TEMPERATURE") or "0.2"),
        "max_tokens": int(_env("AGENT_HARNESS_MODEL_MAX_TOKENS") or "0"),
        "retry_attempts": int(_env("AGENT_HARNESS_MODEL_RETRY_ATTEMPTS") or "4"),
        "retry_backoff_seconds": float(_env("AGENT_HARNESS_MODEL_RETRY_BACKOFF") or "1.0"),
    }


def _write_summary(root: Path, payload: dict[str, object], paths: dict[str, object], query: str) -> Path:
    payload = json.loads(json.dumps(payload, default=str))
    identity = payload.get("identity", {}) if isinstance(payload.get("identity", {}), dict) else {}
    proposal = payload.get("proposal", {}) if isinstance(payload.get("proposal", {}), dict) else {}
    story = payload.get("story", {}) if isinstance(payload.get("story", {}), dict) else {}
    delivery = payload.get("delivery", {}) if isinstance(payload.get("delivery", {}), dict) else {}
    comparison = payload.get("comparison", {}).get("positioning", {}) if isinstance(payload.get("comparison", {}), dict) else {}
    why_use_this = payload.get("why_use_this", []) if isinstance(payload.get("why_use_this", []), list) else []
    release = payload.get("lab", {}).get("release_decision", {}) if isinstance(payload.get("lab", {}), dict) else {}
    generation = payload.get("harness", {}).get("generation", {}) if isinstance(payload.get("harness", {}), dict) else {}
    decision = proposal.get("decision", {}) if isinstance(proposal.get("decision", {}), dict) else {}
    business_summary = proposal.get("business_summary", []) if isinstance(proposal.get("business_summary", []), list) else []
    excerpt = str(delivery.get("delivery_brief_excerpt", "")).strip() or str(delivery.get("final_answer_excerpt", "")).strip()
    if not excerpt:
        excerpt = str(proposal.get("subheadline", "")).strip() or str(story.get("release_need", "")).strip()

    lines = [
        "# Live Showcase Summary",
        "",
        f"- Theme: {story.get('theme', '')}",
        f"- Product sentence: {identity.get('one_liner', '')}",
        f"- Query: {query}",
        f"- Release verdict: {decision.get('status', release.get('decision', ''))}",
        f"- Competitive headline: {comparison.get('headline', '')}",
        f"- Live generation mode: {generation.get('mode', '')}",
        f"- Live model: {generation.get('model', '')}",
        "",
        "## Proposal Summary",
        "",
        *(f"- {item}" for item in business_summary[:4]),
        "",
        "## Why This Demo Matters",
        "",
    ]
    lines.extend(f"- {item}" for item in why_use_this[:6])
    lines.extend(
        [
            "",
            "## Primary Result Excerpt",
            "",
            "```text",
            excerpt,
            "```",
            "",
            "## Artifact Paths",
            "",
            f"- HTML: {paths.get('html', '')}",
            f"- JSON: {paths.get('json', '')}",
            f"- Brief: {paths.get('brief', '')}",
            f"- Manifest: {paths.get('manifest', '')}",
        ]
    )
    summary_path = root / "live_showcase_summary.md"
    summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-model studio showcase demo.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Task/query to run")
    parser.add_argument("--output-dir", default="reports/live_showcase_demo", help="Output directory")
    parser.add_argument("--tag", default="live-agent-harness-report", help="Artifact tag")
    parser.add_argument("--mode", default="deep", help="Harness mode")
    parser.add_argument("--lab-preset", default="broad", help="Research-lab preset")
    parser.add_argument("--lab-repeats", type=int, default=1, help="Research-lab repeats")
    parser.add_argument("--max-live-calls", type=int, default=0, help="Live model call budget, 0 for unlimited")
    args = parser.parse_args()

    live_model = _resolve_live_model()
    constraints = HarnessConstraints(
        enable_live_agent=True,
        max_live_agent_calls=max(0, int(args.max_live_calls)),
        live_agent_timeout_seconds=int(live_model["timeout_seconds"]),
        live_agent_temperature=float(live_model["temperature"]),
    )

    builder = StudioShowcaseBuilder(harness=HarnessEngine())
    payload = builder.build_showcase(
        query=args.query,
        mode=args.mode,
        lab_preset=args.lab_preset,
        lab_repeats=max(1, int(args.lab_repeats)),
        include_marketplace=True,
        include_external=True,
        include_harness_tools=True,
        include_interop_catalog=True,
        constraints=constraints,
        live_model=live_model,
    )
    paths = builder.write_showcase(
        payload=payload,
        output_dir=args.output_dir,
        tag=args.tag,
        export_interop=True,
    )
    output_root = Path(args.output_dir)
    summary_path = _write_summary(output_root, payload, paths, args.query)
    print(
        json.dumps(
            {
                "theme": payload.get("identity", {}),
                "release_decision": payload.get("lab", {}).get("release_decision", {}),
                "generation": payload.get("harness", {}).get("generation", {}),
                "paths": {**paths, "summary": str(summary_path)},
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
