"""Dynamic task understanding, channel deliberation, and graph compilation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.state import SkillCategory
from app.core.task_graph import ExecutableTaskGraph, TaskGraphNode
from app.skills.registry import list_all_skills


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(str(text or ""))]


def _count_markers(lowered: str, markers: list[str]) -> int:
    return sum(1 for marker in markers if marker in lowered)


def _slugify(parts: list[str], fallback: str = "task") -> str:
    text = "-".join(part.strip().lower() for part in parts if part.strip()) or fallback
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or fallback


@dataclass(frozen=True)
class SkillPrior:
    """A ranked skill prior used to bias planning without fixing the workflow."""

    name: str
    score: float
    rationale: list[str] = field(default_factory=list)
    category: str = ""
    tier: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "rationale": list(self.rationale),
            "category": self.category,
            "tier": self.tier,
        }


@dataclass(frozen=True)
class ChannelDeliberation:
    """Agent-style evidence-channel selection result."""

    scores: dict[str, float] = field(default_factory=dict)
    selected: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
            "selected": list(self.selected),
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class TaskProfile:
    """Shared task profile for planner, graph compiler, and runtime surfaces."""

    query: str
    target_hint: str
    evidence_strategy: str
    execution_intent: str
    output_mode: str
    reasoning_style: str
    domains: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    requires_workspace: bool = False
    requires_external_evidence: bool = False
    requires_discovery: bool = True
    requires_validation: bool = False
    requires_command_execution: bool = False
    artifact_targets: list[str] = field(default_factory=list)
    workspace_summary: dict[str, Any] = field(default_factory=dict)
    graph_expansion: dict[str, Any] = field(default_factory=dict)
    skill_priors: list[SkillPrior] = field(default_factory=list)
    deliberation: ChannelDeliberation = field(default_factory=ChannelDeliberation)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "target_hint": self.target_hint,
            "evidence_strategy": self.evidence_strategy,
            "execution_intent": self.execution_intent,
            "output_mode": self.output_mode,
            "reasoning_style": self.reasoning_style,
            "domains": list(self.domains),
            "keywords": list(self.keywords),
            "requires_workspace": self.requires_workspace,
            "requires_external_evidence": self.requires_external_evidence,
            "requires_discovery": self.requires_discovery,
            "requires_validation": self.requires_validation,
            "requires_command_execution": self.requires_command_execution,
            "artifact_targets": list(self.artifact_targets),
            "workspace_summary": dict(self.workspace_summary),
            "graph_expansion": dict(self.graph_expansion),
            "skill_priors": [item.to_dict() for item in self.skill_priors],
            "deliberation": self.deliberation.to_dict(),
            "selected_channels": list(self.deliberation.selected),
        }


def infer_domains(query: str) -> list[str]:
    """Infer topical domains from free-form task text."""

    lowered = str(query or "").lower()
    mapping = {
        "risk": [
            "risk",
            "security",
            "audit",
            "compliance",
            "governance",
            "control",
            "\u98ce\u63a7",
            "\u5b89\u5168",
            "\u5ba1\u8ba1",
            "\u5408\u89c4",
            "\u6cbb\u7406",
        ],
        "enterprise": [
            "enterprise",
            "workflow",
            "operations",
            "stakeholder",
            "board",
            "ops",
            "\u4f01\u4e1a",
            "\u6d41\u7a0b",
            "\u8fd0\u8425",
        ],
        "research": [
            "research",
            "benchmark",
            "experiment",
            "study",
            "evaluation",
            "paper",
            "\u7814\u7a76",
            "\u57fa\u51c6",
            "\u8bc4\u6d4b",
            "\u8bba\u6587",
            "\u5b9e\u9a8c",
        ],
        "fintech": [
            "fintech",
            "bank",
            "payments",
            "insurance",
            "customer support",
            "\u91d1\u878d",
            "\u94f6\u884c",
            "\u652f\u4ed8",
            "\u4fdd\u9669",
        ],
        "engineering": [
            "code",
            "repo",
            "repository",
            "workspace",
            "module",
            "patch",
            "test",
            "bug",
            "\u4ee3\u7801",
            "\u4ed3\u5e93",
            "\u6a21\u5757",
            "\u8865\u4e01",
            "\u6d4b\u8bd5",
            "\u7f3a\u9677",
        ],
    }
    domains = [domain for domain, markers in mapping.items() if any(marker in lowered for marker in markers)]
    return domains or ["general"]


def analyze_task_request(
    query: str,
    *,
    target: str = "general",
    workspace_root: str | Path | None = None,
    skill_limit: int = 4,
    live_model_overrides: dict[str, Any] | None = None,
) -> TaskProfile:
    """Infer a reusable task profile from a free-form request."""

    lowered = str(query or "").strip().lower()
    keywords = list(dict.fromkeys(_tokens(query)))[:8]
    target_hint = str(target or "general").strip().lower() or "general"

    workspace_markers = [
        "repo",
        "repository",
        "workspace",
        "file",
        "module",
        "code",
        "bug",
        "patch",
        "test",
        "refactor",
        "my repo",
        "this repo",
        "my repository",
        "\u4ed3\u5e93",
        "\u4ee3\u7801",
        "\u6587\u4ef6",
        "\u6a21\u5757",
        "\u8865\u4e01",
        "\u6d4b\u8bd5",
        "\u4fee\u590d",
    ]
    external_markers = [
        "research",
        "report",
        "topic",
        "market",
        "trend",
        "state of the art",
        "paper",
        "benchmark",
        "compare",
        "internet",
        "web",
        "latest",
        "\u7814\u7a76",
        "\u62a5\u544a",
        "\u8d8b\u52bf",
        "\u8bba\u6587",
        "\u8bc4\u6d4b",
        "\u5bf9\u6bd4",
        "\u7f51\u4e0a",
        "\u8c03\u7814",
    ]
    code_markers = [
        "patch",
        "test",
        "bug",
        "refactor",
        "fix",
        "implement",
        "validation",
        "execute",
        "run",
        "build",
        "\u4ee3\u7801",
        "\u8865\u4e01",
        "\u6d4b\u8bd5",
        "\u4fee\u590d",
        "\u5b9e\u73b0",
        "\u8fd0\u884c",
        "\u6784\u5efa",
    ]
    ops_markers = [
        "runbook",
        "incident",
        "ops",
        "workflow",
        "playbook",
        "governance",
        "policy",
        "escalation",
        "\u8fd0\u7ef4",
        "\u6d41\u7a0b",
        "\u6cbb\u7406",
        "\u7b56\u7565",
    ]
    benchmark_markers = [
        "benchmark",
        "ablation",
        "evaluation",
        "runner",
        "gaia",
        "swe-bench",
        "webarena",
        "\u03c4-bench",
        "tau-bench",
        "\u57fa\u51c6",
        "\u6d88\u878d",
        "\u8bc4\u6d4b",
        "\u8dd1\u5206",
    ]
    report_markers = ["report", "brief", "summary", "proposal", "\u65b9\u6848", "\u62a5\u544a", "\u603b\u7ed3"]
    runbook_markers = ["runbook", "playbook", "ops", "\u64cd\u4f5c\u624b\u518c", "\u9884\u6848"]
    patch_markers = ["patch", "fix", "implement", "\u4ee3\u7801\u4fee\u6539", "\u8865\u4e01", "\u4fee\u590d"]

    workspace_signal = _count_markers(lowered, workspace_markers)
    external_signal = _count_markers(lowered, external_markers)
    code_signal = _count_markers(lowered, code_markers)
    ops_signal = _count_markers(lowered, ops_markers)
    benchmark_signal = _count_markers(lowered, benchmark_markers)

    if target_hint == "code":
        workspace_signal += 2
        code_signal += 2
    elif target_hint == "research":
        external_signal += 2
    elif target_hint == "ops":
        ops_signal += 2

    execution_intent = "general"
    if benchmark_signal >= max(code_signal, ops_signal, 2):
        execution_intent = "benchmark"
    elif code_signal >= max(ops_signal, external_signal, 2):
        execution_intent = "code"
    elif ops_signal >= max(code_signal, external_signal, 2):
        execution_intent = "ops"
    elif external_signal >= 2:
        execution_intent = "research"
    elif workspace_signal and external_signal:
        execution_intent = "mixed"

    output_mode = "artifact"
    if any(marker in lowered for marker in patch_markers):
        output_mode = "patch"
    elif any(marker in lowered for marker in runbook_markers):
        output_mode = "runbook"
    elif benchmark_signal > 0:
        output_mode = "benchmark"
    elif any(marker in lowered for marker in report_markers):
        output_mode = "report"

    reasoning_style = "deliberate"
    if execution_intent == "benchmark":
        reasoning_style = "comparative"
    elif execution_intent == "code":
        reasoning_style = "debug"
    elif execution_intent == "research":
        reasoning_style = "evidence-led"
    elif execution_intent == "ops":
        reasoning_style = "procedural"

    workspace_summary = inspect_workspace_capabilities(workspace_root)
    skill_priors = select_skill_priors(
        query=query,
        execution_intent=execution_intent,
        output_mode=output_mode,
        limit=skill_limit,
    )
    local_deliberation = deliberate_channels(
        query=query,
        target=target_hint,
        execution_intent=execution_intent,
        output_mode=output_mode,
        skill_priors=skill_priors,
        workspace_root=workspace_root,
        workspace_signal=workspace_signal,
        external_signal=external_signal,
        code_signal=code_signal,
        benchmark_signal=benchmark_signal,
        ops_signal=ops_signal,
    )
    deliberation = refine_deliberation_with_live_model(
        query=query,
        target=target_hint,
        execution_intent=execution_intent,
        output_mode=output_mode,
        skill_priors=skill_priors,
        workspace_summary=workspace_summary,
        local=local_deliberation,
        live_model_overrides=live_model_overrides,
    )

    selected = set(deliberation.selected)
    requires_workspace = "workspace" in selected
    requires_external_evidence = "web" in selected
    requires_discovery = "discovery" in selected or not selected
    requires_validation = execution_intent in {"code", "research", "benchmark", "mixed"} or output_mode in {
        "patch",
        "benchmark",
    }
    requires_command_execution = (
        execution_intent in {"code", "benchmark"}
        and bool(workspace_summary.get("suggested_commands", []))
        and any(
            marker in lowered
            for marker in ["run", "execute", "build", "test", "validation", "\u8fd0\u884c", "\u6267\u884c", "\u6d4b\u8bd5"]
        )
    )

    if requires_workspace and requires_external_evidence:
        evidence_strategy = "hybrid"
    elif requires_workspace:
        evidence_strategy = "workspace"
    elif requires_external_evidence:
        evidence_strategy = "web"
    else:
        evidence_strategy = "minimal"

    graph_expansion = plan_graph_expansion(
        query=query,
        execution_intent=execution_intent,
        output_mode=output_mode,
        selected_channels=deliberation.selected,
        workspace_summary=workspace_summary,
        requires_command_execution=requires_command_execution,
        live_model_overrides=live_model_overrides,
        skill_priors=skill_priors,
    )

    return TaskProfile(
        query=query,
        target_hint=target_hint,
        evidence_strategy=evidence_strategy,
        execution_intent=execution_intent,
        output_mode=output_mode,
        reasoning_style=reasoning_style,
        domains=infer_domains(query),
        keywords=keywords,
        requires_workspace=requires_workspace,
        requires_external_evidence=requires_external_evidence,
        requires_discovery=requires_discovery,
        requires_validation=requires_validation,
        requires_command_execution=requires_command_execution,
        artifact_targets=default_artifact_targets(
            selected_channels=deliberation.selected,
            output_mode=output_mode,
            requires_validation=requires_validation,
            requires_command_execution=requires_command_execution,
        ),
        workspace_summary=workspace_summary,
        graph_expansion=graph_expansion,
        skill_priors=skill_priors,
        deliberation=deliberation,
    )


def select_skill_priors(
    query: str,
    *,
    execution_intent: str = "general",
    output_mode: str = "artifact",
    limit: int = 4,
) -> list[SkillPrior]:
    """Select best-fit skill priors without locking the runtime to one workflow."""

    lowered = str(query or "").lower()
    query_tokens = set(_tokens(query))
    intent_boosts = {
        "code": {"codebase_triage", "decompose_task", "validation_planner", "artifact_synthesis"},
        "research": {"research_brief", "extract_facts", "validate_claims", "artifact_synthesis"},
        "ops": {"ops_runbook", "identify_risks", "prioritize_items", "generate_recommendations"},
        "benchmark": {"benchmark_ablation", "research_brief", "validation_planner", "compare_options"},
        "mixed": {"decompose_task", "artifact_synthesis", "compare_options", "validation_planner"},
        "general": {"decompose_task", "artifact_synthesis", "executive_summary"},
    }
    output_boosts = {
        "patch": {"codebase_triage", "validation_planner"},
        "runbook": {"ops_runbook", "identify_risks"},
        "benchmark": {"benchmark_ablation", "validate_claims"},
        "report": {"research_brief", "extract_facts", "artifact_synthesis"},
        "artifact": {"artifact_synthesis", "executive_summary"},
    }

    priors: list[SkillPrior] = []
    for meta in list_all_skills():
        score = 0.0
        rationale: list[str] = []
        keyword_hits = [keyword for keyword in meta.confidence_keywords if keyword.lower() in lowered]
        if keyword_hits:
            score += 0.38 + 0.11 * len(keyword_hits[:3])
            rationale.append(f"matched keywords: {', '.join(keyword_hits[:3])}")
        if meta.name.lower() in lowered:
            score += 0.42
            rationale.append("skill name appears in query")
        desc_tokens = set(_tokens(meta.description))
        overlap = sorted(query_tokens & desc_tokens)
        if overlap:
            score += min(0.24, 0.06 * len(overlap[:4]))
            rationale.append(f"description overlap: {', '.join(overlap[:3])}")
        if meta.name in intent_boosts.get(execution_intent, set()):
            score += 0.32
            rationale.append(f"aligned with intent={execution_intent}")
        if meta.name in output_boosts.get(output_mode, set()):
            score += 0.18
            rationale.append(f"aligned with output={output_mode}")
        if meta.category in {SkillCategory.ANALYSIS, SkillCategory.REASONING}:
            score += 0.05
        if score <= 0.0:
            continue
        priors.append(
            SkillPrior(
                name=meta.name,
                score=score,
                rationale=rationale,
                category=meta.category.value,
                tier=meta.tier.value,
            )
        )

    priors.sort(key=lambda item: item.score, reverse=True)
    if priors:
        return priors[:limit]

    defaults = {
        "code": ["codebase_triage", "validation_planner", "decompose_task"],
        "research": ["research_brief", "extract_facts", "validate_claims"],
        "ops": ["ops_runbook", "identify_risks", "prioritize_items"],
        "benchmark": ["benchmark_ablation", "compare_options", "validation_planner"],
        "mixed": ["decompose_task", "artifact_synthesis", "validation_planner"],
        "general": ["decompose_task", "artifact_synthesis", "executive_summary"],
    }
    return [
        SkillPrior(name=name, score=0.25 - 0.02 * index, rationale=["intent fallback"])
        for index, name in enumerate(defaults.get(execution_intent, defaults["general"])[:limit])
    ]


def inspect_workspace_capabilities(workspace_root: str | Path | None) -> dict[str, Any]:
    """Summarize executable signals available in a workspace."""

    if not workspace_root:
        return {
            "exists": False,
            "root": "",
            "sample_files": [],
            "languages": [],
            "frameworks": [],
            "has_tests": False,
            "suggested_commands": [],
        }

    root = Path(workspace_root)
    if not root.exists():
        return {
            "exists": False,
            "root": str(root),
            "sample_files": [],
            "languages": [],
            "frameworks": [],
            "has_tests": False,
            "suggested_commands": [],
        }

    sample_files: list[str] = []
    suffixes: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        sample_files.append(rel)
        if path.suffix:
            suffixes.append(path.suffix.lower())
        if len(sample_files) >= 12:
            break

    languages: list[str] = []
    if any(item == ".py" for item in suffixes):
        languages.append("python")
    if any(item in {".js", ".ts", ".tsx", ".jsx"} for item in suffixes):
        languages.append("javascript")
    if any(item == ".rs" for item in suffixes):
        languages.append("rust")
    if any(item == ".go" for item in suffixes):
        languages.append("go")

    frameworks: list[str] = []
    if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists() or (root / "tests").exists():
        frameworks.append("pytest")
    if (root / "package.json").exists():
        frameworks.append("npm")
    if (root / "Cargo.toml").exists():
        frameworks.append("cargo")
    if (root / "go.mod").exists():
        frameworks.append("go")

    suggested_commands: list[str] = []
    tests_root = root / "tests"
    if tests_root.exists():
        targeted = [path.relative_to(root).as_posix() for path in sorted(tests_root.rglob("test_*")) if path.is_file()]
        suggested_commands.extend(f"pytest -q {item}" for item in targeted[:2])
        suggested_commands.append("pytest -q")
    if (root / "package.json").exists():
        suggested_commands.append("npm test")
    if (root / "Cargo.toml").exists():
        suggested_commands.append("cargo test")
    if (root / "go.mod").exists():
        suggested_commands.append("go test ./...")

    deduped_commands: list[str] = []
    for item in suggested_commands:
        if item not in deduped_commands:
            deduped_commands.append(item)

    return {
        "exists": True,
        "root": str(root),
        "sample_files": sample_files,
        "languages": languages,
        "frameworks": frameworks,
        "has_tests": bool(tests_root.exists()),
        "suggested_commands": deduped_commands[:5],
    }


def default_artifact_targets(
    *,
    selected_channels: list[str],
    output_mode: str,
    requires_validation: bool,
    requires_command_execution: bool,
) -> list[str]:
    """Infer artifact targets that make the graph materially reviewable."""

    targets = ["analysis_brief", "deliverable_report"]
    if "workspace" in selected_channels:
        targets.append("workspace_findings")
    if "web" in selected_channels:
        targets.append("evidence_bundle")
    if "risk" in selected_channels:
        targets.append("risk_register")
    if requires_validation:
        targets.append("validation_plan")
    if requires_command_execution:
        targets.append("execution_trace")
    if output_mode == "patch":
        targets.append("patch_plan")
    elif output_mode == "benchmark":
        targets.append("benchmark_matrix")
    elif output_mode == "runbook":
        targets.append("runbook")
    deduped: list[str] = []
    for item in targets:
        if item not in deduped:
            deduped.append(item)
    return deduped


def plan_graph_expansion(
    *,
    query: str,
    execution_intent: str,
    output_mode: str,
    selected_channels: list[str],
    workspace_summary: dict[str, Any],
    requires_command_execution: bool,
    live_model_overrides: dict[str, Any] | None,
    skill_priors: list[SkillPrior],
) -> dict[str, Any]:
    """Plan additional executable graph nodes beyond analysis/report writing."""

    local = _default_graph_expansion(
        execution_intent=execution_intent,
        output_mode=output_mode,
        selected_channels=selected_channels,
        workspace_summary=workspace_summary,
        requires_command_execution=requires_command_execution,
    )
    return refine_graph_expansion_with_live_model(
        query=query,
        execution_intent=execution_intent,
        output_mode=output_mode,
        workspace_summary=workspace_summary,
        skill_priors=skill_priors,
        local=local,
        live_model_overrides=live_model_overrides,
    )


def _default_graph_expansion(
    *,
    execution_intent: str,
    output_mode: str,
    selected_channels: list[str],
    workspace_summary: dict[str, Any],
    requires_command_execution: bool,
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    selected = set(selected_channels)

    if execution_intent in {"code", "mixed"} or output_mode == "patch":
        actions.append(
            {
                "kind": "patch_scaffold",
                "title": "Generate Patch Scaffold",
                "depends_on": ["analysis"],
                "reason": "code-oriented tasks benefit from an explicit patch scaffold",
            }
        )
        actions.append(
            {
                "kind": "patch_draft",
                "title": "Generate Patch Draft",
                "depends_on": ["analysis"],
                "reason": "code-oriented tasks benefit from a concrete draft patch artifact",
            }
        )
    if execution_intent == "benchmark" or output_mode == "benchmark":
        actions.append(
            {
                "kind": "benchmark_run_config",
                "title": "Generate Benchmark Run Config",
                "depends_on": ["analysis"],
                "reason": "benchmark tasks need executable configuration, not just narrative",
            }
        )
        actions.append(
            {
                "kind": "benchmark_manifest",
                "title": "Generate Benchmark Manifest",
                "depends_on": ["analysis"],
                "reason": "benchmark tasks need a portable manifest for reproducibility",
            }
        )
    if execution_intent in {"research", "benchmark", "mixed"} and "web" in selected:
        actions.append(
            {
                "kind": "dataset_pull_spec",
                "title": "Generate Dataset Pull Spec",
                "depends_on": ["analysis"],
                "reason": "external-evidence tasks benefit from a reproducible data collection spec",
            }
        )
        actions.append(
            {
                "kind": "dataset_loader_template",
                "title": "Generate Dataset Loader Template",
                "depends_on": ["analysis"],
                "reason": "external-evidence tasks benefit from a reusable loader template",
            }
        )
    if requires_command_execution and workspace_summary.get("suggested_commands"):
        actions.append(
            {
                "kind": "validation_execution",
                "title": "Execute Suggested Validation",
                "depends_on": ["analysis"],
                "reason": "workspace indicates concrete validation commands are available",
            }
        )

    deduped_actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in actions:
        kind = str(action.get("kind", "")).strip()
        if kind and kind not in seen:
            seen.add(kind)
            deduped_actions.append(action)

    return {
        "actions": deduped_actions,
        "replan_enabled": bool(deduped_actions or requires_command_execution),
        "replan_focus": ["execution", "artifacts", "validation"],
        "rationale": ["local graph expansion selected"],
        "source": "local",
    }


def refine_graph_expansion_with_live_model(
    *,
    query: str,
    execution_intent: str,
    output_mode: str,
    workspace_summary: dict[str, Any],
    skill_priors: list[SkillPrior],
    local: dict[str, Any],
    live_model_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Optionally refine graph expansion with a live model."""

    if not live_model_overrides:
        return local
    try:
        from app.harness.live_agent import CallBudget, LiveModelConfig, LiveModelGateway

        config = LiveModelConfig.from_overrides(live_model_overrides)
        if not config:
            return local
        gateway = LiveModelGateway(config)
        payload = {
            "query": query,
            "execution_intent": execution_intent,
            "output_mode": output_mode,
            "workspace_summary": workspace_summary,
            "skill_priors": [item.to_dict() for item in skill_priors[:4]],
            "local_expansion": local,
        }
        messages = [
            {
                "role": "system",
                "content": (
                "You are expanding a general agent task graph. "
                "Return strict JSON with keys: actions, replan_enabled, replan_focus, rationale. "
                    "Allowed action kinds: patch_scaffold, patch_draft, benchmark_run_config, benchmark_manifest, "
                    "dataset_pull_spec, dataset_loader_template, validation_execution. "
                    "Only propose actions that materially improve executable artifacts."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ]
        text, _meta = gateway.chat(
            messages=messages,
            budget=CallBudget(max_calls=1),
            temperature=0.0,
            require_json=True,
        )
        parsed = _coerce_json_dict(text)
        raw_actions = parsed.get("actions", [])
        actions: list[dict[str, Any]] = []
        for item in raw_actions if isinstance(raw_actions, list) else []:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "")).strip()
            if kind not in {
                "patch_scaffold",
                "patch_draft",
                "benchmark_run_config",
                "benchmark_manifest",
                "dataset_pull_spec",
                "dataset_loader_template",
                "validation_execution",
            }:
                continue
            actions.append(
                {
                    "kind": kind,
                    "title": str(item.get("title", kind.replace("_", " ").title())).strip(),
                    "depends_on": list(item.get("depends_on", ["analysis"])) if isinstance(item.get("depends_on", []), list) else ["analysis"],
                    "reason": str(item.get("reason", "live model expansion")).strip(),
                }
            )
        if not actions:
            return local
        replan_focus = [str(item).strip() for item in parsed.get("replan_focus", []) if str(item).strip()] if isinstance(parsed.get("replan_focus", []), list) else []
        rationale = list(local.get("rationale", []))
        rationale.append("live model refined graph expansion")
        for item in parsed.get("rationale", []) if isinstance(parsed.get("rationale", []), list) else []:
            text_item = str(item).strip()
            if text_item:
                rationale.append(text_item)
        return {
            "actions": actions,
            "replan_enabled": bool(parsed.get("replan_enabled", True)),
            "replan_focus": replan_focus or list(local.get("replan_focus", [])),
            "rationale": rationale[:8],
            "source": "live_model",
        }
    except Exception:
        return local


def refine_deliberation_with_live_model(
    *,
    query: str,
    target: str,
    execution_intent: str,
    output_mode: str,
    skill_priors: list[SkillPrior],
    workspace_summary: dict[str, Any],
    local: ChannelDeliberation,
    live_model_overrides: dict[str, Any] | None,
) -> ChannelDeliberation:
    """Optionally refine local channel selection with a live model."""

    if not live_model_overrides:
        return local

    try:
        from app.harness.live_agent import CallBudget, LiveModelConfig, LiveModelGateway

        config = LiveModelConfig.from_overrides(live_model_overrides)
        if not config:
            return local
        gateway = LiveModelGateway(config)
        payload = {
            "query": query,
            "target": target,
            "execution_intent": execution_intent,
            "output_mode": output_mode,
            "skill_priors": [item.to_dict() for item in skill_priors[:4]],
            "workspace_summary": workspace_summary,
            "local_deliberation": local.to_dict(),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are selecting evidence/action channels for a general agent task planner. "
                    "Return strict JSON with keys: selected_channels, rationale, channel_scores. "
                    "selected_channels must be an array chosen from workspace, web, discovery, risk. "
                    "Use workspace only when local artifacts are likely necessary. "
                    "Use web only when external evidence is likely necessary. "
                    "Prefer discovery for open-ended tasks."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ]
        text, _meta = gateway.chat(
            messages=messages,
            budget=CallBudget(max_calls=1),
            temperature=0.0,
            require_json=True,
        )
        parsed = _coerce_json_dict(text)
        raw_selected = parsed.get("selected_channels", [])
        selected = [str(item).strip() for item in raw_selected if str(item).strip() in {"workspace", "web", "discovery", "risk"}]
        if not selected:
            return local
        raw_scores = parsed.get("channel_scores", {})
        llm_scores = {
            key: _safe_float(raw_scores.get(key), local.scores.get(key, 0.0))
            for key in {"workspace", "web", "discovery", "risk"}
        }
        merged_scores = {
            key: round((local.scores.get(key, 0.0) + llm_scores.get(key, local.scores.get(key, 0.0))) / 2.0, 4)
            for key in {"workspace", "web", "discovery", "risk"}
        }
        rationale = list(local.rationale)
        rationale.append("live model refined channel selection")
        for item in parsed.get("rationale", []) if isinstance(parsed.get("rationale", []), list) else []:
            text_item = str(item).strip()
            if text_item:
                rationale.append(text_item)
        deduped_selected: list[str] = []
        for item in selected:
            if item not in deduped_selected:
                deduped_selected.append(item)
        return ChannelDeliberation(scores=merged_scores, selected=deduped_selected, rationale=rationale[:8])
    except Exception:
        return local


def deliberate_channels(
    *,
    query: str,
    target: str,
    execution_intent: str,
    output_mode: str,
    skill_priors: list[SkillPrior],
    workspace_root: str | Path | None,
    workspace_signal: int,
    external_signal: int,
    code_signal: int,
    benchmark_signal: int,
    ops_signal: int,
) -> ChannelDeliberation:
    """Choose evidence channels from priors and task constraints."""

    del query  # current deliberation is signal-based; query text already folded into signals

    prior_names = {item.name for item in skill_priors}
    workspace_prior_hits = len(
        prior_names & {"codebase_triage", "decompose_task", "validation_planner", "artifact_synthesis"}
    )
    web_prior_hits = len(prior_names & {"research_brief", "extract_facts", "validate_claims", "benchmark_ablation"})
    risk_prior_hits = len(prior_names & {"identify_risks", "ops_runbook", "validation_planner"})

    workspace_score = 0.12 + 0.11 * workspace_signal + 0.14 * workspace_prior_hits
    web_score = 0.12 + 0.11 * external_signal + 0.16 * web_prior_hits + 0.08 * benchmark_signal
    discovery_score = 0.34 + 0.05 * len(skill_priors)
    risk_score = 0.08 + 0.14 * ops_signal + 0.12 * risk_prior_hits

    if target == "code":
        workspace_score += 0.18
    if target == "research":
        web_score += 0.18
    if target == "ops":
        risk_score += 0.18
    if output_mode == "patch":
        workspace_score += 0.16
    if output_mode == "benchmark":
        web_score += 0.12
        workspace_score += 0.08
    if output_mode == "runbook":
        risk_score += 0.12
    if execution_intent == "mixed":
        workspace_score += 0.08
        web_score += 0.08
    if execution_intent == "code":
        workspace_score += 0.12
    if execution_intent == "research":
        web_score += 0.12
    if execution_intent == "benchmark":
        web_score += 0.14
        workspace_score += 0.06
    if execution_intent == "ops":
        risk_score += 0.12
    if workspace_root and Path(workspace_root).exists():
        discovery_score += 0.03

    scores = {
        "workspace": round(min(workspace_score, 1.0), 4),
        "web": round(min(web_score, 1.0), 4),
        "discovery": round(min(discovery_score, 1.0), 4),
        "risk": round(min(risk_score, 1.0), 4),
    }

    selected: list[str] = []
    rationale: list[str] = []
    if scores["discovery"] >= 0.35:
        selected.append("discovery")
        rationale.append("open-ended task warrants capability discovery before committing to a path")
    if scores["workspace"] >= 0.56 or (scores["workspace"] >= 0.44 and scores["workspace"] >= scores["web"]):
        selected.append("workspace")
        rationale.append("workspace selected because local artifacts appear decision-relevant")
    if scores["web"] >= 0.56 or (scores["web"] >= 0.44 and scores["web"] > scores["workspace"]):
        selected.append("web")
        rationale.append("web selected because external evidence is likely needed")
    if scores["risk"] >= 0.5:
        selected.append("risk")
        rationale.append("risk channel selected to constrain execution and validate governance")

    if not selected:
        selected = ["discovery"]
        rationale.append("no strong evidence channel won; defaulting to discovery-first planning")

    return ChannelDeliberation(scores=scores, selected=selected, rationale=rationale)


def build_dynamic_task_graph(
    query: str,
    *,
    target: str = "general",
    workspace_root: str | Path = ".",
    profile: TaskProfile | None = None,
    live_model_overrides: dict[str, Any] | None = None,
) -> tuple[TaskProfile, ExecutableTaskGraph]:
    """Compile an executable task graph from a deliberated task profile."""

    resolved = profile or analyze_task_request(
        query,
        target=target,
        workspace_root=workspace_root,
        live_model_overrides=live_model_overrides,
    )
    keywords = resolved.keywords or ["task"]
    slug = _slugify(keywords[:3], fallback=resolved.execution_intent or "task")

    report_title = {
        "patch": "Patch Preparation Report",
        "runbook": "Operational Runbook",
        "benchmark": "Benchmark Study",
        "report": "Research Report",
        "artifact": "Executable Task Artifact",
    }.get(resolved.output_mode, "Executable Task Artifact")
    report_path = f"reports/{slug}-{resolved.output_mode}-report.md"

    nodes: list[TaskGraphNode] = [
        TaskGraphNode(
            node_id="scope",
            title="Scope Task",
            node_type="routing",
            status="ready",
            notes=[query],
            metrics={
                "evidence_strategy": resolved.evidence_strategy,
                "execution_intent": resolved.execution_intent,
                "output_mode": resolved.output_mode,
                "reasoning_style": resolved.reasoning_style,
                "domains": list(resolved.domains),
                "keywords": list(keywords[:6]),
                "skill_priors": [item.name for item in resolved.skill_priors],
                "selected_channels": list(resolved.deliberation.selected),
                "channel_scores": dict(resolved.deliberation.scores),
                "channel_rationale": list(resolved.deliberation.rationale),
                "artifact_targets": list(resolved.artifact_targets),
                "workspace_summary": dict(resolved.workspace_summary),
                "graph_expansion": dict(resolved.graph_expansion),
            },
        )
    ]
    analysis_sources = ["scope"]
    selected = set(resolved.deliberation.selected)

    if "discovery" in selected:
        nodes.append(
            TaskGraphNode(
                node_id="capabilities",
                title="Discover Relevant Tools",
                node_type="tool_call",
                status="ready",
                depends_on=["scope"],
                metrics={"tool_name": "tool_search", "tool_args": {"query": query, "limit": 8}},
            )
        )
        analysis_sources.append("capabilities")

    if resolved.skill_priors:
        skill_query = " ".join(keywords[:3]) or resolved.execution_intent or query
        nodes.append(
            TaskGraphNode(
                node_id="skill_priors",
                title="Inspect Skill Priors",
                node_type="tool_call",
                status="ready",
                depends_on=["scope"],
                notes=[f"priors: {', '.join(item.name for item in resolved.skill_priors)}"],
                metrics={"tool_name": "code_skill_search", "tool_args": {"query": skill_query, "limit": 6}},
            )
        )
        analysis_sources.append("skill_priors")

    if "workspace" in selected:
        glob = "*.py" if resolved.execution_intent in {"code", "benchmark"} else "*"
        focus_query = keywords[0] if keywords else query
        nodes.extend(
            [
                TaskGraphNode(
                    node_id="workspace_scan",
                    title="Inspect Workspace",
                    node_type="workspace_snapshot",
                    status="ready",
                    depends_on=["scope"],
                    metrics={"area": "workspace", "glob": glob, "max_files": 20, "preview_limit": 6},
                ),
                TaskGraphNode(
                    node_id="workspace_focus",
                    title="Find Workspace Signals",
                    node_type="tool_call",
                    status="ready",
                    depends_on=["workspace_scan"],
                    metrics={
                        "tool_name": "workspace_file_search",
                        "tool_args": {"query": focus_query, "glob": glob, "limit": 8},
                    },
                ),
            ]
        )
        analysis_sources.extend(["workspace_scan", "workspace_focus"])
        nodes.append(
            TaskGraphNode(
                node_id="workspace_artifact",
                title="Write Workspace Findings",
                node_type="file_write",
                status="ready",
                depends_on=["workspace_focus"],
                metrics={
                    "area": "outputs",
                    "relative_path": f"artifacts/{slug}-workspace-findings.json",
                    "source_node_id": "workspace_focus",
                    "result_field": "",
                },
            )
        )

    if "web" in selected:
        domains = resolved.domains or ["general"]
        nodes.extend(
            [
                TaskGraphNode(
                    node_id="external_resources",
                    title="Collect External Resources",
                    node_type="tool_call",
                    status="ready",
                    depends_on=["scope"],
                    metrics={"tool_name": "external_resource_hub", "tool_args": {"query": query, "limit": 6}},
                ),
                TaskGraphNode(
                    node_id="evidence",
                    title="Build Evidence Dossier",
                    node_type="tool_call",
                    status="ready",
                    depends_on=["external_resources"],
                    metrics={
                        "tool_name": "evidence_dossier_builder",
                        "tool_args": {"query": query, "limit": 6, "domains": domains},
                    },
                ),
            ]
        )
        analysis_sources.extend(["external_resources", "evidence"])
        nodes.append(
            TaskGraphNode(
                node_id="evidence_artifact",
                title="Write Evidence Bundle",
                node_type="file_write",
                status="ready",
                depends_on=["evidence"],
                metrics={
                    "area": "outputs",
                    "relative_path": f"artifacts/{slug}-evidence-bundle.json",
                    "source_node_id": "evidence",
                    "result_field": "",
                },
            )
        )

    if "risk" in selected:
        nodes.append(
            TaskGraphNode(
                node_id="risk",
                title="Evaluate Risk and Governance",
                node_type="tool_call",
                status="ready",
                depends_on=["scope"],
                metrics={"tool_name": "policy_risk_matrix", "tool_args": {"query": query, "evidence_limit": 4}},
            )
        )
        analysis_sources.append("risk")
        nodes.append(
            TaskGraphNode(
                node_id="risk_artifact",
                title="Write Risk Register",
                node_type="file_write",
                status="ready",
                depends_on=["risk"],
                metrics={
                    "area": "outputs",
                    "relative_path": f"artifacts/{slug}-risk-register.json",
                    "source_node_id": "risk",
                    "result_field": "",
                },
            )
        )

    primary_skill = resolved.skill_priors[0].name if resolved.skill_priors else "decompose_task"
    nodes.append(
        TaskGraphNode(
            node_id="analysis",
            title="Analyze Task",
            node_type="skill_call",
            status="ready",
            depends_on=[node_id for node_id in analysis_sources if node_id != "scope"] or ["scope"],
            metrics={
                "skill_name": primary_skill,
                "source_node_ids": analysis_sources,
                "prompt": query,
            },
        )
    )
    nodes.append(
        TaskGraphNode(
            node_id="analysis_artifact",
            title="Write Analysis Brief",
            node_type="file_write",
            status="ready",
            depends_on=["analysis"],
            metrics={
                "area": "outputs",
                "relative_path": f"artifacts/{slug}-analysis.md",
                "source_node_id": "analysis",
                "result_field": "output",
                "content_prefix": "# Analysis Brief\n\n",
            },
        )
    )

    synthesis_sources = ["analysis"]
    if resolved.requires_validation:
        validation_skill = "benchmark_ablation" if resolved.execution_intent == "benchmark" else "validation_planner"
        validation_depends = ["analysis"]
        if "evidence" in analysis_sources:
            validation_depends.append("evidence")
        if "workspace_focus" in analysis_sources:
            validation_depends.append("workspace_focus")
        nodes.append(
            TaskGraphNode(
                node_id="validation",
                title="Plan Validation",
                node_type="skill_call",
                status="ready",
                depends_on=list(dict.fromkeys(validation_depends)),
                metrics={
                    "skill_name": validation_skill,
                    "source_node_ids": validation_depends,
                    "prompt": query,
                },
            )
        )
        synthesis_sources.append("validation")
        nodes.append(
            TaskGraphNode(
                node_id="validation_artifact",
                title="Write Validation Plan",
                node_type="file_write",
                status="ready",
                depends_on=["validation"],
                metrics={
                    "area": "outputs",
                    "relative_path": f"artifacts/{slug}-validation.md",
                    "source_node_id": "validation",
                    "result_field": "output",
                    "content_prefix": "# Validation Plan\n\n",
                },
            )
        )

    if resolved.requires_command_execution and resolved.workspace_summary.get("suggested_commands"):
        command_depends = ["analysis"]
        if "workspace_focus" in analysis_sources:
            command_depends.append("workspace_focus")
        if resolved.requires_validation:
            command_depends.append("validation")
        nodes.append(
            TaskGraphNode(
                node_id="execution",
                title="Execute Suggested Validation",
                node_type="command",
                status="ready",
                depends_on=list(dict.fromkeys(command_depends)),
                commands=list(resolved.workspace_summary.get("suggested_commands", [])),
                metrics={
                    "area": "workspace",
                    "timeout_seconds": 60,
                    "command_count": len(resolved.workspace_summary.get("suggested_commands", [])),
                },
            )
        )
        synthesis_sources.append("execution")

    expansion_action_ids: list[str] = []
    for action in resolved.graph_expansion.get("actions", []) if isinstance(resolved.graph_expansion.get("actions", []), list) else []:
        if not isinstance(action, dict):
            continue
        kind = str(action.get("kind", "")).strip()
        if kind == "validation_execution":
            continue
        action_id = f"action_{kind}"
        depends_on = [str(item) for item in action.get("depends_on", ["analysis"]) if str(item)]
        if "validation" in node_ids_from_nodes(nodes) and "validation" not in depends_on:
            depends_on.append("validation")
        nodes.append(
            TaskGraphNode(
                node_id=action_id,
                title=str(action.get("title", kind.replace("_", " ").title())).strip(),
                node_type="workspace_action",
                status="ready",
                depends_on=list(dict.fromkeys(depends_on)),
                metrics={
                    "action_kind": kind,
                    "prompt": query,
                    "source_node_ids": list(dict.fromkeys(depends_on)),
                    "workspace_summary": dict(resolved.workspace_summary),
                },
            )
        )
        expansion_action_ids.append(action_id)
        synthesis_sources.append(action_id)

    if bool(resolved.graph_expansion.get("replan_enabled", False)):
        replan_depends = list(dict.fromkeys(synthesis_sources))
        nodes.append(
            TaskGraphNode(
                node_id="replan",
                title="Replan Graph",
                node_type="graph_replan",
                status="ready",
                depends_on=replan_depends,
                metrics={
                    "prompt": query,
                    "source_node_ids": replan_depends,
                    "replan_focus": list(resolved.graph_expansion.get("replan_focus", [])),
                    "workspace_summary": dict(resolved.workspace_summary),
                    "graph_expansion_source": str(resolved.graph_expansion.get("source", "local")),
                },
            )
        )
        synthesis_sources = ["replan"]

    synthesis_skill = "artifact_synthesis"
    if resolved.output_mode == "runbook" or resolved.execution_intent == "ops":
        synthesis_skill = "ops_runbook"
    elif resolved.output_mode == "benchmark" or resolved.execution_intent == "benchmark":
        synthesis_skill = "benchmark_ablation"

    nodes.append(
        TaskGraphNode(
            node_id="synthesis",
            title="Synthesize Deliverable",
            node_type="skill_call",
            status="ready",
            depends_on=synthesis_sources,
            metrics={
                "skill_name": synthesis_skill,
                "source_node_ids": synthesis_sources,
                "prompt": query,
            },
        )
    )
    nodes.append(
        TaskGraphNode(
            node_id="report",
            title=f"Write {report_title}",
            node_type="file_write",
            status="ready",
            depends_on=["synthesis"],
            metrics={
                "area": "outputs",
                "relative_path": report_path,
                "source_node_id": "synthesis",
                "result_field": "output",
                "content_prefix": f"# {report_title}\n\n",
            },
        )
    )
    if resolved.requires_command_execution and resolved.workspace_summary.get("suggested_commands"):
        nodes.append(
            TaskGraphNode(
                node_id="execution_trace",
                title="Write Execution Trace",
                node_type="file_write",
                status="ready",
                depends_on=["execution"],
                metrics={
                    "area": "outputs",
                    "relative_path": f"artifacts/{slug}-execution-trace.json",
                    "source_node_id": "execution",
                    "result_field": "",
                },
            )
        )

    graph = ExecutableTaskGraph(
        graph_id=f"task-{slug}",
        mission_type=f"{resolved.execution_intent or 'general'}_task",
        query=query,
        nodes=nodes,
    )
    return resolved, graph


def _coerce_json_dict(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def node_ids_from_nodes(nodes: list[TaskGraphNode]) -> set[str]:
    return {item.node_id for item in nodes}
