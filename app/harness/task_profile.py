"""Dynamic task understanding, channel deliberation, and graph compilation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.tasking import allowed_workspace_action_kinds, default_capability_registry, infer_task_spec, plan_capability_path
from app.core.state import SkillCategory
from app.core.task_graph import ExecutableTaskGraph, TaskGraphNode
from app.skills.packages import SkillPackageCatalog
from app.skills.registry import list_all_skills


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}")


def _tokens(text: str) -> list[str]:
    raw_tokens = [token.lower() for token in _TOKEN_RE.findall(str(text or ""))]
    expanded: list[str] = []
    for token in raw_tokens:
        if token not in expanded:
            expanded.append(token)
        if "-" in token or "_" in token:
            for part in re.split(r"[-_]+", token):
                normalized = part.strip().lower()
                if len(normalized) >= 2 and normalized not in expanded:
                    expanded.append(normalized)
    return expanded


def _count_markers(lowered: str, markers: list[str]) -> int:
    tokens = set(_tokens(lowered))
    total = 0
    for marker in markers:
        marker_text = str(marker or "").lower()
        if not marker_text:
            continue
        if re.fullmatch(r"[a-z0-9_-]+", marker_text):
            if marker_text in tokens:
                total += 1
            continue
        if marker_text in lowered:
            total += 1
    return total


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
    task_spec: dict[str, Any] = field(default_factory=dict)
    capability_plan: dict[str, Any] = field(default_factory=dict)
    graph_expansion: dict[str, Any] = field(default_factory=dict)
    skill_priors: list[SkillPrior] = field(default_factory=list)
    package_priors: list[dict[str, Any]] = field(default_factory=list)
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
            "task_spec": dict(self.task_spec),
            "capability_plan": dict(self.capability_plan),
            "graph_expansion": dict(self.graph_expansion),
            "skill_priors": [item.to_dict() for item in self.skill_priors],
            "package_priors": [dict(item) for item in self.package_priors],
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
    package_catalog = SkillPackageCatalog()
    package_priors = []
    for item in package_catalog.suggest(query, target=target_hint, limit=max(4, skill_limit)):
        payload = item.to_dict()
        payload["match_score"] = round(item.score_for_query(query, target=target_hint), 4)
        package_priors.append(payload)
    package_channels = _required_channels_from_packages(package_priors)
    package_validation = any("validation" in [str(req).strip().lower() for req in item.get("runtime_requirements", [])] for item in package_priors)

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
        "tests",
        "bug",
        "refactor",
        "fix",
        "fixes",
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
    webpage_markers = [
        "webpage",
        "website",
        "landing page",
        "landing",
        "html",
        "frontend",
        "web app",
        "component",
        "\u7f51\u9875",
        "\u7f51\u7ad9",
        "\u524d\u7aef",
        "\u754c\u9762",
        "\u9875\u9762",
    ]
    slides_markers = [
        "slides",
        "slide",
        "deck",
        "presentation",
        "ppt",
        "pptx",
        "keynote",
        "\u5e7b\u706f",
        "\u6f14\u793a",
        "\u6c47\u62a5",
    ]
    chart_markers = [
        "chart",
        "charts",
        "graph",
        "graphs",
        "plot",
        "visualization",
        "visualize",
        "data viz",
        "\u56fe\u8868",
        "\u53ef\u89c6\u5316",
        "\u7ed8\u56fe",
    ]
    podcast_markers = [
        "podcast",
        "episode",
        "audio show",
        "audio",
        "interview",
        "\u64ad\u5ba2",
        "\u97f3\u9891",
    ]
    video_markers = [
        "video",
        "storyboard",
        "trailer",
        "promo video",
        "short film",
        "shorts",
        "\u89c6\u9891",
        "\u5206\u955c",
    ]
    image_markers = [
        "image",
        "poster",
        "illustration",
        "thumbnail",
        "render",
        "visual",
        "\u6d77\u62a5",
        "\u63d2\u56fe",
        "\u5c01\u9762",
        "\u914d\u56fe",
    ]
    data_markers = [
        "data analysis",
        "analytics",
        "dataset",
        "csv",
        "table",
        "sql",
        "cohort",
        "segmentation",
        "\u6570\u636e\u5206\u6790",
        "\u6570\u636e\u96c6",
        "\u8868\u683c",
        "\u5206\u7fa4",
    ]
    report_markers = ["report", "brief", "summary", "proposal", "\u65b9\u6848", "\u62a5\u544a", "\u603b\u7ed3"]
    memo_markers = ["memo", "one-pager", "one pager", "brief"]
    runbook_markers = ["runbook", "playbook", "ops", "\u64cd\u4f5c\u624b\u518c", "\u9884\u6848"]
    patch_markers = ["patch", "fix", "fixes", "implement", "tests", "\u4ee3\u7801\u4fee\u6539", "\u8865\u4e01", "\u4fee\u590d"]
    benchmark_runtime_markers = [
        "ablation",
        "runner",
        "run config",
        "run-config",
        "manifest",
        "suite",
        "gaia",
        "swe-bench",
        "webarena",
        "tau-bench",
        "\u6d88\u878d",
        "\u8dd1\u5206",
    ]

    workspace_signal = _count_markers(lowered, workspace_markers)
    external_signal = _count_markers(lowered, external_markers)
    code_signal = _count_markers(lowered, code_markers)
    ops_signal = _count_markers(lowered, ops_markers)
    benchmark_signal = _count_markers(lowered, benchmark_markers)
    report_signal = _count_markers(lowered, report_markers + memo_markers)
    benchmark_runtime_signal = _count_markers(lowered, benchmark_runtime_markers)
    patch_mode_signal = _count_markers(lowered, patch_markers)
    runbook_mode_signal = _count_markers(lowered, runbook_markers)
    webpage_mode_signal = _count_markers(lowered, webpage_markers)
    slides_mode_signal = _count_markers(lowered, slides_markers)
    chart_mode_signal = _count_markers(lowered, chart_markers)
    podcast_mode_signal = _count_markers(lowered, podcast_markers)
    video_mode_signal = _count_markers(lowered, video_markers)
    image_mode_signal = _count_markers(lowered, image_markers)
    data_mode_signal = _count_markers(lowered, data_markers)

    if target_hint == "code":
        workspace_signal += 2
        code_signal += 2
    elif target_hint == "research":
        external_signal += 2
    elif target_hint == "ops":
        ops_signal += 2

    execution_intent = "general"
    if benchmark_runtime_signal >= 1 and benchmark_signal >= max(code_signal, ops_signal, 1):
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
    if patch_mode_signal > 0:
        output_mode = "patch"
    elif runbook_mode_signal > 0:
        output_mode = "runbook"
    elif benchmark_runtime_signal > 0 and report_signal == 0:
        output_mode = "benchmark"
    elif webpage_mode_signal > 0:
        output_mode = "webpage"
    elif slides_mode_signal > 0:
        output_mode = "slides"
    elif chart_mode_signal > 0:
        output_mode = "chart"
    elif podcast_mode_signal > 0:
        output_mode = "podcast"
    elif video_mode_signal > 0:
        output_mode = "video"
    elif image_mode_signal > 0:
        output_mode = "image"
    elif data_mode_signal > 0 and report_signal == 0:
        output_mode = "data"
    elif report_signal > 0:
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
    elif output_mode in {"webpage", "slides", "podcast", "video", "image"}:
        reasoning_style = "creative"
    elif output_mode in {"chart", "data"}:
        reasoning_style = "analytic"

    workspace_summary = inspect_workspace_capabilities(workspace_root)
    provisional_validation = execution_intent in {"code", "benchmark", "mixed"} or output_mode in {
        "patch",
        "benchmark",
        "chart",
        "data",
    }
    if execution_intent == "research" and output_mode in {"report", "artifact"}:
        provisional_validation = False
    provisional_command_execution = (
        execution_intent in {"code", "benchmark"}
        and bool(workspace_summary.get("suggested_commands", []))
        and any(
            marker in lowered
            for marker in ["run", "execute", "build", "test", "validation", "\u8fd0\u884c", "\u6267\u884c", "\u6d4b\u8bd5"]
        )
    )
    task_spec = infer_task_spec(
        query=query,
        target=target_hint,
        domains=infer_domains(query),
        output_mode=output_mode,
        workspace_required=workspace_signal > 0 or "workspace" in package_channels,
        external_required=external_signal > 0 or "web" in package_channels,
        needs_validation=provisional_validation or package_validation,
        needs_command_execution=provisional_command_execution,
    )
    skill_priors = select_skill_priors(
        query=query,
        execution_intent=execution_intent,
        output_mode=output_mode,
        package_priors=package_priors,
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
    capability_plan = plan_capability_path(
        task_spec=task_spec,
        registry=default_capability_registry(),
    )
    merged_selected = list(dict.fromkeys(list(deliberation.selected) + list(capability_plan.get("required_channels", []))))
    merged_selected = list(dict.fromkeys(merged_selected + package_channels))
    if (
        workspace_signal <= 0
        and target_hint == "research"
        and external_signal > 0
        and "web" in merged_selected
        and "workspace" in merged_selected
        and "workspace" not in task_spec.required_channels
    ):
        merged_selected = [item for item in merged_selected if item != "workspace"]
    deliberation = ChannelDeliberation(
        scores=dict(deliberation.scores),
        selected=merged_selected,
        rationale=list(deliberation.rationale)
        + (["capability graph requested additional channels"] if merged_selected != deliberation.selected else [])
        + (["skill packages requested additional channels"] if any(channel not in deliberation.selected for channel in package_channels) else []),
    )

    selected = set(deliberation.selected)
    requires_workspace = "workspace" in selected
    requires_external_evidence = "web" in selected
    requires_discovery = "discovery" in selected or not selected
    requires_validation = provisional_validation
    requires_command_execution = provisional_command_execution

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
        package_priors=package_priors,
        task_spec=task_spec.to_dict(),
        capability_plan=capability_plan,
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
            query=query,
            selected_channels=deliberation.selected,
            output_mode=output_mode,
            requires_validation=requires_validation,
            requires_command_execution=requires_command_execution,
        ),
        workspace_summary=workspace_summary,
        task_spec=task_spec.to_dict(),
        capability_plan=capability_plan,
        graph_expansion=graph_expansion,
        skill_priors=skill_priors,
        package_priors=package_priors,
        deliberation=deliberation,
    )


def select_skill_priors(
    query: str,
    *,
    execution_intent: str = "general",
    output_mode: str = "artifact",
    package_priors: list[dict[str, Any]] | None = None,
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
        "webpage": {"webpage_blueprint", "frontend_critique", "executive_summary"},
        "slides": {"slide_deck_designer", "webpage_blueprint", "executive_summary"},
        "chart": {"chart_storyboard", "data_analysis_plan", "extract_facts"},
        "podcast": {"podcast_episode_plan", "research_brief", "synthesize_perspectives"},
        "video": {"video_storyboard", "image_prompt_pack", "executive_summary"},
        "image": {"image_prompt_pack", "brainstorm_ideas", "frontend_critique"},
        "data": {"data_analysis_plan", "chart_storyboard", "extract_facts", "validation_planner"},
    }

    priors: list[SkillPrior] = []
    package_skill_refs: dict[str, list[str]] = {}
    for package in package_priors or []:
        if not isinstance(package, dict):
            continue
        score = float(package.get("match_score", 0.0) or 0.0)
        if score < 0.42:
            continue
        package_name = str(package.get("name", "")).strip() or "package"
        for skill_name in package.get("skill_refs", []) if isinstance(package.get("skill_refs", []), list) else []:
            skill_text = str(skill_name).strip()
            if not skill_text:
                continue
            package_skill_refs.setdefault(skill_text, []).append(package_name)
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
        package_hits = package_skill_refs.get(meta.name, [])
        if package_hits:
            score += 0.38 + 0.06 * min(len(package_hits), 3)
            rationale.append(f"recommended by packages: {', '.join(package_hits[:2])}")
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


def _required_channels_from_packages(package_priors: list[dict[str, Any]]) -> list[str]:
    channels: list[str] = []
    for item in package_priors:
        if not isinstance(item, dict):
            continue
        score = float(item.get("match_score", 0.0) or 0.0)
        if score < 0.52:
            continue
        requirements = [str(req).strip().lower() for req in item.get("runtime_requirements", []) if str(req).strip()] if isinstance(item.get("runtime_requirements", []), list) else []
        tools = [str(tool).strip().lower() for tool in item.get("tool_refs", []) if str(tool).strip()] if isinstance(item.get("tool_refs", []), list) else []
        if "workspace" in requirements or any(tool.startswith("workspace_") for tool in tools):
            channels.append("workspace")
        if any(req in {"web", "evidence", "external"} for req in requirements) or any(tool in {"external_resource_hub", "evidence_dossier_builder"} for tool in tools):
            channels.append("web")
        if "risk" in requirements or any(tool == "policy_risk_matrix" for tool in tools):
            channels.append("risk")
        if item.get("skill_refs") or item.get("tool_refs"):
            channels.append("discovery")
    deduped: list[str] = []
    for channel in channels:
        if channel not in deduped:
            deduped.append(channel)
    return deduped


def _query_requests_benchmark_artifacts(query: str) -> bool:
    markers = [
        "ablation",
        "runner",
        "run config",
        "run-config",
        "manifest",
        "suite",
        "gaia",
        "swe-bench",
        "webarena",
        "tau-bench",
        "benchmark manifest",
        "benchmark config",
        "evaluation config",
        "run benchmark",
        "\u57fa\u51c6",
        "\u6d88\u878d",
        "\u8dd1\u5206",
    ]
    lowered = str(query or "").lower()
    return _count_markers(lowered, markers) > 0


def _query_requests_data_artifacts(query: str) -> bool:
    markers = [
        "data analysis",
        "analytics",
        "dataset",
        "csv",
        "table",
        "sql",
        "cohort",
        "dataset pull",
        "loader template",
        "data spec",
        "\u6570\u636e\u5206\u6790",
        "\u6570\u636e\u96c6",
        "\u8868\u683c",
    ]
    lowered = str(query or "").lower()
    return _count_markers(lowered, markers) > 0


def _package_graph_expansion_actions(*, package_priors: list[dict[str, Any]], selected_channels: list[str]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    selected = set(selected_channels)
    artifact_map = {
        "patch": "patch_draft",
        "execution_trace": "validation_execution",
        "validation_report": "validation_execution",
        "evidence_pack": "dataset_pull_spec",
        "evidence_bundle": "dataset_pull_spec",
        "benchmark_manifest": "benchmark_manifest",
        "benchmark_config": "benchmark_run_config",
        "chart_pack": "chart_pack_spec",
        "chart": "chart_pack_spec",
        "data_pack": "data_analysis_spec",
        "data": "data_analysis_spec",
        "deck": "slide_deck_plan",
        "slides": "slide_deck_plan",
        "presentation": "slide_deck_plan",
        "webpage": "webpage_blueprint",
        "website": "webpage_blueprint",
    }
    for package in package_priors:
        if not isinstance(package, dict):
            continue
        score = float(package.get("match_score", 0.0) or 0.0)
        if score < 0.58:
            continue
        package_name = str(package.get("name", "package")).strip() or "package"
        artifacts = [str(item).strip().lower() for item in package.get("artifact_kinds", []) if str(item).strip()] if isinstance(package.get("artifact_kinds", []), list) else []
        for artifact in artifacts:
            kind = artifact_map.get(artifact)
            if not kind:
                continue
            if kind in {"dataset_pull_spec"} and "web" not in selected:
                continue
            if kind in {"patch_draft", "validation_execution"} and "workspace" not in selected:
                continue
            actions.append(
                {
                    "kind": kind,
                    "title": kind.replace("_", " ").title(),
                    "depends_on": ["analysis"],
                    "reason": f"skill package {package_name} recommends {artifact} artifacts",
                }
            )
        tools = [str(item).strip() for item in package.get("tool_refs", []) if str(item).strip()] if isinstance(package.get("tool_refs", []), list) else []
        for tool_name in tools:
            lowered = tool_name.lower()
            if lowered in {"external_resource_hub", "evidence_dossier_builder"} and "web" in selected:
                actions.append(
                    {
                        "node_type": "tool_call",
                        "tool_name": tool_name,
                        "tool_args": {"query": f"package-guided collection for {package_name}", "limit": 5},
                        "title": f"Package Probe {tool_name.replace('_', ' ').title()}",
                        "depends_on": ["analysis"],
                        "reason": f"skill package {package_name} recommends {tool_name}",
                    }
                )
            if lowered == "policy_risk_matrix" and "risk" in selected:
                actions.append(
                    {
                        "node_type": "tool_call",
                        "tool_name": tool_name,
                        "tool_args": {"query": f"package-guided risk scan for {package_name}", "evidence_limit": 4},
                        "title": "Package Risk Matrix",
                        "depends_on": ["analysis"],
                        "reason": f"skill package {package_name} recommends explicit risk evaluation",
                    }
                )
    return actions


def requested_output_modes(*, query: str, output_mode: str) -> list[str]:
    """Infer one or more artifact surfaces requested by the user."""

    lowered = str(query or "").lower()
    rows = [
        ("webpage", ["webpage", "website", "landing page", "landing", "html", "frontend", "web app", "\u7f51\u9875", "\u524d\u7aef"]),
        ("slides", ["slides", "slide", "deck", "presentation", "ppt", "keynote", "\u5e7b\u706f", "\u6f14\u793a"]),
        ("chart", ["chart", "charts", "graph", "plot", "visualization", "visualize", "\u56fe\u8868", "\u53ef\u89c6\u5316"]),
        ("podcast", ["podcast", "episode", "audio", "interview", "\u64ad\u5ba2", "\u97f3\u9891"]),
        ("video", ["video", "storyboard", "trailer", "short film", "\u89c6\u9891", "\u5206\u955c"]),
        ("image", ["image", "poster", "illustration", "thumbnail", "render", "\u6d77\u62a5", "\u63d2\u56fe"]),
        ("data", ["data analysis", "analytics", "dataset", "csv", "table", "sql", "cohort", "\u6570\u636e\u5206\u6790", "\u6570\u636e\u96c6"]),
    ]
    requested: list[str] = []
    if output_mode in {name for name, _markers in rows}:
        requested.append(output_mode)
    for name, markers in rows:
        if _count_markers(lowered, markers) > 0 and name not in requested:
            requested.append(name)
    return requested


def default_artifact_targets(
    *,
    query: str,
    selected_channels: list[str],
    output_mode: str,
    requires_validation: bool,
    requires_command_execution: bool,
) -> list[str]:
    """Infer artifact targets that make the graph materially reviewable."""

    targets = ["analysis_brief", "completion_packet", "delivery_bundle", "deliverable_report"]
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
    if output_mode != "benchmark" and _query_requests_benchmark_artifacts(query):
        targets.append("benchmark_matrix")
    if _query_requests_data_artifacts(query):
        targets.append("data_analysis_spec")
    for mode in requested_output_modes(query=query, output_mode=output_mode):
        targets.extend(
            {
                "webpage": ["webpage_blueprint"],
                "slides": ["slide_deck_plan"],
                "chart": ["chart_pack_spec"],
                "podcast": ["podcast_episode_plan"],
                "video": ["video_storyboard"],
                "image": ["image_prompt_pack"],
                "data": ["data_analysis_spec"],
            }.get(mode, [])
        )
    deduped: list[str] = []
    for item in targets:
        if item not in deduped:
            deduped.append(item)
    return deduped


def custom_artifact_actions(*, task_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert custom artifact contracts into executable workspace actions."""

    contracts = task_spec.get("artifact_contracts", []) if isinstance(task_spec.get("artifact_contracts", []), list) else []
    target = str(task_spec.get("target", "")).strip().lower()
    actions: list[dict[str, Any]] = []
    for item in contracts:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip()
        if not kind.startswith("custom:"):
            continue
        title = str(item.get("title", kind.replace("custom:", "").replace("_", " ").title())).strip()
        format_hint = str(item.get("format_hint", "markdown")).strip() or "markdown"
        slug = re.sub(r"[^a-z0-9]+", "-", kind.replace("custom:", "").lower()).strip("-") or "artifact"
        content_type = "application/json" if format_hint == "json" else "text/markdown"
        relative_path = f"briefs/{slug}.json" if content_type == "application/json" else f"briefs/{slug}.md"
        depends_on = ["analysis"]
        if target == "research" and kind in {
            "custom:memo",
            "custom:brief",
            "custom:executive_memo",
            "custom:decision_memo",
            "custom:launch_memo",
            "custom:one_pager",
        }:
            depends_on = ["analysis", "evidence", "external_resources"]
        actions.append(
            {
                "node_type": "workspace_action",
                "kind": kind,
                "title": f"Generate {title}",
                "depends_on": depends_on,
                "reason": f"query explicitly asks for {title.lower()} as a first-class deliverable",
                "relative_path": relative_path,
                "content_type": content_type,
                "format_hint": format_hint,
                "artifact_contract": {
                    "kind": kind,
                    "title": title,
                    "format_hint": format_hint,
                    "required": bool(item.get("required", True)),
                },
            }
        )
    return actions


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
    package_priors: list[dict[str, Any]] | None = None,
    task_spec: dict[str, Any] | None = None,
    capability_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan additional executable graph nodes beyond analysis/report writing."""

    local = _default_graph_expansion(
        query=query,
        execution_intent=execution_intent,
        output_mode=output_mode,
        selected_channels=selected_channels,
        workspace_summary=workspace_summary,
        requires_command_execution=requires_command_execution,
        package_priors=package_priors,
        task_spec=task_spec,
        capability_plan=capability_plan,
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
    query: str,
    execution_intent: str,
    output_mode: str,
    selected_channels: list[str],
    workspace_summary: dict[str, Any],
    requires_command_execution: bool,
    package_priors: list[dict[str, Any]] | None = None,
    task_spec: dict[str, Any] | None = None,
    capability_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    selected = set(selected_channels)
    lowered = str(query or "").lower()
    explicit_data_build = any(
        marker in lowered for marker in ["dataset", "data analysis", "csv", "sql", "dashboard", "\u6570\u636e\u5206\u6790", "\u6570\u636e\u96c6"]
    )

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
    if execution_intent == "benchmark" or output_mode == "benchmark" or _query_requests_benchmark_artifacts(query):
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
    if execution_intent in {"research", "benchmark", "mixed"} and "web" in selected and (
        explicit_data_build or output_mode in {"data", "benchmark"} or _query_requests_data_artifacts(query)
    ):
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
    for mode in requested_output_modes(query=query, output_mode=output_mode):
        actions.extend(
            {
                "webpage": [
                    {
                        "kind": "webpage_blueprint",
                        "title": "Generate Webpage Blueprint",
                        "depends_on": ["analysis"],
                        "reason": "page-design tasks need a concrete first-screen and section blueprint",
                    }
                ],
                "slides": [
                    {
                        "kind": "slide_deck_plan",
                        "title": "Generate Slide Deck Plan",
                        "depends_on": ["analysis"],
                        "reason": "presentation tasks need slide-by-slide structure and proof beats",
                    }
                ],
                "chart": [
                    {
                        "kind": "chart_pack_spec",
                        "title": "Generate Chart Pack Spec",
                        "depends_on": ["analysis"],
                        "reason": "visualization tasks need chart choices and data contracts",
                    }
                ],
                "podcast": [
                    {
                        "kind": "podcast_episode_plan",
                        "title": "Generate Podcast Episode Plan",
                        "depends_on": ["analysis"],
                        "reason": "audio tasks need a segment plan and episode arc",
                    }
                ],
                "video": [
                    {
                        "kind": "video_storyboard",
                        "title": "Generate Video Storyboard",
                        "depends_on": ["analysis"],
                        "reason": "video tasks need scene-by-scene storyboard structure",
                    }
                ],
                "image": [
                    {
                        "kind": "image_prompt_pack",
                        "title": "Generate Image Prompt Pack",
                        "depends_on": ["analysis"],
                        "reason": "image tasks need reusable prompt directions and visual constraints",
                    }
                ],
                "data": [
                    {
                        "kind": "data_analysis_spec",
                        "title": "Generate Data Analysis Spec",
                        "depends_on": ["analysis"],
                        "reason": "data-analysis tasks need questions, metrics, cuts, and output contracts",
                    }
                ],
            }.get(mode, [])
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
    actions.extend(_package_graph_expansion_actions(package_priors=package_priors or [], selected_channels=selected_channels))
    actions.extend(custom_artifact_actions(task_spec=task_spec or {}))
    for step in (capability_plan or {}).get("steps", []) if isinstance((capability_plan or {}).get("steps", []), list) else []:
        if not isinstance(step, dict) or str(step.get("node_type", "")) != "workspace_action":
            continue
        kind = str(step.get("ref", "")).strip()
        if not kind:
            continue
        actions.append(
            {
                "kind": kind,
                "title": str(step.get("title", kind.replace("_", " ").title())).strip(),
                "depends_on": ["analysis"],
                "reason": str(step.get("reason", "capability graph selected workspace action")).strip(),
            }
        )

    deduped_actions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for action in actions:
        node_type = str(action.get("node_type", "workspace_action")).strip() or "workspace_action"
        key = str(
            action.get("kind", action.get("tool_name", action.get("subagent_kind", action.get("title", ""))))
        ).strip()
        if not key:
            continue
        dedupe_key = (node_type, key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped_actions.append(action)

    constrained_nodes = _constrain_live_graph_expansion(
        query=query,
        execution_intent=execution_intent,
        output_mode=output_mode,
        nodes=_normalize_graph_expansion_nodes({"actions": deduped_actions}),
    )
    constrained_actions = [
        {
            "kind": str(item.get("kind", "")).strip(),
            "title": str(item.get("title", "")).strip(),
            "depends_on": list(item.get("depends_on", ["analysis"])) if isinstance(item.get("depends_on", []), list) else ["analysis"],
            "reason": str(item.get("reason", "graph expansion")).strip(),
        }
        for item in constrained_nodes
        if str(item.get("node_type", "")) == "workspace_action"
    ]

    return {
        "actions": constrained_actions,
        "nodes": constrained_nodes,
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

        config = LiveModelConfig.resolve(live_model_overrides)
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
                    "Return strict JSON with keys: nodes, replan_enabled, replan_focus, rationale. "
                    "Each node must include node_type from workspace_action, tool_call, subagent. "
                    f"Allowed workspace_action kinds: {', '.join(sorted(allowed_workspace_action_kinds(include_internal=True)))}. "
                    "You may also emit kind starting with custom: when you include relative_path plus content_type or artifact_contract. "
                    "Allowed tool_call names: tool_search, workspace_file_search, workspace_file_read, external_resource_hub, "
                    "evidence_dossier_builder, code_experiment_design, policy_risk_matrix. "
                    "Allowed subagent kinds: repair_probe, research_probe, benchmark_probe, general_probe. "
                    "Only propose nodes that materially improve executable artifacts or evidence gathering."
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
        nodes = _normalize_graph_expansion_nodes(parsed)
        nodes = _constrain_live_graph_expansion(
            query=query,
            execution_intent=execution_intent,
            output_mode=output_mode,
            nodes=nodes,
        )
        if not nodes:
            return local
        replan_focus = [str(item).strip() for item in parsed.get("replan_focus", []) if str(item).strip()] if isinstance(parsed.get("replan_focus", []), list) else []
        rationale = list(local.get("rationale", []))
        rationale.append("live model refined graph expansion")
        for item in parsed.get("rationale", []) if isinstance(parsed.get("rationale", []), list) else []:
            text_item = str(item).strip()
            if text_item:
                rationale.append(text_item)
        actions = [
            {
                "kind": str(item.get("kind", "")).strip(),
                "title": str(item.get("title", "")).strip(),
                "depends_on": list(item.get("depends_on", ["analysis"])) if isinstance(item.get("depends_on", []), list) else ["analysis"],
                "reason": str(item.get("reason", "live model expansion")).strip(),
            }
            for item in nodes
            if str(item.get("node_type", "")) == "workspace_action"
        ]
        return {
            "actions": actions,
            "nodes": nodes,
            "replan_enabled": bool(parsed.get("replan_enabled", True)),
            "replan_focus": replan_focus or list(local.get("replan_focus", [])),
            "rationale": rationale[:8],
            "source": "live_model",
        }
    except Exception:
        return local


def _constrain_live_graph_expansion(
    *,
    query: str,
    execution_intent: str,
    output_mode: str,
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lowered = str(query or "").lower()
    report_like = output_mode in {"report", "artifact"} and any(
        marker in lowered for marker in ["report", "memo", "brief", "research", "\u62a5\u544a", "\u5907\u5fd8\u5f55"]
    )
    explicit_benchmark_build = any(
        marker in lowered for marker in ["benchmark manifest", "run config", "run-config", "ablation", "runner", "suite"]
    )
    explicit_data_build = any(
        marker in lowered for marker in ["dataset", "data analysis", "csv", "sql", "dashboard", "\u6570\u636e\u5206\u6790", "\u6570\u636e\u96c6"]
    )
    explicit_parallel_agent = any(
        marker in lowered for marker in ["subagent", "delegate", "parallel agent", "multi-agent", "research team"]
    )
    presentation_surface = _count_markers(lowered, ["presentation", "slides", "slide", "deck", "webpage", "website", "landing"]) > 0
    allowed_report_custom_kinds = {
        "custom:memo",
        "custom:brief",
        "custom:executive_memo",
        "custom:decision_memo",
        "custom:launch_memo",
        "custom:one_pager",
        "custom:source_matrix",
        "custom:research_outline",
        "custom:direct_answer_baseline",
    }
    filtered: list[dict[str, Any]] = []
    for item in nodes:
        if not isinstance(item, dict):
            continue
        node_type = str(item.get("node_type", "")).strip()
        if report_like and node_type in {"tool_call", "subagent"} and not explicit_parallel_agent and not presentation_surface:
            tool_name = str(item.get("tool_name", "")).strip()
            research_evidence_query = _count_markers(
                lowered,
                ["investigate", "evidence", "latest", "deep-research", "web", "internet", "sources", "citations"],
            ) > 0
            if node_type == "tool_call" and tool_name in {"external_resource_hub", "evidence_dossier_builder"} and research_evidence_query:
                filtered.append(item)
                continue
            continue
        if node_type != "workspace_action":
            filtered.append(item)
            continue
        kind = str(item.get("kind", "")).strip()
        if report_like:
            if kind.startswith("custom:") and kind not in allowed_report_custom_kinds:
                continue
            if kind in {"dataset_pull_spec", "dataset_loader_template", "data_analysis_spec"} and not explicit_data_build:
                continue
            if kind in {"benchmark_manifest", "benchmark_run_config"} and not (
                execution_intent == "benchmark" or explicit_benchmark_build
            ):
                continue
        filtered.append(item)
    return filtered


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

        config = LiveModelConfig.resolve(live_model_overrides)
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
    lowered = str(query or "").lower()

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

    selected: list[str] = ["discovery"]
    rationale: list[str] = []
    rationale.append("open-ended tasks stay discovery-first so the agent can inspect skills and tools before committing")

    mixed_intent = execution_intent == "mixed" or (
        any(marker in lowered for marker in ["compare", "versus", "vs", "tradeoff", "方案", "对比"])
        and workspace_signal > 0
        and external_signal > 0
    )
    if scores["workspace"] >= 0.5 or (mixed_intent and scores["workspace"] >= 0.4):
        selected.append("workspace")
        rationale.append("workspace selected because local artifacts appear decision-relevant")
    if scores["web"] >= 0.5 or (mixed_intent and scores["web"] >= 0.4):
        selected.append("web")
        rationale.append("web selected because external evidence is likely needed")
    if not any(item in selected for item in {"workspace", "web"}):
        if scores["workspace"] >= 0.44 and scores["workspace"] >= scores["web"]:
            selected.append("workspace")
            rationale.append("workspace selected as the strongest grounded channel under uncertainty")
        elif scores["web"] >= 0.44:
            selected.append("web")
            rationale.append("web selected as the strongest external evidence channel under uncertainty")
    if scores["risk"] >= 0.5:
        selected.append("risk")
        rationale.append("risk channel selected to constrain execution and validate governance")

    deduped_selected: list[str] = []
    for item in selected:
        if item not in deduped_selected:
            deduped_selected.append(item)

    return ChannelDeliberation(scores=scores, selected=deduped_selected, rationale=rationale)


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

    task_contracts = resolved.task_spec.get("artifact_contracts", []) if isinstance(resolved.task_spec.get("artifact_contracts", []), list) else []
    report_title = {
        "patch": "Patch Preparation Report",
        "runbook": "Operational Runbook",
        "benchmark": "Benchmark Study",
        "report": "Research Report",
        "webpage": "Website Blueprint",
        "slides": "Slide Deck Plan",
        "chart": "Chart Pack Brief",
        "podcast": "Podcast Episode Plan",
        "video": "Video Storyboard",
        "image": "Image Prompt Pack",
        "data": "Data Analysis Pack",
        "artifact": "Executable Task Artifact",
    }.get(resolved.output_mode, "Executable Task Artifact")
    if any(
        str(item.get("kind", "")) in {"custom:memo", "custom:executive_memo", "custom:decision_memo", "custom:launch_memo"}
        for item in task_contracts
        if isinstance(item, dict)
    ):
        report_title = "Research Memo"
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
                "task_spec": dict(resolved.task_spec),
                "capability_plan": dict(resolved.capability_plan),
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
    expansion_specs = _normalize_graph_expansion_nodes(resolved.graph_expansion)
    for spec in expansion_specs:
        node_type = str(spec.get("node_type", "workspace_action")).strip() or "workspace_action"
        if node_type == "workspace_action" and str(spec.get("kind", "")).strip() == "validation_execution":
            continue
        depends_on = [str(item) for item in spec.get("depends_on", ["analysis"]) if str(item)]
        if "validation" in node_ids_from_nodes(nodes) and "validation" not in depends_on:
            depends_on.append("validation")
        depends_on = list(dict.fromkeys(depends_on))
        existing_ids = node_ids_from_nodes(nodes)
        action_id = _expansion_node_id(spec, existing_ids)
        if node_type == "workspace_action":
            metrics = {
                "action_kind": str(spec.get("kind", "")).strip(),
                "prompt": query,
                "source_node_ids": list(
                    dict.fromkeys(
                        [str(item) for item in spec.get("source_node_ids", depends_on) if str(item)]
                    )
                )
                or depends_on,
                "workspace_summary": dict(resolved.workspace_summary),
            }
            if str(spec.get("relative_path", "")).strip():
                metrics["relative_path"] = str(spec.get("relative_path", "")).strip()
            if str(spec.get("content_type", "")).strip():
                metrics["content_type"] = str(spec.get("content_type", "")).strip()
            if str(spec.get("format_hint", "")).strip():
                metrics["format_hint"] = str(spec.get("format_hint", "")).strip()
            if isinstance(spec.get("artifact_contract", {}), dict):
                metrics["artifact_contract"] = dict(spec.get("artifact_contract", {}))
        elif node_type == "tool_call":
            metrics = {
                "tool_name": str(spec.get("tool_name", "tool_search")).strip(),
                "tool_args": dict(spec.get("tool_args", {})) if isinstance(spec.get("tool_args", {}), dict) else {},
            }
        else:
            metrics = {
                "subagent_kind": str(spec.get("subagent_kind", "general_probe")).strip(),
                "objective": str(spec.get("objective", query)).strip() or query,
                "prompt": query,
                "source_node_ids": list(
                    dict.fromkeys(
                        [str(item) for item in spec.get("source_node_ids", depends_on) if str(item)]
                    )
                )
                or depends_on,
                "workspace_summary": dict(resolved.workspace_summary),
            }
        nodes.append(
            TaskGraphNode(
                node_id=action_id,
                title=str(
                    spec.get(
                        "title",
                        spec.get("kind", spec.get("tool_name", spec.get("subagent_kind", "Expansion Node"))),
                    )
                ).strip(),
                node_type=node_type,
                status="ready",
                depends_on=depends_on,
                metrics=metrics,
            )
        )
        expansion_action_ids.append(action_id)
        synthesis_sources.append(action_id)

    research_surface = (
        resolved.execution_intent in {"research", "benchmark"}
        or (resolved.output_mode == "report" and "web" in selected)
    )
    if research_surface:
        memo_kinds = {
            "custom:memo",
            "custom:brief",
            "custom:executive_memo",
            "custom:decision_memo",
            "custom:launch_memo",
            "custom:one_pager",
        }
        deferred_memo_node_ids = [
            item.node_id
            for item in nodes
            if isinstance(item, TaskGraphNode)
            and str(item.node_type) == "workspace_action"
            and str(item.metrics.get("action_kind", "")) in memo_kinds
        ]
        research_nodes = [
            (
                "source_matrix",
                "Build Source Matrix",
                "custom:source_matrix",
                f"research/{slug}-source-matrix.md",
                ["Question", "Source", "Usefulness", "Open Gaps"],
            ),
            (
                "report_outline",
                "Build Research Outline",
                "custom:research_outline",
                f"research/{slug}-outline.md",
                ["Core Thesis", "Sections", "Evidence Coverage", "Missing Proof"],
            ),
            (
                "direct_baseline",
                "Draft Direct-Model Baseline",
                "custom:direct_answer_baseline",
                f"research/{slug}-direct-baseline.md",
                ["Baseline Answer", "What It Misses", "What Harness Must Add"],
            ),
        ]
        research_evidence_sources = [
            item.node_id
            for item in nodes
            if isinstance(item, TaskGraphNode)
            and str(item.node_type) in {"tool_call", "skill_call"}
            and (
                item.node_id in {"analysis", "external_resources", "evidence"}
                or str(item.metrics.get("tool_name", "")) in {"external_resource_hub", "evidence_dossier_builder"}
            )
        ]
        prior_sources = list(
            dict.fromkeys(
                [item for item in synthesis_sources if item not in deferred_memo_node_ids]
                + research_evidence_sources
            )
        )
        for node_id, title, kind, relative_path, sections in research_nodes:
            source_ids = list(dict.fromkeys(prior_sources))
            nodes.append(
                TaskGraphNode(
                    node_id=node_id,
                    title=title,
                    node_type="workspace_action",
                    status="ready",
                    depends_on=source_ids,
                    metrics={
                        "action_kind": kind,
                        "prompt": query,
                        "source_node_ids": source_ids,
                        "relative_path": relative_path,
                        "content_type": "text/markdown",
                        "format_hint": "markdown",
                        "artifact_contract": {
                            "title": title,
                            "sections": sections,
                        },
                    },
                )
            )
            prior_sources.append(node_id)
        synthesis_sources = list(dict.fromkeys(prior_sources + deferred_memo_node_ids))
        research_support_ids = ["source_matrix", "report_outline", "direct_baseline"]
        rewritten_nodes: list[TaskGraphNode] = []
        for item in nodes:
            if not isinstance(item, TaskGraphNode):
                rewritten_nodes.append(item)
                continue
            if str(item.node_type) != "workspace_action":
                rewritten_nodes.append(item)
                continue
            action_kind = str(item.metrics.get("action_kind", "")) if isinstance(item.metrics, dict) else ""
            if action_kind not in memo_kinds:
                rewritten_nodes.append(item)
                continue
            memo_depends = list(dict.fromkeys(list(item.depends_on or []) + research_support_ids))
            memo_sources = list(
                dict.fromkeys(
                    list(item.metrics.get("source_node_ids", []))
                    + [node_id for node_id in research_support_ids if node_id]
                )
            )
            rewritten_nodes.append(
                TaskGraphNode(
                    node_id=item.node_id,
                    title=item.title,
                    node_type=item.node_type,
                    status=item.status,
                    depends_on=memo_depends,
                    commands=list(item.commands),
                    notes=list(item.notes),
                    artifacts=list(item.artifacts),
                    metrics={**dict(item.metrics), "source_node_ids": memo_sources},
                )
            )
        nodes = rewritten_nodes

    contracts = resolved.task_spec.get("artifact_contracts", []) if isinstance(resolved.task_spec.get("artifact_contracts", []), list) else []
    custom_contract_count = sum(1 for item in contracts if isinstance(item, dict) and str(item.get("kind", "")).startswith("custom:"))
    requested_surface_count = len(requested_output_modes(query=query, output_mode=resolved.output_mode))

    synthesis_skill = "artifact_synthesis"
    lowered_query = query.lower()
    if resolved.output_mode == "patch" or resolved.execution_intent == "code":
        synthesis_skill = "codebase_triage"
    elif resolved.output_mode == "runbook" or resolved.execution_intent == "ops":
        synthesis_skill = "ops_runbook"
    elif research_surface and any(
        marker in lowered_query for marker in ["report", "research", "improvement", "roadmap", "strategy", "gaps"]
    ):
        synthesis_skill = "artifact_synthesis"
    elif resolved.output_mode == "benchmark" or resolved.execution_intent == "benchmark":
        synthesis_skill = "benchmark_ablation"
    elif custom_contract_count > 0 or requested_surface_count > 1 or len(contracts) > 2:
        synthesis_skill = "artifact_synthesis"
    elif resolved.output_mode == "report" and "web" in selected:
        synthesis_skill = "research_brief"
    elif resolved.output_mode == "webpage":
        synthesis_skill = "webpage_blueprint"
    elif resolved.output_mode == "slides":
        synthesis_skill = "slide_deck_designer"
    elif resolved.output_mode == "chart":
        synthesis_skill = "chart_storyboard"
    elif resolved.output_mode == "podcast":
        synthesis_skill = "podcast_episode_plan"
    elif resolved.output_mode == "video":
        synthesis_skill = "video_storyboard"
    elif resolved.output_mode == "image":
        synthesis_skill = "image_prompt_pack"
    elif resolved.output_mode == "data":
        synthesis_skill = "data_analysis_plan"

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

    completion_packet_depends = list(dict.fromkeys(synthesis_sources + ["report"]))
    nodes.append(
        TaskGraphNode(
            node_id="completion_packet",
            title="Generate Completion Packet",
            node_type="workspace_action",
            status="ready",
            depends_on=completion_packet_depends,
            metrics={
                "action_kind": "completion_packet",
                "prompt": query,
                "source_node_ids": completion_packet_depends,
                "workspace_summary": dict(resolved.workspace_summary),
                "task_spec": dict(resolved.task_spec),
            },
        )
    )

    if bool(resolved.graph_expansion.get("replan_enabled", False)):
        replan_depends = ["completion_packet"]
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
                    "task_spec": dict(resolved.task_spec),
                    "capability_plan": dict(resolved.capability_plan),
                    "graph_expansion_source": str(resolved.graph_expansion.get("source", "local")),
                },
            )
        )

    delivery_bundle_depends = ["completion_packet", "report"]
    if resolved.requires_command_execution and resolved.workspace_summary.get("suggested_commands"):
        delivery_bundle_depends.append("execution_trace")
    nodes.append(
        TaskGraphNode(
            node_id="delivery_bundle",
            title="Generate Delivery Bundle",
            node_type="workspace_action",
            status="ready",
            depends_on=list(dict.fromkeys(delivery_bundle_depends)),
            metrics={
                "action_kind": "delivery_bundle",
                "prompt": query,
                "source_node_ids": list(dict.fromkeys(delivery_bundle_depends)),
                "workspace_summary": dict(resolved.workspace_summary),
                "task_spec": dict(resolved.task_spec),
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


def _normalize_graph_expansion_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    allowed_workspace_actions = allowed_workspace_action_kinds(include_internal=True)
    allowed_tool_names = {
        "tool_search",
        "workspace_file_search",
        "workspace_file_read",
        "external_resource_hub",
        "evidence_dossier_builder",
        "code_experiment_design",
        "policy_risk_matrix",
    }
    allowed_subagent_kinds = {"repair_probe", "research_probe", "benchmark_probe", "general_probe"}

    raw_nodes: list[dict[str, Any]] = []
    nodes_payload = payload.get("nodes", [])
    if isinstance(nodes_payload, list):
        raw_nodes.extend(item for item in nodes_payload if isinstance(item, dict))
    actions_payload = payload.get("actions", [])
    if isinstance(actions_payload, list):
        for item in actions_payload:
            if not isinstance(item, dict):
                continue
            if str(item.get("node_type", "")).strip():
                raw_nodes.append(item)
            else:
                raw_nodes.append({**item, "node_type": "workspace_action"})

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_nodes:
        node_type = str(item.get("node_type", "workspace_action")).strip() or "workspace_action"
        depends_on = [str(dep).strip() for dep in item.get("depends_on", ["analysis"]) if str(dep).strip()] if isinstance(item.get("depends_on", []), list) else ["analysis"]
        reason = str(item.get("reason", "graph expansion")).strip() or "graph expansion"
        title = str(item.get("title", "")).strip()

        if node_type == "workspace_action":
            kind = str(item.get("kind", "")).strip()
            is_custom = kind.startswith("custom:")
            if kind in {"completion_packet", "delivery_bundle"}:
                continue
            if kind not in allowed_workspace_actions and not is_custom:
                continue
            if is_custom and not (
                str(item.get("relative_path", "")).strip()
                or isinstance(item.get("artifact_contract", {}), dict)
            ):
                continue
            key = (node_type, kind)
            if key in seen:
                continue
            seen.add(key)
            node_spec = {
                "node_type": node_type,
                "kind": kind,
                "title": title or kind.replace("_", " ").title(),
                "depends_on": depends_on,
                "reason": reason,
                "source_node_ids": list(
                    item.get("source_node_ids", depends_on)
                    if isinstance(item.get("source_node_ids", depends_on), list)
                    else depends_on
                ),
            }
            if is_custom:
                node_spec["relative_path"] = str(item.get("relative_path", "")).strip()
                node_spec["content_type"] = str(item.get("content_type", "")).strip()
                node_spec["format_hint"] = str(item.get("format_hint", "")).strip()
                if isinstance(item.get("artifact_contract", {}), dict):
                    node_spec["artifact_contract"] = dict(item.get("artifact_contract", {}))
            normalized.append(node_spec)
            continue

        if node_type == "tool_call":
            tool_name = str(item.get("tool_name", "")).strip()
            if tool_name not in allowed_tool_names:
                continue
            key = (node_type, tool_name)
            if key in seen:
                continue
            seen.add(key)
            tool_args = dict(item.get("tool_args", {})) if isinstance(item.get("tool_args", {}), dict) else {}
            normalized.append(
                {
                    "node_type": node_type,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "title": title or tool_name.replace("_", " ").title(),
                    "depends_on": depends_on,
                    "reason": reason,
                }
            )
            continue

        if node_type == "subagent":
            subagent_kind = str(item.get("subagent_kind", "")).strip() or "general_probe"
            if subagent_kind not in allowed_subagent_kinds:
                continue
            key = (node_type, subagent_kind)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "node_type": node_type,
                    "subagent_kind": subagent_kind,
                    "objective": str(item.get("objective", "")).strip(),
                    "title": title or subagent_kind.replace("_", " ").title(),
                    "depends_on": depends_on,
                    "reason": reason,
                    "source_node_ids": list(
                        item.get("source_node_ids", depends_on)
                        if isinstance(item.get("source_node_ids", depends_on), list)
                        else depends_on
                    ),
                }
            )
    return normalized


def _expansion_node_id(spec: dict[str, Any], existing_ids: set[str]) -> str:
    node_type = str(spec.get("node_type", "workspace_action")).strip() or "workspace_action"
    key = str(
        spec.get("kind", spec.get("tool_name", spec.get("subagent_kind", spec.get("title", "node"))))
    ).strip()
    prefix = {"workspace_action": "action", "tool_call": "tool", "subagent": "subagent"}.get(node_type, "node")
    base = f"{prefix}_{_slugify([key], fallback='node')}"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def node_ids_from_nodes(nodes: list[TaskGraphNode]) -> set[str]:
    return {item.node_id for item in nodes}
