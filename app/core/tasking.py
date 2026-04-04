"""Task-spec, capability graph, and state-gap planning primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _norm(text: object) -> str:
    return str(text or "").strip()


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = _norm(item)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _has_marker(*, lowered: str, tokens: set[str], marker: str) -> bool:
    marker_text = _norm(marker).lower()
    if not marker_text:
        return False
    if " " in marker_text:
        return marker_text in lowered
    return marker_text in tokens


def _dedupe_contracts(contracts: list["ArtifactContract"]) -> list["ArtifactContract"]:
    out: list[ArtifactContract] = []
    seen: set[str] = set()
    for item in contracts:
        if item.kind in seen:
            continue
        seen.add(item.kind)
        out.append(item)
    return out


def _custom_document_contracts(lowered: str) -> list["ArtifactContract"]:
    contracts: list[ArtifactContract] = []
    memo_specific = False
    for marker, contract in [
        ("decision memo", ArtifactContract(kind="custom:decision_memo", title="Decision Memo", format_hint="markdown")),
        ("executive memo", ArtifactContract(kind="custom:executive_memo", title="Executive Memo", format_hint="markdown")),
        ("launch memo", ArtifactContract(kind="custom:launch_memo", title="Launch Memo", format_hint="markdown")),
    ]:
        if marker in lowered:
            contracts.append(contract)
            memo_specific = True
    if "memo" in lowered and not memo_specific:
        contracts.append(ArtifactContract(kind="custom:memo", title="Memo", format_hint="markdown"))
    if "one-pager" in lowered or "one pager" in lowered:
        contracts.append(ArtifactContract(kind="custom:one_pager", title="One-Pager", format_hint="markdown"))
    if "brief" in lowered:
        contracts.append(ArtifactContract(kind="custom:brief", title="Brief", format_hint="markdown"))
    if "checklist" in lowered:
        contracts.append(ArtifactContract(kind="custom:checklist", title="Checklist", format_hint="markdown"))
    if "faq" in lowered:
        contracts.append(ArtifactContract(kind="custom:faq", title="FAQ", format_hint="markdown"))
    return _dedupe_contracts(contracts)


def _explicit_artifact_contracts(lowered: str) -> list["ArtifactContract"]:
    contracts: list[ArtifactContract] = []
    markers = [
        (("risk register",), ArtifactContract(kind="risk_register", title="Risk Register", format_hint="json")),
        (
            ("evidence bundle", "evidence packet", "evidence pack", "evidence dossier"),
            ArtifactContract(kind="evidence_bundle", title="Evidence Bundle", format_hint="json"),
        ),
        (("workspace findings", "inspection findings"), ArtifactContract(kind="workspace_findings", title="Workspace Findings", format_hint="json")),
    ]
    for keys, contract in markers:
        if any(key in lowered for key in keys):
            contracts.append(contract)
    return _dedupe_contracts(contracts)


@dataclass(frozen=True)
class ArtifactContract:
    """One concrete output the task expects."""

    kind: str
    title: str
    format_hint: str = ""
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "format_hint": self.format_hint,
            "required": self.required,
        }


@dataclass(frozen=True)
class WorkspaceActionSpec:
    """One executable workspace artifact action visible to planner and runtime."""

    kind: str
    title: str
    default_relative_path: str
    content_type: str
    result_field: str = "output"
    format_hint: str = ""
    internal_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "default_relative_path": self.default_relative_path,
            "content_type": self.content_type,
            "result_field": self.result_field,
            "format_hint": self.format_hint,
            "internal_only": self.internal_only,
        }


@dataclass(frozen=True)
class TaskSpec:
    """Unified task abstraction independent of one task family template."""

    query: str
    goal: str
    target: str = "general"
    domains: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    required_channels: list[str] = field(default_factory=list)
    artifact_contracts: list[ArtifactContract] = field(default_factory=list)
    risk_policy: str = "balanced"
    needs_validation: bool = False
    needs_command_execution: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "goal": self.goal,
            "target": self.target,
            "domains": list(self.domains),
            "constraints": list(self.constraints),
            "success_criteria": list(self.success_criteria),
            "required_channels": list(self.required_channels),
            "artifact_contracts": [item.to_dict() for item in self.artifact_contracts],
            "risk_policy": self.risk_policy,
            "needs_validation": self.needs_validation,
            "needs_command_execution": self.needs_command_execution,
        }


@dataclass(frozen=True)
class Capability:
    """One reusable operator capability available to the planner/runtime."""

    name: str
    title: str
    node_type: str
    ref: str
    phase: str
    produces_channels: list[str] = field(default_factory=list)
    produces_artifacts: list[str] = field(default_factory=list)
    covers_keywords: list[str] = field(default_factory=list)
    default_args: dict[str, Any] = field(default_factory=dict)
    source_node_ids: list[str] = field(default_factory=list)
    requires_channels: list[str] = field(default_factory=list)
    cost_score: float = 1.0
    risk_level: str = "medium"

    def to_step(self, *, reason: str = "", depends_on: list[str] | None = None) -> dict[str, Any]:
        return {
            "capability": self.name,
            "title": self.title,
            "node_type": self.node_type,
            "ref": self.ref,
            "phase": self.phase,
            "produces_channels": list(self.produces_channels),
            "produces_artifacts": list(self.produces_artifacts),
            "default_args": dict(self.default_args),
            "source_node_ids": list(self.source_node_ids),
            "depends_on": list(depends_on or []),
            "reason": reason or self.title,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_step()


@dataclass
class CapabilityRegistry:
    """Registry of planner-visible reusable capabilities."""

    capabilities: dict[str, Capability] = field(default_factory=dict)

    def register(self, capability: Capability) -> None:
        self.capabilities[capability.name] = capability

    def get(self, name: str) -> Capability | None:
        return self.capabilities.get(name)

    def list_all(self) -> list[Capability]:
        return list(self.capabilities.values())


@dataclass(frozen=True)
class TaskWorldState:
    """Observed execution state used for gap-driven replanning."""

    channels: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    completed_capabilities: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    validation_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": list(self.channels),
            "artifacts": list(self.artifacts),
            "completed_capabilities": list(self.completed_capabilities),
            "failures": list(self.failures),
            "validation_ok": self.validation_ok,
        }


@dataclass(frozen=True)
class StateGap:
    """Difference between desired task state and current world state."""

    missing_channels: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    missing_validation: bool = False
    failure_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_channels": list(self.missing_channels),
            "missing_artifacts": list(self.missing_artifacts),
            "missing_validation": self.missing_validation,
            "failure_types": list(self.failure_types),
        }


def infer_task_spec(
    *,
    query: str,
    target: str = "general",
    domains: list[str] | None = None,
    output_mode: str = "artifact",
    workspace_required: bool = False,
    external_required: bool = False,
    needs_validation: bool = False,
    needs_command_execution: bool = False,
) -> TaskSpec:
    """Infer a task spec from a free-form query."""

    lowered = query.lower()
    tokens = set(re.findall(r"[a-z0-9_-]+", lowered))
    contracts: list[ArtifactContract] = []
    if output_mode == "patch":
        contracts.append(ArtifactContract(kind="patch_plan", title="Patch Plan", format_hint="markdown"))
        contracts.append(ArtifactContract(kind="patch_draft", title="Patch Draft", format_hint="diff"))
    elif output_mode == "benchmark":
        contracts.append(ArtifactContract(kind="benchmark_manifest", title="Benchmark Manifest", format_hint="json"))
        contracts.append(ArtifactContract(kind="benchmark_run_config", title="Benchmark Run Config", format_hint="json"))
    elif output_mode == "runbook":
        contracts.append(ArtifactContract(kind="runbook", title="Operational Runbook", format_hint="markdown"))
    elif output_mode == "webpage":
        contracts.append(ArtifactContract(kind="webpage_blueprint", title="Webpage Blueprint", format_hint="markdown"))
    elif output_mode == "slides":
        contracts.append(ArtifactContract(kind="slide_deck_plan", title="Slide Deck Plan", format_hint="markdown"))
    elif output_mode == "chart":
        contracts.append(ArtifactContract(kind="chart_pack_spec", title="Chart Pack Spec", format_hint="json"))
    elif output_mode == "podcast":
        contracts.append(ArtifactContract(kind="podcast_episode_plan", title="Podcast Episode Plan", format_hint="markdown"))
    elif output_mode == "video":
        contracts.append(ArtifactContract(kind="video_storyboard", title="Video Storyboard", format_hint="markdown"))
    elif output_mode == "image":
        contracts.append(ArtifactContract(kind="image_prompt_pack", title="Image Prompt Pack", format_hint="markdown"))
    elif output_mode == "data":
        contracts.append(ArtifactContract(kind="data_analysis_spec", title="Data Analysis Spec", format_hint="json"))
    elif any(marker in lowered for marker in ["webpage", "website", "landing", "frontend", "html"]):
        contracts.append(ArtifactContract(kind="webpage_blueprint", title="Webpage Blueprint", format_hint="markdown"))
    elif any(marker in lowered for marker in ["slides", "slide", "deck", "presentation", "ppt"]):
        contracts.append(ArtifactContract(kind="slide_deck_plan", title="Slide Deck Plan", format_hint="markdown"))
    elif any(marker in lowered for marker in ["chart", "visualization", "plot", "graph"]):
        contracts.append(ArtifactContract(kind="chart_pack_spec", title="Chart Pack Spec", format_hint="json"))
    elif any(marker in lowered for marker in ["podcast", "episode", "audio"]):
        contracts.append(ArtifactContract(kind="podcast_episode_plan", title="Podcast Episode Plan", format_hint="markdown"))
    elif any(marker in lowered for marker in ["video", "storyboard", "trailer"]):
        contracts.append(ArtifactContract(kind="video_storyboard", title="Video Storyboard", format_hint="markdown"))
    elif any(marker in lowered for marker in ["image", "poster", "illustration", "thumbnail"]):
        contracts.append(ArtifactContract(kind="image_prompt_pack", title="Image Prompt Pack", format_hint="markdown"))
    elif any(marker in lowered for marker in ["data", "dataset", "analytics", "dashboard", "sql", "csv"]):
        contracts.append(ArtifactContract(kind="data_analysis_spec", title="Data Analysis Spec", format_hint="json"))
    else:
        contracts.append(ArtifactContract(kind="deliverable_report", title="Deliverable Report", format_hint="markdown"))
    contracts.extend(_custom_document_contracts(lowered))
    contracts.extend(_explicit_artifact_contracts(lowered))
    contracts.append(ArtifactContract(kind="completion_packet", title="Completion Packet", format_hint="json"))
    contracts = _dedupe_contracts(contracts)

    required_channels: list[str] = ["discovery"]
    if workspace_required or any(_has_marker(lowered=lowered, tokens=tokens, marker=marker) for marker in ["repo", "workspace", "code", "file", "module", "patch", "test"]):
        required_channels.append("workspace")
    if external_required or any(_has_marker(lowered=lowered, tokens=tokens, marker=marker) for marker in ["latest", "research", "web", "internet", "benchmark", "market", "paper"]):
        required_channels.append("web")
    if any(_has_marker(lowered=lowered, tokens=tokens, marker=marker) for marker in ["risk", "safety", "governance", "policy", "audit", "compliance"]):
        required_channels.append("risk")

    success = [
        "produce inspectable artifacts instead of a single answer string",
        "leave enough evidence for a reviewer to verify the result",
    ]
    for contract in contracts:
        success.append(f"materialize {contract.title.lower()}")
    if needs_validation:
        success.append("record a validation-ready artifact or check")

    constraints = []
    if needs_command_execution:
        constraints.append("use bounded workspace execution only when local commands are relevant")
    if "web" not in required_channels:
        constraints.append("do not force external evidence unless the task actually needs it")

    return TaskSpec(
        query=query,
        goal=query,
        target=target,
        domains=list(domains or []),
        constraints=_dedupe(constraints),
        success_criteria=_dedupe(success),
        required_channels=_dedupe(required_channels),
        artifact_contracts=contracts,
        risk_policy="strict" if "risk" in required_channels else "balanced",
        needs_validation=needs_validation,
        needs_command_execution=needs_command_execution,
    )


def default_workspace_action_specs() -> dict[str, WorkspaceActionSpec]:
    """Return the canonical workspace-action catalog shared by planner and runtime."""

    specs = [
        WorkspaceActionSpec("patch_scaffold", "Generate Patch Scaffold", "plans/patch-scaffold.md", "text/markdown", "output", "markdown"),
        WorkspaceActionSpec("patch_draft", "Generate Patch Draft", "patches/patch-draft.diff", "text/plain", "output", "diff"),
        WorkspaceActionSpec("benchmark_run_config", "Generate Benchmark Run Config", "benchmarks/run-config.json", "application/json", "config", "json"),
        WorkspaceActionSpec("benchmark_manifest", "Generate Benchmark Manifest", "benchmarks/manifest.json", "application/json", "manifest", "json"),
        WorkspaceActionSpec("completion_packet", "Generate Completion Packet", "packets/completion-packet.json", "application/json", "packet", "json"),
        WorkspaceActionSpec("dataset_pull_spec", "Generate Dataset Pull Spec", "datasets/pull-spec.json", "application/json", "spec", "json"),
        WorkspaceActionSpec("dataset_loader_template", "Generate Dataset Loader Template", "datasets/loader_template.py", "text/plain", "output", "python"),
        WorkspaceActionSpec("webpage_blueprint", "Generate Webpage Blueprint", "web/landing-page-blueprint.md", "text/markdown", "output", "markdown"),
        WorkspaceActionSpec("slide_deck_plan", "Generate Slide Deck Plan", "slides/deck-plan.md", "text/markdown", "output", "markdown"),
        WorkspaceActionSpec("chart_pack_spec", "Generate Chart Pack Spec", "charts/chart-pack.json", "application/json", "spec", "json"),
        WorkspaceActionSpec("podcast_episode_plan", "Generate Podcast Episode Plan", "podcast/episode-plan.md", "text/markdown", "output", "markdown"),
        WorkspaceActionSpec("video_storyboard", "Generate Video Storyboard", "video/storyboard.md", "text/markdown", "output", "markdown"),
        WorkspaceActionSpec("image_prompt_pack", "Generate Image Prompt Pack", "images/prompt-pack.md", "text/markdown", "output", "markdown"),
        WorkspaceActionSpec("data_analysis_spec", "Generate Data Analysis Spec", "analysis/data-analysis-spec.json", "application/json", "spec", "json"),
        WorkspaceActionSpec("validation_execution", "Execute Suggested Validation", "", "application/json", "results", "json", internal_only=True),
    ]
    return {item.kind: item for item in specs}


def allowed_workspace_action_kinds(*, include_internal: bool = True) -> set[str]:
    """Return allowed workspace-action kinds."""

    return {
        item.kind
        for item in default_workspace_action_specs().values()
        if include_internal or not item.internal_only
    }


def workspace_action_result_field(kind: str, *, content_type: str = "") -> str:
    """Return the canonical result field for one workspace action."""

    spec = default_workspace_action_specs().get(str(kind or "").strip())
    if spec:
        return spec.result_field
    return "spec" if str(content_type).strip() == "application/json" else "output"


def default_capability_registry() -> CapabilityRegistry:
    """Return the default capability/operator graph."""

    registry = CapabilityRegistry()
    for capability in [
        Capability(
            name="discover_tools",
            title="Discover Relevant Tools",
            node_type="tool_call",
            ref="tool_search",
            phase="observe",
            produces_channels=["discovery"],
            cost_score=0.5,
            risk_level="low",
        ),
        Capability(
            name="inspect_skill_catalog",
            title="Inspect Skill Priors",
            node_type="tool_call",
            ref="code_skill_search",
            phase="observe",
            produces_channels=["discovery"],
            cost_score=0.4,
            risk_level="low",
        ),
        Capability(
            name="observe_workspace",
            title="Inspect Workspace",
            node_type="tool_call",
            ref="workspace_file_search",
            phase="observe",
            produces_channels=["workspace"],
            produces_artifacts=["workspace_findings"],
            requires_channels=["workspace"],
            cost_score=0.7,
            risk_level="low",
        ),
        Capability(
            name="collect_external_evidence",
            title="Collect External Resources",
            node_type="tool_call",
            ref="external_resource_hub",
            phase="observe",
            produces_channels=["web"],
            cost_score=0.9,
            risk_level="medium",
        ),
        Capability(
            name="build_evidence_dossier",
            title="Build Evidence Dossier",
            node_type="tool_call",
            ref="evidence_dossier_builder",
            phase="observe",
            produces_channels=["web"],
            produces_artifacts=["evidence_bundle"],
            requires_channels=["web"],
            cost_score=1.0,
            risk_level="medium",
        ),
        Capability(
            name="assess_risk",
            title="Evaluate Risk and Governance",
            node_type="tool_call",
            ref="policy_risk_matrix",
            phase="observe",
            produces_channels=["risk"],
            produces_artifacts=["risk_register"],
            requires_channels=["risk"],
            cost_score=0.8,
            risk_level="low",
        ),
        Capability(
            name="decompose_task",
            title="Analyze Task",
            node_type="skill_call",
            ref="decompose_task",
            phase="reason",
            cost_score=0.7,
            risk_level="low",
        ),
        Capability(
            name="plan_validation",
            title="Plan Validation",
            node_type="skill_call",
            ref="validation_planner",
            phase="validate",
            produces_artifacts=["validation_plan"],
            cost_score=0.8,
            risk_level="low",
        ),
        Capability(
            name="produce_completion_packet",
            title="Generate Completion Packet",
            node_type="workspace_action",
            ref="completion_packet",
            phase="produce",
            produces_artifacts=["completion_packet"],
        ),
        Capability(
            name="produce_patch_scaffold",
            title="Generate Patch Scaffold",
            node_type="workspace_action",
            ref="patch_scaffold",
            phase="produce",
            produces_artifacts=["patch_plan"],
            requires_channels=["workspace"],
        ),
        Capability(
            name="produce_patch_draft",
            title="Generate Patch Draft",
            node_type="workspace_action",
            ref="patch_draft",
            phase="produce",
            produces_artifacts=["patch_draft"],
            requires_channels=["workspace"],
        ),
        Capability(
            name="produce_benchmark_manifest",
            title="Generate Benchmark Manifest",
            node_type="workspace_action",
            ref="benchmark_manifest",
            phase="produce",
            produces_artifacts=["benchmark_manifest"],
        ),
        Capability(
            name="produce_benchmark_run_config",
            title="Generate Benchmark Run Config",
            node_type="workspace_action",
            ref="benchmark_run_config",
            phase="produce",
            produces_artifacts=["benchmark_run_config"],
        ),
        Capability(
            name="produce_dataset_pull_spec",
            title="Generate Dataset Pull Spec",
            node_type="workspace_action",
            ref="dataset_pull_spec",
            phase="produce",
            produces_artifacts=["dataset_pull_spec"],
            requires_channels=["web"],
        ),
        Capability(
            name="produce_dataset_loader_template",
            title="Generate Dataset Loader Template",
            node_type="workspace_action",
            ref="dataset_loader_template",
            phase="produce",
            produces_artifacts=["dataset_loader_template"],
            requires_channels=["web"],
        ),
        Capability(
            name="produce_webpage_blueprint",
            title="Generate Webpage Blueprint",
            node_type="workspace_action",
            ref="webpage_blueprint",
            phase="produce",
            produces_artifacts=["webpage_blueprint"],
        ),
        Capability(
            name="produce_slide_deck_plan",
            title="Generate Slide Deck Plan",
            node_type="workspace_action",
            ref="slide_deck_plan",
            phase="produce",
            produces_artifacts=["slide_deck_plan"],
        ),
        Capability(
            name="produce_chart_pack_spec",
            title="Generate Chart Pack Spec",
            node_type="workspace_action",
            ref="chart_pack_spec",
            phase="produce",
            produces_artifacts=["chart_pack_spec"],
        ),
        Capability(
            name="produce_podcast_episode_plan",
            title="Generate Podcast Episode Plan",
            node_type="workspace_action",
            ref="podcast_episode_plan",
            phase="produce",
            produces_artifacts=["podcast_episode_plan"],
        ),
        Capability(
            name="produce_video_storyboard",
            title="Generate Video Storyboard",
            node_type="workspace_action",
            ref="video_storyboard",
            phase="produce",
            produces_artifacts=["video_storyboard"],
        ),
        Capability(
            name="produce_image_prompt_pack",
            title="Generate Image Prompt Pack",
            node_type="workspace_action",
            ref="image_prompt_pack",
            phase="produce",
            produces_artifacts=["image_prompt_pack"],
        ),
        Capability(
            name="produce_data_analysis_spec",
            title="Generate Data Analysis Spec",
            node_type="workspace_action",
            ref="data_analysis_spec",
            phase="produce",
            produces_artifacts=["data_analysis_spec"],
        ),
    ]:
        registry.register(capability)
    return registry


def plan_capability_path(
    *,
    task_spec: TaskSpec,
    registry: CapabilityRegistry,
    world_state: TaskWorldState | None = None,
) -> dict[str, Any]:
    """Plan capabilities by satisfying state gaps instead of task-family templates."""

    state = world_state or TaskWorldState()
    gap = compute_state_gap(task_spec=task_spec, world_state=state)
    steps: list[dict[str, Any]] = []
    planned: set[str] = set()

    def add(name: str, reason: str) -> None:
        capability = registry.get(name)
        if capability is None or name in planned:
            return
        steps.append(capability.to_step(reason=reason))
        planned.add(name)

    if "workspace" in gap.missing_channels:
        add("observe_workspace", "workspace state is required to reduce uncertainty")
    if "web" in gap.missing_channels:
        add("collect_external_evidence", "external evidence is missing for this task")
        add("build_evidence_dossier", "normalize the retrieved evidence into reviewable records")
    if "risk" in gap.missing_channels:
        add("assess_risk", "risk/governance state is required before execution closes")
    if "discovery" in gap.missing_channels:
        add("discover_tools", "task is open-ended and should inspect available operators first")
        add("inspect_skill_catalog", "skill selection should come from explicit capability inspection")

    add("decompose_task", "translate goal and constraints into an executable plan")
    if task_spec.needs_validation and gap.missing_validation:
        add("plan_validation", "validation remains unsatisfied in the current state")

    artifact_to_capability = {
        "completion_packet": "produce_completion_packet",
        "patch_plan": "produce_patch_scaffold",
        "patch_draft": "produce_patch_draft",
        "benchmark_manifest": "produce_benchmark_manifest",
        "benchmark_run_config": "produce_benchmark_run_config",
        "dataset_pull_spec": "produce_dataset_pull_spec",
        "dataset_loader_template": "produce_dataset_loader_template",
        "webpage_blueprint": "produce_webpage_blueprint",
        "slide_deck_plan": "produce_slide_deck_plan",
        "chart_pack_spec": "produce_chart_pack_spec",
        "podcast_episode_plan": "produce_podcast_episode_plan",
        "video_storyboard": "produce_video_storyboard",
        "image_prompt_pack": "produce_image_prompt_pack",
        "data_analysis_spec": "produce_data_analysis_spec",
    }
    for artifact in gap.missing_artifacts:
        capability_name = artifact_to_capability.get(artifact)
        if capability_name:
            add(capability_name, f"artifact gap detected for {artifact}")

    return {
        "steps": steps,
        "required_channels": list(task_spec.required_channels),
        "required_artifacts": [item.kind for item in task_spec.artifact_contracts if item.required],
        "gap": gap.to_dict(),
    }


def build_world_state(*, graph: dict[str, Any], context: dict[str, Any]) -> TaskWorldState:
    """Build a compact world state from graph/context for replanning."""

    channels: list[str] = []
    artifacts: list[str] = []
    completed_capabilities: list[str] = []
    failures: list[str] = []
    validation_ok = False

    node_results = context.get("node_results", {}) if isinstance(context.get("node_results", {}), dict) else {}
    for node in graph.get("nodes", []) if isinstance(graph.get("nodes", []), list) else []:
        if not isinstance(node, dict) or str(node.get("status", "")) != "completed":
            continue
        node_id = str(node.get("node_id", ""))
        metrics = node.get("metrics", {}) if isinstance(node.get("metrics", {}), dict) else {}
        completed_capabilities.append(node_id)
        if node_id.startswith("workspace_") or node_id.startswith("action_patch") or metrics.get("action_kind") in {"patch_scaffold", "patch_draft"}:
            channels.append("workspace")
        if node_id in {"external_resources", "evidence"}:
            channels.append("web")
        if node_id == "risk":
            channels.append("risk")
        if node_id in {"capabilities", "skill_priors"}:
            channels.append("discovery")
        result = node_results.get(node_id, {}) if isinstance(node_results, dict) else {}
        artifact = result.get("artifact", {}) if isinstance(result, dict) else {}
        path = _norm(artifact.get("path", ""))
        if path:
            lowered = path.lower()
            if "packets/completion-packet" in lowered:
                artifacts.append("completion_packet")
            if "web/" in lowered:
                artifacts.append("webpage_blueprint")
            if "slides/" in lowered:
                artifacts.append("slide_deck_plan")
            if "charts/" in lowered:
                artifacts.append("chart_pack_spec")
            if "podcast/" in lowered:
                artifacts.append("podcast_episode_plan")
            if "video/" in lowered:
                artifacts.append("video_storyboard")
            if "images/" in lowered:
                artifacts.append("image_prompt_pack")
            if "analysis/data-analysis-spec" in lowered:
                artifacts.append("data_analysis_spec")
            if "benchmarks/manifest" in lowered:
                artifacts.append("benchmark_manifest")
            if "benchmarks/run-config" in lowered:
                artifacts.append("benchmark_run_config")
            if "datasets/pull-spec" in lowered:
                artifacts.append("dataset_pull_spec")
            if "datasets/loader_template" in lowered:
                artifacts.append("dataset_loader_template")
            if "patch-scaffold" in lowered:
                artifacts.append("patch_plan")
            if "patch-draft" in lowered:
                artifacts.append("patch_draft")
            if "validation" in lowered:
                artifacts.append("validation_plan")
        payload = result.get("result", {}) if isinstance(result, dict) else {}
        artifact_contract = payload.get("artifact_contract", {}) if isinstance(payload, dict) and isinstance(payload.get("artifact_contract", {}), dict) else {}
        custom_kind = _norm(artifact_contract.get("kind", ""))
        if custom_kind:
            artifacts.append(custom_kind)
        if node_id == "execution":
            results = payload.get("results", []) if isinstance(payload, dict) else []
            exit_codes = [int(item.get("exit_code", 1)) for item in results if isinstance(item, dict)]
            validation_ok = bool(exit_codes) and all(code == 0 for code in exit_codes)
            for item in results if isinstance(results, list) else []:
                if not isinstance(item, dict) or int(item.get("exit_code", 0)) == 0:
                    continue
                text = f"{item.get('stdout', '')} {item.get('stderr', '')}".lower()
                if "timeout" in text:
                    failures.append("timeout")
                elif "assert" in text or "traceback" in text or "failed" in text:
                    failures.append("assertion_failure")
                elif "no module named" in text or "command not found" in text or "importerror" in text:
                    failures.append("missing_dependency")
                else:
                    failures.append("execution_failure")
        if node_id == "evidence":
            artifacts.append("evidence_bundle")
        if node_id == "risk":
            artifacts.append("risk_register")
        if node_id == "completion_packet":
            artifacts.append("completion_packet")
        if node_id == "synthesis":
            artifacts.append("deliverable_report")

    return TaskWorldState(
        channels=_dedupe(channels),
        artifacts=_dedupe(artifacts),
        completed_capabilities=_dedupe(completed_capabilities),
        failures=_dedupe(failures),
        validation_ok=validation_ok,
    )


def compute_state_gap(*, task_spec: TaskSpec, world_state: TaskWorldState) -> StateGap:
    """Compute missing channels/artifacts/validation from the current world state."""

    required_artifacts = [item.kind for item in task_spec.artifact_contracts if item.required]
    return StateGap(
        missing_channels=[item for item in task_spec.required_channels if item not in world_state.channels],
        missing_artifacts=[item for item in required_artifacts if item not in world_state.artifacts],
        missing_validation=bool(task_spec.needs_validation and not world_state.validation_ok),
        failure_types=list(world_state.failures),
    )
