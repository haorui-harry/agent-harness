"""Mission-pack protocol for general-purpose agent task routing and delivery."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.tasking import infer_task_spec
from app.core.task_graph import ExecutableTaskGraph, TaskGraphArtifact, TaskGraphNode


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _humanize_action(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = re.sub(r"\bbecause artifact gap detected for\b", "to close the missing", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbecause\b", "-", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text[:1].upper() + text[1:] if text else ""


def _chunk_execution_tracks(execution_plan: list[str], limit: int = 3) -> list[list[str]]:
    rows: list[str] = []
    for item in execution_plan:
        clean = _humanize_action(item)
        if clean:
            rows.append(clean)
    if not rows:
        return []
    if len(rows) <= limit:
        return [[item] for item in rows[:limit]]
    chunks: list[list[str]] = [[] for _ in range(limit)]
    for index, item in enumerate(rows):
        chunks[index % limit].append(item)
    return [chunk for chunk in chunks if chunk]


def _dedupe(items: list[object], limit: int = 8) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for item in items:
        text = _clean_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
        if len(rows) >= limit:
            break
    return rows


def _answer_signal(final_answer: str) -> str:
    text = _clean_text(final_answer)
    if not text:
        return "runtime answer available"
    text = re.sub(r"\[[^\]]+\]", "", text).strip()
    text = re.sub(r"#+\s*", "", text).strip()
    if not text:
        return "runtime answer available"
    return (text[:87] + "...") if len(text) > 90 else text


@dataclass(frozen=True)
class MissionDeliverableBlueprint:
    """One user-visible deliverable emitted by a mission pack."""

    title: str
    description: str
    audience: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "description": self.description,
            "audience": self.audience,
        }


@dataclass(frozen=True)
class MissionProfile:
    """Generalized product profile for one class of user task."""

    name: str
    title: str
    summary: str
    primary_deliverable: str
    target_users: list[str]
    output_views: list[str]
    review_questions: list[str]
    deliverables: list[MissionDeliverableBlueprint] = field(default_factory=list)
    keyword_patterns: list[str] = field(default_factory=list)
    artifact_kinds: set[str] = field(default_factory=set)
    boundary_statement: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "primary_deliverable": self.primary_deliverable,
            "target_users": list(self.target_users),
            "output_views": list(self.output_views),
            "review_questions": list(self.review_questions),
            "deliverables": [item.to_dict() for item in self.deliverables],
            "keyword_patterns": list(self.keyword_patterns),
            "artifact_kinds": sorted(self.artifact_kinds),
            "boundary_statement": self.boundary_statement,
        }


class MissionRegistry:
    """Infer the best mission-pack profile for a query."""

    def __init__(self) -> None:
        self._profiles = self._defaults()
        self._profiles_by_name = {item.name: item for item in self._profiles}

    def infer(self, query: str, task_spec: dict[str, Any] | None = None) -> MissionProfile:
        spec_profile = self._infer_from_task_spec(task_spec or {})
        if spec_profile is not None:
            return spec_profile
        text = query.lower().strip()
        best = self._profiles[0]
        best_score = -1
        for profile in self._profiles:
            score = sum(1 for pattern in profile.keyword_patterns if re.search(pattern, text))
            if score > best_score:
                best = profile
                best_score = score
        return best

    def infer_from_query_context(
        self,
        query: str,
        *,
        task_spec: dict[str, Any] | None = None,
    ) -> MissionProfile:
        payload = task_spec or infer_task_spec(query=query).to_dict()
        return self.infer(query, task_spec=payload)

    def _infer_from_task_spec(self, task_spec: dict[str, Any]) -> MissionProfile | None:
        if not isinstance(task_spec, dict) or not task_spec:
            return None
        primary = _clean_text(task_spec.get("primary_artifact_kind", "")).lower()
        if not primary:
            return None
        for profile in self._profiles:
            if primary in profile.artifact_kinds:
                return profile
        return None

    def list_cards(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._profiles]

    def build_runtime_pack(
        self,
        query: str,
        run: Any,
        run_summary: dict[str, Any],
        profile: MissionProfile | None = None,
    ) -> dict[str, Any]:
        metadata = run.metadata if hasattr(run, "metadata") and isinstance(run.metadata, dict) else {}
        task_spec = metadata.get("task_spec", {}) if isinstance(metadata.get("task_spec", {}), dict) else {}
        profile = profile or self.infer(query, task_spec=task_spec)
        evidence = run_summary.get("evidence", {}) if isinstance(run_summary, dict) else {}
        value_card = run_summary.get("value_card", {}) if isinstance(run_summary, dict) else {}
        recipe = run_summary.get("recipe", {}) if isinstance(run_summary, dict) else {}
        live = run_summary.get("live_agent", {}) if isinstance(run_summary, dict) else {}
        security = run_summary.get("security", {}) if isinstance(run_summary, dict) else {}
        selected_agent = _clean_text(metadata.get("selected_agent", ""))
        selected_skills = metadata.get("selected_skills", []) if isinstance(metadata.get("selected_skills", []), list) else []
        execution_plan = _dedupe(run_summary.get("plan", []), limit=10)
        deliverables = self._runtime_deliverables(
            profile=profile,
            final_answer=_clean_text(getattr(run, "final_answer", "")),
            execution_plan=execution_plan,
            evidence=evidence,
            live=live,
        )
        review_questions = _dedupe(
            list(profile.review_questions)
            + [
                f"What evidence would invalidate the current {profile.title.lower()} recommendation?",
                "Which deliverable is ready for stakeholder review today?",
            ],
            limit=6,
        )
        runtime_state = {
            "completed": bool(getattr(run, "completed", False)),
            "selected_agent": selected_agent,
            "selected_skills": selected_skills[:8],
            "recipe": {
                "name": _clean_text(recipe.get("name", "")),
                "executed_steps": int(recipe.get("executed_steps", 0)),
                "total_steps": int(recipe.get("total_steps", 0)),
            },
            "value_index": float(value_card.get("value_index", 0.0)),
            "band": _clean_text(value_card.get("band", "")),
            "mode": _clean_text(metadata.get("mode", "")),
            "risk_level": _clean_text(metadata.get("risk_level", "")),
        }
        execution_tracks = self._runtime_execution_tracks(
            execution_plan=execution_plan,
            security=security,
            evidence=evidence,
        )
        honest_boundary = self._runtime_boundary(profile)
        pack = {
            "name": profile.name,
            "title": profile.title,
            "summary": profile.summary,
            "primary_deliverable": profile.primary_deliverable,
            "query": query,
            "target_users": list(profile.target_users),
            "output_views": list(profile.output_views),
            "deliverables": deliverables,
            "review_questions": review_questions,
            "execution_tracks": execution_tracks,
            "evidence_snapshot": {
                "record_count": int(evidence.get("record_count", 0)),
                "citation_count": int(evidence.get("citation_count", 0)),
                "citations": list(evidence.get("citations", []))[:6],
            },
            "runtime_state": runtime_state,
            "decision": {
                "status": "ready" if bool(getattr(run, "completed", False)) else "blocked",
                "reason": "runtime_completed" if bool(getattr(run, "completed", False)) else "runtime_incomplete",
                "selected_candidate": selected_agent,
                "value_index": float(value_card.get("value_index", 0.0)),
            },
            "honest_boundary": honest_boundary,
        }
        pack["task_graph"] = self._runtime_task_graph(
            query=query,
            profile=profile,
            runtime_state=runtime_state,
            execution_plan=execution_plan,
            deliverables=deliverables,
            evidence=pack["evidence_snapshot"],
            decision=pack["decision"],
            execution_tracks=execution_tracks,
        )
        return pack

    def build_release_pack(
        self,
        query: str,
        run: Any,
        run_summary: dict[str, Any],
        scenario: dict[str, Any],
        story: dict[str, Any],
        proposal: dict[str, Any],
        lab_payload: dict[str, Any],
        agent_comparison: dict[str, Any],
        profile: MissionProfile | None = None,
    ) -> dict[str, Any]:
        metadata = run.metadata if hasattr(run, "metadata") and isinstance(run.metadata, dict) else {}
        task_spec = metadata.get("task_spec", {}) if isinstance(metadata.get("task_spec", {}), dict) else {}
        base = self.build_runtime_pack(query=query, run=run, run_summary=run_summary, profile=profile)
        evidence = run_summary.get("evidence", {}) if isinstance(run_summary, dict) else {}
        release = lab_payload.get("release_decision", {}) if isinstance(lab_payload, dict) else {}
        target_users = _dedupe(
            list(proposal.get("target_users", []))
            + list(base.get("target_users", []))
            + [story.get("audience_takeaway", "")]
        )
        review_questions = _dedupe(
            list(base.get("review_questions", []))
            + [
                release.get("reason", ""),
                f"Does the evidence packet support expansion beyond {scenario.get('name', 'this mission')}?",
                f"Is {agent_comparison.get('winner', 'the selected agent')} still the right owner if constraints tighten?",
            ],
            limit=6,
        )
        execution_plan = _dedupe(proposal.get("execution_plan", []) or run_summary.get("plan", []), limit=10)
        deliverables = self._release_deliverables(
            base_rows=base.get("deliverables", []),
            execution_plan=execution_plan,
            evidence=evidence,
            release=release,
        )
        honest_boundary = self._runtime_boundary(profile or self.infer(query, task_spec=task_spec))
        base.update(
            {
                "target_users": target_users[:5],
                "review_questions": review_questions,
                "execution_tracks": self._release_execution_tracks(proposal=proposal, execution_plan=execution_plan),
                "deliverables": deliverables,
                "decision": {
                    "status": release.get("decision", "block"),
                    "reason": _clean_text(release.get("reason", "")),
                    "selected_candidate": _clean_text(release.get("selected_candidate", "")),
                    "value_index": float(run_summary.get("value_card", {}).get("value_index", 0.0)),
                },
                "honest_boundary": honest_boundary,
                "release_context": {
                    "scenario_name": _clean_text(scenario.get("name", "")),
                    "theme": _clean_text(story.get("theme", "")),
                    "headline": _clean_text(proposal.get("headline", "")),
                },
            }
        )
        base["task_graph"] = self._release_task_graph(
            query=query,
            profile=profile or self.infer(query, task_spec=task_spec),
            execution_plan=execution_plan,
            deliverables=deliverables,
            evidence=base.get("evidence_snapshot", {}),
            decision=base.get("decision", {}),
            release_context=base.get("release_context", {}),
            execution_tracks=base.get("execution_tracks", []),
            release=release,
            proposal=proposal,
        )
        return base

    @staticmethod
    def _runtime_deliverables(
        profile: MissionProfile,
        final_answer: str,
        execution_plan: list[str],
        evidence: dict[str, Any],
        live: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in profile.deliverables:
            signal = ""
            lowered = item.title.lower()
            if "evidence" in lowered:
                signal = f"{int(evidence.get('record_count', 0))} records / {int(evidence.get('citation_count', 0))} citations"
            elif "execution" in lowered or "playbook" in lowered or "timeline" in lowered or "migration" in lowered:
                signal = f"{len(execution_plan)} execution steps"
            elif "validation" in lowered:
                signal = "validation path attached"
            elif "decision" in lowered or "brief" in lowered or "spec" in lowered:
                signal = _answer_signal(final_answer)
            rows.append(
                {
                    **item.to_dict(),
                    "status": "ready" if final_answer else "draft",
                    "evidence_hint": _clean_text(signal),
                    "live_backed": bool(live.get("success", False)),
                }
            )
        return rows

    @staticmethod
    def _runtime_execution_tracks(
        execution_plan: list[str],
        security: dict[str, Any],
        evidence: dict[str, Any],
    ) -> list[dict[str, Any]]:
        tracks: list[dict[str, Any]] = []
        for index, rows in enumerate(_chunk_execution_tracks(execution_plan), start=1):
            if not rows:
                continue
            tracks.append(
                {
                    "name": f"Track {index}",
                    "focus": " | ".join(rows[:2]),
                    "success": (
                        f"preflight={security.get('preflight_action', '')}; "
                        f"evidence={int(evidence.get('record_count', 0))} records"
                    ),
                }
            )
        return tracks

    @staticmethod
    def _runtime_boundary(profile: MissionProfile) -> str:
        if profile.boundary_statement:
            return profile.boundary_statement
        return (
            "Current strength is packaging execution, evidence, and review structure in one runtime artifact. "
            "Weaker on tasks requiring sustained real-world environment interaction."
        )

    @staticmethod
    def _release_deliverables(
        base_rows: list[dict[str, Any]],
        execution_plan: list[str],
        evidence: dict[str, Any],
        release: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in base_rows:
            row = dict(item)
            title = str(row.get("title", "")).lower()
            if "validation" in title:
                row["evidence_hint"] = _clean_text(release.get("decision", "block"))
            elif "evidence" in title:
                row["evidence_hint"] = f"{int(evidence.get('record_count', 0))} records / {int(evidence.get('citation_count', 0))} citations"
            elif "execution" in title or "playbook" in title or "timeline" in title:
                row["evidence_hint"] = f"{len(execution_plan)} execution steps"
            row["status"] = "ready" if execution_plan else row.get("status", "draft")
            rows.append(row)
        return rows

    @staticmethod
    def _release_execution_tracks(proposal: dict[str, Any], execution_plan: list[str]) -> list[dict[str, Any]]:
        tracks: list[dict[str, Any]] = []
        for index, rows in enumerate(_chunk_execution_tracks(execution_plan), start=1):
            tracks.append(
                {
                    "name": f"Track {index}",
                    "focus": " | ".join(rows[:2]),
                    "success": "Produce reviewable artifacts with evidence and a clear delivery boundary.",
                }
            )
        if tracks:
            return tracks
        phases = proposal.get("phases", []) if isinstance(proposal, dict) else []
        for index, item in enumerate(execution_plan[:3], start=1):
            tracks.append(
                {
                    "name": f"Track {index}",
                    "focus": _humanize_action(item),
                    "success": "Completed with trace, evidence, and reviewable outputs.",
                }
            )
        if tracks:
            return tracks
        for index, phase in enumerate(phases[:3], start=1):
            tracks.append(
                {
                    "name": f"Track {index}",
                    "focus": _clean_text(", ".join(phase.get("actions", [])[:2])),
                    "success": _clean_text(", ".join(phase.get("success_metrics", [])[:3])),
                }
            )
        return tracks

    def _runtime_task_graph(
        self,
        query: str,
        profile: MissionProfile,
        runtime_state: dict[str, Any],
        execution_plan: list[str],
        deliverables: list[dict[str, Any]],
        evidence: dict[str, Any],
        decision: dict[str, Any],
        execution_tracks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        completed = bool(runtime_state.get("completed", False))
        evidence_count = int(evidence.get("record_count", 0))
        citation_count = int(evidence.get("citation_count", 0))
        evidence_status = "completed" if evidence_count or citation_count or completed else "ready"
        selected_skills = runtime_state.get("selected_skills", [])
        recipe = runtime_state.get("recipe", {}) if isinstance(runtime_state.get("recipe", {}), dict) else {}
        recipe_name = _clean_text(recipe.get("name", ""))
        recipe_progress = f"{int(recipe.get('executed_steps', 0))}/{int(recipe.get('total_steps', 0))}"

        nodes = [
            TaskGraphNode(
                node_id="scope_mission",
                title="Scope mission and choose operator",
                node_type="routing",
                status="completed" if runtime_state.get("selected_agent") else "ready",
                notes=[
                    f"selected agent: {runtime_state.get('selected_agent', '') or 'pending'}",
                    f"mode: {runtime_state.get('mode', '') or 'balanced'}",
                    f"skills: {len(selected_skills)}",
                ],
                artifacts=[
                    TaskGraphArtifact(
                        kind="routing_contract",
                        label="Mission routing",
                        status="completed" if runtime_state.get("selected_agent") else "ready",
                        summary=f"recipe={recipe_name or 'auto'} progress={recipe_progress}",
                    )
                ],
                metrics={
                    "selected_skills": len(selected_skills),
                    "value_index": float(runtime_state.get("value_index", 0.0)),
                },
            ),
            TaskGraphNode(
                node_id="assemble_evidence",
                title="Assemble evidence and runtime signals",
                node_type="evidence",
                status=evidence_status,
                depends_on=["scope_mission"],
                notes=[
                    f"records={evidence_count}",
                    f"citations={citation_count}",
                ],
                artifacts=[
                    TaskGraphArtifact(
                        kind="evidence_packet",
                        label="Evidence packet",
                        status=evidence_status,
                        summary=f"{evidence_count} records / {citation_count} citations",
                    )
                ],
                metrics={
                    "record_count": evidence_count,
                    "citation_count": citation_count,
                },
            ),
            TaskGraphNode(
                node_id="build_primary",
                title=f"Build {profile.primary_deliverable}",
                node_type="synthesis",
                status="completed" if completed else "ready",
                depends_on=["scope_mission", "assemble_evidence"],
                commands=execution_plan[:3],
                notes=[
                    f"risk level: {runtime_state.get('risk_level', '') or 'unknown'}",
                    f"band: {runtime_state.get('band', '') or 'unscored'}",
                ],
                artifacts=[
                    TaskGraphArtifact(
                        kind="primary_deliverable",
                        label=profile.primary_deliverable,
                        status="completed" if completed else "ready",
                        summary=profile.summary,
                    )
                ],
                metrics={
                    "plan_steps": len(execution_plan),
                    "track_count": len(execution_tracks),
                },
            ),
            TaskGraphNode(
                node_id="package_views",
                title="Package user-facing deliverable bundle",
                node_type="packaging",
                status="completed" if completed else "ready",
                depends_on=["build_primary"],
                notes=[f"output views: {len(profile.output_views)}"],
                artifacts=self._deliverable_artifacts(deliverables),
                metrics={
                    "deliverable_count": len(deliverables),
                    "ready_deliverables": sum(1 for item in deliverables if item.get("status") == "ready"),
                },
            ),
            TaskGraphNode(
                node_id="governance_review",
                title="Governance and publish review",
                node_type="review",
                status="completed" if decision.get("status") == "ready" else "ready",
                depends_on=["package_views"],
                notes=[
                    f"decision={decision.get('status', '')}",
                    f"reason={decision.get('reason', '')}",
                ],
                artifacts=[
                    TaskGraphArtifact(
                        kind="decision_gate",
                        label="Mission decision",
                        status="completed" if decision.get("status") == "ready" else "ready",
                        summary=f"{decision.get('status', '')}: {decision.get('reason', '')}",
                    )
                ],
                metrics={
                    "execution_tracks": len(execution_tracks),
                },
            ),
        ]
        graph = ExecutableTaskGraph(
            graph_id=f"{profile.name}-runtime-graph",
            mission_type=profile.name,
            query=query,
            nodes=nodes,
        )
        return graph.to_dict()

    def _release_task_graph(
        self,
        query: str,
        profile: MissionProfile,
        execution_plan: list[str],
        deliverables: list[dict[str, Any]],
        evidence: dict[str, Any],
        decision: dict[str, Any],
        release_context: dict[str, Any],
        execution_tracks: list[dict[str, Any]],
        release: dict[str, Any],
        proposal: dict[str, Any],
    ) -> dict[str, Any]:
        release_status = _clean_text(release.get("decision", decision.get("status", "")))
        evidence_count = int(evidence.get("record_count", 0))
        citation_count = int(evidence.get("citation_count", 0))
        evidence_status = "completed" if evidence_count or citation_count or release_status else "ready"
        phases = proposal.get("phases", []) if isinstance(proposal, dict) else []

        nodes = [
            TaskGraphNode(
                node_id="frame_release",
                title="Frame release narrative and scenario boundary",
                node_type="framing",
                status="completed" if release_context else "ready",
                notes=[
                    f"scenario={release_context.get('scenario_name', '')}",
                    f"theme={release_context.get('theme', '')}",
                    f"headline={release_context.get('headline', '')}",
                ],
                artifacts=[
                    TaskGraphArtifact(
                        kind="release_context",
                        label="Release framing",
                        status="completed" if release_context else "ready",
                        summary=_clean_text(release_context.get("headline", profile.primary_deliverable)),
                    )
                ],
                metrics={"phase_count": len(phases)},
            ),
            TaskGraphNode(
                node_id="collect_evidence",
                title="Collect release-grade evidence bundle",
                node_type="evidence",
                status=evidence_status,
                depends_on=["frame_release"],
                notes=[f"records={evidence_count}", f"citations={citation_count}"],
                artifacts=[
                    TaskGraphArtifact(
                        kind="evidence_packet",
                        label="Evidence bundle",
                        status=evidence_status,
                        summary=f"{evidence_count} records / {citation_count} citations",
                    )
                ],
                metrics={"record_count": evidence_count, "citation_count": citation_count},
            ),
            TaskGraphNode(
                node_id="orchestrate_tracks",
                title="Orchestrate multi-track execution",
                node_type="execution_plan",
                status="completed" if execution_tracks or execution_plan else "ready",
                depends_on=["frame_release"],
                commands=execution_plan[:4],
                notes=[
                    _clean_text(f"{item.get('name', '')}: {item.get('focus', '')}")
                    for item in execution_tracks[:3]
                ],
                artifacts=[
                    TaskGraphArtifact(
                        kind="execution_tracks",
                        label="Execution tracks",
                        status="completed" if execution_tracks or execution_plan else "ready",
                        summary=f"{len(execution_tracks) or min(len(execution_plan), 3)} tracks prepared",
                    )
                ],
                metrics={"track_count": len(execution_tracks), "plan_steps": len(execution_plan)},
            ),
            TaskGraphNode(
                node_id="package_release",
                title="Package release deliverables",
                node_type="packaging",
                status="completed" if deliverables else "ready",
                depends_on=["collect_evidence", "orchestrate_tracks"],
                artifacts=self._deliverable_artifacts(deliverables),
                metrics={"deliverable_count": len(deliverables)},
            ),
            TaskGraphNode(
                node_id="release_decision",
                title="Finalize promotion or block decision",
                node_type="review",
                status="completed" if release_status in {"promote", "ship", "ready", "approve"} else "ready",
                depends_on=["package_release"],
                notes=[
                    f"decision={release_status or 'block'}",
                    f"reason={decision.get('reason', '')}",
                ],
                artifacts=[
                    TaskGraphArtifact(
                        kind="release_decision",
                        label="Release decision",
                        status="completed" if release_status in {"promote", "ship", "ready", "approve"} else "ready",
                        summary=f"{release_status or 'block'}: {decision.get('reason', '')}",
                    )
                ],
                metrics={"phases": len(phases)},
            ),
        ]
        graph = ExecutableTaskGraph(
            graph_id=f"{profile.name}-release-graph",
            mission_type=f"{profile.name}_release",
            query=query,
            nodes=nodes,
        )
        return graph.to_dict()

    @staticmethod
    def _deliverable_artifacts(deliverables: list[dict[str, Any]]) -> list[TaskGraphArtifact]:
        rows: list[TaskGraphArtifact] = []
        for item in deliverables[:6]:
            rows.append(
                TaskGraphArtifact(
                    kind="deliverable",
                    label=str(item.get("title", "")),
                    status=str(item.get("status", "draft")),
                    summary=_clean_text(str(item.get("description", "")) or str(item.get("evidence_hint", ""))),
                )
            )
        return rows

    @staticmethod
    def _defaults() -> list[MissionProfile]:
        return [
            MissionProfile(
                name="general",
                title="General Task Pack",
                summary="Universal task execution with planning, evidence collection, and deliverable packaging.",
                primary_deliverable="Task result with execution trace, evidence, and reviewable artifacts.",
                target_users=["requester", "reviewer", "operator"],
                output_views=["result summary", "execution trace", "evidence packet", "deliverable bundle"],
                review_questions=[
                    "Does the result directly answer the original request?",
                    "What evidence supports the conclusion?",
                    "What remains unverified or incomplete?",
                ],
                deliverables=[
                    MissionDeliverableBlueprint("Task Result", "Primary output addressing the user request.", "requester"),
                    MissionDeliverableBlueprint("Execution Trace", "Step-by-step record of actions taken and tools used.", "reviewer"),
                    MissionDeliverableBlueprint("Evidence Packet", "Supporting data, citations, and runtime signals.", "auditor"),
                    MissionDeliverableBlueprint("Delivery Bundle", "Packaged artifacts ready for downstream use.", "operator"),
                ],
                keyword_patterns=[r".*"],
                artifact_kinds={
                    "webpage_blueprint", "slide_deck_plan", "podcast_episode_plan",
                    "video_storyboard", "image_prompt_pack", "chart_pack_spec",
                    "data_analysis_spec", "dataset_pull_spec", "dataset_loader_template",
                    "custom:decision_memo", "custom:executive_memo", "custom:launch_memo",
                    "custom:one_pager", "custom:checklist", "runbook", "risk_register",
                },
                boundary_statement=(
                    "Capable of planning, analysis, evidence gathering, and artifact packaging. "
                    "Weaker on sustained real-world environment interaction and long-running execution loops."
                ),
            ),
            MissionProfile(
                name="research",
                title="Research Pack",
                summary="Evidence-driven research with hypothesis framing, source collection, and synthesis.",
                primary_deliverable="Research output with evidence anchors, analysis, and actionable findings.",
                target_users=["researcher", "analyst", "decision-maker"],
                output_views=["research brief", "evidence packet", "analysis summary", "recommendation"],
                review_questions=[
                    "Is the core finding supported by collected evidence?",
                    "What evidence would invalidate the conclusion?",
                    "What follow-up investigation is needed?",
                ],
                deliverables=[
                    MissionDeliverableBlueprint("Research Brief", "Core findings and supporting evidence.", "researcher"),
                    MissionDeliverableBlueprint("Evidence Collection", "Sources, citations, and data points.", "reviewer"),
                    MissionDeliverableBlueprint("Analysis Summary", "Structured interpretation of findings.", "decision-maker"),
                    MissionDeliverableBlueprint("Recommendations", "Actionable next steps based on evidence.", "operator"),
                ],
                keyword_patterns=[r"(research|study|paper|experiment|evidence|analysis|investigate|evaluate|compare|survey)"],
                artifact_kinds={
                    "deliverable_report", "custom:brief", "custom:memo",
                    "workspace_findings", "evidence_bundle",
                },
                boundary_statement=(
                    "Strong at evidence gathering, synthesis, and structured analysis. "
                    "Weaker on tasks requiring real-time data access or experimental execution."
                ),
            ),
            MissionProfile(
                name="implementation",
                title="Implementation Pack",
                summary="Engineering-oriented execution with code analysis, planning, and validation.",
                primary_deliverable="Implementation plan with architecture decisions, code changes, and validation steps.",
                target_users=["engineer", "tech lead", "code reviewer"],
                output_views=["architecture spec", "implementation plan", "validation checklist", "code artifacts"],
                review_questions=[
                    "Does the implementation address the root cause?",
                    "What tests validate the change?",
                    "What risks does the change introduce?",
                ],
                deliverables=[
                    MissionDeliverableBlueprint("Implementation Plan", "Step-by-step execution path with dependencies.", "engineer"),
                    MissionDeliverableBlueprint("Code Artifacts", "Patches, specs, or scaffolding produced.", "code reviewer"),
                    MissionDeliverableBlueprint("Validation Gates", "Tests and checks that confirm correctness.", "tech lead"),
                    MissionDeliverableBlueprint("Risk Assessment", "Failure modes and mitigation strategies.", "operator"),
                ],
                keyword_patterns=[r"(code|implement|build|fix|refactor|migrate|deploy|architecture|design|engineer|develop|debug|patch)"],
                artifact_kinds={"patch_plan", "patch_draft"},
                boundary_statement=(
                    "Strong at code analysis, architecture planning, and validation design. "
                    "Weaker on executing long multi-file changes or maintaining live environment stability."
                ),
            ),
        ]
