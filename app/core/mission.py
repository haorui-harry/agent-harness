"""Shared mission-pack protocol for runtime and showcase outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.task_graph import ExecutableTaskGraph, TaskGraphArtifact, TaskGraphNode


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


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
class BenchmarkTargetBlueprint:
    """Benchmark family that is relevant to a mission type."""

    name: str
    fit: str
    strength: str
    gap: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "fit": self.fit,
            "strength": self.strength,
            "gap": self.gap,
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
    benchmark_targets: list[BenchmarkTargetBlueprint] = field(default_factory=list)
    keyword_patterns: list[str] = field(default_factory=list)

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
            "benchmark_targets": [item.to_dict() for item in self.benchmark_targets],
            "keyword_patterns": list(self.keyword_patterns),
        }


class MissionRegistry:
    """Infer the best mission-pack profile for a query."""

    def __init__(self) -> None:
        self._profiles = self._defaults()

    def infer(self, query: str) -> MissionProfile:
        text = query.lower().strip()
        best = self._profiles[0]
        best_score = -1
        for profile in self._profiles:
            score = sum(1 for pattern in profile.keyword_patterns if re.search(pattern, text))
            if score > best_score:
                best = profile
                best_score = score
        return best

    def list_cards(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._profiles]

    def build_runtime_pack(
        self,
        query: str,
        run: Any,
        run_summary: dict[str, Any],
        profile: MissionProfile | None = None,
    ) -> dict[str, Any]:
        profile = profile or self.infer(query)
        metadata = run.metadata if hasattr(run, "metadata") and isinstance(run.metadata, dict) else {}
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
                "Which benchmark family is the right proof target for this mission?",
            ],
            limit=6,
        )
        benchmark_targets = self._runtime_benchmark_targets(
            profile=profile,
            completed=bool(getattr(run, "completed", False)),
            live=live,
            selected_skills=selected_skills,
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
            "benchmark_targets": benchmark_targets,
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
            benchmark_targets=benchmark_targets,
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
        benchmark_targets = self._release_benchmark_targets(
            base_rows=base.get("benchmark_targets", []),
            release=release,
        )
        honest_boundary = (
            "Current strength is evidence-backed planning, governance framing, and packaged delivery. "
            "It is not yet a leaderboard winner on web navigation or code-fix benchmarks because the repo "
            "still lacks full browser-actuation loops and code-task specific execution traces."
        )
        base.update(
            {
                "target_users": target_users[:5],
                "review_questions": review_questions,
                "execution_tracks": self._release_execution_tracks(proposal=proposal, execution_plan=execution_plan),
                "deliverables": deliverables,
                "benchmark_targets": benchmark_targets,
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
            profile=profile or self.infer(query),
            execution_plan=execution_plan,
            deliverables=deliverables,
            benchmark_targets=benchmark_targets,
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
            elif "benchmark" in lowered or "validation" in lowered:
                signal = "benchmark family mapped"
            elif "decision" in lowered or "brief" in lowered or "spec" in lowered:
                signal = final_answer[:120] or "runtime answer available"
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
        buckets = [
            ("Track 1", execution_plan[:2]),
            ("Track 2", execution_plan[2:4]),
            ("Track 3", execution_plan[4:6]),
        ]
        for name, rows in buckets:
            if not rows:
                continue
            tracks.append(
                {
                    "name": name,
                    "focus": ", ".join(rows[:2]),
                    "success": (
                        f"preflight={security.get('preflight_action', '')}; "
                        f"evidence={int(evidence.get('record_count', 0))} records"
                    ),
                }
            )
        return tracks

    @staticmethod
    def _runtime_benchmark_targets(
        profile: MissionProfile,
        completed: bool,
        live: dict[str, Any],
        selected_skills: list[Any],
    ) -> list[dict[str, Any]]:
        rows = [item.to_dict() for item in profile.benchmark_targets]
        for row in rows:
            row["current_status"] = "mapped"
            row["current_signal"] = "runtime completed" if completed else "runtime incomplete"
            if row["name"] in {"WebArena", "SWE-bench Verified"}:
                row["current_signal"] = "not yet directly exercised"
            if bool(live.get("success", False)):
                row["current_signal"] += "; live model path exercised"
            if selected_skills:
                row["current_signal"] += f"; skills={len(selected_skills)}"
        return rows

    @staticmethod
    def _runtime_boundary(profile: MissionProfile) -> str:
        if profile.name == "implementation_pack":
            return (
                "Current implementation missions can produce specs, migration plans, and validation checklists, "
                "but they are not yet equivalent to a full code-repair benchmark loop."
            )
        if profile.name == "research_pack":
            return (
                "Current research missions are strong on packaging evidence and promotion logic, "
                "but still need public benchmark execution history to claim superiority."
            )
        return (
            "Current strength is packaging execution, evidence, and review structure in one runtime artifact. "
            "It is weaker on environments that require full browser or repository action loops."
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
            if "benchmark" in title or "validation" in title:
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
        phases = proposal.get("phases", []) if isinstance(proposal, dict) else []
        tracks: list[dict[str, Any]] = []
        for phase in phases[:3]:
            tracks.append(
                {
                    "name": _clean_text(phase.get("phase", "Execution Track")),
                    "focus": _clean_text(", ".join(phase.get("actions", [])[:2])),
                    "success": _clean_text(", ".join(phase.get("success_metrics", [])[:3])),
                }
            )
        if tracks:
            return tracks
        for index, item in enumerate(execution_plan[:3], start=1):
            tracks.append(
                {
                    "name": f"Track {index}",
                    "focus": item,
                    "success": "Completed with trace, evidence, and reviewable outputs.",
                }
            )
        return tracks

    @staticmethod
    def _release_benchmark_targets(
        base_rows: list[dict[str, Any]],
        release: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in base_rows:
            row = dict(item)
            row["current_status"] = "mapped"
            row["current_signal"] = (
                "release-gated with evidence"
                if row.get("name", "") in {"GAIA", "TAU-bench", "TheAgentCompany"}
                else "partially covered"
            )
            if release:
                row["current_signal"] += f"; release={release.get('decision', 'block')}"
            rows.append(row)
        return rows

    def _runtime_task_graph(
        self,
        query: str,
        profile: MissionProfile,
        runtime_state: dict[str, Any],
        execution_plan: list[str],
        deliverables: list[dict[str, Any]],
        benchmark_targets: list[dict[str, Any]],
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
                node_id="benchmark_gate",
                title="Map benchmark and verification gate",
                node_type="evaluation",
                status="completed" if benchmark_targets else "ready",
                depends_on=["build_primary"],
                notes=[
                    str(item.get("current_signal", ""))
                    for item in benchmark_targets[:3]
                    if str(item.get("current_signal", ""))
                ],
                artifacts=self._benchmark_artifacts(benchmark_targets),
                metrics={"benchmark_targets": len(benchmark_targets)},
            ),
            TaskGraphNode(
                node_id="governance_review",
                title="Governance and publish review",
                node_type="review",
                status="completed" if decision.get("status") == "ready" else "ready",
                depends_on=["package_views", "benchmark_gate"],
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
        benchmark_targets: list[dict[str, Any]],
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
                node_id="evaluate_benchmarks",
                title="Evaluate benchmark positioning and release gate",
                node_type="evaluation",
                status="completed" if benchmark_targets else "ready",
                depends_on=["package_release"],
                notes=[
                    str(item.get("current_signal", ""))
                    for item in benchmark_targets[:3]
                    if str(item.get("current_signal", ""))
                ],
                artifacts=self._benchmark_artifacts(benchmark_targets),
                metrics={"benchmark_targets": len(benchmark_targets)},
            ),
            TaskGraphNode(
                node_id="release_decision",
                title="Finalize promotion or block decision",
                node_type="review",
                status="completed" if release_status in {"promote", "ship", "ready", "approve"} else "ready",
                depends_on=["evaluate_benchmarks"],
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
    def _benchmark_artifacts(benchmark_targets: list[dict[str, Any]]) -> list[TaskGraphArtifact]:
        rows: list[TaskGraphArtifact] = []
        for item in benchmark_targets[:5]:
            rows.append(
                TaskGraphArtifact(
                    kind="benchmark_target",
                    label=str(item.get("name", "")),
                    status=str(item.get("current_status", "mapped")),
                    summary=_clean_text(str(item.get("current_signal", "")) or str(item.get("gap", ""))),
                )
            )
        return rows

    @staticmethod
    def _defaults() -> list[MissionProfile]:
        return [
            MissionProfile(
                name="creative_pack",
                title="Creative Media Mission Pack",
                summary="Multi-format package for landing pages, slide decks, visuals, audio, and launch storytelling.",
                primary_deliverable="Experience pack with webpage blueprint, deck arc, media scripts, and visual directions.",
                target_users=["product marketer", "designer", "creative lead", "founder"],
                output_views=["landing page", "slide deck", "video storyboard", "image prompt pack"],
                review_questions=[
                    "Does the first screen explain the audience, value, and output in one glance?",
                    "Which media asset proves the story instead of repeating the slogan?",
                    "What artifact can design or marketing execute next without reinterpretation?",
                ],
                deliverables=[
                    MissionDeliverableBlueprint("Webpage Blueprint", "Hero structure, page sections, proof blocks, and CTA design.", "design and growth"),
                    MissionDeliverableBlueprint("Slide Deck Plan", "Slide-by-slide arc for demos, launches, and executive reviews.", "founder and product marketing"),
                    MissionDeliverableBlueprint("Media Storyboard", "Podcast or video segment structure with beats and proof moments.", "content team"),
                    MissionDeliverableBlueprint("Visual Direction Pack", "Prompt-ready visual directions for hero art, posters, and diagrams.", "creative ops"),
                ],
                benchmark_targets=[
                    BenchmarkTargetBlueprint("TAU-bench", "medium", "Useful for multi-step creative task packaging with tool choices.", "No direct media-generation benchmark loop yet."),
                    BenchmarkTargetBlueprint("TheAgentCompany", "medium", "Matches cross-functional knowledge-work packaging.", "Needs richer long-horizon editing and review state."),
                    BenchmarkTargetBlueprint("GAIA", "low", "Can partially validate evidence-backed content planning.", "Not a direct creative-output benchmark."),
                ],
                keyword_patterns=[r"(webpage|website|landing|frontend|ui|slide|deck|presentation|ppt|podcast|video|storyboard|image|poster|illustration)"],
            ),
            MissionProfile(
                name="analytics_pack",
                title="Analytics Mission Pack",
                summary="Decision-oriented package for data analysis, charting, dashboard framing, and evidence-backed readouts.",
                primary_deliverable="Analysis pack with data questions, chart portfolio, dashboard narrative, and validation notes.",
                target_users=["analyst", "operator", "research lead", "executive reviewer"],
                output_views=["analysis brief", "chart pack", "dashboard spec", "evidence notes"],
                review_questions=[
                    "Which metric actually drives the decision and which metric is only diagnostic?",
                    "Where are the outliers or segments that would invalidate the headline?",
                    "What chart pack or dashboard view is ready for stakeholder consumption today?",
                ],
                deliverables=[
                    MissionDeliverableBlueprint("Analysis Question Set", "Decision questions, cohorts, metrics, and failure checks.", "analytics owner"),
                    MissionDeliverableBlueprint("Chart Portfolio", "Reusable chart specs with data contracts and captions.", "analyst and product"),
                    MissionDeliverableBlueprint("Dashboard Narrative", "How to sequence metrics, alerts, and annotations on the surface.", "ops and leadership"),
                    MissionDeliverableBlueprint("Data Pull Spec", "Reproducible collection rules and quality checks for the dataset.", "data engineering"),
                ],
                benchmark_targets=[
                    BenchmarkTargetBlueprint("GAIA", "medium", "Good fit for evidence-backed multi-step analysis framing.", "Needs stronger public run history on external datasets."),
                    BenchmarkTargetBlueprint("TAU-bench", "medium", "Useful for operational analytics and workflow-oriented analysis loops.", "Needs live connectors into business systems."),
                    BenchmarkTargetBlueprint("TheAgentCompany", "medium", "Fits workplace analysis and reporting tasks.", "Needs stronger persistent memory around datasets and iterations."),
                ],
                keyword_patterns=[r"(data|dataset|analytics|analysis|chart|graph|plot|dashboard|sql|csv|cohort|visualization)"],
            ),
            MissionProfile(
                name="strategy_pack",
                title="Strategy Mission Pack",
                summary="Business-facing package for launch, rollout, and investment decisions.",
                primary_deliverable="Launch strategy packet with execution, evidence, and release gate.",
                target_users=["product lead", "operations lead", "risk owner", "executive sponsor"],
                output_views=["proposal", "execution plan", "evidence packet", "benchmark positioning"],
                review_questions=[
                    "Is the chosen wedge narrow enough to execute and large enough to matter?",
                    "Which release gate blocks expansion first?",
                    "What evidence is still missing for executive sign-off?",
                ],
                deliverables=[
                    MissionDeliverableBlueprint("Decision Memo", "One-page business recommendation with tradeoffs and target wedge.", "executive sponsor"),
                    MissionDeliverableBlueprint("Execution Playbook", "Phased rollout with operators, checkpoints, and fallback path.", "product and operations"),
                    MissionDeliverableBlueprint("Evidence Packet", "Citations, policy references, and runtime signals behind the claim.", "risk and procurement"),
                    MissionDeliverableBlueprint("Interop Export", "External skill-compatible bundle for downstream ecosystems.", "platform team"),
                ],
                benchmark_targets=[
                    BenchmarkTargetBlueprint("GAIA", "medium", "Good match for evidence-backed multi-step reasoning.", "Needs stronger open-web retrieval verification."),
                    BenchmarkTargetBlueprint("TAU-bench", "high", "Strong fit for enterprise workflow planning and tool orchestration.", "Needs deeper real connector coverage."),
                    BenchmarkTargetBlueprint("TheAgentCompany", "medium", "Good fit for knowledge-work packaging and operating decisions.", "Needs richer long-horizon workplace state."),
                ],
                keyword_patterns=[r"(launch|rollout|strategy|board|proposal|market|growth|copilot|enterprise|plan)"],
            ),
            MissionProfile(
                name="research_pack",
                title="Research Mission Pack",
                summary="Research-facing package for study design, evidence review, and promotion decisions.",
                primary_deliverable="Research promotion packet with experiment rationale, evidence, and release criteria.",
                target_users=["research lead", "applied scientist", "review committee"],
                output_views=["research brief", "benchmark report", "promotion packet", "risk register"],
                review_questions=[
                    "Is the claimed gain reproducible beyond the current scenario set?",
                    "Which missing evidence would most likely change the promotion decision?",
                    "What post-launch monitoring is required to validate the lab result?",
                ],
                deliverables=[
                    MissionDeliverableBlueprint("Research Brief", "Hypothesis, operating thesis, and study implications.", "research committee"),
                    MissionDeliverableBlueprint("Benchmark Readout", "Release gate, leaderboard, and evidence trail.", "lab leadership"),
                    MissionDeliverableBlueprint("Promotion Checklist", "What must pass before the result becomes default.", "release committee"),
                    MissionDeliverableBlueprint("Evidence Packet", "External and internal citations linked to the claim.", "reviewers"),
                ],
                benchmark_targets=[
                    BenchmarkTargetBlueprint("GAIA", "high", "Reasoning + retrieval alignment is directly relevant.", "Needs public benchmark execution history."),
                    BenchmarkTargetBlueprint("TheAgentCompany", "medium", "Useful for broader knowledge-work research tasks.", "Needs richer interactive environment state."),
                    BenchmarkTargetBlueprint("WebArena", "low", "Only partial overlap through retrieval and action planning.", "Needs real browser action loops."),
                ],
                keyword_patterns=[r"(research|study|paper|benchmark|experiment|lab|evaluation|hypothesis)"],
            ),
            MissionProfile(
                name="operations_pack",
                title="Operations Mission Pack",
                summary="Operations-facing package for daily execution, dependency control, and governance checkpoints.",
                primary_deliverable="Operational playbook with owners, checkpoints, and escalation paths.",
                target_users=["ops manager", "program manager", "service owner"],
                output_views=["playbook", "timeline", "dependency map", "risk register"],
                review_questions=[
                    "Which dependency can stall execution first?",
                    "Where does human override enter the loop?",
                    "What is the rollback path if live metrics degrade?",
                ],
                deliverables=[
                    MissionDeliverableBlueprint("Operational Playbook", "Practical sequence of actions and handoffs.", "delivery owner"),
                    MissionDeliverableBlueprint("Dependency Timeline", "Critical path and checkpoint plan.", "program manager"),
                    MissionDeliverableBlueprint("Risk Register", "Failure modes, control points, and escalation logic.", "ops lead"),
                    MissionDeliverableBlueprint("Evidence Packet", "Policy and runtime evidence for auditability.", "governance"),
                ],
                benchmark_targets=[
                    BenchmarkTargetBlueprint("TAU-bench", "high", "Best fit for enterprise task flow and tool-mediated work.", "Needs live business connectors."),
                    BenchmarkTargetBlueprint("TheAgentCompany", "medium", "Useful for workplace productivity packaging.", "Needs persistent workplace memory."),
                    BenchmarkTargetBlueprint("GAIA", "low", "Only partial overlap via multi-hop reasoning.", "Less relevant than ops execution fidelity."),
                ],
                keyword_patterns=[r"(ops|operations|timeline|delivery|program|workflow|dependency|playbook|milestone)"],
            ),
            MissionProfile(
                name="implementation_pack",
                title="Implementation Mission Pack",
                summary="Engineering-facing package for architecture, migration, and validation planning.",
                primary_deliverable="Implementation spec with architecture target state, migration steps, and validation gates.",
                target_users=["tech lead", "staff engineer", "platform owner"],
                output_views=["architecture spec", "migration plan", "validation checklist", "risk register"],
                review_questions=[
                    "What execution trace proves the design is implementable?",
                    "Which integration or operability gap is still unowned?",
                    "What benchmark should validate the implementation class?",
                ],
                deliverables=[
                    MissionDeliverableBlueprint("Architecture Spec", "Target state and integration blueprint.", "tech lead"),
                    MissionDeliverableBlueprint("Migration Plan", "Phased delivery path with rollback boundary.", "platform team"),
                    MissionDeliverableBlueprint("Validation Checklist", "Tests, evals, and release gates for implementation.", "engineering manager"),
                    MissionDeliverableBlueprint("Benchmark Mapping", "Which benchmark family actually matters for this build.", "research and engineering"),
                ],
                benchmark_targets=[
                    BenchmarkTargetBlueprint("SWE-bench Verified", "medium", "Relevant once the system closes the code-fix loop.", "Current engine is not yet a code-repair benchmark runner."),
                    BenchmarkTargetBlueprint("WebArena", "low", "Relevant for integration flows with heavy browser action.", "Missing real browser execution layer."),
                    BenchmarkTargetBlueprint("GAIA", "medium", "Useful for architecture reasoning quality.", "Not sufficient for implementation proof."),
                ],
                keyword_patterns=[r"(architecture|design|system|integration|refactor|migration|implementation|build|code)"],
            ),
        ]
