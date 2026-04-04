"""Scenario-aware proposal defaults for flagship showcase generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProposalPillarBlueprint:
    title: str
    summary: str
    integration: str
    live_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "integration": self.integration,
            "live_key": self.live_key,
        }


@dataclass(frozen=True)
class ProposalPhaseBlueprint:
    phase: str
    actions: list[str]
    success_metrics: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "actions": list(self.actions),
            "success_metrics": list(self.success_metrics),
        }


@dataclass(frozen=True)
class ProposalScenario:
    name: str
    theme: str
    release_need: str
    audience_takeaway: str
    headline: str
    strategy_plan: list[str]
    business_summary: list[str]
    critical_risks: list[str]
    impact_labels: list[str]
    keyword_patterns: list[str] = field(default_factory=list)
    pillars: list[ProposalPillarBlueprint] = field(default_factory=list)
    phases: list[ProposalPhaseBlueprint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "theme": self.theme,
            "release_need": self.release_need,
            "audience_takeaway": self.audience_takeaway,
            "headline": self.headline,
            "strategy_plan": list(self.strategy_plan),
            "business_summary": list(self.business_summary),
            "critical_risks": list(self.critical_risks),
            "impact_labels": list(self.impact_labels),
            "keyword_patterns": list(self.keyword_patterns),
            "pillars": [item.to_dict() for item in self.pillars],
            "phases": [item.to_dict() for item in self.phases],
        }


class ProposalRegistry:
    """Infer a concrete business scenario and provide proposal defaults."""

    def __init__(self) -> None:
        self._scenarios = self._defaults()

    def infer(self, query: str) -> ProposalScenario:
        text = query.lower().strip()
        best = self._scenarios[-1]
        best_score = -1
        for scenario in self._scenarios:
            score = sum(1 for pattern in scenario.keyword_patterns if re.search(pattern, text))
            if score > best_score:
                best = scenario
                best_score = score
        return best

    def list_cards(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._scenarios]

    @staticmethod
    def _defaults() -> list[ProposalScenario]:
        return [
            ProposalScenario(
                name="regulated_copilot_launch",
                theme=(
                    "Launching a regulated AI customer-support copilot with revenue pressure, auditability, and "
                    "research credibility held in one operating model."
                ),
                release_need=(
                    "A fintech launch team needs a 90-day plan that can ship a customer-support copilot, "
                    "keep a human override path, satisfy model-risk governance, and generate proof strong enough "
                    "for procurement, compliance, and executive rollout decisions."
                ),
                audience_takeaway=(
                    "This package turns a risky AI feature launch into a staged operating program with "
                    "commercial goals, control checkpoints, and evidence for expansion."
                ),
                headline="90-Day Launch Plan for a Regulated AI Support Copilot",
                strategy_plan=[
                    "Concentrate the launch on one revenue-linked support workflow instead of a broad AI platform story.",
                    "Prove operator control, auditability, and model-risk containment before scaling volume.",
                    "Translate pilot evidence into a release gate that commercial, risk, and research leaders can all sign.",
                ],
                business_summary=[
                    "Start with a narrow customer-support workflow where response time and auditability matter enough to measure.",
                    "Use human override, evidence logging, and gated rollout as first-class launch features rather than afterthoughts.",
                    "Expand only after the pilot proves containment, operator adoption, and measurable service uplift.",
                ],
                critical_risks=[
                    "Customer-facing hallucinations escaping human review.",
                    "Override policy exists on paper but is too slow for live operations.",
                    "Pilot metrics show engagement but not enough compliance evidence for expansion.",
                ],
                impact_labels=[
                    "Pilot throughput lift",
                    "Audit readiness",
                    "Operator adoption",
                    "Expansion readiness",
                ],
                keyword_patterns=[
                    r"(regulated|compliance|audit|policy|governance)",
                    r"(support|service|customer)",
                    r"(copilot|assistant|agent)",
                    r"(fintech|bank|payments|insurance)",
                ],
                pillars=[
                    ProposalPillarBlueprint(
                        title="Commercial Wedge",
                        summary="Target one support workflow with visible revenue or retention pressure.",
                        integration="Connect launch scope to the frontline KPI dashboard and ticket-routing path.",
                        live_key="growth_pillar",
                    ),
                    ProposalPillarBlueprint(
                        title="Control Surface",
                        summary="Human override, audit logging, and policy escalation stay in the request path.",
                        integration="Bind policy checks to case routing, override capture, and review queues.",
                        live_key="governance_pillar",
                    ),
                    ProposalPillarBlueprint(
                        title="Proof Engine",
                        summary="Every release claim is backed by pilot evidence, replayable traces, and lab gates.",
                        integration="Feed pilot traces and release benchmarks into the same decision packet.",
                        live_key="research_pillar",
                    ),
                ],
                phases=[
                    ProposalPhaseBlueprint(
                        phase="Phase 1 - Scope And Control Setup",
                        actions=[
                            "Limit the first release to one support motion with stable documentation and bounded risk.",
                            "Define override ownership, escalation path, and audit log schema before model exposure grows.",
                            "Instrument baseline service metrics and reviewer workload before turning on automation.",
                        ],
                        success_metrics=[
                            "baseline metrics captured",
                            "override policy signed off",
                            "audit schema wired into launch logs",
                        ],
                    ),
                    ProposalPhaseBlueprint(
                        phase="Phase 2 - Pilot And Evidence Collection",
                        actions=[
                            "Run the copilot in shadow or assisted mode with sampled human review.",
                            "Track response quality, override frequency, and policy exceptions in one operating dashboard.",
                            "Build a release packet that procurement, risk, and product leaders can read without extra translation.",
                        ],
                        success_metrics=[
                            "quality threshold sustained",
                            "override load within staffing plan",
                            "evidence packet complete",
                        ],
                    ),
                    ProposalPhaseBlueprint(
                        phase="Phase 3 - Controlled Expansion",
                        actions=[
                            "Open the copilot to more queues only after gates pass on quality, safety, and operator adoption.",
                            "Separate fast rollback levers from growth levers so expansion does not compromise containment.",
                            "Turn pilot proof into the board narrative for regional or workflow expansion.",
                        ],
                        success_metrics=[
                            "gates passed for next queue",
                            "rollback tested",
                            "expansion case approved",
                        ],
                    ),
                ],
            ),
            ProposalScenario(
                name="research_ops_platform",
                theme=(
                    "Launching an applied research platform that must ship product value while preserving experimental rigor."
                ),
                release_need=(
                    "A research and product organization needs one operating plan that can move experiments into production "
                    "without losing reproducibility, auditability, or decision quality."
                ),
                audience_takeaway=(
                    "The proposal connects experiment design, evidence review, and product rollout into one repeatable system."
                ),
                headline="Applied Research Delivery Operating Plan",
                strategy_plan=[
                    "Align experiment throughput with release evidence rather than running research and product on separate tracks.",
                    "Use one scorecard for value, reproducibility, and safety before promotion to production.",
                    "Package experiment results into operating decisions that non-research stakeholders can use."
                ],
                business_summary=[
                    "Move from isolated experiments to a governed release pipeline.",
                    "Treat reproducibility and benchmark history as launch requirements.",
                    "Use staged release criteria so research wins can survive real production scrutiny.",
                ],
                critical_risks=[
                    "Research signals look promising but do not translate to production operating constraints.",
                    "Benchmarks are strong but decision-makers cannot inspect the supporting evidence quickly enough.",
                    "Experimental branches proliferate faster than governance can review them.",
                ],
                impact_labels=["Experiment throughput", "Evidence quality", "Release confidence", "Stakeholder trust"],
                keyword_patterns=[
                    r"(research|lab|experiment|benchmark|study|paper)",
                    r"(researcher|researchers|promotion criteria|release promotion|paper-grade)",
                    r"(improvement|upgrade|roadmap|system gaps|evidence standards|deep research)",
                ],
                pillars=[
                    ProposalPillarBlueprint("Research Throughput", "Increase experiment velocity without breaking comparability.", "Route candidate ideas into benchmark-ready release packets.", "growth_pillar"),
                    ProposalPillarBlueprint("Governed Promotion", "Promote only the candidates that pass quality and safety gates.", "Tie release promotion to the same benchmark trail.", "governance_pillar"),
                    ProposalPillarBlueprint("Decision Readout", "Convert experiment evidence into board-readable launch claims.", "Attach lab and runtime evidence to the operating memo.", "research_pillar"),
                ],
                phases=[
                    ProposalPhaseBlueprint("Phase 1 - Benchmark Foundation", ["Stabilize scenario suites and quality gates.", "Define success metrics before scaling experiments."], ["scenario coverage complete", "quality gates agreed"]),
                    ProposalPhaseBlueprint("Phase 2 - Candidate Promotion", ["Run contenders through repeatable evaluation.", "Package evidence for decision review."], ["promotion packet complete", "release committee review ready"]),
                    ProposalPhaseBlueprint("Phase 3 - Production Rollout", ["Ship the winning candidate with monitoring hooks.", "Track post-launch drift against lab expectations."], ["post-launch drift within threshold", "rollback plan validated"]),
                ],
            ),
            ProposalScenario(
                name="enterprise_ai_rollout",
                theme="Rolling out an enterprise AI operating layer with stronger governance than point-solution copilots.",
                release_need=(
                    "An enterprise team needs a rollout design that can prove operational value, control security risk, "
                    "and avoid becoming another disconnected AI pilot."
                ),
                audience_takeaway=(
                    "The system is framed as an operating layer with phased adoption, evidence gates, and ecosystem leverage."
                ),
                headline="Enterprise AI Operating Layer Rollout Plan",
                strategy_plan=[
                    "Choose one workflow wedge and one governance model before expanding platform scope.",
                    "Integrate discovery, execution, and release evidence instead of running them as isolated workstreams.",
                    "Use interoperability as an expansion lever, not an afterthought.",
                ],
                business_summary=[
                    "Build around a single high-friction workflow first.",
                    "Keep release gates visible to security and operators from day one.",
                    "Turn ecosystem compatibility into a deployment multiplier.",
                ],
                critical_risks=[
                    "Platform scope expands before any workflow proves value.",
                    "Local pilot success cannot be repeated in the next environment.",
                    "Governance review happens after integration choices are already expensive to reverse.",
                ],
                impact_labels=["Workflow time saved", "Deployment readiness", "Governance coverage", "Ecosystem leverage"],
                keyword_patterns=[
                    r"(enterprise|workflow|operations|rollout|deployment|platform)",
                    r"(operating layer|business ops|security|cio)",
                ],
                pillars=[
                    ProposalPillarBlueprint("Workflow Wedge", "Start with one workflow that benefits from orchestration and evidence capture.", "Wire the wedge directly into the rollout scorecard.", "growth_pillar"),
                    ProposalPillarBlueprint("Governance By Default", "Move security and policy checks into the operating loop.", "Attach controls to runtime routing and release decisions.", "governance_pillar"),
                    ProposalPillarBlueprint("Portable Capability", "Export skills and evidence so adoption is not trapped in one runtime.", "Use interoperability bundles to accelerate adjacent teams.", "research_pillar"),
                ],
                phases=[
                    ProposalPhaseBlueprint("Phase 1 - Wedge Selection", ["Choose the first workflow and define owner metrics.", "Instrument current-state baseline."], ["workflow owner assigned", "baseline captured"]),
                    ProposalPhaseBlueprint("Phase 2 - Controlled Deployment", ["Launch with guardrails and evidence collection.", "Verify deployment and security readiness."], ["deployment checklist passed", "evidence bundle generated"]),
                    ProposalPhaseBlueprint("Phase 3 - Expansion Playbook", ["Replicate the pattern to the next workflow.", "Use interop to accelerate ecosystem adoption."], ["second workflow approved", "interop bundle consumed"]),
                ],
            ),
            ProposalScenario(
                name="generic_launch",
                theme="Launching an AI product with growth, governance, and research credibility kept in balance.",
                release_need=(
                    "A launch team needs a concrete operating plan that can expand the product, control governance risk, "
                    "and prove the system is credible enough to release."
                ),
                audience_takeaway=(
                    "This is not just a prompt response. It is a release-ready strategy package with routing evidence, "
                    "evaluation results, and ecosystem-portable artifacts."
                ),
                headline="Flagship AI Platform Launch Plan",
                strategy_plan=[
                    "Synthesize competing business, governance, and research perspectives into one operating thesis.",
                    "Map major release risks before launch and make the downside visible.",
                    "Convert the result into a release recommendation backed by measurable gates.",
                ],
                business_summary=[
                    "Build one operating model that aligns growth, governance, and research instead of optimizing them separately.",
                    "Release in phased checkpoints so the team can prove value before expanding scope.",
                    "Use explicit release gates so launch quality is evidence-backed rather than intuition-backed.",
                ],
                critical_risks=[
                    "Launch scope expands faster than proof quality.",
                    "Governance review arrives after product commitments are already fixed.",
                    "The product story is stronger than the evidence packet behind it.",
                ],
                impact_labels=["Value creation", "Safety posture", "Release confidence", "Ecosystem leverage"],
                keyword_patterns=[],
                pillars=[
                    ProposalPillarBlueprint("Growth Engine", "Business capability bundle", "Integrated into the release stack", "growth_pillar"),
                    ProposalPillarBlueprint("Governance Core", "Business capability bundle", "Integrated into the release stack", "governance_pillar"),
                    ProposalPillarBlueprint("Research Credibility", "Business capability bundle", "Integrated into the release stack", "research_pillar"),
                ],
                phases=[
                    ProposalPhaseBlueprint("Phase 1 - Focus", ["Choose the first release wedge.", "Define operating and risk owners."], ["wedge defined", "owners assigned"]),
                    ProposalPhaseBlueprint("Phase 2 - Prove", ["Run a controlled launch.", "Collect measurable evidence."], ["evidence collected", "gate review ready"]),
                    ProposalPhaseBlueprint("Phase 3 - Expand", ["Scale only after gates pass.", "Package the expansion case."], ["expansion approved", "rollout tracked"]),
                ],
            ),
        ]
