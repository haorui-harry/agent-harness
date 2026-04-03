"""Built-in and extensible skills for the LangGraph Skill Router."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from app.core.state import (
    SkillBudget,
    SkillCategory,
    SkillDependency,
    SkillMetadata,
    SkillTier,
)
from app.ecosystem.marketplace import list_marketplace_skill_metadata


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
    return [part.strip(" -\t") for part in parts if part.strip(" -\t")]


def _keywords(text: str, limit: int = 6) -> list[str]:
    stop = {
        "this",
        "that",
        "with",
        "from",
        "into",
        "about",
        "there",
        "their",
        "would",
        "could",
        "should",
        "have",
        "has",
        "been",
        "were",
        "will",
        "your",
        "ours",
        "they",
        "them",
        "what",
        "when",
        "where",
        "which",
        "because",
        "while",
        "after",
        "before",
        "also",
        "more",
        "most",
        "only",
        "than",
        "then",
        "into",
    }
    tokens = [word.lower() for word in _WORD_RE.findall(str(text or "")) if word.lower() not in stop]
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(limit)]


def _candidate_lines(text: str) -> list[str]:
    lines = [line.strip(" -\t") for line in str(text or "").splitlines() if line.strip(" -\t")]
    return lines or _sentences(text)


def _match_lines(text: str, keywords: list[str], limit: int = 4) -> list[str]:
    rows: list[tuple[int, str]] = []
    for line in _candidate_lines(text):
        lowered = line.lower()
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score:
            rows.append((score, line))
    rows.sort(key=lambda item: (-item[0], len(item[1])))
    out: list[str] = []
    for _, line in rows:
        if line not in out:
            out.append(line)
        if len(out) >= limit:
            break
    return out


def _bullet_block(title: str, rows: list[str]) -> str:
    payload = rows or ["No grounded items extracted from input."]
    return title + "\n" + "\n".join(f"- {row}" for row in payload)


def identify_risks(text: str) -> str:
    """Extract and list key risks from the given text."""

    risks = _match_lines(
        text,
        [
            "risk",
            "issue",
            "blocker",
            "delay",
            "breach",
            "failure",
            "uncertain",
            "audit",
            "compliance",
            "dependency",
            "cost",
        ],
        limit=5,
    )
    if not risks:
        risks = [f"Potential exposure around {keyword}" for keyword in _keywords(text, limit=4)]
    return _bullet_block("Key Risks Identified:", risks) + "\n--- (skill: identify_risks)"


def executive_summary(text: str) -> str:
    """Produce a concise executive summary."""

    sentences = _sentences(text)
    context = sentences[0] if sentences else _normalize_text(text)[:180]
    findings = _match_lines(text, ["must", "need", "result", "impact", "evidence", "decision"], limit=2)
    action = findings[0] if findings else f"Focus next on {', '.join(_keywords(text, limit=3)) or 'the main task'}."
    return (
        "Executive Summary:\n"
        f"- Context: {context[:220]}\n"
        f"- Core finding: {(findings[1] if len(findings) > 1 else action)[:220]}\n"
        f"- Decision implication: {action[:220]}\n"
        "--- (skill: executive_summary)"
    )


def compare_options(text: str) -> str:
    """Compare options or alternatives mentioned in the text."""

    lines = _candidate_lines(text)
    options = [line for line in lines if any(token in line.lower() for token in ["option", "plan", "approach", "strategy"])]
    if len(options) < 2:
        options = lines[:2] if len(lines) >= 2 else [text, "Fallback alternative based on extracted keywords"]
    rows = options[:3]
    rendered = ["Option Comparison:", "| Option | Strengths | Risks |", "|--------|-----------|-------|"]
    for index, row in enumerate(rows, start=1):
        keywords = _keywords(row, limit=3)
        strength = ", ".join(keywords[:2]) or "coverage"
        risk = keywords[-1] if keywords else "ambiguity"
        rendered.append(f"| {index} | {strength} | {risk} |")
    rendered.append(f"Recommendation Basis: prioritize {', '.join(_keywords(text, limit=3)) or 'the clearest option'}")
    rendered.append("--- (skill: compare_options)")
    return "\n".join(rendered)


def extract_facts(text: str) -> str:
    """Extract factual statements from the text."""

    facts = [item for item in _sentences(text) if any(token.isdigit() for token in item) or len(item.split()) >= 6][:5]
    if not facts:
        facts = _candidate_lines(text)[:4]
    quality = "high" if len(facts) >= 3 else "medium"
    return _bullet_block("Extracted Facts:", facts) + f"\nEvidence quality: {quality}\n--- (skill: extract_facts)"


def generate_recommendations(text: str) -> str:
    """Generate actionable recommendations based on analysis."""

    priorities = _keywords(text, limit=3)
    recommendations = [
        f"1. Stabilize {priorities[0] if priorities else 'the core workflow'} with a named owner and deadline.",
        f"2. Validate {priorities[1] if len(priorities) > 1 else 'the highest-risk assumption'} using direct evidence or tests.",
        f"3. Package {priorities[2] if len(priorities) > 2 else 'the deliverable'} into an artifact others can review.",
    ]
    return "Recommendations:\n" + "\n".join(recommendations) + "\n--- (skill: generate_recommendations)"


def brainstorm_ideas(text: str) -> str:
    """Generate creative ideas and angles related to the topic."""

    seeds = _keywords(text, limit=4)
    ideas = [
        f"Reframe {seeds[0] if seeds else 'the task'} as a reusable product surface instead of a one-off output.",
        f"Turn {seeds[1] if len(seeds) > 1 else 'the workflow'} into a measurable experiment with clear pass or fail criteria.",
        f"Expose {seeds[2] if len(seeds) > 2 else 'the strongest capability'} through a smaller public interface or skill.",
        f"Run a high-upside variant optimized around {seeds[3] if len(seeds) > 3 else 'speed and quality together'}.",
    ]
    return _bullet_block("Brainstormed Ideas:", ideas) + "\n--- (skill: brainstorm_ideas)"


def detect_anomalies(text: str) -> str:
    """Detect anomalies, contradictions, and inconsistency patterns."""

    anomalies = _match_lines(text, ["but", "however", "despite", "inconsistent", "unexpected", "error", "fail"], limit=4)
    if not anomalies:
        anomalies = [f"No direct contradiction found; inspect {keyword} for drift." for keyword in _keywords(text, limit=3)]
    return _bullet_block("Anomaly Detection Report:", anomalies) + "\n--- (skill: detect_anomalies)"


def build_timeline(text: str) -> str:
    """Extract events and build a timeline."""

    items = _match_lines(text, ["first", "then", "next", "after", "before", "phase", "day", "week", "month"], limit=5)
    if not items:
        items = _candidate_lines(text)[:3]
    lines = ["Timeline:"] + [f"  [T{idx}] {item}" for idx, item in enumerate(items, start=1)]
    if len(items) >= 2:
        lines.append(f"  Dependencies: T2 depends on T1{' , T3 depends on T2' if len(items) >= 3 else ''}")
    lines.append("--- (skill: build_timeline)")
    return "\n".join(lines)


def synthesize_perspectives(text: str) -> str:
    """Integrate viewpoints to find consensus and divergence."""

    lines = _candidate_lines(text)
    left = lines[0] if lines else "No clear first viewpoint extracted."
    right = lines[1] if len(lines) > 1 else "No contrasting viewpoint extracted."
    common = ", ".join(_keywords(text, limit=3)) or "shared objective"
    return (
        "Perspective Synthesis:\n"
        f"  Viewpoint A: {left[:180]}\n"
        f"  Viewpoint B: {right[:180]}\n"
        f"  Common Ground: {common}\n"
        f"  Key Disagreements: {('different emphasis on ' + common)[:180]}\n"
        f"  Synthesis: combine the strongest claims while validating the highest-risk disagreement.\n"
        "--- (skill: synthesize_perspectives)"
    )


def validate_claims(text: str) -> str:
    """Validate claims and estimate evidence confidence."""

    claims = _candidate_lines(text)[:2]
    sections = ["Claim Validation:"]
    for index, claim in enumerate(claims or ["No explicit claim extracted."], start=1):
        confidence = "HIGH" if any(char.isdigit() for char in claim) else "MEDIUM"
        verdict = "SUPPORTED" if len(claim.split()) >= 6 else "INSUFFICIENT_EVIDENCE"
        evidence = claim if verdict == "SUPPORTED" else "Need stronger external or empirical support."
        sections.append(f"  Claim {index}: {claim[:180]}")
        sections.append(f"    Evidence: {evidence[:180]}")
        sections.append(f"    Confidence: {confidence}")
        sections.append(f"    Verdict: {verdict}")
    sections.append("--- (skill: validate_claims)")
    return "\n".join(sections)


def generate_analogies(text: str) -> str:
    """Generate analogies for complex concepts."""

    focus = ", ".join(_keywords(text, limit=3)) or "the system"
    return (
        "Analogies:\n"
        f"  Concept: {focus}\n"
        f"  Analogy 1: Like an airport control tower coordinating many runways without collisions.\n"
        f"  Analogy 2: Like a research lab notebook that also knows how to run the next experiment.\n"
        f"  Why it works: both analogies emphasize coordination, memory, and verifiable execution.\n"
        "--- (skill: generate_analogies)"
    )


def prioritize_items(text: str) -> str:
    """Rank and prioritize items by urgency and impact."""

    items = _candidate_lines(text)[:4]
    while len(items) < 4:
        items.append(f"Follow-up on {(_keywords(text, limit=4) + ['execution'])[len(items)]}")
    return (
        "Priority Ranking:\n"
        f"  [P0 - Critical] {items[0][:90]} - Impact: HIGH, Urgency: HIGH\n"
        f"  [P1 - High]     {items[1][:90]} - Impact: HIGH, Urgency: MEDIUM\n"
        f"  [P2 - Medium]   {items[2][:90]} - Impact: MEDIUM, Urgency: MEDIUM\n"
        f"  [P3 - Low]      {items[3][:90]} - Impact: LOW, Urgency: LOW\n"
        f"  Rationale: prioritize the items most tied to {', '.join(_keywords(text, limit=3)) or 'execution risk'}.\n"
        "--- (skill: prioritize_items)"
    )


def decompose_task(text: str) -> str:
    """Break a task into an executable checklist with dependencies."""

    keywords = _keywords(text, limit=4)
    tasks = [
        f"[1] Define scope around {keywords[0] if keywords else 'the request'}",
        f"[2] Gather evidence and workspace context for {keywords[1] if len(keywords) > 1 else 'the key objects'}",
        f"[3] Execute the highest-value action touching {keywords[2] if len(keywords) > 2 else 'the core path'}",
        f"[4] Validate outputs and package review artifacts for {keywords[3] if len(keywords) > 3 else 'handoff'}",
    ]
    return "Executable Task Breakdown:\n" + "\n".join(f"- {item}" for item in tasks) + "\n--- (skill: decompose_task)"


def artifact_synthesis(text: str) -> str:
    """Turn mixed notes, logs, and artifact snippets into a grounded synthesis."""

    facts = _candidate_lines(text)[:4]
    focus = ", ".join(_keywords(text, limit=3)) or "the artifact set"
    return (
        "Artifact Synthesis:\n"
        f"- Observed signals: {' | '.join(item[:90] for item in facts) if facts else 'No strong signals extracted.'}\n"
        f"- Stable interpretation: the artifacts converge on {focus}.\n"
        f"- Remaining gap: validate the weakest unsupported assumption before finalizing.\n"
        "--- (skill: artifact_synthesis)"
    )


def validation_planner(text: str) -> str:
    """Produce a validation plan tied to concrete outputs and failure checks."""

    focus = _keywords(text, limit=3)
    checks = [
        f"Check 1: verify output completeness for {focus[0] if focus else 'the main deliverable'}",
        f"Check 2: stress the highest-risk path around {focus[1] if len(focus) > 1 else 'runtime behavior'}",
        f"Check 3: retain an artifact proving {focus[2] if len(focus) > 2 else 'the validation result'}",
    ]
    return "Validation Plan:\n" + "\n".join(f"- {item}" for item in checks) + "\n--- (skill: validation_planner)"


def codebase_triage(text: str) -> str:
    """Summarize likely code hotspots, missing tests, and execution priorities."""

    focus = _keywords(text, limit=4)
    items = [
        f"Hotspot: inspect {focus[0] if focus else 'the primary module'} first.",
        f"Patch target: isolate the defect around {focus[1] if len(focus) > 1 else 'input handling'}.",
        f"Test gap: add regression coverage for {focus[2] if len(focus) > 2 else 'the failing edge case'}.",
        f"Execution note: preserve an artifact for {focus[3] if len(focus) > 3 else 'validation output'}.",
    ]
    return "Codebase Triage:\n" + "\n".join(f"- {item}" for item in items) + "\n--- (skill: codebase_triage)"


def research_brief(text: str) -> str:
    """Turn a topic into a research brief with hypotheses, evidence, and gaps."""

    focus = _keywords(text, limit=3)
    return (
        "Research Brief:\n"
        f"- Question: how should we improve {focus[0] if focus else 'the target system'}?\n"
        f"- Hypothesis: better structure around {focus[1] if len(focus) > 1 else 'task decomposition'} improves results.\n"
        f"- Evidence to gather: benchmarks, failure cases, and direct artifacts touching {focus[2] if len(focus) > 2 else 'runtime behavior'}.\n"
        "- Open gap: separate anecdotal wins from reproducible gains.\n"
        "--- (skill: research_brief)"
    )


def ops_runbook(text: str) -> str:
    """Convert a task into a runbook with triggers, actions, and escalation points."""

    focus = _keywords(text, limit=3)
    return (
        "Ops Runbook:\n"
        f"- Trigger: when {focus[0] if focus else 'the workflow'} deviates from expected state.\n"
        f"- Action: stabilize {focus[1] if len(focus) > 1 else 'the critical path'} and capture evidence.\n"
        f"- Escalation: involve review if {focus[2] if len(focus) > 2 else 'risk or ambiguity'} remains unresolved.\n"
        "- Artifact: leave a handoff note plus validation record.\n"
        "--- (skill: ops_runbook)"
    )


def benchmark_ablation(text: str) -> str:
    """Produce a compact benchmark and ablation design."""

    focus = _keywords(text, limit=3)
    return (
        "Benchmark Ablation Plan:\n"
        f"- Baseline: current pipeline centered on {focus[0] if focus else 'the default configuration'}.\n"
        f"- Ablation A: remove or weaken {focus[1] if len(focus) > 1 else 'dynamic discovery'}.\n"
        f"- Ablation B: tighten validation around {focus[2] if len(focus) > 2 else 'artifact quality'}.\n"
        "- Report: compare pass rate, value, latency, and failure clusters.\n"
        "--- (skill: benchmark_ablation)"
    )


def frontend_critique(text: str) -> str:
    """Generate a sharp product or interface critique with redesign priorities."""

    focus = _keywords(text, limit=3)
    return (
        "Frontend Critique:\n"
        f"- Strength: the surface already communicates {focus[0] if focus else 'the main message'}.\n"
        f"- Weakness: hierarchy around {focus[1] if len(focus) > 1 else 'primary actions'} is not decisive enough.\n"
        f"- Redesign priority: make {focus[2] if len(focus) > 2 else 'the first-screen story'} obvious within one glance.\n"
        "- Finish: package the critique with before or after visual guidance.\n"
        "--- (skill: frontend_critique)"
    )


SKILL_REGISTRY: dict[str, dict[str, Any]] = {
    "identify_risks": {
        "fn": identify_risks,
        "metadata": SkillMetadata(
            name="identify_risks",
            description="Extract and categorize risks from text",
            strengths=["risk detection", "threat analysis", "negative outcome patterns"],
            weaknesses=["may over-flag benign items", "less useful for purely positive content"],
            category=SkillCategory.EXTRACTION,
            output_type="list",
            confidence_keywords=["risk", "threat", "danger", "concern", "issue", "problem"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.2,
            synergies=["prioritize_items", "generate_recommendations", "detect_anomalies"],
            conflicts=[],
        ),
    },
    "executive_summary": {
        "fn": executive_summary,
        "metadata": SkillMetadata(
            name="executive_summary",
            description="Produce a concise executive summary of the content",
            strengths=["brevity", "clarity", "distilling key points"],
            weaknesses=["loses nuance", "not suitable for deep analysis"],
            category=SkillCategory.COMMUNICATION,
            output_type="text",
            confidence_keywords=["summarize", "summary", "overview", "brief", "tldr"],
            tier=SkillTier.BASIC,
            compute_cost=0.8,
            synergies=["extract_facts", "identify_risks"],
            conflicts=[],
        ),
    },
    "compare_options": {
        "fn": compare_options,
        "metadata": SkillMetadata(
            name="compare_options",
            description="Compare alternatives or options mentioned in the content",
            strengths=["structured comparison", "trade-off analysis"],
            weaknesses=["needs clearly defined options", "less useful for single-topic content"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=["compare", "versus", "vs", "alternative", "option", "trade-off"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.2,
            synergies=["generate_recommendations", "extract_facts"],
            conflicts=[],
        ),
    },
    "extract_facts": {
        "fn": extract_facts,
        "metadata": SkillMetadata(
            name="extract_facts",
            description="Pull out factual statements and data points",
            strengths=["precision", "fact isolation", "data extraction"],
            weaknesses=["ignores opinions and context", "can be overly literal"],
            category=SkillCategory.RECALL,
            output_type="list",
            confidence_keywords=["fact", "data", "number", "statistic", "evidence"],
            tier=SkillTier.BASIC,
            compute_cost=0.9,
            synergies=["validate_claims", "build_timeline", "synthesize_perspectives"],
            conflicts=["generate_analogies"],
        ),
    },
    "generate_recommendations": {
        "fn": generate_recommendations,
        "metadata": SkillMetadata(
            name="generate_recommendations",
            description="Generate actionable recommendations",
            strengths=["actionability", "decision support", "prioritization"],
            weaknesses=["may assume context not present", "can be generic"],
            category=SkillCategory.GENERATION,
            output_type="list",
            confidence_keywords=["recommend", "suggest", "advise", "action", "should", "next step"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.1,
            synergies=["identify_risks", "prioritize_items", "compare_options"],
            conflicts=[],
        ),
    },
    "brainstorm_ideas": {
        "fn": brainstorm_ideas,
        "metadata": SkillMetadata(
            name="brainstorm_ideas",
            description="Generate creative ideas and unconventional angles",
            strengths=["creativity", "lateral thinking", "diverse perspectives"],
            weaknesses=["may lack focus", "not all ideas are practical"],
            category=SkillCategory.GENERATION,
            output_type="list",
            confidence_keywords=["idea", "creative", "brainstorm", "innovate", "explore", "what if"],
            tier=SkillTier.BASIC,
            compute_cost=0.9,
            synergies=["generate_analogies", "synthesize_perspectives"],
            conflicts=["validate_claims", "detect_anomalies"],
        ),
    },
    "detect_anomalies": {
        "fn": detect_anomalies,
        "metadata": SkillMetadata(
            name="detect_anomalies",
            description="Detect anomalies, contradictions, and inconsistencies in text",
            strengths=[
                "contradiction detection",
                "pattern deviation",
                "outlier identification",
                "consistency checking",
            ],
            weaknesses=["may flag intentional contrasts", "less useful for subjective content"],
            category=SkillCategory.ANALYSIS,
            output_type="list",
            confidence_keywords=[
                "anomaly",
                "inconsistent",
                "contradiction",
                "unusual",
                "outlier",
                "deviation",
                "bug",
                "error",
            ],
            tier=SkillTier.ADVANCED,
            compute_cost=1.5,
            synergies=["identify_risks", "validate_claims"],
            conflicts=["brainstorm_ideas"],
        ),
    },
    "build_timeline": {
        "fn": build_timeline,
        "metadata": SkillMetadata(
            name="build_timeline",
            description="Extract events and build a chronological timeline with dependencies",
            strengths=["temporal ordering", "dependency mapping", "sequence reconstruction"],
            weaknesses=["requires temporal markers", "less useful for non-sequential content"],
            category=SkillCategory.EXTRACTION,
            output_type="timeline",
            confidence_keywords=["timeline", "sequence", "when", "before", "after", "history", "phase", "schedule"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.3,
            synergies=["extract_facts", "prioritize_items"],
            conflicts=[],
        ),
    },
    "synthesize_perspectives": {
        "fn": synthesize_perspectives,
        "metadata": SkillMetadata(
            name="synthesize_perspectives",
            description="Identify, compare, and synthesize multiple viewpoints into an integrated analysis",
            strengths=["multi-perspective integration", "consensus detection", "nuance preservation"],
            weaknesses=["may create false balance", "requires diverse sources"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=[
                "perspective",
                "viewpoint",
                "opinion",
                "stakeholder",
                "debate",
                "synthesis",
                "integrate",
            ],
            tier=SkillTier.EXPERT,
            compute_cost=2.0,
            synergies=["extract_facts", "validate_claims"],
            conflicts=[],
        ),
    },
    "validate_claims": {
        "fn": validate_claims,
        "metadata": SkillMetadata(
            name="validate_claims",
            description="Evaluate claims against available evidence and assign confidence levels",
            strengths=["evidence evaluation", "claim verification", "confidence assessment", "bias detection"],
            weaknesses=["needs factual content", "cannot verify against external sources"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=["claim", "verify", "evidence", "true", "false", "proof", "validate", "fact check"],
            tier=SkillTier.EXPERT,
            compute_cost=2.0,
            synergies=["extract_facts", "detect_anomalies"],
            conflicts=["brainstorm_ideas"],
        ),
    },
    "generate_analogies": {
        "fn": generate_analogies,
        "metadata": SkillMetadata(
            name="generate_analogies",
            description="Create illuminating analogies to explain complex concepts",
            strengths=["concept mapping", "simplification", "cross-domain transfer", "intuition building"],
            weaknesses=["analogies can mislead if taken literally", "not suitable for strict specs"],
            category=SkillCategory.GENERATION,
            output_type="text",
            confidence_keywords=["analogy", "like", "similar to", "metaphor", "explain", "simplify", "intuition"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.2,
            synergies=["brainstorm_ideas", "synthesize_perspectives"],
            conflicts=["extract_facts"],
        ),
    },
    "prioritize_items": {
        "fn": prioritize_items,
        "metadata": SkillMetadata(
            name="prioritize_items",
            description="Rank and prioritize items by impact, urgency, and dependencies",
            strengths=["impact assessment", "urgency classification", "dependency-aware ordering"],
            weaknesses=["subjective without clear criteria", "hard on equal-priority cases"],
            category=SkillCategory.ANALYSIS,
            output_type="list",
            confidence_keywords=["priority", "rank", "important", "urgent", "first", "order", "triage", "critical"],
            tier=SkillTier.BASIC,
            compute_cost=0.8,
            synergies=["identify_risks", "generate_recommendations"],
            conflicts=[],
        ),
    },
    "decompose_task": {
        "fn": decompose_task,
        "metadata": SkillMetadata(
            name="decompose_task",
            description="Break a request into executable tasks, dependencies, and validation steps",
            strengths=["task decomposition", "dependency ordering", "execution framing"],
            weaknesses=["heuristic on underspecified tasks", "does not execute actions itself"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=["decompose", "break down", "task graph", "checklist", "steps", "workflow"],
            tier=SkillTier.EXPERT,
            compute_cost=1.3,
            synergies=["prioritize_items", "generate_recommendations", "validation_planner"],
            conflicts=[],
        ),
    },
    "artifact_synthesis": {
        "fn": artifact_synthesis,
        "metadata": SkillMetadata(
            name="artifact_synthesis",
            description="Synthesize logs, notes, and outputs into a grounded artifact summary",
            strengths=["artifact grounding", "cross-source compression", "handoff clarity"],
            weaknesses=["depends on input quality", "not a substitute for raw artifacts"],
            category=SkillCategory.COMMUNICATION,
            output_type="text",
            confidence_keywords=["artifact", "log", "trace", "summary", "synthesize", "handoff"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.1,
            synergies=["extract_facts", "executive_summary", "validate_claims"],
            conflicts=[],
        ),
    },
    "validation_planner": {
        "fn": validation_planner,
        "metadata": SkillMetadata(
            name="validation_planner",
            description="Translate a task or artifact into concrete validation checks and evidence hooks",
            strengths=["verification design", "failure anticipation", "artifact-oriented validation"],
            weaknesses=["does not run tests itself", "may miss domain-specific edge cases"],
            category=SkillCategory.ANALYSIS,
            output_type="list",
            confidence_keywords=["validate", "test", "check", "verify", "acceptance", "quality gate"],
            tier=SkillTier.EXPERT,
            compute_cost=1.2,
            synergies=["decompose_task", "identify_risks", "extract_facts"],
            conflicts=[],
        ),
    },
    "codebase_triage": {
        "fn": codebase_triage,
        "metadata": SkillMetadata(
            name="codebase_triage",
            description="Identify code hotspots, patch targets, tests, and validation artifacts",
            strengths=["engineering triage", "code-task framing", "test-gap detection"],
            weaknesses=["works from text summaries unless paired with workspace tools", "does not patch code itself"],
            category=SkillCategory.ANALYSIS,
            output_type="structured",
            confidence_keywords=["codebase", "patch", "bug", "test", "regression", "module"],
            tier=SkillTier.EXPERT,
            compute_cost=1.4,
            synergies=["decompose_task", "validation_planner", "extract_facts"],
            conflicts=[],
        ),
    },
    "research_brief": {
        "fn": research_brief,
        "metadata": SkillMetadata(
            name="research_brief",
            description="Turn a topic into a research question, hypothesis, evidence plan, and open gap list",
            strengths=["research framing", "hypothesis design", "gap analysis"],
            weaknesses=["not a substitute for external evidence gathering", "still heuristic on narrow domains"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=["research", "study", "hypothesis", "paper", "question", "evidence plan"],
            tier=SkillTier.EXPERT,
            compute_cost=1.4,
            synergies=["extract_facts", "benchmark_ablation", "validate_claims"],
            conflicts=[],
        ),
    },
    "ops_runbook": {
        "fn": ops_runbook,
        "metadata": SkillMetadata(
            name="ops_runbook",
            description="Convert operational tasks into trigger-action-escalation runbooks",
            strengths=["ops structure", "incident response framing", "handoff clarity"],
            weaknesses=["does not integrate external ticketing by itself", "assumes the task is operationalizable"],
            category=SkillCategory.GENERATION,
            output_type="structured",
            confidence_keywords=["runbook", "ops", "incident", "workflow", "playbook", "escalation"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.1,
            synergies=["prioritize_items", "generate_recommendations", "validation_planner"],
            conflicts=[],
        ),
    },
    "benchmark_ablation": {
        "fn": benchmark_ablation,
        "metadata": SkillMetadata(
            name="benchmark_ablation",
            description="Produce a benchmark runner plan with ablations, metrics, and failure analysis hooks",
            strengths=["benchmark framing", "ablation design", "evaluation rigor"],
            weaknesses=["does not execute benchmarks", "depends on a real task suite to be conclusive"],
            category=SkillCategory.ANALYSIS,
            output_type="structured",
            confidence_keywords=["benchmark", "ablation", "metric", "runner", "evaluation", "failure cluster"],
            tier=SkillTier.EXPERT,
            compute_cost=1.4,
            synergies=["research_brief", "validation_planner", "compare_options"],
            conflicts=[],
        ),
    },
    "frontend_critique": {
        "fn": frontend_critique,
        "metadata": SkillMetadata(
            name="frontend_critique",
            description="Critique interface hierarchy, story clarity, and redesign priorities",
            strengths=["UI critique", "first-screen clarity", "product storytelling"],
            weaknesses=["not a replacement for visual execution", "works best with screenshots or design text"],
            category=SkillCategory.COMMUNICATION,
            output_type="text",
            confidence_keywords=["ui", "ux", "frontend", "screen", "layout", "visual hierarchy"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.1,
            synergies=["executive_summary", "brainstorm_ideas", "generate_recommendations"],
            conflicts=[],
        ),
    },
}

# Third-party skill adapters (runtime-registered).
EXTERNAL_SKILL_REGISTRY: dict[str, dict[str, Any]] = {}


def get_skill_metadata(name: str) -> SkillMetadata | None:
    entry = SKILL_REGISTRY.get(name)
    if entry:
        return entry["metadata"]
    ext = EXTERNAL_SKILL_REGISTRY.get(name)
    if ext:
        return ext["metadata"]
    return None


def list_all_skills() -> list[SkillMetadata]:
    built_in = [entry["metadata"] for entry in SKILL_REGISTRY.values()]
    market = list_marketplace_skill_metadata()
    external = [entry["metadata"] for entry in EXTERNAL_SKILL_REGISTRY.values()]

    combined: list[SkillMetadata] = []
    seen: set[str] = set()
    for meta in built_in + market + external:
        if meta.name not in seen:
            combined.append(meta)
            seen.add(meta.name)
    return combined


def list_builtin_skills() -> list[SkillMetadata]:
    return [entry["metadata"] for entry in SKILL_REGISTRY.values()]


def list_external_skills() -> list[SkillMetadata]:
    return [entry["metadata"] for entry in EXTERNAL_SKILL_REGISTRY.values()]


def register_external_skill(
    name: str,
    fn: Callable[[str], str],
    metadata: SkillMetadata,
    source: str = "third_party",
) -> None:
    """Register an external skill so it can be routed like built-ins."""

    if not metadata.skill_id:
        metadata.skill_id = name
    if not metadata.summary:
        metadata.summary = metadata.description
    EXTERNAL_SKILL_REGISTRY[name] = {
        "fn": fn,
        "metadata": metadata,
        "source": source,
    }


def load_external_skills_from_file(path: str) -> int:
    """Load external skill specs from a local JSON file.

    Expected schema:
    {
      "skills": [
        {
          "name": "third_party_skill",
          "description": "...",
          "category": "analysis",
          "output_type": "structured",
          "confidence_keywords": ["..."],
          "tier": "advanced",
          "compute_cost": 1.1,
          "synergies": ["identify_risks"],
          "conflicts": [],
          "template": "Result for {query}"
        }
      ]
    }
    """

    spec_path = Path(path)
    if not spec_path.exists():
        return 0

    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    skills = payload.get("skills", [])
    loaded = 0
    for raw in skills:
        name = str(raw.get("name", "")).strip()
        if not name:
            continue

        template = str(raw.get("template", "Third-party output for: {query}"))

        def _make_fn(tpl: str, skill_name: str) -> Callable[[str], str]:
            def _fn(query: str) -> str:
                return f"{tpl.format(query=query)}\n--- (skill: {skill_name})"

            return _fn

        metadata = SkillMetadata(
            name=name,
            description=str(raw.get("description", f"External skill {name}")),
            strengths=list(raw.get("strengths", ["external capability"])),
            weaknesses=list(raw.get("weaknesses", ["unknown reliability"])),
            category=SkillCategory(str(raw.get("category", "analysis"))),
            output_type=str(raw.get("output_type", "text")),
            skill_id=str(raw.get("skill_id", name)),
            owner=str(raw.get("owner", raw.get("source", "third_party"))),
            version=str(raw.get("version", "1.0.0")),
            summary=str(raw.get("summary", raw.get("description", f"External skill {name}"))),
            applicable_tasks=list(raw.get("applicable_tasks", [])),
            contraindications=list(raw.get("contraindications", [])),
            required_inputs=list(raw.get("required_inputs", [])),
            optional_inputs=list(raw.get("optional_inputs", [])),
            output_schema=dict(raw.get("output_schema", {})),
            evidence_style=str(raw.get("evidence_style", "heuristic")),
            reasoning_role=str(raw.get("reasoning_role", "general")),
            latency_profile=str(raw.get("latency_profile", "medium")),
            cost_profile=str(raw.get("cost_profile", "medium")),
            risk_profile=str(raw.get("risk_profile", "medium")),
            interpretability_score=float(raw.get("interpretability_score", 0.65)),
            calibration_score=float(raw.get("calibration_score", 0.55)),
            failure_modes=list(raw.get("failure_modes", [])),
            recovery_suggestions=list(raw.get("recovery_suggestions", [])),
            compatible_with=list(raw.get("compatible_with", [])),
            complements=list(raw.get("complements", [])),
            redundant_with=list(raw.get("redundant_with", [])),
            ideal_position_in_pipeline=str(raw.get("ideal_position_in_pipeline", "middle")),
            supports_parallelism=bool(raw.get("supports_parallelism", True)),
            success_rate_by_domain=dict(raw.get("success_rate_by_domain", {})),
            success_rate_by_agent=dict(raw.get("success_rate_by_agent", {})),
            pairwise_synergy_scores=dict(raw.get("pairwise_synergy_scores", {})),
            pairwise_redundancy_scores=dict(raw.get("pairwise_redundancy_scores", {})),
            drift_flag=bool(raw.get("drift_flag", False)),
            reputation_score=float(raw.get("reputation_score", 0.5)),
            confidence_keywords=list(raw.get("confidence_keywords", [])),
            tier=SkillTier(str(raw.get("tier", "basic"))),
            compute_cost=float(raw.get("compute_cost", 1.0)),
            synergies=list(raw.get("synergies", [])),
            conflicts=list(raw.get("conflicts", [])),
        )
        register_external_skill(
            name=name,
            fn=_make_fn(template, name),
            metadata=metadata,
            source=str(raw.get("source", "third_party")),
        )
        loaded += 1

    return loaded


def get_skill_dependencies(name: str) -> SkillDependency | None:
    """Get skill dependency and enhancement info."""

    meta = get_skill_metadata(name)
    if not meta:
        return None
    return SkillDependency(
        skill_name=name,
        depends_on=[],
        enhances=list(meta.synergies),
        conflicts_with=list(meta.conflicts),
    )


def get_skill_budget(name: str) -> SkillBudget:
    """Get abstract budget for a skill."""

    meta = get_skill_metadata(name)
    cost = meta.compute_cost if meta else 1.0
    return SkillBudget(
        compute_cost=cost,
        latency_estimate_ms=cost * 100.0,
        context_requirement=0.5,
    )


def validate_skill_metadata(metadata: SkillMetadata) -> list[str]:
    """Validate metadata quality for registry/lifecycle checks."""

    issues: list[str] = []
    if not metadata.name.strip():
        issues.append("missing_name")
    if not metadata.description.strip():
        issues.append("missing_description")
    if not metadata.strengths:
        issues.append("missing_strengths")
    if not metadata.confidence_keywords:
        issues.append("missing_confidence_keywords")
    if metadata.compute_cost <= 0:
        issues.append("invalid_compute_cost")
    if not (metadata.summary or metadata.description):
        issues.append("missing_summary")
    if metadata.interpretability_score < 0.3:
        issues.append("low_interpretability")
    return issues


def get_skill_card(name: str) -> dict[str, Any] | None:
    """Return rich SkillCard + lifecycle signals for one skill."""

    meta = get_skill_metadata(name)
    if not meta:
        return None

    from app.memory.learning import get_skill_reliability  # local import to avoid circular startup cost

    card = meta.to_skill_card()
    card["validation_issues"] = validate_skill_metadata(meta)
    card["runtime_reliability"] = round(get_skill_reliability(name), 4)
    card["lifecycle_stage"] = "active" if not card["validation_issues"] else "needs_review"
    return card


def get_skill_lifecycle_status(name: str) -> dict[str, Any]:
    """Return coarse lifecycle status used by demos/inspection."""

    card = get_skill_card(name)
    if not card:
        return {"skill": name, "status": "unknown"}
    stage = card.get("lifecycle_stage", "active")
    drift = bool(card.get("drift_flag", False))
    reliability = float(card.get("runtime_reliability", 0.5))
    if drift:
        stage = "drift_alert"
    elif reliability < 0.4:
        stage = "observation"
    return {
        "skill": name,
        "status": stage,
        "reliability": reliability,
        "validation_issues": card.get("validation_issues", []),
    }


def list_skills_by_category(category: SkillCategory) -> list[SkillMetadata]:
    """List skills by category."""

    return [meta for meta in list_all_skills() if meta.category == category]


def list_skills_by_tier(tier: SkillTier) -> list[SkillMetadata]:
    """List skills by tier."""

    return [meta for meta in list_all_skills() if meta.tier == tier]


def get_synergy_pairs() -> list[tuple[str, str, float]]:
    """Return (skill_a, skill_b, strength) tuples for known synergies."""

    pairs: list[tuple[str, str, float]] = []
    for entry in SKILL_REGISTRY.values():
        meta = entry["metadata"]
        for partner in meta.synergies:
            pairs.append((meta.name, partner, 0.8))
    for entry in EXTERNAL_SKILL_REGISTRY.values():
        meta = entry["metadata"]
        for partner in meta.synergies:
            pairs.append((meta.name, partner, 0.8))
    return pairs


def _fallback_marketplace_skill(name: str, text: str) -> str:
    return (
        f"Marketplace Skill `{name}` Output:\n"
        f"- Query interpreted: {text}\n"
        "- Evidence: synthetic local execution (replace with real tool backend)\n"
        "- Confidence: medium"
    )


def execute_skill(name: str, text: str) -> str:
    """Execute built-in, external, or marketplace-backed skills."""

    entry = SKILL_REGISTRY.get(name)
    if entry:
        return entry["fn"](text)

    ext = EXTERNAL_SKILL_REGISTRY.get(name)
    if ext:
        try:
            return ext["fn"](text)
        except Exception as exc:  # pragma: no cover - defensive
            return f"[ERROR] External skill `{name}` failed: {exc}"

    market = {skill.name for skill in list_marketplace_skill_metadata()}
    if name in market:
        return _fallback_marketplace_skill(name, text)

    return f"[ERROR] Unknown skill: {name}"
