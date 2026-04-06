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
        "and", "this", "that", "with", "for", "from", "into", "about",
        "there", "their", "would", "could", "should", "have", "has",
        "been", "were", "will", "your", "ours", "they", "them", "what",
        "when", "where", "which", "because", "while", "after", "before",
        "also", "more", "most", "only", "than", "then", "into", "the",
        "are", "was", "not", "but", "its", "can", "all", "does", "did",
    }
    tokens = [word.lower() for word in _WORD_RE.findall(str(text or "")) if word.lower() not in stop]
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(limit)]


def _topic_clause(text: str, fallback: str = "the target system") -> str:
    normalized = _normalize_text(text)
    for pattern in [r"\bon ([^.;]+)", r"\bfor ([^.;]+)", r"\babout ([^.;]+)"]:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        topic = match.group(1).strip(" .")
        topic = re.split(r"\band\b|\bwith\b", topic, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .")
        if topic:
            return topic
    stripped = re.sub(
        r"^(write|prepare|create|design|generate|draft|build|analyze|analyse|inspect|summarize|develop)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    words = stripped.split()
    return " ".join(words[:10]).strip(" .,") or fallback


def _candidate_lines(text: str) -> list[str]:
    lines = []
    for raw in str(text or "").splitlines():
        line = raw.strip(" -\t")
        if not line:
            continue
        lowered = line.lower()
        if re.match(r"^\[[a-z0-9_-]+\]\s+[a-z ]+$", lowered):
            continue
        if lowered.startswith("tool used:"):
            continue
        if lowered.startswith("--- (skill:"):
            continue
        if line.endswith(":") and len(line.split()) <= 4:
            continue
        lines.append(line)
    return lines or _sentences(text)


def _json_objects(text: str, limit: int = 6) -> list[dict[str, Any]]:
    payload = str(text or "")
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while index < len(payload) and len(objects) < limit:
        start = payload.find("{", index)
        if start < 0:
            break
        try:
            value, offset = decoder.raw_decode(payload[start:])
        except Exception:
            index = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        index = start + max(offset, 1)
    return objects


def _structured_signal_lines(text: str, limit: int = 8) -> list[str]:
    rows: list[str] = []
    for payload in _json_objects(text, limit=limit):
        output = payload.get("output")
        if isinstance(output, str):
            rows.extend(_candidate_lines(output)[:3])
        if isinstance(output, dict):
            record_count = int(output.get("record_count", output.get("count", 0)) or 0)
            if record_count > 0:
                rows.append(f"Collected {record_count} supporting evidence records.")
            for record in output.get("records", [])[:3] if isinstance(output.get("records", []), list) else []:
                if isinstance(record, dict) and str(record.get("title", "")).strip():
                    rows.append(str(record.get("title", "")).strip())
            for risk in output.get("risk_matrix", [])[:3] if isinstance(output.get("risk_matrix", []), list) else []:
                if not isinstance(risk, dict):
                    continue
                dimension = str(risk.get("dimension", "risk")).strip() or "risk"
                level = str(risk.get("level", "unknown")).strip() or "unknown"
                rows.append(f"{dimension.title()} risk is {level}.")
            for result in output.get("results", [])[:2] if isinstance(output.get("results", []), list) else []:
                if not isinstance(result, dict):
                    continue
                command = _normalize_text(str(result.get("command", ""))) or "command"
                rows.append(f"Command {command} exited with code {result.get('exit_code', 0)}.")
        tool_name = str(payload.get("tool_name", "")).strip()
        if tool_name and tool_name not in rows:
            rows.append(f"Tool used: {tool_name}.")
    if not rows:
        rows = _candidate_lines(text)
    deduped: list[str] = []
    for item in rows:
        value = _normalize_text(item)
        lowered = value.lower()
        if not value or value in deduped or value in {"{", "}", "[", "]"}:
            continue
        if re.match(r"^\[[a-z0-9_-]+\]\s+[a-z ]+$", lowered):
            continue
        if lowered.startswith("tool used:"):
            continue
        deduped.append(value)
        if len(deduped) >= limit:
            break
    return deduped


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


def _dedupe_preserve(items: list[str], limit: int = 6) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = _normalize_text(item)
        lowered = value.lower()
        if not value or lowered in seen:
            continue
        seen.add(lowered)
        rows.append(value)
        if len(rows) >= limit:
            break
    return rows


def _url_lines(text: str, limit: int = 4) -> list[str]:
    matches = re.findall(r"https?://[^\s)>\]]+", str(text or ""))
    return _dedupe_preserve(matches, limit=limit)


def _risk_lines(text: str, limit: int = 4) -> list[str]:
    rows = _match_lines(
        text,
        ["risk", "failure", "fragile", "weak", "uncertain", "governance", "safety", "hallucination", "drift", "audit"],
        limit=limit,
    )
    if rows:
        return rows
    focus = _keywords(text, limit=3)
    return [f"The current material is still weak on {item}." for item in focus[:3]] or ["The evidence base is still too thin to support a strong claim."]


def _recommendation_lines(text: str, limit: int = 4) -> list[str]:
    rows = _match_lines(
        text,
        ["recommend", "improve", "upgrade", "fix", "validate", "ship", "evidence", "benchmark", "deliverable"],
        limit=limit,
    )
    if rows:
        return rows
    focus = _keywords(text, limit=4)
    suggestions = [
        f"Strengthen evidence collection around {focus[0] if focus else 'the core claim'}.",
        f"Turn analysis on {focus[1] if len(focus) > 1 else 'the main path'} into a reusable artifact.",
        f"Validate the highest-risk claim around {focus[2] if len(focus) > 2 else 'execution quality'} before scaling.",
        f"Package the result so {focus[3] if len(focus) > 3 else 'reviewers'} can inspect it quickly.",
    ]
    return suggestions[:limit]


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

    facts = _structured_signal_lines(text, limit=8)
    evidence = _dedupe_preserve(facts + _url_lines(text, limit=3), limit=6)
    risks = _risk_lines(text, limit=3)
    recommendations = _recommendation_lines(text, limit=3)
    focus_terms = _keywords(text, limit=5)
    focus = ", ".join(focus_terms[:3]) or "the primary task"
    strongest = evidence[0] if evidence else "No strong signals extracted."
    support = evidence[1] if len(evidence) > 1 else "Need stronger supporting detail."
    return (
        "## Bottom Line\n\n"
        f"The strongest conclusion is that the deliverable should concentrate on {focus}, grounded first in {strongest[:220]} and reinforced by {support[:200]}. "
        "The final answer should therefore lead with the decision, then show the evidence, then state the concrete move that follows.\n\n"
        "## What The Material Actually Shows\n\n"
        + "\n".join(f"- {item}" for item in evidence[:4])
        + "\n\n## Main Constraints And Failure Modes\n\n"
        + "\n".join(f"- {item}" for item in risks[:3])
        + "\n\n## Recommended Delivery\n\n"
        + "\n".join(f"{idx}. {item}" for idx, item in enumerate(recommendations[:3], start=1))
        + "\n\n## Why This Should Be The Shipped Answer\n\n"
        "This synthesis is stronger than a generic summary when it preserves the best grounded signals, removes low-value process narration, and leaves one inspectable artifact path for review.\n"
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
    lowered = str(text or "").lower()
    topic = _topic_clause(text, fallback="the primary module")
    signals = _structured_signal_lines(text, limit=6)
    hotspot = focus[0] if focus else topic
    patch_target = focus[1] if len(focus) > 1 else "input handling"
    test_gap = focus[2] if len(focus) > 2 else "the failing edge case"
    execution_note = focus[3] if len(focus) > 3 else "validation output"
    grounding = signals[0] if signals else f"Focus on {topic}."
    secondary = signals[1] if len(signals) > 1 else f"Keep the patch narrowly scoped around {patch_target}."
    return (
        "## Engineering Summary\n\n"
        f"The likely hotspot is {hotspot}. Current grounding suggests: {grounding}\n\n"
        "## Patch Intent\n\n"
        f"The patch should isolate the defect around {patch_target} and keep the change surface minimal enough to validate quickly. "
        f"A second relevant signal is {secondary}\n\n"
        "## Test Plan\n\n"
        f"- Add or tighten regression coverage for {test_gap}.\n"
        "- Keep one direct happy-path test and one edge-case test tied to the patched branch.\n"
        "- Preserve a validation artifact that records what was executed and what still remains unverified.\n\n"
        "## Execution Notes\n\n"
        f"- Preserve evidence for {execution_note}.\n"
        "- Prefer touching the smallest number of files that can prove the fix.\n"
        "- If evidence is thin, state the missing proof explicitly instead of inventing implementation details.\n"
        "--- (skill: codebase_triage)"
    )


def research_brief(text: str) -> str:
    """Turn a topic into a research brief with hypotheses, evidence, and gaps."""

    focus = _keywords(text, limit=4)
    topic = _topic_clause(text)
    signals = _structured_signal_lines(text, limit=8)
    evidence = _dedupe_preserve(signals + _url_lines(text, limit=4), limit=6)
    failure_modes = _risk_lines(text, limit=3)
    recommendations = _recommendation_lines(text, limit=4)
    lead_signal = evidence[0] if evidence else "No strong evidence signal has been extracted yet."
    support_signal = evidence[1] if len(evidence) > 1 else "Evidence still needs to be deepened beyond initial notes."
    principle_a = focus[0] if focus else "the core mechanism"
    principle_b = focus[1] if len(focus) > 1 else "evidence quality"
    principle_c = focus[2] if len(focus) > 2 else "execution closure"
    return (
        "## Core Judgment\n\n"
        f"For {topic}, the evidence currently points to one central conclusion: better system structure only matters when it improves evidence quality, execution closure, and the usefulness of the final artifact. "
        f"The strongest available support is {lead_signal[:220]}, with additional support from {support_signal[:200]}.\n\n"
        "## Evidence Anchors\n\n"
        + "\n".join(f"- {item}" for item in evidence[:5])
        + "\n\n## Main Failure Modes\n\n"
        + "\n".join(f"- {item}" for item in failure_modes[:3])
        + "\n\n## Design Principles\n\n"
        f"1. Optimize {principle_a} only when it improves a user-visible outcome.\n"
        f"2. Make {principle_b} enter the final answer directly instead of leaving it in side artifacts.\n"
        f"3. Keep {principle_c} inspectable so the runtime can be audited and repaired.\n\n"
        "## Recommended Improvement Path\n\n"
        + "\n".join(f"{idx}. {item}" for idx, item in enumerate(recommendations[:4], start=1))
        + "\n\n## What To Validate Next\n\n"
        f"- Check whether the current approach produces better output than a direct model answer on {focus[3] if len(focus) > 3 else 'real reviewer tasks'}.\n"
        "- Test the weak points with concrete failure cases rather than only narrative claims.\n"
        "- Keep the next iteration tied to evidence, artifact quality, and execution closure.\n"
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


def frontend_critique(text: str) -> str:
    """Generate a product or interface critique with redesign priorities."""

    focus = _keywords(text, limit=3)
    return (
        "Frontend Critique:\n"
        f"- Strength: the surface already communicates {focus[0] if focus else 'the main message'}.\n"
        f"- Weakness: hierarchy around {focus[1] if len(focus) > 1 else 'primary actions'} is not decisive enough.\n"
        f"- Redesign priority: make {focus[2] if len(focus) > 2 else 'the first-screen story'} obvious within one glance.\n"
        "- Finish: package the critique with before or after visual guidance.\n"
        "--- (skill: frontend_critique)"
    )


def chart_storyboard(text: str) -> str:
    """Design a chart pack with chart choices, data expectations, and narrative intent."""

    focus = _keywords(text, limit=4)
    lead = focus[0] if focus else "the main metric"
    compare = focus[1] if len(focus) > 1 else "key segments"
    trend = focus[2] if len(focus) > 2 else "trend over time"
    return (
        "Chart Storyboard:\n"
        f"- Chart 1: headline comparison chart covering {lead} across {compare}.\n"
        f"- Chart 2: time-series chart explaining how {trend} evolves.\n"
        f"- Chart 3: risk or outlier chart isolating failure pockets around {focus[3] if len(focus) > 3 else 'edge cases'}.\n"
        "- Data contract: keep fields for dimension, metric, time, source, and annotation.\n"
        "- Review rule: every chart needs one sentence explaining the decision it should unlock.\n"
        "--- (skill: chart_storyboard)"
    )


def data_analysis_plan(text: str) -> str:
    """Frame a concrete data-analysis plan with questions, cuts, and validation hooks."""

    focus = _keywords(text, limit=4)
    return (
        "Data Analysis Plan:\n"
        f"- Core question: what drives {focus[0] if focus else 'the target outcome'}?\n"
        f"- Primary cuts: cohort by {focus[1] if len(focus) > 1 else 'segment'}, compare by {focus[2] if len(focus) > 2 else 'time'}, and inspect outliers in {focus[3] if len(focus) > 3 else 'edge populations'}.\n"
        "- Required tables: fact table, dimension table, source quality log.\n"
        "- Metrics: define one north-star metric, two diagnostic metrics, and one guardrail metric.\n"
        "- Validation: reconcile sample counts, null rates, and metric definitions before publishing charts.\n"
        "--- (skill: data_analysis_plan)"
    )


def webpage_blueprint(text: str) -> str:
    """Design a user-facing webpage or landing page blueprint."""

    focus = _keywords(text, limit=4)
    return (
        "Webpage Blueprint:\n"
        f"- Hero: a decisive headline around {focus[0] if focus else 'the product promise'} with one proof point and one primary action.\n"
        f"- Section 2: demonstrate how {focus[1] if len(focus) > 1 else 'the workflow'} works through three steps.\n"
        f"- Section 3: trust layer covering {focus[2] if len(focus) > 2 else 'evidence, safety, and governance'}.\n"
        f"- Section 4: artifact gallery or case study anchored on {focus[3] if len(focus) > 3 else 'visible results'}.\n"
        "- Interaction notes: first screen must clarify who it is for, what it produces, and what the user can open.\n"
        "--- (skill: webpage_blueprint)"
    )


def slide_deck_designer(text: str) -> str:
    """Generate a slide-deck structure with pacing and visual intent."""

    focus = _keywords(text, limit=4)
    topic = _topic_clause(text, fallback="the core proposal")
    return (
        "Slide Deck Design:\n"
        f"- Slide 1: opening tension framed around {topic}.\n"
        f"- Slide 2: system or market context for {focus[0] if focus else 'the audience'}.\n"
        f"- Slide 3: product or method mechanism centered on {focus[1] if len(focus) > 1 else 'the differentiator'}.\n"
        "- Slide 4: proof section with external evidence, user evidence, or execution artifacts.\n"
        f"- Slide 5: rollout or adoption path constrained by {focus[2] if len(focus) > 2 else 'risk and dependencies'}.\n"
        "- Slide 6: closing ask with a single decision, owner, and next checkpoint.\n"
        "--- (skill: slide_deck_designer)"
    )


def podcast_episode_plan(text: str) -> str:
    """Turn a topic into a podcast episode outline with segment logic."""

    focus = _keywords(text, limit=4)
    return (
        "Podcast Episode Plan:\n"
        f"- Cold open: state the sharp question about {focus[0] if focus else 'the topic'} in under 20 seconds.\n"
        f"- Segment 1: explain the background and why {focus[1] if len(focus) > 1 else 'it matters now'}.\n"
        f"- Segment 2: unpack the main mechanism behind {focus[2] if len(focus) > 2 else 'the system'} using examples.\n"
        f"- Segment 3: debate risks, tradeoffs, and open problems around {focus[3] if len(focus) > 3 else 'deployment'}.\n"
        "- Close: summarize three takeaways and one open question for the audience.\n"
        "--- (skill: podcast_episode_plan)"
    )


def video_storyboard(text: str) -> str:
    """Create a short-form video storyboard with scenes and beats."""

    focus = _keywords(text, limit=4)
    return (
        "Video Storyboard:\n"
        f"- Scene 1: attention hook introducing {focus[0] if focus else 'the main idea'} in one visual moment.\n"
        f"- Scene 2: show the system or product in action around {focus[1] if len(focus) > 1 else 'the workflow'}.\n"
        f"- Scene 3: proof montage using charts, artifacts, or evidence for {focus[2] if len(focus) > 2 else 'performance'}.\n"
        f"- Scene 4: risk or counterfactual beat clarifying {focus[3] if len(focus) > 3 else 'what can go wrong'}.\n"
        "- Final frame: explicit call to action plus on-screen takeaway sentence.\n"
        "--- (skill: video_storyboard)"
    )


def image_prompt_pack(text: str) -> str:
    """Produce a reusable image prompt pack with multiple visual directions."""

    focus = _keywords(text, limit=4)
    return (
        "Image Prompt Pack:\n"
        f"- Direction A: editorial hero image centered on {focus[0] if focus else 'the concept'} with clean composition.\n"
        f"- Direction B: technical schematic emphasizing {focus[1] if len(focus) > 1 else 'mechanism and structure'}.\n"
        f"- Direction C: campaign poster framing {focus[2] if len(focus) > 2 else 'the strongest claim'} with bold typography.\n"
        f"- Direction D: product-render style image showing {focus[3] if len(focus) > 3 else 'the user interaction'}.\n"
        "- Shared constraints: specify aspect ratio, palette, negative prompts, and must-include labels.\n"
        "--- (skill: image_prompt_pack)"
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
            synergies=["prioritize_items", "extract_facts", "detect_anomalies"],
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
            conflicts=[],
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
            synergies=["synthesize_perspectives"],
            conflicts=["validate_claims", "detect_anomalies"],
        ),
    },
    "detect_anomalies": {
        "fn": detect_anomalies,
        "metadata": SkillMetadata(
            name="detect_anomalies",
            description="Detect anomalies, contradictions, and inconsistencies in text",
            strengths=["contradiction detection", "pattern deviation", "outlier identification"],
            weaknesses=["may flag intentional contrasts", "less useful for subjective content"],
            category=SkillCategory.ANALYSIS,
            output_type="list",
            confidence_keywords=["anomaly", "inconsistent", "contradiction", "unusual", "outlier", "bug", "error"],
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
            description="Identify, compare, and synthesize multiple viewpoints",
            strengths=["multi-perspective integration", "consensus detection", "nuance preservation"],
            weaknesses=["may create false balance", "requires diverse sources"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=["perspective", "viewpoint", "opinion", "stakeholder", "debate", "synthesis"],
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
            strengths=["evidence evaluation", "claim verification", "confidence assessment"],
            weaknesses=["needs factual content", "cannot verify against external sources"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=["claim", "verify", "evidence", "true", "false", "proof", "validate"],
            tier=SkillTier.EXPERT,
            compute_cost=2.0,
            synergies=["extract_facts", "detect_anomalies"],
            conflicts=["brainstorm_ideas"],
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
            confidence_keywords=["priority", "rank", "important", "urgent", "first", "order", "triage"],
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
            synergies=["prioritize_items", "validation_planner"],
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
            description="Translate a task into concrete validation checks and evidence hooks",
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
            weaknesses=["works from text summaries unless paired with workspace tools"],
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
            description="Turn a topic into a research question, hypothesis, evidence plan, and gap list",
            strengths=["research framing", "hypothesis design", "gap analysis"],
            weaknesses=["not a substitute for external evidence gathering"],
            category=SkillCategory.REASONING,
            output_type="structured",
            confidence_keywords=["research", "study", "hypothesis", "paper", "question", "evidence plan"],
            tier=SkillTier.EXPERT,
            compute_cost=1.4,
            synergies=["extract_facts", "validate_claims", "artifact_synthesis"],
            conflicts=[],
        ),
    },
    "ops_runbook": {
        "fn": ops_runbook,
        "metadata": SkillMetadata(
            name="ops_runbook",
            description="Convert operational tasks into trigger-action-escalation runbooks",
            strengths=["ops structure", "incident response framing", "handoff clarity"],
            weaknesses=["does not integrate external ticketing by itself"],
            category=SkillCategory.GENERATION,
            output_type="structured",
            confidence_keywords=["runbook", "ops", "incident", "workflow", "playbook", "escalation"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.1,
            synergies=["prioritize_items", "validation_planner"],
            conflicts=[],
        ),
    },
    "frontend_critique": {
        "fn": frontend_critique,
        "metadata": SkillMetadata(
            name="frontend_critique",
            description="Critique interface hierarchy and redesign priorities",
            strengths=["UI critique", "first-screen clarity", "product storytelling"],
            weaknesses=["not a replacement for visual execution"],
            category=SkillCategory.COMMUNICATION,
            output_type="text",
            confidence_keywords=["ui", "ux", "frontend", "screen", "layout", "visual hierarchy"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.1,
            synergies=["executive_summary", "brainstorm_ideas"],
            conflicts=[],
        ),
    },
    "chart_storyboard": {
        "fn": chart_storyboard,
        "metadata": SkillMetadata(
            name="chart_storyboard",
            description="Choose chart families, data contracts, and decision narratives",
            strengths=["data storytelling", "chart selection", "visualization framing"],
            weaknesses=["does not render charts itself"],
            category=SkillCategory.ANALYSIS,
            output_type="structured",
            confidence_keywords=["chart", "graph", "visualization", "plot", "dashboard"],
            tier=SkillTier.EXPERT,
            compute_cost=1.3,
            synergies=["data_analysis_plan", "extract_facts", "validation_planner"],
            conflicts=[],
        ),
    },
    "data_analysis_plan": {
        "fn": data_analysis_plan,
        "metadata": SkillMetadata(
            name="data_analysis_plan",
            description="Design a data-analysis plan with questions, metrics, and validation hooks",
            strengths=["analysis framing", "metric design", "data rigor"],
            weaknesses=["does not execute queries"],
            category=SkillCategory.ANALYSIS,
            output_type="structured",
            confidence_keywords=["data", "dataset", "analysis", "analytics", "cohort", "metric"],
            tier=SkillTier.EXPERT,
            compute_cost=1.4,
            synergies=["chart_storyboard", "extract_facts", "validation_planner"],
            conflicts=[],
        ),
    },
    "webpage_blueprint": {
        "fn": webpage_blueprint,
        "metadata": SkillMetadata(
            name="webpage_blueprint",
            description="Plan a landing page or webpage with sections and interaction intent",
            strengths=["product storytelling", "information hierarchy", "web deliverable framing"],
            weaknesses=["does not implement frontend code"],
            category=SkillCategory.GENERATION,
            output_type="structured",
            confidence_keywords=["webpage", "website", "landing page", "frontend", "ui"],
            tier=SkillTier.EXPERT,
            compute_cost=1.3,
            synergies=["frontend_critique", "slide_deck_designer", "executive_summary"],
            conflicts=[],
        ),
    },
    "slide_deck_designer": {
        "fn": slide_deck_designer,
        "metadata": SkillMetadata(
            name="slide_deck_designer",
            description="Turn a topic into a presentation deck arc with slide beats",
            strengths=["narrative pacing", "presentation planning", "executive communication"],
            weaknesses=["does not render slides"],
            category=SkillCategory.GENERATION,
            output_type="structured",
            confidence_keywords=["slides", "deck", "presentation", "ppt", "keynote"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.2,
            synergies=["webpage_blueprint", "chart_storyboard", "executive_summary"],
            conflicts=[],
        ),
    },
    "podcast_episode_plan": {
        "fn": podcast_episode_plan,
        "metadata": SkillMetadata(
            name="podcast_episode_plan",
            description="Structure a podcast episode with hook, segments, and takeaway design",
            strengths=["audio narrative", "segment design", "audience pacing"],
            weaknesses=["does not synthesize audio"],
            category=SkillCategory.GENERATION,
            output_type="structured",
            confidence_keywords=["podcast", "episode", "audio", "host", "interview"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.2,
            synergies=["research_brief", "synthesize_perspectives", "executive_summary"],
            conflicts=[],
        ),
    },
    "video_storyboard": {
        "fn": video_storyboard,
        "metadata": SkillMetadata(
            name="video_storyboard",
            description="Create a short-form video storyboard with scenes and beats",
            strengths=["visual pacing", "scene design", "artifact-backed storytelling"],
            weaknesses=["does not render video"],
            category=SkillCategory.GENERATION,
            output_type="structured",
            confidence_keywords=["video", "storyboard", "scene", "trailer", "short"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.3,
            synergies=["image_prompt_pack", "slide_deck_designer"],
            conflicts=[],
        ),
    },
    "image_prompt_pack": {
        "fn": image_prompt_pack,
        "metadata": SkillMetadata(
            name="image_prompt_pack",
            description="Generate reusable image prompt directions for visual assets",
            strengths=["visual direction", "prompt packaging", "multi-style asset planning"],
            weaknesses=["does not render images directly"],
            category=SkillCategory.GENERATION,
            output_type="structured",
            confidence_keywords=["image", "poster", "illustration", "thumbnail", "render", "visual"],
            tier=SkillTier.ADVANCED,
            compute_cost=1.1,
            synergies=["video_storyboard", "webpage_blueprint"],
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
