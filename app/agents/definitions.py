"""Agent definitions for the LangGraph Skill Router."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.state import AgentPersonality, AgentStyle, ConflictPolicy


@dataclass
class AgentProfile:
    """Describes an agent's identity and routing preferences."""

    name: str
    description: str
    style: AgentStyle
    domains: list[str]
    preferred_skills: list[str] = field(default_factory=list)
    personality: AgentPersonality = field(
        default_factory=lambda: AgentPersonality(
            risk_tolerance=0.5,
            creativity_bias=0.5,
            diversity_preference=0.5,
            confidence_threshold=0.2,
            collaboration_tendency=0.5,
            depth_vs_breadth=0.5,
        )
    )
    conflict_resolution: ConflictPolicy = ConflictPolicy.WEIGHTED
    max_concurrent_skills: int = 3
    collaboration_partners: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)


AGENT_REGISTRY: dict[str, AgentProfile] = {
    "AnalysisAgent": AgentProfile(
        name="AnalysisAgent",
        description=(
            "Deep analytical reasoning - risk assessment, data interpretation, "
            "root cause analysis, quantitative evaluation"
        ),
        style=AgentStyle.CAUTIOUS,
        domains=[
            "analysis",
            "risk",
            "data",
            "evaluation",
            "assessment",
            "audit",
            "root cause",
            "quantitative",
        ],
        preferred_skills=[
            "identify_risks",
            "compare_options",
            "extract_facts",
            "detect_anomalies",
            "validate_claims",
        ],
        personality=AgentPersonality(
            risk_tolerance=0.3,
            creativity_bias=0.2,
            diversity_preference=0.7,
            confidence_threshold=0.3,
            collaboration_tendency=0.6,
            depth_vs_breadth=0.2,
        ),
        conflict_resolution=ConflictPolicy.WEIGHTED,
        collaboration_partners=["CriticAgent", "ResearchAgent"],
        anti_patterns=["brainstorm", "creative writing", "casual chat"],
    ),
    "SummaryAgent": AgentProfile(
        name="SummaryAgent",
        description=(
            "Concise communication - summaries, briefs, overviews, distilling "
            "complex info into digestible format"
        ),
        style=AgentStyle.BALANCED,
        domains=[
            "summary",
            "overview",
            "brief",
            "digest",
            "report",
            "condense",
            "highlight",
            "key points",
        ],
        preferred_skills=["executive_summary", "extract_facts", "prioritize_items"],
        personality=AgentPersonality(
            risk_tolerance=0.4,
            creativity_bias=0.3,
            diversity_preference=0.5,
            confidence_threshold=0.25,
            collaboration_tendency=0.4,
            depth_vs_breadth=0.7,
        ),
        conflict_resolution=ConflictPolicy.HIERARCHICAL,
        collaboration_partners=["AnalysisAgent"],
        anti_patterns=["deep technical analysis", "code generation"],
    ),
    "CreativeAgent": AgentProfile(
        name="CreativeAgent",
        description=(
            "Idea generation and creative exploration - lateral thinking, "
            "unconventional angles, innovation"
        ),
        style=AgentStyle.CREATIVE,
        domains=[
            "brainstorm",
            "idea",
            "creative",
            "innovation",
            "explore",
            "imagine",
            "what if",
            "lateral",
        ],
        preferred_skills=[
            "brainstorm_ideas",
            "generate_recommendations",
            "generate_analogies",
            "synthesize_perspectives",
        ],
        personality=AgentPersonality(
            risk_tolerance=0.9,
            creativity_bias=0.95,
            diversity_preference=0.9,
            confidence_threshold=0.1,
            collaboration_tendency=0.7,
            depth_vs_breadth=0.8,
        ),
        conflict_resolution=ConflictPolicy.DEBATE,
        collaboration_partners=["CriticAgent"],
        anti_patterns=["precise calculation", "strict compliance"],
    ),
    "AdvisorAgent": AgentProfile(
        name="AdvisorAgent",
        description=(
            "Actionable advice and strategic recommendations - decision support, "
            "prioritized action items"
        ),
        style=AgentStyle.AGGRESSIVE,
        domains=[
            "recommend",
            "advise",
            "strategy",
            "plan",
            "decision",
            "action",
            "priority",
            "next step",
        ],
        preferred_skills=[
            "generate_recommendations",
            "identify_risks",
            "prioritize_items",
            "compare_options",
        ],
        personality=AgentPersonality(
            risk_tolerance=0.6,
            creativity_bias=0.4,
            diversity_preference=0.4,
            confidence_threshold=0.35,
            collaboration_tendency=0.3,
            depth_vs_breadth=0.4,
        ),
        conflict_resolution=ConflictPolicy.HIERARCHICAL,
        collaboration_partners=["AnalysisAgent"],
        anti_patterns=["open-ended exploration", "pure research"],
    ),
    "ResearchAgent": AgentProfile(
        name="ResearchAgent",
        description=(
            "Deep research and evidence gathering - systematic investigation, "
            "source synthesis, claim verification"
        ),
        style=AgentStyle.CAUTIOUS,
        domains=[
            "research",
            "investigate",
            "evidence",
            "source",
            "verify",
            "study",
            "literature",
            "deep dive",
        ],
        preferred_skills=[
            "extract_facts",
            "validate_claims",
            "synthesize_perspectives",
            "build_timeline",
        ],
        personality=AgentPersonality(
            risk_tolerance=0.2,
            creativity_bias=0.3,
            diversity_preference=0.8,
            confidence_threshold=0.35,
            collaboration_tendency=0.7,
            depth_vs_breadth=0.15,
        ),
        conflict_resolution=ConflictPolicy.VOTING,
        collaboration_partners=["AnalysisAgent", "CriticAgent"],
        anti_patterns=["quick summary", "creative brainstorm"],
    ),
    "DebugAgent": AgentProfile(
        name="DebugAgent",
        description=(
            "Problem diagnosis and anomaly detection - root cause analysis, "
            "pattern breaking, debugging"
        ),
        style=AgentStyle.CAUTIOUS,
        domains=[
            "debug",
            "diagnose",
            "anomaly",
            "error",
            "bug",
            "root cause",
            "troubleshoot",
            "pattern",
        ],
        preferred_skills=[
            "detect_anomalies",
            "identify_risks",
            "extract_facts",
            "validate_claims",
            "build_timeline",
        ],
        personality=AgentPersonality(
            risk_tolerance=0.3,
            creativity_bias=0.4,
            diversity_preference=0.6,
            confidence_threshold=0.2,
            collaboration_tendency=0.5,
            depth_vs_breadth=0.1,
        ),
        conflict_resolution=ConflictPolicy.WEIGHTED,
        collaboration_partners=["ResearchAgent", "AnalysisAgent"],
        anti_patterns=["summarize", "brainstorm", "recommend"],
    ),
    "PlannerAgent": AgentProfile(
        name="PlannerAgent",
        description=(
            "Strategic planning and sequencing - roadmaps, timelines, dependency "
            "mapping, resource allocation"
        ),
        style=AgentStyle.BALANCED,
        domains=[
            "plan",
            "roadmap",
            "timeline",
            "sequence",
            "phase",
            "milestone",
            "schedule",
            "dependency",
        ],
        preferred_skills=[
            "build_timeline",
            "prioritize_items",
            "generate_recommendations",
            "compare_options",
        ],
        personality=AgentPersonality(
            risk_tolerance=0.4,
            creativity_bias=0.3,
            diversity_preference=0.6,
            confidence_threshold=0.3,
            collaboration_tendency=0.6,
            depth_vs_breadth=0.5,
        ),
        conflict_resolution=ConflictPolicy.HIERARCHICAL,
        collaboration_partners=["AdvisorAgent", "AnalysisAgent"],
        anti_patterns=["pure research", "creative writing"],
    ),
    "CriticAgent": AgentProfile(
        name="CriticAgent",
        description=(
            "Critical evaluation and challenge - devil's advocate, assumption "
            "testing, weakness identification"
        ),
        style=AgentStyle.AGGRESSIVE,
        domains=[
            "critique",
            "challenge",
            "weakness",
            "flaw",
            "assumption",
            "counter",
            "devil",
            "stress test",
        ],
        preferred_skills=[
            "validate_claims",
            "identify_risks",
            "detect_anomalies",
            "compare_options",
        ],
        personality=AgentPersonality(
            risk_tolerance=0.5,
            creativity_bias=0.6,
            diversity_preference=0.7,
            confidence_threshold=0.15,
            collaboration_tendency=0.8,
            depth_vs_breadth=0.3,
        ),
        conflict_resolution=ConflictPolicy.DEBATE,
        collaboration_partners=["CreativeAgent", "AnalysisAgent"],
        anti_patterns=["positive brainstorm", "executive summary"],
    ),
}


def get_agent(name: str) -> AgentProfile | None:
    """Get a single agent by name."""

    return AGENT_REGISTRY.get(name)


def list_agents() -> list[AgentProfile]:
    """Return all built-in agents."""

    return list(AGENT_REGISTRY.values())


def find_collaborators(agent_name: str) -> list[AgentProfile]:
    """Find preferred collaborators for a given agent."""

    agent = get_agent(agent_name)
    if not agent:
        return []
    collaborators: list[AgentProfile] = []
    for name in agent.collaboration_partners:
        candidate = get_agent(name)
        if candidate:
            collaborators.append(candidate)
    return collaborators


def agents_for_domains(domains: list[str]) -> list[tuple[AgentProfile, int]]:
    """Return agents ranked by explicit domain hit counts."""

    results: list[tuple[AgentProfile, int]] = []
    lowered = [domain.lower() for domain in domains]
    for agent in list_agents():
        hits = sum(1 for domain in lowered if any(domain in d.lower() for d in agent.domains))
        if hits > 0:
            results.append((agent, hits))
    results.sort(key=lambda item: item[1], reverse=True)
    return results
