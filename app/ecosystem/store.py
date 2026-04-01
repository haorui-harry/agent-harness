"""Local skill marketplace storage and import/export utilities."""

from __future__ import annotations

import json
from pathlib import Path

from app.ecosystem.models import MarketplaceSkill, from_dict, to_dict

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MARKETPLACE_FILE = DATA_DIR / "marketplace.json"


def _bootstrap_payload() -> dict:
    return {
        "skills": [
            {
                "skill_id": "market.risk_heatmap",
                "provider": "core-labs",
                "version": "0.1.0",
                "tags": ["risk", "analysis", "governance"],
                "installs": 120,
                "uses": 87,
                "success": 77,
                "failures": 10,
                "avg_rating": 4.6,
                "rating_count": 29,
                "last_updated": "2026-01-15",
                "trending_score": 0.18,
                "version_history": [
                    {"version": "0.1.0", "changelog": "Initial release", "date": "2025-12-01"}
                ],
                "compatibility": ["identify_risks", "prioritize_items"],
                "metadata": {
                    "name": "risk_heatmap",
                    "description": "Score risks by likelihood and impact for prioritization.",
                    "strengths": ["risk scoring", "prioritization", "portfolio view"],
                    "weaknesses": ["limited narrative quality"],
                    "category": "analysis",
                    "output_type": "structured",
                    "confidence_keywords": ["risk", "impact", "likelihood", "prioritize"],
                    "tier": "advanced",
                    "compute_cost": 1.4,
                    "synergies": ["identify_risks", "prioritize_items"],
                    "conflicts": [],
                },
            },
            {
                "skill_id": "market.evidence_matrix",
                "provider": "research-hub",
                "version": "0.2.1",
                "tags": ["research", "evidence", "comparison"],
                "installs": 95,
                "uses": 71,
                "success": 65,
                "failures": 6,
                "avg_rating": 4.4,
                "rating_count": 22,
                "last_updated": "2026-02-03",
                "trending_score": 0.21,
                "version_history": [
                    {"version": "0.2.1", "changelog": "Confidence calibration", "date": "2026-02-03"}
                ],
                "compatibility": ["extract_facts", "validate_claims"],
                "metadata": {
                    "name": "evidence_matrix",
                    "description": "Build claim-evidence-confidence matrix from source text.",
                    "strengths": ["evidence synthesis", "fact confidence", "traceability"],
                    "weaknesses": ["creative ideation"],
                    "category": "recall",
                    "output_type": "structured",
                    "confidence_keywords": ["evidence", "claim", "confidence", "source"],
                    "tier": "expert",
                    "compute_cost": 1.8,
                    "synergies": ["extract_facts", "validate_claims"],
                    "conflicts": ["brainstorm_ideas"],
                },
            },
            {
                "skill_id": "market.board_brief",
                "provider": "exec-ai",
                "version": "1.0.0",
                "tags": ["communication", "executive", "brief"],
                "installs": 180,
                "uses": 150,
                "success": 133,
                "failures": 17,
                "avg_rating": 4.8,
                "rating_count": 55,
                "last_updated": "2026-03-11",
                "trending_score": 0.25,
                "version_history": [
                    {"version": "1.0.0", "changelog": "Stable release", "date": "2026-01-20"}
                ],
                "compatibility": ["executive_summary", "generate_recommendations"],
                "metadata": {
                    "name": "board_brief",
                    "description": "Produce board-level concise summary with recommendations.",
                    "strengths": ["executive tone", "clarity", "decision framing"],
                    "weaknesses": ["low-level extraction"],
                    "category": "communication",
                    "output_type": "text",
                    "confidence_keywords": ["summary", "board", "brief", "recommendation"],
                    "tier": "advanced",
                    "compute_cost": 1.1,
                    "synergies": ["executive_summary", "prioritize_items"],
                    "conflicts": [],
                },
            },
            {
                "skill_id": "market.sentiment_scanner",
                "provider": "nlp-works",
                "version": "1.2.0",
                "tags": ["sentiment", "analysis", "opinion", "emotion"],
                "installs": 95,
                "uses": 82,
                "success": 74,
                "failures": 8,
                "avg_rating": 4.3,
                "rating_count": 31,
                "last_updated": "2026-02-18",
                "trending_score": 0.16,
                "version_history": [
                    {"version": "1.2.0", "changelog": "Improved tone scoring", "date": "2026-02-18"}
                ],
                "compatibility": ["synthesize_perspectives", "executive_summary"],
                "metadata": {
                    "name": "sentiment_scanner",
                    "description": "Analyze sentiment polarity and emotional tone across text segments.",
                    "strengths": ["sentiment detection", "emotion classification", "tone analysis"],
                    "weaknesses": ["sarcasm detection", "multilingual support"],
                    "category": "analysis",
                    "output_type": "structured",
                    "confidence_keywords": ["sentiment", "emotion", "tone", "feeling", "positive", "negative"],
                    "tier": "advanced",
                    "compute_cost": 1.3,
                    "synergies": ["synthesize_perspectives"],
                    "conflicts": [],
                },
            },
            {
                "skill_id": "market.dependency_mapper",
                "provider": "arch-ai",
                "version": "0.3.0",
                "tags": ["dependency", "graph", "relationship", "mapping"],
                "installs": 68,
                "uses": 45,
                "success": 39,
                "failures": 6,
                "avg_rating": 4.1,
                "rating_count": 15,
                "last_updated": "2026-01-30",
                "trending_score": 0.10,
                "version_history": [
                    {"version": "0.3.0", "changelog": "Added causal mapping", "date": "2026-01-30"}
                ],
                "compatibility": ["build_timeline", "prioritize_items"],
                "metadata": {
                    "name": "dependency_mapper",
                    "description": "Map dependencies, relationships, and causal chains in structured content.",
                    "strengths": ["relationship extraction", "causal chain detection", "dependency graphing"],
                    "weaknesses": ["needs structured input", "less useful for narrative text"],
                    "category": "extraction",
                    "output_type": "graph",
                    "confidence_keywords": ["dependency", "relationship", "cause", "effect", "chain", "graph"],
                    "tier": "expert",
                    "compute_cost": 1.7,
                    "synergies": ["build_timeline", "extract_facts"],
                    "conflicts": [],
                },
            },
            {
                "skill_id": "market.scenario_generator",
                "provider": "futures-lab",
                "version": "0.1.0",
                "tags": ["scenario", "future", "simulation", "what-if"],
                "installs": 42,
                "uses": 28,
                "success": 24,
                "failures": 4,
                "avg_rating": 4.5,
                "rating_count": 12,
                "last_updated": "2026-03-20",
                "trending_score": 0.33,
                "version_history": [
                    {"version": "0.1.0", "changelog": "Initial release", "date": "2026-03-20"}
                ],
                "compatibility": ["brainstorm_ideas", "generate_recommendations"],
                "metadata": {
                    "name": "scenario_generator",
                    "description": "Generate alternative scenarios and what-if analyses based on input conditions.",
                    "strengths": ["scenario planning", "what-if analysis", "alternative futures"],
                    "weaknesses": ["speculative by nature", "needs clear assumptions"],
                    "category": "generation",
                    "output_type": "structured",
                    "confidence_keywords": ["scenario", "what if", "future", "alternative", "simulate", "possibility"],
                    "tier": "advanced",
                    "compute_cost": 1.4,
                    "synergies": ["identify_risks", "generate_recommendations"],
                    "conflicts": [],
                },
            },
        ]
    }


def ensure_store() -> None:
    """Ensure marketplace file exists and contains baseline skills."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed = _bootstrap_payload()

    if not MARKETPLACE_FILE.exists():
        MARKETPLACE_FILE.write_text(json.dumps(seed, indent=2), encoding="utf-8")
        return

    current = json.loads(MARKETPLACE_FILE.read_text(encoding="utf-8"))
    current_skills = current.get("skills", [])
    current_ids = {row.get("skill_id") for row in current_skills}

    changed = False
    for row in seed["skills"]:
        if row["skill_id"] not in current_ids:
            current_skills.append(row)
            changed = True

    if changed:
        current["skills"] = current_skills
        MARKETPLACE_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")


def load_marketplace() -> list[MarketplaceSkill]:
    """Load marketplace skills from disk."""

    ensure_store()
    payload = json.loads(MARKETPLACE_FILE.read_text(encoding="utf-8"))
    return [from_dict(item) for item in payload.get("skills", [])]


def save_marketplace(skills: list[MarketplaceSkill]) -> None:
    """Persist marketplace skills to disk."""

    ensure_store()
    payload = {"skills": [to_dict(item) for item in skills]}
    MARKETPLACE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def import_marketplace_from_file(path: str) -> int:
    """Import third-party marketplace skills from another JSON file."""

    source = Path(path)
    if not source.exists():
        return 0

    ensure_store()
    external_payload = json.loads(source.read_text(encoding="utf-8"))
    external_rows = external_payload.get("skills", [])

    current_payload = json.loads(MARKETPLACE_FILE.read_text(encoding="utf-8"))
    current_rows = current_payload.get("skills", [])
    index = {row.get("skill_id"): row for row in current_rows}

    imported = 0
    for row in external_rows:
        skill_id = row.get("skill_id")
        if not skill_id:
            continue
        if skill_id not in index:
            current_rows.append(row)
            imported += 1

    if imported:
        current_payload["skills"] = current_rows
        MARKETPLACE_FILE.write_text(json.dumps(current_payload, indent=2), encoding="utf-8")

    return imported


def export_marketplace_snapshot(path: str) -> None:
    """Export current marketplace snapshot to target path."""

    ensure_store()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(MARKETPLACE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
