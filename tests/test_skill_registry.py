"""Tests for SkillCard metadata and lifecycle helpers."""

from app.skills.registry import (
    get_skill_card,
    get_skill_lifecycle_status,
    list_all_skills,
    validate_skill_metadata,
)


def test_skill_card_contains_core_fields() -> None:
    card = get_skill_card("identify_risks")
    assert card is not None
    assert card["name"] == "identify_risks"
    assert "failure_modes" in card
    assert "lifecycle_stage" in card
    assert "runtime_reliability" in card


def test_skill_lifecycle_status_known_skill() -> None:
    status = get_skill_lifecycle_status("identify_risks")
    assert status["skill"] == "identify_risks"
    assert status["status"] in {"active", "needs_review", "drift_alert", "observation"}


def test_metadata_validation_runs_for_all_skills() -> None:
    skills = list_all_skills()
    assert len(skills) >= 12
    for meta in skills[:12]:
        issues = validate_skill_metadata(meta)
        assert isinstance(issues, list)
