"""Tests for system modes and policy bundles."""

from app.policy.center import RiskLevel, SystemMode, infer_risk_level, policy_for_mode


def test_policy_modes_exist() -> None:
    for mode in [SystemMode.FAST, SystemMode.BALANCED, SystemMode.DEEP, SystemMode.SAFETY_CRITICAL]:
        bundle = policy_for_mode(mode)
        payload = bundle.to_dict()
        assert payload["mode"] == mode.value
        assert payload["budget"]["max_skills"] >= 1


def test_safety_mode_requires_verifier() -> None:
    bundle = policy_for_mode(SystemMode.SAFETY_CRITICAL)
    assert bundle.risk.require_verifier is True
    assert bundle.risk.require_human_approval is True


def test_risk_inference() -> None:
    assert infer_risk_level("audit this legal compliance plan") == RiskLevel.CRITICAL
    assert infer_risk_level("summarize this short note") in {RiskLevel.LOW, RiskLevel.MEDIUM}
