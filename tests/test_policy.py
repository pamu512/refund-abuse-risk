from __future__ import annotations

from refund_abuse_risk.config import load_operating_point, load_policy
from refund_abuse_risk.schemas.models import SuggestedTier
from refund_abuse_risk.scoring.policy import combine_scores, evaluate_hard_gates, tier_for_score


def test_hard_gate_user_refund_cap() -> None:
    policy = load_policy()
    features = {
        "user_id": "u1",
        "driver_id": "d1",
        "vendor_id": "v1",
        "device_id": "dev1",
        "user_refund_count_7d": 9,
        "user_refund_count_30d": 9,
        "user_refund_rate_30d": 0.1,
        "user_refund_gmv_pct_30d": 0.1,
        "driver_refund_count_30d": 0,
        "driver_refund_rate_30d": 0,
        "vendor_refund_count_30d": 0,
        "vendor_refund_rate_30d": 0,
        "vendor_refund_gmv_pct_30d": 0,
        "strong_fraud_label": 0,
    }
    gated, items = evaluate_hard_gates(
        features,
        entity_scores={"user": 40, "driver": 10, "vendor": 10},
        link_scores={"ud": 10, "uv": 10, "vd": 10, "uvd": 10},
        device_cluster_score=10,
        policy=policy,
        market="SG",
        vertical="food",
    )
    assert gated is True
    assert any(i.reason_code == "USER_REFUND_COUNT_7D" for i in items)


def test_hard_gate_uvd_and_device() -> None:
    policy = load_policy()
    features = {
        "user_id": "u1",
        "driver_id": "d1",
        "vendor_id": "v1",
        "device_id": "dev1",
        "user_refund_count_7d": 0,
        "user_refund_count_30d": 0,
        "user_refund_rate_30d": 0,
        "user_refund_gmv_pct_30d": 0,
        "driver_refund_count_30d": 0,
        "driver_refund_rate_30d": 0,
        "vendor_refund_count_30d": 0,
        "vendor_refund_rate_30d": 0,
        "vendor_refund_gmv_pct_30d": 0,
        "strong_fraud_label": 0,
    }
    gated, items = evaluate_hard_gates(
        features,
        entity_scores={"user": 10, "driver": 10, "vendor": 10},
        link_scores={"ud": 10, "uv": 10, "vd": 10, "uvd": 90},
        device_cluster_score=85,
        policy=policy,
        market="SG",
        vertical="food",
    )
    assert gated is True
    codes = {i.reason_code for i in items}
    assert "LINK_UVD_HARD" in codes
    assert "DEVICE_CLUSTER_HARD" in codes


def test_hard_gate_forces_auto_deny() -> None:
    op = load_operating_point()
    combined, tier = combine_scores(10.0, 10.0, 20.0, op, hard_gated=True)
    assert tier == SuggestedTier.AUTO_DENY
    assert combined >= 76.0


def test_tier_bands() -> None:
    op = load_operating_point()
    assert tier_for_score(10, op) == SuggestedTier.AUTO_APPROVE
    assert tier_for_score(40, op) == SuggestedTier.SOFT_FRICTION
    assert tier_for_score(60, op) == SuggestedTier.HOLD_REVIEW
    assert tier_for_score(90, op) == SuggestedTier.AUTO_DENY
