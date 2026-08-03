from __future__ import annotations

from refund_abuse_risk.config import load_operating_point, load_policy
from refund_abuse_risk.schemas.models import SuggestedTier
from refund_abuse_risk.scoring.policy import (
    combine_scores,
    evaluate_hard_gates,
    tier_for_score,
    tier_from_heads,
)
from refund_abuse_risk.scoring.thresholds import (
    pattern_flag_recall,
    recommend_head_thresholds,
    threshold_at_recall,
)


def test_strong_fraud_is_only_hard_gate() -> None:
    policy = load_policy()
    features = {
        "user_id": "u1",
        "driver_id": "d1",
        "vendor_id": "v1",
        "device_id": "dev1",
        "user_days_since_signup": 45,
        "user_lifetime_orders": 20,
        "user_lifetime_refund_count": 9,
        "user_refund_to_ltv_ratio": 0.9,
        "user_orders_30d": 12,
        "user_refund_count_7d": 9,
        "user_refund_count_30d": 9,
        "user_refund_rate_30d": 0.9,
        "user_refund_gmv_pct_30d": 0.9,
        "combined_refund_count_30d": 20,
        "related_max_refund_rate_30d": 0.9,
        "strong_fraud_label": 0,
    }
    gated, items = evaluate_hard_gates(
        features,
        entity_scores={"user": 90, "driver": 90, "vendor": 90},
        link_scores={"ud": 90, "uv": 90, "vd": 90, "uvd": 90},
        device_cluster_score=90,
        policy=policy,
        market="SG",
        vertical="food",
    )
    assert gated is False
    assert items == []


def test_strong_fraud_hard_gates() -> None:
    policy = load_policy()
    gated, items = evaluate_hard_gates(
        {
            "user_id": "u1",
            "prior_strong_fraud": 1,
        },
        entity_scores={},
        link_scores={},
        device_cluster_score=0,
        policy=policy,
        market="SG",
        vertical="food",
    )
    assert gated is True
    assert any(i.reason_code == "STRONG_FRAUD_LABEL" for i in items)


def test_hard_gate_forces_auto_deny() -> None:
    op = load_operating_point()
    combined, tier = combine_scores(10.0, 10.0, 20.0, op, hard_gated=True)
    assert tier == SuggestedTier.AUTO_DENY
    assert combined >= 90.0


def test_tier_from_heads() -> None:
    op = load_operating_point()
    assert tier_from_heads(10, 10, op) == SuggestedTier.AUTO_APPROVE
    assert tier_from_heads(40, 10, op) == SuggestedTier.SOFT_FRICTION
    assert tier_from_heads(55, 10, op) == SuggestedTier.HOLD_REVIEW
    assert tier_from_heads(10, 70, op) == SuggestedTier.AUTO_DENY


def test_tier_bands_secondary() -> None:
    op = load_operating_point()
    assert tier_for_score(10, op) == SuggestedTier.AUTO_APPROVE
    assert tier_for_score(40, op) == SuggestedTier.SOFT_FRICTION
    assert tier_for_score(60, op) == SuggestedTier.HOLD_REVIEW
    assert tier_for_score(90, op) == SuggestedTier.AUTO_DENY


def test_combine_uses_decision_score_not_prior_banding() -> None:
    op = load_operating_point()
    low_prior_combined, tier_low = combine_scores(
        80.0, 10.0, 5.0, op, hard_gated=False, decision_score=80.0
    )
    high_prior_combined, tier_high = combine_scores(
        80.0, 10.0, 95.0, op, hard_gated=False, decision_score=80.0
    )
    # Prior must not change tier under decision_primary.
    assert tier_low == tier_high
    assert low_prior_combined == high_prior_combined
    assert tier_low == SuggestedTier.AUTO_DENY


def test_high_decision_score_raises_tier() -> None:
    op = load_operating_point()
    _, tier_low = combine_scores(40.0, 10.0, 20.0, op, hard_gated=False, decision_score=40.0)
    _, tier_high = combine_scores(40.0, 90.0, 20.0, op, hard_gated=False, decision_score=90.0)
    assert tier_high == SuggestedTier.AUTO_DENY
    assert tier_low == SuggestedTier.SOFT_FRICTION


def test_threshold_at_recall_and_recommend() -> None:
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    s = [90, 80, 40, 30, 20, 10, 5, 4, 3, 2]
    thr = threshold_at_recall(y, s, 0.98)
    assert thr is not None
    assert thr <= 40.0
    rec = recommend_head_thresholds(y, s, y, s, target_recall=0.98)
    assert rec["abuse_soft_friction"] <= rec["abuse_hold_review"] <= rec["abuse_auto_deny"]
    assert pattern_flag_recall(y, y, s, s, rec["abuse_soft_friction"], rec["fraud_soft_friction"]) >= 0.98
