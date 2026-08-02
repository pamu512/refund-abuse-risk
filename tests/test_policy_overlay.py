from __future__ import annotations

from refund_abuse_risk.config import load_operating_point, load_policy
from refund_abuse_risk.schemas.models import SuggestedTier
from refund_abuse_risk.scoring.policy import combine_scores, evaluate_hard_gates


def test_strong_fraud_hard_gate():
    hard, items = evaluate_hard_gates(
        {
            "user_id": "U",
            "driver_id": "D",
            "vendor_id": "V",
            "device_id": "DEV",
            "strong_fraud_label": 1,
            "user_refund_count_7d": 0,
            "user_refund_rate_30d": 0,
        },
        entity_scores={"user": 10, "driver": 0, "vendor": 0},
        link_scores={"ud": 0, "uv": 0, "vd": 0, "uvd": 0},
        device_cluster_score=0,
        policy=load_policy(),
        market="SG",
        vertical="food",
    )
    assert hard is True
    assert any(i.reason_code == "STRONG_FRAUD_LABEL" for i in items)
    _, tier = combine_scores(10, 10, 10, load_operating_point(), hard_gated=True)
    assert tier == SuggestedTier.AUTO_DENY


def test_user_refund_cap_hard_gate():
    hard, items = evaluate_hard_gates(
        {
            "user_id": "U_ABUSE",
            "driver_id": "D",
            "vendor_id": "V",
            "device_id": "DEV",
            "strong_fraud_label": 0,
            "user_refund_count_7d": 9,
            "user_refund_count_30d": 20,
            "user_refund_rate_30d": 0.5,
            "user_refund_gmv_pct_30d": 0.5,
            "driver_refund_count_30d": 0,
            "driver_refund_rate_30d": 0,
            "vendor_refund_count_30d": 0,
            "vendor_refund_rate_30d": 0,
            "vendor_refund_gmv_pct_30d": 0,
        },
        entity_scores={"user": 70, "driver": 0, "vendor": 0},
        link_scores={"ud": 0, "uv": 0, "vd": 0, "uvd": 0},
        device_cluster_score=10,
        policy=load_policy(),
        market="SG",
        vertical="food",
    )
    assert hard is True
    assert any(i.reason_code == "USER_REFUND_COUNT_7D" for i in items)


def test_uvd_link_hard_gate():
    hard, items = evaluate_hard_gates(
        {
            "user_id": "U",
            "driver_id": "D",
            "vendor_id": "V",
            "device_id": "DEV",
            "strong_fraud_label": 0,
            "user_refund_count_7d": 0,
            "user_refund_count_30d": 0,
            "user_refund_rate_30d": 0,
            "user_refund_gmv_pct_30d": 0,
            "driver_refund_count_30d": 0,
            "driver_refund_rate_30d": 0,
            "vendor_refund_count_30d": 0,
            "vendor_refund_rate_30d": 0,
            "vendor_refund_gmv_pct_30d": 0,
        },
        entity_scores={"user": 0, "driver": 0, "vendor": 0},
        link_scores={"ud": 0, "uv": 0, "vd": 0, "uvd": 90},
        device_cluster_score=10,
        policy=load_policy(),
        market="SG",
        vertical="food",
    )
    assert hard is True
    assert any(i.reason_code == "LINK_UVD_HARD" for i in items)
