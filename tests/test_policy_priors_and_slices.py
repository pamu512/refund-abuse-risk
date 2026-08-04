from __future__ import annotations

from refund_abuse_risk.config import load_vertical_policy
from refund_abuse_risk.features.builders import FEATURE_COLUMNS, build_order_feature_row
from refund_abuse_risk.features.policy_priors import (
    POLICY_FEATURE_COLUMNS,
    policy_prior_features,
    resolve_policy_priors,
)
from refund_abuse_risk.scoring.decision import recommended_overlays_from_slices


def test_policy_priors_vertical_and_market_override() -> None:
    cfg = load_vertical_policy()
    food = resolve_policy_priors(market="SG", vertical="food", cfg=cfg)
    qcom = resolve_policy_priors(market="SG", vertical="qcommerce", cfg=cfg)
    us_food = resolve_policy_priors(market="US", vertical="food", cfg=cfg)
    us_q = resolve_policy_priors(market="US", vertical="qcommerce", cfg=cfg)
    assert food["claim_window_hours"] == 36
    assert qcom["claim_window_hours"] == 18
    assert us_food["claim_window_hours"] == 24
    assert us_q["claim_window_hours"] == 12
    assert us_food["remedy_cash_bias"] > food["remedy_cash_bias"]


def test_claim_window_remaining_frac_clamped() -> None:
    feats = policy_prior_features(
        {
            "market": "US",
            "vertical": "qcommerce",
            "delivered_ts": "2026-08-01T00:00:00Z",
            "claim_ts": "2026-08-01T06:00:00Z",  # 6h into 12h window
        }
    )
    assert feats["hours_since_delivery"] == 6.0
    assert abs(feats["claim_window_remaining_frac"] - 0.5) < 1e-6

    late = policy_prior_features(
        {
            "market": "US",
            "vertical": "qcommerce",
            "delivered_ts": "2026-08-01T00:00:00Z",
            "claim_ts": "2026-08-02T00:00:00Z",  # past window
        }
    )
    assert late["claim_window_remaining_frac"] == 0.0

    missing = policy_prior_features({"market": "SG", "vertical": "food"})
    assert missing["hours_since_delivery"] == 0.0
    assert missing["claim_window_remaining_frac"] == 0.0

    # event_ts alone (no claim_ts) must not fabricate a 0-hour window hit
    event_only = policy_prior_features(
        {
            "market": "SG",
            "vertical": "food",
            "status": "delivered",
            "event_ts": "2026-08-01T00:00:00Z",
        }
    )
    assert event_only["claim_window_remaining_frac"] == 0.0


def test_policy_features_on_order_row() -> None:
    import pandas as pd

    order = {
        "order_id": "O1",
        "user_id": "U1",
        "driver_id": "D1",
        "vendor_id": "V1",
        "device_id": "DEV1",
        "market": "US",
        "vertical": "food",
        "amount": 20.0,
        "status": "delivered",
        "event_ts": "2026-08-01T12:00:00Z",
        "delivered_ts": "2026-08-01T10:00:00Z",
        "claim_ts": "2026-08-01T12:00:00Z",
    }
    hist = pd.DataFrame(columns=["user_id", "event_ts", "is_refund", "amount"])
    devices = pd.DataFrame(
        [{"user_id": "U1", "device_id": "DEV1", "cluster_id": "C1", "last_seen_ts": order["event_ts"]}]
    )
    row = build_order_feature_row(order, hist, devices)
    for col in POLICY_FEATURE_COLUMNS:
        assert col in FEATURE_COLUMNS
        assert col in row
    assert row["policy_claim_window_hours"] == 24.0


def test_recommended_overlays_skips_thin() -> None:
    overlays = recommended_overlays_from_slices(
        {
            "SG|food": {
                "market": "SG",
                "vertical": "food",
                "soft_friction": 30,
                "hold_review": 45,
                "auto_deny": 70,
                "ok": True,
                "promote_eligible": True,
                "thin_slice": False,
            },
            "ID|qcommerce": {
                "market": "ID",
                "vertical": "qcommerce",
                "soft_friction": 28,
                "hold_review": 40,
                "auto_deny": 65,
                "ok": False,
                "promote_eligible": False,
                "thin_slice": True,
            },
        }
    )
    assert len(overlays) == 1
    assert overlays[0]["market"] == "SG"
    assert overlays[0]["soft_friction"] == 30.0
