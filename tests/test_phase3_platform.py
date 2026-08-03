from __future__ import annotations

import pandas as pd

from refund_abuse_risk.config import load_effect_rules, load_refund_budget
from refund_abuse_risk.control_plane.effects import resolve_refund_effect
from refund_abuse_risk.features.builders import build_order_feature_row
from refund_abuse_risk.integrations.device_vision import (
    adapt_claim_vision,
    adapt_device_intelligence,
    merge_platform_signals,
)
from refund_abuse_risk.labels.discovery import mint_weak_labels_from_uv_anomaly
from refund_abuse_risk.schemas.models import RefundEffect, SuggestedTier
from refund_abuse_risk.scoring.budget import evaluate_refund_budget
from refund_abuse_risk.scoring.monitoring import (
    expected_calibration_error,
    population_stability_index,
)


def test_device_vision_adapters_normalize() -> None:
    device = adapt_device_intelligence(
        {"shield_score": 0.82, "emulator": 1, "mock_location": 1, "same_device_as_courier": 1}
    )
    assert 80.0 <= device["device_risk_score"] <= 85.0
    assert device["is_emulator"] == 1.0
    assert device["is_gps_spoof"] == 1.0
    assert device["customer_courier_same_device"] == 1.0

    vision = adapt_claim_vision(
        {"has_image": 1, "manipulation_score": 88, "in_app_capture": 0, "pin_required": 1}
    )
    assert vision["claim_has_image"] == 1.0
    assert abs(vision["claim_image_ai_risk"] - 0.88) < 1e-6
    assert vision["claim_in_app_capture"] == 0.0


def test_nested_payload_merges_into_features() -> None:
    history = pd.DataFrame(
        [
            {
                "order_id": "h1",
                "user_id": "u1",
                "driver_id": "d1",
                "vendor_id": "v1",
                "amount": 20,
                "is_refund": 0,
                "event_ts": "2026-07-01T00:00:00Z",
                "claim_reason": "",
            }
        ]
    )
    devices = pd.DataFrame(
        [{"user_id": "u1", "device_id": "dev1", "cluster_id": "c1", "last_seen_ts": "2026-07-20T00:00:00Z"}]
    )
    order = {
        "order_id": "o1",
        "user_id": "u1",
        "driver_id": "d1",
        "vendor_id": "v1",
        "device_id": "dev1",
        "market": "SG",
        "vertical": "food",
        "amount": 25,
        "status": "delivered",
        "event_ts": "2026-07-20T00:00:00Z",
        "device_intelligence": {"risk_score": 75, "is_cloned_app": 1},
        "claim_vision": {"claim_image_ai_risk": 0.91, "claim_in_app_capture": 0, "pin_required": 1},
    }
    feat = build_order_feature_row(order, history, devices)
    assert feat["device_risk_score"] == 75.0
    assert feat["is_cloned_app"] == 1.0
    assert feat["claim_image_ai_risk"] == 0.91
    assert feat["pin_required"] == 1.0


def test_effect_rules_live_and_shadow_and_kill_switch() -> None:
    cfg = load_effect_rules()
    # PIN fail → live step-up
    live = resolve_refund_effect(
        SuggestedTier.AUTO_APPROVE,
        {"pin_required": 1, "pin_verified": 0},
        decision_score=40,
        effect_cfg=cfg,
    )
    assert live.final_effect == RefundEffect.REFUND_STEP_UP
    assert live.matched_mode == "live"
    assert "EFFECT_PIN_FAIL" in live.reason_codes

    # Same device → shadow block (base preserved)
    shadow = resolve_refund_effect(
        SuggestedTier.SOFT_FRICTION,
        {"customer_courier_same_device": 1},
        decision_score=50,
        effect_cfg=cfg,
    )
    assert shadow.final_effect == RefundEffect.REFUND_STEP_UP
    assert shadow.shadow_effect == RefundEffect.REFUND_BLOCK
    assert shadow.matched_mode == "shadow"

    killed = resolve_refund_effect(
        SuggestedTier.AUTO_APPROVE,
        {"pin_required": 1, "pin_verified": 0},
        decision_score=40,
        effect_cfg={**cfg, "kill_switch": True},
    )
    assert killed.final_effect == RefundEffect.REFUND_AUTO_GRANT
    assert killed.kill_switch is True


def test_refund_budget_pressure() -> None:
    cfg = load_refund_budget()
    ok = evaluate_refund_budget({"user_refund_count_30d": 1, "user_refund_gmv_pct_30d": 0.05}, cfg)
    assert ok.pressure == "ok"
    elev = evaluate_refund_budget({"user_refund_count_30d": 3, "user_refund_gmv_pct_30d": 0.1}, cfg)
    assert elev.pressure == "elevated"
    assert elev.suggested_effect == "refund_step_up"
    exh = evaluate_refund_budget({"user_refund_count_30d": 5, "user_refund_gmv_pct_30d": 0.4}, cfg)
    assert exh.pressure == "exhausted"
    assert exh.suggested_effect == "refund_manual_review"


def test_ece_and_psi_monitoring() -> None:
    y = [0, 0, 1, 1, 1, 0, 1, 0]
    s = [10, 20, 70, 80, 90, 30, 60, 15]
    ece = expected_calibration_error(y, s, n_bins=5)
    assert ece["n"] == 8
    assert ece["ece"] is not None
    assert 0.0 <= float(ece["ece"]) <= 1.0

    psi = population_stability_index([10, 20, 30, 40, 50, 60], [12, 22, 28, 55, 70, 80])
    assert psi["psi"] is not None
    assert float(psi["psi"]) >= 0.0


def test_discovery_mints_weak_labels_without_overwriting_proven() -> None:
    history = pd.DataFrame(
        [
            {
                "order_id": f"h{i}",
                "user_id": "u_bad",
                "vendor_id": "v_bad",
                "driver_id": "d1",
                "market": "SG",
                "vertical": "food",
                "amount": 30,
                "is_refund": 1 if i < 4 else 0,
                "event_ts": f"2026-06-{10 + i:02d}T00:00:00Z",
            }
            for i in range(6)
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "o_new",
                "user_id": "u_bad",
                "vendor_id": "v_bad",
                "market": "SG",
                "vertical": "food",
                "event_ts": "2026-06-20T00:00:00Z",
                "fraud_label": 0,
                "fraud_label_source": "",
                "abuse_label": 0,
            },
            {
                "order_id": "o_proven",
                "user_id": "u_bad",
                "vendor_id": "v_bad",
                "market": "SG",
                "vertical": "food",
                "event_ts": "2026-06-20T00:00:00Z",
                "fraud_label": 1,
                "fraud_label_source": "proven",
                "abuse_label": 1,
            },
        ]
    )
    out = mint_weak_labels_from_uv_anomaly(orders, history)
    row_new = out.loc[out["order_id"] == "o_new"].iloc[0]
    row_proven = out.loc[out["order_id"] == "o_proven"].iloc[0]
    assert int(out.attrs.get("discovery_minted", 0)) >= 1
    assert int(row_new["abuse_label_weak"]) == 1
    assert str(row_new["fraud_label_source"]) == "discovery"
    assert str(row_proven["fraud_label_source"]) == "proven"


def test_merge_prefers_nonzero_order_fields() -> None:
    merged = merge_platform_signals(
        {"device_risk_score": 40, "claim_has_image": 0},
        device_payload={"device_risk_score": 90},
        vision_payload={"has_image": 1},
    )
    assert merged["device_risk_score"] == 40.0
    assert merged["claim_has_image"] == 1.0
