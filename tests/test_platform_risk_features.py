from __future__ import annotations

import pandas as pd

from refund_abuse_risk.config import load_label_weights
from refund_abuse_risk.features.builders import build_order_feature_row
from refund_abuse_risk.model.two_head import apply_proxy_fraud_labels


def test_device_integrity_and_claim_proof_features() -> None:
    history = pd.DataFrame(
        [
            {
                "order_id": "h1",
                "user_id": "u1",
                "driver_id": "d1",
                "vendor_id": "v1",
                "amount": 40,
                "is_refund": 1,
                "event_ts": "2026-07-10T00:00:00Z",
                "claim_reason": "missing_item",
            }
        ]
    )
    devices = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "device_id": "dev1",
                "cluster_id": "c1",
                "last_seen_ts": "2026-07-20T00:00:00Z",
                "device_risk_score": 90,
                "is_emulator": 0,
                "is_cloned_app": 1,
                "is_gps_spoof": 1,
                "is_tampered": 1,
            }
        ]
    )
    order = {
        "order_id": "o1",
        "user_id": "u1",
        "driver_id": "d1",
        "vendor_id": "v1",
        "device_id": "dev1",
        "market": "SG",
        "vertical": "food",
        "amount": 40,
        "status": "delivered",
        "claim_reason": "missing_item",
        "event_ts": "2026-07-20T00:00:00Z",
        "customer_courier_same_device": 1,
        "claim_has_image": 1,
        "claim_image_ai_risk": 0.9,
        "claim_in_app_capture": 0,
        "pin_required": 1,
        "pin_verified": 0,
        "delivery_geofence_ok": 0,
    }
    feat = build_order_feature_row(order, history, devices)
    assert feat["device_risk_score"] == 90.0
    assert feat["is_cloned_app"] == 1.0
    assert feat["is_gps_spoof"] == 1.0
    assert feat["customer_courier_same_device"] == 1.0
    assert feat["claim_image_ai_risk"] == 0.9
    assert feat["pin_required"] == 1.0
    assert feat["pin_verified"] == 0.0
    assert feat["delivery_geofence_ok"] == 0.0


def test_integrity_proxy_path_mints_fraud_label() -> None:
    frame = pd.DataFrame(
        [
            {
                "device_cluster_size": 1,
                "accounts_per_device": 1,
                "uvd_refund_lift": 1.0,
                "uvd_refund_share": 0.1,
                "uvd_cooccur": 5,
                "user_refund_rate_30d": 0.5,
                "user_orders_30d": 8,
                "device_risk_score": 85,
                "is_emulator": 0,
                "is_cloned_app": 1,
                "is_gps_spoof": 1,
                "customer_courier_same_device": 1,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
            }
        ]
    )
    out = apply_proxy_fraud_labels(frame, load_label_weights())
    assert int(out.iloc[0]["fraud_label"]) == 1
    assert str(out.iloc[0]["fraud_label_source"]) == "proxy"
