from __future__ import annotations

import pandas as pd

from refund_abuse_risk.features.builders import build_order_feature_row
from refund_abuse_risk.scoring.monitoring import (
    evaluate_monitoring_gates,
    summarize_ops_metrics,
)


def test_ops_gate_blocks_high_hold_and_override_rates() -> None:
    ops = summarize_ops_metrics(
        tier_counts={"hold_review": 40, "auto_approve": 60},
        effects_dual_run={
            "n": 100,
            "live_override_count": 50,
            "shadow_override_count": 0,
            "final_effect_counts": {"refund_auto_grant": 20},
        },
        amounts=[10.0] * 100,
        final_effects=["refund_auto_grant"] * 20 + ["refund_block"] * 80,
    )
    assert abs(ops["hold_rate"] - 0.4) < 1e-9
    assert abs(ops["live_override_rate"] - 0.5) < 1e-9
    assert abs(ops["refund_dollar_per_order"] - 2.0) < 1e-9

    ok = evaluate_monitoring_gates(
        decision_ece=0.05,
        psi_train_test=0.1,
        monitoring_cfg={
            "max_decision_ece": 0.2,
            "max_psi_train_test": 0.35,
            "max_hold_rate": 0.55,
            "max_live_override_rate": 0.35,
        },
        ops_metrics=ops,
    )
    assert ok["ok"] is False
    assert any("live_override_rate" in r for r in ok["reasons"])


def test_ops_snapshot_from_config_gates_cs_queue() -> None:
    gate = evaluate_monitoring_gates(
        decision_ece=0.01,
        psi_train_test=0.01,
        monitoring_cfg={
            "max_cs_queue_depth": 100,
            "ops_snapshot": {"cs_queue_depth": 250},
        },
        ops_metrics={"hold_rate": 0.1},
    )
    assert gate["ok"] is False
    assert any("cs_queue_depth" in r for r in gate["reasons"])


def test_pit_feature_replay_ignores_future_history() -> None:
    """G4: top features from full history must match as-of-truncated history."""
    order = {
        "order_id": "O_now",
        "user_id": "U1",
        "driver_id": "D1",
        "vendor_id": "V1",
        "device_id": "DEV1",
        "market": "SG",
        "vertical": "food",
        "amount": 40.0,
        "status": "delivered",
        "claim_reason": "missing_item",
        "event_ts": "2026-07-10T12:00:00Z",
    }
    history = pd.DataFrame(
        [
            {
                "order_id": "H1",
                "user_id": "U1",
                "driver_id": "D1",
                "vendor_id": "V1",
                "device_id": "DEV1",
                "market": "SG",
                "vertical": "food",
                "amount": 30.0,
                "is_refund": 1,
                "status": "delivered",
                "claim_reason": "missing_item",
                "event_ts": "2026-07-01T00:00:00Z",
            },
            {
                "order_id": "H2",
                "user_id": "U1",
                "driver_id": "D1",
                "vendor_id": "V1",
                "device_id": "DEV1",
                "market": "SG",
                "vertical": "food",
                "amount": 30.0,
                "is_refund": 1,
                "status": "delivered",
                "claim_reason": "quality",
                "event_ts": "2026-07-05T00:00:00Z",
            },
            # Future — must not affect PIT features.
            {
                "order_id": "H_future",
                "user_id": "U1",
                "driver_id": "D1",
                "vendor_id": "V1",
                "device_id": "DEV1",
                "market": "SG",
                "vertical": "food",
                "amount": 99.0,
                "is_refund": 1,
                "status": "delivered",
                "claim_reason": "missing_item",
                "event_ts": "2026-07-20T00:00:00Z",
            },
        ]
    )
    devices = pd.DataFrame(
        [{"device_id": "DEV1", "user_id": "U1", "cluster_id": "C1", "accounts_per_device": 1}]
    )
    past = history.loc[
        pd.to_datetime(history["event_ts"], utc=True)
        <= pd.Timestamp("2026-07-10T12:00:00Z", tz="UTC")
    ]
    full = build_order_feature_row(order, history, devices)
    replay = build_order_feature_row(order, past, devices)
    # Top train/serve-sensitive columns — must be bit-identical under PIT replay.
    keys = [
        "user_orders_30d",
        "user_refund_count_30d",
        "user_refund_rate_30d",
        "user_lifetime_orders",
        "user_lifetime_refund_count",
        "uvd_cooccur",
        "uv_edge_anomaly",
    ]
    for k in keys:
        assert abs(float(full[k]) - float(replay[k])) < 1e-9, k
    # Sanity: past refunds are visible.
    assert float(full["user_lifetime_refund_count"]) >= 2.0
