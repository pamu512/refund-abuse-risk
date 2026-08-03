from __future__ import annotations

import pandas as pd

from refund_abuse_risk.config import load_label_weights
from refund_abuse_risk.features.builders import (
    PROXY_MINT_FEATURE_COLUMNS,
    build_order_feature_row,
    fraud_model_feature_columns,
)
from refund_abuse_risk.graph.bipartite import bipartite_features_as_of
from refund_abuse_risk.model.two_head import TwoHeadModel, apply_proxy_fraud_labels
from refund_abuse_risk.scoring.thresholds import recommend_head_thresholds


def test_bipartite_as_of_ignores_future_refunds() -> None:
    history = pd.DataFrame(
        [
            {
                "order_id": "h1",
                "user_id": "U",
                "vendor_id": "V",
                "market": "SG",
                "vertical": "food",
                "is_refund": 0,
                "event_ts": "2026-07-01T00:00:00Z",
            },
            {
                "order_id": "h2",
                "user_id": "U",
                "vendor_id": "V",
                "market": "SG",
                "vertical": "food",
                "is_refund": 0,
                "event_ts": "2026-07-02T00:00:00Z",
            },
            # Future concentrated refunds — must not leak into earlier score.
            {
                "order_id": "h3",
                "user_id": "U",
                "vendor_id": "V",
                "market": "SG",
                "vertical": "food",
                "is_refund": 1,
                "event_ts": "2026-07-20T00:00:00Z",
            },
            {
                "order_id": "h4",
                "user_id": "U",
                "vendor_id": "V",
                "market": "SG",
                "vertical": "food",
                "is_refund": 1,
                "event_ts": "2026-07-21T00:00:00Z",
            },
            {
                "order_id": "h5",
                "user_id": "U",
                "vendor_id": "V",
                "market": "SG",
                "vertical": "food",
                "is_refund": 1,
                "event_ts": "2026-07-22T00:00:00Z",
            },
        ]
    )
    early = {
        "order_id": "o-early",
        "user_id": "U",
        "vendor_id": "V",
        "market": "SG",
        "vertical": "food",
        "event_ts": "2026-07-03T00:00:00Z",
    }
    feat = bipartite_features_as_of(early, history)
    assert feat["uv_edge_anomaly"] == 0.0


def test_proxy_mint_features_held_out_of_fraud_head() -> None:
    cols = fraud_model_feature_columns(exclude_proxy_mint=True)
    for c in PROXY_MINT_FEATURE_COLUMNS:
        assert c not in cols
    wide = fraud_model_feature_columns(exclude_proxy_mint=False)
    for c in PROXY_MINT_FEATURE_COLUMNS:
        assert c in wide


def test_fit_excludes_mint_features_by_default() -> None:
    # Minimal frame with both classes.
    n = 20
    rows = []
    for i in range(n):
        rows.append(
            {
                "abuse_label": 1 if i < 8 else 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
                "abuse_label_weak": 0,
                **{c: 0.0 for c in __import__(
                    "refund_abuse_risk.features.builders", fromlist=["FEATURE_COLUMNS"]
                ).FEATURE_COLUMNS},
                "device_cluster_size": 8 if i < 5 else 1,
                "accounts_per_device": 5 if i < 5 else 1,
                "uvd_refund_lift": 3.0 if i < 5 else 1.0,
                "uvd_refund_share": 0.8 if i < 5 else 0.0,
                "uvd_cooccur": 6 if i < 5 else 0,
                "user_refund_rate_30d": 0.5 if i < 5 else 0.05,
                "user_orders_30d": 10,
            }
        )
    frame = pd.DataFrame(rows)
    model = TwoHeadModel().fit(frame, load_label_weights())
    for c in PROXY_MINT_FEATURE_COLUMNS:
        assert c not in model.fraud_feature_columns


def test_costed_thresholds_report_ok_and_floors() -> None:
    # Separable scores: positives high, negatives low → cost feasible.
    y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    s = [90, 85, 80, 75, 70, 20, 15, 10, 5, 1]
    rec = recommend_head_thresholds(y, s, y, s, target_recall=0.8, min_precision_at_soft=0.5)
    assert rec["ok"] is True
    assert rec["cost_feasible"] is True
    assert rec["floor_would_bind"] == []

    # Soft floor would bind with tiny scores → not ok.
    y2 = [1, 1, 0, 0]
    s2 = [5, 4, 3, 2]
    rec2 = recommend_head_thresholds(
        y2, s2, y2, s2, target_recall=1.0, min_precision_at_soft=None, apply_floors=False
    )
    assert rec2["soft_floor_would_bind"] or not rec2["cost_feasible"] or rec2["floor_would_bind"]
    assert rec2["ok"] is False


def test_baseline_cols_not_required_for_row_build() -> None:
    history = pd.DataFrame(
        [
            {
                "order_id": "h1",
                "user_id": "u1",
                "driver_id": "d1",
                "vendor_id": "v1",
                "amount": 10,
                "is_refund": 0,
                "event_ts": "2026-07-01T00:00:00Z",
                "claim_reason": "",
            }
        ]
    )
    devices = pd.DataFrame(
        [{"user_id": "u1", "device_id": "dev1", "cluster_id": "c1", "last_seen_ts": "2026-07-01T00:00:00Z"}]
    )
    order = {
        "order_id": "o1",
        "user_id": "u1",
        "driver_id": "d1",
        "vendor_id": "v1",
        "device_id": "dev1",
        "market": "SG",
        "vertical": "food",
        "amount": 10,
        "status": "delivered",
        "event_ts": "2026-07-02T00:00:00Z",
        "claim_reason": "",
    }
    feat = build_order_feature_row(order, history, devices)
    assert feat["user_baseline_lift"] == 1.0
    labeled = apply_proxy_fraud_labels(pd.DataFrame([feat]), load_label_weights())
    assert "fraud_label" in labeled.columns
