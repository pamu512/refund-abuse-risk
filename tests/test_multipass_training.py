from __future__ import annotations

import numpy as np
import pandas as pd

from refund_abuse_risk.training.multipass import (
    mint_isolation_forest_discovery,
    train_multipass,
)


def _toy_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        bad = i < 20
        rows.append(
            {
                "order_id": f"O{i}",
                "user_id": f"U{i%40}",
                "vendor_id": f"V{i%10}",
                "driver_id": f"D{i%8}",
                "market": "SG",
                "vertical": "food",
                "abuse_label": int(bad and i < 12),
                "abuse_label_weak": 0,
                "fraud_label": int(bad and i < 8),
                "fraud_label_source": "proven" if bad and i < 8 else "",
                "strong_fraud_label": int(bad and i < 8),
                "weak_policy_negative": 0,
                "device_cluster_size": 10 if bad else 1,
                "accounts_per_device": 5 if bad else 1,
                "uvd_refund_lift": 3.0 if bad else 1.0,
                "uvd_refund_share": 0.8 if bad else 0.1,
                "uvd_cooccur": 8 if bad else 2,
                "device_risk_score": 90 if bad else 5,
                "user_refund_rate_30d": 0.6 if bad else 0.05,
                "uv_edge_anomaly": 2.0 if bad else 0.0,
                "user_bipartite_anomaly": 1.0 if bad else 0.0,
                "is_cloned_app": 1 if bad else 0,
                "is_gps_spoof": 1 if bad else 0,
                "customer_courier_same_device": 1 if bad else 0,
                "user_orders_30d": 10,
                "user_refund_count_30d": 6 if bad else 0,
                "order_amount": 40,
                "is_food": 1,
                "is_qcommerce": 0,
                "order_status_delivered": 1,
                "event_ts": f"2026-07-{(i % 28) + 1:02d}T00:00:00Z",
            }
        )
    frame = pd.DataFrame(rows)
    # Fill remaining model columns with zeros if missing later in fit via apply_proxy.
    return frame


def test_isolation_forest_mints_discovery_without_touching_proven() -> None:
    frame = _toy_frame()
    out, stats = mint_isolation_forest_discovery(frame, contamination=0.15, score_percentile=85)
    assert stats["iforest_minted"] >= 1
    proven = out[out["fraud_label_source"] == "proven"]
    assert (proven["fraud_label_source"] == "proven").all()


def test_multipass_runs_supervised_and_unsupervised() -> None:
    from refund_abuse_risk.features.builders import FEATURE_COLUMNS

    frame = _toy_frame(80)
    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0
    model, reports, _ = train_multipass(frame, history=None, n_passes=3, random_state=1)
    assert model.decision_stacker is not None
    assert len(reports) == 3
    assert "isolation_forest" in reports[0].unsupervised
    assert reports[-1].supervised["n"] == 80


def test_multipass_early_stops_on_flat_decision_mean() -> None:
    from refund_abuse_risk.features.builders import FEATURE_COLUMNS

    frame = _toy_frame(80)
    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0
    _model, reports, _ = train_multipass(
        frame,
        history=None,
        n_passes=8,
        random_state=1,
        early_stop_delta=1e9,  # any tiny move counts as flat
        early_stop_patience=2,
    )
    assert len(reports) < 8
    assert reports[-1].supervised.get("early_stopped") is True
