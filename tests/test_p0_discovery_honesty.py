from __future__ import annotations

import pandas as pd

from refund_abuse_risk.labels.discovery_controls import (
    cap_discovery_labels,
    mint_uv_asof_daily,
)
from refund_abuse_risk.training.serve_features import stratified_order_sample


def test_asof_daily_ignores_future_history_refunds() -> None:
    history = pd.DataFrame(
        [
            {
                "order_id": "h1",
                "user_id": "u1",
                "vendor_id": "v1",
                "driver_id": "d1",
                "market": "SG",
                "vertical": "food",
                "amount": 20,
                "is_refund": 1,
                "event_ts": "2026-06-01T00:00:00Z",
            },
            {
                "order_id": "h2",
                "user_id": "u1",
                "vendor_id": "v1",
                "driver_id": "d1",
                "market": "SG",
                "vertical": "food",
                "amount": 20,
                "is_refund": 1,
                "event_ts": "2026-06-02T00:00:00Z",
            },
            {
                "order_id": "h3",
                "user_id": "u1",
                "vendor_id": "v1",
                "driver_id": "d1",
                "market": "SG",
                "vertical": "food",
                "amount": 20,
                "is_refund": 1,
                "event_ts": "2026-06-03T00:00:00Z",
            },
            # Future mass that must not mint early orders.
            *[
                {
                    "order_id": f"hf{i}",
                    "user_id": "u1",
                    "vendor_id": "v1",
                    "driver_id": "d1",
                    "market": "SG",
                    "vertical": "food",
                    "amount": 20,
                    "is_refund": 1,
                    "event_ts": f"2026-07-{10+i:02d}T00:00:00Z",
                }
                for i in range(6)
            ],
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "o_early",
                "user_id": "u1",
                "vendor_id": "v1",
                "market": "SG",
                "vertical": "food",
                "event_ts": "2026-06-02T12:00:00Z",
                "fraud_label": 0,
                "fraud_label_source": "",
                "abuse_label": 0,
            },
            {
                "order_id": "o_late",
                "user_id": "u1",
                "vendor_id": "v1",
                "market": "SG",
                "vertical": "food",
                "event_ts": "2026-07-20T00:00:00Z",
                "fraud_label": 0,
                "fraud_label_source": "",
                "abuse_label": 0,
            },
            {
                "order_id": "o_proven",
                "user_id": "u1",
                "vendor_id": "v1",
                "market": "SG",
                "vertical": "food",
                "event_ts": "2026-07-20T00:00:00Z",
                "fraud_label": 1,
                "fraud_label_source": "proven",
                "abuse_label": 1,
                "strong_fraud_label": 1,
            },
        ]
    )
    out, stats = mint_uv_asof_daily(orders, history)
    assert stats["mode"] == "asof_daily"
    early = out.loc[out["order_id"] == "o_early"].iloc[0]
    late = out.loc[out["order_id"] == "o_late"].iloc[0]
    proven = out.loc[out["order_id"] == "o_proven"].iloc[0]
    # Early day: only prior-day history before 2026-06-02 → not enough for elevated edge.
    assert str(early["fraud_label_source"]) != "discovery"
    assert str(proven["fraud_label_source"]) == "proven"
    # Late day sees June+July prior history → can mint.
    assert int(stats["uv_minted"]) >= 1
    assert str(late["fraud_label_source"]) == "discovery"


def test_stratified_sample_preserves_majority() -> None:
    rows = []
    for i in range(1000):
        pos = i < 250
        rows.append({"abuse_label": int(pos), "fraud_label": 0, "order_id": f"O{i}"})
    sample = stratified_order_sample(pd.DataFrame(rows), n=200, random_state=0)
    rate = float(sample["abuse_label"].astype(int).mean())
    assert len(sample) == 200
    assert 0.35 <= rate <= 0.55  # ~2x oversample of 25%, capped at 50%


def test_cap_discovery_limits_volume() -> None:
    rows = []
    for i in range(100):
        rows.append(
            {
                "fraud_label": 1 if i < 40 else 0,
                "fraud_label_source": "discovery" if i < 40 else ("proven" if i < 50 else ""),
                "abuse_label": 1 if i < 40 else 0,
                "abuse_label_weak": 1 if i < 40 else 0,
                "strong_fraud_label": 1 if 40 <= i < 50 else 0,
                "unsupervised_anomaly_score": float(100 - i),
            }
        )
    frame = pd.DataFrame(rows)
    out, stats = cap_discovery_labels(
        frame, max_fraction_of_rows=0.05, max_fraction_of_fraud_positives=0.35
    )
    n_disc = int((out["fraud_label_source"] == "discovery").sum())
    assert n_disc <= stats["cap"]
    assert stats["demoted"] >= 1
    # Proven untouched.
    assert int((out["fraud_label_source"] == "proven").sum()) == 10
