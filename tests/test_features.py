from __future__ import annotations

import pandas as pd

from refund_abuse_risk.features.builders import FEATURE_COLUMNS, build_order_feature_row


def test_build_order_feature_row_refund_stats() -> None:
    history = pd.DataFrame(
        [
            {
                "order_id": "h1",
                "user_id": "u1",
                "driver_id": "d1",
                "vendor_id": "v1",
                "amount": 50,
                "is_refund": 1,
                "event_ts": "2026-07-10T00:00:00Z",
            },
            {
                "order_id": "h2",
                "user_id": "u1",
                "driver_id": "d1",
                "vendor_id": "v1",
                "amount": 50,
                "is_refund": 1,
                "event_ts": "2026-07-12T00:00:00Z",
            },
            {
                "order_id": "h3",
                "user_id": "u1",
                "driver_id": "d1",
                "vendor_id": "v1",
                "amount": 50,
                "is_refund": 0,
                "event_ts": "2026-07-15T00:00:00Z",
            },
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
        "amount": 60,
        "status": "delivered",
        "claim_reason": "missing_item",
        "event_ts": "2026-07-20T00:00:00Z",
    }
    feat = build_order_feature_row(order, history, devices)
    assert set(FEATURE_COLUMNS) == set(feat)
    assert feat["user_refund_count_30d"] == 2.0
    assert abs(feat["user_refund_rate_30d"] - (2 / 3)) < 1e-9
    assert feat["claim_reason_missing_item"] == 1.0
    assert feat["uvd_cooccur"] == 3.0
