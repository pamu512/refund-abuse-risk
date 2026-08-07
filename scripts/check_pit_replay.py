#!/usr/bin/env python3
"""CI / overnight gate: PIT feature replay must ignore future history."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from refund_abuse_risk.features.builders import build_order_feature_row

PIT_KEYS = (
    "user_orders_30d",
    "user_refund_count_30d",
    "user_refund_rate_30d",
    "user_lifetime_orders",
    "user_lifetime_refund_count",
    "uvd_cooccur",
    "uv_edge_anomaly",
    "uv_edge_lift",
    "uv_edge_n_orders",
    "uv_edge_elevated",
    "driver_refund_rate_30d",
    "vendor_refund_rate_30d",
)


def main() -> int:
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
    as_of = pd.Timestamp("2026-07-10T12:00:00Z", tz="UTC")
    past = history.loc[pd.to_datetime(history["event_ts"], utc=True) <= as_of]
    full = build_order_feature_row(order, history, devices)
    replay = build_order_feature_row(order, past, devices)
    failures: list[str] = []
    for k in PIT_KEYS:
        if k not in full or k not in replay:
            failures.append(f"missing key {k}")
            continue
        if abs(float(full[k]) - float(replay[k])) > 1e-9:
            failures.append(f"{k}: full={full[k]} replay={replay[k]}")
    if failures:
        print("PIT replay FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PIT replay OK ({len(PIT_KEYS)} keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
