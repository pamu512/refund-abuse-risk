from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from refund_abuse_risk.config import load_sdk_ingest
from refund_abuse_risk.integrations.sdk_ingest import (
    apply_sdk_signals_to_orders,
    attach_sdk_signals,
    normalize_sdk_event,
)
from refund_abuse_risk.features.builders import build_order_feature_row


def test_normalize_rejects_low_confidence() -> None:
    cfg = load_sdk_ingest()
    ok = normalize_sdk_event(
        {
            "order_id": "o1",
            "source": "fingerprint",
            "confidence": 0.9,
            "payload": {"risk_score": 80, "is_emulator": 1},
        },
        cfg,
    )
    assert ok is not None
    assert ok["accepted"] is True
    assert ok["features"]["device_risk_score"] == 80.0

    bad = normalize_sdk_event(
        {
            "order_id": "o1",
            "source": "device",
            "confidence": 0.1,
            "payload": {"risk_score": 99},
        },
        cfg,
    )
    assert bad is not None
    assert bad["accepted"] is False


def test_apply_sdk_signals_to_orders_latest_wins() -> None:
    orders = pd.DataFrame(
        [{"order_id": "o1", "user_id": "u1", "market": "SG", "vertical": "food"}]
    )
    events = [
        {
            "order_id": "o1",
            "source": "shield",
            "confidence": 0.7,
            "event_ts": "2026-07-01T00:00:00Z",
            "payload": {"risk_score": 40},
        },
        {
            "order_id": "o1",
            "source": "device",
            "confidence": 0.95,
            "event_ts": "2026-07-02T00:00:00Z",
            "payload": {"risk_score": 88, "is_gps_spoof": 1},
        },
        {
            "order_id": "o1",
            "source": "vision",
            "confidence": 0.8,
            "event_ts": "2026-07-02T01:00:00Z",
            "payload": {"claim_image_ai_risk": 0.92, "pin_required": 1, "pin_verified": 0},
        },
    ]
    out = apply_sdk_signals_to_orders(orders, events, load_sdk_ingest())
    assert int(out.attrs["sdk_events_applied"]) == 2  # latest device + vision
    row = out.iloc[0]
    assert float(row["device_risk_score"]) == 88.0
    assert float(row["is_gps_spoof"]) == 1.0
    assert float(row["claim_image_ai_risk"]) == 0.92
    assert float(row["device_signal_confidence"]) == 0.95
    assert isinstance(row["device_intelligence"], dict)


def test_attach_sdk_signals_feeds_feature_builder() -> None:
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
    }
    enriched = attach_sdk_signals(
        order,
        device_event={"payload": {"shield_score": 0.77, "cloned_app": 1}, "confidence": 0.9},
        vision_event={"payload": {"manipulation_score": 85, "has_image": 1}, "confidence": 0.85},
        cfg=load_sdk_ingest(),
    )
    feat = build_order_feature_row(enriched, history, devices)
    assert feat["device_risk_score"] == 77.0
    assert feat["is_cloned_app"] == 1.0
    assert abs(feat["claim_image_ai_risk"] - 0.85) < 1e-6


def test_ingest_script_jsonl(tmp_path: Path) -> None:
    import runpy
    import sys

    root = Path(__file__).resolve().parents[1]
    orders = tmp_path / "orders.csv"
    events = tmp_path / "events.jsonl"
    out = tmp_path / "orders.sdk.csv"
    pd.DataFrame([{"order_id": "O1", "user_id": "u1"}]).to_csv(orders, index=False)
    events.write_text(
        json.dumps(
            {
                "order_id": "O1",
                "source": "incognia",
                "confidence": 0.9,
                "payload": {"risk_score": 71, "is_tampered": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    argv = sys.argv
    try:
        sys.argv = [
            "ingest_sdk_signals.py",
            "--orders",
            str(orders),
            "--events",
            str(events),
            "--out",
            str(out),
            "--json-summary",
        ]
        runpy.run_path(str(root / "scripts" / "ingest_sdk_signals.py"), run_name="__main__")
    finally:
        sys.argv = argv
    frame = pd.read_csv(out)
    assert float(frame.iloc[0]["device_risk_score"]) == 71.0
