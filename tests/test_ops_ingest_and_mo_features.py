from __future__ import annotations

from pathlib import Path

import pandas as pd

from refund_abuse_risk.features.builders import FEATURE_COLUMNS, build_order_feature_row
from refund_abuse_risk.graph.bipartite import bipartite_features_as_of, score_uv_bipartite
from refund_abuse_risk.integrations.ops_ingest import (
    load_ops_snapshot_file,
    merge_ops_snapshot,
    normalize_ops_snapshot,
)
from refund_abuse_risk.integrations.sdk_ingest import attach_sdk_signals
from refund_abuse_risk.scoring.monitoring import evaluate_monitoring_gates


def test_normalize_and_merge_ops_snapshot(tmp_path: Path) -> None:
    snap = normalize_ops_snapshot(
        {
            "as_of": "2026-08-03T00:00:00Z",
            "metrics": {"cs_queue_depth": 200, "refund_dollar_per_order": 9.5, "junk": "x"},
        }
    )
    assert snap["cs_queue_depth"] == 200.0
    assert "junk" not in snap
    path = tmp_path / "ops.json"
    path.write_text(
        '{"metrics": {"hold_rate": 0.22, "live_override_rate": 0.03}}',
        encoding="utf-8",
    )
    loaded = load_ops_snapshot_file(path)
    mon = merge_ops_snapshot({"max_hold_rate": 0.5, "ops_snapshot": {}}, loaded)
    assert mon["ops_snapshot"]["hold_rate"] == 0.22
    gate = evaluate_monitoring_gates(
        decision_ece=None,
        psi_train_test=None,
        monitoring_cfg={**mon, "max_cs_queue_depth": 100},
    )
    # No cs in loaded sidecar → pass; inject breach via merge.
    mon2 = merge_ops_snapshot(mon, {"cs_queue_depth": 250})
    bad = evaluate_monitoring_gates(
        decision_ece=None,
        psi_train_test=None,
        monitoring_cfg={**mon2, "max_cs_queue_depth": 100},
    )
    assert bad["ok"] is False


def test_example_ops_snapshot_file_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    snap = load_ops_snapshot_file(root / "examples" / "ops_snapshot.example.json")
    assert snap["cs_queue_depth"] == 85.0
    assert snap["refund_dollar_per_order"] == 4.2


def test_mo_tag_features_on_elevated_edge() -> None:
    rows = []
    for i in range(12):
        rows.append(
            {
                "order_id": f"H{i}",
                "user_id": "U1",
                "vendor_id": "V1",
                "market": "SG",
                "vertical": "food",
                "is_refund": 1 if i < 8 else 0,
                "event_ts": f"2026-07-0{(i % 9) + 1}T00:00:00Z",
            }
        )
    hist = pd.DataFrame(rows)
    edges, _ = score_uv_bipartite(
        hist,
        {
            "null_baseline": {"enabled": False},
            "min_edge_orders": 3,
            "min_edge_refunds": 2,
            "min_lift": 1.2,
            "min_refund_rate": 0.2,
        },
    )
    assert int(edges["elevated"].sum()) >= 1
    feat = bipartite_features_as_of(
        {
            "order_id": "O1",
            "user_id": "U1",
            "vendor_id": "V1",
            "market": "SG",
            "vertical": "food",
            "event_ts": "2026-07-15T00:00:00Z",
        },
        hist,
        {"null_baseline": {"enabled": False}, "min_edge_orders": 3, "min_edge_refunds": 2, "min_lift": 1.2, "min_refund_rate": 0.2},
    )
    assert feat["uv_edge_elevated"] == 1.0
    assert feat["uv_edge_n_orders"] >= 3.0
    assert feat["uv_edge_lift"] >= 1.0
    assert (
        feat["uv_mo_possible_collusion"]
        + feat["uv_mo_user_scatter"]
        + feat["uv_mo_elevated_uv"]
        >= 1.0
    )
    for col in (
        "uv_edge_lift",
        "uv_mo_possible_collusion",
        "uv_mo_user_scatter",
        "uv_mo_elevated_uv",
    ):
        assert col in FEATURE_COLUMNS


def test_claim_path_sdk_attach_then_features() -> None:
    order = {
        "order_id": "O1",
        "user_id": "U1",
        "driver_id": "D1",
        "vendor_id": "V1",
        "device_id": "DEV1",
        "market": "SG",
        "vertical": "food",
        "amount": 25,
        "status": "delivered",
        "event_ts": "2026-07-10T00:00:00Z",
    }
    enriched = attach_sdk_signals(
        order,
        device_event={
            "payload": {"risk_score": 90, "emulator": 1, "cloned_app": 1, "gps_spoof": 1},
            "confidence": 0.95,
        },
        vision_event={
            "payload": {"has_image": 1, "ai_risk": 0.8, "in_app_capture": 1},
            "confidence": 0.9,
        },
    )
    assert float(enriched["device_risk_score"]) >= 80
    assert float(enriched["claim_has_image"]) == 1.0
    hist = pd.DataFrame(
        [
            {
                "order_id": "H0",
                "user_id": "U1",
                "driver_id": "D1",
                "vendor_id": "V1",
                "device_id": "DEV1",
                "market": "SG",
                "vertical": "food",
                "amount": 20,
                "is_refund": 0,
                "event_ts": "2026-07-01T00:00:00Z",
            }
        ]
    )
    devices = pd.DataFrame(
        [{"device_id": "DEV1", "user_id": "U1", "cluster_id": "C1"}]
    )
    feat = build_order_feature_row(enriched, hist, devices)
    assert float(feat["device_risk_score"]) >= 80
    assert float(feat["is_emulator"]) == 1.0
    assert float(feat["claim_image_ai_risk"]) > 0
