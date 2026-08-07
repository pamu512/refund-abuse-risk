"""Platform-depth contracts: overlays in serve, slice calibrator on path, PIT CI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from refund_abuse_risk.config import (
    load_operating_point,
    merge_decision_threshold_overlays,
)
from refund_abuse_risk.features.builders import build_order_feature_row
from refund_abuse_risk.model.two_head import TwoHeadModel
from refund_abuse_risk.pipeline.score import score_feature_row, train_two_head
from refund_abuse_risk.schemas.models import SuggestedTier
from refund_abuse_risk.scoring.decision import (
    resolve_decision_thresholds,
    tier_from_decision_score,
)
from refund_abuse_risk.scoring.slice_calibrator import SliceCalibrator

ROOT = Path(__file__).resolve().parents[1]


def test_default_op_ships_non_empty_overlays_for_serve() -> None:
    op = load_operating_point()
    overlays = op.get("decision_threshold_overlays") or []
    assert len(overlays) >= 1
    thr_sg = resolve_decision_thresholds(op, market="SG", vertical="food")
    thr_global = resolve_decision_thresholds(op, market="ZZ", vertical="other")
    assert thr_sg["soft_friction"] != thr_global["soft_friction"]
    assert tier_from_decision_score(33, op, market="SG", vertical="food") == (
        SuggestedTier.SOFT_FRICTION
    )
    assert tier_from_decision_score(33, op, market="ZZ", vertical="other") == (
        SuggestedTier.AUTO_APPROVE
    )


def test_overlay_sidecar_merges_when_inline_empty(tmp_path: Path) -> None:
    side = tmp_path / "overlays.yaml"
    side.write_text(
        "decision_threshold_overlays:\n"
        "  - market: MY\n"
        "    vertical: food\n"
        "    soft_friction: 12\n"
        "    hold_review: 22\n"
        "    auto_deny: 88\n",
        encoding="utf-8",
    )
    op = {
        "decision_thresholds": {
            "soft_friction": 35,
            "hold_review": 50,
            "auto_deny": 75,
        },
        "decision_threshold_overlays": [],
        "decision_threshold_overlays_file": str(side),
    }
    merged = merge_decision_threshold_overlays(op, config_dir=tmp_path)
    assert len(merged["decision_threshold_overlays"]) == 1
    thr = resolve_decision_thresholds(merged, market="MY", vertical="food")
    assert thr["soft_friction"] == 12


def test_score_feature_row_applies_slice_overlay_and_calibrator(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 80
    rows = []
    hist = []
    for i in range(n):
        market = "SG" if i % 2 == 0 else "ID"
        oid = f"O{i}"
        ts = f"2026-06-{(i % 28) + 1:02d}T12:00:00Z"
        is_bad = i % 5 == 0
        rows.append(
            {
                "order_id": oid,
                "user_id": f"U{i % 20}",
                "driver_id": f"D{i % 15}",
                "vendor_id": f"V{i % 12}",
                "device_id": f"DEV{i % 10}",
                "market": market,
                "vertical": "food",
                "amount": float(20 + (i % 30)),
                "status": "delivered",
                "claim_reason": "missing_item" if is_bad else "",
                "event_ts": ts,
                "abuse_label": int(is_bad),
                "fraud_label": int(is_bad and i % 10 == 0),
                "fraud_label_source": "proven" if is_bad and i % 10 == 0 else "",
                "is_refund": int(is_bad),
            }
        )
        hist.append(
            {
                "order_id": f"H{i}",
                "user_id": f"U{i % 20}",
                "driver_id": f"D{i % 15}",
                "vendor_id": f"V{i % 12}",
                "device_id": f"DEV{i % 10}",
                "market": market,
                "vertical": "food",
                "amount": 15.0,
                "is_refund": int(rng.random() < 0.2),
                "status": "delivered",
                "claim_reason": "",
                "event_ts": f"2026-05-{(i % 28) + 1:02d}T00:00:00Z",
            }
        )
    orders = pd.DataFrame(rows)
    history = pd.DataFrame(hist)
    devices = pd.DataFrame(
        [
            {
                "device_id": f"DEV{i}",
                "user_id": f"U{i}",
                "cluster_id": f"C{i % 5}",
                "accounts_per_device": 1,
            }
            for i in range(20)
        ]
    )
    users = pd.DataFrame(
        [{"user_id": f"U{i}", "signup_ts": "2026-01-01T00:00:00Z"} for i in range(20)]
    )
    model = train_two_head(orders, history, devices, users=users)
    assert model.slice_calibrator is not None

    feat = build_order_feature_row(rows[0], history, devices, users=users)
    # Serve path needs slice keys; feature builder returns numeric columns only.
    serve_row = {
        **feat,
        "order_id": rows[0]["order_id"],
        "market": rows[0]["market"],
        "vertical": rows[0]["vertical"],
        "amount": rows[0]["amount"],
        "event_ts": rows[0]["event_ts"],
    }
    scored = model.predict_proba(pd.DataFrame([serve_row])).iloc[0]
    assert "decision_score_raw" in scored.index
    assert bool(scored["slice_calibrator_applied"]) is True

    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = TwoHeadModel.load(path)
    assert loaded.slice_calibrator is not None
    a = model.predict_proba(pd.DataFrame([serve_row]))["decision_score"].iloc[0]
    b = loaded.predict_proba(pd.DataFrame([serve_row]))["decision_score"].iloc[0]
    assert float(a) == pytest.approx(float(b))

    op = load_operating_point()
    snap = score_feature_row(serve_row, model, operating_point=op, update_baselines=False)
    assert snap.market == "SG"
    assert "slice_calibrator_applied" in snap.reason_codes
    # Overlay differs from global → reason code on serve path.
    assert any(c.startswith("slice_overlay:") for c in snap.reason_codes)


def test_pit_replay_ci_contract_ignores_future_history_and_bipartite() -> None:
    """CI gate: full history == as-of-truncated history for PIT-sensitive columns."""
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
    past = history.loc[
        pd.to_datetime(history["event_ts"], utc=True)
        <= pd.Timestamp("2026-07-10T12:00:00Z", tz="UTC")
    ]
    full = build_order_feature_row(order, history, devices)
    replay = build_order_feature_row(order, past, devices)
    keys = [
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
    ]
    for k in keys:
        assert k in full and k in replay
        assert abs(float(full[k]) - float(replay[k])) < 1e-9, k
    assert float(full["user_lifetime_refund_count"]) >= 2.0


def test_slice_calibrator_identity_when_disabled() -> None:
    scores = np.array([10, 20, 80, 90], dtype=float)
    y = np.array([0, 0, 1, 1])
    markets = np.array(["SG"] * 4)
    verticals = np.array(["food"] * 4)
    off = SliceCalibrator(enabled=False).fit(scores, y, markets, verticals)
    assert np.allclose(off.transform(scores, markets, verticals), scores)
