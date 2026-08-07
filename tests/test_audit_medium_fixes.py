"""Regression tests for audit Medium findings (M1/M2/M3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from refund_abuse_risk.config import load_operating_point
from refund_abuse_risk.oot.validate import validate_oot_pack
from refund_abuse_risk.scoring.decision import resolve_decision_thresholds
from refund_abuse_risk.scoring.monitoring import evaluate_monitoring_gates

ROOT = Path(__file__).resolve().parents[1]


def test_m1_pack_schema_uses_min_pack_n_not_holdout(tmp_path: Path) -> None:
    floors = yaml.safe_load(
        (ROOT / "config" / "oot_floors.prod.yaml").read_text(encoding="utf-8")
    )
    assert int(floors["min_pack_n"]) >= 100
    assert "min_holdout_n" in floors

    pd.DataFrame(
        [
            {
                "order_id": f"o{i}",
                "event_ts": f"2026-07-0{(i % 9) + 1}T00:00:00Z",
            }
            for i in range(50)
        ]
    ).to_csv(tmp_path / "orders.csv", index=False)
    (tmp_path / "manifest.yaml").write_text(
        "profile: prod_shaped\n"
        "floors:\n  min_pack_n: 100\n  min_holdout_n: 100\n  min_proven_positives: 0\n",
        encoding="utf-8",
    )
    result = validate_oot_pack(tmp_path)
    assert result["ok"] is False
    assert any("min_pack_n" in e for e in result["errors"])
    assert not any("min_holdout_n=" in e for e in result["errors"])


def test_m1_holdout_without_pack_n_is_schema_error(tmp_path: Path) -> None:
    pd.DataFrame(
        [{"order_id": "o1", "event_ts": "2026-07-01T00:00:00Z"}]
    ).to_csv(tmp_path / "orders.csv", index=False)
    (tmp_path / "manifest.yaml").write_text(
        "profile: demo\nfloors:\n  min_holdout_n: 30\n",
        encoding="utf-8",
    )
    result = validate_oot_pack(tmp_path)
    assert result["ok"] is False
    assert any("min_pack_n" in e for e in result["errors"])


def test_m2_overlay_prefers_more_specific_match() -> None:
    op = {
        "decision_thresholds": {
            "soft_friction": 35,
            "hold_review": 50,
            "auto_deny": 75,
        },
        "decision_threshold_overlays": [
            # Broader market-only listed first — must lose to market×vertical.
            {"market": "SG", "soft_friction": 25, "hold_review": 40, "auto_deny": 70},
            {
                "market": "SG",
                "vertical": "food",
                "soft_friction": 20,
                "hold_review": 38,
                "auto_deny": 60,
            },
        ],
    }
    thr = resolve_decision_thresholds(op, market="SG", vertical="food")
    assert thr["soft_friction"] == 20
    assert thr["auto_deny"] == 60
    thr_mkt = resolve_decision_thresholds(op, market="SG", vertical="qcommerce")
    assert thr_mkt["soft_friction"] == 25


def test_m3_default_op_enforces_ops_snapshot_age() -> None:
    op = load_operating_point()
    mon = op.get("monitoring") or {}
    assert mon.get("max_ops_snapshot_age_hours") == 24
    assert not mon.get("ops_snapshot"), "default OP must not ship a stale baked ops_snapshot"

    fresh = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    stale = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    base_mon = {k: v for k, v in mon.items() if k != "ops_snapshot"}
    good = evaluate_monitoring_gates(
        decision_ece=0.01,
        psi_train_test=0.01,
        decision_brier=0.05,
        ece_usable=True,
        monitoring_cfg={
            **base_mon,
            "ops_snapshot": {
                "as_of": fresh,
                "hold_rate": 0.1,
                "live_override_rate": 0.01,
                "refund_grant_rate": 0.5,
                "refund_dollar_per_order": 1.0,
                "cs_queue_depth": 10,
                "shadow_override_rate": 0.01,
            },
        },
    )
    assert good["ok"] is True, good["reasons"]

    bad = evaluate_monitoring_gates(
        decision_ece=0.01,
        psi_train_test=0.01,
        decision_brier=0.05,
        ece_usable=True,
        monitoring_cfg={
            **base_mon,
            "ops_snapshot": {
                "as_of": stale,
                "hold_rate": 0.1,
                "live_override_rate": 0.01,
                "refund_grant_rate": 0.5,
                "refund_dollar_per_order": 1.0,
                "cs_queue_depth": 10,
                "shadow_override_rate": 0.01,
            },
        },
    )
    assert bad["ok"] is False
    assert any("ops_snapshot_age" in r for r in bad["reasons"])
