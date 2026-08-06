"""Production readiness A+C: pack schema, dispositions gate, freshness, overlays."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from refund_abuse_risk.oot.validate import validate_oot_pack
from refund_abuse_risk.scoring.decision import resolve_decision_thresholds
from refund_abuse_risk.scoring.monitoring import evaluate_monitoring_gates
from refund_abuse_risk.training.closed_loop import (
    assert_dispositions_present,
    load_training_orders,
)

ROOT = Path(__file__).resolve().parents[1]


def test_prod_floors_file_stricter_than_default() -> None:
    demo = yaml.safe_load(
        (ROOT / "config" / "oot_floors.default.yaml").read_text(encoding="utf-8")
    )
    prod = yaml.safe_load(
        (ROOT / "config" / "oot_floors.prod.yaml").read_text(encoding="utf-8")
    )
    assert int(prod["min_holdout_n"]) > int(demo["min_holdout_n"])
    assert int(prod["min_proven_positives"]) > int(demo["min_proven_positives"])
    assert float(prod["max_decision_ece"]) <= float(demo["max_decision_ece"])


def test_validate_pack_missing_dispositions_fails(tmp_path: Path) -> None:
    orders = pd.DataFrame(
        [
            {
                "order_id": f"o{i}",
                "event_ts": f"2026-07-0{(i % 9) + 1}T00:00:00Z",
            }
            for i in range(120)
        ]
    )
    orders.to_csv(tmp_path / "orders.csv", index=False)
    (tmp_path / "manifest.yaml").write_text(
        "profile: prod_shaped\nrequire_dispositions: true\n"
        "floors:\n  min_holdout_n: 100\n  min_proven_positives: 15\n",
        encoding="utf-8",
    )
    result = validate_oot_pack(tmp_path)
    assert result["ok"] is False
    assert any("dispositions" in e for e in result["errors"])


def test_seed_and_validate_prod_shaped_pack() -> None:
    import subprocess
    import sys

    pack = ROOT / "data" / "oot_packs" / "prod_shaped_v1"
    proc = subprocess.run(
        [sys.executable, "scripts/seed_oot_pack.py", "--profile", "prod_shaped"],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    result = validate_oot_pack(
        pack, floors_path=ROOT / "config" / "oot_floors.prod.yaml"
    )
    assert result["ok"] is True, result
    assert result["n_proven"] >= 15
    assert result["profile"] == "prod_shaped"


def test_require_dispositions_fails_without_proven(tmp_path: Path) -> None:
    orders = pd.DataFrame(
        [
            {
                "order_id": "o1",
                "event_ts": "2026-07-01T00:00:00Z",
                "user_id": "u",
                "driver_id": "d",
                "vendor_id": "v",
                "amount": 10,
            }
        ]
    )
    orders.to_csv(tmp_path / "orders.csv", index=False)
    with pytest.raises(ValueError, match="require_dispositions"):
        load_training_orders(tmp_path, require_dispositions=True, min_train_proven=1)


def test_require_dispositions_passes_with_labeled_proven(tmp_path: Path) -> None:
    orders = pd.DataFrame(
        [
            {
                "order_id": "o1",
                "event_ts": "2026-07-01T00:00:00Z",
                "user_id": "u",
                "driver_id": "d",
                "vendor_id": "v",
                "amount": 10,
                "fraud_label_source": "proven",
                "fraud_label": 1,
            }
        ]
    )
    orders.to_csv(tmp_path / "orders.labeled.csv", index=False)
    frame, stats = load_training_orders(
        tmp_path, require_dispositions=True, min_train_proven=1
    )
    assert len(frame) == 1
    assert stats["proven_count"] == 1
    assert_dispositions_present(frame, stats, min_proven=1)


def test_ops_snapshot_freshness_gate() -> None:
    fresh = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace(
        "+00:00", "Z"
    )
    stale = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat().replace(
        "+00:00", "Z"
    )
    ok = evaluate_monitoring_gates(
        decision_ece=0.01,
        psi_train_test=0.01,
        monitoring_cfg={
            "max_ops_snapshot_age_hours": 24,
            "ops_snapshot": {"as_of": fresh, "hold_rate": 0.1},
        },
    )
    assert ok["ok"] is True
    bad = evaluate_monitoring_gates(
        decision_ece=0.01,
        psi_train_test=0.01,
        monitoring_cfg={
            "max_ops_snapshot_age_hours": 24,
            "ops_snapshot": {"as_of": stale, "hold_rate": 0.1},
        },
    )
    assert bad["ok"] is False
    assert any("ops_snapshot_age" in r for r in bad["reasons"])


def test_promote_overlays_and_rollback(tmp_path: Path) -> None:
    import importlib.util

    op = {
        "policy_version": "0.6.2",
        "decision_thresholds": {
            "soft_friction": 35,
            "hold_review": 50,
            "auto_deny": 75,
        },
        "decision_threshold_overlays": [],
    }
    op_path = tmp_path / "op.yaml"
    op_path.write_text(yaml.safe_dump(op), encoding="utf-8")
    overlays = tmp_path / "overlays.yaml"
    overlays.write_text(
        yaml.safe_dump(
            {
                "decision_threshold_overlays": [
                    {
                        "market": "SG",
                        "vertical": "food",
                        "soft_friction": 30,
                        "hold_review": 45,
                        "auto_deny": 70,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    backup_dir = tmp_path / "backups"

    spec = importlib.util.spec_from_file_location(
        "promote_overlays", ROOT / "scripts" / "promote_overlays.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    (tmp_path / "empty.yaml").write_text(
        "decision_threshold_overlays: []\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        mod.promote(
            overlays_path=tmp_path / "empty.yaml",
            op_path=op_path,
            backup_dir=backup_dir,
            dry_run=False,
            require_non_empty=True,
            backup_keep=5,
        )

    summary = mod.promote(
        overlays_path=overlays,
        op_path=op_path,
        backup_dir=backup_dir,
        dry_run=False,
        require_non_empty=True,
        backup_keep=5,
    )
    assert summary["n_overlays"] == 1
    live = yaml.safe_load(op_path.read_text(encoding="utf-8"))
    assert live["policy_version"] == "0.6.3"
    thr = resolve_decision_thresholds(live, market="SG", vertical="food")
    assert thr["soft_friction"] == 30
    thr_other = resolve_decision_thresholds(live, market="ID", vertical="food")
    assert thr_other["soft_friction"] == 35

    rolled = mod.rollback(op_path=op_path, backup_dir=backup_dir)
    assert "restored_from" in rolled
    restored = yaml.safe_load(op_path.read_text(encoding="utf-8"))
    assert restored["decision_threshold_overlays"] == []
