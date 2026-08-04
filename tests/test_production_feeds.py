from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from refund_abuse_risk.integrations.feeds import run_feeds


def _write_fixture_tree(root: Path) -> dict:
    fixtures = root / "data" / "feeds" / "fixtures"
    (fixtures / "dispositions").mkdir(parents=True)
    (fixtures / "sdk_events").mkdir(parents=True)
    (fixtures / "ops_snapshot").mkdir(parents=True)

    orders = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "user_id": "U1",
                "abuse_label": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "weak_policy_negative": 0,
                "event_ts": "2026-07-01T00:00:00Z",
                "market": "SG",
                "vertical": "food",
                "amount": 20,
            },
            {
                "order_id": "O2",
                "user_id": "U2",
                "abuse_label": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "weak_policy_negative": 0,
                "event_ts": "2026-07-02T00:00:00Z",
                "market": "SG",
                "vertical": "food",
                "amount": 30,
            },
        ]
    )
    orders.to_csv(root / "data" / "orders.csv", index=False)
    pd.DataFrame(
        [
            {
                "order_id": "O2",
                "disposition": "chargeback_lost",
                "disposition_ts": "2026-07-10T00:00:00Z",
            }
        ]
    ).to_csv(fixtures / "dispositions" / "dispositions.csv", index=False)
    (fixtures / "sdk_events" / "sdk_events.jsonl").write_text(
        json.dumps(
            {
                "order_id": "O1",
                "source": "device",
                "confidence": 0.9,
                "payload": {"risk_score": 80, "emulator": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (fixtures / "ops_snapshot" / "ops_snapshot.json").write_text(
        json.dumps(
            {
                "as_of": "2026-08-04T00:00:00Z",
                "source": "test",
                "metrics": {"cs_queue_depth": 10, "hold_rate": 0.1},
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "stage_root": "data/feeds/staging",
        "orders_path": "data/orders.csv",
        "orders_labeled_out": "data/orders.labeled.csv",
        "ops_sidecar": "data/ops_snapshot.json",
        "feeds": {
            "dispositions": {
                "enabled": True,
                "source": {
                    "type": "local_dir",
                    "uri": "data/feeds/fixtures/dispositions",
                    "glob": "*.csv",
                },
                "required_columns": ["order_id", "disposition", "disposition_ts"],
                "apply": {"kind": "dispositions"},
            },
            "sdk_events": {
                "enabled": True,
                "source": {
                    "type": "local_dir",
                    "uri": "data/feeds/fixtures/sdk_events",
                    "glob": "*.jsonl",
                },
                "required_columns": ["order_id"],
                "apply": {"kind": "sdk_events"},
            },
            "ops_snapshot": {
                "enabled": True,
                "source": {
                    "type": "local_dir",
                    "uri": "data/feeds/fixtures/ops_snapshot",
                    "glob": "*.json",
                },
                "required_columns": [],
                "apply": {"kind": "ops_snapshot"},
            },
        },
    }
    return cfg


def test_run_feeds_dry_run_no_writes(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    cfg = _write_fixture_tree(tmp_path)
    report = run_feeds(cfg, dry_run=True, root=tmp_path)
    assert report["ok"] is True
    assert not (tmp_path / "data" / "orders.labeled.csv").exists()
    assert not (tmp_path / "data" / "ops_snapshot.json").exists()
    assert report["feeds"]["dispositions"]["apply"]["dry_run"] is True


def test_run_feeds_apply_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    cfg = _write_fixture_tree(tmp_path)
    report = run_feeds(cfg, root=tmp_path)
    assert report["ok"] is True
    labeled = pd.read_csv(tmp_path / "data" / "orders.labeled.csv")
    assert int(labeled.loc[labeled["order_id"] == "O2", "fraud_label"].iloc[0]) == 1
    ops = json.loads((tmp_path / "data" / "ops_snapshot.json").read_text(encoding="utf-8"))
    assert ops["cs_queue_depth"] == 10.0


def test_run_feeds_stage_only(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    cfg = _write_fixture_tree(tmp_path)
    report = run_feeds(cfg, stage_only=True, root=tmp_path)
    assert report["ok"] is True
    assert "apply" not in report["feeds"]["ops_snapshot"]
    staged = list((tmp_path / "data" / "feeds" / "staging").rglob("*"))
    assert any(p.is_file() for p in staged)


def test_unsupported_source_type(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    cfg = _write_fixture_tree(tmp_path)
    cfg["feeds"]["ops_snapshot"]["source"]["type"] = "s3"
    report = run_feeds(cfg, only=["ops_snapshot"], root=tmp_path)
    assert report["ok"] is False
    assert "NotImplementedError" in report["feeds"]["ops_snapshot"]["error"]
