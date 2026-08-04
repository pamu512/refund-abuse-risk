from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

from refund_abuse_risk.integrations.feeds import pull_sqlite, run_feeds


def test_pull_sqlite_dispositions_and_ops(tmp_path: Path) -> None:
    db = tmp_path / "wh.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame(
            [
                {
                    "order_id": "O1",
                    "disposition": "chargeback_lost",
                    "disposition_ts": "2026-07-10T00:00:00Z",
                }
            ]
        ).to_sql("dispositions", conn, index=False)
        pd.DataFrame(
            [
                {
                    "as_of": "2026-08-01T00:00:00Z",
                    "hold_rate": 0.2,
                    "live_override_rate": 0.05,
                    "shadow_override_rate": 0.1,
                    "refund_grant_rate": 0.5,
                    "refund_dollar_per_order": 3.0,
                    "cs_queue_depth": 40,
                }
            ]
        ).to_sql("ops_metrics", conn, index=False)

    disp = pull_sqlite(
        "dispositions",
        {
            "uri": str(db),
            "query": "SELECT order_id, disposition, disposition_ts FROM dispositions",
            "format": "csv",
        },
        root=tmp_path,
        stage_root=tmp_path / "stage",
    )
    assert disp.suffix == ".csv"
    assert "chargeback_lost" in disp.read_text(encoding="utf-8")

    ops = pull_sqlite(
        "ops_snapshot",
        {
            "uri": str(db),
            "query": "SELECT hold_rate, cs_queue_depth FROM ops_metrics LIMIT 1",
            "format": "json",
        },
        root=tmp_path,
        stage_root=tmp_path / "stage",
    )
    payload = json.loads(ops.read_text(encoding="utf-8"))
    assert payload["metrics"]["cs_queue_depth"] == 40.0


def test_run_feeds_sqlite_apply(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    wh = data / "feeds" / "warehouse"
    wh.mkdir(parents=True)
    db = wh / "demo.db"
    pd.DataFrame(
        [
            {
                "order_id": "O2",
                "user_id": "U2",
                "abuse_label": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "event_ts": "2026-07-01T00:00:00Z",
            },
            {
                "order_id": "O1",
                "user_id": "U1",
                "abuse_label": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "event_ts": "2026-07-01T00:00:00Z",
            },
        ]
    ).to_csv(data / "orders.csv", index=False)
    with sqlite3.connect(db) as conn:
        pd.DataFrame(
            [
                {
                    "order_id": "O2",
                    "disposition": "chargeback_lost",
                    "disposition_ts": "2026-07-15T00:00:00Z",
                }
            ]
        ).to_sql("dispositions", conn, index=False)
        pd.DataFrame(
            [{"hold_rate": 0.1, "cs_queue_depth": 12.0, "refund_dollar_per_order": 2.0}]
        ).to_sql("ops_metrics", conn, index=False)

    cfg = {
        "stage_root": "data/feeds/staging",
        "orders_path": "data/orders.csv",
        "orders_labeled_out": "data/orders.labeled.csv",
        "ops_sidecar": "data/ops_snapshot.json",
        "feeds": {
            "dispositions": {
                "enabled": True,
                "source": {
                    "type": "sqlite",
                    "uri": "data/feeds/warehouse/demo.db",
                    "query": "SELECT order_id, disposition, disposition_ts FROM dispositions",
                    "format": "csv",
                },
                "required_columns": ["order_id", "disposition", "disposition_ts"],
                "apply": {"kind": "dispositions"},
            },
            "ops_snapshot": {
                "enabled": True,
                "source": {
                    "type": "sqlite",
                    "uri": "data/feeds/warehouse/demo.db",
                    "query": "SELECT hold_rate, cs_queue_depth, refund_dollar_per_order FROM ops_metrics LIMIT 1",
                    "format": "json",
                },
                "required_columns": [],
                "apply": {"kind": "ops_snapshot"},
            },
        },
    }
    report = run_feeds(cfg, root=tmp_path)
    assert report["ok"] is True, report
    labeled = pd.read_csv(data / "orders.labeled.csv")
    assert int(labeled.loc[labeled["order_id"] == "O2", "fraud_label"].iloc[0]) == 1
    ops = json.loads((data / "ops_snapshot.json").read_text(encoding="utf-8"))
    assert ops["cs_queue_depth"] == 12.0


def test_oot_floor_check_helper() -> None:
    from scripts.eval_oot_pack import _check_floors

    floors = yaml.safe_load(
        Path("config/oot_floors.default.yaml").read_text(encoding="utf-8")
    )
    ok_metrics = {
        "n_holdout": 100,
        "n_proven": 5,
        "fraud_proven_average_precision": 0.5,
        "fraud_proven_precision_at_soft": 0.3,
        "decision_ece": 0.1,
    }
    assert _check_floors(ok_metrics, floors) == []
    bad = dict(ok_metrics)
    bad["fraud_proven_average_precision"] = 0.01
    assert any("proven_ap" in f for f in _check_floors(bad, floors))
