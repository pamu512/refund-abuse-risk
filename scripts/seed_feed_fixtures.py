#!/usr/bin/env python3
"""Seed data/feeds/fixtures (+ optional sqlite warehouse) from demo/examples."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _seed_warehouse(fixtures: Path, warehouse_db: Path) -> None:
    warehouse_db.parent.mkdir(parents=True, exist_ok=True)
    if warehouse_db.exists():
        warehouse_db.unlink()
    disp_path = fixtures / "dispositions" / "dispositions.csv"
    ops_path = fixtures / "ops_snapshot" / "ops_snapshot.json"
    with sqlite3.connect(warehouse_db) as conn:
        if disp_path.exists():
            pd.read_csv(disp_path).to_sql("dispositions", conn, index=False, if_exists="replace")
        else:
            pd.DataFrame(
                columns=["order_id", "disposition", "disposition_ts"]
            ).to_sql("dispositions", conn, index=False, if_exists="replace")
        metrics = {
            "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hold_rate": 0.18,
            "live_override_rate": 0.04,
            "shadow_override_rate": 0.11,
            "refund_grant_rate": 0.62,
            "refund_dollar_per_order": 4.2,
            "cs_queue_depth": 85.0,
        }
        if ops_path.exists():
            raw = json.loads(ops_path.read_text(encoding="utf-8"))
            m = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else raw
            metrics["as_of"] = str(raw.get("as_of") or metrics["as_of"])
            for k in list(metrics):
                if k != "as_of" and k in m:
                    metrics[k] = float(m[k])
        pd.DataFrame([metrics]).to_sql("ops_metrics", conn, index=False, if_exists="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=ROOT / "data" / "feeds" / "fixtures",
    )
    parser.add_argument(
        "--with-warehouse",
        action="store_true",
        help="Also build data/feeds/warehouse/demo.db from fixtures",
    )
    parser.add_argument(
        "--warehouse-db",
        type=Path,
        default=ROOT / "data" / "feeds" / "warehouse" / "demo.db",
    )
    args = parser.parse_args()

    fixtures = args.fixtures_root
    (fixtures / "dispositions").mkdir(parents=True, exist_ok=True)
    (fixtures / "sdk_events").mkdir(parents=True, exist_ok=True)
    (fixtures / "ops_snapshot").mkdir(parents=True, exist_ok=True)

    disp_src = args.data_dir / "dispositions.csv"
    orders = args.data_dir / "orders.csv"
    if disp_src.exists():
        shutil.copy2(disp_src, fixtures / "dispositions" / "dispositions.csv")
    elif orders.exists():
        # Minimal fixture so runner validates without closed-loop generate.
        od = pd.read_csv(orders).head(20)
        rows = []
        for _, r in od.iterrows():
            rows.append(
                {
                    "order_id": r["order_id"],
                    "disposition": "refund_granted_policy",
                    "disposition_ts": str(r.get("event_ts") or "2026-08-01T00:00:00Z"),
                }
            )
        pd.DataFrame(rows).to_csv(fixtures / "dispositions" / "dispositions.csv", index=False)

    sdk_example = ROOT / "examples" / "sdk_events.example.jsonl"
    sdk_large = args.data_dir / "sdk_events.jsonl"
    sdk_dst = fixtures / "sdk_events" / "sdk_events.jsonl"
    if sdk_large.exists():
        shutil.copy2(sdk_large, sdk_dst)
    elif sdk_example.exists():
        shutil.copy2(sdk_example, sdk_dst)

    ops_example = ROOT / "examples" / "ops_snapshot.example.json"
    ops_dst = fixtures / "ops_snapshot" / "ops_snapshot.json"
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if ops_example.exists():
        raw = json.loads(ops_example.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw["as_of"] = as_of
            ops_dst.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        else:
            shutil.copy2(ops_example, ops_dst)
    else:
        ops_dst.write_text(
            json.dumps(
                {
                    "as_of": as_of,
                    "source": "fixture",
                    "metrics": {
                        "cs_queue_depth": 85,
                        "refund_dollar_per_order": 4.2,
                        "hold_rate": 0.18,
                        "live_override_rate": 0.04,
                        "shadow_override_rate": 0.11,
                        "refund_grant_rate": 0.62,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    warehouse_ok = False
    if args.with_warehouse:
        _seed_warehouse(fixtures, args.warehouse_db)
        warehouse_ok = args.warehouse_db.exists()

    print(
        json.dumps(
            {
                "fixtures_root": str(fixtures),
                "dispositions": (fixtures / "dispositions" / "dispositions.csv").exists(),
                "sdk_events": sdk_dst.exists(),
                "ops_snapshot": ops_dst.exists(),
                "warehouse_db": str(args.warehouse_db) if warehouse_ok else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
