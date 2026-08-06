#!/usr/bin/env python3
"""Build a labeled OOT pack directory (demo CI or prod_shaped schema fixture)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _seed_demo(args: argparse.Namespace) -> dict:
    orders_path = args.data_dir / "orders.labeled.csv"
    if not (args.prefer_labeled and orders_path.exists()):
        orders_path = args.data_dir / "orders.csv"
    if not orders_path.exists():
        raise SystemExit(f"No orders at {orders_path}; run generate_demo_data.py first")

    orders = pd.read_csv(orders_path)
    if "event_ts" not in orders.columns:
        raise SystemExit("orders require event_ts for time-OOT")
    orders = orders.sort_values("event_ts").reset_index(drop=True)
    if args.max_orders and len(orders) > args.max_orders:
        orders = orders.tail(int(args.max_orders)).reset_index(drop=True)

    pack = args.pack_dir
    pack.mkdir(parents=True, exist_ok=True)
    orders.to_csv(pack / "orders.csv", index=False)

    for name in ("history.csv", "devices.csv", "users.csv", "dispositions.csv"):
        src = args.data_dir / name
        if src.exists():
            shutil.copy2(src, pack / name)

    floors = yaml.safe_load(
        (ROOT / "config" / "oot_floors.default.yaml").read_text(encoding="utf-8")
    )
    if str(pack.name).startswith("demo"):
        floors = {
            **floors,
            "min_holdout_n": max(
                10, min(int(floors.get("min_holdout_n", 30)), max(10, len(orders) // 6))
            ),
            "min_proven_positives": 1,
            "min_fraud_proven_precision_at_soft": 0.05,
            "min_fraud_proven_average_precision": 0.15,
            "max_decision_ece": 0.45,
        }
    manifest = {
        "pack_id": pack.name,
        "profile": "demo",
        "source_orders": str(orders_path),
        "n_orders": int(len(orders)),
        "holdout_days": 7,
        "floors": floors,
        "notes": "Synthetic/demo pack — not production lift proof.",
    }
    (pack / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return {"pack_dir": str(pack), "n_orders": int(len(orders)), "profile": "demo"}


def _seed_prod_shaped(args: argparse.Namespace) -> dict:
    """
    Build a prod_shaped fixture: enough orders + dispositions for schema floors.

    Metrics on this pack are still synthetic — validate_oot_pack is the CI gate;
    eval_oot_pack against prod AP floors may fail and that is expected.
    """
    pack = args.pack_dir
    pack.mkdir(parents=True, exist_ok=True)
    floors = yaml.safe_load(
        (ROOT / "config" / "oot_floors.prod.yaml").read_text(encoding="utf-8")
    )
    min_n = max(int(floors.get("min_holdout_n", 100)), int(args.max_orders or 120))
    min_proven = int(floors.get("min_proven_positives", 15))

    base_orders = args.data_dir / "orders.labeled.csv"
    if not base_orders.exists():
        base_orders = args.data_dir / "orders.csv"
    template = pd.read_csv(base_orders) if base_orders.exists() else pd.DataFrame()

    rows: list[dict] = []
    disp_rows: list[dict] = []
    base_ts = pd.Timestamp("2026-06-01", tz="UTC")
    markets = ["SG", "ID", "MY"]
    verticals = ["food", "qcommerce"]
    proven_outcomes = ["chargeback_lost", "bank_dispute_lost", "investigator_confirmed_fraud"]

    for i in range(min_n):
        oid = f"PS-{i:04d}"
        is_proven = i < min_proven
        market = markets[i % len(markets)]
        vertical = verticals[i % len(verticals)]
        ts = base_ts + pd.Timedelta(days=i % 40, hours=i % 20)
        row = {
            "order_id": oid,
            "user_id": f"U{i % 40}",
            "driver_id": f"D{i % 25}",
            "vendor_id": f"V{i % 20}",
            "device_id": f"DEV{i % 30}",
            "event_ts": ts.isoformat().replace("+00:00", "Z"),
            "amount": float(10 + (i % 40)),
            "status": "delivered",
            "market": market,
            "vertical": vertical,
            "is_refund": 1 if is_proven or i % 5 == 0 else 0,
            "claim_reason": "missing_item" if is_proven else "",
        }
        if len(template) and "fraud_label" in template.columns:
            row["fraud_label"] = 1 if is_proven else 0
            row["abuse_label"] = 1 if is_proven or i % 7 == 0 else 0
            row["fraud_label_source"] = "proven" if is_proven else ""
            row["strong_fraud_label"] = 1 if is_proven else 0
        rows.append(row)
        if is_proven:
            disp_rows.append(
                {
                    "order_id": oid,
                    "outcome": proven_outcomes[i % len(proven_outcomes)],
                    "disposition_ts": (ts + pd.Timedelta(days=8)).isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "source": "prod_shaped_fixture",
                }
            )

    orders = pd.DataFrame(rows)
    dispositions = pd.DataFrame(disp_rows)
    orders.to_csv(pack / "orders.csv", index=False)
    dispositions.to_csv(pack / "dispositions.csv", index=False)

    # Minimal history/devices so serve-path eval can run if someone tries.
    hist = []
    for i, r in enumerate(rows):
        hist.append(
            {
                "order_id": f"H-{r['order_id']}",
                "user_id": r["user_id"],
                "driver_id": r["driver_id"],
                "vendor_id": r["vendor_id"],
                "amount": r["amount"],
                "is_refund": r["is_refund"],
                "event_ts": (
                    pd.Timestamp(r["event_ts"]) - pd.Timedelta(days=3)
                ).isoformat().replace("+00:00", "Z"),
                "market": r["market"],
                "vertical": r["vertical"],
            }
        )
    pd.DataFrame(hist).to_csv(pack / "history.csv", index=False)
    devices = pd.DataFrame(
        [
            {
                "user_id": f"U{i}",
                "device_id": f"DEV{i}",
                "cluster_id": f"C{i % 10}",
                "last_seen_ts": "2026-07-15T00:00:00Z",
            }
            for i in range(40)
        ]
    )
    devices.to_csv(pack / "devices.csv", index=False)
    users = pd.DataFrame(
        [{"user_id": f"U{i}", "signup_ts": "2026-01-01T00:00:00Z"} for i in range(40)]
    )
    users.to_csv(pack / "users.csv", index=False)

    manifest = {
        "pack_id": pack.name,
        "profile": "prod_shaped",
        "require_dispositions": True,
        "floors_file": "config/oot_floors.prod.yaml",
        "floors": floors,
        "n_orders": int(len(orders)),
        "n_proven_dispositions": int(len(dispositions)),
        "holdout_days": 7,
        "notes": (
            "Prod-shaped schema fixture with dispositions. "
            "Not production lift proof — eval_oot_pack prod AP may fail on synth."
        ),
    }
    (pack / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return {
        "pack_dir": str(pack),
        "n_orders": int(len(orders)),
        "n_proven": int(len(dispositions)),
        "profile": "prod_shaped",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=None,
        help="Default: data/oot_packs/demo_v1 or prod_shaped_v1 by profile",
    )
    parser.add_argument(
        "--profile",
        choices=("demo", "prod_shaped"),
        default="demo",
    )
    parser.add_argument("--max-orders", type=int, default=None)
    parser.add_argument("--prefer-labeled", action="store_true", default=True)
    args = parser.parse_args()

    if args.pack_dir is None:
        name = "prod_shaped_v1" if args.profile == "prod_shaped" else "demo_v1"
        args.pack_dir = ROOT / "data" / "oot_packs" / name
    if args.max_orders is None:
        args.max_orders = 400 if args.profile == "demo" else 120

    if args.profile == "prod_shaped":
        out = _seed_prod_shaped(args)
    else:
        out = _seed_demo(args)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
