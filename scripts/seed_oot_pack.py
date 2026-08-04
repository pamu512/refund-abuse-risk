#!/usr/bin/env python3
"""Build a labeled OOT pack directory from demo/large data (CI / local honesty)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=ROOT / "data" / "oot_packs" / "demo_v1",
    )
    parser.add_argument("--max-orders", type=int, default=400)
    parser.add_argument("--prefer-labeled", action="store_true", default=True)
    args = parser.parse_args()

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

    for name in ("history.csv", "devices.csv", "users.csv"):
        src = args.data_dir / name
        if src.exists():
            shutil.copy2(src, pack / name)

    floors = yaml.safe_load(
        (ROOT / "config" / "oot_floors.default.yaml").read_text(encoding="utf-8")
    )
    # Demo packs: size-aware floors so CI can green; prod packs keep default floors.
    if str(pack.name).startswith("demo"):
        floors = {
            **floors,
            "min_holdout_n": max(10, min(int(floors.get("min_holdout_n", 30)), max(10, len(orders) // 6))),
            "min_proven_positives": 1,
            "min_fraud_proven_precision_at_soft": 0.05,
            "min_fraud_proven_average_precision": 0.15,
            "max_decision_ece": 0.45,
        }
    manifest = {
        "pack_id": pack.name,
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
    print(json.dumps({"pack_dir": str(pack), "n_orders": int(len(orders))}, indent=2))


if __name__ == "__main__":
    main()
