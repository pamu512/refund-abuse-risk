#!/usr/bin/env python3
"""P1b: Apply labeling functions → weak/proxy labels (never overwrite proven)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from refund_abuse_risk.features.builders import build_order_feature_frame
from refund_abuse_risk.labels.weak_supervision import mint_weak_labels_from_lfs

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--orders",
        type=Path,
        default=None,
        help="Orders CSV (default: data/orders.labeled.csv or orders.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: data/orders.weak_labeled.csv)",
    )
    parser.add_argument(
        "--with-features",
        action="store_true",
        help="Build serve-path features before LFs (needs history/devices)",
    )
    args = parser.parse_args()

    orders_path = args.orders
    if orders_path is None:
        labeled = args.data_dir / "orders.labeled.csv"
        orders_path = labeled if labeled.exists() else args.data_dir / "orders.csv"
    if not orders_path.is_file():
        print(f"orders not found: {orders_path}", file=sys.stderr)
        return 1

    orders = pd.read_csv(orders_path)
    if args.with_features:
        history = pd.read_csv(args.data_dir / "history.csv")
        devices = pd.read_csv(args.data_dir / "devices.csv")
        users_path = args.data_dir / "users.csv"
        users = pd.read_csv(users_path) if users_path.exists() else None
        frame = build_order_feature_frame(orders, history, devices, users=users)
        # Keep label/source columns from orders.
        for col in ("fraud_label", "abuse_label", "fraud_label_source", "order_id"):
            if col in orders.columns and col not in frame.columns:
                frame[col] = orders[col].to_numpy()
        if "claim_reason" in orders.columns and "claim_reason" not in frame.columns:
            frame["claim_reason"] = orders["claim_reason"].to_numpy()
        minted = mint_weak_labels_from_lfs(frame)
    else:
        minted = mint_weak_labels_from_lfs(orders)

    out = args.out or (args.data_dir / "orders.weak_labeled.csv")
    minted.to_csv(out, index=False)
    n_weak = int((minted.get("weak_label", pd.Series(dtype=int)) == 1).sum())
    n_proven = int(
        minted.get("fraud_label_source", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .eq("proven")
        .sum()
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "n_rows": int(len(minted)),
                "n_weak_positive": n_weak,
                "n_proven_untouched": n_proven,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
