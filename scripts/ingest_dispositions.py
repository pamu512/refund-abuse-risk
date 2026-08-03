#!/usr/bin/env python3
"""Apply refund/claim dispositions onto order training labels (closed-loop)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from refund_abuse_risk.config import load_disposition_labels
from refund_abuse_risk.labels.dispositions import apply_dispositions_to_orders

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=Path, default=ROOT / "data" / "orders.csv")
    parser.add_argument(
        "--dispositions", type=Path, default=ROOT / "data" / "dispositions.csv"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: data/orders.labeled.csv (does not overwrite source)",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Only use dispositions with disposition_ts <= as-of (ISO)",
    )
    parser.add_argument("--json-summary", action="store_true")
    args = parser.parse_args()

    cfg = load_disposition_labels()
    orders = pd.read_csv(args.orders)
    dispositions = pd.read_csv(args.dispositions)
    labeled = apply_dispositions_to_orders(
        orders, dispositions, cfg, as_of=args.as_of
    )
    out = args.out or (ROOT / "data" / "orders.labeled.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out, index=False)

    summary = {
        "orders_in": int(len(orders)),
        "dispositions_in": int(len(dispositions)),
        "dispositions_applied": int(labeled.attrs.get("dispositions_applied", 0)),
        "dispositions_skipped_lag": int(labeled.attrs.get("dispositions_skipped_lag", 0)),
        "lag_days": cfg.get("lag_days"),
        "unknown_dispositions": list(labeled.attrs.get("dispositions_unknown", [])),
        "out": str(out),
        "fraud_proven": int(
            (labeled.get("fraud_label_source", pd.Series(dtype=str)).astype(str).str.lower() == "proven").sum()
        ),
        "strong_fraud": int(pd.to_numeric(labeled.get("strong_fraud_label", 0), errors="coerce").fillna(0).sum()),
        "abuse_positives": int(pd.to_numeric(labeled.get("abuse_label", 0), errors="coerce").fillna(0).sum()),
    }
    if args.json_summary:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Applied {summary['dispositions_applied']} dispositions → {out} "
            f"(abuse+={summary['abuse_positives']}, proven={summary['fraud_proven']})"
        )


if __name__ == "__main__":
    main()
