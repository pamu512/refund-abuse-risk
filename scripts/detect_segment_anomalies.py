#!/usr/bin/env python3
"""P0: Scan market×vertical refund-rate anomalies; write proposed rules (shadow only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from refund_abuse_risk.ops.segment_anomaly import (
    detect_segment_anomalies,
    proposals_to_yaml,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orders",
        type=Path,
        default=ROOT / "data" / "orders.csv",
    )
    parser.add_argument("--baseline-days", type=int, default=14)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--min-orders-per-day", type=int, default=5)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data" / "proposed_rules" / "segment_anomalies.json",
    )
    parser.add_argument(
        "--out-yaml",
        type=Path,
        default=ROOT / "data" / "proposed_rules" / "segment_anomalies.proposed.yaml",
    )
    args = parser.parse_args()

    if not args.orders.is_file():
        print(f"orders not found: {args.orders}", file=sys.stderr)
        return 1
    orders = pd.read_csv(args.orders)
    report = detect_segment_anomalies(
        orders,
        baseline_days=args.baseline_days,
        gap_days=args.gap_days,
        z_threshold=args.z_threshold,
        min_orders_per_day=args.min_orders_per_day,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    doc = proposals_to_yaml(report)
    args.out_yaml.write_text(
        yaml.safe_dump(doc, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_anomalies": report["n_anomalies"],
                "n_segments_scanned": report["n_segments_scanned"],
                "auto_enforce": False,
                "out_json": str(args.out_json),
                "out_yaml": str(args.out_yaml),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
