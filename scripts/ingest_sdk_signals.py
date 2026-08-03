#!/usr/bin/env python3
"""Ingest device / claim-vision SDK envelopes onto orders (vendor-agnostic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from refund_abuse_risk.config import load_sdk_ingest
from refund_abuse_risk.integrations.sdk_ingest import apply_sdk_signals_to_orders

ROOT = Path(__file__).resolve().parents[1]


def _load_events(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "events" in data:
            return pd.DataFrame(data["events"])
        raise ValueError("JSON must be a list of events or {events: [...]}")
    return pd.read_csv(path)


def _serialize_nested(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("device_intelligence", "claim_vision"):
        if col not in out.columns:
            continue
        out[col] = out[col].map(
            lambda v: json.dumps(v) if isinstance(v, dict) else ("" if pd.isna(v) else v)
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=Path, default=ROOT / "data" / "orders.csv")
    parser.add_argument(
        "--events",
        type=Path,
        required=True,
        help="SDK envelopes: .jsonl / .json / .csv (payload JSON string ok in CSV)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: data/orders.sdk.csv (does not overwrite source)",
    )
    parser.add_argument("--json-summary", action="store_true")
    args = parser.parse_args()

    cfg = load_sdk_ingest()
    orders = pd.read_csv(args.orders)
    events = _load_events(args.events)
    enriched = apply_sdk_signals_to_orders(orders, events, cfg)
    out = args.out or (ROOT / "data" / "orders.sdk.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    _serialize_nested(enriched).to_csv(out, index=False)

    summary = {
        "orders_in": int(len(orders)),
        "events_in": int(enriched.attrs.get("sdk_events_in", 0)),
        "events_applied": int(enriched.attrs.get("sdk_events_applied", 0)),
        "events_rejected_confidence": int(
            enriched.attrs.get("sdk_events_rejected_confidence", 0)
        ),
        "min_confidence": cfg.get("min_confidence"),
        "out": str(out),
        "device_signals": int(
            (pd.to_numeric(enriched.get("device_signal_confidence", 0), errors="coerce").fillna(0) > 0).sum()
        ),
        "vision_signals": int(
            (pd.to_numeric(enriched.get("vision_signal_confidence", 0), errors="coerce").fillna(0) > 0).sum()
        ),
    }
    if args.json_summary:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Applied {summary['events_applied']} SDK signals → {out} "
            f"(device={summary['device_signals']}, vision={summary['vision_signals']}, "
            f"rejected_conf={summary['events_rejected_confidence']})"
        )


if __name__ == "__main__":
    main()
