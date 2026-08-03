#!/usr/bin/env python3
"""Generate dispositions / chargebacks / QA sample + optional SDK confidences (roadmap §3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from refund_abuse_risk.config import load_disposition_labels
from refund_abuse_risk.integrations.sdk_ingest import apply_sdk_signals_to_orders
from refund_abuse_risk.labels.dispositions import apply_dispositions_to_orders

ROOT = Path(__file__).resolve().parents[1]


def _synth_dispositions(orders: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mint lagged dispositions including chargebacks + weak-policy auto-grants."""
    rows: list[dict] = []
    ts = pd.to_datetime(orders["event_ts"], utc=True, errors="coerce")
    for i, row in orders.iterrows():
        oid = str(row["order_id"])
        event = ts.loc[i]
        if pd.isna(event):
            continue
        abuse = int(row.get("abuse_label", 0) or 0) >= 1
        fraud = int(row.get("fraud_label", 0) or 0) >= 1
        src = str(row.get("fraud_label_source", "") or "").lower()
        weak = int(row.get("weak_policy_negative", 0) or 0) >= 1
        # Expand weak_policy coverage: soft-looking cleans get occasional auto-grants.
        if not abuse and not fraud and rng.random() < 0.08:
            weak = True
        lag = int(rng.integers(7, 21))
        disp_ts = (event + pd.Timedelta(days=lag)).isoformat()
        if src == "proven" or (fraud and rng.random() < 0.55):
            # Orthogonal hard truth: some proven via chargeback, some investigator.
            disp = "chargeback_lost" if rng.random() < 0.45 else "investigator_confirmed_fraud"
        elif fraud:
            disp = "manual_denied_fraud"
        elif abuse and rng.random() < 0.7:
            disp = "manual_denied_abuse"
        elif weak or (not abuse and not fraud and rng.random() < 0.12):
            disp = "weak_policy_auto_grant"
        elif rng.random() < 0.05:
            disp = "investigator_cleared"
        else:
            continue
        rows.append({"order_id": oid, "disposition": disp, "disposition_ts": disp_ts})
    return pd.DataFrame(rows)


def _qa_sample(dispositions: pd.DataFrame, rng: np.random.Generator, n: int = 50) -> pd.DataFrame:
    if dispositions.empty:
        return dispositions
    hard = dispositions[
        dispositions["disposition"].isin(
            ["investigator_confirmed_fraud", "chargeback_lost", "manual_denied_fraud"]
        )
    ]
    soft = dispositions[~dispositions.index.isin(hard.index)]
    n_hard = min(len(hard), max(1, n // 2))
    n_soft = min(len(soft), n - n_hard)
    parts = []
    if n_hard:
        parts.append(hard.sample(n=n_hard, random_state=int(rng.integers(0, 1_000_000))))
    if n_soft:
        parts.append(soft.sample(n=n_soft, random_state=int(rng.integers(0, 1_000_000))))
    out = pd.concat(parts, ignore_index=True) if parts else dispositions.head(0)
    out["qa_status"] = "pending_review"
    return out


def _sdk_events_for_orders(orders: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """Realistic confidence draws (Beta) — not flat 70–95."""
    events: list[dict] = []
    for _, row in orders.iterrows():
        oid = str(row["order_id"])
        risky = int(row.get("fraud_label", 0) or 0) >= 1 or int(row.get("abuse_label", 0) or 0) >= 1
        if rng.random() > (0.35 if risky else 0.08):
            continue
        # Risky → higher device risk / lower confidence noise; clean → sparse low scores.
        conf = float(np.clip(rng.beta(8, 2) if risky else rng.beta(2, 5), 0.05, 0.99))
        risk = float(np.clip(rng.beta(6, 2) * 100 if risky else rng.beta(2, 8) * 100, 0, 100))
        events.append(
            {
                "order_id": oid,
                "source": "device",
                "vendor": "synth_shield",
                "confidence": conf,
                "event_ts": str(row.get("event_ts", "")),
                "payload": {
                    "risk_score": risk,
                    "emulator": bool(risky and rng.random() < 0.4),
                    "cloned_app": bool(risky and rng.random() < 0.35),
                    "gps_spoof": bool(risky and rng.random() < 0.25),
                    "tampered": bool(risky and rng.random() < 0.3),
                    "customer_courier_same_device": bool(risky and rng.random() < 0.2),
                },
            }
        )
        if risky and rng.random() < 0.5:
            vconf = float(np.clip(rng.beta(7, 3), 0.05, 0.99))
            events.append(
                {
                    "order_id": oid,
                    "source": "vision",
                    "vendor": "synth_vision",
                    "confidence": vconf,
                    "event_ts": str(row.get("event_ts", "")),
                    "payload": {
                        "has_image": True,
                        "ai_risk": float(rng.beta(5, 3)),
                        "in_app_capture": bool(rng.random() < 0.7),
                        "pin_required": bool(rng.random() < 0.4),
                        "pin_verified": bool(rng.random() < 0.5),
                        "geofence_ok": bool(rng.random() < 0.8),
                    },
                }
            )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qa-n", type=int, default=80)
    parser.add_argument("--max-orders", type=int, default=20000, help="Cap for speed")
    parser.add_argument("--apply", action="store_true", help="Write orders.labeled.csv")
    parser.add_argument("--with-sdk", action="store_true", help="Also mint SDK events JSONL")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    orders_path = args.data_dir / "orders.csv"
    orders = pd.read_csv(orders_path)
    if args.max_orders and len(orders) > args.max_orders:
        orders = orders.sample(n=int(args.max_orders), random_state=int(args.seed)).reset_index(
            drop=True
        )

    # Expand weak_policy_negative rate on unlabeled soft rows before disposition mint.
    if "weak_policy_negative" not in orders.columns:
        orders["weak_policy_negative"] = 0
    clean = (
        (orders.get("abuse_label", 0).astype(int) < 1)
        & (orders.get("fraud_label", 0).astype(int) < 1)
        & (orders["weak_policy_negative"].astype(int) < 1)
    )
    flip = clean & (rng.random(len(orders)) < 0.10)
    orders.loc[flip, "weak_policy_negative"] = 1

    dispositions = _synth_dispositions(orders, rng)
    disp_path = args.data_dir / "dispositions.csv"
    dispositions.to_csv(disp_path, index=False)
    qa = _qa_sample(dispositions, rng, n=int(args.qa_n))
    qa_path = args.data_dir / "dispositions_qa_sample.csv"
    qa.to_csv(qa_path, index=False)

    summary: dict = {
        "orders": int(len(orders)),
        "dispositions": int(len(dispositions)),
        "chargeback_lost": int((dispositions["disposition"] == "chargeback_lost").sum()),
        "weak_policy_auto_grant": int(
            (dispositions["disposition"] == "weak_policy_auto_grant").sum()
        ),
        "qa_sample": int(len(qa)),
        "dispositions_path": str(disp_path),
        "qa_path": str(qa_path),
    }

    if args.apply:
        cfg = load_disposition_labels()
        labeled = apply_dispositions_to_orders(orders, dispositions, cfg)
        out = args.data_dir / "orders.labeled.csv"
        labeled.to_csv(out, index=False)
        summary["orders_labeled"] = str(out)
        summary["dispositions_applied"] = int(labeled.attrs.get("dispositions_applied", 0))
        summary["proven_after"] = int(
            (labeled["fraud_label_source"].astype(str).str.lower() == "proven").sum()
        )

    if args.with_sdk:
        events = _sdk_events_for_orders(orders, rng)
        ev_path = args.data_dir / "sdk_events.jsonl"
        with open(ev_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
        enriched = apply_sdk_signals_to_orders(orders, events)
        sdk_orders = args.data_dir / "orders.sdk.csv"
        enriched.to_csv(sdk_orders, index=False)
        summary["sdk_events"] = int(len(events))
        summary["sdk_events_path"] = str(ev_path)
        summary["sdk_orders_path"] = str(sdk_orders)
        summary["sdk_applied"] = int(enriched.attrs.get("sdk_events_applied", 0))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
