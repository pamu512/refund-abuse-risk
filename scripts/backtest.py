#!/usr/bin/env python3
"""Backtest abuse/fraud heads with proven vs proxy slices."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from refund_abuse_risk.config import load_label_weights
from refund_abuse_risk.features.builders import build_order_feature_frame
from refund_abuse_risk.model.two_head import TwoHeadModel, apply_proxy_fraud_labels
from refund_abuse_risk.pipeline.score import precompute_orders, train_two_head

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _safe_auc(y_true, y_score) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score) -> float | None:
    if sum(y_true) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def main() -> None:
    if not (DATA / "orders.csv").exists():
        import runpy

        runpy.run_path(str(ROOT / "scripts" / "generate_demo_data.py"), run_name="__main__")

    orders = pd.read_csv(DATA / "orders.csv")
    history = pd.read_csv(DATA / "history.csv")
    devices = pd.read_csv(DATA / "devices.csv")
    label_weights = load_label_weights()

    # Simple holdout by order_id suffix hash.
    mask = orders["order_id"].astype(str).str.len() % 2 == 0
    train_orders = orders.loc[mask].reset_index(drop=True)
    test_orders = orders.loc[~mask].reset_index(drop=True)
    if train_orders.empty or test_orders.empty:
        train_orders = orders.iloc[: max(len(orders) // 2, 1)].reset_index(drop=True)
        test_orders = orders.iloc[len(train_orders) :].reset_index(drop=True)

    model = train_two_head(train_orders, history, devices, label_weights=label_weights)
    feat = build_order_feature_frame(test_orders, history, devices)
    feat = apply_proxy_fraud_labels(feat, label_weights)
    scored = model.predict_proba(feat)

    abuse_y = scored["abuse_label"].astype(int).tolist()
    fraud_y = scored["fraud_label"].astype(int).tolist()
    metrics = {
        "n_train": int(len(train_orders)),
        "n_test": int(len(test_orders)),
        "abuse": {
            "roc_auc": _safe_auc(abuse_y, scored["abuse_score"] / 100.0),
            "average_precision": _safe_ap(abuse_y, scored["abuse_score"] / 100.0),
            "positive_rate": float(sum(abuse_y) / len(abuse_y)) if abuse_y else 0.0,
        },
        "fraud_all": {
            "roc_auc": _safe_auc(fraud_y, scored["fraud_score"] / 100.0),
            "average_precision": _safe_ap(fraud_y, scored["fraud_score"] / 100.0),
            "positive_rate": float(sum(fraud_y) / len(fraud_y)) if fraud_y else 0.0,
        },
    }

    for source in ("proven", "proxy"):
        # Evaluate fraud score treating only this source as positive vs clean negatives.
        src_mask = scored["fraud_label_source"].astype(str).str.lower().eq(source)
        neg_mask = scored["fraud_label"].astype(int).eq(0)
        slice_df = scored.loc[src_mask | neg_mask]
        y = slice_df["fraud_label"].astype(int).tolist()
        # For proven/proxy slice, positives are those with that source.
        y = [
            1 if (lab == 1 and str(src).lower() == source) else 0
            for lab, src in zip(
                slice_df["fraud_label"].astype(int),
                slice_df["fraud_label_source"].astype(str),
                strict=True,
            )
        ]
        metrics[f"fraud_{source}"] = {
            "n": int(len(slice_df)),
            "roc_auc": _safe_auc(y, slice_df["fraud_score"] / 100.0),
            "average_precision": _safe_ap(y, slice_df["fraud_score"] / 100.0),
            "positive_rate": float(sum(y) / len(y)) if y else 0.0,
        }

    cache = precompute_orders(test_orders, history, devices, model)
    tier_counts: dict[str, int] = {}
    hard_gate_count = 0
    for snap in cache.all_snapshots():
        tier_counts[snap.suggested_tier.value] = tier_counts.get(snap.suggested_tier.value, 0) + 1
        hard_gate_count += int(snap.hard_gated)
    metrics["tiers"] = tier_counts
    metrics["hard_gated"] = hard_gate_count

    out = ROOT / "examples" / "csv_demo" / "backtest_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
