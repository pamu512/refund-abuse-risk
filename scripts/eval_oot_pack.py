#!/usr/bin/env python3
"""Evaluate a labeled OOT pack against honesty floors (serve-path train + time holdout)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from refund_abuse_risk.config import load_label_weights, load_operating_point, load_yaml
from refund_abuse_risk.features.builders import build_order_feature_frame
from refund_abuse_risk.model.two_head import TwoHeadModel, apply_proxy_fraud_labels
from refund_abuse_risk.scoring.monitoring import expected_calibration_error
from refund_abuse_risk.scoring.thresholds import precision_at_threshold
from refund_abuse_risk.training.splits import time_based_order_split

ROOT = Path(__file__).resolve().parents[1]


def _safe_ap(y_true, y_score) -> float | None:
    if sum(y_true) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def _check_floors(metrics: dict[str, Any], floors: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    n = int(metrics.get("n_holdout") or 0)
    if n < int(floors.get("min_holdout_n") or 0):
        fails.append(f"n_holdout={n} < min_holdout_n={floors.get('min_holdout_n')}")
    n_proven = int(metrics.get("n_proven") or 0)
    if n_proven < int(floors.get("min_proven_positives") or 0):
        fails.append(
            f"n_proven={n_proven} < min_proven_positives={floors.get('min_proven_positives')}"
        )
    ap = metrics.get("fraud_proven_average_precision")
    min_ap = floors.get("min_fraud_proven_average_precision")
    if min_ap is not None and ap is not None and float(ap) < float(min_ap):
        fails.append(f"proven_ap={ap:.4f} < {min_ap}")
    prec = metrics.get("fraud_proven_precision_at_soft")
    min_p = floors.get("min_fraud_proven_precision_at_soft")
    if min_p is not None and prec is not None and float(prec) < float(min_p):
        fails.append(f"proven_precision_at_soft={prec:.4f} < {min_p}")
    ece = metrics.get("decision_ece")
    max_ece = floors.get("max_decision_ece")
    if max_ece is not None and ece is not None and float(ece) > float(max_ece):
        fails.append(f"decision_ece={ece:.4f} > {max_ece}")
    return fails


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=ROOT / "data" / "oot_packs" / "demo_v1",
    )
    parser.add_argument(
        "--floors",
        type=Path,
        default=ROOT / "config" / "oot_floors.default.yaml",
    )
    parser.add_argument("--max-train-rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pack = args.pack_dir
    manifest_path = pack / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    floors = load_yaml(args.floors)
    floors = {**floors, **(manifest.get("floors") or {})}

    orders = pd.read_csv(pack / "orders.csv")
    history = (
        pd.read_csv(pack / "history.csv")
        if (pack / "history.csv").exists()
        else pd.DataFrame()
    )
    devices = (
        pd.read_csv(pack / "devices.csv")
        if (pack / "devices.csv").exists()
        else pd.DataFrame(columns=["user_id", "device_id", "cluster_id", "last_seen_ts"])
    )
    users = pd.read_csv(pack / "users.csv") if (pack / "users.csv").exists() else None

    holdout_days = float(manifest.get("holdout_days") or 7)
    train_orders, test_orders, split = time_based_order_split(
        orders, holdout_days=holdout_days, min_train=20, min_test=10
    )
    if not split.get("ok"):
        print(json.dumps({"ok": False, "error": "time_oot_split_failed", "split": split}, indent=2))
        sys.exit(1)

    max_rows = int(args.max_train_rows)
    if len(train_orders) > max_rows:
        train_orders = train_orders.sample(n=max_rows, random_state=int(args.seed))

    train_feat = build_order_feature_frame(train_orders, history, devices, users=users)
    test_feat = build_order_feature_frame(test_orders, history, devices, users=users)
    lw = load_label_weights()
    train_feat = apply_proxy_fraud_labels(train_feat, lw)
    test_feat = apply_proxy_fraud_labels(test_feat, lw)

    op = load_operating_point()
    model = TwoHeadModel(model_version=str(op.get("model_version", "oot")))
    model.fit(train_feat, lw)
    scored = model.predict_proba(test_feat)

    proven = scored["fraud_label_source"].astype(str).str.lower().eq("proven")
    y_proven = proven.astype(int).to_numpy()
    decision = scored["decision_score"].astype(float).to_numpy()
    pattern_y = (
        (scored["abuse_label"].astype(int) >= 1) | (scored["fraud_label"].astype(int) >= 1)
    ).astype(int).to_numpy()
    soft = float((op.get("decision_thresholds") or {}).get("soft_friction", 35))

    metrics: dict[str, Any] = {
        "pack_dir": str(pack),
        "split": split,
        "n_train": int(len(train_feat)),
        "n_holdout": int(len(scored)),
        "n_proven": int(y_proven.sum()),
        "fraud_proven_average_precision": _safe_ap(y_proven.tolist(), (decision / 100.0).tolist()),
        "fraud_proven_precision_at_soft": precision_at_threshold(y_proven, decision, soft),
        "decision_ece": expected_calibration_error(pattern_y, decision).get("ece"),
        "soft_friction": soft,
        "feature_source": "serve",
    }
    fails = _check_floors(metrics, floors)
    metrics["floors"] = floors
    metrics["ok"] = len(fails) == 0
    metrics["failures"] = fails
    print(json.dumps(metrics, indent=2))
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
