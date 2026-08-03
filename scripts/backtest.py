#!/usr/bin/env python3
"""Backtest abuse/fraud heads; optionally tune thresholds for ~98% pattern recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from refund_abuse_risk.config import load_guardrails, load_label_weights, load_operating_point
from refund_abuse_risk.control_plane.audit import PolicyAuditLog
from refund_abuse_risk.control_plane.hil import HilProposalStore
from refund_abuse_risk.control_plane.tuner import run_threshold_tuner
from refund_abuse_risk.features.builders import build_order_feature_frame
from refund_abuse_risk.model.two_head import apply_proxy_fraud_labels
from refund_abuse_risk.pipeline.score import precompute_orders, train_two_head
from refund_abuse_risk.scoring.thresholds import (
    pattern_flag_recall,
    precision_at_threshold,
    recommend_head_thresholds,
    recall_at_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OP_PATH = ROOT / "config" / "operating_point.default.yaml"
CONTROL_DB = ROOT / "data" / "control_plane.db"


def _safe_auc(y_true, y_score) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score) -> float | None:
    if sum(y_true) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Run HIL-gated tuner: auto-step within max_abs_delta, queue remainder for approval. "
            "Also writes examples/csv_demo/tuned_thresholds.yaml with the full proposal."
        ),
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="With --tune, apply the bounded auto-step to operating_point (still HIL-gates large moves)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Pattern recall target for --tune (default: from operating_point)",
    )
    args = parser.parse_args()

    if not (DATA / "orders.csv").exists():
        import runpy

        runpy.run_path(str(ROOT / "scripts" / "generate_demo_data.py"), run_name="__main__")

    orders = pd.read_csv(DATA / "orders.csv")
    history = pd.read_csv(DATA / "history.csv")
    devices = pd.read_csv(DATA / "devices.csv")
    users = pd.read_csv(DATA / "users.csv") if (DATA / "users.csv").exists() else None
    label_weights = load_label_weights()
    operating_point = load_operating_point()
    thr = operating_point.get("head_thresholds") or {}
    target_recall = float(
        args.target_recall
        if args.target_recall is not None
        else thr.get("target_pattern_recall", 0.98)
    )

    # Simple holdout by order_id suffix hash.
    mask = orders["order_id"].astype(str).str.len() % 2 == 0
    train_orders = orders.loc[mask].reset_index(drop=True)
    test_orders = orders.loc[~mask].reset_index(drop=True)
    if train_orders.empty or test_orders.empty:
        train_orders = orders.iloc[: max(len(orders) // 2, 1)].reset_index(drop=True)
        test_orders = orders.iloc[len(train_orders) :].reset_index(drop=True)

    model = train_two_head(train_orders, history, devices, users=users, label_weights=label_weights)
    feat = build_order_feature_frame(test_orders, history, devices, users=users)
    feat = apply_proxy_fraud_labels(feat, label_weights)
    scored = model.predict_proba(feat)

    abuse_y = scored["abuse_label"].astype(int).to_numpy()
    fraud_y = scored["fraud_label"].astype(int).to_numpy()
    abuse_s = (scored["abuse_score"] / 1.0).to_numpy()
    fraud_s = (scored["fraud_score"] / 1.0).to_numpy()

    recommended = recommend_head_thresholds(
        abuse_y, abuse_s, fraud_y, fraud_s, target_recall=target_recall
    )

    soft_a = float(thr.get("abuse_soft_friction", recommended["abuse_soft_friction"]))
    soft_f = float(thr.get("fraud_soft_friction", recommended["fraud_soft_friction"]))
    hold_a = float(thr.get("abuse_hold_review", recommended["abuse_hold_review"]))
    hold_f = float(thr.get("fraud_hold_review", recommended["fraud_hold_review"]))
    deny_a = float(thr.get("abuse_auto_deny", recommended["abuse_auto_deny"]))
    deny_f = float(thr.get("fraud_auto_deny", recommended["fraud_auto_deny"]))

    metrics = {
        "n_train": int(len(train_orders)),
        "n_test": int(len(test_orders)),
        "decision_mode": operating_point.get("decision_mode"),
        "abuse": {
            "roc_auc": _safe_auc(abuse_y.tolist(), (abuse_s / 100.0).tolist()),
            "average_precision": _safe_ap(abuse_y.tolist(), (abuse_s / 100.0).tolist()),
            "positive_rate": float(abuse_y.mean()) if len(abuse_y) else 0.0,
            "recall_at_soft": recall_at_threshold(abuse_y, abuse_s, soft_a),
            "precision_at_soft": precision_at_threshold(abuse_y, abuse_s, soft_a),
            "recall_at_hold": recall_at_threshold(abuse_y, abuse_s, hold_a),
            "recall_at_deny": recall_at_threshold(abuse_y, abuse_s, deny_a),
        },
        "fraud_all": {
            "roc_auc": _safe_auc(fraud_y.tolist(), (fraud_s / 100.0).tolist()),
            "average_precision": _safe_ap(fraud_y.tolist(), (fraud_s / 100.0).tolist()),
            "positive_rate": float(fraud_y.mean()) if len(fraud_y) else 0.0,
            "recall_at_soft": recall_at_threshold(fraud_y, fraud_s, soft_f),
            "precision_at_soft": precision_at_threshold(fraud_y, fraud_s, soft_f),
            "recall_at_hold": recall_at_threshold(fraud_y, fraud_s, hold_f),
            "recall_at_deny": recall_at_threshold(fraud_y, fraud_s, deny_f),
        },
        "pattern_detection": {
            "target_recall": target_recall,
            "recall_at_soft_or": pattern_flag_recall(
                abuse_y, fraud_y, abuse_s, fraud_s, soft_a, soft_f
            ),
            "recall_at_hold_or": pattern_flag_recall(
                abuse_y, fraud_y, abuse_s, fraud_s, hold_a, hold_f
            ),
            "current_thresholds": {
                "abuse_soft_friction": soft_a,
                "fraud_soft_friction": soft_f,
                "abuse_hold_review": hold_a,
                "fraud_hold_review": hold_f,
                "abuse_auto_deny": deny_a,
                "fraud_auto_deny": deny_f,
            },
            "recommended_thresholds": recommended,
        },
    }

    for source in ("proven", "proxy"):
        src_mask = scored["fraud_label_source"].astype(str).str.lower().eq(source)
        neg_mask = scored["fraud_label"].astype(int).eq(0)
        slice_df = scored.loc[src_mask | neg_mask]
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
            "roc_auc": _safe_auc(y, (slice_df["fraud_score"] / 100.0).tolist()),
            "average_precision": _safe_ap(y, (slice_df["fraud_score"] / 100.0).tolist()),
            "positive_rate": float(sum(y) / len(y)) if y else 0.0,
        }

    cache = precompute_orders(
        test_orders, history, devices, model, users=users, operating_point=operating_point
    )
    tier_counts: dict[str, int] = {}
    hard_gate_count = 0
    for snap in cache.all_snapshots():
        tier_counts[snap.suggested_tier.value] = tier_counts.get(snap.suggested_tier.value, 0) + 1
        hard_gate_count += int(snap.hard_gated)
    metrics["tiers"] = tier_counts
    metrics["hard_gated"] = hard_gate_count

    if args.tune:
        head = {
            "target_pattern_recall": target_recall,
            "abuse_soft_friction": round(float(recommended["abuse_soft_friction"]), 2),
            "fraud_soft_friction": round(float(recommended["fraud_soft_friction"]), 2),
            "abuse_hold_review": round(float(recommended["abuse_hold_review"]), 2),
            "fraud_hold_review": round(float(recommended["fraud_hold_review"]), 2),
            "abuse_auto_deny": round(float(recommended["abuse_auto_deny"]), 2),
            "fraud_auto_deny": round(float(recommended["fraud_auto_deny"]), 2),
        }
        decision = run_threshold_tuner(
            current_thresholds=dict(thr),
            abuse_y=abuse_y,
            abuse_scores=abuse_s,
            fraud_y=fraud_y,
            fraud_scores=fraud_s,
            guardrails=load_guardrails(),
            audit=PolicyAuditLog(CONTROL_DB),
            hil_store=HilProposalStore(CONTROL_DB),
            operating_point_path=OP_PATH,
            apply=bool(args.write_config),
        )
        tuned_path = ROOT / "examples" / "csv_demo" / "tuned_thresholds.yaml"
        tuned_path.parent.mkdir(parents=True, exist_ok=True)
        tuned_path.write_text(
            yaml.safe_dump(
                {
                    "head_thresholds": head,
                    "notes": (
                        "Full ML proposal from holdout. Auto-apply only steps within "
                        "policy_guardrails.auto_apply.max_abs_delta; remainder needs HIL "
                        "(scripts/run_tuner.py --approve <id>)."
                    ),
                    "tuner_decision": decision.to_dict(),
                    "metrics_snapshot": {
                        "pattern_recall_at_soft": recommended.get("pattern_recall_at_soft"),
                        "abuse_precision_at_soft": recommended.get("abuse_precision_at_soft"),
                        "fraud_precision_at_soft": recommended.get("fraud_precision_at_soft"),
                    },
                },
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        metrics["tuned_wrote"] = str(tuned_path)
        metrics["tuned_thresholds"] = head
        metrics["tuner_decision"] = decision.to_dict()
        if args.write_config:
            metrics["promoted_to"] = str(OP_PATH)
            metrics["hil_pending"] = decision.proposal_id

    out = ROOT / "examples" / "csv_demo" / "backtest_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
