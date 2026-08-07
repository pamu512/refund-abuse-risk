#!/usr/bin/env python3
"""Backtest abuse/fraud heads; optionally tune thresholds for ~98% pattern recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
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
from refund_abuse_risk.training.splits import time_based_order_split
from refund_abuse_risk.scoring.decision import (
    recommend_decision_thresholds,
    recommend_decision_thresholds_by_slice,
    recommended_overlays_from_slices,
)
from refund_abuse_risk.scoring.monitoring import (
    brier_score,
    evaluate_monitoring_gates,
    expected_calibration_error,
    population_stability_index,
    summarize_ops_metrics,
)
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
        "--write-slice-overlays",
        type=Path,
        default=None,
        help="Write promote-eligible market×vertical decision_threshold_overlays YAML fragment",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Pattern recall target for --tune (default: from operating_point)",
    )
    parser.add_argument(
        "--oot-days",
        type=float,
        default=7.0,
        help="Primary holdout: last N days by event_ts (adaptive if span is shorter)",
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

    train_orders, test_orders, split_stats = time_based_order_split(
        orders, holdout_days=float(args.oot_days), min_train=10, min_test=5
    )
    if train_orders.empty or test_orders.empty:
        train_orders = orders.iloc[: max(len(orders) // 2, 1)].reset_index(drop=True)
        test_orders = orders.iloc[len(train_orders) :].reset_index(drop=True)
        split_stats = {
            **split_stats,
            "ok": False,
            "mode": "positional_fallback",
            "temporal_ok": False,
        }

    model = train_two_head(train_orders, history, devices, users=users, label_weights=label_weights)
    feat = build_order_feature_frame(test_orders, history, devices, users=users)
    feat = apply_proxy_fraud_labels(feat, label_weights)
    scored = model.predict_proba(feat)

    abuse_y = scored["abuse_label"].astype(int).to_numpy()
    fraud_y = scored["fraud_label"].astype(int).to_numpy()
    abuse_s = scored["abuse_score"].to_numpy(dtype=float)
    fraud_s = scored["fraud_score"].to_numpy(dtype=float)
    decision_s = scored["decision_score"].to_numpy(dtype=float)
    pattern_y = ((abuse_y >= 1) | (fraud_y >= 1)).astype(int)

    min_prec = thr.get("min_precision_at_soft", 0.15)
    max_fp = thr.get("max_fp_rate_at_soft")
    max_fp_dollars = thr.get("max_fp_refund_dollars_mean")
    amounts = (
        scored["amount"].astype(float).to_numpy()
        if "amount" in scored.columns
        else None
    )
    recommended_heads = recommend_head_thresholds(
        abuse_y,
        abuse_s,
        fraud_y,
        fraud_s,
        target_recall=target_recall,
        min_precision_at_soft=float(min_prec) if min_prec is not None else None,
        max_fp_rate_at_soft=float(max_fp) if max_fp is not None else None,
        apply_floors=False,
    )
    proven_mask = scored["fraud_label_source"].astype(str).str.lower().eq("proven")
    if "strong_fraud_label" in scored.columns:
        proven_mask = proven_mask | scored["strong_fraud_label"].astype(float).ge(1)
    proven_y = proven_mask.astype(int).to_numpy()
    source = scored["fraud_label_source"].astype(str).str.lower()
    proxy_only = source.eq("proxy") & (fraud_y >= 1) & (abuse_y < 1) & (proven_y < 1)
    # Promote ladder: proven fraud OR abuse — exclude proxy-only fraud mass.
    ladder_y = ((proven_y >= 1) | (abuse_y >= 1)).astype(int)
    ladder_y = np.where(proxy_only.to_numpy(), 0, ladder_y).astype(int)
    if int(ladder_y.sum()) < 3:
        ladder_y = pattern_y
        ladder_label = "pattern_fallback"
    else:
        ladder_label = "proven_or_abuse"
    mon_cfg_early = operating_point.get("monitoring") or {}
    recommended_decision = recommend_decision_thresholds(
        ladder_y,
        decision_s,
        target_recall=target_recall,
        min_precision_at_soft=float(min_prec) if min_prec is not None else None,
        max_fp_rate_at_soft=float(max_fp) if max_fp is not None else None,
        amounts=amounts,
        max_fp_refund_dollars_mean=(
            float(max_fp_dollars) if max_fp_dollars is not None else None
        ),
        apply_floors=False,
        precision_bootstrap_n=int(mon_cfg_early.get("precision_bootstrap_n") or 400),
        require_precision_bootstrap_ci=bool(
            mon_cfg_early.get("require_precision_bootstrap_ci")
        ),
    )
    recommended_decision["ladder_label"] = ladder_label
    recommended = recommended_decision  # primary promote signal
    slice_frame = scored.copy()
    slice_frame["pattern_y"] = ladder_y
    recommended_by_slice = recommend_decision_thresholds_by_slice(
        slice_frame,
        target_recall=target_recall,
        min_precision_at_soft=float(min_prec) if min_prec is not None else None,
        max_fp_rate_at_soft=float(max_fp) if max_fp is not None else None,
        max_fp_refund_dollars_mean=(
            float(max_fp_dollars) if max_fp_dollars is not None else None
        ),
        min_slice_n=20,
        min_slice_positives=3,
    )

    dthr = operating_point.get("decision_thresholds") or {}
    soft_d = float(dthr.get("soft_friction", recommended_decision["soft_friction"]))
    hold_d = float(dthr.get("hold_review", recommended_decision["hold_review"]))
    deny_d = float(dthr.get("auto_deny", recommended_decision["auto_deny"]))

    soft_a = float(thr.get("abuse_soft_friction", recommended_heads["abuse_soft_friction"]))
    soft_f = float(thr.get("fraud_soft_friction", recommended_heads["fraud_soft_friction"]))
    hold_a = float(thr.get("abuse_hold_review", recommended_heads["abuse_hold_review"]))
    hold_f = float(thr.get("fraud_hold_review", recommended_heads["fraud_hold_review"]))
    deny_a = float(thr.get("abuse_auto_deny", recommended_heads["abuse_auto_deny"]))
    deny_f = float(thr.get("fraud_auto_deny", recommended_heads["fraud_auto_deny"]))

    # Primary truth: proven fraud only (proxy metrics are secondary).
    fraud_proven_y = proven_y

    metrics = {
        "n_train": int(len(train_orders)),
        "n_test": int(len(test_orders)),
        "split": split_stats,
        "primary_holdout": "time_oot",
        "decision_mode": operating_point.get("decision_mode"),
        "honesty": {
            "recommended_ok": recommended.get("ok"),
            "cost_feasible": recommended.get("cost_feasible"),
            "soft_threshold_usable": recommended.get("soft_threshold_usable"),
            "ladder_label": ladder_label,
            "floor_would_bind": recommended.get("floor_would_bind"),
            "exclude_proxy_mint_features": bool(
                (label_weights.get("proxy_rules") or {}).get(
                    "exclude_mint_features_from_fraud_head", True
                )
            ),
            "primary_fraud_metric": "proven_only",
            "primary_decision_metric": "proven_ladder_when_supported",
            "primary_holdout": "time_oot",
            "stacker_fit_mode": getattr(model.decision_stacker, "fit_mode", None),
            "stack_label": getattr(model.decision_stacker, "stack_label", None),
        },
        "decision": {
            "roc_auc": _safe_auc(pattern_y.tolist(), (decision_s / 100.0).tolist()),
            "average_precision": _safe_ap(pattern_y.tolist(), (decision_s / 100.0).tolist()),
            "positive_rate": float(pattern_y.mean()) if len(pattern_y) else 0.0,
            "recall_at_soft": recall_at_threshold(pattern_y, decision_s, soft_d),
            "precision_at_soft": precision_at_threshold(pattern_y, decision_s, soft_d),
            "recall_at_hold": recall_at_threshold(pattern_y, decision_s, hold_d),
            "recall_at_deny": recall_at_threshold(pattern_y, decision_s, deny_d),
            "current_thresholds": {
                "soft_friction": soft_d,
                "hold_review": hold_d,
                "auto_deny": deny_d,
            },
            "recommended_thresholds": recommended_decision,
            "recommended_thresholds_by_slice": recommended_by_slice,
        },
        "abuse": {
            "roc_auc": _safe_auc(abuse_y.tolist(), (abuse_s / 100.0).tolist()),
            "average_precision": _safe_ap(abuse_y.tolist(), (abuse_s / 100.0).tolist()),
            "positive_rate": float(abuse_y.mean()) if len(abuse_y) else 0.0,
            "recall_at_soft": recall_at_threshold(abuse_y, abuse_s, soft_a),
            "precision_at_soft": precision_at_threshold(abuse_y, abuse_s, soft_a),
            "recall_at_hold": recall_at_threshold(abuse_y, abuse_s, hold_a),
            "recall_at_deny": recall_at_threshold(abuse_y, abuse_s, deny_a),
        },
        "fraud_proven_only": {
            "roc_auc": _safe_auc(fraud_proven_y.tolist(), (fraud_s / 100.0).tolist()),
            "average_precision": _safe_ap(fraud_proven_y.tolist(), (fraud_s / 100.0).tolist()),
            "positive_rate": float(fraud_proven_y.mean()) if len(fraud_proven_y) else 0.0,
            "n_positives": int(fraud_proven_y.sum()),
            "recall_at_soft": recall_at_threshold(fraud_proven_y, fraud_s, soft_f),
            "precision_at_soft": precision_at_threshold(fraud_proven_y, fraud_s, soft_f),
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
            "min_precision_at_soft": min_prec,
            "recall_at_soft_or": pattern_flag_recall(
                abuse_y, fraud_y, abuse_s, fraud_s, soft_a, soft_f
            ),
            "recall_at_hold_or": pattern_flag_recall(
                abuse_y, fraud_y, abuse_s, fraud_s, hold_a, hold_f
            ),
            "current_head_thresholds": {
                "abuse_soft_friction": soft_a,
                "fraud_soft_friction": soft_f,
                "abuse_hold_review": hold_a,
                "fraud_hold_review": hold_f,
                "abuse_auto_deny": deny_a,
                "fraud_auto_deny": deny_f,
            },
            "recommended_head_thresholds": recommended_heads,
        },
    }

    # OOT slices by market × vertical (pattern + proven honesty).
    slice_metrics: dict[str, Any] = {}
    min_slice_n = 20
    min_slice_proven = 3
    max_slice_ece = (operating_point.get("monitoring") or {}).get("max_decision_ece")
    slice_ece_ok = True
    slice_ece_failures: list[str] = []
    _mon = operating_point.get("monitoring") or {}
    slice_ece_kw = {
        "binning": str(_mon.get("ece_binning") or "adaptive"),
        "min_positives": int(_mon.get("min_ece_positives") or 5),
        "min_bin_n": int(_mon.get("min_ece_bin_n") or 3),
    }
    for (market, vertical), grp in scored.groupby(
        [scored["market"].astype(str), scored["vertical"].astype(str)], sort=True
    ):
        py = (
            (grp["abuse_label"].astype(int) >= 1) | (grp["fraud_label"].astype(int) >= 1)
        ).astype(int)
        ds = grp["decision_score"].astype(float)
        proven_g = grp["fraud_label_source"].astype(str).str.lower().eq("proven")
        proven_y = proven_g.astype(int)
        key = f"{market}|{vertical}"
        # Report pattern ECE; gate only on proven ECE (never green via pattern fallback).
        ece_pat = expected_calibration_error(py.to_numpy(), ds.to_numpy(), **slice_ece_kw)
        ece_proven = expected_calibration_error(
            proven_y.to_numpy(), ds.to_numpy(), **slice_ece_kw
        )
        thin = len(grp) < min_slice_n or int(proven_y.sum()) < min_slice_proven
        ece_val = ece_proven.get("ece") if int(proven_y.sum()) >= min_slice_proven else None
        ece_ok = True
        if not thin and max_slice_ece is not None:
            if ece_val is None or ece_proven.get("usable") is False:
                ece_ok = False
                slice_ece_ok = False
                slice_ece_failures.append(f"{key}:proven_ece_missing_or_unusable")
            elif float(ece_val) > float(max_slice_ece):
                ece_ok = False
                slice_ece_ok = False
                slice_ece_failures.append(key)
        slice_metrics[key] = {
            "n": int(len(grp)),
            "n_proven": int(proven_y.sum()),
            "thin_slice": bool(thin),
            "positive_rate": float(py.mean()) if len(py) else 0.0,
            "roc_auc": _safe_auc(py.tolist(), (ds / 100.0).tolist()),
            "average_precision": _safe_ap(py.tolist(), (ds / 100.0).tolist()),
            "recall_at_soft": recall_at_threshold(py.to_numpy(), ds.to_numpy(), soft_d),
            "precision_at_soft": precision_at_threshold(py.to_numpy(), ds.to_numpy(), soft_d),
            "fraud_proven_average_precision": _safe_ap(
                proven_y.tolist(), (ds / 100.0).tolist()
            ),
            "fraud_proven_precision_at_soft": precision_at_threshold(
                proven_y.to_numpy(), ds.to_numpy(), soft_d
            ),
            "decision_ece": ece_pat.get("ece"),
            "decision_ece_proven": ece_proven.get("ece"),
            "ece_gate": ece_val,
            "ece_ok": bool(ece_ok),
        }
    metrics["slices"] = slice_metrics
    _n_total = int(len(scored))
    _n_thin = int(sum(int(v["n"]) for v in slice_metrics.values() if v.get("thin_slice")))
    slice_thin_mass = float(_n_thin / _n_total) if _n_total else 0.0
    recommended_overlays = recommended_overlays_from_slices(recommended_by_slice)
    metrics["recommended_overlays"] = recommended_overlays
    metrics["recommended_by_slice"] = {
        k: {
            "ok": v.get("ok"),
            "promote_eligible": v.get("promote_eligible"),
            "thin_slice": v.get("thin_slice"),
            "n": v.get("n"),
            "soft_friction": v.get("soft_friction"),
            "hold_review": v.get("hold_review"),
            "auto_deny": v.get("auto_deny"),
        }
        for k, v in recommended_by_slice.items()
    }

    # Calibration / drift (train scores vs holdout) — ECE on decision, PSI train→test.
    # Ops gates finalized after dual-run effects below.
    train_feat = build_order_feature_frame(train_orders, history, devices, users=users)
    train_feat = apply_proxy_fraud_labels(train_feat, label_weights)
    train_scored = model.predict_proba(train_feat)
    train_decision = train_scored["decision_score"].to_numpy(dtype=float)
    # Promote ECE uses proven labels when support allows (audit H1 + calibration package).
    mon_pre = operating_point.get("monitoring") or {}
    min_proven_ece = int(mon_pre.get("min_proven_for_ece") or mon_pre.get("min_ece_positives") or 5)
    ece_binning = str(mon_pre.get("ece_binning") or "adaptive")
    min_ece_pos = int(mon_pre.get("min_ece_positives") or 5)
    min_ece_bin = int(mon_pre.get("min_ece_bin_n") or 3)
    ece_kwargs = {
        "binning": ece_binning,
        "min_positives": min_ece_pos,
        "min_bin_n": min_ece_bin,
    }
    if int(proven_y.sum()) >= min_proven_ece:
        decision_ece = expected_calibration_error(proven_y, decision_s, **ece_kwargs)
        decision_ece["label"] = "proven_fraud"
        decision_brier = brier_score(proven_y, decision_s)
        decision_brier["label"] = "proven_fraud"
    else:
        decision_ece = {
            "ece": None,
            "n": int(len(decision_s)),
            "n_positives": int(proven_y.sum()),
            "label": "proven_insufficient",
            "usable": False,
            "usable_reasons": ["proven_insufficient"],
            "binning": ece_binning,
            "pattern_ece_diagnostic": expected_calibration_error(
                pattern_y, decision_s, **ece_kwargs
            ).get("ece"),
        }
        decision_brier = {
            "brier": None,
            "n": int(len(decision_s)),
            "n_positives": int(proven_y.sum()),
            "label": "proven_insufficient",
        }
    psi = population_stability_index(train_decision, decision_s)
    metrics["monitoring"] = {
        "decision_ece": decision_ece,
        "decision_brier": decision_brier,
        "decision_ece_pattern_diagnostic": expected_calibration_error(
            pattern_y, decision_s, **ece_kwargs
        ),
        "decision_psi_train_vs_test": psi,
        "abuse_ece": expected_calibration_error(abuse_y, abuse_s, **ece_kwargs),
        "fraud_ece": expected_calibration_error(fraud_y, fraud_s, **ece_kwargs),
    }

    for source in ("proven", "proxy", "discovery"):
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
    base_effects: dict[str, int] = {}
    final_effects: dict[str, int] = {}
    shadow_effects: dict[str, int] = {}
    live_overrides = 0
    shadow_overrides = 0
    budget_floors = 0
    row_amounts: list[float] = []
    row_final_effects: list[str] = []
    shadow_hits = 0
    shadow_true = 0
    for snap in cache.all_snapshots():
        tier_counts[snap.suggested_tier.value] = tier_counts.get(snap.suggested_tier.value, 0) + 1
        hard_gate_count += int(snap.hard_gated)
        ed = snap.effect_decision or {}
        base = str(ed.get("base_effect") or snap.refund_effect.value)
        final = str(ed.get("final_effect") or snap.refund_effect.value)
        base_effects[base] = base_effects.get(base, 0) + 1
        final_effects[final] = final_effects.get(final, 0) + 1
        row_final_effects.append(final)
        amt = getattr(snap, "amount", None)
        if amt is None and hasattr(snap, "features"):
            amt = (snap.features or {}).get("order_amount")
        row_amounts.append(float(amt or 0.0))
        if ed.get("matched_mode") == "live":
            live_overrides += 1
        if ed.get("matched_mode") == "shadow" and ed.get("shadow_effect"):
            shadow_overrides += 1
            sh = str(ed["shadow_effect"])
            shadow_effects[sh] = shadow_effects.get(sh, 0) + 1
            # Precision of shadow overrides vs pattern labels (roadmap §4).
            shadow_hits += 1
            oid = str(getattr(snap, "order_id", "") or "")
            if oid and oid in scored["order_id"].astype(str).to_numpy():
                row = scored.loc[scored["order_id"].astype(str) == oid].iloc[0]
                if int(row.get("abuse_label", 0) or 0) >= 1 or int(row.get("fraud_label", 0) or 0) >= 1:
                    shadow_true += 1
        if ed.get("budget_floored") or (snap.refund_budget or {}).get("floored"):
            budget_floors += 1
    metrics["tiers"] = tier_counts
    metrics["hard_gated"] = hard_gate_count
    metrics["effects_dual_run"] = {
        "n": int(len(cache.all_snapshots())),
        "base_effect_counts": base_effects,
        "final_effect_counts": final_effects,
        "shadow_effect_counts": shadow_effects,
        "live_override_count": live_overrides,
        "shadow_override_count": shadow_overrides,
        "budget_floor_count": budget_floors,
        "shadow_override_precision": (
            float(shadow_true / shadow_hits) if shadow_hits else None
        ),
        "shadow_override_labeled_n": int(shadow_hits),
    }
    # Prefer order amounts from scored holdout when snap amount is missing.
    if "amount" in scored.columns and len(scored) == len(row_amounts):
        row_amounts = scored["amount"].astype(float).fillna(0.0).tolist()
    ops = summarize_ops_metrics(
        tier_counts=tier_counts,
        effects_dual_run=metrics["effects_dual_run"],
        amounts=row_amounts,
        final_effects=row_final_effects,
    )
    monitoring_cfg = dict(operating_point.get("monitoring") or {})
    # Live ops feed: data/ops_snapshot.json from scripts/ingest_ops_snapshot.py
    ops_sidecar = ROOT / "data" / "ops_snapshot.json"
    if ops_sidecar.exists():
        from refund_abuse_risk.integrations.ops_ingest import (
            load_ops_snapshot_file,
            merge_ops_snapshot,
        )

        monitoring_cfg = merge_ops_snapshot(monitoring_cfg, load_ops_snapshot_file(ops_sidecar))
        metrics["monitoring"]["ops_sidecar"] = str(ops_sidecar)
    mon_gate = evaluate_monitoring_gates(
        decision_ece=decision_ece.get("ece"),
        psi_train_test=psi.get("psi"),
        monitoring_cfg=monitoring_cfg,
        ops_metrics=ops,
        decision_brier=decision_brier.get("brier"),
        ece_usable=decision_ece.get("usable"),
        ece_usable_reasons=list(decision_ece.get("usable_reasons") or []),
    )
    metrics["monitoring"]["ops"] = ops
    metrics["monitoring"]["gate"] = mon_gate
    metrics["monitoring"]["slices"] = {
        "ok": bool(slice_ece_ok),
        "max_decision_ece": max_slice_ece,
        "failures": slice_ece_failures,
        "min_slice_n": min_slice_n,
        "min_slice_proven": min_slice_proven,
        "thin_slice_mass": slice_thin_mass,
        "n_thin_rows": _n_thin,
        "n_rows": _n_total,
    }
    monitoring_ok = bool(mon_gate.get("ok", True)) and bool(slice_ece_ok)
    metrics["monitoring"]["ok"] = monitoring_ok
    metrics["honesty"]["monitoring_ok"] = monitoring_ok
    metrics["honesty"]["slices_ece_ok"] = bool(slice_ece_ok)
    temporal_ok = bool(split_stats.get("temporal_ok", False))
    metrics["honesty"]["temporal_ok"] = temporal_ok
    metrics["honesty"]["promote_ok"] = bool(
        recommended.get("ok") and monitoring_ok and temporal_ok
    )
    if not temporal_ok:
        metrics["honesty"]["promote_block_reasons"] = list(
            metrics["honesty"].get("promote_block_reasons") or []
        ) + ["temporal_oot_not_honest"]
    metrics["honesty"]["ops_gate"] = True

    if args.write_slice_overlays is not None:
        out_path = Path(args.write_slice_overlays)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            yaml.safe_dump(
                {"decision_threshold_overlays": recommended_overlays},
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        metrics["slice_overlays_wrote"] = str(out_path)

    if args.tune:
        decision_proposal = {
            "soft_friction": round(float(recommended_decision["soft_friction"]), 2),
            "hold_review": round(float(recommended_decision["hold_review"]), 2),
            "auto_deny": round(float(recommended_decision["auto_deny"]), 2),
        }
        head = {
            "target_pattern_recall": target_recall,
            "abuse_soft_friction": round(float(recommended_heads["abuse_soft_friction"]), 2),
            "fraud_soft_friction": round(float(recommended_heads["fraud_soft_friction"]), 2),
            "abuse_hold_review": round(float(recommended_heads["abuse_hold_review"]), 2),
            "fraud_hold_review": round(float(recommended_heads["fraud_hold_review"]), 2),
            "abuse_auto_deny": round(float(recommended_heads["abuse_auto_deny"]), 2),
            "fraud_auto_deny": round(float(recommended_heads["fraud_auto_deny"]), 2),
        }
        # Head tuner still HIL-gates legacy head knobs; decision ladder written to tuned yaml.
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
            apply=bool(args.write_config) and bool(recommended_heads.get("ok")),
        )
        if args.write_config and recommended_decision.get("ok") and monitoring_ok:
            op_live = yaml.safe_load(OP_PATH.read_text(encoding="utf-8")) or {}
            op_live["decision_thresholds"] = decision_proposal
            op_live["decision_threshold_overlays"] = recommended_overlays
            OP_PATH.write_text(
                yaml.safe_dump(op_live, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
        tuned_path = ROOT / "examples" / "csv_demo" / "tuned_thresholds.yaml"
        tuned_path.parent.mkdir(parents=True, exist_ok=True)
        tuned_path.write_text(
            yaml.safe_dump(
                {
                    "decision_thresholds": decision_proposal,
                    "decision_threshold_overlays": [
                        {
                            "market": rec["market"],
                            "vertical": rec["vertical"],
                            "soft_friction": round(float(rec["soft_friction"]), 2),
                            "hold_review": round(float(rec["hold_review"]), 2),
                            "auto_deny": round(float(rec["auto_deny"]), 2),
                            "ok": bool(rec.get("ok")),
                            "thin_slice": bool(rec.get("thin_slice")),
                            "n": rec.get("n"),
                        }
                        for rec in recommended_by_slice.values()
                    ],
                    "head_thresholds": head,
                    "notes": (
                        "decision_thresholds are primary under decision_primary. "
                        "Overlays apply per market×vertical (first match). "
                        "Head knobs remain for evidence/baseline under_threshold. "
                        "Promote requires recommended.ok + monitoring.ok."
                    ),
                    "tuner_decision": decision.to_dict(),
                    "metrics_snapshot": {
                        "decision_recall_at_soft": recommended_decision.get("recall_at_soft"),
                        "decision_precision_at_soft": recommended_decision.get("precision_at_soft"),
                        "decision_ok": recommended_decision.get("ok"),
                        "stacker_fit_mode": getattr(model.decision_stacker, "fit_mode", None),
                    },
                },
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        metrics["tuned_wrote"] = str(tuned_path)
        metrics["tuned_decision_thresholds"] = decision_proposal
        metrics["tuned_head_thresholds"] = head
        metrics["tuner_decision"] = decision.to_dict()
        if args.write_config:
            promote_ok = bool(
                recommended_decision.get("ok", False) and monitoring_ok and temporal_ok
            )
            if not promote_ok:
                metrics["promoted_to"] = None
                reasons = []
                if not recommended_decision.get("ok", False):
                    reasons.append(
                        "decision recommended.ok is false (cost infeasible or soft floor would bind)"
                    )
                if not temporal_ok:
                    reasons.append("temporal_oot_not_honest (adaptive/positional split)")
                if not mon_gate.get("ok", True):
                    reasons.extend(mon_gate.get("reasons") or ["monitoring gate failed"])
                if not slice_ece_ok:
                    reasons.append(
                        "slice ECE failed: " + ",".join(slice_ece_failures or ["unknown"])
                    )
                metrics["promote_blocked"] = "; ".join(reasons)
            else:
                metrics["promoted_to"] = str(OP_PATH)
                metrics["hil_pending"] = decision.proposal_id

    out = ROOT / "examples" / "csv_demo" / "backtest_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
