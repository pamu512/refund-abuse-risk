#!/usr/bin/env python3
"""Train two-head model — P0 serve-path features + capped as-of discovery multipass."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score

from refund_abuse_risk.config import load_label_weights, load_operating_point
from refund_abuse_risk.model.two_head import TwoHeadModel, apply_proxy_fraud_labels
from refund_abuse_risk.training.closed_loop import load_training_orders
from refund_abuse_risk.training.multipass import train_multipass
from refund_abuse_risk.training.serve_features import build_serve_training_frame
from refund_abuse_risk.training.splits import time_based_order_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "large"
DEFAULT_MODEL = ROOT / "models" / "two_head_large.joblib"


def _load_feature_frame(data_dir: Path) -> pd.DataFrame:
    pq = data_dir / "feature_frame.parquet"
    gz = data_dir / "feature_frame.csv.gz"
    csv = data_dir / "feature_frame.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if gz.exists():
        return pd.read_csv(gz)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"No feature_frame in {data_dir}. Run scripts/generate_large_demo_data.py first."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--feature-source",
        choices=("serve", "frame"),
        default="serve",
        help="serve=build_order_feature_frame (P0); frame=precomputed synthetic frame",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=4,
        help="Unsupervised→supervised cycles (default 4; use 1 for supervised only)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Train row cap (serve path default 5000; 0 = all for frame source)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=180,
        help="History window for serve-path feature build",
    )
    parser.add_argument(
        "--early-stop-delta",
        type=float,
        default=0.05,
        help="Stop multipass when |Δ decision_score_mean| stays below this",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,
        help="Consecutive flat passes before early stop",
    )
    parser.add_argument(
        "--oot-days",
        type=float,
        default=7.0,
        help="Hold out last N days for primary OOT metrics (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip UV discovery (IsolationForest + self-train only)",
    )
    parser.add_argument(
        "--no-closed-loop",
        action="store_true",
        help="Use raw orders.csv only (skip labeled/dispositions/SDK overlays)",
    )
    args = parser.parse_args()

    t0 = time.time()
    label_weights = load_label_weights()
    op = load_operating_point()
    model_version = str(op.get("model_version", "0.6.2"))
    feature_stats: dict = {"feature_source": args.feature_source}

    history = None
    hist_path = args.data_dir / "history.csv"
    if hist_path.exists():
        history = pd.read_csv(hist_path)

    oot_feat = None
    split_stats: dict = {"mode": "disabled", "ok": False}
    hist_for_features = history if history is not None else pd.DataFrame()

    if args.feature_source == "serve":
        if args.no_closed_loop:
            orders = pd.read_csv(args.data_dir / "orders.csv")
            closed_loop_stats = {"orders_source": str(args.data_dir / "orders.csv"), "disabled": True}
        else:
            orders, closed_loop_stats = load_training_orders(args.data_dir)
        devices = pd.read_csv(args.data_dir / "devices.csv")
        users = (
            pd.read_csv(args.data_dir / "users.csv")
            if (args.data_dir / "users.csv").exists()
            else None
        )
        max_rows = int(args.max_rows) if args.max_rows else 5000
        train_orders, test_orders = orders, orders.iloc[0:0]
        if float(args.oot_days) > 0:
            train_orders, test_orders, split_stats = time_based_order_split(
                orders, holdout_days=float(args.oot_days), min_train=50, min_test=20
            )
        feat, feature_stats = build_serve_training_frame(
            train_orders,
            hist_for_features,
            devices,
            users=users,
            max_rows=max_rows,
            lookback_days=int(args.lookback_days),
            random_state=int(args.seed),
        )
        feature_stats["closed_loop"] = closed_loop_stats
        if len(test_orders):
            oot_cap = min(2000, len(test_orders))
            oot_feat, oot_stats = build_serve_training_frame(
                test_orders,
                hist_for_features,
                devices,
                users=users,
                max_rows=oot_cap,
                lookback_days=int(args.lookback_days),
                random_state=int(args.seed) + 1,
            )
            feature_stats["oot"] = oot_stats
        # Discovery UV should use the same trimmed history window when possible.
        if history is not None and int(args.lookback_days) > 0:
            from refund_abuse_risk.training.serve_features import trim_history_lookback

            as_of = pd.to_datetime(feat["event_ts"], utc=True, errors="coerce").max()
            history = trim_history_lookback(
                history, lookback_days=int(args.lookback_days), as_of=as_of
            )
    else:
        feat_all = _load_feature_frame(args.data_dir)
        if float(args.oot_days) > 0:
            feat, oot_feat, split_stats = time_based_order_split(
                feat_all, holdout_days=float(args.oot_days), min_train=50, min_test=20
            )
        else:
            feat = feat_all
        if args.max_rows and len(feat) > args.max_rows:
            feat = feat.sample(n=int(args.max_rows), random_state=int(args.seed)).reset_index(
                drop=True
            )
        if oot_feat is not None and len(oot_feat) > 2000:
            oot_feat = oot_feat.sample(n=2000, random_state=int(args.seed) + 1).reset_index(
                drop=True
            )
        feature_stats = {
            "feature_source": "frame",
            "orders_sampled": int(len(feat)),
            "warning": "synthetic feature_frame — not serve-path parity",
        }

    if args.no_history:
        history = None

    n_passes = max(1, int(args.passes))
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    pass_reports: list[dict] = []

    if n_passes == 1:
        labeled = apply_proxy_fraud_labels(feat, label_weights)
        model = TwoHeadModel(model_version=model_version)
        model.fit(labeled, label_weights)
        scored = model.predict_proba(labeled)
        summary_extra = {
            "mode": "supervised_only",
            "stacker_fit_mode": getattr(model.decision_stacker, "fit_mode", None),
            "abuse_positive_rate": float(scored["abuse_label"].astype(int).mean()),
            "fraud_positive_rate": float(scored["fraud_label"].astype(int).mean()),
            "decision_score_mean": float(scored["decision_score"].mean()),
            "discovery_rate": float(
                (scored["fraud_label_source"].astype(str).str.lower() == "discovery").mean()
            ),
        }
    else:
        model, reports, _final = train_multipass(
            feat,
            history=history,
            n_passes=n_passes,
            label_weights=label_weights,
            model_version=model_version,
            random_state=int(args.seed),
            early_stop_delta=float(args.early_stop_delta),
            early_stop_patience=int(args.early_stop_patience),
        )
        pass_reports = [r.to_dict() for r in reports]
        last = pass_reports[-1]["supervised"] if pass_reports else {}
        last_u = pass_reports[-1]["unsupervised"] if pass_reports else {}
        summary_extra = {
            "mode": "unsupervised_supervised_multipass",
            "stacker_fit_mode": last.get("stacker_fit_mode"),
            "abuse_positive_rate": last.get("abuse_positive_rate"),
            "fraud_positive_rate": last.get("fraud_positive_rate"),
            "decision_score_mean": last.get("decision_score_mean"),
            "discovery_rate": last_u.get("discovery_rate"),
            "discovery_cap": last_u.get("discovery_cap"),
            "uv_mode": (last_u.get("uv") or {}).get("mode"),
            "n_passes_ran": len(pass_reports),
            "early_stopped": bool(last.get("early_stopped")),
            "passes": pass_reports,
        }

    oot_metrics: dict = {}
    if oot_feat is not None and len(oot_feat) and model is not None:
        oot_labeled = apply_proxy_fraud_labels(oot_feat, label_weights)
        oot_scored = model.predict_proba(oot_labeled)
        abuse_y = oot_scored["abuse_label"].astype(int).to_numpy()
        fraud_y = oot_scored["fraud_label"].astype(int).to_numpy()
        pattern_y = ((abuse_y >= 1) | (fraud_y >= 1)).astype(int)
        decision_s = oot_scored["decision_score"].to_numpy(dtype=float) / 100.0
        proven = oot_scored["fraud_label_source"].astype(str).str.lower().eq("proven").astype(int)
        fraud_s = oot_scored["fraud_score"].to_numpy(dtype=float) / 100.0

        def _auc(y, s):
            return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else None

        def _ap(y, s):
            return float(average_precision_score(y, s)) if int(y.sum()) > 0 else None

        from refund_abuse_risk.scoring.thresholds import precision_at_threshold

        soft = float((load_operating_point().get("decision_thresholds") or {}).get("soft_friction", 35))
        fraud_scores_100 = oot_scored["fraud_score"].to_numpy(dtype=float)
        oot_metrics = {
            "primary_holdout": "time_oot",
            "primary_metric": "fraud_proven_average_precision",
            "n": int(len(oot_scored)),
            "fraud_proven_roc_auc": _auc(proven.to_numpy(), fraud_s),
            "fraud_proven_average_precision": _ap(proven.to_numpy(), fraud_s),
            "fraud_proven_precision_at_soft": precision_at_threshold(
                proven.to_numpy(), fraud_scores_100, soft
            ),
            "decision_roc_auc": _auc(pattern_y, decision_s),
            "decision_average_precision": _ap(pattern_y, decision_s),
            "abuse_positive_rate": float(abuse_y.mean()),
            "fraud_positive_rate": float(fraud_y.mean()),
            "decision_score_mean": float(oot_scored["decision_score"].mean()),
        }
    summary_extra["split"] = split_stats
    summary_extra["oot"] = oot_metrics

    model.model_version = model_version
    model.save(args.model_out)

    summary = {
        "data_dir": str(args.data_dir),
        "model_out": str(args.model_out),
        "n_train": int(len(feat)),
        "n_passes": n_passes,
        "history_used": bool(history is not None and len(history) > 0),
        "model_version": model.model_version,
        "elapsed_sec": round(time.time() - t0, 2),
        "feature_stats": feature_stats,
        "discovery_config": label_weights.get("discovery"),
        **summary_extra,
    }
    summary_path = args.model_out.with_suffix(".train_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    compact = {k: v for k, v in summary.items() if k != "passes"}
    if pass_reports:
        compact["pass_decision_means"] = [
            p["supervised"].get("decision_score_mean") for p in pass_reports
        ]
        compact["pass_discovery_rates"] = [
            p["unsupervised"].get("discovery_rate") for p in pass_reports
        ]
        compact["pass_discovery_demoted"] = [
            (p["unsupervised"].get("discovery_cap") or {}).get("demoted") for p in pass_reports
        ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
