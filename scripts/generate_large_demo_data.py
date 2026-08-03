#!/usr/bin/env python3
"""Generate ~500k-line mock history+orders and a train-ready feature frame.

Feature columns are computed from generation-time aggregates (fast path for
large mock training). CSVs remain compatible with the normal score pipeline.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from refund_abuse_risk.features.builders import FEATURE_COLUMNS
from refund_abuse_risk.baselines.store import zero_baseline_features
from refund_abuse_risk.graph.bipartite import zero_bipartite_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "large"


SEGMENTS = (
    # name, weight, refund_p, abuse, fraud, fraud_source, strong, weak_neg
    # More weak_policy_negative coverage so soft auto-grants ≠ clean (roadmap §3.4).
    ("clean", 0.66, 0.04, 0, 0, "", 0, 0),
    ("abuse", 0.12, 0.55, 1, 0, "", 0, 0),
    ("burn", 0.05, 0.70, 1, 0, "", 0, 0),
    ("proxy", 0.05, 0.45, 1, 1, "proxy", 0, 0),
    ("fraud", 0.03, 0.60, 1, 1, "proven", 1, 0),
    ("weak", 0.09, 0.35, 0, 0, "", 0, 1),
)


def _assign_segments(n: int, rng: np.random.Generator) -> np.ndarray:
    weights = np.array([s[1] for s in SEGMENTS], dtype=float)
    weights /= weights.sum()
    return rng.choice(np.arange(len(SEGMENTS)), size=n, p=weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500_000, help="Target history+orders rows")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--orders-share",
        type=float,
        default=0.20,
        help="Fraction of --rows that are scored orders (rest = history)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    n_total = int(args.rows)
    n_orders = max(1_000, int(n_total * float(args.orders_share)))
    n_history = max(n_total - n_orders, n_orders * 3)
    # Reconcile to exact target when possible.
    if n_history + n_orders != n_total:
        n_history = n_total - n_orders
    n_users = max(2_000, n_orders)  # one scored order per user by default
    n_drivers = max(200, n_users // 40)
    n_vendors = max(150, n_users // 50)
    n_clusters = max(100, n_users // 80)

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    markets = np.array(["SG", "ID", "MY"])
    verticals = np.array(["food", "qcommerce"])

    seg_idx = _assign_segments(n_users, rng)
    user_ids = np.array([f"U{i:06d}" for i in range(n_users)])
    driver_ids = np.array([f"D{i:04d}" for i in range(n_drivers)])
    vendor_ids = np.array([f"V{i:04d}" for i in range(n_vendors)])
    device_ids = np.array([f"DEV{i:06d}" for i in range(n_users)])
    cluster_ids = np.array([f"C{i:04d}" for i in range(n_clusters)])

    user_driver = rng.integers(0, n_drivers, size=n_users)
    user_vendor = rng.integers(0, n_vendors, size=n_users)
    user_cluster = rng.integers(0, n_clusters, size=n_users)
    # Proxy/fraud farm: concentrate on few clusters.
    farm_clusters = np.arange(min(8, n_clusters))
    for i, s in enumerate(seg_idx):
        if SEGMENTS[s][0] in {"proxy", "fraud"}:
            user_cluster[i] = int(rng.choice(farm_clusters))

    user_market = markets[rng.integers(0, len(markets), size=n_users)]
    user_vertical = verticals[rng.integers(0, len(verticals), size=n_users)]
    signup_days = rng.integers(30, 400, size=n_users)

    # History: allocate ~n_history events across users (variable lifetime length).
    hist_per_user = rng.integers(4, 16, size=n_users).astype(int)
    # Scale to target history count.
    scale = n_history / max(int(hist_per_user.sum()), 1)
    hist_per_user = np.maximum(3, np.floor(hist_per_user * scale).astype(int))
    # Spread remainder across users — dumping onto [0] made one user span decades.
    diff = int(n_history - int(hist_per_user.sum()))
    if diff != 0:
        order = rng.permutation(n_users)
        step = 1 if diff > 0 else -1
        for i in range(abs(diff)):
            u = int(order[i % n_users])
            hist_per_user[u] = max(3, int(hist_per_user[u]) + step)

    hist_user = np.repeat(np.arange(n_users), hist_per_user)
    n_hist = len(hist_user)
    hist_j = np.concatenate([np.arange(k) for k in hist_per_user])
    refund_p = np.array([SEGMENTS[s][2] for s in seg_idx], dtype=float)[hist_user]
    # Clean users almost never refund early in life.
    is_refund = (rng.random(n_hist) < refund_p).astype(int)
    # Spread each user's events across a fixed calendar window (not hist_j as absolute day).
    span_days = 360
    hist_ts = []
    for u, j in zip(hist_user, hist_j, strict=True):
        k = int(hist_per_user[int(u)])
        day = int(round((j / max(k - 1, 1)) * (span_days - 1))) if k > 1 else span_days // 2
        day = int(np.clip(day + int(rng.integers(0, 2)), 0, span_days - 1))
        hist_ts.append((base + timedelta(days=day, hours=int(rng.integers(0, 24)))).isoformat())
    hist_reason = np.where(
        is_refund == 1,
        rng.choice(["missing_item", "quality", "wrong_order"], size=n_hist),
        "",
    )
    history = pd.DataFrame(
        {
            "order_id": [f"H{i:07d}" for i in range(n_hist)],
            "user_id": user_ids[hist_user],
            "driver_id": driver_ids[user_driver[hist_user]],
            "vendor_id": vendor_ids[user_vendor[hist_user]],
            "device_id": device_ids[hist_user],
            "market": user_market[hist_user],
            "vertical": user_vertical[hist_user],
            "amount": np.round(rng.uniform(12, 90, size=n_hist), 2),
            "is_refund": is_refund,
            "event_ts": hist_ts,
            "status": "delivered",
            "claim_reason": hist_reason,
        }
    )

    # Pre-aggregates for feature frame (mock-scale train path).
    hist_df = history.copy()
    hist_df["event_ts"] = pd.to_datetime(hist_df["event_ts"], utc=True)
    as_of = hist_df["event_ts"].max() + pd.Timedelta(days=2)
    win = hist_df["event_ts"] >= (as_of - pd.Timedelta(days=30))
    h30 = hist_df.loc[win]
    user_agg = h30.groupby("user_id").agg(
        user_orders_30d=("order_id", "size"),
        user_refund_count_30d=("is_refund", "sum"),
        user_gmv_30d=("amount", "sum"),
        user_refund_gmv_30d=("amount", lambda s: float(s[h30.loc[s.index, "is_refund"] == 1].sum()) if len(s) else 0.0),
    )
    # Safer refund GMV:
    tmp = h30.copy()
    tmp["refund_amt"] = np.where(tmp["is_refund"] == 1, tmp["amount"], 0.0)
    user_agg = tmp.groupby("user_id").agg(
        user_orders_30d=("order_id", "size"),
        user_refund_count_30d=("is_refund", "sum"),
        user_gmv_30d=("amount", "sum"),
        user_refund_gmv_30d=("refund_amt", "sum"),
        user_refund_count_7d=("is_refund", "sum"),  # approx; refined below
    )
    # 7d window
    w7 = hist_df["event_ts"] >= (as_of - pd.Timedelta(days=7))
    u7 = hist_df.loc[w7].groupby("user_id")["is_refund"].sum().rename("user_refund_count_7d")
    user_agg = user_agg.drop(columns=["user_refund_count_7d"], errors="ignore").join(u7, how="left")
    user_agg["user_refund_count_7d"] = user_agg["user_refund_count_7d"].fillna(0)
    user_agg["user_refund_rate_30d"] = user_agg["user_refund_count_30d"] / user_agg["user_orders_30d"].clip(lower=1)
    user_agg["user_refund_gmv_pct_30d"] = user_agg["user_refund_gmv_30d"] / user_agg["user_gmv_30d"].clip(lower=1)

    life = hist_df.groupby("user_id").agg(
        user_lifetime_orders=("order_id", "size"),
        user_lifetime_refund_count=("is_refund", "sum"),
        user_lifetime_gmv=("amount", "sum"),
    )
    life_ref = hist_df.assign(refund_amt=np.where(hist_df["is_refund"] == 1, hist_df["amount"], 0.0)).groupby("user_id")[
        "refund_amt"
    ].sum().rename("user_lifetime_refund_gmv")
    life = life.join(life_ref, how="left")
    life["user_ltv_net"] = life["user_lifetime_gmv"] - life["user_lifetime_refund_gmv"]
    life["user_refund_to_ltv_ratio"] = life["user_lifetime_refund_gmv"] / life["user_ltv_net"].abs().clip(lower=1)

    driver_agg = tmp.groupby("driver_id").agg(
        driver_orders_30d=("order_id", "size"),
        driver_refund_count_30d=("is_refund", "sum"),
    )
    driver_agg["driver_refund_rate_30d"] = (
        driver_agg["driver_refund_count_30d"] / driver_agg["driver_orders_30d"].clip(lower=1)
    )
    vendor_agg = tmp.groupby("vendor_id").agg(
        vendor_orders_30d=("order_id", "size"),
        vendor_refund_count_30d=("is_refund", "sum"),
        vendor_gmv_30d=("amount", "sum"),
        vendor_refund_gmv_30d=("refund_amt", "sum"),
    )
    vendor_agg["vendor_refund_rate_30d"] = (
        vendor_agg["vendor_refund_count_30d"] / vendor_agg["vendor_orders_30d"].clip(lower=1)
    )
    vendor_agg["vendor_refund_gmv_pct_30d"] = (
        vendor_agg["vendor_refund_gmv_30d"] / vendor_agg["vendor_gmv_30d"].clip(lower=1)
    )

    # Farm clusters inflate accounts_per_device / device_cluster_size.
    accounts_per_cluster = pd.Series(user_cluster).value_counts()

    # Scored orders: one per user (n_orders may be < n_users → sample)
    order_users = np.arange(n_users)
    if n_orders < n_users:
        # Stratified-ish: keep all minority segments, sample clean.
        minority = np.array([SEGMENTS[s][0] != "clean" for s in seg_idx])
        keep = np.where(minority)[0]
        need = n_orders - len(keep)
        if need > 0:
            clean_idx = np.where(~minority)[0]
            pick = rng.choice(clean_idx, size=min(need, len(clean_idx)), replace=False)
            order_users = np.concatenate([keep, pick])
        else:
            order_users = keep[:n_orders]
        rng.shuffle(order_users)
        order_users = order_users[:n_orders]
    else:
        # Extra orders: sample users with replacement for remainder.
        extra = n_orders - n_users
        order_users = np.concatenate(
            [np.arange(n_users), rng.choice(np.arange(n_users), size=extra, replace=True)]
        )

    n_ord = len(order_users)
    order_ids = np.array([f"O{i:07d}" for i in range(n_ord)])
    # Spread scored orders across the last 30 calendar days so time-OOT holdouts work.
    order_span_days = 30
    score_start = as_of - pd.Timedelta(days=order_span_days)
    order_offsets = rng.integers(0, order_span_days * 24, size=n_ord)
    order_ts = [
        (score_start + pd.Timedelta(hours=int(h))).isoformat() for h in order_offsets
    ]
    amounts = np.round(rng.uniform(15, 95, size=n_ord), 2)
    claim_reasons = []
    abuse_label = []
    abuse_weak = []
    fraud_label = []
    fraud_src = []
    strong = []
    weak_neg = []
    for ui in order_users:
        name, _w, _p, ab, fr, src, st, wn = SEGMENTS[int(seg_idx[ui])]
        abuse_label.append(ab)
        abuse_weak.append(1 if name in {"abuse", "burn", "weak"} and ab else 0)
        fraud_label.append(fr)
        fraud_src.append(src)
        strong.append(st)
        weak_neg.append(wn)
        if ab or fr:
            claim_reasons.append(str(rng.choice(["missing_item", "quality", "wrong_order"])))
        else:
            claim_reasons.append("")

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "user_id": user_ids[order_users],
            "driver_id": driver_ids[user_driver[order_users]],
            "vendor_id": vendor_ids[user_vendor[order_users]],
            "device_id": device_ids[order_users],
            "market": user_market[order_users],
            "vertical": user_vertical[order_users],
            "amount": amounts,
            "status": "delivered",
            "claim_reason": claim_reasons,
            "event_ts": order_ts,
            "abuse_label": abuse_label,
            "abuse_label_weak": abuse_weak,
            "fraud_label": fraud_label,
            "fraud_label_source": fraud_src,
            "strong_fraud_label": strong,
            "weak_policy_negative": weak_neg,
            "prior_strong_fraud": strong,
            "customer_courier_same_device": [
                1 if SEGMENTS[int(seg_idx[ui])][0] in {"proxy", "fraud"} and rng.random() < 0.5 else 0
                for ui in order_users
            ],
            "claim_has_image": [1 if abuse_label[i] or fraud_label[i] else 0 for i in range(n_ord)],
            "claim_image_ai_risk": [
                float(rng.uniform(0.75, 0.95)) if SEGMENTS[int(seg_idx[ui])][0] == "burn" else 0.0
                for ui in order_users
            ],
            "claim_in_app_capture": [
                0 if SEGMENTS[int(seg_idx[ui])][0] == "burn" else 1 for ui in order_users
            ],
            "pin_required": [1 if SEGMENTS[int(seg_idx[ui])][0] == "clean" else int(rng.random() < 0.3) for ui in order_users],
            "pin_verified": [1 if SEGMENTS[int(seg_idx[ui])][0] == "clean" else 0 for ui in order_users],
            "delivery_geofence_ok": [1 if SEGMENTS[int(seg_idx[ui])][0] == "clean" else int(rng.random() < 0.4) for ui in order_users],
        }
    )

    devices = pd.DataFrame(
        {
            "user_id": user_ids,
            "device_id": device_ids,
            "cluster_id": [f"C{c:04d}" for c in user_cluster],
            "last_seen_ts": as_of.isoformat(),
            "device_risk_score": [
                float(rng.uniform(70, 95)) if SEGMENTS[int(seg_idx[i])][0] in {"proxy", "fraud"} else float(rng.uniform(0, 25))
                for i in range(n_users)
            ],
            "is_emulator": [1 if SEGMENTS[int(seg_idx[i])][0] == "proxy" and rng.random() < 0.6 else 0 for i in range(n_users)],
            "is_cloned_app": [1 if SEGMENTS[int(seg_idx[i])][0] in {"proxy", "fraud"} else 0 for i in range(n_users)],
            "is_gps_spoof": [1 if SEGMENTS[int(seg_idx[i])][0] == "fraud" and rng.random() < 0.5 else 0 for i in range(n_users)],
            "is_tampered": [1 if SEGMENTS[int(seg_idx[i])][0] in {"proxy", "fraud"} and rng.random() < 0.4 else 0 for i in range(n_users)],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "signup_ts": [(base - timedelta(days=int(d))).isoformat() for d in signup_days],
        }
    )

    # Feature frame for fast training.
    feat_rows: list[dict[str, float]] = []
    meta_rows: list[dict] = []
    u_lookup = user_agg
    life_lookup = life
    d_lookup = driver_agg
    v_lookup = vendor_agg
    dev_map = devices.set_index("device_id")
    zeros_base = zero_baseline_features()
    zeros_bip = zero_bipartite_features()

    for i, row in orders.iterrows():
        uid = str(row["user_id"])
        did = str(row["driver_id"])
        vid = str(row["vendor_id"])
        deid = str(row["device_id"])
        u = u_lookup.loc[uid] if uid in u_lookup.index else None
        lf = life_lookup.loc[uid] if uid in life_lookup.index else None
        dr = d_lookup.loc[did] if did in d_lookup.index else None
        ve = v_lookup.loc[vid] if vid in v_lookup.index else None
        dev = dev_map.loc[deid] if deid in dev_map.index else None
        orders_30 = float(u["user_orders_30d"]) if u is not None else 1.0
        ref_30 = float(u["user_refund_count_30d"]) if u is not None else 0.0
        rate = float(u["user_refund_rate_30d"]) if u is not None else 0.0
        gmv_pct = float(u["user_refund_gmv_pct_30d"]) if u is not None else 0.0
        cluster = str(dev["cluster_id"]) if dev is not None else "C0"
        c_size = float(accounts_per_cluster.get(int(cluster[1:]), 1)) if cluster.startswith("C") else 1.0
        # Link lifts: elevated for proxy/fraud segments.
        ui = int(str(uid).lstrip("U"))
        seg_name = SEGMENTS[int(seg_idx[ui])][0]
        lift = 2.5 if seg_name in {"proxy", "fraud"} else (1.6 if seg_name in {"abuse", "burn"} else 1.0)
        cooccur = 8.0 if seg_name in {"proxy", "fraud"} else 3.0
        feat = {**zeros_base, **zeros_bip}
        feat.update(
            {
                "order_amount": float(row["amount"]),
                "user_orders_30d": orders_30,
                "user_refund_count_7d": float(u["user_refund_count_7d"]) if u is not None else 0.0,
                "user_refund_count_30d": ref_30,
                "user_refund_rate_30d": rate,
                "user_refund_gmv_pct_30d": gmv_pct,
                "user_days_since_signup": float(signup_days[ui]),
                "user_lifetime_orders": float(lf["user_lifetime_orders"]) if lf is not None else orders_30,
                "user_lifetime_refund_count": float(lf["user_lifetime_refund_count"]) if lf is not None else ref_30,
                "user_lifetime_gmv": float(lf["user_lifetime_gmv"]) if lf is not None else float(row["amount"]),
                "user_lifetime_refund_gmv": float(lf["user_lifetime_refund_gmv"]) if lf is not None else 0.0,
                "user_ltv_net": float(lf["user_ltv_net"]) if lf is not None else float(row["amount"]),
                "user_refund_to_ltv_ratio": float(lf["user_refund_to_ltv_ratio"]) if lf is not None else 0.0,
                "related_account_count": max(1.0, c_size - 1.0) if seg_name in {"proxy", "fraud"} else 0.0,
                "related_refund_count_30d": ref_30 * (0.8 if seg_name in {"proxy", "fraud"} else 0.0),
                "related_refund_gmv_30d": float(u["user_refund_gmv_30d"]) * 0.5 if u is not None and seg_name in {"proxy", "fraud"} else 0.0,
                "related_max_refund_rate_30d": rate if seg_name in {"proxy", "fraud"} else 0.0,
                "related_max_orders_30d": orders_30 if seg_name in {"proxy", "fraud"} else 0.0,
                "combined_refund_count_30d": ref_30 * (1.5 if seg_name in {"proxy", "fraud"} else 1.0),
                "combined_refund_gmv_30d": float(u["user_refund_gmv_30d"]) if u is not None else 0.0,
                "driver_orders_30d": float(dr["driver_orders_30d"]) if dr is not None else 1.0,
                "driver_refund_count_30d": float(dr["driver_refund_count_30d"]) if dr is not None else 0.0,
                "driver_refund_rate_30d": float(dr["driver_refund_rate_30d"]) if dr is not None else 0.0,
                "vendor_orders_30d": float(ve["vendor_orders_30d"]) if ve is not None else 1.0,
                "vendor_refund_count_30d": float(ve["vendor_refund_count_30d"]) if ve is not None else 0.0,
                "vendor_refund_rate_30d": float(ve["vendor_refund_rate_30d"]) if ve is not None else 0.0,
                "vendor_refund_gmv_pct_30d": float(ve["vendor_refund_gmv_pct_30d"]) if ve is not None else 0.0,
                "accounts_per_device": max(1.0, c_size / 5.0) if seg_name in {"proxy", "fraud"} else 1.0,
                "devices_per_account": 1.0,
                "device_cluster_size": max(1.0, c_size if seg_name in {"proxy", "fraud"} else 1.0),
                "device_churn_30d": 0.0,
                "ud_cooccur": cooccur,
                "ud_refund_lift": lift,
                "ud_refund_share": min(0.9, rate),
                "uv_cooccur": cooccur,
                "uv_refund_lift": lift,
                "uv_refund_share": min(0.9, rate),
                "vd_cooccur": cooccur * 0.7,
                "vd_refund_lift": lift * 0.9,
                "uvd_cooccur": cooccur,
                "uvd_refund_lift": lift,
                "uvd_refund_share": min(0.95, rate + (0.2 if seg_name in {"proxy", "fraud"} else 0.0)),
                "reason_repeat_rate_30d": min(1.0, rate),
                "is_food": 1.0 if row["vertical"] == "food" else 0.0,
                "is_qcommerce": 1.0 if row["vertical"] == "qcommerce" else 0.0,
                "claim_reason_missing_item": 1.0 if "missing" in str(row["claim_reason"]) else 0.0,
                "claim_reason_quality": 1.0 if "quality" in str(row["claim_reason"]) else 0.0,
                "claim_reason_wrong_order": 1.0 if "wrong" in str(row["claim_reason"]) else 0.0,
                "order_status_delivered": 1.0,
                "device_risk_score": float(dev["device_risk_score"]) if dev is not None else 0.0,
                "is_emulator": float(dev["is_emulator"]) if dev is not None else 0.0,
                "is_cloned_app": float(dev["is_cloned_app"]) if dev is not None else 0.0,
                "is_gps_spoof": float(dev["is_gps_spoof"]) if dev is not None else 0.0,
                "is_tampered": float(dev["is_tampered"]) if dev is not None else 0.0,
                "customer_courier_same_device": float(row["customer_courier_same_device"]),
                "claim_has_image": float(row["claim_has_image"]),
                "claim_image_ai_risk": float(row["claim_image_ai_risk"]),
                "claim_in_app_capture": float(row["claim_in_app_capture"]),
                "pin_required": float(row["pin_required"]),
                "pin_verified": float(row["pin_verified"]),
                "delivery_geofence_ok": float(row["delivery_geofence_ok"]),
                "uv_edge_anomaly": 2.0 if seg_name in {"proxy", "fraud"} else 0.0,
                "user_bipartite_anomaly": 1.5 if seg_name in {"proxy", "fraud"} else 0.0,
                "vendor_bipartite_anomaly": 1.0 if seg_name in {"proxy", "fraud"} else 0.0,
            }
        )
        # Ensure all FEATURE_COLUMNS present.
        for col in FEATURE_COLUMNS:
            feat.setdefault(col, 0.0)
        feat_rows.append({k: float(feat[k]) for k in FEATURE_COLUMNS})
        meta_rows.append(
            {
                "order_id": row["order_id"],
                "user_id": uid,
                "driver_id": did,
                "vendor_id": vid,
                "device_id": deid,
                "market": row["market"],
                "vertical": row["vertical"],
                "amount": row["amount"],
                "status": row["status"],
                "event_ts": row["event_ts"],
                "claim_reason": row["claim_reason"],
                "abuse_label": int(row["abuse_label"]),
                "abuse_label_weak": int(row["abuse_label_weak"]),
                "fraud_label": int(row["fraud_label"]),
                "fraud_label_source": row["fraud_label_source"],
                "strong_fraud_label": int(row["strong_fraud_label"]),
                "weak_policy_negative": int(row["weak_policy_negative"]),
                "prior_strong_fraud": int(row["prior_strong_fraud"]),
            }
        )

    feature_frame = pd.concat([pd.DataFrame(meta_rows), pd.DataFrame(feat_rows)], axis=1)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    history.to_csv(out / "history.csv", index=False)
    orders.to_csv(out / "orders.csv", index=False)
    devices.to_csv(out / "devices.csv", index=False)
    users.to_csv(out / "users.csv", index=False)
    feature_frame.to_parquet(out / "feature_frame.parquet", index=False)
    # CSV fallback if parquet engine missing in some envs — also write csv.gz
    feature_frame.to_csv(out / "feature_frame.csv.gz", index=False, compression="gzip")

    summary = {
        "out": str(out),
        "history_rows": int(len(history)),
        "orders_rows": int(len(orders)),
        "devices_rows": int(len(devices)),
        "users_rows": int(len(users)),
        "feature_frame_rows": int(len(feature_frame)),
        "history_plus_orders": int(len(history) + len(orders)),
        "label_rates": {
            "abuse": float(np.mean(abuse_label)),
            "fraud": float(np.mean(fraud_label)),
            "proven": float(np.mean([1 if s == "proven" else 0 for s in fraud_src])),
            "proxy": float(np.mean([1 if s == "proxy" else 0 for s in fraud_src])),
        },
        "seed": int(args.seed),
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
