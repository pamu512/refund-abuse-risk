from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from refund_abuse_risk.baselines.store import BASELINE_FEATURE_KINDS, zero_baseline_features
from refund_abuse_risk.graph.bipartite import (
    bipartite_feature_lookups,
    features_for_order as bipartite_features_for_order,
    score_uv_bipartite,
    zero_bipartite_features,
)

_BASELINE_FEATURE_COLUMNS: list[str] = []
for _kind in BASELINE_FEATURE_KINDS:
    _BASELINE_FEATURE_COLUMNS.extend(
        [
            f"{_kind}_baseline_lift",
            f"{_kind}_baseline_under_rate",
            f"{_kind}_baseline_elevated_streak",
            f"{_kind}_is_clean_baseline",
        ]
    )
_BASELINE_FEATURE_COLUMNS.extend(
    [
        "user_cohort_lift",
        "driver_cohort_lift",
        "vendor_cohort_lift",
        "device_cohort_lift",
        "tenure_bucket_code",
    ]
)
_BIPARTITE_FEATURE_COLUMNS = [
    "uv_edge_anomaly",
    "user_bipartite_anomaly",
    "vendor_bipartite_anomaly",
]

# Shared columns available on every scored row.
FEATURE_COLUMNS: list[str] = [
    "order_amount",
    "user_orders_30d",
    "user_refund_count_7d",
    "user_refund_count_30d",
    "user_refund_rate_30d",
    "user_refund_gmv_pct_30d",
    "user_days_since_signup",
    "user_lifetime_orders",
    "user_lifetime_refund_count",
    "user_lifetime_gmv",
    "user_lifetime_refund_gmv",
    "user_ltv_net",
    "user_refund_to_ltv_ratio",
    "related_account_count",
    "related_refund_count_30d",
    "related_refund_gmv_30d",
    "related_max_refund_rate_30d",
    "related_max_orders_30d",
    "combined_refund_count_30d",
    "combined_refund_gmv_30d",
    "driver_orders_30d",
    "driver_refund_count_30d",
    "driver_refund_rate_30d",
    "vendor_orders_30d",
    "vendor_refund_count_30d",
    "vendor_refund_rate_30d",
    "vendor_refund_gmv_pct_30d",
    "accounts_per_device",
    "devices_per_account",
    "device_cluster_size",
    "device_churn_30d",
    "ud_cooccur",
    "ud_refund_lift",
    "ud_refund_share",
    "uv_cooccur",
    "uv_refund_lift",
    "uv_refund_share",
    "vd_cooccur",
    "vd_refund_lift",
    "uvd_cooccur",
    "uvd_refund_lift",
    "uvd_refund_share",
    "reason_repeat_rate_30d",
    "is_food",
    "is_qcommerce",
    "claim_reason_missing_item",
    "claim_reason_quality",
    "claim_reason_wrong_order",
    "order_status_delivered",
    *_BASELINE_FEATURE_COLUMNS,
    *_BIPARTITE_FEATURE_COLUMNS,
]

# Abuse head: behavioral rate / claim / GMV / tenure-LTV patterns.
ABUSE_FEATURE_COLUMNS: list[str] = [
    "order_amount",
    "user_orders_30d",
    "user_refund_count_7d",
    "user_refund_count_30d",
    "user_refund_rate_30d",
    "user_refund_gmv_pct_30d",
    "user_days_since_signup",
    "user_lifetime_orders",
    "user_lifetime_refund_count",
    "user_ltv_net",
    "user_refund_to_ltv_ratio",
    "related_account_count",
    "related_refund_count_30d",
    "related_refund_gmv_30d",
    "related_max_refund_rate_30d",
    "combined_refund_count_30d",
    "combined_refund_gmv_30d",
    "driver_refund_rate_30d",
    "vendor_refund_rate_30d",
    "vendor_refund_gmv_pct_30d",
    "reason_repeat_rate_30d",
    "devices_per_account",
    "uv_refund_share",
    "is_food",
    "is_qcommerce",
    "claim_reason_missing_item",
    "claim_reason_quality",
    "claim_reason_wrong_order",
    "order_status_delivered",
    "user_baseline_lift",
    "user_baseline_under_rate",
    "user_baseline_elevated_streak",
    "user_is_clean_baseline",
    "uv_baseline_lift",
    "uv_baseline_under_rate",
    "user_cohort_lift",
    "tenure_bucket_code",
    "uv_edge_anomaly",
    "user_bipartite_anomaly",
]

# Fraud head: graph / device collusion signals (avoids pure rate-cap identity).
FRAUD_FEATURE_COLUMNS: list[str] = [
    "accounts_per_device",
    "device_cluster_size",
    "device_churn_30d",
    "ud_cooccur",
    "ud_refund_lift",
    "ud_refund_share",
    "uv_cooccur",
    "uv_refund_lift",
    "uv_refund_share",
    "vd_cooccur",
    "vd_refund_lift",
    "uvd_cooccur",
    "uvd_refund_lift",
    "uvd_refund_share",
    "related_account_count",
    "related_refund_count_30d",
    "combined_refund_count_30d",
    "combined_refund_gmv_30d",
    "driver_refund_rate_30d",
    "vendor_refund_rate_30d",
    "user_refund_rate_30d",
    "user_orders_30d",
    "is_food",
    "is_qcommerce",
    "driver_baseline_lift",
    "driver_baseline_under_rate",
    "driver_baseline_elevated_streak",
    "driver_is_clean_baseline",
    "vendor_baseline_lift",
    "vendor_baseline_under_rate",
    "vendor_baseline_elevated_streak",
    "vendor_is_clean_baseline",
    "device_baseline_lift",
    "device_baseline_under_rate",
    "device_is_clean_baseline",
    "ud_baseline_lift",
    "uvd_baseline_lift",
    "uvd_baseline_under_rate",
    "uvd_baseline_elevated_streak",
    "uvd_is_clean_baseline",
    "driver_cohort_lift",
    "vendor_cohort_lift",
    "device_cohort_lift",
    "uv_edge_anomaly",
    "user_bipartite_anomaly",
    "vendor_bipartite_anomaly",
]


def _parse_ts(value: Any) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _window_mask(ts: pd.Series, as_of: datetime, days: int) -> pd.Series:
    as_of_ts = _as_utc_timestamp(as_of)
    start = as_of_ts - pd.Timedelta(days=days)
    parsed = pd.to_datetime(ts, utc=True)
    return (parsed <= as_of_ts) & (parsed >= start)


def _history_as_of(
    history: pd.DataFrame,
    as_of: datetime,
    *,
    exclude_order_id: str | None = None,
) -> pd.DataFrame:
    """Point-in-time history: events at/before as_of, excluding the scored order."""
    if history.empty:
        return history
    out = history.copy()
    if exclude_order_id:
        out = out[out["order_id"].astype(str) != str(exclude_order_id)]
    if out.empty:
        return out
    mask = pd.to_datetime(out["event_ts"], utc=True) <= _as_utc_timestamp(as_of)
    return out.loc[mask]


def _shrink_rate(refunds: float, orders: float, baseline: float, prior_strength: float = 3.0) -> float:
    """Shrink empirical rate toward baseline when support is thin."""
    if orders <= 0:
        return 0.0
    return (refunds + prior_strength * baseline) / (orders + prior_strength)


def _smoothed_lift(refunds: float, n: float, baseline: float, prior_strength: float = 2.0) -> float:
    if n <= 0:
        return 1.0
    rate = (refunds + prior_strength * baseline) / (n + prior_strength)
    return rate / baseline if baseline > 0 else 1.0


def _entity_stats(
    history: pd.DataFrame,
    entity_col: str,
    entity_id: str,
    as_of: datetime,
    baseline_rate: float,
) -> dict[str, float]:
    empty = {
        "orders_30d": 0.0,
        "refund_count_7d": 0.0,
        "refund_count_30d": 0.0,
        "refund_rate_30d": 0.0,
        "refund_gmv_pct_30d": 0.0,
    }
    if history.empty or not entity_id:
        return empty
    subset = history[history[entity_col].astype(str) == str(entity_id)]
    if subset.empty:
        return empty
    m7 = subset[_window_mask(subset["event_ts"], as_of, 7)]
    m30 = subset[_window_mask(subset["event_ts"], as_of, 30)]
    orders_30 = float(len(m30))
    refunds_7 = float(m7["is_refund"].sum()) if len(m7) else 0.0
    refunds_30 = float(m30["is_refund"].sum()) if len(m30) else 0.0
    gmv_30 = float(m30["amount"].sum()) if len(m30) else 0.0
    refund_gmv_30 = float(m30.loc[m30["is_refund"] == 1, "amount"].sum()) if len(m30) else 0.0
    # GMV pct also shrunk when few orders.
    gmv_pct = (refund_gmv_30 / gmv_30) if gmv_30 else 0.0
    if orders_30 < 3 and gmv_30:
        gmv_pct = (refund_gmv_30 + baseline_rate * gmv_30 * 0.5) / (gmv_30 + 0.5 * gmv_30)
    return {
        "orders_30d": orders_30,
        "refund_count_7d": refunds_7,
        "refund_count_30d": refunds_30,
        "refund_rate_30d": _shrink_rate(refunds_30, orders_30, baseline_rate),
        "refund_gmv_pct_30d": float(gmv_pct),
    }


def _link_stats(
    history: pd.DataFrame,
    cols: list[str],
    ids: list[str],
    as_of: datetime,
    baseline_rate: float,
    user_refunds_30d: float,
) -> dict[str, float]:
    if history.empty or any(not x for x in ids):
        return {"cooccur": 0.0, "refund_lift": 1.0, "refund_share": 0.0}
    mask = pd.Series(True, index=history.index)
    for col, eid in zip(cols, ids, strict=True):
        mask &= history[col].astype(str) == str(eid)
    subset = history.loc[mask]
    subset = subset[_window_mask(subset["event_ts"], as_of, 90)]
    n = float(len(subset))
    if n == 0:
        return {"cooccur": 0.0, "refund_lift": 1.0, "refund_share": 0.0}
    refunds = float(subset["is_refund"].sum())
    lift = _smoothed_lift(refunds, n, baseline_rate)
    # Share of this user's refunds that occur on this link (collusion concentration).
    share = (refunds / user_refunds_30d) if user_refunds_30d > 0 else 0.0
    return {"cooccur": n, "refund_lift": lift, "refund_share": float(min(share, 1.0))}


def _device_stats(
    devices: pd.DataFrame,
    user_id: str,
    device_id: str,
) -> dict[str, float]:
    if devices.empty:
        return {
            "accounts_per_device": 1.0,
            "devices_per_account": 1.0,
            "device_cluster_size": 1.0,
            "device_churn_30d": 0.0,
        }
    d = devices.copy()
    d["user_id"] = d["user_id"].astype(str)
    d["device_id"] = d["device_id"].astype(str)
    accounts = d.loc[d["device_id"] == str(device_id), "user_id"].nunique() if device_id else 1
    user_devices = d.loc[d["user_id"] == str(user_id), "device_id"].nunique() if user_id else 1
    cluster_id = None
    if device_id and "cluster_id" in d.columns:
        rows = d.loc[d["device_id"] == str(device_id), "cluster_id"]
        cluster_id = rows.iloc[0] if len(rows) else None
    if cluster_id is not None:
        cluster_size = int(d.loc[d["cluster_id"] == cluster_id, "user_id"].nunique())
    else:
        cluster_size = max(int(accounts), 1)
    churn = float(user_devices)
    return {
        "accounts_per_device": float(max(accounts, 1)),
        "devices_per_account": float(max(user_devices, 1)),
        "device_cluster_size": float(max(cluster_size, 1)),
        "device_churn_30d": churn,
    }


def _reason_repeat_rate(history: pd.DataFrame, user_id: str, as_of: datetime) -> float:
    if history.empty or not user_id or "claim_reason" not in history.columns:
        return 0.0
    subset = history[history["user_id"].astype(str) == str(user_id)]
    subset = subset[_window_mask(subset["event_ts"], as_of, 30)]
    refunds = subset[subset["is_refund"] == 1]
    if len(refunds) < 2:
        return 0.0
    reasons = refunds["claim_reason"].astype(str).str.lower().replace({"": "unknown", "nan": "unknown"})
    top = float(reasons.value_counts(normalize=True).iloc[0])
    return top


def _signup_ts(
    user_id: str,
    users: pd.DataFrame | None,
    history: pd.DataFrame,
    order: dict[str, Any],
) -> pd.Timestamp | None:
    if order.get("signup_ts"):
        return _as_utc_timestamp(order["signup_ts"])
    if users is not None and not users.empty and user_id:
        rows = users.loc[users["user_id"].astype(str) == str(user_id)]
        if len(rows) and pd.notna(rows.iloc[0].get("signup_ts")):
            return _as_utc_timestamp(rows.iloc[0]["signup_ts"])
    if not history.empty and user_id:
        subset = history.loc[history["user_id"].astype(str) == str(user_id)]
        if len(subset):
            return pd.to_datetime(subset["event_ts"], utc=True).min()
    return None


def _related_user_ids(
    history: pd.DataFrame,
    devices: pd.DataFrame,
    *,
    user_id: str,
    device_id: str,
    driver_id: str,
    vendor_id: str,
) -> set[str]:
    """Related accounts: device/cluster + UVD triad peers (same driver AND vendor)."""
    related: set[str] = set()
    if not user_id:
        return related

    if devices is not None and not devices.empty:
        d = devices.copy()
        d["user_id"] = d["user_id"].astype(str)
        d["device_id"] = d["device_id"].astype(str)
        if device_id:
            related |= set(d.loc[d["device_id"] == str(device_id), "user_id"])
            if "cluster_id" in d.columns:
                clusters = d.loc[d["device_id"] == str(device_id), "cluster_id"]
                if len(clusters):
                    cid = clusters.iloc[0]
                    related |= set(d.loc[d["cluster_id"] == cid, "user_id"].astype(str))

    # Collusion ring peers only — same driver+vendor avoids popular-driver fanout.
    if history is not None and not history.empty and driver_id and vendor_id:
        h = history
        mask = (h["driver_id"].astype(str) == str(driver_id)) & (
            h["vendor_id"].astype(str) == str(vendor_id)
        )
        related |= set(h.loc[mask, "user_id"].astype(str))

    related.discard(str(user_id))
    related.discard("")
    return related


def _related_refund_stats(
    history: pd.DataFrame,
    related_ids: set[str],
    as_of: datetime,
    baseline_rate: float,
) -> dict[str, float]:
    empty = {
        "related_account_count": 0.0,
        "related_refund_count_30d": 0.0,
        "related_refund_gmv_30d": 0.0,
        "related_max_refund_rate_30d": 0.0,
        "related_max_orders_30d": 0.0,
    }
    if history.empty or not related_ids:
        return empty
    subset = history[history["user_id"].astype(str).isin(related_ids)]
    if subset.empty:
        return {**empty, "related_account_count": float(len(related_ids))}
    m30 = subset[_window_mask(subset["event_ts"], as_of, 30)]
    refund_count = float(m30["is_refund"].sum()) if len(m30) else 0.0
    refund_gmv = float(m30.loc[m30["is_refund"] == 1, "amount"].sum()) if len(m30) else 0.0
    max_rate = 0.0
    max_orders = 0.0
    for rid in related_ids:
        stats = _entity_stats(history, "user_id", rid, as_of, baseline_rate)
        max_rate = max(max_rate, float(stats["refund_rate_30d"]))
        max_orders = max(max_orders, float(stats["orders_30d"]))
    return {
        "related_account_count": float(len(related_ids)),
        "related_refund_count_30d": refund_count,
        "related_refund_gmv_30d": refund_gmv,
        "related_max_refund_rate_30d": max_rate,
        "related_max_orders_30d": max_orders,
    }


def _user_lifetime_stats(
    history: pd.DataFrame,
    user_id: str,
    as_of: datetime,
    signup: pd.Timestamp | None,
) -> dict[str, float]:
    empty = {
        "days_since_signup": 0.0,
        "lifetime_orders": 0.0,
        "lifetime_refund_count": 0.0,
        "lifetime_gmv": 0.0,
        "lifetime_refund_gmv": 0.0,
        "ltv_net": 0.0,
        "refund_to_ltv_ratio": 0.0,
    }
    as_of_ts = _as_utc_timestamp(as_of)
    if signup is not None:
        days = max((as_of_ts - signup).total_seconds() / 86400.0, 0.0)
    else:
        days = 0.0
    if history.empty or not user_id:
        empty["days_since_signup"] = days
        return empty
    subset = history[history["user_id"].astype(str) == str(user_id)]
    if subset.empty:
        empty["days_since_signup"] = days
        return empty
    lifetime_orders = float(len(subset))
    lifetime_refund_count = float(subset["is_refund"].sum())
    lifetime_gmv = float(subset["amount"].sum())
    lifetime_refund_gmv = float(subset.loc[subset["is_refund"] == 1, "amount"].sum())
    ltv_net = lifetime_gmv - lifetime_refund_gmv
    ratio = (lifetime_refund_gmv / lifetime_gmv) if lifetime_gmv > 0 else 0.0
    if signup is None and lifetime_orders:
        first = pd.to_datetime(subset["event_ts"], utc=True).min()
        days = max((as_of_ts - first).total_seconds() / 86400.0, 0.0)
    return {
        "days_since_signup": float(days),
        "lifetime_orders": lifetime_orders,
        "lifetime_refund_count": lifetime_refund_count,
        "lifetime_gmv": lifetime_gmv,
        "lifetime_refund_gmv": lifetime_refund_gmv,
        "ltv_net": float(ltv_net),
        "refund_to_ltv_ratio": float(ratio),
    }


def build_order_feature_row(
    order: dict[str, Any],
    history: pd.DataFrame,
    devices: pd.DataFrame,
    users: pd.DataFrame | None = None,
) -> dict[str, float]:
    as_of = _parse_ts(order.get("event_ts") or order.get("order_ts"))
    order_id = str(order.get("order_id", "") or "")
    hist = _history_as_of(history, as_of, exclude_order_id=order_id or None)

    user_id = str(order.get("user_id", ""))
    driver_id = str(order.get("driver_id", ""))
    vendor_id = str(order.get("vendor_id", ""))
    device_id = str(order.get("device_id", ""))

    baseline_rate = float(hist["is_refund"].mean()) if len(hist) else 0.05
    baseline_rate = float(min(max(baseline_rate, 0.02), 0.5))

    signup = _signup_ts(user_id, users, hist, order)
    life = _user_lifetime_stats(hist, user_id, as_of, signup)
    user = _entity_stats(hist, "user_id", user_id, as_of, baseline_rate)
    driver = _entity_stats(hist, "driver_id", driver_id, as_of, baseline_rate)
    vendor = _entity_stats(hist, "vendor_id", vendor_id, as_of, baseline_rate)
    device = _device_stats(devices, user_id, device_id)
    related_ids = _related_user_ids(
        hist,
        devices,
        user_id=user_id,
        device_id=device_id,
        driver_id=driver_id,
        vendor_id=vendor_id,
    )
    related = _related_refund_stats(hist, related_ids, as_of, baseline_rate)
    user_refund_gmv_30d = 0.0
    if not hist.empty and user_id:
        u30 = hist[hist["user_id"].astype(str) == user_id]
        u30 = u30[_window_mask(u30["event_ts"], as_of, 30)]
        if len(u30):
            user_refund_gmv_30d = float(u30.loc[u30["is_refund"] == 1, "amount"].sum())
    combined_refund_count = user["refund_count_30d"] + related["related_refund_count_30d"]
    combined_refund_gmv = user_refund_gmv_30d + related["related_refund_gmv_30d"]
    ud = _link_stats(
        hist, ["user_id", "driver_id"], [user_id, driver_id], as_of, baseline_rate, user["refund_count_30d"]
    )
    uv = _link_stats(
        hist, ["user_id", "vendor_id"], [user_id, vendor_id], as_of, baseline_rate, user["refund_count_30d"]
    )
    vd = _link_stats(
        hist, ["vendor_id", "driver_id"], [vendor_id, driver_id], as_of, baseline_rate, user["refund_count_30d"]
    )
    uvd = _link_stats(
        hist,
        ["user_id", "vendor_id", "driver_id"],
        [user_id, vendor_id, driver_id],
        as_of,
        baseline_rate,
        user["refund_count_30d"],
    )

    vertical = str(order.get("vertical", "food")).lower()
    reason = str(order.get("claim_reason", "") or "").lower()
    status = str(order.get("status", "") or "").lower()
    amount = float(order.get("amount", 0.0) or 0.0)

    return {
        "order_amount": amount,
        "user_orders_30d": user["orders_30d"],
        "user_refund_count_7d": user["refund_count_7d"],
        "user_refund_count_30d": user["refund_count_30d"],
        "user_refund_rate_30d": user["refund_rate_30d"],
        "user_refund_gmv_pct_30d": user["refund_gmv_pct_30d"],
        "user_days_since_signup": life["days_since_signup"],
        "user_lifetime_orders": life["lifetime_orders"],
        "user_lifetime_refund_count": life["lifetime_refund_count"],
        "user_lifetime_gmv": life["lifetime_gmv"],
        "user_lifetime_refund_gmv": life["lifetime_refund_gmv"],
        "user_ltv_net": life["ltv_net"],
        "user_refund_to_ltv_ratio": life["refund_to_ltv_ratio"],
        "related_account_count": related["related_account_count"],
        "related_refund_count_30d": related["related_refund_count_30d"],
        "related_refund_gmv_30d": related["related_refund_gmv_30d"],
        "related_max_refund_rate_30d": related["related_max_refund_rate_30d"],
        "related_max_orders_30d": related["related_max_orders_30d"],
        "combined_refund_count_30d": float(combined_refund_count),
        "combined_refund_gmv_30d": float(combined_refund_gmv),
        "driver_orders_30d": driver["orders_30d"],
        "driver_refund_count_30d": driver["refund_count_30d"],
        "driver_refund_rate_30d": driver["refund_rate_30d"],
        "vendor_orders_30d": vendor["orders_30d"],
        "vendor_refund_count_30d": vendor["refund_count_30d"],
        "vendor_refund_rate_30d": vendor["refund_rate_30d"],
        "vendor_refund_gmv_pct_30d": vendor["refund_gmv_pct_30d"],
        "accounts_per_device": device["accounts_per_device"],
        "devices_per_account": device["devices_per_account"],
        "device_cluster_size": device["device_cluster_size"],
        "device_churn_30d": device["device_churn_30d"],
        "ud_cooccur": ud["cooccur"],
        "ud_refund_lift": ud["refund_lift"],
        "ud_refund_share": ud["refund_share"],
        "uv_cooccur": uv["cooccur"],
        "uv_refund_lift": uv["refund_lift"],
        "uv_refund_share": uv["refund_share"],
        "vd_cooccur": vd["cooccur"],
        "vd_refund_lift": vd["refund_lift"],
        "uvd_cooccur": uvd["cooccur"],
        "uvd_refund_lift": uvd["refund_lift"],
        "uvd_refund_share": uvd["refund_share"],
        "reason_repeat_rate_30d": _reason_repeat_rate(hist, user_id, as_of),
        "is_food": 1.0 if vertical == "food" else 0.0,
        "is_qcommerce": 1.0 if vertical in {"qcommerce", "q_commerce", "quick_commerce"} else 0.0,
        "claim_reason_missing_item": 1.0 if "missing" in reason else 0.0,
        "claim_reason_quality": 1.0 if "quality" in reason or "spoil" in reason else 0.0,
        "claim_reason_wrong_order": 1.0 if "wrong" in reason else 0.0,
        "order_status_delivered": 1.0 if status == "delivered" else 0.0,
        **zero_baseline_features(),
        **zero_bipartite_features(),
    }


def build_order_feature_frame(
    orders: pd.DataFrame,
    history: pd.DataFrame,
    devices: pd.DataFrame,
    users: pd.DataFrame | None = None,
    bipartite_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    edges, nodes = score_uv_bipartite(history, bipartite_cfg)
    edge_map, node_map = bipartite_feature_lookups(edges, nodes)
    rows: list[dict[str, float]] = []
    meta: list[dict[str, Any]] = []
    for _, order in orders.iterrows():
        od = order.to_dict()
        feat = build_order_feature_row(od, history, devices, users=users)
        feat.update(bipartite_features_for_order(od, edge_map=edge_map, node_map=node_map))
        rows.append(feat)
        meta.append(
            {
                "order_id": od.get("order_id"),
                "user_id": od.get("user_id"),
                "driver_id": od.get("driver_id"),
                "vendor_id": od.get("vendor_id"),
                "device_id": od.get("device_id"),
                "market": od.get("market"),
                "vertical": od.get("vertical"),
                "amount": od.get("amount"),
                "status": od.get("status"),
                "event_ts": od.get("event_ts"),
                "claim_reason": od.get("claim_reason"),
                "abuse_label": od.get("abuse_label", 0),
                "abuse_label_weak": od.get("abuse_label_weak", 0),
                "fraud_label": od.get("fraud_label", 0),
                "fraud_label_source": od.get("fraud_label_source", ""),
                "strong_fraud_label": od.get("strong_fraud_label", 0),
                "weak_policy_negative": od.get("weak_policy_negative", 0),
            }
        )
    feat_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    meta_df = pd.DataFrame(meta)
    return pd.concat([meta_df, feat_df], axis=1)
