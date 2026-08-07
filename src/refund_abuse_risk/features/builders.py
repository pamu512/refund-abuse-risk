from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from refund_abuse_risk.baselines.store import BASELINE_FEATURE_KINDS, zero_baseline_features
from refund_abuse_risk.features.policy_priors import POLICY_FEATURE_COLUMNS, policy_prior_features
from refund_abuse_risk.graph.bipartite import (
    bipartite_features_as_of,
    zero_bipartite_features,
)

# Features used to mint proxy fraud labels — excluded from fraud head when
# label_weights.proxy_rules.exclude_mint_features_from_fraud_head is true.
PROXY_MINT_FEATURE_COLUMNS: tuple[str, ...] = (
    "device_cluster_size",
    "accounts_per_device",
    "uvd_refund_lift",
    "uvd_refund_share",
    "uvd_cooccur",
    "device_risk_score",
    "is_emulator",
    "is_cloned_app",
    "is_gps_spoof",
    "customer_courier_same_device",
)

# Baseline/cohort lifts are gate-only (post-score). Kept on the row for evidence
# but omitted from head training columns so train/serve stay aligned.
GATE_ONLY_FEATURE_COLUMNS: tuple[str, ...] = (
    "user_baseline_lift",
    "user_baseline_under_rate",
    "user_baseline_elevated_streak",
    "user_is_clean_baseline",
    "uv_baseline_lift",
    "uv_baseline_under_rate",
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
    "user_cohort_lift",
    "driver_cohort_lift",
    "vendor_cohort_lift",
    "device_cohort_lift",
    "tenure_bucket_code",
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
    "uv_edge_lift",
    "uv_edge_n_orders",
    "uv_edge_elevated",
    "uv_mo_possible_collusion",
    "uv_mo_user_scatter",
    "uv_mo_elevated_uv",
]
_PLATFORM_RISK_FEATURE_COLUMNS = [
    # Device integrity (Fingerprint/SHIELD-style feed columns; 0 if absent).
    "device_risk_score",
    "is_emulator",
    "is_cloned_app",
    "is_gps_spoof",
    "is_tampered",
    "customer_courier_same_device",
    # Claim media + delivery proof.
    "claim_has_image",
    "claim_image_ai_risk",
    "claim_in_app_capture",
    "pin_required",
    "pin_verified",
    "delivery_geofence_ok",
]

_POLICY_FEATURE_COLUMNS = list(POLICY_FEATURE_COLUMNS)

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
    *_PLATFORM_RISK_FEATURE_COLUMNS,
    *_POLICY_FEATURE_COLUMNS,
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
    "uv_edge_anomaly",
    "user_bipartite_anomaly",
    "uv_edge_lift",
    "uv_edge_elevated",
    "claim_has_image",
    "claim_image_ai_risk",
    "claim_in_app_capture",
    "pin_required",
    "pin_verified",
    *_POLICY_FEATURE_COLUMNS,
]

# Fraud head: graph / device collusion signals (avoids pure rate-cap identity).
# PROXY_MINT_FEATURE_COLUMNS are omitted by default; added back only when
# exclude_mint_features_from_fraud_head is false.
FRAUD_FEATURE_COLUMNS: list[str] = [
    "device_churn_30d",
    "ud_cooccur",
    "ud_refund_lift",
    "ud_refund_share",
    "uv_cooccur",
    "uv_refund_lift",
    "uv_refund_share",
    "vd_cooccur",
    "vd_refund_lift",
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
    "uv_edge_anomaly",
    "user_bipartite_anomaly",
    "vendor_bipartite_anomaly",
    "uv_edge_lift",
    "uv_edge_n_orders",
    "uv_edge_elevated",
    "uv_mo_possible_collusion",
    "uv_mo_user_scatter",
    "uv_mo_elevated_uv",
    "is_tampered",
    "claim_image_ai_risk",
    "pin_required",
    "pin_verified",
    "delivery_geofence_ok",
    "policy_claim_window_hours",
    "policy_photo_prior",
    "policy_remedy_cash_bias",
    "claim_window_remaining_frac",
]


# Rate supports used inside apply_proxy_fraud_labels — hold out of fraud head with mint.
_PROXY_MINT_SUPPORT_COLUMNS: tuple[str, ...] = (
    "user_refund_rate_30d",
    "user_orders_30d",
)


def fraud_model_feature_columns(*, exclude_proxy_mint: bool = True) -> list[str]:
    """Fraud-head columns; mint features held out by default (honesty gate)."""
    cols = list(FRAUD_FEATURE_COLUMNS)
    if exclude_proxy_mint:
        drop = set(PROXY_MINT_FEATURE_COLUMNS) | set(_PROXY_MINT_SUPPORT_COLUMNS)
        cols = [c for c in cols if c not in drop]
    else:
        for c in PROXY_MINT_FEATURE_COLUMNS:
            if c not in cols:
                cols.append(c)
    return cols


def abuse_model_feature_columns(*, exclude_proxy_mint: bool = True) -> list[str]:
    """Abuse-head columns; drop pure proxy-mint graph/device features when excluded."""
    cols = list(ABUSE_FEATURE_COLUMNS)
    if exclude_proxy_mint:
        drop = set(PROXY_MINT_FEATURE_COLUMNS)
        cols = [c for c in cols if c not in drop]
    return cols


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
    # Prefer pre-parsed `_ts` (set by build_order_feature_frame). Never copy the full frame.
    ts = history["_ts"] if "_ts" in history.columns else pd.to_datetime(history["event_ts"], utc=True)
    mask = ts <= _as_utc_timestamp(as_of)
    if exclude_order_id:
        mask = mask & (history["order_id"].astype(str) != str(exclude_order_id))
    return history.loc[mask]


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


def _as_float(row: dict[str, Any] | pd.Series, key: str, default: float = 0.0) -> float:
    if key not in row:
        return float(default)
    val = row[key]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _zero_platform_risk_features() -> dict[str, float]:
    return {k: 0.0 for k in _PLATFORM_RISK_FEATURE_COLUMNS}


def _device_integrity(
    devices: pd.DataFrame,
    device_id: str,
    order: dict[str, Any] | None = None,
) -> dict[str, float]:
    out = {
        "device_risk_score": 0.0,
        "is_emulator": 0.0,
        "is_cloned_app": 0.0,
        "is_gps_spoof": 0.0,
        "is_tampered": 0.0,
    }
    if devices is not None and not devices.empty and device_id:
        rows = devices.loc[devices["device_id"].astype(str) == str(device_id)]
        if not rows.empty:
            row = rows.iloc[0]
            out["device_risk_score"] = max(0.0, min(100.0, _as_float(row, "device_risk_score", 0.0)))
            for key in ("is_emulator", "is_cloned_app", "is_gps_spoof", "is_tampered"):
                out[key] = 1.0 if _as_float(row, key, 0.0) >= 1 else 0.0
    # Nested adapter / order fields fill gaps (device table wins when non-zero).
    if order:
        ord_risk = max(0.0, min(100.0, _as_float(order, "device_risk_score", 0.0)))
        if out["device_risk_score"] <= 0 and ord_risk > 0:
            out["device_risk_score"] = ord_risk
        for key in ("is_emulator", "is_cloned_app", "is_gps_spoof", "is_tampered"):
            if out[key] < 1 and _as_float(order, key, 0.0) >= 1:
                out[key] = 1.0
    return out


def _claim_delivery_features(order: dict[str, Any]) -> dict[str, float]:
    return {
        "customer_courier_same_device": 1.0
        if _as_float(order, "customer_courier_same_device", 0.0) >= 1
        else 0.0,
        "claim_has_image": 1.0 if _as_float(order, "claim_has_image", 0.0) >= 1 else 0.0,
        "claim_image_ai_risk": max(0.0, min(1.0, _as_float(order, "claim_image_ai_risk", 0.0))),
        "claim_in_app_capture": 1.0 if _as_float(order, "claim_in_app_capture", 0.0) >= 1 else 0.0,
        "pin_required": 1.0 if _as_float(order, "pin_required", 0.0) >= 1 else 0.0,
        "pin_verified": 1.0 if _as_float(order, "pin_verified", 0.0) >= 1 else 0.0,
        "delivery_geofence_ok": 1.0 if _as_float(order, "delivery_geofence_ok", 0.0) >= 1 else 0.0,
    }


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
    from refund_abuse_risk.integrations.device_vision import merge_platform_signals

    # Nested vendor payloads (Fingerprint/SHIELD / vision) → flat feature columns.
    order = merge_platform_signals(
        dict(order),
        device_payload=order.get("device_intelligence") or order.get("device_payload"),
        vision_payload=order.get("claim_vision") or order.get("vision_payload"),
    )
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
        **_zero_platform_risk_features(),
        **_device_integrity(devices, device_id, order),
        **_claim_delivery_features(order),
        **policy_prior_features(order),
        # As-of UV bipartite (no future leakage).
        **bipartite_features_as_of(order, hist),
    }


def build_order_feature_frame(
    orders: pd.DataFrame,
    history: pd.DataFrame,
    devices: pd.DataFrame,
    users: pd.DataFrame | None = None,
    bipartite_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    _ = bipartite_cfg  # reserved; per-order as-of path uses defaults for now
    rows: list[dict[str, float]] = []
    meta: list[dict[str, Any]] = []
    # Parse timestamps once for as-of filters (ponytail: ceiling = still O(n) row loop).
    hist = history
    if hist is not None and not hist.empty and "_ts" not in hist.columns:
        hist = hist.copy()
        hist["_ts"] = pd.to_datetime(hist["event_ts"], utc=True)
    for _, order in orders.iterrows():
        od = order.to_dict()
        feat = build_order_feature_row(od, hist, devices, users=users)
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
                # Account prior from dispositions (safe for hard gates).
                "prior_strong_fraud": od.get("prior_strong_fraud", 0),
            }
        )
    feat_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    meta_df = pd.DataFrame(meta)
    return pd.concat([meta_df, feat_df], axis=1)
