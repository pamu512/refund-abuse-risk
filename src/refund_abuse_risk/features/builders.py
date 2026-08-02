from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "order_amount",
    "user_refund_count_7d",
    "user_refund_count_30d",
    "user_refund_rate_30d",
    "user_refund_gmv_pct_30d",
    "driver_refund_count_30d",
    "driver_refund_rate_30d",
    "vendor_refund_count_30d",
    "vendor_refund_rate_30d",
    "vendor_refund_gmv_pct_30d",
    "accounts_per_device",
    "devices_per_account",
    "device_cluster_size",
    "device_churn_30d",
    "ud_cooccur",
    "ud_refund_lift",
    "uv_cooccur",
    "uv_refund_lift",
    "vd_cooccur",
    "vd_refund_lift",
    "uvd_cooccur",
    "uvd_refund_lift",
    "is_food",
    "is_qcommerce",
    "claim_reason_missing_item",
    "claim_reason_quality",
    "claim_reason_wrong_order",
    "order_status_delivered",
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


def _entity_stats(
    history: pd.DataFrame,
    entity_col: str,
    entity_id: str,
    as_of: datetime,
) -> dict[str, float]:
    if history.empty or entity_id is None or entity_id == "":
        return {
            "refund_count_7d": 0.0,
            "refund_count_30d": 0.0,
            "refund_rate_30d": 0.0,
            "refund_gmv_pct_30d": 0.0,
        }
    subset = history[history[entity_col].astype(str) == str(entity_id)]
    m7 = subset[_window_mask(subset["event_ts"], as_of, 7)]
    m30 = subset[_window_mask(subset["event_ts"], as_of, 30)]
    orders_30 = float(len(m30))
    refunds_7 = float(m7["is_refund"].sum()) if len(m7) else 0.0
    refunds_30 = float(m30["is_refund"].sum()) if len(m30) else 0.0
    gmv_30 = float(m30["amount"].sum()) if len(m30) else 0.0
    refund_gmv_30 = float(m30.loc[m30["is_refund"] == 1, "amount"].sum()) if len(m30) else 0.0
    return {
        "refund_count_7d": refunds_7,
        "refund_count_30d": refunds_30,
        "refund_rate_30d": (refunds_30 / orders_30) if orders_30 else 0.0,
        "refund_gmv_pct_30d": (refund_gmv_30 / gmv_30) if gmv_30 else 0.0,
    }


def _link_stats(
    history: pd.DataFrame,
    cols: list[str],
    ids: list[str],
    as_of: datetime,
    baseline_rate: float,
) -> dict[str, float]:
    if history.empty or any(not x for x in ids):
        return {"cooccur": 0.0, "refund_lift": 1.0}
    mask = pd.Series(True, index=history.index)
    for col, eid in zip(cols, ids, strict=True):
        mask &= history[col].astype(str) == str(eid)
    subset = history.loc[mask]
    subset = subset[_window_mask(subset["event_ts"], as_of, 90)]
    n = float(len(subset))
    if n == 0:
        return {"cooccur": 0.0, "refund_lift": 1.0}
    rate = float(subset["is_refund"].mean())
    lift = (rate / baseline_rate) if baseline_rate > 0 else (rate * 10.0 if rate else 1.0)
    return {"cooccur": n, "refund_lift": lift}


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
    churn = 0.0
    if "last_seen_ts" in d.columns and user_id:
        user_rows = d.loc[d["user_id"] == str(user_id)]
        if len(user_rows):
            churn = float(user_rows["device_id"].nunique())
    return {
        "accounts_per_device": float(max(accounts, 1)),
        "devices_per_account": float(max(user_devices, 1)),
        "device_cluster_size": float(max(cluster_size, 1)),
        "device_churn_30d": churn,
    }


def build_order_feature_row(
    order: dict[str, Any],
    history: pd.DataFrame,
    devices: pd.DataFrame,
) -> dict[str, float]:
    as_of = _parse_ts(order.get("event_ts") or order.get("order_ts"))
    user_id = str(order.get("user_id", ""))
    driver_id = str(order.get("driver_id", ""))
    vendor_id = str(order.get("vendor_id", ""))
    device_id = str(order.get("device_id", ""))

    baseline_rate = float(history["is_refund"].mean()) if len(history) else 0.05
    baseline_rate = max(baseline_rate, 0.01)

    user = _entity_stats(history, "user_id", user_id, as_of)
    driver = _entity_stats(history, "driver_id", driver_id, as_of)
    vendor = _entity_stats(history, "vendor_id", vendor_id, as_of)
    device = _device_stats(devices, user_id, device_id)
    ud = _link_stats(history, ["user_id", "driver_id"], [user_id, driver_id], as_of, baseline_rate)
    uv = _link_stats(history, ["user_id", "vendor_id"], [user_id, vendor_id], as_of, baseline_rate)
    vd = _link_stats(history, ["vendor_id", "driver_id"], [vendor_id, driver_id], as_of, baseline_rate)
    uvd = _link_stats(
        history,
        ["user_id", "vendor_id", "driver_id"],
        [user_id, vendor_id, driver_id],
        as_of,
        baseline_rate,
    )

    vertical = str(order.get("vertical", "food")).lower()
    reason = str(order.get("claim_reason", "") or "").lower()
    status = str(order.get("status", "") or "").lower()
    amount = float(order.get("amount", 0.0) or 0.0)

    return {
        "order_amount": amount,
        "user_refund_count_7d": user["refund_count_7d"],
        "user_refund_count_30d": user["refund_count_30d"],
        "user_refund_rate_30d": user["refund_rate_30d"],
        "user_refund_gmv_pct_30d": user["refund_gmv_pct_30d"],
        "driver_refund_count_30d": driver["refund_count_30d"],
        "driver_refund_rate_30d": driver["refund_rate_30d"],
        "vendor_refund_count_30d": vendor["refund_count_30d"],
        "vendor_refund_rate_30d": vendor["refund_rate_30d"],
        "vendor_refund_gmv_pct_30d": vendor["refund_gmv_pct_30d"],
        "accounts_per_device": device["accounts_per_device"],
        "devices_per_account": device["devices_per_account"],
        "device_cluster_size": device["device_cluster_size"],
        "device_churn_30d": device["device_churn_30d"],
        "ud_cooccur": ud["cooccur"],
        "ud_refund_lift": ud["refund_lift"],
        "uv_cooccur": uv["cooccur"],
        "uv_refund_lift": uv["refund_lift"],
        "vd_cooccur": vd["cooccur"],
        "vd_refund_lift": vd["refund_lift"],
        "uvd_cooccur": uvd["cooccur"],
        "uvd_refund_lift": uvd["refund_lift"],
        "is_food": 1.0 if vertical == "food" else 0.0,
        "is_qcommerce": 1.0 if vertical in {"qcommerce", "q_commerce", "quick_commerce"} else 0.0,
        "claim_reason_missing_item": 1.0 if "missing" in reason else 0.0,
        "claim_reason_quality": 1.0 if "quality" in reason or "spoil" in reason else 0.0,
        "claim_reason_wrong_order": 1.0 if "wrong" in reason else 0.0,
        "order_status_delivered": 1.0 if status == "delivered" else 0.0,
    }


def build_order_feature_frame(
    orders: pd.DataFrame,
    history: pd.DataFrame,
    devices: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    meta: list[dict[str, Any]] = []
    for _, order in orders.iterrows():
        od = order.to_dict()
        feat = build_order_feature_row(od, history, devices)
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
                "claim_reason": od.get("claim_reason"),
                "abuse_label": od.get("abuse_label", 0),
                "fraud_label": od.get("fraud_label", 0),
                "fraud_label_source": od.get("fraud_label_source", ""),
                "strong_fraud_label": od.get("strong_fraud_label", 0),
            }
        )
    feat_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    meta_df = pd.DataFrame(meta)
    return pd.concat([meta_df, feat_df], axis=1)
