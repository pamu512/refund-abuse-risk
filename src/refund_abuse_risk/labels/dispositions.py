"""Closed-loop: refund/claim dispositions → training labels."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

LABEL_COLS = (
    "abuse_label",
    "abuse_label_weak",
    "fraud_label",
    "fraud_label_source",
    "strong_fraud_label",
    "weak_policy_negative",
)


def apply_dispositions_to_orders(
    orders: pd.DataFrame,
    dispositions: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Overlay disposition outcomes onto order label columns.

    Expected disposition columns: order_id, disposition [, disposition_ts].
    Lag policy: disposition_ts >= order.event_ts + lag_days (default 7).
    Optional as_of: only dispositions with disposition_ts <= as_of are used.
    """
    out = orders.copy()
    for col in LABEL_COLS:
        if col not in out.columns:
            out[col] = 0 if col != "fraud_label_source" else ""
    if dispositions is None or dispositions.empty or "order_id" not in dispositions.columns:
        out.attrs["dispositions_applied"] = 0
        out.attrs["dispositions_skipped_lag"] = 0
        out.attrs["dispositions_unknown"] = []
        return out
    if "disposition" not in dispositions.columns:
        raise ValueError("dispositions require a 'disposition' column")

    outcomes = cfg.get("outcomes") or {}
    lag_days = int(cfg.get("lag_days", 0) or 0)
    require_event = bool(cfg.get("require_order_event_ts", True))
    disp = dispositions.copy()
    disp["order_id"] = disp["order_id"].astype(str)
    disp["disposition"] = disp["disposition"].astype(str).str.strip().str.lower()
    if "disposition_ts" in disp.columns:
        disp["disp_ts"] = pd.to_datetime(disp["disposition_ts"], utc=True, errors="coerce")
    else:
        disp["disp_ts"] = pd.NaT

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize("UTC")
        else:
            as_of_ts = as_of_ts.tz_convert("UTC")
        disp = disp[disp["disp_ts"].isna() | (disp["disp_ts"] <= as_of_ts)]

    if cfg.get("prefer_latest", True):
        disp = disp.sort_values("disp_ts", kind="mergesort")
        disp = disp.drop_duplicates(subset=["order_id"], keep="last")
    else:
        disp = disp.drop_duplicates(subset=["order_id"], keep="last")

    out["order_id"] = out["order_id"].astype(str)
    event_map: dict[str, pd.Timestamp] = {}
    if "event_ts" in out.columns:
        for row in out.itertuples(index=False):
            ts = pd.to_datetime(getattr(row, "event_ts", None), utc=True, errors="coerce")
            event_map[str(row.order_id)] = ts

    applied = 0
    skipped_lag = 0
    unknown: set[str] = set()
    for row in disp.itertuples(index=False):
        key = str(row.disposition)
        patch = outcomes.get(key)
        if patch is None:
            for ok, ov in outcomes.items():
                if str(ok).lower() == key:
                    patch = ov
                    break
        if patch is None:
            unknown.add(key)
            continue
        oid = str(row.order_id)
        mask = out["order_id"] == oid
        if not mask.any():
            continue

        disp_ts = getattr(row, "disp_ts", pd.NaT)
        event_ts = event_map.get(oid, pd.NaT)
        if lag_days > 0:
            if require_event and pd.isna(event_ts):
                skipped_lag += 1
                continue
            if not pd.isna(event_ts) and not pd.isna(disp_ts):
                earliest = event_ts + timedelta(days=lag_days)
                if disp_ts < earliest:
                    skipped_lag += 1
                    continue
            elif require_event and pd.isna(disp_ts):
                skipped_lag += 1
                continue

        for col, val in patch.items():
            if col not in LABEL_COLS:
                continue
            out.loc[mask, col] = val
        applied += 1

    out.attrs["dispositions_applied"] = applied
    out.attrs["dispositions_skipped_lag"] = skipped_lag
    out.attrs["dispositions_unknown"] = sorted(unknown)
    return out
