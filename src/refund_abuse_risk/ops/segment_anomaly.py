"""Segment refund-rate anomaly detection → proposed rules (never auto-enforce).

DoorDash/RADAR-shaped: daily time series per market×vertical, moving-window
z-score with a gap between baseline and test day. Output is advisory only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


def _refund_flag(frame: pd.DataFrame) -> pd.Series:
    if "is_refund" in frame.columns:
        return frame["is_refund"].fillna(0).astype(float).gt(0)
    if "claim_reason" in frame.columns:
        return frame["claim_reason"].fillna("").astype(str).str.len().gt(0)
    raise ValueError("orders need is_refund or claim_reason for refund-rate anomaly")


def build_daily_refund_series(
    orders: pd.DataFrame,
    *,
    min_orders_per_day: int = 5,
) -> pd.DataFrame:
    """Return rows: segment, day, n, refund_rate."""
    if orders is None or len(orders) == 0:
        return pd.DataFrame(columns=["segment", "day", "n", "refund_rate"])
    df = orders.copy()
    if "event_ts" not in df.columns and "order_ts" not in df.columns:
        raise ValueError("orders need event_ts or order_ts")
    ts_col = "event_ts" if "event_ts" in df.columns else "order_ts"
    df["_day"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.floor("D")
    df = df.dropna(subset=["_day"])
    df["_refund"] = _refund_flag(df).astype(float)
    m = df["market"].astype(str).str.upper() if "market" in df.columns else "ALL"
    v = df["vertical"].astype(str).str.lower() if "vertical" in df.columns else "all"
    df["_segment"] = m.astype(str) + "|" + v.astype(str)
    grouped = (
        df.groupby(["_segment", "_day"], sort=True)
        .agg(n=("_refund", "size"), refunds=("_refund", "sum"))
        .reset_index()
    )
    grouped["refund_rate"] = grouped["refunds"] / grouped["n"].clip(lower=1)
    grouped = grouped.rename(columns={"_segment": "segment", "_day": "day"})
    grouped = grouped.loc[grouped["n"] >= int(min_orders_per_day)].copy()
    return grouped[["segment", "day", "n", "refund_rate"]]


def _zscore_anomaly(
    rates: np.ndarray,
    *,
    baseline_days: int,
    gap_days: int,
    z_threshold: float,
) -> dict[str, Any] | None:
    """
    First ``baseline_days`` form baseline; skip ``gap_days``; last point is test.

    Requires len(rates) >= baseline_days + gap_days + 1.
    """
    need = int(baseline_days) + int(gap_days) + 1
    if len(rates) < need:
        return None
    baseline = np.asarray(rates[: int(baseline_days)], dtype=float)
    test = float(rates[-1])
    mu = float(baseline.mean())
    sigma = float(baseline.std(ddof=0))
    if sigma < 1e-9:
        sigma = 1e-9
    z = (test - mu) / sigma
    if z < float(z_threshold):
        return None
    return {
        "baseline_mean": mu,
        "baseline_std": sigma,
        "test_rate": test,
        "z": float(z),
        "baseline_days": int(baseline_days),
        "gap_days": int(gap_days),
    }


def detect_segment_anomalies(
    orders: pd.DataFrame,
    *,
    baseline_days: int = 14,
    gap_days: int = 3,
    z_threshold: float = 3.0,
    min_orders_per_day: int = 5,
    min_segment_days: int | None = None,
) -> dict[str, Any]:
    """
    Scan market×vertical daily refund rates; return anomalies + proposed rules.

    Proposed rules are **advisory** (mode=shadow). Callers must not auto-promote.
    """
    series = build_daily_refund_series(orders, min_orders_per_day=min_orders_per_day)
    need = int(baseline_days) + int(gap_days) + 1
    if min_segment_days is None:
        min_segment_days = need
    anomalies: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []

    for segment, grp in series.groupby("segment", sort=True):
        grp = grp.sort_values("day")
        if len(grp) < int(min_segment_days):
            continue
        rates = grp["refund_rate"].to_numpy(dtype=float)
        hit = _zscore_anomaly(
            rates,
            baseline_days=baseline_days,
            gap_days=gap_days,
            z_threshold=z_threshold,
        )
        if hit is None:
            continue
        parts = str(segment).split("|", 1)
        market = parts[0] if parts else "ALL"
        vertical = parts[1] if len(parts) > 1 else "all"
        test_day = grp["day"].iloc[-1]
        day_n = int(grp["n"].iloc[-1])
        row = {
            "segment": str(segment),
            "market": market,
            "vertical": vertical,
            "test_day": pd.Timestamp(test_day).strftime("%Y-%m-%d"),
            "n_test_day": day_n,
            **hit,
        }
        anomalies.append(row)
        # Soft proposal: tighten soft_friction on the slice (shadow only).
        proposals.append(
            {
                "kind": "decision_threshold_overlay",
                "mode": "shadow",
                "auto_enforce": False,
                "market": market,
                "vertical": vertical,
                "reason": (
                    f"refund_rate z={hit['z']:.2f} on {row['test_day']} "
                    f"(baseline_mean={hit['baseline_mean']:.3f})"
                ),
                "suggested": {
                    "soft_friction_delta": -5,
                    "hold_review_delta": -5,
                    "note": (
                        "Tighten ladder 5pts vs current slice/global; "
                        "merge via scripts/ingest_proposed_rules.py (shadow by default)"
                    ),
                },
                "source": "segment_anomaly_zscore",
            }
        )

    return {
        "ok": True,
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_segments_scanned": int(series["segment"].nunique()) if len(series) else 0,
        "n_anomalies": len(anomalies),
        "params": {
            "baseline_days": int(baseline_days),
            "gap_days": int(gap_days),
            "z_threshold": float(z_threshold),
            "min_orders_per_day": int(min_orders_per_day),
        },
        "anomalies": anomalies,
        "proposed_rules": proposals,
        "auto_enforce": False,
    }


def proposals_to_yaml(report: dict[str, Any]) -> dict[str, Any]:
    """YAML-ready document for ingest (never auto-written to live OP)."""
    return {
        "auto_enforce": False,
        "mode": "shadow",
        "as_of": report.get("as_of"),
        "n_anomalies": report.get("n_anomalies"),
        "proposed_rules": list(report.get("proposed_rules") or []),
        "notes": (
            "Advisory only; auto_enforce is always false at discovery. "
            "Merge with scripts/ingest_proposed_rules.py (enabled + optional "
            "INGEST_RULES_LIVE / overnight INGEST_PROPOSED_RULES). "
            "promote_overlays remains the backtest-driven overlay path."
        ),
    }
