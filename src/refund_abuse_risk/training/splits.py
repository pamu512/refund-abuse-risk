"""Train/holdout splits — time-based OOT is the primary evaluation cut."""

from __future__ import annotations

from typing import Any

import pandas as pd


def time_based_order_split(
    orders: pd.DataFrame,
    *,
    holdout_days: float = 7.0,
    min_train: int = 20,
    min_test: int = 10,
    fallback_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Split orders into train / OOT test by ``event_ts``.

    Test = last ``holdout_days`` (UTC). If the order span is shorter than that,
    use the last ``fallback_frac`` of the span so short demo windows still OOT.
    Falls back to a positional half-split only when timestamps are unusable.
    """
    if orders is None or orders.empty:
        empty = orders.iloc[0:0].copy() if orders is not None else pd.DataFrame()
        return empty, empty, {"mode": "empty", "ok": False, "temporal_ok": False}

    if "event_ts" not in orders.columns:
        n = len(orders)
        cut = max(n // 2, 1)
        train = orders.iloc[:cut].reset_index(drop=True)
        test = orders.iloc[cut:].reset_index(drop=True)
        return train, test, {
            "mode": "positional_fallback",
            "ok": bool(len(train) and len(test)),
            "temporal_ok": False,
            "reason": "missing_event_ts",
        }

    ts = pd.to_datetime(orders["event_ts"], utc=True, errors="coerce")
    valid = ts.notna()
    if int(valid.sum()) < (min_train + min_test):
        n = len(orders)
        cut = max(n // 2, 1)
        train = orders.iloc[:cut].reset_index(drop=True)
        test = orders.iloc[cut:].reset_index(drop=True)
        return train, test, {
            "mode": "positional_fallback",
            "ok": bool(len(train) and len(test)),
            "temporal_ok": False,
            "reason": "insufficient_valid_timestamps",
        }

    tmin = ts.loc[valid].min()
    tmax = ts.loc[valid].max()
    span = tmax - tmin
    requested = pd.Timedelta(days=float(holdout_days))
    used_adaptive = False
    hold = requested
    if span <= requested:
        hold = max(span * float(fallback_frac), pd.Timedelta(hours=1))
        used_adaptive = True
    cut_ts = tmax - hold

    train_mask = valid & (ts < cut_ts)
    test_mask = valid & (ts >= cut_ts)
    train = orders.loc[train_mask].reset_index(drop=True)
    test = orders.loc[test_mask].reset_index(drop=True)

    if len(train) < min_train or len(test) < min_test:
        # Widen test to last fallback_frac of rows by time rank.
        ranked = ts.loc[valid].sort_values()
        n_valid = len(ranked)
        n_test = max(min_test, int(round(n_valid * float(fallback_frac))))
        n_test = min(n_test, n_valid - min_train) if n_valid > min_train else max(1, n_valid // 5)
        cut_ts = ranked.iloc[n_valid - n_test]
        train_mask = valid & (ts < cut_ts)
        test_mask = valid & (ts >= cut_ts)
        train = orders.loc[train_mask].reset_index(drop=True)
        test = orders.loc[test_mask].reset_index(drop=True)
        used_adaptive = True

    holdout_days_used = float(hold / pd.Timedelta(days=1))
    # Honest temporal OOT: requested holdout calendar days, not adaptive/positional.
    temporal_ok = (not used_adaptive) and (
        holdout_days_used + 1e-9 >= float(holdout_days) * 0.99
    )
    stats: dict[str, Any] = {
        "mode": "time_oot" if not used_adaptive else "time_oot_adaptive",
        "ok": bool(len(train) >= 1 and len(test) >= 1),
        "temporal_ok": bool(temporal_ok),
        "holdout_days_requested": float(holdout_days),
        "holdout_days_used": holdout_days_used,
        "adaptive": bool(used_adaptive),
        "cut_ts": cut_ts.isoformat() if hasattr(cut_ts, "isoformat") else str(cut_ts),
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "span_days": float(span / pd.Timedelta(days=1)),
        "train_ts_min": ts.loc[train_mask].min().isoformat() if len(train) else None,
        "train_ts_max": ts.loc[train_mask].max().isoformat() if len(train) else None,
        "test_ts_min": ts.loc[test_mask].min().isoformat() if len(test) else None,
        "test_ts_max": ts.loc[test_mask].max().isoformat() if len(test) else None,
    }
    return train, test, stats
