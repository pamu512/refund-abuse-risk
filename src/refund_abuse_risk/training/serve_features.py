"""Build training frames via the serve-path feature builder (P0 train≈serve)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from refund_abuse_risk.features.builders import build_order_feature_frame


def stratified_order_sample(
    orders: pd.DataFrame,
    *,
    n: int,
    random_state: int = 42,
    minority_oversample: float = 2.0,
    minority_frac_cap: float = 0.5,
) -> pd.DataFrame:
    """Downsample while mildly oversampling abuse/fraud — never a 100% minority bag."""
    if n <= 0 or len(orders) <= n:
        return orders.reset_index(drop=True)
    rng = np.random.default_rng(int(random_state))
    frame = orders.copy()
    abuse = frame.get("abuse_label", pd.Series(0, index=frame.index)).astype(int) >= 1
    fraud = frame.get("fraud_label", pd.Series(0, index=frame.index)).astype(int) >= 1
    minority = abuse | fraud
    min_idx = frame.index[minority].to_numpy()
    maj_idx = frame.index[~minority].to_numpy()
    pop_rate = float(minority.mean()) if len(frame) else 0.0
    if len(min_idx) == 0 or pop_rate <= 0.0:
        keep = rng.choice(frame.index.to_numpy(), size=n, replace=False)
        return frame.loc[keep].reset_index(drop=True)

    target_frac = min(float(minority_frac_cap), max(pop_rate * float(minority_oversample), pop_rate))
    n_min = min(len(min_idx), max(1, int(round(n * target_frac))))
    n_maj = n - n_min
    if n_maj > len(maj_idx):
        n_maj = len(maj_idx)
        n_min = n - n_maj
    keep_min = rng.choice(min_idx, size=n_min, replace=False)
    keep_maj = (
        rng.choice(maj_idx, size=n_maj, replace=False) if n_maj > 0 else np.array([], dtype=min_idx.dtype)
    )
    keep = np.concatenate([keep_min, keep_maj])
    rng.shuffle(keep)
    return frame.loc[keep].reset_index(drop=True)


def trim_history_lookback(
    history: pd.DataFrame,
    *,
    lookback_days: int,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep recent history only (speeds serve-path rebuild; still PIT within window)."""
    if history is None or history.empty or lookback_days <= 0:
        return history
    ts = pd.to_datetime(history["event_ts"], utc=True)
    end = as_of if as_of is not None else ts.max()
    start = end - pd.Timedelta(days=int(lookback_days))
    return history.loc[(ts >= start) & (ts <= end)].copy()


def build_serve_training_frame(
    orders: pd.DataFrame,
    history: pd.DataFrame,
    devices: pd.DataFrame,
    users: pd.DataFrame | None = None,
    *,
    max_rows: int = 5000,
    lookback_days: int = 180,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Sample orders and build features with ``build_order_feature_frame`` (serve path).
    """
    sample = stratified_order_sample(orders, n=int(max_rows), random_state=random_state)
    hist = history
    if lookback_days > 0 and not history.empty:
        as_of = pd.to_datetime(sample["event_ts"], utc=True, errors="coerce").max()
        if pd.isna(as_of):
            as_of = pd.to_datetime(history["event_ts"], utc=True).max()
        hist = trim_history_lookback(history, lookback_days=lookback_days, as_of=as_of)
    # Keep history for sampled users (+ same-device neighbors). Vendor/driver OR-filters
    # pull nearly the full lookback (few shared IDs) and make serve rebuild unusable.
    hist_feat = hist
    if hist is not None and not hist.empty and len(sample) and "user_id" in sample.columns:
        users_s = set(sample["user_id"].astype(str))
        if "device_id" in sample.columns and devices is not None and not devices.empty:
            if "device_id" in devices.columns and "user_id" in devices.columns:
                sample_devs = set(sample["device_id"].astype(str))
                neighbor = devices.loc[
                    devices["device_id"].astype(str).isin(sample_devs), "user_id"
                ].astype(str)
                users_s |= set(neighbor.tolist())
        hist_feat = hist.loc[hist["user_id"].astype(str).isin(users_s)]
    feat = build_order_feature_frame(sample, hist_feat, devices, users=users)
    stats = {
        "feature_source": "serve",
        "orders_in": int(len(orders)),
        "orders_sampled": int(len(sample)),
        "history_rows_lookback": int(len(hist)) if hist is not None else 0,
        "history_rows_used": int(len(hist_feat)) if hist_feat is not None else 0,
        "lookback_days": int(lookback_days),
    }
    return feat, stats
