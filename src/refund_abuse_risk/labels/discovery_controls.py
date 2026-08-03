"""P0 discovery honesty: as-of UV mint + volume caps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from refund_abuse_risk.graph.bipartite import score_uv_bipartite


def discovery_config(label_weights: dict[str, Any] | None) -> dict[str, Any]:
    lw = label_weights or {}
    raw = dict(lw.get("discovery") or {})
    return {
        "mint_mode": str(raw.get("mint_mode", "asof_daily")),  # asof_daily | batch
        "max_fraction_of_rows": float(raw.get("max_fraction_of_rows", 0.05)),
        "max_fraction_of_fraud_positives": float(
            raw.get("max_fraction_of_fraud_positives", 0.35)
        ),
        "iforest_max_fraction": float(raw.get("iforest_max_fraction", 0.02)),
        "self_train_max_fraction": float(raw.get("self_train_max_fraction", 0.01)),
    }


def _protected_mask(frame: pd.DataFrame) -> np.ndarray:
    src = frame["fraud_label_source"].astype(str).str.lower().to_numpy()
    strong = frame.get("strong_fraud_label", pd.Series(0, index=frame.index)).astype(float).to_numpy()
    return (src == "proven") | (strong >= 1)


def mint_uv_asof_daily(
    frame: pd.DataFrame,
    history: pd.DataFrame | None,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Mint discovery labels from UV edges scored on history **before each order day**.

    For orders on calendar day D (UTC), edges use only ``event_ts < D`` — no
    same-day / future refund leakage into labels.
    """
    out = frame.copy()
    for col, default in (
        ("abuse_label", 0),
        ("abuse_label_weak", 0),
        ("fraud_label", 0),
        ("fraud_label_source", ""),
        ("strong_fraud_label", 0),
    ):
        if col not in out.columns:
            out[col] = default
    out["fraud_label_source"] = (
        out["fraud_label_source"].astype("string").fillna("").astype(str).replace({"nan": "", "<NA>": ""})
    )
    stats: dict[str, Any] = {
        "mode": "asof_daily",
        "uv_days": 0,
        "uv_elevated_edges_total": 0,
        "uv_minted": 0,
        "enabled": True,
    }
    if history is None or history.empty or "event_ts" not in out.columns:
        stats["enabled"] = False
        return out, stats

    hist = history.copy()
    hist["_ts"] = pd.to_datetime(hist["event_ts"], utc=True)
    out_ts = pd.to_datetime(out["event_ts"], utc=True, errors="coerce")
    if out_ts.isna().all():
        stats["enabled"] = False
        return out, stats
    out = out.copy()
    out["_ts"] = out_ts
    out["_day"] = out["_ts"].dt.floor("D")
    assign = np.zeros(len(out), dtype=bool)
    elev_total = 0
    days = 0
    for day, idx in out.groupby("_day", sort=True).groups.items():
        if pd.isna(day):
            continue
        days += 1
        cut = pd.Timestamp(day)
        if cut.tzinfo is None:
            cut = cut.tz_localize("UTC")
        else:
            cut = cut.tz_convert("UTC")
        hist_day = hist.loc[hist["_ts"] < cut]
        edges, _nodes = score_uv_bipartite(hist_day, cfg)
        if edges.empty or "elevated" not in edges.columns:
            continue
        elev = edges[edges["elevated"]]
        elev_total += int(len(elev))
        if elev.empty:
            continue
        elev_key = elev.assign(
            _market=elev["market"].astype(str),
            _vertical=elev["vertical"].astype(str),
            _user_id=elev["user_id"].astype(str),
            _vendor_id=elev["vendor_id"].astype(str),
            _uv=1,
        )[["_market", "_vertical", "_user_id", "_vendor_id", "_uv"]].drop_duplicates()
        chunk = out.loc[idx, ["market", "vertical", "user_id", "vendor_id"]].copy()
        chunk["_market"] = chunk["market"].astype(str).str.upper().replace({"": "ALL"})
        chunk["_vertical"] = chunk["vertical"].astype(str).str.lower().replace({"": "all"})
        chunk["_user_id"] = chunk["user_id"].astype(str)
        chunk["_vendor_id"] = chunk["vendor_id"].astype(str)
        merged = chunk.merge(
            elev_key, on=["_market", "_vertical", "_user_id", "_vendor_id"], how="left"
        )
        hit = merged["_uv"].fillna(0).to_numpy() >= 1
        # Map back to positional mask in `out`
        pos = out.index.get_indexer(idx)
        assign[pos[hit]] = True

    protected = _protected_mask(out)
    assign = assign & ~protected
    out.loc[assign, "abuse_label"] = 1
    out.loc[assign, "abuse_label_weak"] = 1
    out.loc[assign, "fraud_label"] = 1
    out.loc[assign, "fraud_label_source"] = "discovery"
    out = out.drop(columns=["_ts", "_day"], errors="ignore")
    stats["uv_days"] = int(days)
    stats["uv_elevated_edges_total"] = int(elev_total)
    stats["uv_minted"] = int(assign.sum())
    return out, stats


def cap_discovery_labels(
    frame: pd.DataFrame,
    *,
    max_fraction_of_rows: float = 0.05,
    max_fraction_of_fraud_positives: float = 0.35,
    random_state: int = 42,
    score_col: str = "unsupervised_anomaly_score",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Limit discovery-labeled rows so they cannot dominate training.

    Keeps highest ``score_col`` (else random). Demoted rows lose discovery
    fraud/abuse_weak marks (proven/proxy untouched).
    """
    out = frame.copy()
    src = out["fraud_label_source"].astype(str).str.lower()
    disc = src.eq("discovery")
    n = len(out)
    n_disc = int(disc.sum())
    stats: dict[str, Any] = {
        "discovery_before": n_disc,
        "discovery_after": n_disc,
        "demoted": 0,
        "max_fraction_of_rows": float(max_fraction_of_rows),
        "max_fraction_of_fraud_positives": float(max_fraction_of_fraud_positives),
    }
    if n == 0 or n_disc == 0:
        return out, stats

    cap_rows = max(0, int(np.floor(n * float(max_fraction_of_rows))))
    # Also cap vs non-discovery fraud positives.
    other_pos = int(((out["fraud_label"].astype(int) >= 1) & ~disc).sum())
    cap_vs_pos = (
        max(0, int(np.floor(other_pos * float(max_fraction_of_fraud_positives) / max(1e-9, 1.0 - float(max_fraction_of_fraud_positives)))))
        if other_pos > 0
        else cap_rows
    )
    # If other_pos=0, only row cap applies.
    cap = min(cap_rows, cap_vs_pos) if other_pos > 0 else cap_rows
    # Allow at least a tiny discovery set when cap_rows > 0.
    if cap_rows > 0:
        cap = max(cap, min(cap_rows, 1))
    stats["cap"] = int(cap)
    if n_disc <= cap:
        return out, stats

    disc_idx = out.index[disc]
    if score_col in out.columns:
        scores = out.loc[disc_idx, score_col].astype(float).fillna(-np.inf)
        keep = scores.sort_values(ascending=False).head(cap).index
    else:
        rng = np.random.default_rng(int(random_state))
        keep = pd.Index(rng.choice(disc_idx.to_numpy(), size=cap, replace=False))
    drop = disc_idx.difference(keep)
    out.loc[drop, "fraud_label"] = 0
    out.loc[drop, "fraud_label_source"] = ""
    # Only clear abuse if it was weak discovery mark.
    weak = out.loc[drop, "abuse_label_weak"].astype(float) >= 1
    drop_weak = drop[weak.to_numpy()]
    out.loc[drop_weak, "abuse_label"] = 0
    out.loc[drop_weak, "abuse_label_weak"] = 0
    stats["discovery_after"] = int((out["fraud_label_source"].astype(str).str.lower() == "discovery").sum())
    stats["demoted"] = int(len(drop))
    return out, stats
