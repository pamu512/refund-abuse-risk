"""Mint weak training labels from elevated UV bipartite discovery (as-of)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from refund_abuse_risk.graph.bipartite import history_as_of, score_uv_bipartite


def _edge_elevated(
    edges: pd.DataFrame,
    *,
    market: str,
    vertical: str,
    user_id: str,
    vendor_id: str,
) -> bool:
    if edges is None or edges.empty or "elevated" not in edges.columns:
        return False
    hit = edges[
        (edges["market"].astype(str) == market)
        & (edges["vertical"].astype(str) == vertical)
        & (edges["user_id"].astype(str) == user_id)
        & (edges["vendor_id"].astype(str) == vendor_id)
        & edges["elevated"].astype(bool)
    ]
    return not hit.empty


def mint_weak_labels_from_uv_anomaly(
    orders: pd.DataFrame,
    history: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    For orders whose UV edge is elevated **as-of event_ts**, mint weak abuse
    (+ optional discovery fraud). Never overwrites proven fraud labels.

    Requires order `event_ts` (or `order_ts`); rows without a timestamp are skipped.
    """
    out = orders.copy()
    for col, default in (
        ("abuse_label", 0),
        ("abuse_label_weak", 0),
        ("fraud_label", 0),
        ("fraud_label_source", ""),
    ):
        if col not in out.columns:
            out[col] = default

    minted = 0
    skipped_no_ts = 0
    for idx, row in out.iterrows():
        src = str(row.get("fraud_label_source", "") or "").lower()
        if src == "proven":
            continue
        as_of = row.get("event_ts") or row.get("order_ts")
        if as_of is None or (isinstance(as_of, float) and pd.isna(as_of)) or str(as_of) == "":
            skipped_no_ts += 1
            continue
        market = str(row.get("market", "")).upper() or "ALL"
        vertical = str(row.get("vertical", "")).lower() or "all"
        user_id = str(row.get("user_id", "") or "")
        vendor_id = str(row.get("vendor_id", "") or "")
        if not user_id or not vendor_id:
            continue
        hist = history_as_of(
            history,
            as_of=as_of,
            exclude_order_id=str(row.get("order_id", "") or "") or None,
        )
        edges, _nodes = score_uv_bipartite(hist, cfg)
        if not _edge_elevated(
            edges,
            market=market,
            vertical=vertical,
            user_id=user_id,
            vendor_id=vendor_id,
        ):
            continue
        out.at[idx, "abuse_label"] = 1
        out.at[idx, "abuse_label_weak"] = 1
        # Weak fraud only — never touches proven (continued above).
        out.at[idx, "fraud_label"] = 1
        out.at[idx, "fraud_label_source"] = "discovery"
        minted += 1
    out.attrs["discovery_minted"] = minted
    out.attrs["discovery_skipped_no_ts"] = skipped_no_ts
    return out
