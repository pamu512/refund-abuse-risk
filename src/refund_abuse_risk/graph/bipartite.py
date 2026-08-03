"""User↔vendor bipartite anomaly scores (offline discovery; no GNN).

ponytail: statistical lift vs market×vertical base rate + log-support.
Upgrade path: GraphBEAN / edge-attributed autoencoder on the same edge table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CFG: dict[str, Any] = {
    "enabled": True,
    "version": "0.1.0",
    "relation": "uv",
    "min_edge_orders": 3,
    "min_edge_refunds": 2,
    "min_lift": 2.0,
    "min_refund_rate": 0.25,
    "node_score": "max_edge",
}


def zero_bipartite_features() -> dict[str, float]:
    return {
        "uv_edge_anomaly": 0.0,
        "user_bipartite_anomaly": 0.0,
        "vendor_bipartite_anomaly": 0.0,
        # Edge attributes / MO tags (features, not only discovery labels).
        "uv_edge_lift": 1.0,
        "uv_edge_n_orders": 0.0,
        "uv_edge_elevated": 0.0,
        "uv_mo_possible_collusion": 0.0,
        "uv_mo_user_scatter": 0.0,
        "uv_mo_elevated_uv": 0.0,
    }


def _score_uv_edges_raw(frame: pd.DataFrame, c: dict[str, Any]) -> pd.DataFrame:
    """Edge table + elevated flags without null baseline (used by null shuffles)."""
    edges = (
        frame.groupby(["market", "vertical", "user_id", "vendor_id"], as_index=False)
        .agg(n_orders=("is_refund", "size"), n_refunds=("is_refund", "sum"))
    )
    edges["refund_rate"] = edges["n_refunds"] / edges["n_orders"].clip(lower=1)
    base = (
        frame.groupby(["market", "vertical"], as_index=False)
        .agg(base_orders=("is_refund", "size"), base_refunds=("is_refund", "sum"))
    )
    edges = edges.merge(base, on=["market", "vertical"], how="left")
    loo_orders = (edges["base_orders"] - edges["n_orders"]).clip(lower=0)
    loo_refunds = (edges["base_refunds"] - edges["n_refunds"]).clip(lower=0)
    edges["base_rate"] = (loo_refunds / loo_orders.clip(lower=1)).fillna(0.0)
    eps = 1e-6
    edges["lift"] = edges["refund_rate"] / edges["base_rate"].clip(lower=eps)
    edges["edge_anomaly"] = edges["lift"] * np.log1p(edges["n_orders"]) * edges["refund_rate"]
    edges = edges.drop(columns=["base_orders", "base_refunds"], errors="ignore")
    min_n = int(c.get("min_edge_orders", 3))
    min_r = int(c.get("min_edge_refunds", 2))
    min_lift = float(c.get("min_lift", 2.0))
    min_rate = float(c.get("min_refund_rate", 0.25))
    edges["elevated"] = (
        (edges["n_orders"] >= min_n)
        & (edges["n_refunds"] >= min_r)
        & (edges["lift"] >= min_lift)
        & (edges["refund_rate"] >= min_rate)
    )
    return edges


def apply_edge_shuffle_null(
    history: pd.DataFrame,
    edges: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Demote elevated edges whose anomaly is not above an edge-shuffle null percentile.

    Shuffles ``vendor_id`` within market×vertical so refund mass is preserved but
    user↔vendor pairing is destroyed — a cheap GraphBEAN-precursor honesty check.
    """
    null_cfg = dict(cfg.get("null_baseline") or {})
    stats: dict[str, Any] = {
        "enabled": bool(null_cfg.get("enabled", True)),
        "elevated_before": int(edges["elevated"].sum()) if len(edges) and "elevated" in edges else 0,
        "elevated_after": 0,
        "demoted": 0,
        "null_threshold": None,
    }
    if not stats["enabled"] or edges.empty or history is None or history.empty:
        stats["elevated_after"] = stats["elevated_before"]
        return edges, stats
    # Tiny frames: null is unstable / expensive relative to signal.
    if len(history) < int(null_cfg.get("min_history_rows", 40)):
        stats["elevated_after"] = stats["elevated_before"]
        stats["skipped"] = "history_too_small"
        return edges, stats

    n_shuffles = max(1, int(null_cfg.get("n_shuffles", 3)))
    pct = float(null_cfg.get("percentile", 95.0))
    seed = int(null_cfg.get("random_state", 42))
    rng = np.random.default_rng(seed)

    frame = history
    null_vals: list[np.ndarray] = []
    for _ in range(n_shuffles):
        shuffled = frame.copy()
        for _, idx in shuffled.groupby(["market", "vertical"], sort=False).groups.items():
            vendors = shuffled.loc[idx, "vendor_id"].to_numpy().copy()
            rng.shuffle(vendors)
            shuffled.loc[idx, "vendor_id"] = vendors
        null_edges = _score_uv_edges_raw(shuffled, cfg)
        if len(null_edges):
            null_vals.append(null_edges["edge_anomaly"].to_numpy(dtype=float))
    if not null_vals:
        stats["elevated_after"] = stats["elevated_before"]
        stats["skipped"] = "empty_null"
        return edges, stats

    pool = np.concatenate(null_vals)
    thr = float(np.percentile(pool, pct)) if len(pool) else 0.0
    out = edges.copy()
    before = out["elevated"].astype(bool).to_numpy()
    keep = before & (out["edge_anomaly"].astype(float).to_numpy() > thr)
    out["elevated"] = keep
    stats["null_threshold"] = thr
    stats["elevated_after"] = int(keep.sum())
    stats["demoted"] = int(before.sum() - keep.sum())
    return out, stats


def score_uv_bipartite(
    history: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score UV edges and nodes from settled history rows.

    Expects columns: user_id, vendor_id, market, vertical, is_refund (0/1).
    Optional: order_id (dedupe), amount.
    """
    c = {**DEFAULT_CFG, **(cfg or {})}
    empty_edges = pd.DataFrame(
        columns=[
            "market",
            "vertical",
            "user_id",
            "vendor_id",
            "n_orders",
            "n_refunds",
            "refund_rate",
            "base_rate",
            "lift",
            "edge_anomaly",
            "elevated",
            "mo_tag",
        ]
    )
    empty_nodes = pd.DataFrame(
        columns=[
            "market",
            "vertical",
            "entity_kind",
            "entity_id",
            "n_partners",
            "n_orders",
            "n_refunds",
            "node_anomaly",
            "elevated",
            "mo_tag",
        ]
    )
    if history is None or history.empty or not c.get("enabled", True):
        return empty_edges, empty_nodes

    frame = history.copy()
    need = {"user_id", "vendor_id", "is_refund"}
    if not need.issubset(set(frame.columns)):
        return empty_edges, empty_nodes
    if "market" not in frame.columns:
        frame["market"] = "ALL"
    if "vertical" not in frame.columns:
        frame["vertical"] = "all"
    frame["user_id"] = frame["user_id"].astype(str)
    frame["vendor_id"] = frame["vendor_id"].astype(str)
    frame["market"] = frame["market"].astype(str).str.upper().replace({"": "ALL"})
    frame["vertical"] = frame["vertical"].astype(str).str.lower().replace({"": "all"})
    frame["is_refund"] = pd.to_numeric(frame["is_refund"], errors="coerce").fillna(0).astype(int)
    frame = frame[(frame["user_id"] != "") & (frame["vendor_id"] != "")]
    if frame.empty:
        return empty_edges, empty_nodes

    if "order_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["order_id"], keep="last")

    edges = _score_uv_edges_raw(frame, c)
    edges, _null_stats = apply_edge_shuffle_null(frame, edges, c)

    user_deg = edges.groupby(["market", "vertical", "user_id"], as_index=False).agg(
        n_partners=("vendor_id", "nunique")
    )
    vendor_deg = edges.groupby(["market", "vertical", "vendor_id"], as_index=False).agg(
        n_partners=("user_id", "nunique")
    )
    edges = edges.merge(user_deg, on=["market", "vertical", "user_id"], how="left", suffixes=("", "_u"))
    edges = edges.merge(
        vendor_deg.rename(columns={"n_partners": "vendor_n_partners"}),
        on=["market", "vertical", "vendor_id"],
        how="left",
    )
    edges["n_partners"] = edges["n_partners"].fillna(1).astype(int)
    edges["vendor_n_partners"] = edges["vendor_n_partners"].fillna(1).astype(int)

    def _mo(row: pd.Series) -> str:
        if not bool(row["elevated"]):
            return ""
        # Tight pair with repeated refunds → collusion-ish; scatter user → farm.
        if int(row["n_partners"]) <= 2 and int(row["vendor_n_partners"]) <= 8:
            return "possible_collusion"
        if int(row["n_partners"]) >= 4:
            return "user_scatter_refunds"
        return "elevated_uv"

    edges["mo_tag"] = edges.apply(_mo, axis=1)

    # Node scores from elevated edges (max).
    elev = edges[edges["elevated"]].copy()
    if elev.empty:
        nodes = empty_nodes.copy()
        return edges.drop(columns=["n_partners", "vendor_n_partners"], errors="ignore"), nodes

    user_nodes = (
        elev.groupby(["market", "vertical", "user_id"], as_index=False)
        .agg(
            node_anomaly=("edge_anomaly", "max"),
            n_partners=("vendor_id", "nunique"),
            n_orders=("n_orders", "sum"),
            n_refunds=("n_refunds", "sum"),
            mo_tag=("mo_tag", lambda s: s.value_counts().index[0] if len(s) else ""),
        )
        .rename(columns={"user_id": "entity_id"})
    )
    user_nodes["entity_kind"] = "user"
    user_nodes["elevated"] = True

    vendor_nodes = (
        elev.groupby(["market", "vertical", "vendor_id"], as_index=False)
        .agg(
            node_anomaly=("edge_anomaly", "max"),
            n_partners=("user_id", "nunique"),
            n_orders=("n_orders", "sum"),
            n_refunds=("n_refunds", "sum"),
            mo_tag=("mo_tag", lambda s: s.value_counts().index[0] if len(s) else ""),
        )
        .rename(columns={"vendor_id": "entity_id"})
    )
    vendor_nodes["entity_kind"] = "vendor"
    vendor_nodes["elevated"] = True

    nodes = pd.concat([user_nodes, vendor_nodes], ignore_index=True)
    nodes = nodes[
        [
            "market",
            "vertical",
            "entity_kind",
            "entity_id",
            "n_partners",
            "n_orders",
            "n_refunds",
            "node_anomaly",
            "elevated",
            "mo_tag",
        ]
    ]
    edge_out = edges.drop(columns=["n_partners", "vendor_n_partners"], errors="ignore")
    return edge_out, nodes


def bipartite_feature_lookups(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str, str, str], float],
    dict[tuple[str, str, str, str], float],
    dict[tuple[str, str, str, str], dict[str, Any]],
]:
    """Maps edge anomaly, node anomaly, and rich edge attrs (lift/n/mo_tag)."""
    edge_map: dict[tuple[str, str, str, str], float] = {}
    node_map: dict[tuple[str, str, str, str], float] = {}
    edge_attrs: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if edges is not None and not edges.empty:
        for row in edges.itertuples(index=False):
            key = (
                str(row.market),
                str(row.vertical),
                str(row.user_id),
                str(row.vendor_id),
            )
            edge_map[key] = float(row.edge_anomaly)
            edge_attrs[key] = {
                "lift": float(getattr(row, "lift", 1.0) or 1.0),
                "n_orders": float(getattr(row, "n_orders", 0) or 0),
                "elevated": bool(getattr(row, "elevated", False)),
                "mo_tag": str(getattr(row, "mo_tag", "") or ""),
            }
    if nodes is not None and not nodes.empty:
        for row in nodes.itertuples(index=False):
            key = (
                str(row.market),
                str(row.vertical),
                str(row.entity_kind),
                str(row.entity_id),
            )
            node_map[key] = float(row.node_anomaly)
    return edge_map, node_map, edge_attrs


def features_for_order(
    order: dict[str, Any],
    *,
    edge_map: dict[tuple[str, str, str, str], float] | None = None,
    node_map: dict[tuple[str, str, str, str], float] | None = None,
    edge_attrs: dict[tuple[str, str, str, str], dict[str, Any]] | None = None,
) -> dict[str, float]:
    out = zero_bipartite_features()
    if not edge_map and not node_map and not edge_attrs:
        return out
    market = str(order.get("market", "") or "ALL").upper() or "ALL"
    vertical = str(order.get("vertical", "") or "all").lower() or "all"
    user_id = str(order.get("user_id", "") or "")
    vendor_id = str(order.get("vendor_id", "") or "")
    key = (market, vertical, user_id, vendor_id)
    if edge_map and user_id and vendor_id:
        out["uv_edge_anomaly"] = float(edge_map.get(key, 0.0))
    if edge_attrs and user_id and vendor_id and key in edge_attrs:
        attr = edge_attrs[key]
        out["uv_edge_lift"] = float(attr.get("lift", 1.0) or 1.0)
        out["uv_edge_n_orders"] = float(attr.get("n_orders", 0.0) or 0.0)
        out["uv_edge_elevated"] = 1.0 if attr.get("elevated") else 0.0
        tag = str(attr.get("mo_tag", "") or "")
        out["uv_mo_possible_collusion"] = 1.0 if tag == "possible_collusion" else 0.0
        out["uv_mo_user_scatter"] = 1.0 if tag == "user_scatter_refunds" else 0.0
        out["uv_mo_elevated_uv"] = 1.0 if tag == "elevated_uv" else 0.0
    if node_map:
        if user_id:
            out["user_bipartite_anomaly"] = float(
                node_map.get((market, vertical, "user", user_id), 0.0)
            )
        if vendor_id:
            out["vendor_bipartite_anomaly"] = float(
                node_map.get((market, vertical, "vendor", vendor_id), 0.0)
            )
    return out


def history_as_of(
    history: pd.DataFrame,
    *,
    as_of: Any,
    exclude_order_id: str | None = None,
) -> pd.DataFrame:
    """Point-in-time history: event_ts <= as_of, optional order exclusion."""
    if history is None or history.empty:
        return history if history is not None else pd.DataFrame()
    out = history
    if exclude_order_id and "order_id" in out.columns:
        out = out[out["order_id"].astype(str) != str(exclude_order_id)]
    if out.empty or "event_ts" not in out.columns:
        return out
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    else:
        as_of_ts = as_of_ts.tz_convert("UTC")
    ts = pd.to_datetime(out["event_ts"], utc=True)
    return out.loc[ts <= as_of_ts]


def bipartite_features_as_of(
    order: dict[str, Any],
    history: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """UV anomaly features using only history at/before the order timestamp."""
    as_of = order.get("event_ts") or order.get("order_ts")
    if as_of is None:
        return zero_bipartite_features()
    hist = history_as_of(
        history,
        as_of=as_of,
        exclude_order_id=str(order.get("order_id", "") or "") or None,
    )
    edges, nodes = score_uv_bipartite(hist, cfg)
    edge_map, node_map, edge_attrs = bipartite_feature_lookups(edges, nodes)
    return features_for_order(
        order, edge_map=edge_map, node_map=node_map, edge_attrs=edge_attrs
    )


def export_partitions(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    root: Path | str,
    as_of_date: str,
    version: str,
    edge_table: str = "bipartite_uv_edges",
    node_table: str = "bipartite_uv_nodes",
) -> tuple[Path, Path]:
    root = Path(root)
    edge_path = root / edge_table / f"as_of_date={as_of_date}" / "part.csv"
    node_path = root / node_table / f"as_of_date={as_of_date}" / "part.csv"
    for path, frame in ((edge_path, edges), (node_path, nodes)):
        path.parent.mkdir(parents=True, exist_ok=True)
        out = frame.copy()
        if not out.empty:
            out.insert(0, "as_of_date", as_of_date)
            out.insert(1, "anomaly_version", version)
        else:
            out = pd.DataFrame(columns=["as_of_date", "anomaly_version", *list(frame.columns)])
        out.to_csv(path, index=False)
    return edge_path, node_path
