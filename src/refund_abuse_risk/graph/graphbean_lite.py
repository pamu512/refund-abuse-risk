"""GraphBEAN-lite: bipartite node+edge reconstruction anomaly (no GNN).

ponytail: Ridge feature-decoder + degree structure residual on the UV edge
table. Ceiling = linear reconstruction; upgrade = Grab GraphBEAN / RGCN on the
same edge schema. Outputs node/edge scores + MO tags for shadow actioning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from refund_abuse_risk.graph.bipartite import score_uv_bipartite

DEFAULT_GB_CFG: dict[str, Any] = {
    "enabled": True,
    "version": "0.1.0-lite",
    "min_edges": 20,
    "edge_z_threshold": 2.5,
    "node_z_threshold": 2.5,
    "ridge_alpha": 1.0,
}


def _node_feature_table(edges: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    id_col = "user_id" if kind == "user" else "vendor_id"
    g = edges.groupby(["market", "vertical", id_col], as_index=False).agg(
        n_orders=("n_orders", "sum"),
        n_refunds=("n_refunds", "sum"),
        n_partners=("vendor_id" if kind == "user" else "user_id", "nunique"),
        mean_lift=("lift", "mean"),
        mean_edge_anomaly=("edge_anomaly", "mean"),
    )
    g["refund_rate"] = g["n_refunds"] / g["n_orders"].clip(lower=1)
    g = g.rename(columns={id_col: "entity_id"})
    g["entity_kind"] = kind
    return g


def _tag_mo(edge_row: pd.Series, user_partners: int, vendor_partners: int) -> str:
    if float(edge_row.get("recon_z", 0) or 0) < 1.5 and not bool(edge_row.get("elevated")):
        return ""
    if user_partners <= 2 and vendor_partners <= 8 and float(edge_row.get("lift", 1)) >= 1.5:
        return "possible_collusion"
    if user_partners >= 4 and float(edge_row.get("refund_rate", 0)) >= 0.3:
        return "user_scatter_refunds"
    if float(edge_row.get("lift", 1)) >= 2.0:
        return "elevated_uv"
    if float(edge_row.get("recon_z", 0) or 0) >= 2.5:
        return "recon_anomaly"
    return ""


def run_graphbean_lite(
    history: pd.DataFrame,
    *,
    bipartite_cfg: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Fit a lite autoencoder-style model on UV edges; return scored edges/nodes.

    Never auto-enforces — caller writes proposals / features only.
    """
    c = {**DEFAULT_GB_CFG, **(cfg or {})}
    edges, _nodes_uv = score_uv_bipartite(history, bipartite_cfg)
    empty = {
        "ok": True,
        "version": c["version"],
        "n_edges": 0,
        "n_anomalous_edges": 0,
        "n_anomalous_nodes": 0,
        "edges": edges,
        "nodes": pd.DataFrame(),
        "auto_enforce": False,
    }
    if not c.get("enabled", True) or edges is None or len(edges) < int(c["min_edges"]):
        empty["ok"] = len(edges) >= int(c["min_edges"]) if edges is not None else False
        empty["reason"] = "insufficient_edges"
        return empty

    users = _node_feature_table(edges, kind="user")
    vendors = _node_feature_table(edges, kind="vendor")
    u = users.rename(
        columns={
            "n_orders": "u_n_orders",
            "n_refunds": "u_n_refunds",
            "n_partners": "u_n_partners",
            "refund_rate": "u_refund_rate",
            "mean_lift": "u_mean_lift",
            "mean_edge_anomaly": "u_mean_edge_anomaly",
            "entity_id": "user_id",
        }
    )
    v = vendors.rename(
        columns={
            "n_orders": "v_n_orders",
            "n_refunds": "v_n_refunds",
            "n_partners": "v_n_partners",
            "refund_rate": "v_refund_rate",
            "mean_lift": "v_mean_lift",
            "mean_edge_anomaly": "v_mean_edge_anomaly",
            "entity_id": "vendor_id",
        }
    )
    feat_cols = [
        "u_n_orders",
        "u_refund_rate",
        "u_n_partners",
        "u_mean_lift",
        "v_n_orders",
        "v_refund_rate",
        "v_n_partners",
        "v_mean_lift",
    ]
    merged = edges.merge(
        u[
            [
                "market",
                "vertical",
                "user_id",
                "u_n_orders",
                "u_refund_rate",
                "u_n_partners",
                "u_mean_lift",
                "u_mean_edge_anomaly",
            ]
        ],
        on=["market", "vertical", "user_id"],
        how="left",
    ).merge(
        v[
            [
                "market",
                "vertical",
                "vendor_id",
                "v_n_orders",
                "v_refund_rate",
                "v_n_partners",
                "v_mean_lift",
                "v_mean_edge_anomaly",
            ]
        ],
        on=["market", "vertical", "vendor_id"],
        how="left",
    )
    for col in feat_cols:
        merged[col] = merged[col].fillna(0.0)

    x = merged[feat_cols].to_numpy(dtype=float)
    # Feature decoder targets: edge refund_rate + log1p(n_orders) + lift
    y = np.column_stack(
        [
            merged["refund_rate"].to_numpy(dtype=float),
            np.log1p(merged["n_orders"].to_numpy(dtype=float)),
            np.log1p(merged["lift"].clip(lower=0).to_numpy(dtype=float)),
        ]
    )
    model = Ridge(alpha=float(c["ridge_alpha"]), random_state=42)
    model.fit(x, y)
    y_hat = model.predict(x)
    recon_err = np.mean((y - y_hat) ** 2, axis=1)

    # Structure residual: unexpected dense pairing vs degree product prior.
    deg_prior = (
        merged["u_n_partners"].to_numpy(dtype=float)
        * merged["v_n_partners"].to_numpy(dtype=float)
    )
    deg_prior = deg_prior / max(float(deg_prior.mean()), 1e-6)
    struct_err = np.abs(np.log1p(merged["n_orders"].to_numpy(dtype=float)) - np.log1p(deg_prior))
    combined = recon_err + 0.25 * struct_err
    mu = float(combined.mean())
    sigma = float(combined.std()) or 1e-9
    z = (combined - mu) / sigma

    out_edges = merged.copy()
    out_edges["recon_error"] = recon_err
    out_edges["structure_error"] = struct_err
    out_edges["graphbean_score"] = combined
    out_edges["recon_z"] = z
    out_edges["graphbean_anomalous"] = z >= float(c["edge_z_threshold"])
    out_edges["mo_tag"] = [
        _tag_mo(out_edges.iloc[i], int(out_edges.iloc[i]["u_n_partners"]), int(out_edges.iloc[i]["v_n_partners"]))
        for i in range(len(out_edges))
    ]

    # Node scores: max edge graphbean_score among incident edges.
    user_nodes = (
        out_edges.groupby(["market", "vertical", "user_id"], as_index=False)
        .agg(
            graphbean_score=("graphbean_score", "max"),
            recon_z=("recon_z", "max"),
            n_partners=("vendor_id", "nunique"),
            n_orders=("n_orders", "sum"),
            mo_tag=("mo_tag", lambda s: next((x for x in s if x), "")),
        )
        .rename(columns={"user_id": "entity_id"})
    )
    user_nodes["entity_kind"] = "user"
    vendor_nodes = (
        out_edges.groupby(["market", "vertical", "vendor_id"], as_index=False)
        .agg(
            graphbean_score=("graphbean_score", "max"),
            recon_z=("recon_z", "max"),
            n_partners=("user_id", "nunique"),
            n_orders=("n_orders", "sum"),
            mo_tag=("mo_tag", lambda s: next((x for x in s if x), "")),
        )
        .rename(columns={"vendor_id": "entity_id"})
    )
    vendor_nodes["entity_kind"] = "vendor"
    nodes = pd.concat([user_nodes, vendor_nodes], ignore_index=True)
    n_mu = float(nodes["graphbean_score"].mean()) if len(nodes) else 0.0
    n_sd = float(nodes["graphbean_score"].std()) or 1e-9
    nodes["node_z"] = (nodes["graphbean_score"] - n_mu) / n_sd
    nodes["graphbean_anomalous"] = nodes["node_z"] >= float(c["node_z_threshold"])

    proposals: list[dict[str, Any]] = []
    for _, row in out_edges.loc[out_edges["graphbean_anomalous"]].head(50).iterrows():
        proposals.append(
            {
                "kind": "graphbean_edge",
                "mode": "shadow",
                "auto_enforce": False,
                "market": str(row["market"]),
                "vertical": str(row["vertical"]),
                "user_id": str(row["user_id"]),
                "vendor_id": str(row["vendor_id"]),
                "mo_tag": str(row.get("mo_tag") or ""),
                "recon_z": float(row["recon_z"]),
                "suggested_challenge": (
                    "identity_verify"
                    if str(row.get("mo_tag")) == "possible_collusion"
                    else "payment_verify"
                ),
                "reason": f"graphbean_lite z={float(row['recon_z']):.2f} mo={row.get('mo_tag')}",
            }
        )

    return {
        "ok": True,
        "version": c["version"],
        "n_edges": int(len(out_edges)),
        "n_anomalous_edges": int(out_edges["graphbean_anomalous"].sum()),
        "n_anomalous_nodes": int(nodes["graphbean_anomalous"].sum()),
        "edges": out_edges,
        "nodes": nodes,
        "proposed_actions": proposals,
        "auto_enforce": False,
        "params": {
            "edge_z_threshold": float(c["edge_z_threshold"]),
            "node_z_threshold": float(c["node_z_threshold"]),
            "min_edges": int(c["min_edges"]),
        },
    }
