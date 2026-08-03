#!/usr/bin/env python3
"""Offline user↔vendor bipartite anomaly discovery → warehouse partitions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from refund_abuse_risk.config import load_bipartite_anomaly
from refund_abuse_risk.graph.bipartite import export_partitions, score_uv_bipartite

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=ROOT / "data" / "history.csv",
        help="History CSV with user_id, vendor_id, is_refund, market, vertical",
    )
    parser.add_argument(
        "--warehouse-root",
        type=Path,
        default=ROOT / "data" / "warehouse",
    )
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--json-summary", action="store_true")
    parser.add_argument(
        "--mint-weak-labels",
        type=Path,
        default=None,
        help="Orders CSV to mint discovery weak labels into (writes alongside with .discovery.csv)",
    )
    args = parser.parse_args()

    cfg = load_bipartite_anomaly()
    as_of = args.as_of_date or datetime.now(timezone.utc).date().isoformat()
    history = pd.read_csv(args.history)
    edges, nodes = score_uv_bipartite(history, cfg)
    wh = cfg.get("warehouse") or {}
    edge_path, node_path = export_partitions(
        edges,
        nodes,
        root=args.warehouse_root,
        as_of_date=as_of,
        version=str(cfg.get("version", "0.1.0")),
        edge_table=str(wh.get("edge_table", "bipartite_uv_edges")),
        node_table=str(wh.get("node_table", "bipartite_uv_nodes")),
    )
    n_elev = int(edges["elevated"].sum()) if not edges.empty and "elevated" in edges.columns else 0
    summary = {
        "relation": cfg.get("relation", "uv"),
        "as_of_date": as_of,
        "version": cfg.get("version"),
        "n_edges": int(len(edges)),
        "n_elevated_edges": n_elev,
        "n_nodes": int(len(nodes)),
        "edge_path": str(edge_path),
        "node_path": str(node_path),
    }
    if args.mint_weak_labels is not None:
        from refund_abuse_risk.labels.discovery import mint_weak_labels_from_uv_anomaly

        orders = pd.read_csv(args.mint_weak_labels)
        labeled = mint_weak_labels_from_uv_anomaly(orders, history, cfg)
        out_path = args.mint_weak_labels.with_suffix(".discovery.csv")
        labeled.to_csv(out_path, index=False)
        summary["discovery_minted"] = int(labeled.attrs.get("discovery_minted", 0))
        summary["discovery_path"] = str(out_path)
    if args.json_summary:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"UV bipartite: {summary['n_edges']} edges "
            f"({summary['n_elevated_edges']} elevated), "
            f"{summary['n_nodes']} nodes → {edge_path.parent}"
        )


if __name__ == "__main__":
    main()
