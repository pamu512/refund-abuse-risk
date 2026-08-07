#!/usr/bin/env python3
"""P2a: GraphBEAN-lite UV reconstruction anomalies → proposed actions (shadow only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from refund_abuse_risk.config import load_bipartite_anomaly
from refund_abuse_risk.graph.graphbean_lite import run_graphbean_lite

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=ROOT / "data" / "history.csv",
    )
    parser.add_argument("--min-edges", type=int, default=20)
    parser.add_argument("--edge-z", type=float, default=2.5)
    parser.add_argument("--node-z", type=float, default=2.5)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data" / "proposed_rules" / "graphbean_lite.json",
    )
    parser.add_argument(
        "--out-yaml",
        type=Path,
        default=ROOT / "data" / "proposed_rules" / "graphbean_lite.proposed.yaml",
    )
    args = parser.parse_args()

    if not args.history.is_file():
        print(f"history not found: {args.history}", file=sys.stderr)
        return 1

    history = pd.read_csv(args.history)
    bipartite_cfg = load_bipartite_anomaly()
    result = run_graphbean_lite(
        history,
        bipartite_cfg=bipartite_cfg,
        cfg={
            "min_edges": args.min_edges,
            "edge_z_threshold": args.edge_z,
            "node_z_threshold": args.node_z,
        },
    )
    report = {
        "ok": result["ok"],
        "version": result["version"],
        "n_edges": result["n_edges"],
        "n_anomalous_edges": result["n_anomalous_edges"],
        "n_anomalous_nodes": result["n_anomalous_nodes"],
        "auto_enforce": False,
        "params": result.get("params"),
        "proposed_actions": result.get("proposed_actions") or [],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    doc = {
        "mode": "shadow",
        "auto_enforce": False,
        "source": "graphbean_lite",
        "version": result["version"],
        "n_anomalous_edges": result["n_anomalous_edges"],
        "actions": result.get("proposed_actions") or [],
    }
    args.out_yaml.write_text(
        yaml.safe_dump(doc, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_edges": report["n_edges"],
                "n_anomalous_edges": report["n_anomalous_edges"],
                "n_anomalous_nodes": report["n_anomalous_nodes"],
                "n_proposed_actions": len(report["proposed_actions"]),
                "auto_enforce": False,
                "out_json": str(args.out_json),
                "out_yaml": str(args.out_yaml),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
