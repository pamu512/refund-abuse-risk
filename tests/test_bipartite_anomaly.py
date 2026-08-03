from __future__ import annotations

from pathlib import Path

import pandas as pd

from refund_abuse_risk.graph.bipartite import (
    bipartite_feature_lookups,
    export_partitions,
    features_for_order,
    score_uv_bipartite,
)


def _hist() -> pd.DataFrame:
    rows: list[dict] = []
    # Clean background: many UV pairs, rare refunds → low base rate.
    for i in range(40):
        rows.append(
            {
                "order_id": f"C-{i}",
                "user_id": f"Uc{i}",
                "vendor_id": f"Vc{i % 8}",
                "market": "SG",
                "vertical": "food",
                "is_refund": 0,
            }
        )
    # Collusive UV: same user×vendor, many refunds.
    for i in range(8):
        rows.append(
            {
                "order_id": f"BAD-{i}",
                "user_id": "Ubad",
                "vendor_id": "Vbad",
                "market": "SG",
                "vertical": "food",
                "is_refund": 1 if i < 6 else 0,
            }
        )
    return pd.DataFrame(rows)


def test_uv_edge_elevated_for_collusion_pair() -> None:
    cfg = {
        "min_edge_orders": 3,
        "min_edge_refunds": 2,
        "min_lift": 2.0,
        "min_refund_rate": 0.25,
    }
    edges, nodes = score_uv_bipartite(_hist(), cfg)
    bad = edges[(edges["user_id"] == "Ubad") & (edges["vendor_id"] == "Vbad")]
    assert len(bad) == 1
    assert bool(bad.iloc[0]["elevated"]) is True
    assert float(bad.iloc[0]["edge_anomaly"]) > 0
    assert bad.iloc[0]["mo_tag"] in {"possible_collusion", "elevated_uv", "user_scatter_refunds"}
    assert any(
        (n.entity_kind == "user" and n.entity_id == "Ubad") for n in nodes.itertuples(index=False)
    )


def test_feature_lookup_and_export(tmp_path: Path) -> None:
    edges, nodes = score_uv_bipartite(
        _hist(),
        {"min_edge_orders": 3, "min_edge_refunds": 2, "min_lift": 2.0, "min_refund_rate": 0.25},
    )
    edge_map, node_map = bipartite_feature_lookups(edges, nodes)
    feat = features_for_order(
        {"user_id": "Ubad", "vendor_id": "Vbad", "market": "SG", "vertical": "food"},
        edge_map=edge_map,
        node_map=node_map,
    )
    assert feat["uv_edge_anomaly"] > 0
    assert feat["user_bipartite_anomaly"] > 0
    assert feat["vendor_bipartite_anomaly"] > 0
    edge_path, node_path = export_partitions(
        edges, nodes, root=tmp_path, as_of_date="2026-08-03", version="0.1.0"
    )
    assert "edge_anomaly" in edge_path.read_text(encoding="utf-8")
    assert "node_anomaly" in node_path.read_text(encoding="utf-8")
