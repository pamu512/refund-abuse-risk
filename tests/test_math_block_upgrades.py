from __future__ import annotations

import numpy as np
import pandas as pd

from refund_abuse_risk.graph.bipartite import apply_edge_shuffle_null, score_uv_bipartite
from refund_abuse_risk.model.two_head import balance_class_weights
from refund_abuse_risk.scoring.decision import DecisionStacker


def test_balance_class_weights_equalizes_mass() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    w = np.array([1.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    out = balance_class_weights(y, w)
    assert abs(out[y == 1].sum() - out[y == 0].sum()) < 1e-9
    assert out[0] / out[1] == w[0] / w[1]


def test_stacker_uses_slice_one_hots() -> None:
    abuse = np.array([20.0, 80.0, 20.0, 80.0, 15.0, 85.0])
    fraud = np.array([20.0, 80.0, 20.0, 80.0, 15.0, 85.0])
    y = np.array([0, 1, 0, 1, 0, 1])
    markets = np.array(["SG", "SG", "ID", "ID", "SG", "ID"])
    verticals = np.array(["food", "food", "food", "food", "qcommerce", "qcommerce"])
    stacker = DecisionStacker().fit(
        abuse, fraud, y, fit_mode="oof", markets=markets, verticals=verticals
    )
    assert stacker.use_slices is True
    assert stacker.market_levels
    scores = stacker.predict_scores(abuse, fraud, markets=markets, verticals=verticals)
    assert scores[1] > scores[0]


def test_edge_shuffle_null_filters_vs_disabled() -> None:
    rows = []
    for i in range(40):
        rows.append(
            {
                "user_id": "U_farm",
                "vendor_id": "V_farm",
                "market": "SG",
                "vertical": "food",
                "is_refund": 1 if i < 25 else 0,
                "order_id": f"F{i}",
            }
        )
    for i in range(120):
        rows.append(
            {
                "user_id": f"U{i}",
                "vendor_id": f"V{i % 15}",
                "market": "SG",
                "vertical": "food",
                "is_refund": int(i % 4 == 0),
                "order_id": f"R{i}",
            }
        )
    hist = pd.DataFrame(rows)
    cfg_base = {
        "min_edge_orders": 3,
        "min_edge_refunds": 2,
        "min_lift": 1.5,
        "min_refund_rate": 0.2,
    }
    edges_on, _ = score_uv_bipartite(
        hist,
        {
            **cfg_base,
            "null_baseline": {
                "enabled": True,
                "n_shuffles": 4,
                "percentile": 95.0,
                "min_history_rows": 10,
            },
        },
    )
    edges_off, _ = score_uv_bipartite(
        hist, {**cfg_base, "null_baseline": {"enabled": False}}
    )
    assert int(edges_on["elevated"].sum()) <= int(edges_off["elevated"].sum())
    # Direct null call exposes threshold audit fields.
    _, stats = apply_edge_shuffle_null(
        hist,
        edges_off,
        {
            "null_baseline": {
                "enabled": True,
                "n_shuffles": 3,
                "percentile": 95.0,
                "min_history_rows": 10,
            },
            **cfg_base,
        },
    )
    assert stats["null_threshold"] is not None
    assert stats["elevated_after"] <= stats["elevated_before"]


def test_null_baseline_disabled_passthrough() -> None:
    hist = pd.DataFrame(
        [
            {
                "user_id": "u",
                "vendor_id": "v",
                "market": "SG",
                "vertical": "food",
                "is_refund": 1,
                "order_id": f"o{i}",
            }
            for i in range(10)
        ]
    )
    raw = score_uv_bipartite(hist, {"null_baseline": {"enabled": False}})[0]
    filtered, stats = apply_edge_shuffle_null(
        hist, raw, {"null_baseline": {"enabled": False}}
    )
    assert stats["enabled"] is False
    assert int(filtered["elevated"].sum()) == int(raw["elevated"].sum())
