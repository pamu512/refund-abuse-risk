"""Regression tests for remaining audit High findings (H1/H2/H3/H5/H8)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from refund_abuse_risk.config import load_label_weights
from refund_abuse_risk.features.builders import (
    abuse_model_feature_columns,
    fraud_model_feature_columns,
)
from refund_abuse_risk.scoring.monitoring import expected_calibration_error
from refund_abuse_risk.training.splits import time_based_order_split


def test_h1_ece_reports_scale_and_positives() -> None:
    out = expected_calibration_error([0, 1], [10, 90], scores_are_0_100=True)
    assert out["scores_are_0_100"] is True
    assert out["n_positives"] == 1
    probs = expected_calibration_error([0, 1], [0.1, 0.9], scores_are_0_100=False)
    assert probs["scores_are_0_100"] is False


def test_h2_class_balance_default_off() -> None:
    lw = load_label_weights()
    assert lw.get("class_balance") is False


def test_h3_mint_features_excluded_from_both_heads() -> None:
    fraud_cols = fraud_model_feature_columns(exclude_proxy_mint=True)
    abuse_cols = abuse_model_feature_columns(exclude_proxy_mint=True)
    assert "device_cluster_size" not in fraud_cols
    assert "uvd_refund_lift" not in fraud_cols
    assert "user_refund_rate_30d" not in fraud_cols
    assert "device_cluster_size" not in abuse_cols
    # Abuse still keeps behavioral rates (not pure mint-graph features).
    assert "user_refund_rate_30d" in abuse_cols


def test_h5_adaptive_split_marks_temporal_not_ok() -> None:
    # 3-day span cannot honor 7-day holdout.
    rows = []
    for i in range(40):
        rows.append(
            {
                "order_id": f"o{i}",
                "event_ts": f"2026-07-0{(i % 3) + 1}T12:00:00Z",
            }
        )
    orders = pd.DataFrame(rows)
    _tr, _te, stats = time_based_order_split(
        orders, holdout_days=7, min_train=5, min_test=5
    )
    assert stats.get("adaptive") is True or stats.get("mode") == "time_oot_adaptive"
    assert stats.get("temporal_ok") is False


def test_h5_positional_fallback_temporal_not_ok() -> None:
    orders = pd.DataFrame([{"order_id": f"o{i}"} for i in range(20)])
    _tr, _te, stats = time_based_order_split(orders, holdout_days=7)
    assert stats["mode"] == "positional_fallback"
    assert stats["temporal_ok"] is False


def test_h8_serve_rate_limit_and_audit(caplog) -> None:
    from http.client import HTTPConnection
    import threading
    from http.server import HTTPServer

    from refund_abuse_risk.pipeline.score import OrderRiskCache
    from refund_abuse_risk.schemas.models import (
        EntityScores,
        EvidencePack,
        LinkScores,
        OrderRiskSnapshot,
        RefundEffect,
        SuggestedTier,
    )
    from refund_abuse_risk.serve.api import ScoreApiConfig, make_handler_class

    cache = OrderRiskCache()
    cache.put(
        OrderRiskSnapshot(
            order_id="O1",
            market="SG",
            vertical="food",
            abuse_score=10,
            fraud_score=10,
            decision_score=10,
            entity_scores=EntityScores(),
            link_scores=LinkScores(),
            suggested_tier=SuggestedTier.AUTO_APPROVE,
            refund_effect=RefundEffect.REFUND_AUTO_GRANT,
            evidence_pack=EvidencePack(),
        ),
        {"order_id": "O1"},
    )
    cfg = ScoreApiConfig(
        token="t",
        cache=cache,
        rate_limit_per_minute=2,
        audit_log=True,
    )
    server = HTTPServer(("127.0.0.1", 0), make_handler_class(cfg))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with caplog.at_level(logging.INFO, logger="refund_abuse_risk.serve"):
            conn = HTTPConnection("127.0.0.1", port, timeout=5)
            for _ in range(2):
                conn.request(
                    "GET", "/v1/orders/O1/risk", headers={"Authorization": "Bearer t"}
                )
                assert conn.getresponse().status == 200
            conn.request(
                "GET", "/v1/orders/O1/risk", headers={"Authorization": "Bearer t"}
            )
            assert conn.getresponse().status == 429
    finally:
        server.shutdown()
