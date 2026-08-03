from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from refund_abuse_risk.config import load_label_weights
from refund_abuse_risk.features.builders import build_order_feature_frame
from refund_abuse_risk.graph.entities import link_key
from refund_abuse_risk.model.two_head import TwoHeadModel, apply_proxy_fraud_labels
from refund_abuse_risk.pipeline.score import (
    claim_path_read,
    on_entity_risk_change,
    precompute_orders,
    train_two_head,
)
from refund_abuse_risk.schemas.models import SuggestedTier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def demo_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import runpy

    runpy.run_path(str(ROOT / "scripts" / "generate_demo_data.py"), run_name="__main__")
    return (
        pd.read_csv(DATA / "orders.csv"),
        pd.read_csv(DATA / "history.csv"),
        pd.read_csv(DATA / "devices.csv"),
        pd.read_csv(DATA / "users.csv"),
    )


def test_proxy_labels_applied(demo_frames) -> None:
    orders, history, devices, users = demo_frames
    feat = build_order_feature_frame(orders, history, devices, users=users)
    labeled = apply_proxy_fraud_labels(feat, load_label_weights())
    proxy = labeled[labeled["fraud_label_source"] == "proxy"]
    proven = labeled[labeled["fraud_label_source"] == "proven"]
    assert len(proven) >= 1
    assert len(proxy) >= 1


def test_two_head_train_and_score(demo_frames) -> None:
    orders, history, devices, users = demo_frames
    model = train_two_head(orders, history, devices, users=users)
    feat = build_order_feature_frame(orders.head(5), history, devices, users=users)
    scored = model.predict_proba(feat)
    assert "abuse_score" in scored.columns
    assert "fraud_score" in scored.columns
    assert scored["abuse_score"].between(0, 100).all()
    assert scored["fraud_score"].between(0, 100).all()


def test_precompute_and_claim_path(demo_frames) -> None:
    orders, history, devices, users = demo_frames
    model = train_two_head(orders, history, devices, users=users)
    cache = precompute_orders(orders, history, devices, model, users=users)
    snap = claim_path_read("O-FRAUD-0", cache)
    assert snap.order_id == "O-FRAUD-0"
    assert snap.hard_gated is True
    assert snap.suggested_tier == SuggestedTier.AUTO_DENY
    assert "STRONG_FRAUD_LABEL" in snap.reason_codes


def test_new_user_not_hard_gated_on_thin_history(demo_frames) -> None:
    orders, history, devices, users = demo_frames
    model = train_two_head(orders, history, devices, users=users)
    cache = precompute_orders(orders, history, devices, model, users=users)
    snap = claim_path_read("O-NEW-0", cache)
    assert snap.hard_gated is False


def test_pattern_orders_flagged_by_heads_not_gates(demo_frames) -> None:
    """Ring / LTV-burn / serial abuse should be caught by scores, not hard gates."""
    orders, history, devices, users = demo_frames
    model = train_two_head(orders, history, devices, users=users)
    cache = precompute_orders(orders, history, devices, model, users=users)

    for order_id in ("O-NEWRING-0", "O-BURN-0", "O-ABUSE-0"):
        snap = claim_path_read(order_id, cache)
        assert snap.hard_gated is False
        assert max(snap.abuse_score, snap.fraud_score) >= 30.0
        assert snap.suggested_tier != SuggestedTier.AUTO_APPROVE


def test_entity_risk_change_rescores(demo_frames) -> None:
    orders, history, devices, users = demo_frames
    model = train_two_head(orders, history, devices, users=users)
    cache = precompute_orders(orders, history, devices, model, users=users)
    key = link_key("driver", "DF0")
    refreshed = on_entity_risk_change(
        {key},
        orders,
        history,
        devices,
        model,
        cache,
        new_scores={key: cache.standing_entity_score(key) + 25.0},
        users=users,
    )
    assert len(refreshed) >= 1


def test_model_save_load(tmp_path, demo_frames) -> None:
    orders, history, devices, users = demo_frames
    model = train_two_head(orders, history, devices, users=users)
    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = TwoHeadModel.load(path)
    feat = build_order_feature_frame(orders.head(3), history, devices, users=users)
    a = model.predict_proba(feat)["abuse_score"].tolist()
    b = loaded.predict_proba(feat)["abuse_score"].tolist()
    assert a == pytest.approx(b)
