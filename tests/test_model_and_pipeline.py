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
def demo_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (DATA / "orders.csv").exists():
        import runpy

        runpy.run_path(str(ROOT / "scripts" / "generate_demo_data.py"), run_name="__main__")
    return (
        pd.read_csv(DATA / "orders.csv"),
        pd.read_csv(DATA / "history.csv"),
        pd.read_csv(DATA / "devices.csv"),
    )


def test_proxy_labels_applied(demo_frames) -> None:
    orders, history, devices = demo_frames
    feat = build_order_feature_frame(orders, history, devices)
    labeled = apply_proxy_fraud_labels(feat, load_label_weights())
    proxy = labeled[labeled["fraud_label_source"] == "proxy"]
    proven = labeled[labeled["fraud_label_source"] == "proven"]
    assert len(proven) >= 1
    assert len(proxy) >= 1


def test_two_head_train_and_score(demo_frames) -> None:
    orders, history, devices = demo_frames
    model = train_two_head(orders, history, devices)
    feat = build_order_feature_frame(orders.head(5), history, devices)
    scored = model.predict_proba(feat)
    assert "abuse_score" in scored.columns
    assert "fraud_score" in scored.columns
    assert scored["abuse_score"].between(0, 100).all()
    assert scored["fraud_score"].between(0, 100).all()


def test_precompute_and_claim_path(demo_frames) -> None:
    orders, history, devices = demo_frames
    model = train_two_head(orders, history, devices)
    cache = precompute_orders(orders, history, devices, model)
    snap = claim_path_read("O-FRAUD-0", cache)
    assert snap.order_id == "O-FRAUD-0"
    assert snap.hard_gated is True
    assert snap.suggested_tier == SuggestedTier.AUTO_DENY
    assert snap.evidence_pack.items
    assert "STRONG_FRAUD_LABEL" in snap.reason_codes or any(
        c.startswith("USER_") or c.startswith("LINK_") or c.startswith("DEVICE_")
        for c in snap.reason_codes
    )


def test_entity_risk_change_rescores(demo_frames) -> None:
    orders, history, devices = demo_frames
    model = train_two_head(orders, history, devices)
    cache = precompute_orders(orders, history, devices, model)
    key = link_key("driver", "DF0")
    refreshed = on_entity_risk_change(
        {key},
        orders,
        history,
        devices,
        model,
        cache,
        new_scores={key: cache.standing_entity_score(key) + 25.0},
    )
    assert len(refreshed) >= 1


def test_model_save_load(tmp_path, demo_frames) -> None:
    orders, history, devices = demo_frames
    model = train_two_head(orders, history, devices)
    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = TwoHeadModel.load(path)
    feat = build_order_feature_frame(orders.head(3), history, devices)
    a = model.predict_proba(feat)["abuse_score"].tolist()
    b = loaded.predict_proba(feat)["abuse_score"].tolist()
    assert a == pytest.approx(b)
