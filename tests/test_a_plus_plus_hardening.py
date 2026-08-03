from __future__ import annotations

import pandas as pd

from refund_abuse_risk.config import load_label_weights, load_refund_budget
from refund_abuse_risk.labels.discovery import mint_weak_labels_from_uv_anomaly
from refund_abuse_risk.model.two_head import apply_proxy_fraud_labels, sample_weights_for_frame
from refund_abuse_risk.schemas.models import RefundEffect
from refund_abuse_risk.scoring.budget import apply_budget_effect_floor, evaluate_refund_budget
from refund_abuse_risk.scoring.monitoring import evaluate_monitoring_gates


def _collusion_history() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append(
            {
                "order_id": f"h{i}",
                "user_id": "u_bad",
                "vendor_id": "v_bad",
                "driver_id": "d1",
                "market": "SG",
                "vertical": "food",
                "amount": 30,
                "is_refund": 1 if i < 4 else 0,
                "event_ts": f"2026-06-{10 + i:02d}T00:00:00Z",
            }
        )
    return pd.DataFrame(rows)


def test_discovery_is_as_of_and_skips_future_history() -> None:
    history = _collusion_history()
    # Order before any elevated support exists → no mint.
    early = pd.DataFrame(
        [
            {
                "order_id": "o_early",
                "user_id": "u_bad",
                "vendor_id": "v_bad",
                "market": "SG",
                "vertical": "food",
                "event_ts": "2026-06-10T12:00:00Z",
                "fraud_label": 0,
                "fraud_label_source": "",
                "abuse_label": 0,
            }
        ]
    )
    out_early = mint_weak_labels_from_uv_anomaly(early, history)
    assert int(out_early.attrs.get("discovery_minted", 0)) == 0

    late = pd.DataFrame(
        [
            {
                "order_id": "o_late",
                "user_id": "u_bad",
                "vendor_id": "v_bad",
                "market": "SG",
                "vertical": "food",
                "event_ts": "2026-06-20T00:00:00Z",
                "fraud_label": 0,
                "fraud_label_source": "",
                "abuse_label": 0,
            },
            {
                "order_id": "o_proven",
                "user_id": "u_bad",
                "vendor_id": "v_bad",
                "market": "SG",
                "vertical": "food",
                "event_ts": "2026-06-20T00:00:00Z",
                "fraud_label": 1,
                "fraud_label_source": "proven",
                "abuse_label": 1,
            },
        ]
    )
    out = mint_weak_labels_from_uv_anomaly(late, history)
    assert int(out.attrs.get("discovery_minted", 0)) >= 1
    row_new = out.loc[out["order_id"] == "o_late"].iloc[0]
    row_proven = out.loc[out["order_id"] == "o_proven"].iloc[0]
    assert str(row_new["fraud_label_source"]) == "discovery"
    assert int(row_new["abuse_label_weak"]) == 1
    assert str(row_proven["fraud_label_source"]) == "proven"


def test_discovery_weight_and_proxy_does_not_overwrite() -> None:
    lw = load_label_weights()
    frame = pd.DataFrame(
        [
            {
                "fraud_label": 1,
                "fraud_label_source": "discovery",
                "device_cluster_size": 10,
                "accounts_per_device": 10,
                "uvd_refund_lift": 3.0,
                "uvd_refund_share": 0.9,
                "uvd_cooccur": 10,
                "user_refund_rate_30d": 0.9,
                "user_orders_30d": 20,
                "device_risk_score": 90,
                "is_emulator": 1,
                "is_cloned_app": 1,
                "is_gps_spoof": 1,
                "customer_courier_same_device": 1,
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
            }
        ]
    )
    out = apply_proxy_fraud_labels(frame, lw)
    assert str(out.iloc[0]["fraud_label_source"]) == "discovery"
    w = sample_weights_for_frame(out, lw, head="fraud")
    assert abs(float(w[0]) - float(lw["fraud_discovery_weight"])) < 1e-9


def test_monitoring_gate_blocks_on_ece_breach() -> None:
    ok = evaluate_monitoring_gates(
        decision_ece=0.05,
        psi_train_test=0.1,
        monitoring_cfg={"max_decision_ece": 0.2, "max_psi_train_test": 0.35},
    )
    assert ok["ok"] is True
    bad = evaluate_monitoring_gates(
        decision_ece=0.5,
        psi_train_test=0.1,
        monitoring_cfg={"max_decision_ece": 0.2, "max_psi_train_test": 0.35},
    )
    assert bad["ok"] is False
    assert bad["reasons"]


def test_monitoring_gate_blocks_on_ops_snapshot() -> None:
    bad = evaluate_monitoring_gates(
        decision_ece=0.01,
        psi_train_test=0.01,
        monitoring_cfg={
            "max_refund_dollar_per_order": 5.0,
            "ops_snapshot": {"refund_dollar_per_order": 12.0},
        },
    )
    assert bad["ok"] is False
    assert any("refund_dollar_per_order" in r for r in bad["reasons"])


def test_budget_floors_weaker_effect() -> None:
    cfg = load_refund_budget()
    budget = evaluate_refund_budget(
        {"user_refund_count_30d": 5, "user_refund_gmv_pct_30d": 0.5}, cfg
    )
    assert budget.pressure == "exhausted"
    effect, budget2, reasons = apply_budget_effect_floor(
        RefundEffect.REFUND_AUTO_GRANT, budget, cfg
    )
    assert effect == RefundEffect.REFUND_MANUAL_REVIEW
    assert budget2.floored is True
    assert "BUDGET_FLOOR" in reasons

    # Already stricter → no floor.
    effect2, _, reasons2 = apply_budget_effect_floor(
        RefundEffect.REFUND_BLOCK, budget, cfg
    )
    assert effect2 == RefundEffect.REFUND_BLOCK
    assert reasons2 == []
