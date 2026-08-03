from __future__ import annotations

from pathlib import Path

from refund_abuse_risk.baselines.cohort import CohortBaselineStore
from refund_abuse_risk.baselines.gate import (
    apply_precision_discount_to_operating_point,
    evaluate_baseline_gate,
)
from refund_abuse_risk.baselines.store import BehaviorBaselineStore
from refund_abuse_risk.config import load_behavior_baselines, load_operating_point
from refund_abuse_risk.schemas.models import SuggestedTier
from refund_abuse_risk.scoring.policy import combine_scores


def _cfg(**overrides):
    cfg = load_behavior_baselines()
    cfg["baseline_qualify"] = {
        **cfg["baseline_qualify"],
        "min_observations": 5,
        "min_under_rate": 0.8,
        "lookback_events": 20,
        "lookback_days": 30,
        "revoke_under_rate": 0.4,
        "pair_min_cooccur": 0,
    }
    cfg["elevation"] = {
        **cfg["elevation"],
        "min_abs_delta": 15.0,
        "min_lift": 1.5,
        "min_elevated_streak": 2,
        "require_clean_baseline": True,
    }
    cfg["precision_discount"] = {
        **cfg["precision_discount"],
        "max_auto_relax_points": 5.0,
    }
    cfg["settled_statuses"] = ["delivered", "claim"]
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def _order(i: int, **extra):
    base = {
        "order_id": f"O-{i}",
        "user_id": "U1",
        "driver_id": "D1",
        "vendor_id": "V1",
        "device_id": "DEV1",
        "market": "SG",
        "vertical": "food",
        "status": "delivered",
        "event_ts": f"2026-07-{1 + (i % 28):02d}T12:00:00Z",
        "user_days_since_signup": 120,
        "ud_cooccur": 5,
        "uv_cooccur": 5,
        "vd_cooccur": 5,
        "uvd_cooccur": 5,
    }
    base.update(extra)
    return base


def test_head_score_clean_then_abuse_discount(tmp_path: Path) -> None:
    store = BehaviorBaselineStore(tmp_path / "b.db")
    cfg = _cfg()
    op = load_operating_point()

    for i in range(5):
        evaluate_baseline_gate(
            _order(i),
            abuse_score=10.0,
            fraud_score=5.0,
            store=store,
            baseline_cfg=cfg,
            operating_point=op,
            update_store=True,
        )
    user = store.get("user:U1", market="SG", vertical="food")
    assert user is not None
    assert user.is_clean_baseline is True
    assert user.score_head == "abuse"

    r1 = evaluate_baseline_gate(
        _order(10),
        abuse_score=55.0,
        fraud_score=5.0,
        store=store,
        baseline_cfg=cfg,
        operating_point=op,
    )
    assert r1.abuse_precision_discount == 0.0

    r2 = evaluate_baseline_gate(
        _order(11),
        abuse_score=60.0,
        fraud_score=5.0,
        store=store,
        baseline_cfg=cfg,
        operating_point=op,
    )
    assert r2.abuse_precision_discount > 0.0
    assert r2.fraud_precision_discount == 0.0
    assert any(s.entity_kind == "user" and s.elevated for s in r2.signals)


def test_idempotent_per_order(tmp_path: Path) -> None:
    store = BehaviorBaselineStore(tmp_path / "b.db")
    cfg = _cfg()
    op = load_operating_point()
    order = _order(0)
    evaluate_baseline_gate(
        order, abuse_score=10.0, fraud_score=5.0, store=store, baseline_cfg=cfg, operating_point=op
    )
    evaluate_baseline_gate(
        order, abuse_score=10.0, fraud_score=5.0, store=store, baseline_cfg=cfg, operating_point=op
    )
    user = store.get("user:U1", market="SG", vertical="food")
    assert user is not None
    assert user.n_observations == 1


def test_unsettle_status_skips_write(tmp_path: Path) -> None:
    store = BehaviorBaselineStore(tmp_path / "b.db")
    cfg = _cfg()
    op = load_operating_point()
    evaluate_baseline_gate(
        _order(0, status="placed"),
        abuse_score=10.0,
        fraud_score=5.0,
        store=store,
        baseline_cfg=cfg,
        operating_point=op,
    )
    assert store.get("user:U1", market="SG", vertical="food") is None


def test_head_attributed_relax_does_not_warp_other_head() -> None:
    op = load_operating_point()
    adjusted = apply_precision_discount_to_operating_point(
        op,
        abuse_relax_points=5.0,
        fraud_relax_points=0.0,
        abuse_precision_discount=0.05,
        fraud_precision_discount=0.0,
    )
    assert adjusted["head_thresholds"]["abuse_soft_friction"] == op["head_thresholds"][
        "abuse_soft_friction"
    ] - 5.0
    assert adjusted["head_thresholds"]["fraud_soft_friction"] == op["head_thresholds"][
        "fraud_soft_friction"
    ]
    soft = float(op["head_thresholds"]["abuse_soft_friction"])
    _, tier_before = combine_scores(soft - 1, 0, 0, op, hard_gated=False)
    _, tier_after = combine_scores(soft - 1, 0, 0, adjusted, hard_gated=False)
    assert tier_before == SuggestedTier.AUTO_APPROVE
    assert tier_after == SuggestedTier.SOFT_FRICTION


def test_hil_caps_large_relax(tmp_path: Path) -> None:
    store = BehaviorBaselineStore(tmp_path / "b.db")
    cfg = _cfg()
    cfg["precision_discount"]["max_auto_relax_points"] = 2.0
    cfg["precision_discount"]["threshold_relax_per_pp"] = 2.0
    op = load_operating_point()
    for i in range(5):
        evaluate_baseline_gate(
            _order(i),
            abuse_score=8.0,
            fraud_score=5.0,
            store=store,
            baseline_cfg=cfg,
            operating_point=op,
        )
    for i in range(5, 8):
        r = evaluate_baseline_gate(
            _order(i),
            abuse_score=90.0,
            fraud_score=5.0,
            store=store,
            baseline_cfg=cfg,
            operating_point=op,
        )
    assert r.abuse_relax_points <= 2.0 + 1e-9
    assert r.hil_required is True
    assert any(i.reason_code == "BASELINE_HIL_RELAX" for i in r.evidence_items)


def test_warehouse_snapshot_export(tmp_path: Path) -> None:
    store = BehaviorBaselineStore(tmp_path / "b.db")
    cohort = CohortBaselineStore(tmp_path / "b.db")
    cfg = _cfg()
    op = load_operating_point()
    for i in range(5):
        evaluate_baseline_gate(
            _order(i, user_id="U3", driver_id="D3", vendor_id="V3", market="ID"),
            abuse_score=5.0,
            fraud_score=4.0,
            store=store,
            cohort_store=cohort,
            baseline_cfg=cfg,
            operating_point=op,
        )
    out = tmp_path / "as_of_date=2026-08-03" / "part.csv"
    store.export_csv(out, as_of_date="2026-08-03", baseline_version="0.3.0")
    text = out.read_text(encoding="utf-8")
    assert "as_of_date" in text
    assert "baseline_version" in text
    assert "score_head" in text
    cohort_out = tmp_path / "cohort" / "part.csv"
    cohort.export_csv(cohort_out, as_of_date="2026-08-03", baseline_version="0.3.0")
    assert "tenure_bucket" in cohort_out.read_text(encoding="utf-8")


def test_cohort_elevation_without_clean_self(tmp_path: Path) -> None:
    """Stable-high self-baseline still discounts via cohort peer lift."""
    db = tmp_path / "b.db"
    store = BehaviorBaselineStore(db)
    cohort = CohortBaselineStore(db)
    cfg = _cfg()
    cfg["cohort"] = {
        **cfg.get("cohort", {}),
        "enabled": True,
        "min_support": 10,
        "lookback_events": 200,
        "min_lift": 1.5,
        "min_abs_delta": 15.0,
    }
    op = load_operating_point()
    # Peer cohort: many low-abuse orders from other users.
    for i in range(15):
        evaluate_baseline_gate(
            _order(i, user_id=f"Upeer{i}", driver_id=f"Dp{i}", vendor_id=f"Vp{i}", device_id=f"DVp{i}"),
            abuse_score=12.0,
            fraud_score=8.0,
            store=store,
            cohort_store=cohort,
            baseline_cfg=cfg,
            operating_point=op,
        )
    # Target user: only high scores → never qualifies as clean self-baseline.
    for i in range(15, 20):
        evaluate_baseline_gate(
            _order(i, user_id="Ubad"),
            abuse_score=70.0,
            fraud_score=8.0,
            store=store,
            cohort_store=cohort,
            baseline_cfg=cfg,
            operating_point=op,
        )
    user = store.get("user:Ubad", market="SG", vertical="food")
    assert user is not None
    assert user.is_clean_baseline is False

    hit = evaluate_baseline_gate(
        _order(99, user_id="Ubad"),
        abuse_score=75.0,
        fraud_score=8.0,
        store=store,
        cohort_store=cohort,
        baseline_cfg=cfg,
        operating_point=op,
    )
    assert hit.abuse_precision_discount > 0.0
    assert any(i.reason_code == "COHORT_ELEVATED_USER" for i in hit.evidence_items)
    assert hit.cohort_features["user_cohort_lift"] > 1.0


def test_device_baseline_and_trust_credit(tmp_path: Path) -> None:
    store = BehaviorBaselineStore(tmp_path / "b.db")
    cfg = _cfg()
    cfg["trust_credit"] = {
        "enabled": True,
        "require_user_clean": True,
        "require_device_clean": True,
        "max_abuse_score": 25.0,
        "max_fraud_score": 20.0,
        "abuse_tighten_points": 5.0,
        "fraud_tighten_points": 5.0,
    }
    op = load_operating_point()
    for i in range(5):
        evaluate_baseline_gate(
            _order(i),
            abuse_score=8.0,
            fraud_score=4.0,
            store=store,
            baseline_cfg=cfg,
            operating_point=op,
        )
    device = store.get("device:DEV1", market="SG", vertical="food")
    assert device is not None
    assert device.is_clean_baseline is True
    assert device.score_head == "fraud"

    trusted = evaluate_baseline_gate(
        _order(20),
        abuse_score=10.0,
        fraud_score=5.0,
        store=store,
        baseline_cfg=cfg,
        operating_point=op,
    )
    assert trusted.trust_credit is True
    assert trusted.abuse_tighten_points == 5.0
    assert any(i.reason_code == "BASELINE_TRUST_CREDIT" for i in trusted.evidence_items)

    tightened = apply_precision_discount_to_operating_point(
        op,
        abuse_tighten_points=trusted.abuse_tighten_points,
        fraud_tighten_points=trusted.fraud_tighten_points,
    )
    assert tightened["head_thresholds"]["abuse_soft_friction"] == op["head_thresholds"][
        "abuse_soft_friction"
    ] + 5.0
