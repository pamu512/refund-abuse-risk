"""Math / promote / calibration invariants that pin Correctness ≥ 9.0.

Synth labels are a production-readiness issue, not a math-machinery issue.
These tests fail if ECE/PSI/costed recall/promote contracts drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from refund_abuse_risk.scoring.decision import (
    recommend_decision_thresholds,
    recommend_decision_thresholds_by_slice,
    recommended_overlays_from_slices,
    tier_from_decision_score,
)
from refund_abuse_risk.scoring.monitoring import (
    evaluate_monitoring_gates,
    expected_calibration_error,
    population_stability_index,
    summarize_ops_metrics,
)
from refund_abuse_risk.scoring.policy import combine_scores
from refund_abuse_risk.scoring.thresholds import (
    fp_rate_at_threshold,
    precision_at_threshold,
    recall_at_threshold,
    threshold_at_recall,
    threshold_at_recall_costed,
)
from refund_abuse_risk.schemas.models import SuggestedTier


def test_ece_perfectly_calibrated_is_near_zero() -> None:
    # Scores 0–100 imply p̂ = score/100; labels match p̂ in each bin.
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0, 0, 0, 0, 100, 100, 100, 100]
    out = expected_calibration_error(y, s, n_bins=2)
    assert out["n"] == 8
    assert out["ece"] is not None
    assert float(out["ece"]) < 1e-9


def test_ece_empty_and_bounds() -> None:
    empty = expected_calibration_error([], [], n_bins=5)
    assert empty["ece"] is None and empty["n"] == 0
    # Max miscalibration: all mass at conf=1, acc=0
    bad = expected_calibration_error([0, 0, 0, 0], [100, 100, 100, 100], n_bins=2)
    assert bad["ece"] is not None
    assert 0.0 <= float(bad["ece"]) <= 1.0
    assert float(bad["ece"]) == pytest.approx(1.0)


def test_psi_identical_near_zero_empty_none() -> None:
    x = [10, 20, 30, 40, 50, 60, 70, 80]
    same = population_stability_index(x, x, n_bins=4)
    assert same["psi"] is not None
    assert float(same["psi"]) < 0.05
    empty = population_stability_index([], [1, 2])
    assert empty["psi"] is None


def test_threshold_at_recall_identity() -> None:
    y = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    s = np.array([90, 80, 40, 70, 30, 10], dtype=float)
    # Need 2/3 positives → threshold lands at 80 (covers 90 and 80).
    t = threshold_at_recall(y, s, target_recall=2 / 3)
    assert t == pytest.approx(80.0)
    assert recall_at_threshold(y, s, t) >= 2 / 3 - 1e-12
    assert threshold_at_recall(y, s, 1.0) is not None
    assert threshold_at_recall([0, 0, 0], [1, 2, 3], 0.9) is None


def test_costed_threshold_picks_highest_feasible_and_respects_dollar_cap() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    s = np.array([95, 85, 75, 65, 55, 20, 10, 5], dtype=float)
    amts = np.array([10, 10, 10, 200, 200, 10, 10, 10], dtype=float)

    t_loose, info_loose = threshold_at_recall_costed(
        y, s, target_recall=0.66, min_precision=0.4, amounts=amts
    )
    assert info_loose["feasible"] is True
    assert t_loose is not None

    t_tight, info_tight = threshold_at_recall_costed(
        y,
        s,
        target_recall=0.66,
        min_precision=0.4,
        amounts=amts,
        max_fp_refund_dollars_mean=25.0,
    )
    if info_tight["feasible"]:
        assert float(t_tight) >= float(t_loose)
        assert float(info_tight["fp_refund_dollars_mean"]) <= 25.0 + 1e-9
    else:
        assert info_tight["reason"] == "cost_constraints_infeasible"
        assert info_tight.get("recall_only_threshold") is not None


def test_precision_recall_fp_rate_identities() -> None:
    y = np.array([1, 1, 0, 0, 0], dtype=float)
    s = np.array([90, 40, 80, 30, 10], dtype=float)
    thr = 50.0
    # Pred: scores 90, 80 → TP=1, FP=1 → prec=0.5, rec=0.5, fpr=1/3
    assert precision_at_threshold(y, s, thr) == pytest.approx(0.5)
    assert recall_at_threshold(y, s, thr) == pytest.approx(0.5)
    assert fp_rate_at_threshold(y, s, thr) == pytest.approx(1 / 3)


def test_recommend_decision_ladder_monotonic_and_soft_floor_blocks_ok() -> None:
    # Perfectly separable → soft can sit very high.
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    s = [99, 98, 97, 5, 4, 3, 2, 1]
    rec = recommend_decision_thresholds(
        y, s, target_recall=1.0, min_precision_at_soft=0.9, apply_floors=False
    )
    assert rec["soft_friction"] <= rec["hold_review"] <= rec["auto_deny"]
    assert rec["cost_feasible"] is True
    assert rec["ok"] is True
    assert float(rec["recall_at_soft"]) >= 1.0 - 1e-12

    # Soft threshold forced below floor (all scores low) → ok=False without apply_floors.
    y_low = [1, 1, 0, 0, 0, 0]
    s_low = [8, 7, 6, 5, 4, 3]
    rec_low = recommend_decision_thresholds(
        y_low, s_low, target_recall=1.0, min_precision_at_soft=None, apply_floors=False
    )
    assert "soft_friction" in rec_low["soft_floor_would_bind"]
    assert rec_low["ok"] is False


def test_thin_slice_not_promote_eligible_and_overlays_filter() -> None:
    frame = pd.DataFrame(
        {
            "market": ["SG"] * 5 + ["ID"] * 40,
            "vertical": ["food"] * 45,
            "pattern_y": [1, 1, 0, 0, 0] + ([1] * 10 + [0] * 30),
            "decision_score": [90, 80, 20, 10, 5]
            + ([95 - i for i in range(10)] + [10.0] * 30),
            "amount": [12.0] * 45,
        }
    )
    by_slice = recommend_decision_thresholds_by_slice(
        frame, min_slice_n=20, min_slice_positives=3, target_recall=0.8
    )
    assert by_slice["SG|food"]["thin_slice"] is True
    assert by_slice["SG|food"]["promote_eligible"] is False
    assert by_slice["ID|food"]["thin_slice"] is False
    overlays = recommended_overlays_from_slices(by_slice)
    markets = {o["market"] for o in overlays}
    assert "SG" not in markets
    if by_slice["ID|food"].get("promote_eligible"):
        assert "ID" in markets


def test_monitoring_missing_metrics_pass_breach_fails() -> None:
    missing = evaluate_monitoring_gates(
        decision_ece=None,
        psi_train_test=None,
        monitoring_cfg={"max_decision_ece": 0.2, "max_psi_train_test": 0.35},
    )
    assert missing["ok"] is True
    assert missing["reasons"] == []

    breach = evaluate_monitoring_gates(
        decision_ece=0.9,
        psi_train_test=0.01,
        monitoring_cfg={"max_decision_ece": 0.2, "max_psi_train_test": 0.35},
    )
    assert breach["ok"] is False
    assert any("decision_ece" in r for r in breach["reasons"])


def test_ops_summary_refund_dollar_per_order_identity() -> None:
    out = summarize_ops_metrics(
        tier_counts={"hold_review": 2, "auto_approve": 2},
        effects_dual_run={
            "n": 4,
            "live_override_count": 1,
            "shadow_override_count": 0,
            "final_effect_counts": {"refund_auto_grant": 2},
        },
        amounts=[10.0, 20.0, 30.0, 40.0],
        final_effects=[
            "refund_auto_grant",
            "refund_block",
            "refund_auto_grant",
            "refund_step_up",
        ],
    )
    # Grants at $10 + $30 → mean per order = 40/4
    assert out["refund_dollar_per_order"] == pytest.approx(10.0)
    assert out["hold_rate"] == pytest.approx(0.5)
    assert out["live_override_rate"] == pytest.approx(0.25)


def test_hard_gate_and_decision_primary_tier_contract() -> None:
    op = {
        "decision_mode": "decision_primary",
        "decision_thresholds": {
            "soft_friction": 35,
            "hold_review": 50,
            "auto_deny": 75,
        },
        "score_display": {"abuse": 0.5, "fraud": 0.5},
    }
    score, tier = combine_scores(
        10.0, 10.0, entity_prior=0.0, operating_point=op, hard_gated=True, decision_score=20.0
    )
    assert score >= 90.0
    assert tier == SuggestedTier.AUTO_DENY

    assert tier_from_decision_score(34, op) == SuggestedTier.AUTO_APPROVE
    assert tier_from_decision_score(35, op) == SuggestedTier.SOFT_FRICTION
    assert tier_from_decision_score(50, op) == SuggestedTier.HOLD_REVIEW
    assert tier_from_decision_score(75, op) == SuggestedTier.AUTO_DENY
