from __future__ import annotations

import numpy as np

from refund_abuse_risk.config import load_head_hyperparams
from refund_abuse_risk.model.two_head import _make_head
from refund_abuse_risk.scoring.decision import recommend_decision_thresholds
from refund_abuse_risk.scoring.slice_calibrator import SliceCalibrator
from refund_abuse_risk.scoring.thresholds import threshold_at_recall_costed


def test_per_head_hyperparams_differ() -> None:
    hp = load_head_hyperparams()
    assert hp["abuse"]["max_depth"] != hp["fraud"]["max_depth"]
    abuse = _make_head("abuse", hp)
    fraud = _make_head("fraud", hp)
    assert abuse.named_steps["clf"].estimator.max_depth == hp["abuse"]["max_depth"]
    assert fraud.named_steps["clf"].estimator.max_depth == hp["fraud"]["max_depth"]


def test_fp_refund_dollar_cost_constraint() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([90, 80, 70, 60, 55, 20, 10, 5], dtype=float)
    # High-dollar FPs at mid scores.
    amts = np.array([10, 10, 10, 100, 100, 10, 10, 10], dtype=float)
    t_loose, info_loose = threshold_at_recall_costed(
        y, s, target_recall=0.66, min_precision=0.4, amounts=amts
    )
    t_tight, info_tight = threshold_at_recall_costed(
        y,
        s,
        target_recall=0.66,
        min_precision=0.4,
        amounts=amts,
        max_fp_refund_dollars_mean=20.0,
    )
    assert info_loose["feasible"] is True
    # Tight $ budget should force a higher (stricter) threshold or mark infeasible.
    if info_tight["feasible"]:
        assert float(t_tight) >= float(t_loose)


def test_slice_calibrator_fits_global() -> None:
    scores = np.array([10, 20, 80, 90, 15, 85], dtype=float)
    y = np.array([0, 0, 1, 1, 0, 1])
    markets = np.array(["SG"] * 6)
    verticals = np.array(["food"] * 6)
    cal = SliceCalibrator(min_slice_n=4, min_slice_positives=2).fit(
        scores, y, markets, verticals
    )
    out = cal.transform(scores, markets, verticals)
    assert out[3] > out[0]


def test_recommend_decision_accepts_amounts() -> None:
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    s = [90, 85, 80, 40, 30, 20, 10, 5]
    amts = [12, 12, 12, 12, 12, 12, 12, 12]
    rec = recommend_decision_thresholds(
        y, s, target_recall=0.66, min_precision_at_soft=0.4, amounts=amts
    )
    assert "soft_friction" in rec
