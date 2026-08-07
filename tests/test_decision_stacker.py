from __future__ import annotations

import numpy as np

from refund_abuse_risk.config import load_operating_point
from refund_abuse_risk.schemas.models import SuggestedTier
from refund_abuse_risk.scoring.decision import (
    DecisionStacker,
    recommend_decision_thresholds,
    tier_from_decision_score,
)
from refund_abuse_risk.scoring.policy import combine_scores


def test_stacker_ranks_joint_risk() -> None:
    abuse = np.array([10.0, 80.0, 20.0, 70.0])
    fraud = np.array([10.0, 20.0, 80.0, 70.0])
    y = np.array([0, 1, 1, 1])
    stacker = DecisionStacker().fit(abuse, fraud, y)
    scores = stacker.predict_scores(abuse, fraud)
    assert scores[0] < scores[1]
    assert scores[0] < scores[2]


def test_tier_from_decision_score_ladder() -> None:
    op = load_operating_point()
    assert tier_from_decision_score(10, op) == SuggestedTier.AUTO_APPROVE
    assert tier_from_decision_score(40, op) == SuggestedTier.SOFT_FRICTION
    assert tier_from_decision_score(55, op) == SuggestedTier.HOLD_REVIEW
    assert tier_from_decision_score(80, op) == SuggestedTier.AUTO_DENY


def test_combine_decision_primary() -> None:
    op = load_operating_point()
    assert op["decision_mode"] == "decision_primary"
    _, tier = combine_scores(90, 10, 0, op, hard_gated=False, decision_score=20)
    # Decision score wins over high abuse head.
    assert tier == SuggestedTier.AUTO_APPROVE


def test_recommend_decision_thresholds_costed() -> None:
    y = [1] * 16 + [0] * 24
    s = [95 - i for i in range(16)] + [20.0] * 24
    rec = recommend_decision_thresholds(y, s, target_recall=0.75, min_precision_at_soft=0.5)
    assert rec["ok"] is True
    assert rec["precision_ci_ok"] is True
    assert rec["soft_friction"] <= rec["hold_review"] <= rec["auto_deny"]
