from __future__ import annotations

import numpy as np
import pandas as pd

from refund_abuse_risk.config import load_operating_point
from refund_abuse_risk.schemas.models import SuggestedTier
from refund_abuse_risk.scoring.decision import (
    DecisionStacker,
    recommend_decision_thresholds_by_slice,
    resolve_decision_thresholds,
    tier_from_decision_score,
)
from refund_abuse_risk.scoring.policy import combine_scores


def test_resolve_decision_threshold_overlay() -> None:
    op = {
        "decision_thresholds": {
            "soft_friction": 35,
            "hold_review": 50,
            "auto_deny": 75,
        },
        "decision_threshold_overlays": [
            {
                "market": "SG",
                "vertical": "food",
                "soft_friction": 20,
                "hold_review": 40,
                "auto_deny": 60,
            }
        ],
    }
    thr = resolve_decision_thresholds(op, market="SG", vertical="food")
    assert thr["soft_friction"] == 20
    assert thr["auto_deny"] == 60
    # Unmatched slice keeps global.
    thr2 = resolve_decision_thresholds(op, market="ID", vertical="food")
    assert thr2["soft_friction"] == 35
    assert tier_from_decision_score(25, op, market="SG", vertical="food") == (
        SuggestedTier.SOFT_FRICTION
    )
    assert tier_from_decision_score(25, op, market="ID", vertical="food") == (
        SuggestedTier.AUTO_APPROVE
    )


def test_combine_scores_uses_slice_ladder() -> None:
    op = load_operating_point()
    op = dict(op)
    op["decision_threshold_overlays"] = [
        {
            "market": "SG",
            "vertical": "qcommerce",
            "soft_friction": 10,
            "hold_review": 20,
            "auto_deny": 90,
        }
    ]
    _, tier = combine_scores(
        10,
        10,
        0,
        op,
        hard_gated=False,
        decision_score=15,
        market="SG",
        vertical="qcommerce",
    )
    assert tier == SuggestedTier.SOFT_FRICTION


def test_recommend_by_slice_marks_thin() -> None:
    frame = pd.DataFrame(
        {
            "market": ["SG", "SG", "ID", "ID"],
            "vertical": ["food", "food", "food", "food"],
            "pattern_y": [1, 0, 1, 1],
            "decision_score": [80.0, 10.0, 70.0, 60.0],
        }
    )
    recs = recommend_decision_thresholds_by_slice(
        frame, min_slice_n=10, min_slice_positives=3
    )
    assert recs["SG|food"]["thin_slice"] is True
    assert recs["SG|food"]["ok"] is False


def test_stacker_records_fit_mode() -> None:
    abuse = np.array([10.0, 80.0, 20.0, 70.0, 15.0, 75.0])
    fraud = np.array([10.0, 20.0, 80.0, 70.0, 12.0, 65.0])
    y = np.array([0, 1, 1, 1, 0, 1])
    stacker = DecisionStacker().fit(abuse, fraud, y, fit_mode="oof")
    assert stacker.fit_mode == "oof"
    scores = stacker.predict_scores(abuse, fraud)
    assert scores[0] < scores[1]


def test_two_head_stacker_prefers_oof_on_demo() -> None:
    from pathlib import Path

    import runpy

    from refund_abuse_risk.pipeline.score import train_two_head

    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    runpy.run_path(str(root / "scripts" / "generate_demo_data.py"), run_name="__main__")
    orders = pd.read_csv(data / "orders.csv")
    history = pd.read_csv(data / "history.csv")
    devices = pd.read_csv(data / "devices.csv")
    users = pd.read_csv(data / "users.csv")
    model = train_two_head(orders, history, devices, users=users)
    assert model.decision_stacker is not None
    assert model.decision_stacker.fit_mode == "oof"
