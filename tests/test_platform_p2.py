"""P2a GraphBEAN-lite proposals + P2b risk-challenge resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from refund_abuse_risk.control_plane.challenges import (
    resolve_risk_challenge,
    tier_to_risk_challenge,
)
from refund_abuse_risk.graph.graphbean_lite import run_graphbean_lite
from refund_abuse_risk.schemas.models import RiskChallenge, SuggestedTier


def _dense_uv_history(n_users: int = 12, n_vendors: int = 8, orders_per_edge: int = 4) -> pd.DataFrame:
    """Enough UV edges for GraphBEAN-lite min_edges=20 with a few hot colluders."""
    rows: list[dict] = []
    oid = 0
    rng = np.random.default_rng(7)
    for u in range(n_users):
        for v in range(n_vendors):
            # Skip some pairs so density is uneven.
            if (u + v) % 3 == 0:
                continue
            hot = u < 2 and v < 2
            for _ in range(orders_per_edge):
                is_refund = int(rng.random() < (0.7 if hot else 0.08))
                rows.append(
                    {
                        "order_id": f"H{oid}",
                        "user_id": f"U{u}",
                        "vendor_id": f"V{v}",
                        "market": "SG",
                        "vertical": "food",
                        "is_refund": is_refund,
                    }
                )
                oid += 1
    return pd.DataFrame(rows)


def test_p2a_graphbean_lite_proposals_shadow_only() -> None:
    history = _dense_uv_history()
    result = run_graphbean_lite(
        history,
        cfg={"min_edges": 20, "edge_z_threshold": 1.5, "node_z_threshold": 1.5},
    )
    assert result["ok"] is True
    assert result["auto_enforce"] is False
    assert result["n_edges"] >= 20
    actions = result["proposed_actions"]
    assert isinstance(actions, list)
    for a in actions:
        assert a["auto_enforce"] is False
        assert a["mode"] == "shadow"
        assert a["kind"] == "graphbean_edge"


def test_p2a_graphbean_lite_skips_tiny_graphs() -> None:
    tiny = pd.DataFrame(
        [
            {
                "order_id": "1",
                "user_id": "U1",
                "vendor_id": "V1",
                "market": "SG",
                "vertical": "food",
                "is_refund": 1,
            }
        ]
    )
    result = run_graphbean_lite(tiny, cfg={"min_edges": 20})
    assert result["n_edges"] == 0 or result["n_anomalous_edges"] == 0
    assert result["auto_enforce"] is False


def test_p2b_tier_defaults() -> None:
    assert tier_to_risk_challenge(SuggestedTier.AUTO_APPROVE) == RiskChallenge.NONE
    assert tier_to_risk_challenge(SuggestedTier.SOFT_FRICTION) == RiskChallenge.PAYMENT_VERIFY
    assert tier_to_risk_challenge(SuggestedTier.HOLD_REVIEW) == RiskChallenge.IDENTITY_VERIFY


def test_p2b_challenge_rules_live_and_shadow() -> None:
    cfg = {
        "challenges_enabled": True,
        "kill_switch": False,
        "challenge_rules": [
            {
                "id": "soft_payment_verify",
                "mode": "live",
                "when": {"tiers": ["soft_friction"], "min_decision_score": 30},
                "challenge": "payment_verify",
                "reason_code": "CHALLENGE_SOFT_PAYMENT",
            },
            {
                "id": "collusion_identity_shadow",
                "mode": "shadow",
                "when": {
                    "min_uv_mo_possible_collusion": 1,
                    "min_decision_score": 35,
                },
                "challenge": "identity_verify",
                "reason_code": "CHALLENGE_COLLUSION_SHADOW",
            },
        ],
    }
    live = resolve_risk_challenge(
        SuggestedTier.SOFT_FRICTION,
        {},
        decision_score=40.0,
        effect_cfg=cfg,
    )
    assert live.final_challenge == RiskChallenge.PAYMENT_VERIFY
    assert live.matched_mode == "live"
    assert "CHALLENGE_SOFT_PAYMENT" in live.reason_codes

    # Auto-approve: no soft rule; collusion shadow can still match.
    shadow = resolve_risk_challenge(
        SuggestedTier.AUTO_APPROVE,
        {"uv_mo_possible_collusion": 1.0},
        decision_score=40.0,
        effect_cfg=cfg,
    )
    assert shadow.final_challenge == RiskChallenge.NONE
    assert shadow.shadow_challenge == RiskChallenge.IDENTITY_VERIFY
    assert shadow.matched_mode == "shadow"


def test_p2b_challenges_disabled_and_kill_switch() -> None:
    base_cfg = {
        "challenges_enabled": False,
        "challenge_rules": [
            {
                "id": "soft_payment_verify",
                "mode": "live",
                "when": {"tiers": ["soft_friction"]},
                "challenge": "payment_verify",
            }
        ],
    }
    off = resolve_risk_challenge(
        SuggestedTier.SOFT_FRICTION,
        {},
        decision_score=50.0,
        effect_cfg=base_cfg,
    )
    assert off.final_challenge == RiskChallenge.NONE

    killed = resolve_risk_challenge(
        SuggestedTier.SOFT_FRICTION,
        {},
        decision_score=50.0,
        effect_cfg={**base_cfg, "challenges_enabled": True, "kill_switch": True},
    )
    assert killed.final_challenge == RiskChallenge.PAYMENT_VERIFY
    assert "CHALLENGE_KILL_SWITCH" in killed.reason_codes
