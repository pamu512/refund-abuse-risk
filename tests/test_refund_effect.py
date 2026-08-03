from __future__ import annotations

from refund_abuse_risk.schemas.models import RefundEffect, SuggestedTier
from refund_abuse_risk.scoring.policy import tier_from_heads, tier_to_refund_effect
from refund_abuse_risk.config import load_operating_point


def test_tier_to_refund_effect_mapping() -> None:
    assert tier_to_refund_effect(SuggestedTier.AUTO_APPROVE) == RefundEffect.REFUND_AUTO_GRANT
    assert tier_to_refund_effect(SuggestedTier.SOFT_FRICTION) == RefundEffect.REFUND_STEP_UP
    assert tier_to_refund_effect(SuggestedTier.HOLD_REVIEW) == RefundEffect.REFUND_MANUAL_REVIEW
    assert tier_to_refund_effect(SuggestedTier.AUTO_DENY) == RefundEffect.REFUND_BLOCK


def test_heads_drive_progressive_refund_effects() -> None:
    op = load_operating_point()
    assert tier_to_refund_effect(tier_from_heads(10, 10, op)) == RefundEffect.REFUND_AUTO_GRANT
    assert tier_to_refund_effect(tier_from_heads(40, 10, op)) == RefundEffect.REFUND_STEP_UP
    assert tier_to_refund_effect(tier_from_heads(55, 10, op)) == RefundEffect.REFUND_MANUAL_REVIEW
    assert tier_to_refund_effect(tier_from_heads(10, 70, op)) == RefundEffect.REFUND_BLOCK
