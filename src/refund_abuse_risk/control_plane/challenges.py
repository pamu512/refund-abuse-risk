"""Map tiers / rules → risk challenges (Uber penny-drop-shaped friction)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from refund_abuse_risk.schemas.models import RiskChallenge, SuggestedTier


@dataclass
class ChallengeDecision:
    base_challenge: RiskChallenge
    final_challenge: RiskChallenge
    matched_rule_id: str | None = None
    matched_mode: str | None = None  # shadow | live
    shadow_challenge: RiskChallenge | None = None
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_challenge": self.base_challenge.value,
            "final_challenge": self.final_challenge.value,
            "matched_rule_id": self.matched_rule_id,
            "matched_mode": self.matched_mode,
            "shadow_challenge": (
                self.shadow_challenge.value if self.shadow_challenge else None
            ),
            "reason_codes": list(self.reason_codes),
        }


_TIER_TO_CHALLENGE: dict[SuggestedTier, RiskChallenge] = {
    SuggestedTier.AUTO_APPROVE: RiskChallenge.NONE,
    SuggestedTier.SOFT_FRICTION: RiskChallenge.PAYMENT_VERIFY,
    SuggestedTier.HOLD_REVIEW: RiskChallenge.IDENTITY_VERIFY,
    SuggestedTier.AUTO_DENY: RiskChallenge.IDENTITY_VERIFY,
}


def tier_to_risk_challenge(tier: SuggestedTier) -> RiskChallenge:
    return _TIER_TO_CHALLENGE.get(tier, RiskChallenge.NONE)


def _num(features: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(features.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _challenge_rule_matches(
    rule: dict[str, Any], features: dict[str, Any], decision_score: float, tier: SuggestedTier
) -> bool:
    when = rule.get("when") or {}
    if "min_decision_score" in when and decision_score < float(when["min_decision_score"]):
        return False
    if "tiers" in when:
        allowed = {str(t).lower() for t in (when.get("tiers") or [])}
        if tier.value not in allowed:
            return False
    if "min_uv_mo_possible_collusion" in when:
        if _num(features, "uv_mo_possible_collusion") < float(
            when["min_uv_mo_possible_collusion"]
        ):
            return False
    if "min_claim_image_ai_risk" in when and _num(features, "claim_image_ai_risk") < float(
        when["min_claim_image_ai_risk"]
    ):
        return False
    return True


def resolve_risk_challenge(
    tier: SuggestedTier,
    features: dict[str, Any],
    *,
    decision_score: float,
    effect_cfg: dict[str, Any] | None,
) -> ChallengeDecision:
    """
    Tier default challenge, then optional ``challenge_rules`` overrides.

    Lives under effect_rules YAML as ``challenge_rules`` so one config owns
    refund effects + challenges. kill_switch on effect_cfg disables overrides
    (tier default still applies).
    """
    base = tier_to_risk_challenge(tier)
    cfg = effect_cfg or {}
    if not cfg.get("challenges_enabled", True):
        return ChallengeDecision(base_challenge=base, final_challenge=RiskChallenge.NONE)
    kill = bool(cfg.get("kill_switch", False))
    if kill:
        return ChallengeDecision(
            base_challenge=base,
            final_challenge=base,
            reason_codes=["CHALLENGE_KILL_SWITCH"],
        )

    for rule in cfg.get("challenge_rules") or []:
        if not _challenge_rule_matches(rule, features, decision_score, tier):
            continue
        mode = str(rule.get("mode", "shadow")).lower()
        try:
            challenge = RiskChallenge(str(rule.get("challenge")))
        except ValueError:
            continue
        reason = str(rule.get("reason_code") or f"CHALLENGE_{rule.get('id', 'rule')}")
        if mode == "live":
            return ChallengeDecision(
                base_challenge=base,
                final_challenge=challenge,
                matched_rule_id=str(rule.get("id")),
                matched_mode="live",
                reason_codes=[reason],
            )
        return ChallengeDecision(
            base_challenge=base,
            final_challenge=base,
            matched_rule_id=str(rule.get("id")),
            matched_mode="shadow",
            shadow_challenge=challenge,
            reason_codes=[reason, "CHALLENGE_SHADOW"],
        )

    return ChallengeDecision(base_challenge=base, final_challenge=base)
