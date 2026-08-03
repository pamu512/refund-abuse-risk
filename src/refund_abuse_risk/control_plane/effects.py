"""Shadow → live refund-effect rules with global kill switch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from refund_abuse_risk.schemas.models import RefundEffect, SuggestedTier
from refund_abuse_risk.scoring.policy import tier_to_refund_effect


@dataclass
class EffectDecision:
    base_effect: RefundEffect
    final_effect: RefundEffect
    matched_rule_id: str | None = None
    matched_mode: str | None = None  # shadow | live
    shadow_effect: RefundEffect | None = None
    reason_codes: list[str] = field(default_factory=list)
    kill_switch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_effect": self.base_effect.value,
            "final_effect": self.final_effect.value,
            "matched_rule_id": self.matched_rule_id,
            "matched_mode": self.matched_mode,
            "shadow_effect": self.shadow_effect.value if self.shadow_effect else None,
            "reason_codes": list(self.reason_codes),
            "kill_switch": self.kill_switch,
        }


def _num(features: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(features.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _rule_matches(rule: dict[str, Any], features: dict[str, Any], decision_score: float) -> bool:
    when = rule.get("when") or {}
    if "min_decision_score" in when and decision_score < float(when["min_decision_score"]):
        return False
    if "min_device_risk_score" in when and _num(features, "device_risk_score") < float(
        when["min_device_risk_score"]
    ):
        return False
    if "min_claim_image_ai_risk" in when and _num(features, "claim_image_ai_risk") < float(
        when["min_claim_image_ai_risk"]
    ):
        return False
    for key in (
        "pin_required",
        "pin_verified",
        "claim_in_app_capture",
        "customer_courier_same_device",
        "claim_has_image",
    ):
        if key not in when:
            continue
        want = float(when[key])
        got = _num(features, key)
        if want >= 1 and got < 1:
            return False
        if want < 1 and got >= 1:
            return False
    return True


def resolve_refund_effect(
    tier: SuggestedTier,
    features: dict[str, Any],
    *,
    decision_score: float,
    effect_cfg: dict[str, Any] | None,
) -> EffectDecision:
    """Apply ordered effect rules; kill_switch disables overrides."""
    base = tier_to_refund_effect(tier)
    cfg = effect_cfg or {}
    kill = bool(cfg.get("kill_switch", False))
    if not cfg.get("enabled", True) or kill:
        return EffectDecision(
            base_effect=base,
            final_effect=base,
            kill_switch=kill,
            reason_codes=["EFFECT_KILL_SWITCH"] if kill else [],
        )

    for rule in cfg.get("rules") or []:
        if not _rule_matches(rule, features, decision_score):
            continue
        mode = str(rule.get("mode", "shadow")).lower()
        try:
            effect = RefundEffect(str(rule.get("effect")))
        except ValueError:
            continue
        reason = str(rule.get("reason_code") or f"EFFECT_{rule.get('id', 'rule')}")
        if mode == "live":
            return EffectDecision(
                base_effect=base,
                final_effect=effect,
                matched_rule_id=str(rule.get("id")),
                matched_mode="live",
                reason_codes=[reason],
            )
        # shadow: keep base, record override suggestion
        return EffectDecision(
            base_effect=base,
            final_effect=base,
            matched_rule_id=str(rule.get("id")),
            matched_mode="shadow",
            shadow_effect=effect,
            reason_codes=[reason, "EFFECT_SHADOW"],
        )

    return EffectDecision(base_effect=base, final_effect=base)
