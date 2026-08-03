from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from refund_abuse_risk.config import resolve_policy
from refund_abuse_risk.schemas.models import (
    EvidenceItem,
    EvidencePack,
    RefundEffect,
    SuggestedTier,
)

_TIER_TO_REFUND_EFFECT: dict[SuggestedTier, RefundEffect] = {
    SuggestedTier.AUTO_APPROVE: RefundEffect.REFUND_AUTO_GRANT,
    SuggestedTier.SOFT_FRICTION: RefundEffect.REFUND_STEP_UP,
    SuggestedTier.HOLD_REVIEW: RefundEffect.REFUND_MANUAL_REVIEW,
    SuggestedTier.AUTO_DENY: RefundEffect.REFUND_BLOCK,
}


def tier_to_refund_effect(tier: SuggestedTier) -> RefundEffect:
    """Map risk tier → progressive refund UX effect (Glovo-style traffic light)."""
    return _TIER_TO_REFUND_EFFECT.get(tier, RefundEffect.REFUND_MANUAL_REVIEW)


def policy_hash(policy: dict[str, Any]) -> str:
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def tier_for_score(score: float, operating_point: dict[str, Any]) -> SuggestedTier:
    """Map a display score to tier (secondary; primary path uses head thresholds)."""
    tiers = operating_point.get("tiers") or {}
    if score <= float((tiers.get("auto_approve") or {}).get("max_score", 25)):
        return SuggestedTier.AUTO_APPROVE
    if score <= float((tiers.get("soft_friction") or {}).get("max_score", 50)):
        return SuggestedTier.SOFT_FRICTION
    if score <= float((tiers.get("hold_review") or {}).get("max_score", 75)):
        return SuggestedTier.HOLD_REVIEW
    return SuggestedTier.AUTO_DENY


def tier_from_heads(
    abuse_score: float,
    fraud_score: float,
    operating_point: dict[str, Any],
) -> SuggestedTier:
    """
    Learning-primary tier: OR of abuse/fraud heads against operating thresholds.

    Soft thresholds are meant to catch ~target_pattern_recall of labeled patterns.
    """
    thr = operating_point.get("head_thresholds") or {}
    abuse = float(abuse_score)
    fraud = float(fraud_score)
    if abuse >= float(thr.get("abuse_auto_deny", 75)) or fraud >= float(
        thr.get("fraud_auto_deny", 65)
    ):
        return SuggestedTier.AUTO_DENY
    if abuse >= float(thr.get("abuse_hold_review", 50)) or fraud >= float(
        thr.get("fraud_hold_review", 45)
    ):
        return SuggestedTier.HOLD_REVIEW
    if abuse >= float(thr.get("abuse_soft_friction", 35)) or fraud >= float(
        thr.get("fraud_soft_friction", 30)
    ):
        return SuggestedTier.SOFT_FRICTION
    return SuggestedTier.AUTO_APPROVE


def evaluate_hard_gates(
    features: dict[str, Any],
    *,
    entity_scores: dict[str, float],
    link_scores: dict[str, float],
    device_cluster_score: float,
    policy: dict[str, Any],
    market: str,
    vertical: str,
) -> tuple[bool, list[EvidenceItem]]:
    """
    Hard gates are safety overrides only.

    Learning-primary mode: only proven/strong fraud labels force auto_deny.
    Behavioral patterns (rates, rings, LTV burn) are model features, not gates.
    """
    _ = entity_scores, link_scores, device_cluster_score
    resolved = resolve_policy(policy, market=market, vertical=vertical, entity_type="user")
    gates = resolved.get("hard_gates") or {}
    items: list[EvidenceItem] = []

    if gates.get("strong_fraud_label", True) and float(features.get("strong_fraud_label", 0) or 0) >= 1:
        items.append(
            EvidenceItem(
                reason_code="STRONG_FRAUD_LABEL",
                metric="strong_fraud_label",
                value=True,
                threshold=True,
                entity_ids={"user_id": str(features.get("user_id", ""))},
                weight=1.0,
            )
        )

    return (len(items) > 0), items


def build_evidence_pack(
    *,
    features: dict[str, Any],
    abuse_score: float,
    fraud_score: float,
    entity_scores: dict[str, float],
    link_scores: dict[str, float],
    device_cluster_score: float,
    entity_prior: float,
    hard_gate_items: list[EvidenceItem],
    policy: dict[str, Any],
    market: str,
    vertical: str,
) -> EvidencePack:
    resolved = resolve_policy(policy, market=market, vertical=vertical, entity_type="user")
    weights = resolved.get("evidence_weights") or {}
    contributions = {
        "abuse_score": float(abuse_score),
        "fraud_score": float(fraud_score),
        "entity_prior": float(entity_prior),
        "user": float(entity_scores.get("user", 0)) * float(weights.get("user", 0.25)),
        "driver": float(entity_scores.get("driver", 0)) * float(weights.get("driver", 0.15)),
        "vendor": float(entity_scores.get("vendor", 0)) * float(weights.get("vendor", 0.15)),
        "ud": float(link_scores.get("ud", 0)) * float(weights.get("ud", 0.10)),
        "uv": float(link_scores.get("uv", 0)) * float(weights.get("uv", 0.10)),
        "vd": float(link_scores.get("vd", 0)) * float(weights.get("vd", 0.08)),
        "uvd": float(link_scores.get("uvd", 0)) * float(weights.get("uvd", 0.12)),
        "device": float(device_cluster_score) * float(weights.get("device", 0.05)),
    }
    items = list(hard_gate_items)
    # Soft evidence: behavioral features above informative floors (not decision authority).
    metric_hints = [
        ("USER_REFUND_COUNT_7D", "user_refund_count_7d", features.get("user_refund_count_7d"), 2.0),
        ("USER_REFUND_RATE_30D", "user_refund_rate_30d", features.get("user_refund_rate_30d"), 0.15),
        ("USER_REFUND_GMV_PCT_30D", "user_refund_gmv_pct_30d", features.get("user_refund_gmv_pct_30d"), 0.15),
        ("USER_LIFETIME_REFUNDS", "user_lifetime_refund_count", features.get("user_lifetime_refund_count"), 3.0),
        ("USER_REFUND_TO_LTV_SOFT", "user_refund_to_ltv_ratio", features.get("user_refund_to_ltv_ratio"), 0.25),
        ("COMBINED_REFUND_COUNT", "combined_refund_count_30d", features.get("combined_refund_count_30d"), 3.0),
        ("RELATED_REFUND_COUNT", "related_refund_count_30d", features.get("related_refund_count_30d"), 2.0),
        ("RELATED_MAX_RATE", "related_max_refund_rate_30d", features.get("related_max_refund_rate_30d"), 0.25),
        ("UVD_REFUND_LIFT", "uvd_refund_lift", features.get("uvd_refund_lift"), 1.5),
        ("UVD_REFUND_SHARE", "uvd_refund_share", features.get("uvd_refund_share"), 0.4),
        ("DEVICE_MULTI_ACCOUNT", "accounts_per_device", features.get("accounts_per_device"), 2.0),
        ("DEVICE_CLUSTER_SIZE", "device_cluster_size", features.get("device_cluster_size"), 3.0),
        ("REASON_REPEAT_RATE", "reason_repeat_rate_30d", features.get("reason_repeat_rate_30d"), 0.6),
        ("ABUSE_HEAD", "abuse_score", abuse_score, 35.0),
        ("FRAUD_HEAD", "fraud_score", fraud_score, 30.0),
    ]
    existing = {i.reason_code for i in items}
    for code, metric, value, min_value in metric_hints:
        if code in existing or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric < float(min_value):
            continue
        items.append(
            EvidenceItem(
                reason_code=code,
                metric=metric,
                value=numeric,
                threshold=float(min_value),
                entity_ids={
                    "user_id": str(features.get("user_id", "")),
                    "driver_id": str(features.get("driver_id", "")),
                    "vendor_id": str(features.get("vendor_id", "")),
                },
                weight=float(contributions.get(metric.split("_")[0].lower(), 0.1)),
            )
        )
    for kind, score in entity_scores.items():
        code = f"ENTITY_{kind.upper()}_EXTREME"
        if float(score) >= 90 and code not in existing:
            items.append(
                EvidenceItem(
                    reason_code=code,
                    metric=f"{kind}_score",
                    value=float(score),
                    threshold=90.0,
                    entity_ids={f"{kind}_id": str(features.get(f"{kind}_id", ""))},
                    weight=0.8,
                )
            )
    return EvidencePack(items=items, contributions=contributions)


def combine_scores(
    abuse_score: float,
    fraud_score: float,
    entity_prior: float,
    operating_point: dict[str, Any],
    *,
    hard_gated: bool,
) -> tuple[float, SuggestedTier]:
    """
    Learning-primary decision: tier from head thresholds; display score from heads.

    entity_prior is retained for evidence/monitoring only — it does not band or
    override calibrated head probabilities.
    """
    _ = entity_prior
    abuse = float(abuse_score)
    fraud = float(fraud_score)
    display_cfg = operating_point.get("score_display") or {}
    w_abuse = float(display_cfg.get("abuse", 0.5))
    w_fraud = float(display_cfg.get("fraud", 0.5))
    total = w_abuse + w_fraud
    if total <= 0:
        w_abuse, w_fraud, total = 0.5, 0.5, 1.0
    # Keep max floor so one extreme head remains visible in the display score.
    blended = (w_abuse * abuse + w_fraud * fraud) / total
    combined = float(np.clip(max(blended, max(abuse, fraud)), 0.0, 100.0))

    if hard_gated:
        return max(combined, 90.0), SuggestedTier.AUTO_DENY

    mode = str(operating_point.get("decision_mode", "learning_primary"))
    if mode == "learning_primary":
        return combined, tier_from_heads(abuse, fraud, operating_point)
    return combined, tier_for_score(combined, operating_point)
