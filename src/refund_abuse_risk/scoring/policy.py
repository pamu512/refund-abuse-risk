from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from refund_abuse_risk.config import resolve_policy
from refund_abuse_risk.schemas.models import (
    EvidenceItem,
    EvidencePack,
    SuggestedTier,
)


def policy_hash(policy: dict[str, Any]) -> str:
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def band_for_prior(entity_prior: float, operating_point: dict[str, Any]) -> dict[str, float]:
    bands = operating_point.get("bands") or {}
    ordered = sorted(
        bands.items(),
        key=lambda kv: float((kv[1] or {}).get("max_entity_prior", 100)),
    )
    for _name, band in ordered:
        if entity_prior <= float(band.get("max_entity_prior", 100)):
            return {
                "floor": float(band.get("score_floor", 0)),
                "ceiling": float(band.get("score_ceiling", 100)),
            }
    return {"floor": 0.0, "ceiling": 100.0}


def apply_band(raw_score: float, entity_prior: float, operating_point: dict[str, Any]) -> float:
    band = band_for_prior(entity_prior, operating_point)
    # Order/claim score may move only inside the entity-prior band.
    return float(np.clip(raw_score, band["floor"], band["ceiling"]))


def tier_for_score(score: float, operating_point: dict[str, Any]) -> SuggestedTier:
    tiers = operating_point.get("tiers") or {}
    if score <= float((tiers.get("auto_approve") or {}).get("max_score", 25)):
        return SuggestedTier.AUTO_APPROVE
    if score <= float((tiers.get("soft_friction") or {}).get("max_score", 50)):
        return SuggestedTier.SOFT_FRICTION
    if score <= float((tiers.get("hold_review") or {}).get("max_score", 75)):
        return SuggestedTier.HOLD_REVIEW
    return SuggestedTier.AUTO_DENY


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
    resolved = resolve_policy(policy, market=market, vertical=vertical, entity_type="user")
    gates = resolved.get("hard_gates") or {}
    items: list[EvidenceItem] = []

    if gates.get("strong_fraud_label") and float(features.get("strong_fraud_label", 0) or 0) >= 1:
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

    user_gates = gates.get("user") or {}
    checks = [
        ("USER_REFUND_COUNT_7D", "user_refund_count_7d", user_gates.get("max_refund_count_7d"), "user"),
        ("USER_REFUND_COUNT_30D", "user_refund_count_30d", user_gates.get("max_refund_count_30d"), "user"),
        ("USER_REFUND_RATE_30D", "user_refund_rate_30d", user_gates.get("max_refund_rate_30d"), "user"),
        (
            "USER_REFUND_GMV_PCT_30D",
            "user_refund_gmv_pct_30d",
            user_gates.get("max_refund_gmv_pct_30d"),
            "user",
        ),
    ]
    driver_gates = gates.get("driver") or {}
    checks.extend(
        [
            (
                "DRIVER_REFUND_COUNT_30D",
                "driver_refund_count_30d",
                driver_gates.get("max_refund_count_30d"),
                "driver",
            ),
            (
                "DRIVER_REFUND_RATE_30D",
                "driver_refund_rate_30d",
                driver_gates.get("max_refund_rate_30d"),
                "driver",
            ),
        ]
    )
    vendor_gates = gates.get("vendor") or {}
    checks.extend(
        [
            (
                "VENDOR_REFUND_COUNT_30D",
                "vendor_refund_count_30d",
                vendor_gates.get("max_refund_count_30d"),
                "vendor",
            ),
            (
                "VENDOR_REFUND_RATE_30D",
                "vendor_refund_rate_30d",
                vendor_gates.get("max_refund_rate_30d"),
                "vendor",
            ),
            (
                "VENDOR_REFUND_GMV_PCT_30D",
                "vendor_refund_gmv_pct_30d",
                vendor_gates.get("max_refund_gmv_pct_30d"),
                "vendor",
            ),
        ]
    )

    for code, metric, threshold, entity_type in checks:
        if threshold is None:
            continue
        value = float(features.get(metric, 0) or 0)
        if value > float(threshold):
            eid_key = f"{entity_type}_id"
            items.append(
                EvidenceItem(
                    reason_code=code,
                    metric=metric,
                    value=value,
                    threshold=float(threshold),
                    entity_ids={eid_key: str(features.get(eid_key, ""))},
                    weight=1.0,
                )
            )

    link_gates = gates.get("link") or {}
    for kind, score in link_scores.items():
        thr_key = f"{kind}_score_hard"
        thr = link_gates.get(thr_key)
        if thr is None:
            continue
        if float(score) >= float(thr):
            items.append(
                EvidenceItem(
                    reason_code=f"LINK_{kind.upper()}_HARD",
                    metric=f"{kind}_score",
                    value=float(score),
                    threshold=float(thr),
                    entity_ids={
                        "user_id": str(features.get("user_id", "")),
                        "driver_id": str(features.get("driver_id", "")),
                        "vendor_id": str(features.get("vendor_id", "")),
                    },
                    weight=1.0,
                )
            )

    device_thr = link_gates.get("device_cluster_score_hard")
    if device_thr is not None and float(device_cluster_score) >= float(device_thr):
        items.append(
            EvidenceItem(
                reason_code="DEVICE_CLUSTER_HARD",
                metric="device_cluster_score",
                value=float(device_cluster_score),
                threshold=float(device_thr),
                entity_ids={
                    "user_id": str(features.get("user_id", "")),
                    "device_id": str(features.get("device_id", "")),
                },
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
    # Top non-gate contributors for investigators.
    metric_hints = [
        ("USER_REFUND_RATE_7D", "user_refund_count_7d", features.get("user_refund_count_7d")),
        ("USER_REFUND_RATE_30D", "user_refund_rate_30d", features.get("user_refund_rate_30d")),
        ("USER_REFUND_GMV_PCT_30D", "user_refund_gmv_pct_30d", features.get("user_refund_gmv_pct_30d")),
        ("UVD_REFUND_LIFT", "uvd_refund_lift", features.get("uvd_refund_lift")),
        ("DEVICE_MULTI_ACCOUNT", "accounts_per_device", features.get("accounts_per_device")),
        ("DEVICE_CLUSTER_SIZE", "device_cluster_size", features.get("device_cluster_size")),
    ]
    existing = {i.reason_code for i in items}
    for code, metric, value in metric_hints:
        if code in existing or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        items.append(
            EvidenceItem(
                reason_code=code,
                metric=metric,
                value=numeric,
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
    raw = max(float(abuse_score), float(fraud_score))
    banded = apply_band(raw, entity_prior, operating_point)
    if hard_gated:
        # Hard gates cannot be softened by a clean order band.
        combined = max(banded, 76.0, float(entity_prior))
        return float(np.clip(combined, 0.0, 100.0)), SuggestedTier.AUTO_DENY
    combined = float(np.clip(banded, 0.0, 100.0))
    return combined, tier_for_score(combined, operating_point)
