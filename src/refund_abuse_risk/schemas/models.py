from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SuggestedTier(str, Enum):
    AUTO_APPROVE = "auto_approve"
    SOFT_FRICTION = "soft_friction"
    HOLD_REVIEW = "hold_review"
    AUTO_DENY = "auto_deny"


class EntityScores(BaseModel):
    user: float = 0.0
    driver: float = 0.0
    vendor: float = 0.0


class LinkScores(BaseModel):
    ud: float = 0.0
    uv: float = 0.0
    vd: float = 0.0
    uvd: float = 0.0


class EvidenceItem(BaseModel):
    reason_code: str
    metric: str
    value: float | str | bool | None = None
    threshold: float | str | bool | None = None
    entity_ids: dict[str, str] = Field(default_factory=dict)
    weight: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class EvidencePack(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)
    contributions: dict[str, float] = Field(default_factory=dict)


class OrderRiskSnapshot(BaseModel):
    order_id: str
    market: str
    vertical: str
    abuse_score: float
    fraud_score: float
    entity_scores: EntityScores
    link_scores: LinkScores
    device_cluster_score: float = 0.0
    suggested_tier: SuggestedTier
    reason_codes: list[str] = Field(default_factory=list)
    evidence_pack: EvidencePack = Field(default_factory=EvidencePack)
    hard_gated: bool = False
    model_version: str = "0.1.0"
    policy_version: str = "0.1.0"
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entity_prior: float = 0.0
    combined_score: float = 0.0


class LifecycleEvent(str, Enum):
    PLACED = "placed"
    PICKED_UP = "picked_up"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CLAIM = "claim"
