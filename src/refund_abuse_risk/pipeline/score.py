from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from refund_abuse_risk.config import load_label_weights, load_operating_point, load_policy
from refund_abuse_risk.features.builders import build_order_feature_frame, build_order_feature_row
from refund_abuse_risk.graph.entities import EntityGraphIndex
from refund_abuse_risk.model.two_head import (
    TwoHeadModel,
    device_cluster_score_from_features,
    entity_prior_from_features,
    entity_scores_from_features,
    link_scores_from_features,
)
from refund_abuse_risk.schemas.models import (
    EntityScores,
    LifecycleEvent,
    LinkScores,
    OrderRiskSnapshot,
)
from refund_abuse_risk.scoring.policy import (
    build_evidence_pack,
    combine_scores,
    evaluate_hard_gates,
    policy_hash,
)


class OrderRiskCache:
    """Precomputed order snapshots for sync claim-path reads."""

    def __init__(self) -> None:
        self._snapshots: dict[str, OrderRiskSnapshot] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._entity_scores: dict[str, float] = {}
        self.graph = EntityGraphIndex()

    def get(self, order_id: str) -> OrderRiskSnapshot | None:
        return self._snapshots.get(order_id)

    def put(self, snapshot: OrderRiskSnapshot, order: dict[str, Any]) -> None:
        self._snapshots[snapshot.order_id] = snapshot
        self._orders[snapshot.order_id] = dict(order)
        self.graph.index_order(snapshot.order_id, order)

    def all_snapshots(self) -> list[OrderRiskSnapshot]:
        return list(self._snapshots.values())

    def standing_entity_score(self, key: str) -> float:
        return float(self._entity_scores.get(key, 0.0))

    def set_standing_entity_score(self, key: str, score: float) -> None:
        self._entity_scores[key] = float(score)


def score_feature_row(
    row: dict[str, Any] | pd.Series,
    model: TwoHeadModel,
    *,
    policy: dict[str, Any] | None = None,
    operating_point: dict[str, Any] | None = None,
) -> OrderRiskSnapshot:
    policy = policy or load_policy()
    operating_point = operating_point or load_operating_point()
    frame = pd.DataFrame([dict(row)])
    scored = model.predict_proba(frame).iloc[0]
    features = dict(row)

    abuse_score = float(scored["abuse_score"])
    fraud_score = float(scored["fraud_score"])
    entity_scores = entity_scores_from_features(features)
    link_scores = link_scores_from_features(features)
    device_score = device_cluster_score_from_features(features)
    prior = entity_prior_from_features(features)

    hard_gated, gate_items = evaluate_hard_gates(
        features,
        entity_scores=entity_scores,
        link_scores=link_scores,
        device_cluster_score=device_score,
        policy=policy,
        market=str(features.get("market", "")),
        vertical=str(features.get("vertical", "food")),
    )
    combined, tier = combine_scores(
        abuse_score,
        fraud_score,
        prior,
        operating_point,
        hard_gated=hard_gated,
    )
    evidence = build_evidence_pack(
        features=features,
        abuse_score=abuse_score,
        fraud_score=fraud_score,
        entity_scores=entity_scores,
        link_scores=link_scores,
        device_cluster_score=device_score,
        entity_prior=prior,
        hard_gate_items=gate_items,
        policy=policy,
        market=str(features.get("market", "")),
        vertical=str(features.get("vertical", "food")),
    )
    reason_codes = [item.reason_code for item in evidence.items]
    return OrderRiskSnapshot(
        order_id=str(features.get("order_id", "")),
        market=str(features.get("market", "")),
        vertical=str(features.get("vertical", "")),
        abuse_score=abuse_score,
        fraud_score=fraud_score,
        entity_scores=EntityScores(**entity_scores),
        link_scores=LinkScores(**link_scores),
        device_cluster_score=device_score,
        suggested_tier=tier,
        reason_codes=reason_codes,
        evidence_pack=evidence,
        hard_gated=hard_gated,
        model_version=str(operating_point.get("model_version", model.model_version)),
        policy_version=str(operating_point.get("policy_version", policy_hash(policy))),
        scored_at=datetime.now(timezone.utc),
        entity_prior=prior,
        combined_score=combined,
    )


def precompute_orders(
    orders: pd.DataFrame,
    history: pd.DataFrame,
    devices: pd.DataFrame,
    model: TwoHeadModel,
    *,
    cache: OrderRiskCache | None = None,
    policy: dict[str, Any] | None = None,
    operating_point: dict[str, Any] | None = None,
) -> OrderRiskCache:
    cache = cache or OrderRiskCache()
    policy = policy or load_policy()
    operating_point = operating_point or load_operating_point()
    feat = build_order_feature_frame(orders, history, devices)
    scored = model.predict_proba(feat)
    for _, row in scored.iterrows():
        snap = score_feature_row(row, model, policy=policy, operating_point=operating_point)
        order = row.to_dict()
        cache.put(snap, order)
        # Update standing entity/link scores for risk-change detection.
        from refund_abuse_risk.graph.entities import link_key

        cache.set_standing_entity_score(link_key("user", str(order.get("user_id", ""))), snap.entity_scores.user)
        cache.set_standing_entity_score(
            link_key("driver", str(order.get("driver_id", ""))), snap.entity_scores.driver
        )
        cache.set_standing_entity_score(
            link_key("vendor", str(order.get("vendor_id", ""))), snap.entity_scores.vendor
        )
        cache.set_standing_entity_score(
            link_key("uvd", str(order.get("user_id", "")), str(order.get("vendor_id", "")), str(order.get("driver_id", ""))),
            snap.link_scores.uvd,
        )
    return cache


def refresh_order(
    order: dict[str, Any],
    history: pd.DataFrame,
    devices: pd.DataFrame,
    model: TwoHeadModel,
    cache: OrderRiskCache,
    *,
    event: LifecycleEvent | str | None = None,
    policy: dict[str, Any] | None = None,
    operating_point: dict[str, Any] | None = None,
) -> OrderRiskSnapshot:
    _ = event  # lifecycle marker for callers/logging; features use current order row
    feat = build_order_feature_row(order, history, devices)
    row = {**order, **feat}
    snap = score_feature_row(row, model, policy=policy, operating_point=operating_point)
    cache.put(snap, order)
    return snap


def on_entity_risk_change(
    changed_keys: set[str],
    open_orders: pd.DataFrame,
    history: pd.DataFrame,
    devices: pd.DataFrame,
    model: TwoHeadModel,
    cache: OrderRiskCache,
    *,
    new_scores: dict[str, float],
    operating_point: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> list[OrderRiskSnapshot]:
    """Rescore open orders linked to entity/link keys whose risk moved materially."""
    operating_point = operating_point or load_operating_point()
    delta = float(operating_point.get("entity_risk_change_delta", 10.0))
    material_keys: set[str] = set()
    for key in changed_keys:
        old = cache.standing_entity_score(key)
        new = float(new_scores.get(key, old))
        if abs(new - old) >= delta:
            material_keys.add(key)
            cache.set_standing_entity_score(key, new)
    if not material_keys:
        return []

    touched = cache.graph.orders_touched_by_keys(material_keys)
    if open_orders.empty or not touched:
        return []
    subset = open_orders[open_orders["order_id"].astype(str).isin(touched)]
    refreshed: list[OrderRiskSnapshot] = []
    for _, order in subset.iterrows():
        refreshed.append(
            refresh_order(
                order.to_dict(),
                history,
                devices,
                model,
                cache,
                event=LifecycleEvent.CLAIM,
                policy=policy,
                operating_point=operating_point,
            )
        )
    return refreshed


def train_two_head(
    train_orders: pd.DataFrame,
    history: pd.DataFrame,
    devices: pd.DataFrame,
    *,
    label_weights: dict[str, Any] | None = None,
) -> TwoHeadModel:
    label_weights = label_weights or load_label_weights()
    feat = build_order_feature_frame(train_orders, history, devices)
    model = TwoHeadModel()
    model.fit(feat, label_weights)
    return model


def claim_path_read(order_id: str, cache: OrderRiskCache) -> OrderRiskSnapshot:
    """Sync claim path: read precomputed snapshot only (no graph walk)."""
    snap = cache.get(order_id)
    if snap is None:
        raise KeyError(f"No precomputed snapshot for order_id={order_id}")
    return snap
