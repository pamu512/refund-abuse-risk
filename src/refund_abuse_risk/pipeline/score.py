from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from refund_abuse_risk.baselines.cohort import CohortBaselineStore
from refund_abuse_risk.baselines.gate import (
    apply_precision_discount_to_operating_point,
    evaluate_baseline_gate,
)
from refund_abuse_risk.baselines.store import BehaviorBaselineStore
from refund_abuse_risk.config import (
    load_behavior_baselines,
    load_effect_rules,
    load_label_weights,
    load_operating_point,
    load_policy,
    load_refund_budget,
)
from refund_abuse_risk.control_plane.challenges import resolve_risk_challenge
from refund_abuse_risk.control_plane.effects import resolve_refund_effect
from refund_abuse_risk.features.builders import build_order_feature_frame, build_order_feature_row
from refund_abuse_risk.graph.entities import EntityGraphIndex
from refund_abuse_risk.integrations.device_vision import merge_platform_signals
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
    RiskChallenge,
)
from refund_abuse_risk.scoring.budget import apply_budget_effect_floor, evaluate_refund_budget
from refund_abuse_risk.scoring.decision import operating_point_for_slice
from refund_abuse_risk.scoring.policy import (
    build_evidence_pack,
    combine_scores,
    evaluate_hard_gates,
    policy_hash,
)

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_BASELINE_DB = _ROOT / "data" / "behavior_baselines.db"

# Same-order training labels must not drive hard gates / scores (honesty).
_LABEL_LEAK_KEYS = (
    "abuse_label",
    "abuse_label_weak",
    "fraud_label",
    "fraud_label_source",
    "strong_fraud_label",
    "weak_policy_negative",
)


def _strip_label_leak(features: dict[str, Any]) -> dict[str, Any]:
    out = dict(features)
    for key in _LABEL_LEAK_KEYS:
        out.pop(key, None)
    return out


class OrderRiskCache:
    """Precomputed order snapshots for sync claim-path reads."""

    def __init__(
        self,
        baseline_store: BehaviorBaselineStore | None = None,
        cohort_store: CohortBaselineStore | None = None,
    ) -> None:
        self._snapshots: dict[str, OrderRiskSnapshot] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._entity_scores: dict[str, float] = {}
        self.graph = EntityGraphIndex()
        self.baseline_store = baseline_store
        self.cohort_store = cohort_store

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
    baseline_store: BehaviorBaselineStore | None = None,
    cohort_store: CohortBaselineStore | None = None,
    baseline_cfg: dict[str, Any] | None = None,
    effect_cfg: dict[str, Any] | None = None,
    budget_cfg: dict[str, Any] | None = None,
    update_baselines: bool = True,
    decision_archive: Any | None = None,
) -> OrderRiskSnapshot:
    policy = policy or load_policy()
    operating_point = operating_point or load_operating_point()
    baseline_cfg = baseline_cfg if baseline_cfg is not None else load_behavior_baselines()
    effect_cfg = effect_cfg if effect_cfg is not None else load_effect_rules()
    budget_cfg = budget_cfg if budget_cfg is not None else load_refund_budget()
    raw = dict(row)
    raw = merge_platform_signals(
        raw,
        device_payload=raw.get("device_intelligence") or raw.get("device_payload"),
        vision_payload=raw.get("claim_vision") or raw.get("vision_payload"),
    )
    features = _strip_label_leak(raw)
    # Attach read-only baseline features for gate/evidence (not model head inputs).
    if baseline_store is not None:
        features.update(baseline_store.feature_map_for_order(features))
    scored = model.predict_proba(pd.DataFrame([features])).iloc[0]

    abuse_score = float(scored["abuse_score"])
    fraud_score = float(scored["fraud_score"])
    decision_score = float(scored.get("decision_score", max(abuse_score, fraud_score)))
    entity_scores = entity_scores_from_features(features)
    link_scores = link_scores_from_features(features)
    device_score = device_cluster_score_from_features(features)
    prior = entity_prior_from_features(features)

    baseline_gate = evaluate_baseline_gate(
        features,
        abuse_score=abuse_score,
        fraud_score=fraud_score,
        store=baseline_store,
        cohort_store=cohort_store,
        baseline_cfg=baseline_cfg,
        operating_point=operating_point,
        update_store=update_baselines and baseline_store is not None,
    )
    # Cohort lifts available for next-order model features / evidence.
    features.update(baseline_gate.cohort_features)
    market = str(features.get("market", ""))
    vertical = str(features.get("vertical", "food"))
    # Slice ladder first, then baseline precision warp on that ladder.
    slice_op = operating_point_for_slice(
        operating_point, market=market, vertical=vertical
    )
    decision_op = apply_precision_discount_to_operating_point(
        slice_op,
        abuse_precision_discount=baseline_gate.abuse_precision_discount,
        fraud_precision_discount=baseline_gate.fraud_precision_discount,
        abuse_relax_points=baseline_gate.abuse_relax_points,
        fraud_relax_points=baseline_gate.fraud_relax_points,
        abuse_tighten_points=baseline_gate.abuse_tighten_points,
        fraud_tighten_points=baseline_gate.fraud_tighten_points,
    )

    hard_gated, gate_items = evaluate_hard_gates(
        features,
        entity_scores=entity_scores,
        link_scores=link_scores,
        device_cluster_score=device_score,
        policy=policy,
        market=market,
        vertical=vertical,
    )
    gate_items = list(gate_items) + list(baseline_gate.evidence_items)
    combined, tier = combine_scores(
        abuse_score,
        fraud_score,
        prior,
        decision_op,
        hard_gated=hard_gated,
        decision_score=decision_score,
        market=market,
        vertical=vertical,
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
    effect = resolve_refund_effect(
        tier,
        features,
        decision_score=decision_score,
        effect_cfg=effect_cfg,
    )
    challenge = resolve_risk_challenge(
        tier,
        features,
        decision_score=decision_score,
        effect_cfg=effect_cfg,
    )
    budget = evaluate_refund_budget(features, budget_cfg)
    final_effect, budget, budget_reasons = apply_budget_effect_floor(
        effect.final_effect, budget, budget_cfg
    )
    effect_meta = effect.to_dict()
    effect_meta["final_effect"] = final_effect.value
    effect_meta["budget_floored"] = bool(budget.floored)
    effect_meta["challenge"] = challenge.to_dict()
    reason_codes = [item.reason_code for item in evidence.items]
    reason_codes.extend(effect.reason_codes)
    reason_codes.extend(challenge.reason_codes)
    reason_codes.extend(budget_reasons)
    if challenge.final_challenge != RiskChallenge.NONE:
        reason_codes.append(f"challenge:{challenge.final_challenge.value}")
    if bool(scored.get("slice_calibrator_applied")):
        reason_codes.append("slice_calibrator_applied")
    if operating_point.get("decision_threshold_overlays"):
        thr = (slice_op.get("decision_thresholds") or {})
        global_thr = (operating_point.get("decision_thresholds") or {})
        if thr != global_thr:
            reason_codes.append(f"slice_overlay:{market}|{vertical}")
    snap = OrderRiskSnapshot(
        order_id=str(features.get("order_id", "")),
        market=str(features.get("market", "")),
        vertical=str(features.get("vertical", "")),
        abuse_score=abuse_score,
        fraud_score=fraud_score,
        decision_score=decision_score,
        entity_scores=EntityScores(**entity_scores),
        link_scores=LinkScores(**link_scores),
        device_cluster_score=device_score,
        suggested_tier=tier,
        refund_effect=final_effect,
        shadow_refund_effect=effect.shadow_effect,
        risk_challenge=challenge.final_challenge,
        shadow_risk_challenge=challenge.shadow_challenge,
        reason_codes=reason_codes,
        evidence_pack=evidence,
        hard_gated=hard_gated,
        model_version=str(operating_point.get("model_version", model.model_version)),
        policy_version=str(operating_point.get("policy_version", policy_hash(policy))),
        scored_at=datetime.now(timezone.utc),
        entity_prior=prior,
        combined_score=combined,
        precision_discount=float(baseline_gate.precision_discount),
        threshold_relax_points=float(baseline_gate.threshold_relax_points),
        baseline_hil_required=bool(baseline_gate.hil_required),
        baseline_gate=baseline_gate.to_dict(),
        effect_decision=effect_meta,
        refund_budget=budget.to_dict(),
    )
    archive = decision_archive
    if archive is None:
        env_path = os.environ.get("DECISION_ARCHIVE_PATH")
        if env_path:
            from refund_abuse_risk.ops.decision_archive import DecisionArchive

            archive = DecisionArchive(env_path)
    if archive is not None:
        archive.append_snapshot(snap)
    return snap


def precompute_orders(
    orders: pd.DataFrame,
    history: pd.DataFrame,
    devices: pd.DataFrame,
    model: TwoHeadModel,
    *,
    users: pd.DataFrame | None = None,
    cache: OrderRiskCache | None = None,
    policy: dict[str, Any] | None = None,
    operating_point: dict[str, Any] | None = None,
    baseline_store: BehaviorBaselineStore | None = None,
    baseline_cfg: dict[str, Any] | None = None,
    update_baselines: bool = True,
) -> OrderRiskCache:
    baseline_cfg = baseline_cfg if baseline_cfg is not None else load_behavior_baselines()
    if baseline_store is None and baseline_cfg.get("enabled", True):
        baseline_store = BehaviorBaselineStore(_DEFAULT_BASELINE_DB)
    cohort_store = CohortBaselineStore(_DEFAULT_BASELINE_DB)
    cache = cache or OrderRiskCache(baseline_store=baseline_store, cohort_store=cohort_store)
    if cache.baseline_store is None:
        cache.baseline_store = baseline_store
    if cache.cohort_store is None:
        cache.cohort_store = cohort_store
    policy = policy or load_policy()
    operating_point = operating_point or load_operating_point()
    feat = build_order_feature_frame(orders, history, devices, users=users)
    scored = model.predict_proba(feat)
    for _, row in scored.iterrows():
        snap = score_feature_row(
            row,
            model,
            policy=policy,
            operating_point=operating_point,
            baseline_store=cache.baseline_store,
            cohort_store=cache.cohort_store,
            baseline_cfg=baseline_cfg,
            update_baselines=update_baselines,
        )
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
    users: pd.DataFrame | None = None,
    event: LifecycleEvent | str | None = None,
    policy: dict[str, Any] | None = None,
    operating_point: dict[str, Any] | None = None,
    device_sdk_event: dict[str, Any] | None = None,
    vision_sdk_event: dict[str, Any] | None = None,
) -> OrderRiskSnapshot:
    from refund_abuse_risk.config import load_sdk_ingest
    from refund_abuse_risk.integrations.sdk_ingest import attach_sdk_signals

    _ = event  # lifecycle marker for callers/logging; features use current order row
    order_in = dict(order)
    if device_sdk_event is not None or vision_sdk_event is not None:
        order_in = attach_sdk_signals(
            order_in,
            device_event=device_sdk_event,
            vision_event=vision_sdk_event,
            cfg=load_sdk_ingest(),
        )
    feat = build_order_feature_row(order_in, history, devices, users=users)
    row = {**order_in, **feat}
    # Lifecycle refresh: only settled events write baselines (config settled_statuses).
    if event is not None:
        row["lifecycle_event"] = event.value if isinstance(event, LifecycleEvent) else str(event)
    snap = score_feature_row(
        row,
        model,
        policy=policy,
        operating_point=operating_point,
        baseline_store=cache.baseline_store,
        cohort_store=cache.cohort_store,
    )
    cache.put(snap, order_in)
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
    users: pd.DataFrame | None = None,
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
                users=users,
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
    users: pd.DataFrame | None = None,
    label_weights: dict[str, Any] | None = None,
    operating_point: dict[str, Any] | None = None,
) -> TwoHeadModel:
    label_weights = label_weights or load_label_weights()
    operating_point = operating_point or load_operating_point()
    feat = build_order_feature_frame(train_orders, history, devices, users=users)
    model = TwoHeadModel(model_version=str(operating_point.get("model_version", "0.2.0")))
    model.fit(feat, label_weights)
    return model


def claim_path_read(order_id: str, cache: OrderRiskCache) -> OrderRiskSnapshot:
    """Sync claim path: read precomputed snapshot only (no graph walk)."""
    snap = cache.get(order_id)
    if snap is None:
        raise KeyError(f"No precomputed snapshot for order_id={order_id}")
    return snap
