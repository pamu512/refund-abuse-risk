"""Baseline elevation gate → head-attributed precision discount."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from refund_abuse_risk.baselines.cohort import CohortBaselineStore, tenure_bucket
from refund_abuse_risk.baselines.store import (
    BaselineRow,
    BehaviorBaselineStore,
    entity_keys_from_order,
)
from refund_abuse_risk.schemas.models import EvidenceItem


@dataclass
class EntityBaselineSignal:
    entity_key: str
    entity_kind: str
    entity_id: str
    score_head: str
    current_score: float
    baseline_mean: float
    under_rate: float
    is_clean_baseline: bool
    elevated_streak: int
    lift: float
    abs_delta: float
    precision_discount: float
    elevated: bool
    cohort_lift: float = 1.0
    source: str = "self"  # self | cohort


@dataclass
class BaselineGateResult:
    enabled: bool
    under_threshold_abuse: float = 0.0
    under_threshold_fraud: float = 0.0
    precision_discount: float = 0.0
    abuse_precision_discount: float = 0.0
    fraud_precision_discount: float = 0.0
    abuse_relax_points: float = 0.0
    fraud_relax_points: float = 0.0
    abuse_tighten_points: float = 0.0
    fraud_tighten_points: float = 0.0
    threshold_relax_points: float = 0.0
    hil_required: bool = False
    hil_relax_remainder: float = 0.0
    trust_credit: bool = False
    effective_target_precision: float | None = None
    signals: list[EntityBaselineSignal] = field(default_factory=list)
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    updated: bool = False
    cohort_features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "under_threshold_abuse": self.under_threshold_abuse,
            "under_threshold_fraud": self.under_threshold_fraud,
            "precision_discount": self.precision_discount,
            "abuse_precision_discount": self.abuse_precision_discount,
            "fraud_precision_discount": self.fraud_precision_discount,
            "abuse_relax_points": self.abuse_relax_points,
            "fraud_relax_points": self.fraud_relax_points,
            "abuse_tighten_points": self.abuse_tighten_points,
            "fraud_tighten_points": self.fraud_tighten_points,
            "threshold_relax_points": self.threshold_relax_points,
            "hil_required": self.hil_required,
            "hil_relax_remainder": self.hil_relax_remainder,
            "trust_credit": self.trust_credit,
            "effective_target_precision": self.effective_target_precision,
            "updated": self.updated,
            "cohort_features": self.cohort_features,
            "signals": [
                {
                    "entity_key": s.entity_key,
                    "entity_kind": s.entity_kind,
                    "entity_id": s.entity_id,
                    "score_head": s.score_head,
                    "current_score": s.current_score,
                    "baseline_mean": s.baseline_mean,
                    "under_rate": s.under_rate,
                    "is_clean_baseline": s.is_clean_baseline,
                    "elevated_streak": s.elevated_streak,
                    "lift": s.lift,
                    "abs_delta": s.abs_delta,
                    "precision_discount": s.precision_discount,
                    "elevated": s.elevated,
                    "cohort_lift": s.cohort_lift,
                    "source": s.source,
                }
                for s in self.signals
            ],
        }


def under_thresholds(
    baseline_cfg: dict[str, Any],
    operating_point: dict[str, Any],
) -> tuple[float, float]:
    under = baseline_cfg.get("under_threshold") or {}
    thr = operating_point.get("head_thresholds") or {}
    abuse = float(thr.get("abuse_soft_friction", under.get("abuse_fallback", 35.0)))
    fraud = float(thr.get("fraud_soft_friction", under.get("fraud_fallback", 30.0)))
    return abuse, fraud


def is_settled_event(order: dict[str, Any], baseline_cfg: dict[str, Any]) -> bool:
    allowed = {str(s).lower() for s in (baseline_cfg.get("settled_statuses") or [])}
    if not allowed:
        return True
    status = str(order.get("status", "") or order.get("lifecycle_event", "") or "").lower()
    if status in allowed:
        return True
    # Explicit claim path marker.
    if str(order.get("lifecycle_event", "")).lower() in allowed:
        return True
    return False


def _discount_for_lift(
    lift: float,
    abs_delta: float,
    *,
    min_lift: float,
    min_delta: float,
    per_lift_unit: float,
    max_discount: float,
) -> float:
    if lift < min_lift and abs_delta < min_delta:
        return 0.0
    lift_part = max(0.0, lift - 1.0) * float(per_lift_unit)
    delta_part = max(0.0, abs_delta / 100.0) * float(max_discount)
    return float(min(max_discount, max(lift_part, delta_part)))


def evaluate_baseline_gate(
    order: dict[str, Any],
    *,
    abuse_score: float,
    fraud_score: float,
    store: BehaviorBaselineStore | None,
    baseline_cfg: dict[str, Any],
    operating_point: dict[str, Any],
    update_store: bool = True,
    cohort_store: CohortBaselineStore | None = None,
) -> BaselineGateResult:
    if not baseline_cfg.get("enabled", True) or store is None:
        return BaselineGateResult(enabled=False)

    abuse_under, fraud_under = under_thresholds(baseline_cfg, operating_point)
    qualify = baseline_cfg.get("baseline_qualify") or {}
    elev_cfg = baseline_cfg.get("elevation") or {}
    disc_cfg = baseline_cfg.get("precision_discount") or {}
    cohort_cfg = baseline_cfg.get("cohort") or {}
    trust_cfg = baseline_cfg.get("trust_credit") or {}
    kind_heads = baseline_cfg.get("kind_heads") or {}
    kinds = baseline_cfg.get("entity_kinds") or list(kind_heads.keys())
    market = str(order.get("market", "") or "")
    vertical = str(order.get("vertical", "") or "")
    order_id = str(order.get("order_id", "") or "")
    bucket = tenure_bucket(
        float(order.get("user_days_since_signup", 0) or 0),
        cohort_cfg.get("tenure_buckets"),
    )

    min_obs = int(qualify.get("min_observations", 10))
    min_under_rate = float(qualify.get("min_under_rate", 0.80))
    revoke_under = float(qualify.get("revoke_under_rate", 0.50))
    lookback_events = int(qualify.get("lookback_events", 50))
    lookback_days = int(qualify.get("lookback_days", 30))
    pair_min = float(qualify.get("pair_min_cooccur", 3))
    min_delta = float(elev_cfg.get("min_abs_delta", 15.0))
    min_lift = float(elev_cfg.get("min_lift", 1.5))
    require_clean = bool(elev_cfg.get("require_clean_baseline", True))
    min_elev_streak = int(elev_cfg.get("min_elevated_streak", 2))
    max_discount = float(disc_cfg.get("max", 0.10))
    per_lift = float(disc_cfg.get("per_lift_unit", 0.04))
    relax_per_pp = float(disc_cfg.get("threshold_relax_per_pp", 1.0))
    max_auto_relax = float(disc_cfg.get("max_auto_relax_points", 5.0))
    cohort_min_support = int(cohort_cfg.get("min_support", 30))
    cohort_min_lift = float(cohort_cfg.get("min_lift", 1.5))
    cohort_min_delta = float(cohort_cfg.get("min_abs_delta", 15.0))

    settled = is_settled_event(order, baseline_cfg)
    do_update = bool(update_store and settled and order_id)

    # Update peer cohorts once per settled order (abuse + fraud heads).
    cohort_features: dict[str, float] = {
        "user_cohort_lift": 1.0,
        "driver_cohort_lift": 1.0,
        "vendor_cohort_lift": 1.0,
        "device_cohort_lift": 1.0,
        "tenure_bucket_code": {"new": 0.0, "mid": 1.0, "mature": 2.0}.get(bucket, 1.0),
    }
    if cohort_store is not None and cohort_cfg.get("enabled", True):
        if do_update:
            cohort_store.update_score(
                market=market,
                vertical=vertical,
                tenure_bucket_name=bucket,
                score_head="abuse",
                score=abuse_score,
                lookback_events=int(cohort_cfg.get("lookback_events", 500)),
            )
            cohort_store.update_score(
                market=market,
                vertical=vertical,
                tenure_bucket_name=bucket,
                score_head="fraud",
                score=fraud_score,
                lookback_events=int(cohort_cfg.get("lookback_events", 500)),
            )
        for head, score, feat_key in (
            ("abuse", abuse_score, "user_cohort_lift"),
            ("fraud", fraud_score, "driver_cohort_lift"),
        ):
            lift, _, _ = cohort_store.lift(
                score,
                market=market,
                vertical=vertical,
                tenure_bucket_name=bucket,
                score_head=head,
                min_support=cohort_min_support,
            )
            cohort_features[feat_key] = float(lift)
        # Vendor/device share fraud cohort for this market×vertical×tenure.
        cohort_features["vendor_cohort_lift"] = cohort_features["driver_cohort_lift"]
        cohort_features["device_cohort_lift"] = cohort_features["driver_cohort_lift"]

    payloads = entity_keys_from_order(
        order,
        kinds=kinds,
        kind_heads=kind_heads,
        abuse_score=abuse_score,
        fraud_score=fraud_score,
    )
    signals: list[EntityBaselineSignal] = []
    evidence: list[EvidenceItem] = []
    any_updated = False

    for payload in payloads:
        kind = str(payload["entity_kind"])
        head = str(payload["score_head"])
        under_thr = fraud_under if head == "fraud" else abuse_under
        score = float(payload["score"])
        parts = dict(payload["parts"])
        row: BaselineRow | None
        if do_update:
            before_dup = store.has_observation(
                order_id, payload["entity_key"], market=market, vertical=vertical
            )
            row = store.update_observation(
                order_id=order_id,
                entity_key=payload["entity_key"],
                entity_kind=kind,
                entity_id=payload["entity_id"],
                score=score,
                score_head=head,
                under_threshold=under_thr,
                event_ts=order.get("event_ts"),
                market=market,
                vertical=vertical,
                user_id=str(parts.get("user_id", "")),
                driver_id=str(parts.get("driver_id", "")),
                vendor_id=str(parts.get("vendor_id", "")),
                lookback_events=lookback_events,
                lookback_days=lookback_days,
                min_observations=min_obs,
                min_under_rate=min_under_rate,
                revoke_under_rate=revoke_under,
                elevated_min_delta=min_delta,
                elevated_min_lift=min_lift,
                pair_min_cooccur=pair_min,
                support=float(payload.get("support", 1.0)),
            )
            if row is not None and not before_dup:
                any_updated = True
        else:
            row = store.get(payload["entity_key"], market=market, vertical=vertical)
        if row is None:
            continue

        pair_kinds = {"ud", "uv", "vd", "uvd"}
        if kind in pair_kinds and float(payload.get("support", 0)) < pair_min:
            discount, elevated, lift, abs_delta = 0.0, False, 1.0, 0.0
            source = "self"
        else:
            discount, elevated, lift, abs_delta = _signal_discount(
                row,
                current_score=score,
                require_clean=require_clean,
                min_elev_streak=min_elev_streak,
                min_lift=min_lift,
                min_delta=min_delta,
                per_lift=per_lift,
                max_discount=max_discount,
            )
            source = "self"

        c_lift = 1.0
        if cohort_store is not None and cohort_cfg.get("enabled", True):
            c_lift, c_p50, c_n = cohort_store.lift(
                score,
                market=market,
                vertical=vertical,
                tenure_bucket_name=bucket,
                score_head=head,
                min_support=cohort_min_support,
            )
            # Cohort elevation does not require a clean self-baseline (stable abusers).
            if c_n >= cohort_min_support and (
                c_lift >= cohort_min_lift or (score - c_p50) >= cohort_min_delta
            ):
                c_disc = _discount_for_lift(
                    c_lift,
                    score - c_p50,
                    min_lift=cohort_min_lift,
                    min_delta=cohort_min_delta,
                    per_lift_unit=per_lift,
                    max_discount=max_discount,
                )
                if c_disc > discount:
                    discount, elevated, source = c_disc, True, "cohort"

        signals.append(
            EntityBaselineSignal(
                entity_key=payload["entity_key"],
                entity_kind=kind,
                entity_id=payload["entity_id"],
                score_head=head,
                current_score=score,
                baseline_mean=float(row.baseline_mean),
                under_rate=float(row.under_rate),
                is_clean_baseline=bool(row.is_clean_baseline),
                elevated_streak=int(row.elevated_streak),
                lift=lift,
                abs_delta=abs_delta,
                precision_discount=discount,
                elevated=elevated,
                cohort_lift=float(c_lift),
                source=source,
            )
        )
        if row.is_clean_baseline and score < under_thr:
            evidence.append(
                EvidenceItem(
                    reason_code=f"BASELINE_CLEAN_{kind.upper()}",
                    metric="under_rate",
                    value=float(row.under_rate),
                    threshold=min_under_rate,
                    entity_ids={k: v for k, v in parts.items() if v},
                    weight=0.2,
                    details={
                        "baseline_mean": row.baseline_mean,
                        "n_observations": row.n_observations,
                        "score_head": head,
                    },
                )
            )
        if elevated and discount > 0:
            code = (
                f"COHORT_ELEVATED_{kind.upper()}"
                if source == "cohort"
                else f"BASELINE_ELEVATED_{kind.upper()}"
            )
            evidence.append(
                EvidenceItem(
                    reason_code=code,
                    metric="precision_discount",
                    value=discount,
                    threshold=max_discount,
                    entity_ids={k: v for k, v in parts.items() if v},
                    weight=0.6,
                    details={
                        "current_score": score,
                        "baseline_mean": row.baseline_mean,
                        "lift": lift,
                        "cohort_lift": c_lift,
                        "abs_delta": abs_delta,
                        "elevated_streak": row.elevated_streak,
                        "score_head": head,
                        "source": source,
                        "tenure_bucket": bucket,
                    },
                )
            )

    abuse_disc = max(
        (s.precision_discount for s in signals if s.score_head == "abuse"), default=0.0
    )
    fraud_disc = max(
        (s.precision_discount for s in signals if s.score_head == "fraud"), default=0.0
    )
    abuse_relax = abuse_disc * 100.0 * relax_per_pp
    fraud_relax = fraud_disc * 100.0 * relax_per_pp
    raw_relax = max(abuse_relax, fraud_relax)
    hil_required = raw_relax > max_auto_relax + 1e-9
    abuse_relax_auto = min(abuse_relax, max_auto_relax)
    fraud_relax_auto = min(fraud_relax, max_auto_relax)
    if hil_required:
        evidence.append(
            EvidenceItem(
                reason_code="BASELINE_HIL_RELAX",
                metric="threshold_relax_points",
                value=raw_relax,
                threshold=max_auto_relax,
                entity_ids={},
                weight=0.5,
                details={
                    "abuse_relax": abuse_relax,
                    "fraud_relax": fraud_relax,
                    "auto_capped_to": max_auto_relax,
                },
            )
        )

    # Trust credit: clean user+device and low scores → raise thresholds (easier approve).
    trust = False
    abuse_tighten = 0.0
    fraud_tighten = 0.0
    if trust_cfg.get("enabled", True) and abuse_disc <= 0 and fraud_disc <= 0:
        by_kind = {s.entity_kind: s for s in signals}
        user_ok = (not trust_cfg.get("require_user_clean", True)) or (
            by_kind.get("user") is not None and by_kind["user"].is_clean_baseline
        )
        device_ok = (not trust_cfg.get("require_device_clean", True)) or (
            by_kind.get("device") is not None and by_kind["device"].is_clean_baseline
        )
        if (
            user_ok
            and device_ok
            and abuse_score <= float(trust_cfg.get("max_abuse_score", 25))
            and fraud_score <= float(trust_cfg.get("max_fraud_score", 20))
        ):
            trust = True
            abuse_tighten = float(trust_cfg.get("abuse_tighten_points", 5))
            fraud_tighten = float(trust_cfg.get("fraud_tighten_points", 5))
            evidence.append(
                EvidenceItem(
                    reason_code="BASELINE_TRUST_CREDIT",
                    metric="trust_credit",
                    value=True,
                    threshold=True,
                    entity_ids={
                        "user_id": str(order.get("user_id", "")),
                        "device_id": str(order.get("device_id", "")),
                    },
                    weight=0.3,
                    details={
                        "abuse_tighten_points": abuse_tighten,
                        "fraud_tighten_points": fraud_tighten,
                    },
                )
            )

    best_disc = max(abuse_disc, fraud_disc)
    target = float((operating_point.get("head_thresholds") or {}).get("target_pattern_recall", 0.98))

    return BaselineGateResult(
        enabled=True,
        under_threshold_abuse=abuse_under,
        under_threshold_fraud=fraud_under,
        precision_discount=float(best_disc),
        abuse_precision_discount=float(abuse_disc),
        fraud_precision_discount=float(fraud_disc),
        abuse_relax_points=float(abuse_relax_auto),
        fraud_relax_points=float(fraud_relax_auto),
        abuse_tighten_points=float(abuse_tighten),
        fraud_tighten_points=float(fraud_tighten),
        threshold_relax_points=float(max(abuse_relax_auto, fraud_relax_auto)),
        hil_required=hil_required,
        hil_relax_remainder=float(max(0.0, raw_relax - max_auto_relax)),
        trust_credit=trust,
        effective_target_precision=float(max(0.5, target - best_disc)),
        signals=signals,
        evidence_items=evidence,
        updated=any_updated,
        cohort_features=cohort_features,
    )


def _signal_discount(
    row: BaselineRow,
    *,
    current_score: float,
    require_clean: bool,
    min_elev_streak: int,
    min_lift: float,
    min_delta: float,
    per_lift: float,
    max_discount: float,
) -> tuple[float, bool, float, float]:
    eps = 1e-6
    baseline = float(row.baseline_mean)
    lift = float(current_score) / max(baseline, eps)
    abs_delta = float(current_score) - baseline
    elevated_raw = abs_delta >= min_delta or lift >= min_lift
    if require_clean and not row.is_clean_baseline:
        return 0.0, False, lift, abs_delta
    if int(row.elevated_streak) < int(min_elev_streak):
        return 0.0, elevated_raw, lift, abs_delta
    if not elevated_raw:
        return 0.0, False, lift, abs_delta
    discount = _discount_for_lift(
        lift,
        abs_delta,
        min_lift=min_lift,
        min_delta=min_delta,
        per_lift_unit=per_lift,
        max_discount=max_discount,
    )
    return discount, discount > 0, lift, abs_delta


def apply_precision_discount_to_operating_point(
    operating_point: dict[str, Any],
    *,
    precision_discount: float = 0.0,
    threshold_relax_points: float = 0.0,
    abuse_precision_discount: float | None = None,
    fraud_precision_discount: float | None = None,
    abuse_relax_points: float | None = None,
    fraud_relax_points: float | None = None,
    abuse_tighten_points: float = 0.0,
    fraud_tighten_points: float = 0.0,
) -> dict[str, Any]:
    """Relax (discount) or tighten (trust credit) head thresholds independently."""
    abuse_relax = float(
        abuse_relax_points
        if abuse_relax_points is not None
        else threshold_relax_points
    )
    fraud_relax = float(
        fraud_relax_points
        if fraud_relax_points is not None
        else threshold_relax_points
    )
    abuse_disc = float(
        abuse_precision_discount if abuse_precision_discount is not None else precision_discount
    )
    fraud_disc = float(
        fraud_precision_discount if fraud_precision_discount is not None else precision_discount
    )
    abuse_tighten = float(abuse_tighten_points)
    fraud_tighten = float(fraud_tighten_points)
    if (
        abuse_relax <= 0
        and fraud_relax <= 0
        and abuse_disc <= 0
        and fraud_disc <= 0
        and abuse_tighten <= 0
        and fraud_tighten <= 0
    ):
        return operating_point

    op = copy.deepcopy(operating_point)
    thr = dict(op.get("head_thresholds") or {})
    for key in ("abuse_soft_friction", "abuse_hold_review", "abuse_auto_deny"):
        if key not in thr:
            continue
        val = float(thr[key])
        if abuse_relax > 0:
            val = max(0.0, val - abuse_relax)
        if abuse_tighten > 0:
            val = min(100.0, val + abuse_tighten)
        thr[key] = val
    for key in ("fraud_soft_friction", "fraud_hold_review", "fraud_auto_deny"):
        if key not in thr:
            continue
        val = float(thr[key])
        if fraud_relax > 0:
            val = max(0.0, val - fraud_relax)
        if fraud_tighten > 0:
            val = min(100.0, val + fraud_tighten)
        thr[key] = val
    if "target_pattern_recall" in thr and max(abuse_disc, fraud_disc) > 0:
        thr["target_pattern_recall"] = max(
            0.5, float(thr["target_pattern_recall"]) - max(abuse_disc, fraud_disc)
        )
    op["head_thresholds"] = thr
    return op
