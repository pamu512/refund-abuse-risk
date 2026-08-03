"""Threshold tuner: auto-step within delta limits; HIL for oversized moves."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from refund_abuse_risk.control_plane.audit import PolicyAuditLog
from refund_abuse_risk.control_plane.guardrails import (
    THRESHOLD_KEYS,
    GuardrailError,
    clamp_head_thresholds,
    max_auto_delta,
    validate_head_thresholds,
)
from refund_abuse_risk.control_plane.hil import HilProposalStore
from refund_abuse_risk.scoring.thresholds import (
    pattern_flag_recall,
    recommend_head_thresholds,
)


@dataclass
class TuningDecision:
    action: str  # apply | pending_hil | reject | noop
    decision: str  # accepted | pending | rejected | recorded
    reason: str
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    proposed_full: dict[str, float] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    hil_keys: list[str] = field(default_factory=list)
    proposal_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "decision": self.decision,
            "reason": self.reason,
            "before": self.before,
            "after": self.after,
            "proposed_full": self.proposed_full,
            "deltas": self.deltas,
            "hil_keys": self.hil_keys,
            "proposal_id": self.proposal_id,
            "metrics": self.metrics,
        }


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _in_cooldown(audit: PolicyAuditLog, *, market: str, vertical: str, minutes: int) -> bool:
    last = audit.last_apply_at(market=market, vertical=vertical)
    if last is None:
        return False
    return _parse_ts(last) > datetime.now(timezone.utc) - timedelta(minutes=int(minutes))


def enforce_monotone_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    """Keep soft <= hold <= deny after partial auto-steps."""
    out = dict(thresholds)
    for head in ("abuse", "fraud"):
        soft_k = f"{head}_soft_friction"
        hold_k = f"{head}_hold_review"
        deny_k = f"{head}_auto_deny"
        if soft_k in out and hold_k in out:
            out[hold_k] = max(float(out[hold_k]), float(out[soft_k]))
        if hold_k in out and deny_k in out:
            out[deny_k] = max(float(out[deny_k]), float(out[hold_k]))
        elif soft_k in out and deny_k in out:
            out[deny_k] = max(float(out[deny_k]), float(out[soft_k]))
    return out


def plan_auto_step(
    current: dict[str, float],
    proposed: dict[str, float],
    guardrails: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """
    Move each key toward proposed by at most max_auto_delta.

    Returns (auto_applied_values, deltas, keys_still_needing_hil).
    """
    applied: dict[str, float] = dict(current)
    hil_keys: list[str] = []
    for key in THRESHOLD_KEYS:
        if key not in proposed:
            continue
        old = float(current.get(key, proposed[key]))
        target = float(proposed[key])
        max_d = max_auto_delta(guardrails, key)
        raw_delta = target - old
        step = max(-max_d, min(max_d, raw_delta))
        new_val = old + step
        # Round score thresholds to 2dp; keep recall as-is.
        if key == "target_pattern_recall":
            new_val = round(new_val, 4)
            target = round(target, 4)
        else:
            new_val = round(new_val, 2)
            target = round(target, 2)
        applied[key] = new_val
        if abs(target - new_val) > 1e-6:
            hil_keys.append(key)
    applied = enforce_monotone_thresholds(applied)
    deltas = {
        k: round(float(applied[k]) - float(current.get(k, applied[k])), 4)
        for k in applied
        if k in proposed or k in current
    }
    return applied, deltas, hil_keys


def write_operating_point_thresholds(
    operating_point_path: Path | str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    path = Path(operating_point_path)
    op = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    head = dict(op.get("head_thresholds") or {})
    head.update({k: float(v) for k, v in thresholds.items()})
    op["head_thresholds"] = head
    path.write_text(
        yaml.safe_dump(op, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return head


def run_threshold_tuner(
    *,
    current_thresholds: dict[str, Any],
    abuse_y: Any,
    abuse_scores: Any,
    fraud_y: Any,
    fraud_scores: Any,
    guardrails: dict[str, Any],
    audit: PolicyAuditLog,
    hil_store: HilProposalStore,
    operating_point_path: Path | str | None = None,
    apply: bool = True,
    market: str = "",
    vertical: str = "",
    actor: str = "tuner",
) -> TuningDecision:
    """
    Recommend thresholds from scores; auto-apply a bounded step; queue HIL for the rest.

    Similar spirit to offline-cancel-risk: ML adjusts inside guardrails, but oversized
    moves require human approval before the full target is written.
    """
    auto_cfg = guardrails.get("auto_apply") or {}
    target_recall = float(
        current_thresholds.get("target_pattern_recall")
        or (guardrails.get("bounds") or {})
        .get("head_thresholds.target_pattern_recall", {})
        .get("min", 0.98)
    )
    # Prefer explicit current target when present.
    if "target_pattern_recall" in current_thresholds:
        target_recall = float(current_thresholds["target_pattern_recall"])

    min_prec = current_thresholds.get("min_precision_at_soft", 0.15)
    max_fp = current_thresholds.get("max_fp_rate_at_soft")
    recommended = recommend_head_thresholds(
        abuse_y,
        abuse_scores,
        fraud_y,
        fraud_scores,
        target_recall=target_recall,
        min_precision_at_soft=float(min_prec) if min_prec is not None else None,
        max_fp_rate_at_soft=float(max_fp) if max_fp is not None else None,
        apply_floors=False,
    )
    if not recommended.get("ok", True):
        audit.append(
            actor=actor,
            action="reject",
            market=market,
            vertical=vertical,
            before=dict(current_thresholds),
            after={k: recommended.get(k) for k in THRESHOLD_KEYS if k in recommended},
            decision="rejected",
            reason=(
                "honesty: cost constraints infeasible or floors would bind; "
                f"floor_would_bind={recommended.get('floor_would_bind')}; "
                f"cost_feasible={recommended.get('cost_feasible')}"
            ),
        )
        return TuningDecision(
            action="reject",
            decision="rejected",
            reason="recommended thresholds not ok (cost/floor honesty gate)",
            before={k: float(current_thresholds.get(k, 0)) for k in THRESHOLD_KEYS if k in current_thresholds},
            proposed_full={k: float(recommended[k]) for k in THRESHOLD_KEYS if k in recommended},
            metrics={
                "ok": recommended.get("ok"),
                "floor_would_bind": recommended.get("floor_would_bind"),
                "cost_feasible": recommended.get("cost_feasible"),
                "abuse_soft_cost_info": recommended.get("abuse_soft_cost_info"),
                "fraud_soft_cost_info": recommended.get("fraud_soft_cost_info"),
            },
        )

    proposed = clamp_head_thresholds(recommended, guardrails)
    # Keep target_pattern_recall from current unless recommend overwrote with same key.
    if "target_pattern_recall" in current_thresholds:
        proposed["target_pattern_recall"] = float(current_thresholds["target_pattern_recall"])

    try:
        validate_head_thresholds(proposed, guardrails)
    except GuardrailError as exc:
        audit.append(
            actor=actor,
            action="reject",
            market=market,
            vertical=vertical,
            before=dict(current_thresholds),
            after=proposed,
            decision="rejected",
            reason=f"guardrail:{exc}",
        )
        return TuningDecision(
            action="reject",
            decision="rejected",
            reason=str(exc),
            before={k: float(current_thresholds.get(k, 0)) for k in proposed},
            proposed_full=proposed,
        )

    current = {
        k: float(current_thresholds[k])
        for k in THRESHOLD_KEYS
        if k in current_thresholds
    }
    # Fill missing current keys from proposed so deltas are defined.
    for k, v in proposed.items():
        current.setdefault(k, float(v))

    soft_a = float(current.get("abuse_soft_friction", proposed["abuse_soft_friction"]))
    soft_f = float(current.get("fraud_soft_friction", proposed["fraud_soft_friction"]))
    recall_before = pattern_flag_recall(
        abuse_y, fraud_y, abuse_scores, fraud_scores, soft_a, soft_f
    )
    recall_after = pattern_flag_recall(
        abuse_y,
        fraud_y,
        abuse_scores,
        fraud_scores,
        float(proposed["abuse_soft_friction"]),
        float(proposed["fraud_soft_friction"]),
    )
    lift = 0.0
    if recall_before is not None and recall_after is not None:
        lift = float(recall_after) - float(recall_before)
    metrics = {
        "pattern_recall_before": recall_before,
        "pattern_recall_after": recall_after,
        "pattern_recall_lift": lift,
        "recommended": recommended,
    }

    min_lift = float(auto_cfg.get("min_pattern_recall_lift", 0.0))
    if lift < min_lift:
        audit.append(
            actor=actor,
            action="suggest",
            market=market,
            vertical=vertical,
            before=current,
            after=proposed,
            metrics_before={"pattern_recall": recall_before},
            metrics_after={"pattern_recall": recall_after},
            constraints=auto_cfg,
            decision="rejected",
            reason="pattern_recall_lift_below_min",
        )
        return TuningDecision(
            action="reject",
            decision="rejected",
            reason="pattern_recall_lift_below_min",
            before=current,
            proposed_full=proposed,
            metrics=metrics,
        )

    cooldown = int(auto_cfg.get("cooldown_minutes", 60))
    if _in_cooldown(audit, market=market, vertical=vertical, minutes=cooldown):
        audit.append(
            actor=actor,
            action="suggest",
            market=market,
            vertical=vertical,
            before=current,
            after=proposed,
            metrics_after=metrics,
            constraints=auto_cfg,
            decision="rejected",
            reason="cooldown",
        )
        return TuningDecision(
            action="reject",
            decision="rejected",
            reason="cooldown",
            before=current,
            proposed_full=proposed,
            metrics=metrics,
        )

    applied, deltas, hil_keys = plan_auto_step(current, proposed, guardrails)
    changed = any(abs(d) > 1e-9 for d in deltas.values())
    if not changed and not hil_keys:
        audit.append(
            actor=actor,
            action="noop",
            market=market,
            vertical=vertical,
            before=current,
            after=applied,
            metrics_after=metrics,
            decision="recorded",
            reason="no_change",
        )
        return TuningDecision(
            action="noop",
            decision="recorded",
            reason="no_change",
            before=current,
            after=applied,
            proposed_full=proposed,
            deltas=deltas,
            metrics=metrics,
        )

    proposal_id: str | None = None
    if apply and hil_keys:
        proposal_id = hil_store.create(
            current=current,
            proposed=proposed,
            auto_step=applied,
            metrics=metrics,
            market=market,
            vertical=vertical,
            reason="delta_exceeds_max_auto",
        )

    if apply and changed:
        if operating_point_path is None:
            raise ValueError("operating_point_path required when apply=True")
        write_operating_point_thresholds(operating_point_path, applied)

    if hil_keys:
        if apply:
            audit.append(
                actor=actor,
                action="pending_hil",
                market=market,
                vertical=vertical,
                before=current,
                after=applied,
                metrics_before={"pattern_recall": recall_before},
                metrics_after=metrics,
                constraints={
                    **auto_cfg,
                    "hil_keys": hil_keys,
                    "proposal_id": proposal_id,
                    "proposed_full": proposed,
                },
                decision="pending",
                reason="delta_exceeds_max_auto",
            )
        return TuningDecision(
            action="pending_hil",
            decision="pending",
            reason="delta_exceeds_max_auto" if apply else "dry_run_delta_exceeds_max_auto",
            before=current,
            after=applied,
            proposed_full=proposed,
            deltas=deltas,
            hil_keys=hil_keys,
            proposal_id=proposal_id,
            metrics=metrics,
        )

    if apply:
        audit.append(
            actor=actor,
            action="apply",
            market=market,
            vertical=vertical,
            before=current,
            after=applied,
            metrics_before={"pattern_recall": recall_before},
            metrics_after=metrics,
            constraints=auto_cfg,
            decision="accepted",
            reason="within_max_auto_delta",
        )
    return TuningDecision(
        action="apply",
        decision="accepted" if apply else "recorded",
        reason="within_max_auto_delta" if apply else "dry_run_within_max_auto_delta",
        before=current,
        after=applied,
        proposed_full=proposed,
        deltas=deltas,
        metrics=metrics,
    )


def approve_hil_proposal(
    proposal_id: str,
    *,
    hil_store: HilProposalStore,
    audit: PolicyAuditLog,
    guardrails: dict[str, Any],
    operating_point_path: Path | str,
    decided_by: str = "hil",
) -> TuningDecision:
    """Human approves full proposed thresholds (still must pass guardrails)."""
    row = hil_store.get(proposal_id)
    if row is None:
        raise KeyError(f"unknown proposal_id={proposal_id}")
    if row["status"] != "pending":
        raise ValueError(f"proposal already {row['status']}")
    proposed = enforce_monotone_thresholds(clamp_head_thresholds(row["proposed"], guardrails))
    validate_head_thresholds(proposed, guardrails)
    write_operating_point_thresholds(operating_point_path, proposed)
    hil_store.mark(proposal_id, status="approved", decided_by=decided_by)
    audit.append(
        actor=decided_by,
        action="hil_approve",
        market=row.get("market", ""),
        vertical=row.get("vertical", ""),
        before=row.get("current"),
        after=proposed,
        metrics_after=row.get("metrics"),
        constraints={"proposal_id": proposal_id},
        decision="accepted",
        reason="hil_approved",
    )
    return TuningDecision(
        action="apply",
        decision="accepted",
        reason="hil_approved",
        before=dict(row.get("current") or {}),
        after=proposed,
        proposed_full=proposed,
        proposal_id=proposal_id,
        metrics=dict(row.get("metrics") or {}),
    )


def reject_hil_proposal(
    proposal_id: str,
    *,
    hil_store: HilProposalStore,
    audit: PolicyAuditLog,
    decided_by: str = "hil",
    reason: str = "hil_rejected",
) -> TuningDecision:
    row = hil_store.mark(proposal_id, status="rejected", decided_by=decided_by)
    audit.append(
        actor=decided_by,
        action="hil_reject",
        market=row.get("market", ""),
        vertical=row.get("vertical", ""),
        before=row.get("current"),
        after=row.get("proposed"),
        metrics_after=row.get("metrics"),
        constraints={"proposal_id": proposal_id},
        decision="rejected",
        reason=reason,
    )
    return TuningDecision(
        action="reject",
        decision="rejected",
        reason=reason,
        before=dict(row.get("current") or {}),
        proposed_full=dict(row.get("proposed") or {}),
        proposal_id=proposal_id,
        metrics=dict(row.get("metrics") or {}),
    )
