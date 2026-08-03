from __future__ import annotations

from pathlib import Path

import yaml

from refund_abuse_risk.config import load_guardrails
from refund_abuse_risk.control_plane.audit import PolicyAuditLog
from refund_abuse_risk.control_plane.guardrails import GuardrailError, validate_head_thresholds
from refund_abuse_risk.control_plane.hil import HilProposalStore
from refund_abuse_risk.control_plane.tuner import (
    approve_hil_proposal,
    plan_auto_step,
    reject_hil_proposal,
    run_threshold_tuner,
)


def test_plan_auto_step_caps_delta() -> None:
    guardrails = load_guardrails()
    current = {
        "abuse_soft_friction": 35.0,
        "fraud_soft_friction": 30.0,
        "abuse_hold_review": 50.0,
        "fraud_hold_review": 45.0,
        "abuse_auto_deny": 75.0,
        "fraud_auto_deny": 65.0,
        "target_pattern_recall": 0.98,
    }
    proposed = dict(current)
    proposed["abuse_soft_friction"] = 10.0  # -25 → needs HIL; auto steps -5
    proposed["abuse_auto_deny"] = 90.0  # +15 → max deny delta 3
    applied, deltas, hil_keys = plan_auto_step(current, proposed, guardrails)
    assert applied["abuse_soft_friction"] == 30.0
    assert applied["abuse_auto_deny"] == 78.0
    assert "abuse_soft_friction" in hil_keys
    assert "abuse_auto_deny" in hil_keys
    assert abs(deltas["abuse_soft_friction"] + 5.0) < 1e-6


def test_small_change_auto_applies(tmp_path: Path) -> None:
    guardrails = load_guardrails()
    op_path = tmp_path / "operating_point.yaml"
    op_path.write_text(
        yaml.safe_dump(
            {
                "head_thresholds": {
                    "target_pattern_recall": 0.98,
                    "abuse_soft_friction": 35,
                    "fraud_soft_friction": 30,
                    "abuse_hold_review": 50,
                    "fraud_hold_review": 45,
                    "abuse_auto_deny": 75,
                    "fraud_auto_deny": 65,
                }
            }
        ),
        encoding="utf-8",
    )
    db = tmp_path / "cp.db"
    # Scores where recommended soft ≈ current → tiny / no move, or small move.
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    # Threshold at 98% recall lands near 40; current soft 35 → delta +5 exactly → auto, no HIL
    s = [90, 80, 40, 30, 20, 10, 5, 4, 3, 2]
    decision = run_threshold_tuner(
        current_thresholds=yaml.safe_load(op_path.read_text())["head_thresholds"],
        abuse_y=y,
        abuse_scores=s,
        fraud_y=y,
        fraud_scores=s,
        guardrails=guardrails,
        audit=PolicyAuditLog(db),
        hil_store=HilProposalStore(db),
        operating_point_path=op_path,
        apply=True,
    )
    assert decision.action in {"apply", "pending_hil", "noop"}
    written = yaml.safe_load(op_path.read_text())["head_thresholds"]
    # Auto path never jumps more than max_abs_delta from the starting soft.
    assert abs(float(written["abuse_soft_friction"]) - 35.0) <= 5.0 + 1e-6


def test_large_change_queues_hil_and_partial_step(tmp_path: Path) -> None:
    guardrails = load_guardrails()
    # Force tiny max delta for a clear HIL case.
    guardrails = {
        **guardrails,
        "auto_apply": {
            **guardrails["auto_apply"],
            "max_abs_delta": 2.0,
            "max_abs_delta_by_key": {
                "abuse_auto_deny": 2.0,
                "fraud_auto_deny": 2.0,
            },
            "cooldown_minutes": 0,
            "min_pattern_recall_lift": -1.0,
        },
    }
    op_path = tmp_path / "operating_point.yaml"
    current = {
        "target_pattern_recall": 0.98,
        "abuse_soft_friction": 35,
        "fraud_soft_friction": 30,
        "abuse_hold_review": 50,
        "fraud_hold_review": 45,
        "abuse_auto_deny": 75,
        "fraud_auto_deny": 65,
    }
    op_path.write_text(yaml.safe_dump({"head_thresholds": current}), encoding="utf-8")
    db = tmp_path / "cp.db"
    # Strong positives at high scores → recommend pushes soft way down from 35.
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    s = [95, 90, 85, 20, 15, 10, 8, 5, 3, 1]
    decision = run_threshold_tuner(
        current_thresholds=current,
        abuse_y=y,
        abuse_scores=s,
        fraud_y=y,
        fraud_scores=s,
        guardrails=guardrails,
        audit=PolicyAuditLog(db),
        hil_store=HilProposalStore(db),
        operating_point_path=op_path,
        apply=True,
    )
    assert decision.action == "pending_hil"
    assert decision.proposal_id
    assert decision.hil_keys
    written = yaml.safe_load(op_path.read_text())["head_thresholds"]
    # Partial step applied, not full jump to proposed.
    assert abs(float(written["abuse_soft_friction"]) - 35.0) <= 2.0 + 1e-6
    assert abs(float(written["abuse_soft_friction"]) - float(decision.proposed_full["abuse_soft_friction"])) > 1e-6

    approved = approve_hil_proposal(
        decision.proposal_id,
        hil_store=HilProposalStore(db),
        audit=PolicyAuditLog(db),
        guardrails=guardrails,
        operating_point_path=op_path,
        decided_by="tester",
    )
    assert approved.decision == "accepted"
    final = yaml.safe_load(op_path.read_text())["head_thresholds"]
    assert abs(float(final["abuse_soft_friction"]) - float(decision.proposed_full["abuse_soft_friction"])) < 1e-6


def test_reject_hil(tmp_path: Path) -> None:
    db = tmp_path / "cp.db"
    hil = HilProposalStore(db)
    pid = hil.create(
        current={"abuse_soft_friction": 35},
        proposed={"abuse_soft_friction": 10},
        reason="test",
    )
    decision = reject_hil_proposal(pid, hil_store=hil, audit=PolicyAuditLog(db), decided_by="tester")
    assert decision.decision == "rejected"
    assert hil.get(pid)["status"] == "rejected"


def test_guardrail_rejects_out_of_bounds() -> None:
    guardrails = load_guardrails()
    try:
        validate_head_thresholds({"abuse_soft_friction": 1.0}, guardrails)
        raised = False
    except GuardrailError:
        raised = True
    assert raised
