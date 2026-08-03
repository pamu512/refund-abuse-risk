#!/usr/bin/env python3
"""Recommend + auto-step head thresholds; queue HIL when the move is too large."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from refund_abuse_risk.config import load_guardrails, load_label_weights, load_operating_point
from refund_abuse_risk.control_plane.audit import PolicyAuditLog
from refund_abuse_risk.control_plane.hil import HilProposalStore
from refund_abuse_risk.control_plane.tuner import (
    approve_hil_proposal,
    reject_hil_proposal,
    run_threshold_tuner,
)
from refund_abuse_risk.features.builders import build_order_feature_frame
from refund_abuse_risk.model.two_head import apply_proxy_fraud_labels
from refund_abuse_risk.pipeline.score import train_two_head

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OP_PATH = ROOT / "config" / "operating_point.default.yaml"
CONTROL_DB = ROOT / "data" / "control_plane.db"


def _holdout_scores() -> tuple[object, object, object, object]:
    if not (DATA / "orders.csv").exists():
        import runpy

        runpy.run_path(str(ROOT / "scripts" / "generate_demo_data.py"), run_name="__main__")

    orders = pd.read_csv(DATA / "orders.csv")
    history = pd.read_csv(DATA / "history.csv")
    devices = pd.read_csv(DATA / "devices.csv")
    users = pd.read_csv(DATA / "users.csv") if (DATA / "users.csv").exists() else None
    label_weights = load_label_weights()

    from refund_abuse_risk.training.splits import time_based_order_split

    train_orders, test_orders, _split = time_based_order_split(
        orders, holdout_days=7.0, min_train=10, min_test=5
    )
    if train_orders.empty or test_orders.empty:
        train_orders = orders.iloc[: max(len(orders) // 2, 1)].reset_index(drop=True)
        test_orders = orders.iloc[len(train_orders) :].reset_index(drop=True)

    model = train_two_head(train_orders, history, devices, users=users, label_weights=label_weights)
    feat = build_order_feature_frame(test_orders, history, devices, users=users)
    feat = apply_proxy_fraud_labels(feat, label_weights)
    scored = model.predict_proba(feat)
    return (
        scored["abuse_label"].astype(int).to_numpy(),
        scored["abuse_score"].to_numpy(),
        scored["fraud_label"].astype(int).to_numpy(),
        scored["fraud_score"].to_numpy(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="")
    parser.add_argument("--vertical", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute decision without writing operating_point",
    )
    parser.add_argument("--approve", metavar="PROPOSAL_ID", help="Approve a pending HIL proposal")
    parser.add_argument("--reject", metavar="PROPOSAL_ID", help="Reject a pending HIL proposal")
    parser.add_argument("--list-pending", action="store_true", help="List pending HIL proposals")
    parser.add_argument("--decided-by", default="ops")
    args = parser.parse_args(argv)

    guardrails = load_guardrails()
    audit = PolicyAuditLog(CONTROL_DB)
    hil = HilProposalStore(CONTROL_DB)

    if args.list_pending:
        print(json.dumps(hil.list_pending(), indent=2))
        return 0

    if args.approve:
        decision = approve_hil_proposal(
            args.approve,
            hil_store=hil,
            audit=audit,
            guardrails=guardrails,
            operating_point_path=OP_PATH,
            decided_by=args.decided_by,
        )
        print(json.dumps(decision.to_dict(), indent=2))
        return 0

    if args.reject:
        decision = reject_hil_proposal(
            args.reject,
            hil_store=hil,
            audit=audit,
            decided_by=args.decided_by,
        )
        print(json.dumps(decision.to_dict(), indent=2))
        return 0

    op = load_operating_point()
    current = dict(op.get("head_thresholds") or {})
    abuse_y, abuse_s, fraud_y, fraud_s = _holdout_scores()
    decision = run_threshold_tuner(
        current_thresholds=current,
        abuse_y=abuse_y,
        abuse_scores=abuse_s,
        fraud_y=fraud_y,
        fraud_scores=fraud_s,
        guardrails=guardrails,
        audit=audit,
        hil_store=hil,
        operating_point_path=OP_PATH,
        apply=not args.dry_run,
        market=args.market,
        vertical=args.vertical,
    )
    print(json.dumps(decision.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
