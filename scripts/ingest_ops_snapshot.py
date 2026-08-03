#!/usr/bin/env python3
"""Ingest live ops metrics into monitoring.ops_snapshot (promote-gate feed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from refund_abuse_risk.integrations.ops_ingest import (
    apply_ops_snapshot_to_operating_point,
    load_ops_snapshot_file,
    write_ops_snapshot_sidecar,
)
from refund_abuse_risk.scoring.monitoring import evaluate_monitoring_gates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OP = ROOT / "config" / "operating_point.default.yaml"
DEFAULT_SIDECAR = ROOT / "data" / "ops_snapshot.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        required=True,
        help="Ops envelope: .json / .jsonl / .yaml (flat metrics or {metrics:{...}})",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=DEFAULT_SIDECAR,
        help="Write normalized snapshot JSON (read by backtest)",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Also merge into operating_point.monitoring.ops_snapshot",
    )
    parser.add_argument("--operating-point", type=Path, default=DEFAULT_OP)
    parser.add_argument(
        "--check-gate",
        action="store_true",
        help="Evaluate monitoring gate with this snapshot (ECE/PSI treated as unknown)",
    )
    args = parser.parse_args()

    snapshot = load_ops_snapshot_file(args.events)
    if not snapshot:
        raise SystemExit("No usable ops metrics in events file")

    write_ops_snapshot_sidecar(args.sidecar, snapshot)
    summary: dict = {
        "sidecar": str(args.sidecar),
        "snapshot": snapshot,
        "wrote_config": False,
    }
    if args.write_config:
        apply_ops_snapshot_to_operating_point(args.operating_point, snapshot, write=True)
        summary["wrote_config"] = True
        summary["operating_point"] = str(args.operating_point)

    if args.check_gate:
        import yaml

        op = yaml.safe_load(args.operating_point.read_text(encoding="utf-8")) or {}
        mon = dict(op.get("monitoring") or {})
        mon["ops_snapshot"] = {**(mon.get("ops_snapshot") or {}), **snapshot}
        gate = evaluate_monitoring_gates(
            decision_ece=None,
            psi_train_test=None,
            monitoring_cfg=mon,
            ops_metrics=None,
        )
        summary["gate"] = gate

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
