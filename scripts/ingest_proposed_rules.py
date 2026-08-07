#!/usr/bin/env python3
"""Ingest data/proposed_rules into OP overlays / effect_rules (shadow by default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from refund_abuse_risk.config import load_yaml
from refund_abuse_risk.ops.rule_ingest import run_ingest, summary_to_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "rule_ingest.default.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when unknown kinds or soft gate drops occur",
    )
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repo root for relative source/target paths",
    )
    args = parser.parse_args()

    try:
        cfg = load_yaml(args.config)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"config: {exc}"}), file=sys.stderr)
        return 1

    try:
        summary = run_ingest(
            cfg=cfg,
            root=args.root,
            dry_run=bool(args.dry_run),
            strict=bool(args.strict),
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1

    text = summary_to_json(summary)
    sys.stdout.write(text)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text, encoding="utf-8")

    if summary.get("skipped") and summary.get("reason") == "disabled":
        return 0
    if summary.get("ok") is False or summary.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
