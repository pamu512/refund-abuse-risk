#!/usr/bin/env python3
"""Pull production-shaped feeds from configured sources → stage → apply."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from refund_abuse_risk.config import load_yaml
from refund_abuse_risk.integrations.feeds import load_feeds_config, run_feeds

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Feeds YAML (default: config/feeds.default.yaml)",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated feed names (dispositions,sdk_events,ops_snapshot)",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Copy extracts under stage_root; do not apply",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate + stage + report apply plan without writing labeled/ops outputs",
    )
    parser.add_argument("--as-of", default=None, help="Staging stamp / disposition as-of hint")
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config else load_feeds_config()
    only = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else None
    report = run_feeds(
        cfg,
        only=only,
        stage_only=bool(args.stage_only),
        dry_run=bool(args.dry_run),
        as_of=args.as_of,
        root=ROOT,
    )
    print(json.dumps(report, indent=2))
    if not report.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
