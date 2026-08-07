#!/usr/bin/env python3
"""P1a: Query the append-only decision archive."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from refund_abuse_risk.ops.decision_archive import DecisionArchive

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "decision_archive.db"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("DECISION_ARCHIVE_PATH", DEFAULT)),
    )
    parser.add_argument("--order-id", type=str, default="")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    if not args.db.is_file():
        print(json.dumps({"ok": False, "error": f"missing archive {args.db}"}))
        return 1
    arch = DecisionArchive(args.db)
    if args.order_id:
        rows = arch.list_for_order(args.order_id, limit=args.limit)
    else:
        rows = arch.list_recent(limit=args.limit)
    print(json.dumps({"ok": True, "n": len(rows), "rows": rows}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
