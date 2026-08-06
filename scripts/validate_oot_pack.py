#!/usr/bin/env python3
"""Validate OOT pack schema / disposition contract (not model lift)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from refund_abuse_risk.oot.validate import validate_oot_pack

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=ROOT / "data" / "oot_packs" / "prod_shaped_v1",
    )
    parser.add_argument(
        "--floors",
        type=Path,
        default=None,
        help="Floors YAML (default: manifest floors_file / floors / oot_floors.prod for prod_shaped)",
    )
    parser.add_argument(
        "--require-dispositions",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()

    floors = args.floors
    if floors is None and "prod_shaped" in str(args.pack_dir):
        floors = ROOT / "config" / "oot_floors.prod.yaml"

    result = validate_oot_pack(
        args.pack_dir,
        floors_path=floors,
        require_dispositions=args.require_dispositions,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
