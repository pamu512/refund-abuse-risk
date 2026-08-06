#!/usr/bin/env python3
"""Promote decision_threshold_overlays into operating_point (with backup/rollback)."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OP = ROOT / "config" / "operating_point.default.yaml"
DEFAULT_BACKUP_DIR = ROOT / "config" / "backups"


def bump_policy_version(version: Any) -> str:
    text = str(version or "0.0.0")
    m = re.match(r"^(.*?)(\d+)(\D*)$", text)
    if not m:
        return f"{text}.overlays1"
    head, num, tail = m.group(1), m.group(2), m.group(3)
    return f"{head}{int(num) + 1}{tail}"


def load_overlays(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"overlays file must be a mapping: {path}")
    overlays = data.get("decision_threshold_overlays")
    if overlays is None:
        raise SystemExit(f"missing decision_threshold_overlays in {path}")
    if not isinstance(overlays, list):
        raise SystemExit("decision_threshold_overlays must be a list")
    return [dict(x) for x in overlays if isinstance(x, dict)]


def backup_op(op_path: Path, backup_dir: Path, *, keep: int = 5) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"operating_point.{stamp}.yaml"
    shutil.copy2(op_path, dest)
    existing = sorted(backup_dir.glob("operating_point.*.yaml"))
    while len(existing) > int(keep):
        existing[0].unlink(missing_ok=True)
        existing = sorted(backup_dir.glob("operating_point.*.yaml"))
    return dest


def newest_backup(backup_dir: Path) -> Path | None:
    existing = sorted(backup_dir.glob("operating_point.*.yaml"))
    return existing[-1] if existing else None


def promote(
    *,
    overlays_path: Path,
    op_path: Path,
    backup_dir: Path,
    dry_run: bool,
    require_non_empty: bool,
    backup_keep: int,
) -> dict[str, Any]:
    overlays = load_overlays(overlays_path)
    if require_non_empty and not overlays:
        raise SystemExit("require_non_empty: decision_threshold_overlays is empty")
    op = yaml.safe_load(op_path.read_text(encoding="utf-8")) or {}
    if not isinstance(op, dict):
        raise SystemExit(f"invalid operating point: {op_path}")
    new_version = bump_policy_version(op.get("policy_version"))
    summary = {
        "op": str(op_path),
        "n_overlays": len(overlays),
        "policy_version": new_version,
        "dry_run": bool(dry_run),
        "overlays": [
            {"market": o.get("market"), "vertical": o.get("vertical")} for o in overlays
        ],
    }
    if dry_run:
        return summary
    backup = backup_op(op_path, backup_dir, keep=backup_keep)
    op["decision_threshold_overlays"] = overlays
    op["policy_version"] = new_version
    op_path.write_text(
        yaml.safe_dump(op, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    summary["backup"] = str(backup)
    return summary


def rollback(*, op_path: Path, backup_dir: Path) -> dict[str, Any]:
    src = newest_backup(backup_dir)
    if src is None:
        raise SystemExit(f"no backups under {backup_dir}")
    shutil.copy2(src, op_path)
    return {"op": str(op_path), "restored_from": str(src)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlays", type=Path, default=None)
    parser.add_argument("--op", type=Path, default=DEFAULT_OP)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--backup-keep", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-non-empty", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()

    if args.rollback:
        print(yaml.safe_dump(rollback(op_path=args.op, backup_dir=args.backup_dir), sort_keys=False))
        return 0
    if args.overlays is None:
        print("--overlays PATH required unless --rollback", file=sys.stderr)
        return 2
    summary = promote(
        overlays_path=args.overlays,
        op_path=args.op,
        backup_dir=args.backup_dir,
        dry_run=args.dry_run,
        require_non_empty=args.require_non_empty,
        backup_keep=args.backup_keep,
    )
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
