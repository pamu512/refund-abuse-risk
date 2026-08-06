#!/usr/bin/env python3
"""Run the declarative overnight ops profile (config/ops.overnight.yaml)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "config" / "ops.overnight.yaml"


def _load_profile(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "steps" not in data:
        raise SystemExit(f"invalid overnight profile: {path}")
    return data


def _format_argv(argv: list[str], paths: dict[str, str]) -> list[str]:
    out: list[str] = []
    for part in argv:
        try:
            out.append(part.format(**paths))
        except KeyError as exc:
            raise SystemExit(f"unknown path placeholder in {part!r}: {exc}") from exc
    return out


def promote_ok_from_metrics(path: Path) -> bool:
    """Read backtest metrics JSON; True only when honesty.promote_ok is true."""
    import json

    if not path.is_file():
        return False
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        return False
    return bool((metrics.get("honesty") or {}).get("promote_ok"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(os.environ.get("OPS_OVERNIGHT_PROFILE", DEFAULT_PROFILE)),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print steps only")
    parser.add_argument(
        "--require-promote",
        action="store_true",
        help="Fail if backtest honesty.promote_ok is false (prod)",
    )
    args = parser.parse_args()

    profile = _load_profile(args.profile.resolve())
    paths = {str(k): str(v) for k, v in (profile.get("paths") or {}).items()}
    # Env overrides for Downstream cutover.
    if os.environ.get("FEEDS_CONFIG"):
        paths["feeds_config"] = os.environ["FEEDS_CONFIG"]
    if os.environ.get("DATA_DIR"):
        paths["data_dir"] = os.environ["DATA_DIR"]
    if os.environ.get("MODEL_OUT"):
        paths["model_out"] = os.environ["MODEL_OUT"]
    if os.environ.get("OOT_PACK"):
        paths["oot_pack"] = os.environ["OOT_PACK"]

    abort = bool(profile.get("abort_on_step_failure", True))
    require_promote = bool(args.require_promote or profile.get("require_promote"))

    for step in profile["steps"]:
        step_id = str(step.get("id", "?"))
        when_env = step.get("when_env")
        if when_env and not os.environ.get(str(when_env)):
            print(
                f"[ops_overnight] skip {step_id} (env {when_env} unset)",
                flush=True,
            )
            continue
        argv = _format_argv(list(step.get("argv") or []), paths)
        optional = bool(step.get("optional"))
        print(f"[ops_overnight] === {step_id} ===", flush=True)
        print(f"[ops_overnight] {' '.join(argv)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(argv, cwd=str(ROOT), check=False)
        if proc.returncode != 0:
            if optional:
                print(
                    f"[ops_overnight] optional step {step_id} failed "
                    f"(exit {proc.returncode}); continuing",
                    flush=True,
                )
                continue
            if abort:
                print(
                    f"[ops_overnight] abort at {step_id} (exit {proc.returncode})",
                    file=sys.stderr,
                )
                return int(proc.returncode)
        if require_promote and step_id == "backtest":
            metrics_path = ROOT / "examples" / "csv_demo" / "backtest_metrics.json"
            if not promote_ok_from_metrics(metrics_path):
                print(
                    "[ops_overnight] abort: honesty.promote_ok is not true "
                    f"({metrics_path})",
                    file=sys.stderr,
                )
                return 1
            print("[ops_overnight] require_promote: honesty.promote_ok=true", flush=True)

    print("[ops_overnight] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
