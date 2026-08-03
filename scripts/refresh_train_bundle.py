#!/usr/bin/env python3
"""One-shot: closed-loop labels → serve-path train with time-OOT proven metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "large")
    parser.add_argument("--max-orders", type=int, default=8000)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "models" / "two_head_closed_loop.joblib",
    )
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_generate:
        subprocess.check_call(
            [
                py,
                str(ROOT / "scripts" / "generate_closed_loop_labels.py"),
                "--data-dir",
                str(args.data_dir),
                "--max-orders",
                str(args.max_orders),
                "--qa-n",
                "80",
                "--apply",
                "--with-sdk",
                "--seed",
                str(args.seed),
            ]
        )

    subprocess.check_call(
        [
            py,
            "-u",
            str(ROOT / "scripts" / "train_model.py"),
            "--data-dir",
            str(args.data_dir),
            "--feature-source",
            "serve",
            "--max-rows",
            str(args.max_rows),
            "--passes",
            str(args.passes),
            "--oot-days",
            "7",
            "--model-out",
            str(args.model_out),
            "--seed",
            str(args.seed),
        ]
    )

    summary_path = args.model_out.with_suffix(".train_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    print(
        json.dumps(
            {
                "model_out": str(args.model_out),
                "closed_loop": (summary.get("feature_stats") or {}).get("closed_loop"),
                "oot": summary.get("oot"),
                "discovery_rate": summary.get("discovery_rate"),
                "n_passes_ran": summary.get("n_passes_ran"),
                "early_stopped": summary.get("early_stopped"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
