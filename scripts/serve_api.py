#!/usr/bin/env python3
"""Local fixture precompute + claim-path HTTP demo.

Scores an in-repo CSV fixture with train_two_head / precompute_orders
(DecisionStacker). HTTP GETs only read the precomputed cache — this is not a
live scorer.

Env:
  SCORE_API_TOKEN   (required)
  SCORE_API_HOST    default 127.0.0.1
  SCORE_API_PORT    default 8080

Fixture:
  --data-dir PATH   orders.csv + history.csv + devices.csv (+ optional users.csv)
                    default: data/oot_packs/prod_shaped_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from refund_abuse_risk.pipeline.score import (
    OrderRiskCache,
    precompute_orders,
    train_two_head,
)
from refund_abuse_risk.serve.api import config_from_env, serve_forever

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "data" / "oot_packs" / "prod_shaped_v1"
_REQUIRED_CSVS = ("orders.csv", "history.csv", "devices.csv")


def score_fixture_dir(data_dir: Path) -> OrderRiskCache:
    data_dir = Path(data_dir)
    missing = [name for name in _REQUIRED_CSVS if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"fixture CSVs missing in {data_dir}: {', '.join(missing)}"
        )
    orders = pd.read_csv(data_dir / "orders.csv")
    if orders.empty:
        raise FileNotFoundError(f"fixture orders.csv is empty: {data_dir / 'orders.csv'}")
    history = pd.read_csv(data_dir / "history.csv")
    devices = pd.read_csv(data_dir / "devices.csv")
    users_path = data_dir / "users.csv"
    users = pd.read_csv(users_path) if users_path.is_file() else None
    model = train_two_head(orders, history, devices, users=users)
    return precompute_orders(orders, history, devices, model, users=users)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Fixture CSV dir (orders/history/devices). Precompute only — not live scoring.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cache = score_fixture_dir(args.data_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    cfg = config_from_env(cache)
    print(
        json.dumps(
            {
                "mode": "fixture_precompute",
                "data_dir": str(Path(args.data_dir).resolve()),
                "snapshots": len(cache.all_snapshots()),
                "note": "claim-path reads of precomputed snapshots; not live scoring",
            }
        ),
        flush=True,
    )
    serve_forever(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
