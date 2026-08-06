#!/usr/bin/env python3
"""Run the minimal claim-path score HTTP API.

Env:
  SCORE_API_TOKEN   (required)
  SCORE_API_HOST    default 127.0.0.1
  SCORE_API_PORT    default 8080

Cache:
  --cache-jsonl PATH  one OrderRiskSnapshot JSON object per line
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from refund_abuse_risk.pipeline.score import OrderRiskCache
from refund_abuse_risk.schemas.models import OrderRiskSnapshot
from refund_abuse_risk.serve.api import config_from_env, serve_forever

ROOT = Path(__file__).resolve().parents[1]


def load_cache_jsonl(path: Path) -> OrderRiskCache:
    cache = OrderRiskCache()
    if not path.is_file():
        raise SystemExit(f"cache jsonl not found: {path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            snap = OrderRiskSnapshot.model_validate(json.loads(line))
            cache.put(snap, {"order_id": snap.order_id})
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-jsonl",
        type=Path,
        default=ROOT / "data" / "score_cache.jsonl",
        help="Precomputed snapshots (claim-path read only)",
    )
    args = parser.parse_args()
    cache = load_cache_jsonl(args.cache_jsonl)
    cfg = config_from_env(cache)
    print(f"[serve_api] loaded {len(cache.all_snapshots())} snapshots", flush=True)
    serve_forever(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
