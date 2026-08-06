#!/usr/bin/env bash
# Thin wrapper — prefer: python scripts/ops_overnight.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python scripts/ops_overnight.py "$@"
