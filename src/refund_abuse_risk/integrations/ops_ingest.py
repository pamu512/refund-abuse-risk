"""Ingest live ops metrics into monitoring.ops_snapshot for promote gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

# Keys understood by evaluate_monitoring_gates (+ audit fields allowed).
OPS_METRIC_KEYS = (
    "hold_rate",
    "live_override_rate",
    "shadow_override_rate",
    "refund_grant_rate",
    "refund_dollar_per_order",
    "cs_queue_depth",
)

DEFAULT_CFG: dict[str, Any] = {
    "enabled": True,
    "require_as_of": False,
}


def normalize_ops_snapshot(
    payload: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Normalize an ops envelope into a flat snapshot dict.

    Accepts either flat metrics or ``{metrics: {...}, as_of, source}``.
    Non-finite / unknown keys are dropped; known keys coerced to float.
    """
    c = {**DEFAULT_CFG, **(cfg or {})}
    if not c.get("enabled", True):
        return {}
    if not isinstance(payload, dict):
        return {}

    raw = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    out: dict[str, Any] = {}
    for key in OPS_METRIC_KEYS:
        if key not in raw or raw[key] is None:
            continue
        try:
            out[key] = float(raw[key])
        except (TypeError, ValueError):
            continue

    as_of = payload.get("as_of") or payload.get("event_ts") or raw.get("as_of")
    if as_of:
        out["as_of"] = str(as_of)
    elif c.get("require_as_of"):
        return {}
    source = payload.get("source") or raw.get("source")
    if source:
        out["source"] = str(source)
    return out


def load_ops_snapshot_file(path: Path | str) -> dict[str, Any]:
    """Load JSON / JSONL (last object) / YAML ops envelope."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        last: dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                last = obj
        return normalize_ops_snapshot(last)
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
        return normalize_ops_snapshot(data if isinstance(data, dict) else {})
    data = json.loads(text)
    if isinstance(data, list):
        data = data[-1] if data else {}
    return normalize_ops_snapshot(data if isinstance(data, dict) else {})


def merge_ops_snapshot(
    monitoring_cfg: dict[str, Any] | None,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Return monitoring cfg with ops_snapshot merged (snapshot wins on overlap)."""
    cfg = dict(monitoring_cfg or {})
    current = dict(cfg.get("ops_snapshot") or {})
    for k, v in snapshot.items():
        if k in OPS_METRIC_KEYS or k in {"as_of", "source"}:
            current[k] = v
    cfg["ops_snapshot"] = current
    return cfg


def write_ops_snapshot_sidecar(path: Path | str, snapshot: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return path


def apply_ops_snapshot_to_operating_point(
    operating_point_path: Path | str,
    snapshot: dict[str, Any],
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Merge snapshot into operating_point.monitoring.ops_snapshot (optionally persist)."""
    path = Path(operating_point_path)
    op = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mon = merge_ops_snapshot(op.get("monitoring") or {}, snapshot)
    op["monitoring"] = mon
    if write:
        path.write_text(
            yaml.safe_dump(op, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    return op
