"""Validate / clamp head-threshold overlays within ops guardrails."""

from __future__ import annotations

from typing import Any


class GuardrailError(ValueError):
    """Proposal violates configured min/max guardrails."""


THRESHOLD_KEYS = (
    "target_pattern_recall",
    "abuse_soft_friction",
    "fraud_soft_friction",
    "abuse_hold_review",
    "fraud_hold_review",
    "abuse_auto_deny",
    "fraud_auto_deny",
)


def bound_for(guardrails: dict[str, Any], key: str) -> tuple[float, float] | None:
    bounds = guardrails.get("bounds") or {}
    path = f"head_thresholds.{key}"
    row = bounds.get(path)
    if row is None:
        return None
    return float(row["min"]), float(row["max"])


def validate_head_thresholds(
    thresholds: dict[str, Any],
    guardrails: dict[str, Any],
) -> None:
    for key, value in thresholds.items():
        if key not in THRESHOLD_KEYS:
            raise GuardrailError(f"param not tunable under guardrails: {key}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise GuardrailError(f"{key} must be numeric")
        bounds = bound_for(guardrails, key)
        if bounds is None:
            raise GuardrailError(f"param not tunable under guardrails: {key}")
        lo, hi = bounds
        v = float(value)
        if v < lo or v > hi:
            raise GuardrailError(f"{key}={v} outside guardrail [{lo}, {hi}]")


def clamp_head_thresholds(
    thresholds: dict[str, Any],
    guardrails: dict[str, Any],
) -> dict[str, float]:
    """Clamp known keys into bounds; drop unknown keys."""
    out: dict[str, float] = {}
    for key in THRESHOLD_KEYS:
        if key not in thresholds:
            continue
        v = float(thresholds[key])
        bounds = bound_for(guardrails, key)
        if bounds is None:
            continue
        lo, hi = bounds
        out[key] = float(min(hi, max(lo, v)))
    return out


def max_auto_delta(guardrails: dict[str, Any], key: str) -> float:
    cfg = guardrails.get("auto_apply") or {}
    by_key = cfg.get("max_abs_delta_by_key") or {}
    if key in by_key:
        return float(by_key[key])
    return float(cfg.get("max_abs_delta", 5.0))
