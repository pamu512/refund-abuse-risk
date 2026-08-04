"""Vertical / market policy priors → order features."""

from __future__ import annotations

from typing import Any

import pandas as pd

from refund_abuse_risk.config import load_vertical_policy

POLICY_FEATURE_COLUMNS: tuple[str, ...] = (
    "policy_claim_window_hours",
    "policy_photo_prior",
    "policy_remedy_cash_bias",
    "policy_returns_allowed",
    "hours_since_delivery",
    "claim_window_remaining_frac",
)

_VERTICAL_ALIASES = {
    "q_commerce": "qcommerce",
    "quick_commerce": "qcommerce",
    "quickcommerce": "qcommerce",
    "instamart": "qcommerce",
    "groceries": "grocery",
}


def normalize_vertical(vertical: str | None) -> str:
    v = str(vertical or "food").strip().lower()
    return _VERTICAL_ALIASES.get(v, v)


def resolve_policy_priors(
    *,
    market: str | None,
    vertical: str | None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Market overlay wins, else vertical default, else fallback."""
    cfg = cfg if cfg is not None else load_vertical_policy()
    fallback = dict(cfg.get("fallback") or {})
    vert = normalize_vertical(vertical)
    mkt = str(market or "").strip().upper()
    base = dict((cfg.get("defaults") or {}).get(vert) or fallback)
    for ov in cfg.get("overlays") or []:
        if str(ov.get("market", "")).strip().upper() != mkt:
            continue
        if normalize_vertical(ov.get("vertical")) != vert:
            continue
        for key in (
            "claim_window_hours",
            "photo_prior",
            "remedy_cash_bias",
            "returns_allowed",
        ):
            if key in ov and ov[key] is not None:
                base[key] = ov[key]
        break
    return {
        "claim_window_hours": float(base.get("claim_window_hours", 36)),
        "photo_prior": float(base.get("photo_prior", 0.4)),
        "remedy_cash_bias": float(base.get("remedy_cash_bias", 0.45)),
        "returns_allowed": float(base.get("returns_allowed", 0)),
    }


def _hours_since_delivery(order: dict[str, Any]) -> float | None:
    """Hours from delivery to claim; None if clocks missing (avoid event=event → always 0)."""
    delivered = order.get("delivered_ts") or order.get("delivery_ts")
    claim_at = order.get("claim_ts") or order.get("as_of_ts")
    # Fallback: delivered status + event_ts as delivery, only if claim_ts present.
    if delivered is None and str(order.get("status", "")).lower() == "delivered":
        delivered = order.get("event_ts")
    if delivered is None or claim_at is None:
        return None
    try:
        t0 = pd.Timestamp(delivered)
        t1 = pd.Timestamp(claim_at)
        if t0.tzinfo is None:
            t0 = t0.tz_localize("UTC")
        else:
            t0 = t0.tz_convert("UTC")
        if t1.tzinfo is None:
            t1 = t1.tz_localize("UTC")
        else:
            t1 = t1.tz_convert("UTC")
    except (TypeError, ValueError):
        return None
    hours = (t1 - t0).total_seconds() / 3600.0
    if hours < 0:
        return 0.0
    return float(hours)


def policy_prior_features(
    order: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    priors = resolve_policy_priors(
        market=order.get("market"),
        vertical=order.get("vertical"),
        cfg=cfg,
    )
    window = max(1e-6, float(priors["claim_window_hours"]))
    hours = _hours_since_delivery(order)
    if hours is None:
        hours_f = 0.0
        remaining = 0.0
    else:
        hours_f = float(hours)
        remaining = max(0.0, min(1.0, (window - hours_f) / window))
    return {
        "policy_claim_window_hours": float(priors["claim_window_hours"]),
        "policy_photo_prior": float(priors["photo_prior"]),
        "policy_remedy_cash_bias": float(priors["remedy_cash_bias"]),
        "policy_returns_allowed": float(priors["returns_allowed"]),
        "hours_since_delivery": hours_f,
        "claim_window_remaining_frac": remaining,
    }

