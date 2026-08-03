"""Normalize device-intelligence + claim-vision vendor payloads into feature columns.

Supports Fingerprint / SHIELD-shaped dicts and in-app vision scores without
taking a hard dependency on any vendor SDK.
"""

from __future__ import annotations

from typing import Any


def _f(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                continue
    return float(default)


def _flag(payload: dict[str, Any], *keys: str) -> float:
    return 1.0 if _f(payload, *keys, default=0.0) >= 1 else 0.0


def adapt_device_intelligence(payload: dict[str, Any] | None) -> dict[str, float]:
    """
    Map vendor device payload → platform risk feature columns.

    Accepted aliases (examples):
      risk_score | device_risk_score | shield_score
      emulator | is_emulator
      cloned_app | is_cloned_app | app_cloners
      gps_spoof | is_gps_spoof | mock_location
      tampered | is_tampered | rooted
      same_device_as_courier | customer_courier_same_device
    """
    p = payload or {}
    risk = _f(p, "device_risk_score", "risk_score", "shield_score", "fingerprint_risk")
    # Some vendors use 0-1; normalize to 0-100.
    if 0.0 < risk <= 1.0:
        risk *= 100.0
    return {
        "device_risk_score": max(0.0, min(100.0, risk)),
        "is_emulator": _flag(p, "is_emulator", "emulator"),
        "is_cloned_app": _flag(p, "is_cloned_app", "cloned_app", "app_cloners"),
        "is_gps_spoof": _flag(p, "is_gps_spoof", "gps_spoof", "mock_location"),
        "is_tampered": _flag(p, "is_tampered", "tampered", "rooted", "jailbroken"),
        "customer_courier_same_device": _flag(
            p, "customer_courier_same_device", "same_device_as_courier"
        ),
    }


def adapt_claim_vision(payload: dict[str, Any] | None) -> dict[str, float]:
    """
    Map claim media / vision payload → feature columns.

    Accepted aliases:
      has_image | claim_has_image
      ai_risk | claim_image_ai_risk | manipulation_score (0-1)
      in_app_capture | claim_in_app_capture
      pin_required | pin_verified | delivery_geofence_ok
    """
    p = payload or {}
    ai = _f(p, "claim_image_ai_risk", "ai_risk", "manipulation_score", "deepfake_score")
    if ai > 1.0:
        ai = ai / 100.0
    return {
        "claim_has_image": _flag(p, "claim_has_image", "has_image"),
        "claim_image_ai_risk": max(0.0, min(1.0, ai)),
        "claim_in_app_capture": _flag(p, "claim_in_app_capture", "in_app_capture"),
        "pin_required": _flag(p, "pin_required"),
        "pin_verified": _flag(p, "pin_verified"),
        "delivery_geofence_ok": _flag(p, "delivery_geofence_ok", "geofence_ok"),
    }


def merge_platform_signals(
    order: dict[str, Any],
    *,
    device_payload: dict[str, Any] | None = None,
    vision_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Overlay adapted signals onto an order/feature dict (order values win if already set > 0)."""
    out = dict(order)
    device = adapt_device_intelligence(device_payload)
    vision = adapt_claim_vision(vision_payload)
    for key, val in {**device, **vision}.items():
        existing = out.get(key)
        try:
            existing_f = float(existing) if existing is not None else 0.0
        except (TypeError, ValueError):
            existing_f = 0.0
        # Prefer explicit non-zero order fields; else take adapter.
        out[key] = existing_f if existing_f > 0 else val
    return out
