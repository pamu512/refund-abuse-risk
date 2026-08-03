"""Vendor-agnostic SDK signal ingest for device intelligence + claim vision.

Does not depend on Fingerprint/SHIELD/etc. SDKs — accepts JSON envelopes and
routes payloads through ``device_vision`` adapters onto orders.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from refund_abuse_risk.integrations.device_vision import (
    adapt_claim_vision,
    adapt_device_intelligence,
    merge_platform_signals,
)

DEVICE_FEATURE_KEYS = (
    "device_risk_score",
    "is_emulator",
    "is_cloned_app",
    "is_gps_spoof",
    "is_tampered",
    "customer_courier_same_device",
)
VISION_FEATURE_KEYS = (
    "claim_has_image",
    "claim_image_ai_risk",
    "claim_in_app_capture",
    "pin_required",
    "pin_verified",
    "delivery_geofence_ok",
)

DEFAULT_CFG: dict[str, Any] = {
    "enabled": True,
    "min_confidence": 0.50,
    "write_nested_payloads": True,
    "prefer_higher_confidence": True,
    "sources": {
        "device": {
            "aliases": ["device", "device_intelligence", "fingerprint", "shield", "incognia"]
        },
        "vision": {
            "aliases": ["vision", "claim_vision", "image", "media", "pin"]
        },
    },
}


def _as_conf(value: Any, default: float = 1.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return float(default)
        c = float(value)
    except (TypeError, ValueError):
        return float(default)
    if c > 1.0:
        c = c / 100.0
    return max(0.0, min(1.0, c))


def classify_source(source: str | None, cfg: dict[str, Any] | None = None) -> str | None:
    """Map free-form source/vendor string → device | vision | None."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    raw = str(source or "").strip().lower()
    if not raw:
        return None
    for kind, meta in (c.get("sources") or {}).items():
        aliases = [str(a).lower() for a in (meta or {}).get("aliases") or []]
        if raw == str(kind).lower() or raw in aliases:
            return str(kind)
    return None


def normalize_sdk_event(
    event: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Normalize one SDK envelope.

    Expected keys (aliases in parentheses):
      order_id
      source | kind | signal_type  → device | vision (+ vendor aliases)
      payload | signals | data     → vendor dict
      confidence | conf           → 0–1 (default 1.0)
      event_ts | captured_at | ts
      vendor (optional)
    """
    c = {**DEFAULT_CFG, **(cfg or {})}
    if not c.get("enabled", True):
        return None
    if not isinstance(event, dict):
        return None
    order_id = str(event.get("order_id", "") or "").strip()
    if not order_id:
        return None
    kind = classify_source(
        event.get("source") or event.get("kind") or event.get("signal_type"),
        c,
    )
    if kind is None:
        # Infer from nested keys when source omitted.
        if event.get("device_intelligence") or event.get("device_payload"):
            kind = "device"
        elif event.get("claim_vision") or event.get("vision_payload"):
            kind = "vision"
        else:
            return None

    payload = (
        event.get("payload")
        or event.get("signals")
        or event.get("data")
        or event.get("device_intelligence")
        or event.get("device_payload")
        or event.get("claim_vision")
        or event.get("vision_payload")
        or {}
    )
    if not isinstance(payload, dict):
        return None

    confidence = _as_conf(event.get("confidence", event.get("conf")), default=1.0)
    min_c = float(c.get("min_confidence", 0.50))
    accepted = confidence >= min_c

    if kind == "device":
        features = adapt_device_intelligence(payload) if accepted else {k: 0.0 for k in DEVICE_FEATURE_KEYS}
    else:
        features = adapt_claim_vision(payload) if accepted else {k: 0.0 for k in VISION_FEATURE_KEYS}

    return {
        "order_id": order_id,
        "source": kind,
        "vendor": str(event.get("vendor", "") or ""),
        "confidence": confidence,
        "accepted": accepted,
        "event_ts": event.get("event_ts") or event.get("captured_at") or event.get("ts") or "",
        "payload": payload,
        "features": features,
    }


def attach_sdk_signals(
    order: dict[str, Any],
    *,
    device_event: dict[str, Any] | None = None,
    vision_event: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enrich a single order dict for claim-path / refresh (sync)."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    out = dict(order)
    device_payload = None
    vision_payload = None
    if device_event:
        norm = normalize_sdk_event(
            {**device_event, "order_id": device_event.get("order_id") or out.get("order_id"), "source": "device"},
            c,
        )
        if norm and norm["accepted"]:
            device_payload = norm["payload"]
            out["device_signal_confidence"] = norm["confidence"]
            if c.get("write_nested_payloads", True):
                out["device_intelligence"] = dict(norm["payload"])
            out.update(norm["features"])
        elif norm:
            out["device_signal_confidence"] = norm["confidence"]
    if vision_event:
        norm = normalize_sdk_event(
            {**vision_event, "order_id": vision_event.get("order_id") or out.get("order_id"), "source": "vision"},
            c,
        )
        if norm and norm["accepted"]:
            vision_payload = norm["payload"]
            out["vision_signal_confidence"] = norm["confidence"]
            if c.get("write_nested_payloads", True):
                out["claim_vision"] = dict(norm["payload"])
            out.update(norm["features"])
        elif norm:
            out["vision_signal_confidence"] = norm["confidence"]
    # Prefer merge for any residual nested payloads already on the order.
    return merge_platform_signals(out, device_payload=device_payload, vision_payload=vision_payload)


def _events_frame(events: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        return events.copy()
    return pd.DataFrame(list(events))


def apply_sdk_signals_to_orders(
    orders: pd.DataFrame,
    events: pd.DataFrame | list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Join SDK envelopes onto orders by order_id (latest accepted signal per source).

    Writes flat platform feature columns, confidence columns, and optional nested
    payloads. Rows without signals are unchanged.
    """
    c = {**DEFAULT_CFG, **(cfg or {})}
    out = orders.copy()
    for col in (
        "device_signal_confidence",
        "vision_signal_confidence",
        *DEVICE_FEATURE_KEYS,
        *VISION_FEATURE_KEYS,
    ):
        if col not in out.columns:
            out[col] = 0.0
    for col in ("device_intelligence", "claim_vision"):
        if col not in out.columns:
            out[col] = None

    ev = _events_frame(events)
    if ev.empty or not c.get("enabled", True):
        out.attrs["sdk_events_in"] = 0
        out.attrs["sdk_events_applied"] = 0
        out.attrs["sdk_events_rejected_confidence"] = 0
        return out

    normalized: list[dict[str, Any]] = []
    rejected = 0
    for row in ev.to_dict(orient="records"):
        # CSV may store payload as JSON string.
        payload = row.get("payload")
        if isinstance(payload, str) and payload.strip().startswith(("{", "[")):
            import json

            try:
                row = dict(row)
                row["payload"] = json.loads(payload)
            except json.JSONDecodeError:
                continue
        norm = normalize_sdk_event(row, c)
        if norm is None:
            continue
        if not norm["accepted"]:
            rejected += 1
            continue
        normalized.append(norm)

    applied = 0
    if normalized:
        ndf = pd.DataFrame(normalized)
        # Latest by event_ts per (order_id, source); empty ts sorts last → keep last row.
        ndf["_ts"] = pd.to_datetime(ndf["event_ts"], utc=True, errors="coerce")
        ndf = ndf.sort_values(["order_id", "source", "_ts"], kind="mergesort")
        latest = ndf.groupby(["order_id", "source"], as_index=False).tail(1)

        prefer = bool(c.get("prefer_higher_confidence", True))
        write_nested = bool(c.get("write_nested_payloads", True))
        for row in latest.to_dict(orient="records"):
            oid = str(row["order_id"])
            mask = out["order_id"].astype(str) == oid
            if not mask.any():
                continue
            idx = out.index[mask][0]
            conf = float(row["confidence"])
            feats: dict[str, float] = row["features"] if isinstance(row["features"], dict) else {}
            if row["source"] == "device":
                prev = float(out.at[idx, "device_signal_confidence"] or 0)
                if prefer and prev > conf:
                    continue
                out.at[idx, "device_signal_confidence"] = conf
                for key, val in feats.items():
                    out.at[idx, key] = val
                if write_nested:
                    out.at[idx, "device_intelligence"] = row["payload"]
            else:
                prev = float(out.at[idx, "vision_signal_confidence"] or 0)
                if prefer and prev > conf:
                    continue
                out.at[idx, "vision_signal_confidence"] = conf
                for key, val in feats.items():
                    out.at[idx, key] = val
                if write_nested:
                    out.at[idx, "claim_vision"] = row["payload"]
            applied += 1

    out.attrs["sdk_events_in"] = int(len(ev))
    out.attrs["sdk_events_applied"] = int(applied)
    out.attrs["sdk_events_rejected_confidence"] = int(rejected)
    return out
