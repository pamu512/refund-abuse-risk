"""Load orders with closed-loop dispositions + SDK signals for training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from refund_abuse_risk.config import load_disposition_labels, load_sdk_ingest
from refund_abuse_risk.integrations.sdk_ingest import apply_sdk_signals_to_orders
from refund_abuse_risk.labels.dispositions import apply_dispositions_to_orders


def assert_dispositions_present(
    orders: pd.DataFrame,
    stats: dict[str, Any],
    *,
    min_proven: int = 1,
) -> None:
    """
    Raise ValueError when training data lacks disposition/proven mass.

    Used by ``--require-dispositions`` so synth-only paths cannot silently train.
    Accepts pre-baked ``orders.labeled.csv`` or applied ``dispositions.csv``.
    """
    proven_n = 0
    if "fraud_label_source" in orders.columns and len(orders):
        proven_n = int(
            (orders["fraud_label_source"].astype(str).str.lower() == "proven").sum()
        )
    stats["proven_count"] = proven_n
    if proven_n < int(min_proven):
        raise ValueError(
            f"require_dispositions: proven_count={proven_n} < min_proven={min_proven} "
            f"(source={stats.get('orders_source')})"
        )


def load_training_orders(
    data_dir: Path,
    *,
    prefer_labeled: bool = True,
    apply_dispositions: bool = True,
    apply_sdk: bool = True,
    as_of: str | None = None,
    require_dispositions: bool = False,
    min_train_proven: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Resolve the best training orders table under ``data_dir``.

    Preference:
      1. orders.labeled.csv (if prefer_labeled)
      2. orders.csv (+ optional dispositions.csv / sdk_events.jsonl overlays)
      3. orders.sdk.csv as SDK base if present and apply_sdk
    """
    data_dir = Path(data_dir)
    stats: dict[str, Any] = {
        "orders_source": None,
        "dispositions_applied": 0,
        "sdk_events_applied": 0,
        "proven_rate": None,
        "weak_policy_negative_rate": None,
    }

    labeled_path = data_dir / "orders.labeled.csv"
    base_path = data_dir / "orders.csv"
    sdk_orders_path = data_dir / "orders.sdk.csv"

    if prefer_labeled and labeled_path.exists():
        orders = pd.read_csv(labeled_path)
        stats["orders_source"] = str(labeled_path)
        stats["dispositions_applied"] = -1  # pre-baked
        stats["dispositions_path"] = str(data_dir / "dispositions.csv")
    elif base_path.exists():
        orders = pd.read_csv(base_path)
        stats["orders_source"] = str(base_path)
    elif sdk_orders_path.exists():
        orders = pd.read_csv(sdk_orders_path)
        stats["orders_source"] = str(sdk_orders_path)
        apply_sdk = False  # already enriched
    else:
        raise FileNotFoundError(f"No orders.csv / orders.labeled.csv under {data_dir}")

    # Dispositions overlay when starting from raw orders.
    disp_path = data_dir / "dispositions.csv"
    if (
        apply_dispositions
        and disp_path.exists()
        and stats["orders_source"] == str(base_path)
    ):
        cfg = load_disposition_labels()
        orders = apply_dispositions_to_orders(
            orders, pd.read_csv(disp_path), cfg, as_of=as_of
        )
        stats["dispositions_applied"] = int(orders.attrs.get("dispositions_applied", 0))
        stats["dispositions_path"] = str(disp_path)

    # SDK overlay: prefer jsonl events, else merge columns from orders.sdk.csv.
    sdk_events = data_dir / "sdk_events.jsonl"
    platform_cols = {
        "device_risk_score",
        "is_emulator",
        "is_cloned_app",
        "is_gps_spoof",
        "is_tampered",
        "customer_courier_same_device",
        "claim_has_image",
        "claim_image_ai_risk",
        "claim_in_app_capture",
        "pin_required",
        "pin_verified",
        "delivery_geofence_ok",
        "device_signal_confidence",
        "vision_signal_confidence",
    }
    if apply_sdk and sdk_events.exists():
        events = _load_sdk_events(sdk_events)
        orders = apply_sdk_signals_to_orders(orders, events, load_sdk_ingest())
        stats["sdk_events_applied"] = int(orders.attrs.get("sdk_events_applied", 0))
        stats["sdk_events_path"] = str(sdk_events)
    elif apply_sdk and sdk_orders_path.exists():
        sdk = pd.read_csv(sdk_orders_path)
        cols = [c for c in sdk.columns if c in platform_cols]
        if cols:
            merge_cols = ["order_id", *cols]
            orders = orders.drop(columns=[c for c in cols if c in orders.columns], errors="ignore")
            orders = orders.merge(sdk[merge_cols], on="order_id", how="left")
            stats["sdk_events_applied"] = int(orders[cols[0]].fillna(0).ne(0).sum())
            stats["sdk_events_path"] = str(sdk_orders_path)

    if "fraud_label_source" in orders.columns and len(orders):
        stats["proven_rate"] = float(
            (orders["fraud_label_source"].astype(str).str.lower() == "proven").mean()
        )
    if "weak_policy_negative" in orders.columns and len(orders):
        stats["weak_policy_negative_rate"] = float(
            orders["weak_policy_negative"].astype(float).fillna(0).ge(1).mean()
        )
    if "fraud_label_source" in orders.columns and len(orders):
        stats["proven_count"] = int(
            (orders["fraud_label_source"].astype(str).str.lower() == "proven").sum()
        )
    if require_dispositions:
        assert_dispositions_present(orders, stats, min_proven=min_train_proven)
    return orders, stats


def _load_sdk_events(path: Path) -> pd.DataFrame:
    import json

    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)
