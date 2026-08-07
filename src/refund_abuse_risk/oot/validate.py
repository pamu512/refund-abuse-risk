"""OOT pack schema / disposition contract validation (not lift metrics)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from refund_abuse_risk.config import load_disposition_labels, load_yaml

PROVEN_OUTCOMES_DEFAULT = (
    "chargeback_lost",
    "bank_dispute_lost",
    "investigator_confirmed_fraud",
    "manual_denied_fraud",
)

REQUIRED_ORDER_COLS = ("order_id", "event_ts")
REQUIRED_DISP_COLS = ("order_id", "outcome")


def _proven_outcomes(disp_cfg: dict[str, Any] | None = None) -> set[str]:
    cfg = disp_cfg or load_disposition_labels()
    out: set[str] = set()
    for name, row in (cfg.get("outcomes") or {}).items():
        if str((row or {}).get("fraud_label_source", "")).lower() == "proven":
            out.add(str(name))
    return out or set(PROVEN_OUTCOMES_DEFAULT)


def count_proven(
    orders: pd.DataFrame,
    dispositions: pd.DataFrame | None,
    *,
    disp_cfg: dict[str, Any] | None = None,
) -> int:
    """Count proven-labeled orders via dispositions and/or fraud_label_source."""
    proven = _proven_outcomes(disp_cfg)
    ids: set[str] = set()
    if dispositions is not None and len(dispositions) and "outcome" in dispositions.columns:
        mask = dispositions["outcome"].astype(str).isin(proven)
        ids.update(dispositions.loc[mask, "order_id"].astype(str))
    if "fraud_label_source" in orders.columns:
        m = orders["fraud_label_source"].astype(str).str.lower() == "proven"
        ids.update(orders.loc[m, "order_id"].astype(str))
    return int(len(ids))


def validate_oot_pack(
    pack_dir: Path | str,
    *,
    floors_path: Path | str | None = None,
    require_dispositions: bool | None = None,
) -> dict[str, Any]:
    """
    Validate pack layout + disposition/proven contract (schema only).

    Uses ``min_pack_n`` for pack size — never ``min_holdout_n`` (eval-only after
    the time split). Does **not** measure model lift; that is ``eval_oot_pack.py``.
    """
    pack = Path(pack_dir)
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    manifest_path = pack / "manifest.yaml"
    if manifest_path.exists():
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            manifest = raw
    else:
        errors.append("missing manifest.yaml")

    profile = str(manifest.get("profile") or ("demo" if pack.name.startswith("demo") else ""))
    if profile and profile not in {"demo", "prod_shaped"}:
        errors.append(f"unknown profile={profile!r} (expected demo|prod_shaped)")

    floors: dict[str, Any] = {}
    if floors_path is not None:
        floors = load_yaml(floors_path)
    elif manifest.get("floors_file"):
        fp = Path(str(manifest["floors_file"]))
        if not fp.is_file():
            # Manifest paths are repo-relative.
            root = Path(__file__).resolve().parents[3]
            fp = root / fp
        floors = load_yaml(fp)
    elif isinstance(manifest.get("floors"), dict):
        floors = dict(manifest["floors"])

    floors = {**floors, **(manifest.get("floors") or {})}

    orders_path = pack / "orders.csv"
    if not orders_path.exists():
        errors.append("missing orders.csv")
        return {
            "ok": False,
            "schema_only": True,
            "errors": errors,
            "profile": profile,
            "n_orders": 0,
            "n_proven": 0,
        }

    orders = pd.read_csv(orders_path)
    for col in REQUIRED_ORDER_COLS:
        if col not in orders.columns:
            errors.append(f"orders.csv missing column {col}")

    disp_path = pack / "dispositions.csv"
    dispositions = pd.read_csv(disp_path) if disp_path.exists() else None
    req_disp = (
        bool(manifest.get("require_dispositions"))
        if require_dispositions is None
        else bool(require_dispositions)
    )
    if profile == "prod_shaped":
        req_disp = True if require_dispositions is None else bool(require_dispositions)

    if req_disp:
        if dispositions is None:
            errors.append("require_dispositions but dispositions.csv missing")
        else:
            for col in REQUIRED_DISP_COLS:
                if col not in dispositions.columns:
                    errors.append(f"dispositions.csv missing column {col}")

    n_proven = count_proven(orders, dispositions)
    min_proven = int(floors.get("min_proven_positives") or 0)
    if min_proven and n_proven < min_proven:
        errors.append(f"n_proven={n_proven} < min_proven_positives={min_proven}")

    # Pack size ≠ holdout size. Holdout floors are enforced in eval_oot_pack after split.
    min_pack = int(floors.get("min_pack_n") or 0)
    if min_pack and len(orders) < min_pack:
        errors.append(f"n_orders={len(orders)} < min_pack_n={min_pack}")
    if "min_holdout_n" in floors and "min_pack_n" not in floors:
        errors.append(
            "floors.min_holdout_n set without min_pack_n "
            "(pack schema must use min_pack_n; holdout is eval-only)"
        )

    return {
        "ok": len(errors) == 0,
        "schema_only": True,
        "errors": errors,
        "profile": profile or None,
        "n_orders": int(len(orders)),
        "n_proven": int(n_proven),
        "require_dispositions": req_disp,
        "floors": {
            k: floors.get(k)
            for k in ("min_pack_n", "min_holdout_n", "min_proven_positives")
            if k in floors
        },
    }
