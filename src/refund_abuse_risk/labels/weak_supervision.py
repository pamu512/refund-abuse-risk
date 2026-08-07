"""Snorkel-lite labeling functions → weak proxy labels (never overwrite proven).

P1b from platform research (Swiggy DeFraudNet-shaped). No Snorkel dependency:
each LF returns 1 (abuse/fraud-ish), 0 (legit), or None (abstain). Combined via
majority of non-abstain votes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

# LF(row_dict) -> 1 | 0 | None
LabelingFunction = Callable[[dict[str, Any]], int | None]


def lf_high_user_refund_rate(row: dict[str, Any], *, min_rate: float = 0.35) -> int | None:
    rate = row.get("user_refund_rate_30d")
    if rate is None or (isinstance(rate, float) and pd.isna(rate)):
        return None
    return 1 if float(rate) >= float(min_rate) else 0


def lf_elevated_uv_edge(row: dict[str, Any]) -> int | None:
    elev = row.get("uv_edge_elevated")
    if elev is None or (isinstance(elev, float) and pd.isna(elev)):
        return None
    return 1 if float(elev) >= 0.5 else 0


def lf_claim_with_thin_history(row: dict[str, Any]) -> int | None:
    reason = str(row.get("claim_reason") or "").strip()
    if not reason:
        return None
    lifetime = row.get("user_lifetime_orders")
    if lifetime is None or (isinstance(lifetime, float) and pd.isna(lifetime)):
        return None
    # First-/early-order claims are noisier — weak positive only when very thin.
    if float(lifetime) <= 2.0:
        return 1
    return 0


def lf_device_multi_account(row: dict[str, Any], *, min_accounts: float = 3.0) -> int | None:
    n = row.get("accounts_per_device")
    if n is None:
        n = row.get("device_cluster_size")
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return None
    return 1 if float(n) >= float(min_accounts) else 0


def default_labeling_functions() -> list[tuple[str, LabelingFunction]]:
    return [
        ("high_user_refund_rate", lf_high_user_refund_rate),
        ("elevated_uv_edge", lf_elevated_uv_edge),
        ("claim_thin_history", lf_claim_with_thin_history),
        ("device_multi_account", lf_device_multi_account),
    ]


def combine_lf_votes(votes: list[int]) -> tuple[int | None, float]:
    """Majority of {0,1}; ties → abstain. Confidence = majority / n_votes."""
    if not votes:
        return None, 0.0
    pos = sum(1 for v in votes if int(v) == 1)
    neg = sum(1 for v in votes if int(v) == 0)
    n = pos + neg
    if n == 0:
        return None, 0.0
    if pos == neg:
        return None, 0.0
    label = 1 if pos > neg else 0
    conf = max(pos, neg) / n
    return label, float(conf)


def apply_labeling_functions(
    frame: pd.DataFrame,
    labeling_functions: list[tuple[str, LabelingFunction]] | None = None,
) -> pd.DataFrame:
    """
    Add weak columns; never changes proven fraud rows' fraud_label_source.

    Columns: ``weak_label``, ``weak_label_confidence``, ``weak_lf_votes`` (JSON-ish str),
    ``abuse_label_weak``. Optionally sets discovery fraud when weak_label=1 and not proven.
    """
    lfs = labeling_functions or default_labeling_functions()
    out = frame.copy()
    if "fraud_label_source" not in out.columns:
        out["fraud_label_source"] = ""
    if "abuse_label" not in out.columns:
        out["abuse_label"] = 0
    if "fraud_label" not in out.columns:
        out["fraud_label"] = 0

    weak_labels: list[int] = []
    confidences: list[float] = []
    vote_strs: list[str] = []
    abuse_weak: list[int] = []

    for _, row in out.iterrows():
        rd = row.to_dict()
        votes: list[int] = []
        named: list[str] = []
        for name, fn in lfs:
            v = fn(rd)
            if v is None:
                continue
            vi = int(v)
            if vi not in (0, 1):
                continue
            votes.append(vi)
            named.append(f"{name}={vi}")
        label, conf = combine_lf_votes(votes)
        weak_labels.append(-1 if label is None else int(label))
        confidences.append(float(conf))
        vote_strs.append(";".join(named))
        abuse_weak.append(0 if label is None else int(label))

    out["weak_label"] = weak_labels
    out["weak_label_confidence"] = confidences
    out["weak_lf_votes"] = vote_strs
    out["abuse_label_weak"] = abuse_weak

    # Apply weak abuse for training mass; never overwrite proven fraud source.
    for idx in out.index:
        src = str(out.at[idx, "fraud_label_source"] or "").lower()
        if src == "proven":
            continue
        w = int(out.at[idx, "weak_label"])
        if w < 0:
            continue
        if w == 1:
            out.at[idx, "abuse_label"] = max(int(out.at[idx, "abuse_label"] or 0), 1)
            # Discovery-tier fraud only when still unlabeled.
            if int(out.at[idx, "fraud_label"] or 0) == 0 and src in ("", "proxy", "discovery"):
                if float(out.at[idx, "weak_label_confidence"]) >= 0.75:
                    out.at[idx, "fraud_label"] = 1
                    out.at[idx, "fraud_label_source"] = "discovery"
        # w == 0: leave existing labels; abstention of negative is intentional.
    return out


def mint_weak_labels_from_lfs(
    orders: pd.DataFrame,
    *,
    labeling_functions: list[tuple[str, LabelingFunction]] | None = None,
) -> pd.DataFrame:
    """Public entry: apply default or custom LFs to an orders/feature frame."""
    return apply_labeling_functions(orders, labeling_functions=labeling_functions)
