"""Tune head thresholds for pattern recall (learning-primary operating point)."""

from __future__ import annotations

from typing import Any

import numpy as np


def threshold_at_recall(
    y_true: np.ndarray | list[float],
    scores: np.ndarray | list[float],
    target_recall: float = 0.98,
) -> float | None:
    """
    Lowest score threshold such that recall of positives is >= target_recall.

    Scores are "higher = more risky". Returns None if there are no positives.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same shape")
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    target = float(np.clip(target_recall, 0.0, 1.0))
    order = np.argsort(-s)
    y_sorted = y[order]
    s_sorted = s[order]
    tp = 0.0
    for i, lab in enumerate(y_sorted):
        tp += lab
        if tp / n_pos >= target:
            return float(s_sorted[i])
    return float(s_sorted[-1])


def precision_at_threshold(
    y_true: np.ndarray | list[float],
    scores: np.ndarray | list[float],
    threshold: float,
) -> float | None:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    pred = s >= float(threshold)
    n_pred = int(pred.sum())
    if n_pred == 0:
        return None
    return float(y[pred].sum() / n_pred)


def recall_at_threshold(
    y_true: np.ndarray | list[float],
    scores: np.ndarray | list[float],
    threshold: float,
) -> float | None:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    return float(y[s >= float(threshold)].sum() / n_pos)


def pattern_flag_recall(
    abuse_y: np.ndarray | list[float],
    fraud_y: np.ndarray | list[float],
    abuse_scores: np.ndarray | list[float],
    fraud_scores: np.ndarray | list[float],
    abuse_thr: float,
    fraud_thr: float,
) -> float | None:
    """Recall of (abuse|fraud) positives when either head clears its threshold."""
    a_y = np.asarray(abuse_y, dtype=float)
    f_y = np.asarray(fraud_y, dtype=float)
    a_s = np.asarray(abuse_scores, dtype=float)
    f_s = np.asarray(fraud_scores, dtype=float)
    pattern = ((a_y >= 1) | (f_y >= 1)).astype(float)
    n_pos = int(pattern.sum())
    if n_pos == 0:
        return None
    flagged = (a_s >= float(abuse_thr)) | (f_s >= float(fraud_thr))
    return float(pattern[flagged].sum() / n_pos)


def recommend_head_thresholds(
    abuse_y: np.ndarray | list[float],
    abuse_scores: np.ndarray | list[float],
    fraud_y: np.ndarray | list[float],
    fraud_scores: np.ndarray | list[float],
    *,
    target_recall: float = 0.98,
) -> dict[str, Any]:
    """
    Suggest soft / hold / deny head thresholds from validation scores.

    Soft ≈ catch ~target_recall of labeled patterns (the 98% mass).
    Hold ≈ slightly stricter. Deny ≈ high-precision slice (~50% recall of positives).
    """
    abuse_soft = threshold_at_recall(abuse_y, abuse_scores, target_recall)
    fraud_soft = threshold_at_recall(fraud_y, fraud_scores, target_recall)
    abuse_hold = threshold_at_recall(abuse_y, abuse_scores, max(0.85, target_recall - 0.08))
    fraud_hold = threshold_at_recall(fraud_y, fraud_scores, max(0.85, target_recall - 0.08))
    abuse_deny = threshold_at_recall(abuse_y, abuse_scores, 0.50)
    fraud_deny = threshold_at_recall(fraud_y, fraud_scores, 0.50)

    # Fallbacks when a head has no positives in the slice.
    def _or(value: float | None, default: float) -> float:
        return float(default if value is None else value)

    out = {
        "target_pattern_recall": float(target_recall),
        "abuse_soft_friction": _or(abuse_soft, 35.0),
        "fraud_soft_friction": _or(fraud_soft, 30.0),
        "abuse_hold_review": _or(abuse_hold, 50.0),
        "fraud_hold_review": _or(fraud_hold, 45.0),
        "abuse_auto_deny": _or(abuse_deny, 75.0),
        "fraud_auto_deny": _or(fraud_deny, 65.0),
    }
    # Enforce monotone soft <= hold <= deny, with floors so tiny holdouts
    # cannot collapse deny into near-zero (demo / sparse validation safeguard).
    floors = {
        "abuse_soft_friction": 10.0,
        "fraud_soft_friction": 10.0,
        "abuse_hold_review": 25.0,
        "fraud_hold_review": 25.0,
        "abuse_auto_deny": 50.0,
        "fraud_auto_deny": 50.0,
    }
    for key, floor in floors.items():
        out[key] = max(float(out[key]), floor)
    for head in ("abuse", "fraud"):
        soft_k = f"{head}_soft_friction"
        hold_k = f"{head}_hold_review"
        deny_k = f"{head}_auto_deny"
        soft, hold, deny = out[soft_k], out[hold_k], out[deny_k]
        hold = max(hold, soft)
        deny = max(deny, hold)
        out[soft_k], out[hold_k], out[deny_k] = soft, hold, deny

    out["pattern_recall_at_soft"] = pattern_flag_recall(
        abuse_y,
        fraud_y,
        abuse_scores,
        fraud_scores,
        out["abuse_soft_friction"],
        out["fraud_soft_friction"],
    )
    out["abuse_precision_at_soft"] = precision_at_threshold(
        abuse_y, abuse_scores, out["abuse_soft_friction"]
    )
    out["fraud_precision_at_soft"] = precision_at_threshold(
        fraud_y, fraud_scores, out["fraud_soft_friction"]
    )
    return out
