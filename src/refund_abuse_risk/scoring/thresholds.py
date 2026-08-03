"""Tune head thresholds for pattern recall under precision / FP constraints."""

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


def threshold_at_recall_costed(
    y_true: np.ndarray | list[float],
    scores: np.ndarray | list[float],
    *,
    target_recall: float = 0.98,
    min_precision: float | None = None,
    max_fp_rate: float | None = None,
    amounts: np.ndarray | list[float] | None = None,
    max_fp_refund_dollars_mean: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """
    Strictest (highest) threshold that still hits target_recall under cost constraints.

    Among thresholds with Recall >= target_recall AND Precision >= min_precision
    AND FP-rate <= max_fp_rate AND (optional) mean FP refund $ <= budget,
    pick the maximum score cutoff (least spammy soft tier).
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    amts = None if amounts is None else np.asarray(amounts, dtype=float)
    if amts is not None and len(amts) != len(y):
        amts = None
    info: dict[str, Any] = {
        "feasible": False,
        "target_recall": float(target_recall),
        "min_precision": min_precision,
        "max_fp_rate": max_fp_rate,
        "max_fp_refund_dollars_mean": max_fp_refund_dollars_mean,
    }
    n = int(len(y))
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0:
        info["reason"] = "no_positives"
        return None, info

    # Candidate cutoffs = unique scores descending.
    candidates = np.unique(s)[::-1]
    best_t: float | None = None
    best_stats: dict[str, float] = {}
    for t in candidates:
        pred = s >= float(t)
        tp = float(y[pred].sum())
        fp_mask = (y < 0.5) & pred
        fp = float(fp_mask.sum())
        n_hat = int(pred.sum())
        prec = (tp / n_hat) if n_hat else 0.0
        rec = tp / n_pos
        fpr = (fp / n_neg) if n_neg > 0 else 0.0
        fp_dollar_mean = None
        if amts is not None and fp > 0:
            fp_dollar_mean = float(amts[fp_mask].mean())
        elif amts is not None:
            fp_dollar_mean = 0.0
        if rec + 1e-12 < float(target_recall):
            continue
        if min_precision is not None and prec + 1e-12 < float(min_precision):
            continue
        if max_fp_rate is not None and fpr > float(max_fp_rate) + 1e-12:
            continue
        if (
            max_fp_refund_dollars_mean is not None
            and fp_dollar_mean is not None
            and fp_dollar_mean > float(max_fp_refund_dollars_mean) + 1e-12
        ):
            continue
        # Feasible — keep highest t (first in descending scan).
        best_t = float(t)
        best_stats = {"recall": rec, "precision": prec, "fp_rate": fpr}
        if fp_dollar_mean is not None:
            best_stats["fp_refund_dollars_mean"] = fp_dollar_mean
        break

    if best_t is None:
        # Fall back to recall-only and mark infeasible under cost constraints.
        fallback = threshold_at_recall(y, s, target_recall)
        info["reason"] = "cost_constraints_infeasible"
        info["recall_only_threshold"] = fallback
        return fallback, info

    info["feasible"] = True
    info.update(best_stats)
    return best_t, info


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


def fp_rate_at_threshold(
    y_true: np.ndarray | list[float],
    scores: np.ndarray | list[float],
    threshold: float,
) -> float | None:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    n_neg = int((y < 0.5).sum())
    if n_neg == 0:
        return None
    fp = float(((y < 0.5) & (s >= float(threshold))).sum())
    return fp / n_neg


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
    min_precision_at_soft: float | None = 0.15,
    max_fp_rate_at_soft: float | None = None,
    apply_floors: bool = False,
) -> dict[str, Any]:
    """
    Suggest soft / hold / deny head thresholds from validation scores.

    Soft ≈ catch ~target_recall under optional precision/FP constraints.
    Hold ≈ slightly stricter recall. Deny ≈ ~50% recall slice.

    Floors are NOT applied silently. If apply_floors=False (default), raw
    suggestions are returned and `floor_would_bind` lists keys that would change.
    """
    abuse_soft, abuse_soft_info = threshold_at_recall_costed(
        abuse_y,
        abuse_scores,
        target_recall=target_recall,
        min_precision=min_precision_at_soft,
        max_fp_rate=max_fp_rate_at_soft,
    )
    fraud_soft, fraud_soft_info = threshold_at_recall_costed(
        fraud_y,
        fraud_scores,
        target_recall=target_recall,
        min_precision=min_precision_at_soft,
        max_fp_rate=max_fp_rate_at_soft,
    )
    abuse_hold = threshold_at_recall(abuse_y, abuse_scores, max(0.85, target_recall - 0.08))
    fraud_hold = threshold_at_recall(fraud_y, fraud_scores, max(0.85, target_recall - 0.08))
    abuse_deny = threshold_at_recall(abuse_y, abuse_scores, 0.50)
    fraud_deny = threshold_at_recall(fraud_y, fraud_scores, 0.50)

    def _or(value: float | None, default: float) -> float:
        return float(default if value is None else value)

    raw = {
        "target_pattern_recall": float(target_recall),
        "min_precision_at_soft": min_precision_at_soft,
        "max_fp_rate_at_soft": max_fp_rate_at_soft,
        "abuse_soft_friction": _or(abuse_soft, 35.0),
        "fraud_soft_friction": _or(fraud_soft, 30.0),
        "abuse_hold_review": _or(abuse_hold, 50.0),
        "fraud_hold_review": _or(fraud_hold, 45.0),
        "abuse_auto_deny": _or(abuse_deny, 75.0),
        "fraud_auto_deny": _or(fraud_deny, 65.0),
    }
    for head in ("abuse", "fraud"):
        soft_k = f"{head}_soft_friction"
        hold_k = f"{head}_hold_review"
        deny_k = f"{head}_auto_deny"
        soft, hold, deny = raw[soft_k], raw[hold_k], raw[deny_k]
        hold = max(hold, soft)
        deny = max(deny, hold)
        raw[soft_k], raw[hold_k], raw[deny_k] = soft, hold, deny

    floors = {
        "abuse_soft_friction": 10.0,
        "fraud_soft_friction": 10.0,
        "abuse_hold_review": 25.0,
        "fraud_hold_review": 25.0,
        "abuse_auto_deny": 50.0,
        "fraud_auto_deny": 50.0,
    }
    floor_would_bind: list[str] = []
    floored = dict(raw)
    for key, floor in floors.items():
        if float(floored[key]) < float(floor):
            floor_would_bind.append(key)
            if apply_floors:
                floored[key] = float(floor)
    if apply_floors:
        for head in ("abuse", "fraud"):
            soft_k = f"{head}_soft_friction"
            hold_k = f"{head}_hold_review"
            deny_k = f"{head}_auto_deny"
            floored[hold_k] = max(float(floored[hold_k]), float(floored[soft_k]))
            floored[deny_k] = max(float(floored[deny_k]), float(floored[hold_k]))

    out = floored if apply_floors else raw
    out["floor_would_bind"] = floor_would_bind
    out["floors_applied"] = bool(apply_floors and floor_would_bind)
    out["cost_feasible"] = bool(abuse_soft_info.get("feasible") and fraud_soft_info.get("feasible"))
    out["abuse_soft_cost_info"] = abuse_soft_info
    out["fraud_soft_cost_info"] = fraud_soft_info
    # Soft floors rewrite the recall/cost contract; hold/deny floors are warnings only.
    soft_floor_bind = [k for k in floor_would_bind if k.endswith("_soft_friction")]
    out["soft_floor_would_bind"] = soft_floor_bind
    out["ok"] = bool(out["cost_feasible"] and not soft_floor_bind)

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
    out["abuse_fp_rate_at_soft"] = fp_rate_at_threshold(
        abuse_y, abuse_scores, out["abuse_soft_friction"]
    )
    out["fraud_fp_rate_at_soft"] = fp_rate_at_threshold(
        fraud_y, fraud_scores, out["fraud_soft_friction"]
    )
    return out
