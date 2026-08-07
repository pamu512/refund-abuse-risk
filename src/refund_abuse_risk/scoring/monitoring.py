"""Calibration / drift / ops monitoring helpers + promote gates."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


def _parse_as_of(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _as_probs(
    scores: np.ndarray,
    *,
    scores_are_0_100: bool,
) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if scores_are_0_100:
        s = s / 100.0
    return np.clip(s, 0.0, 1.0)


def brier_score(
    y_true: np.ndarray | list[float],
    scores_0_100: np.ndarray | list[float],
    *,
    scores_are_0_100: bool = True,
) -> dict[str, Any]:
    """Mean squared error of p̂ vs binary labels (lower is better)."""
    y = np.asarray(y_true, dtype=float)
    p = _as_probs(np.asarray(scores_0_100, dtype=float), scores_are_0_100=scores_are_0_100)
    if len(y) == 0:
        return {"brier": None, "n": 0, "n_positives": 0}
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "n": int(len(y)),
        "n_positives": int(y.sum()),
        "scores_are_0_100": bool(scores_are_0_100),
    }


def expected_calibration_error(
    y_true: np.ndarray | list[float],
    scores_0_100: np.ndarray | list[float],
    *,
    n_bins: int = 10,
    scores_are_0_100: bool = True,
    binning: str = "fixed",
    min_positives: int = 0,
    min_bin_n: int = 0,
) -> dict[str, Any]:
    """
    ECE on scores vs binary labels.

    Default: scores in 0–100 are treated as 100·p̂ (stacker/head display scale).
    Pass ``scores_are_0_100=False`` when scores are already probabilities in [0, 1].

    ``binning``: ``fixed`` (equal-width) or ``adaptive`` (equal-frequency quantiles).
    When ``min_positives`` / ``min_bin_n`` are set, ``usable`` is False if support
    is too thin for a honest promote gate.
    """
    y = np.asarray(y_true, dtype=float)
    s = _as_probs(np.asarray(scores_0_100, dtype=float), scores_are_0_100=scores_are_0_100)
    strategy = str(binning or "fixed").lower().strip()
    if strategy not in {"fixed", "adaptive"}:
        raise ValueError(f"binning must be 'fixed' or 'adaptive', got {binning!r}")

    if len(y) == 0:
        return {
            "ece": None,
            "n": 0,
            "n_bins": n_bins,
            "label": None,
            "binning": strategy,
            "usable": False,
            "usable_reasons": ["empty"],
            "n_positives": 0,
            "scores_are_0_100": bool(scores_are_0_100),
        }

    n = len(y)
    n_pos = int(y.sum())
    rows: list[dict[str, float]] = []
    ece = 0.0

    if strategy == "adaptive":
        qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
        edges = np.unique(np.quantile(s, qs))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0], dtype=float)
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            if i < len(edges) - 2:
                mask = (s >= lo) & (s < hi)
            else:
                mask = (s >= lo) & (s <= hi)
            if not mask.any():
                continue
            conf = float(s[mask].mean())
            acc = float(y[mask].mean())
            w = float(mask.sum()) / n
            ece += w * abs(acc - conf)
            rows.append(
                {"lo": lo, "hi": hi, "n": float(mask.sum()), "acc": acc, "conf": conf}
            )
    else:
        bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
        for i in range(len(bins) - 1):
            lo, hi = float(bins[i]), float(bins[i + 1])
            mask = (s >= lo) & (s < hi) if i < len(bins) - 2 else (s >= lo) & (s <= hi)
            if not mask.any():
                continue
            conf = float(s[mask].mean())
            acc = float(y[mask].mean())
            w = float(mask.sum()) / n
            ece += w * abs(acc - conf)
            rows.append(
                {"lo": lo, "hi": hi, "n": float(mask.sum()), "acc": acc, "conf": conf}
            )

    usable_reasons: list[str] = []
    if int(min_positives) > 0 and n_pos < int(min_positives):
        usable_reasons.append(
            f"n_positives={n_pos} < min_positives={int(min_positives)}"
        )
    if int(min_bin_n) > 0:
        thin_bins = [r for r in rows if float(r["n"]) < float(min_bin_n)]
        if thin_bins:
            usable_reasons.append(
                f"{len(thin_bins)} occupied bin(s) below min_bin_n={int(min_bin_n)}"
            )
        if len(rows) < 2:
            usable_reasons.append("fewer than 2 occupied bins")

    usable = len(usable_reasons) == 0
    return {
        "ece": float(ece),
        "n": int(n),
        "n_bins": int(n_bins),
        "bins": rows,
        "n_positives": n_pos,
        "scores_are_0_100": bool(scores_are_0_100),
        "binning": strategy,
        "usable": bool(usable),
        "usable_reasons": usable_reasons,
        "min_positives": int(min_positives),
        "min_bin_n": int(min_bin_n),
    }


def population_stability_index(
    expected: np.ndarray | list[float],
    actual: np.ndarray | list[float],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """PSI between two score distributions (train vs serve / week-over-week)."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if len(e) == 0 or len(a) == 0:
        return {"psi": None, "n_expected": int(len(e)), "n_actual": int(len(a))}
    qs = np.linspace(0, 100, int(n_bins) + 1)
    edges = np.unique(np.percentile(e, qs))
    if len(edges) < 2:
        return {"psi": 0.0, "n_expected": int(len(e)), "n_actual": int(len(a))}
    e_counts = np.histogram(e, bins=edges)[0].astype(float)
    a_counts = np.histogram(a, bins=edges)[0].astype(float)
    e_pct = (e_counts + 1e-6) / (e_counts.sum() + 1e-6 * len(e_counts))
    a_pct = (a_counts + 1e-6) / (a_counts.sum() + 1e-6 * len(a_counts))
    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return {
        "psi": psi,
        "n_expected": int(len(e)),
        "n_actual": int(len(a)),
        "n_bins": int(len(edges) - 1),
    }


def summarize_ops_metrics(
    *,
    tier_counts: dict[str, int] | None = None,
    effects_dual_run: dict[str, Any] | None = None,
    amounts: np.ndarray | list[float] | None = None,
    final_effects: list[str] | None = None,
) -> dict[str, Any]:
    """
    Offline ops proxies for promote (CS / refund $ / overrides).

    - hold_rate ≈ CS queue pressure from hold_review tier
    - live_override_rate / shadow_override_rate from dual-run effects
    - refund_grant_rate + refund_dollar_per_order from final effects × amounts
    """
    tiers = {str(k).lower(): int(v) for k, v in (tier_counts or {}).items()}
    dual = effects_dual_run or {}
    n = int(dual.get("n") or sum(tiers.values()) or 0)
    hold = int(tiers.get("hold_review", 0))
    live_ov = int(dual.get("live_override_count", 0))
    shadow_ov = int(dual.get("shadow_override_count", 0))
    final_counts = {
        str(k).lower(): int(v) for k, v in (dual.get("final_effect_counts") or {}).items()
    }
    grant_n = int(final_counts.get("refund_auto_grant", 0))

    refund_dollar_per_order = None
    if amounts is not None and final_effects is not None and len(amounts) == len(final_effects) and len(amounts):
        amts = np.asarray(amounts, dtype=float)
        grants = np.asarray([str(e).lower() == "refund_auto_grant" for e in final_effects])
        refund_dollar_per_order = float(amts[grants].sum() / len(amts)) if grants.any() else 0.0
    elif n > 0 and amounts is not None and len(amounts) == n:
        # Fallback: grant_rate × mean amount when per-row effects unavailable.
        mean_amt = float(np.asarray(amounts, dtype=float).mean())
        refund_dollar_per_order = mean_amt * (grant_n / n)

    return {
        "n": n,
        "hold_rate": float(hold / n) if n else None,
        "live_override_rate": float(live_ov / n) if n else None,
        "shadow_override_rate": float(shadow_ov / n) if n else None,
        "refund_grant_rate": float(grant_n / n) if n else None,
        "refund_dollar_per_order": refund_dollar_per_order,
        "live_override_count": live_ov,
        "shadow_override_count": shadow_ov,
        "hold_count": hold,
        "refund_grant_count": grant_n,
    }


def evaluate_monitoring_gates(
    *,
    decision_ece: float | None,
    psi_train_test: float | None,
    monitoring_cfg: dict[str, Any] | None,
    ops_metrics: dict[str, Any] | None = None,
    decision_brier: float | None = None,
    ece_usable: bool | None = None,
    ece_usable_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """
    Honesty promote gate for calibration / drift / ops.

    Fail-closed: if a ceiling is configured and the metric is missing → ok=False.
    When ``max_decision_ece`` is set and ``ece_usable`` is False → ok=False.
    Breach of a present metric → ok=False.
    ``monitoring.ops_snapshot`` (online feed) is merged over offline ops_metrics.
    """
    cfg = monitoring_cfg or {}
    max_ece = cfg.get("max_decision_ece")
    max_brier = cfg.get("max_decision_brier")
    max_psi = cfg.get("max_psi_train_test")
    reasons: list[str] = []
    ok = True
    if max_ece is not None:
        if ece_usable is False:
            ok = False
            detail = ",".join(ece_usable_reasons or ["unusable"])
            reasons.append(f"decision_ece unusable with max_decision_ece configured ({detail})")
        elif decision_ece is None:
            ok = False
            reasons.append("decision_ece missing with max_decision_ece configured")
        elif float(decision_ece) > float(max_ece):
            ok = False
            reasons.append(
                f"decision_ece={float(decision_ece):.4f} > max_decision_ece={float(max_ece)}"
            )
    if max_brier is not None:
        if decision_brier is None:
            ok = False
            reasons.append("decision_brier missing with max_decision_brier configured")
        elif float(decision_brier) > float(max_brier):
            ok = False
            reasons.append(
                f"decision_brier={float(decision_brier):.4f} > "
                f"max_decision_brier={float(max_brier)}"
            )
    if max_psi is not None:
        if psi_train_test is None:
            ok = False
            reasons.append("psi_train_test missing with max_psi_train_test configured")
        elif float(psi_train_test) > float(max_psi):
            ok = False
            reasons.append(
                f"psi_train_test={float(psi_train_test):.4f} > max_psi_train_test={float(max_psi)}"
            )

    ops = dict(ops_metrics or {})
    snapshot = cfg.get("ops_snapshot") or {}
    if isinstance(snapshot, dict):
        for k, v in snapshot.items():
            if v is not None:
                ops[str(k)] = v

    ops_checks = (
        ("hold_rate", "max_hold_rate"),
        ("live_override_rate", "max_live_override_rate"),
        ("refund_grant_rate", "max_refund_grant_rate"),
        ("refund_dollar_per_order", "max_refund_dollar_per_order"),
        ("cs_queue_depth", "max_cs_queue_depth"),
        ("shadow_override_rate", "max_shadow_override_rate"),
    )
    for metric_key, cfg_key in ops_checks:
        ceiling = cfg.get(cfg_key)
        if ceiling is None:
            continue
        value = ops.get(metric_key)
        if value is None:
            ok = False
            reasons.append(f"{metric_key} missing with {cfg_key} configured")
            continue
        if float(value) > float(ceiling):
            ok = False
            reasons.append(f"{metric_key}={float(value):.4f} > {cfg_key}={float(ceiling)}")

    max_age_h = cfg.get("max_ops_snapshot_age_hours")
    as_of = ops.get("as_of")
    if max_age_h is not None:
        age_hours = None
        parsed = _parse_as_of(as_of)
        if parsed is None:
            ok = False
            reasons.append(
                f"ops_snapshot.as_of missing/unparseable with "
                f"max_ops_snapshot_age_hours={float(max_age_h)}"
            )
        else:
            now = datetime.now(timezone.utc)
            age_hours = (now - parsed).total_seconds() / 3600.0
            if age_hours > float(max_age_h) + 1e-9:
                ok = False
                reasons.append(
                    f"ops_snapshot_age_hours={age_hours:.2f} > "
                    f"max_ops_snapshot_age_hours={float(max_age_h)}"
                )
        ops = dict(ops)
        if age_hours is not None:
            ops["age_hours"] = float(age_hours)

    return {
        "ok": ok,
        "max_decision_ece": float(max_ece) if max_ece is not None else None,
        "max_decision_brier": float(max_brier) if max_brier is not None else None,
        "max_psi_train_test": float(max_psi) if max_psi is not None else None,
        "decision_ece": float(decision_ece) if decision_ece is not None else None,
        "decision_brier": float(decision_brier) if decision_brier is not None else None,
        "ece_usable": ece_usable,
        "psi_train_test": float(psi_train_test) if psi_train_test is not None else None,
        "ops": ops,
        "reasons": reasons,
    }
