"""Joint decision score (stacker) + single-ladder tiers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from refund_abuse_risk.schemas.models import SuggestedTier

_THRESHOLD_KEYS = ("soft_friction", "hold_review", "auto_deny")


@dataclass
class DecisionStacker:
    """
    Calibrated joint score from abuse/fraud head scores.

    Prefer fitting on OOF head scores (see TwoHeadModel.fit); falls back to
    in-sample when CV is infeasible (tiny / single-class folds).
    Optional market×vertical one-hots (slice-aware stacker).
    """

    model: Any = None
    model_version: str = "0.1.0"
    fit_mode: str = "unset"  # oof | in_sample | unset
    market_levels: list[str] | None = None
    vertical_levels: list[str] | None = None
    use_slices: bool = False

    def _slice_matrix(
        self,
        n: int,
        markets: np.ndarray | None,
        verticals: np.ndarray | None,
    ) -> np.ndarray | None:
        if not self.use_slices or not self.market_levels or not self.vertical_levels:
            return None
        m_levels = self.market_levels
        v_levels = self.vertical_levels
        m = (
            _norm_slice_labels(markets, n, upper=True)
            if markets is not None
            else np.array(["ALL"] * n)
        )
        v = (
            _norm_slice_labels(verticals, n, upper=False)
            if verticals is not None
            else np.array(["all"] * n)
        )
        cols = []
        for level in m_levels:
            cols.append((m == level).astype(float))
        cols.append((~np.isin(m, m_levels)).astype(float))  # market other
        for level in v_levels:
            cols.append((v == level).astype(float))
        cols.append((~np.isin(v, v_levels)).astype(float))  # vertical other
        return np.column_stack(cols)

    def _design_matrix(
        self,
        abuse_scores: np.ndarray,
        fraud_scores: np.ndarray,
        markets: np.ndarray | None = None,
        verticals: np.ndarray | None = None,
        *,
        scores_are_0_100: bool = True,
    ) -> np.ndarray:
        a = np.asarray(abuse_scores, dtype=float)
        f = np.asarray(fraud_scores, dtype=float)
        if scores_are_0_100:
            a = a / 100.0
            f = f / 100.0
        base = np.column_stack([a, f, np.maximum(a, f)])
        slice_x = self._slice_matrix(len(a), markets, verticals)
        if slice_x is None:
            return base
        return np.column_stack([base, slice_x])

    def fit(
        self,
        abuse_scores: np.ndarray,
        fraud_scores: np.ndarray,
        pattern_y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        fit_mode: str = "in_sample",
        markets: np.ndarray | list[str] | None = None,
        verticals: np.ndarray | list[str] | None = None,
    ) -> DecisionStacker:
        y = np.asarray(pattern_y, dtype=int)
        classes = np.unique(y)
        self.fit_mode = str(fit_mode)
        m_arr = np.asarray(markets) if markets is not None else None
        v_arr = np.asarray(verticals) if verticals is not None else None
        if m_arr is not None and v_arr is not None and len(m_arr) == len(y):
            m_norm = _norm_slice_labels(m_arr, len(y), upper=True)
            v_norm = _norm_slice_labels(v_arr, len(y), upper=False)
            # Cap cardinality so tiny slices don't explode features.
            self.market_levels = sorted(set(m_norm.tolist()))[:12]
            self.vertical_levels = sorted(set(v_norm.tolist()))[:8]
            self.use_slices = True
        else:
            self.market_levels = None
            self.vertical_levels = None
            self.use_slices = False
        if len(classes) < 2:
            self.model = None
            return self
        x = self._design_matrix(abuse_scores, fraud_scores, m_arr, v_arr, scores_are_0_100=True)
        clf = LogisticRegression(C=1.0, max_iter=300, random_state=42)
        try:
            clf.fit(x, y, sample_weight=sample_weight)
        except TypeError:
            clf.fit(x, y)
        self.model = clf
        return self

    def predict_scores(
        self,
        abuse_scores: np.ndarray,
        fraud_scores: np.ndarray,
        markets: np.ndarray | list[str] | None = None,
        verticals: np.ndarray | list[str] | None = None,
    ) -> np.ndarray:
        a = np.asarray(abuse_scores, dtype=float)
        f = np.asarray(fraud_scores, dtype=float)
        if self.model is None:
            return np.clip(np.maximum(a, f), 0.0, 100.0)
        m_arr = np.asarray(markets) if markets is not None else None
        v_arr = np.asarray(verticals) if verticals is not None else None
        x = self._design_matrix(a, f, m_arr, v_arr, scores_are_0_100=True)
        proba = self.model.predict_proba(x)
        classes = list(self.model.classes_)
        if 1 in classes:
            p = proba[:, classes.index(1)]
        else:
            p = np.zeros(len(a), dtype=float)
        return np.clip(p * 100.0, 0.0, 100.0)

    def score_one(
        self,
        abuse_score: float,
        fraud_score: float,
        market: str = "",
        vertical: str = "",
    ) -> float:
        return float(
            self.predict_scores(
                np.array([abuse_score]),
                np.array([fraud_score]),
                markets=np.array([market]),
                verticals=np.array([vertical]),
            )[0]
        )


def _norm_slice_labels(values: np.ndarray, n: int, *, upper: bool) -> np.ndarray:
    out = np.asarray(
        [str(x) if x is not None else "" for x in np.asarray(values).reshape(-1)],
        dtype=object,
    )
    if len(out) != n:
        out = np.array([""] * n, dtype=object)
    if upper:
        return np.array([s.upper() if s else "ALL" for s in out], dtype=object)
    return np.array([s.lower() if s else "all" for s in out], dtype=object)


def resolve_decision_thresholds(
    operating_point: dict[str, Any],
    *,
    market: str = "",
    vertical: str = "",
) -> dict[str, Any]:
    """Merge global decision_thresholds with market×vertical overlay (first match)."""
    base = dict(operating_point.get("decision_thresholds") or {})
    mkt = str(market or "").upper()
    vert = str(vertical or "").lower()
    for overlay in operating_point.get("decision_threshold_overlays") or []:
        ov_m = str(overlay.get("market", "") or "").upper()
        ov_v = str(overlay.get("vertical", "") or "").lower()
        if ov_m and ov_m != mkt:
            continue
        if ov_v and ov_v != vert:
            continue
        for key in _THRESHOLD_KEYS:
            if key in overlay and overlay[key] is not None:
                base[key] = float(overlay[key])
        break
    return base


def operating_point_for_slice(
    operating_point: dict[str, Any],
    *,
    market: str = "",
    vertical: str = "",
) -> dict[str, Any]:
    """Copy OP with decision_thresholds resolved for market×vertical."""
    op = deepcopy(operating_point)
    op["decision_thresholds"] = resolve_decision_thresholds(
        operating_point, market=market, vertical=vertical
    )
    return op


def tier_from_decision_score(
    decision_score: float,
    operating_point: dict[str, Any],
    *,
    market: str = "",
    vertical: str = "",
) -> SuggestedTier:
    """Ladder on joint decision_score (heads are evidence only)."""
    thr = resolve_decision_thresholds(operating_point, market=market, vertical=vertical)
    score = float(decision_score)
    if score >= float(thr.get("auto_deny", 75)):
        return SuggestedTier.AUTO_DENY
    if score >= float(thr.get("hold_review", 50)):
        return SuggestedTier.HOLD_REVIEW
    if score >= float(thr.get("soft_friction", 35)):
        return SuggestedTier.SOFT_FRICTION
    return SuggestedTier.AUTO_APPROVE


def recommend_decision_thresholds(
    pattern_y: np.ndarray | list[float],
    decision_scores: np.ndarray | list[float],
    *,
    target_recall: float = 0.98,
    min_precision_at_soft: float | None = 0.15,
    max_fp_rate_at_soft: float | None = None,
    amounts: np.ndarray | list[float] | None = None,
    max_fp_refund_dollars_mean: float | None = None,
    apply_floors: bool = False,
) -> dict[str, Any]:
    """Costed soft/hold/deny on the joint decision score (precision / FP / $)."""
    from refund_abuse_risk.scoring.thresholds import (
        fp_rate_at_threshold,
        precision_at_threshold,
        recall_at_threshold,
        threshold_at_recall,
        threshold_at_recall_costed,
    )

    soft, soft_info = threshold_at_recall_costed(
        pattern_y,
        decision_scores,
        target_recall=target_recall,
        min_precision=min_precision_at_soft,
        max_fp_rate=max_fp_rate_at_soft,
        amounts=amounts,
        max_fp_refund_dollars_mean=max_fp_refund_dollars_mean,
    )
    hold = threshold_at_recall(pattern_y, decision_scores, max(0.85, target_recall - 0.08))
    deny = threshold_at_recall(pattern_y, decision_scores, 0.50)

    def _or(v: float | None, default: float) -> float:
        return float(default if v is None else v)

    raw = {
        "soft_friction": _or(soft, 35.0),
        "hold_review": _or(hold, 50.0),
        "auto_deny": _or(deny, 75.0),
        "target_pattern_recall": float(target_recall),
        "min_precision_at_soft": min_precision_at_soft,
    }
    raw["hold_review"] = max(raw["hold_review"], raw["soft_friction"])
    raw["auto_deny"] = max(raw["auto_deny"], raw["hold_review"])

    floors = {"soft_friction": 10.0, "hold_review": 25.0, "auto_deny": 50.0}
    floor_would_bind = [k for k, fl in floors.items() if float(raw[k]) < fl]
    out = dict(raw)
    if apply_floors:
        for k, fl in floors.items():
            out[k] = max(float(out[k]), fl)
        out["hold_review"] = max(out["hold_review"], out["soft_friction"])
        out["auto_deny"] = max(out["auto_deny"], out["hold_review"])

    soft_floor = [k for k in floor_would_bind if k == "soft_friction"]
    out["floor_would_bind"] = floor_would_bind
    out["soft_floor_would_bind"] = soft_floor
    out["cost_feasible"] = bool(soft_info.get("feasible"))
    out["soft_cost_info"] = soft_info
    out["ok"] = bool(out["cost_feasible"] and not soft_floor)
    out["recall_at_soft"] = recall_at_threshold(pattern_y, decision_scores, out["soft_friction"])
    out["precision_at_soft"] = precision_at_threshold(
        pattern_y, decision_scores, out["soft_friction"]
    )
    out["fp_rate_at_soft"] = fp_rate_at_threshold(
        pattern_y, decision_scores, out["soft_friction"]
    )
    return out


def recommend_decision_thresholds_by_slice(
    frame_scores: Any,
    *,
    market_col: str = "market",
    vertical_col: str = "vertical",
    label_col: str = "pattern_y",
    score_col: str = "decision_score",
    amount_col: str = "amount",
    target_recall: float = 0.98,
    min_precision_at_soft: float | None = 0.15,
    max_fp_rate_at_soft: float | None = None,
    max_fp_refund_dollars_mean: float | None = None,
    min_slice_n: int = 20,
    min_slice_positives: int = 3,
) -> dict[str, Any]:
    """
    Costed ladders per market×vertical.

    Thin slices return ok=False and are skipped for overlay promote.
    """
    import pandas as pd

    if not isinstance(frame_scores, pd.DataFrame):
        raise TypeError("frame_scores must be a DataFrame")
    out: dict[str, Any] = {}
    for (market, vertical), grp in frame_scores.groupby(
        [frame_scores[market_col].astype(str), frame_scores[vertical_col].astype(str)],
        sort=True,
    ):
        y = grp[label_col].astype(int).to_numpy()
        s = grp[score_col].astype(float).to_numpy()
        amts = (
            grp[amount_col].astype(float).to_numpy()
            if amount_col in grp.columns
            else None
        )
        key = f"{market}|{vertical}"
        rec = recommend_decision_thresholds(
            y,
            s,
            target_recall=target_recall,
            min_precision_at_soft=min_precision_at_soft,
            max_fp_rate_at_soft=max_fp_rate_at_soft,
            amounts=amts,
            max_fp_refund_dollars_mean=max_fp_refund_dollars_mean,
            apply_floors=False,
        )
        thin = len(grp) < int(min_slice_n) or int(y.sum()) < int(min_slice_positives)
        if thin:
            rec = dict(rec)
            rec["ok"] = False
            rec["thin_slice"] = True
            rec["promote_eligible"] = False
        else:
            rec["thin_slice"] = False
            # Overlay promote only when costed ok and floors clear.
            rec["promote_eligible"] = bool(rec.get("ok")) and not bool(rec.get("thin_slice"))
        rec["n"] = int(len(grp))
        rec["n_positives"] = int(y.sum())
        rec["market"] = str(market)
        rec["vertical"] = str(vertical)
        out[key] = rec
    return out


def recommended_overlays_from_slices(recommended_by_slice: dict[str, Any]) -> list[dict[str, Any]]:
    """Promote-eligible costed ladders → operating_point overlay shape."""
    overlays: list[dict[str, Any]] = []
    for rec in recommended_by_slice.values():
        if not rec.get("promote_eligible", rec.get("ok")) or rec.get("thin_slice"):
            continue
        overlays.append(
            {
                "market": rec["market"],
                "vertical": rec["vertical"],
                "soft_friction": round(float(rec["soft_friction"]), 2),
                "hold_review": round(float(rec["hold_review"]), 2),
                "auto_deny": round(float(rec["auto_deny"]), 2),
            }
        )
    return overlays

