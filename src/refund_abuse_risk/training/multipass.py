"""Multi-pass training: unsupervised discovery → supervised two-head fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from refund_abuse_risk.config import load_bipartite_anomaly, load_label_weights
from refund_abuse_risk.graph.bipartite import score_uv_bipartite
from refund_abuse_risk.labels.discovery_controls import (
    cap_discovery_labels,
    discovery_config,
    mint_uv_asof_daily,
)
from refund_abuse_risk.model.two_head import TwoHeadModel, apply_proxy_fraud_labels

# Feature subspace for unsupervised anomaly (fraud/collusion-shaped).
_UNSUPERVISED_FEATURE_COLS = (
    "device_cluster_size",
    "accounts_per_device",
    "uvd_refund_lift",
    "uvd_refund_share",
    "uvd_cooccur",
    "device_risk_score",
    "user_refund_rate_30d",
    "uv_edge_anomaly",
    "user_bipartite_anomaly",
    "uv_edge_lift",
    "uv_edge_elevated",
    "uv_mo_possible_collusion",
    "uv_mo_user_scatter",
    "is_cloned_app",
    "is_gps_spoof",
    "customer_courier_same_device",
)


@dataclass
class PassReport:
    pass_idx: int
    unsupervised: dict[str, Any] = field(default_factory=dict)
    supervised: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_idx,
            "unsupervised": self.unsupervised,
            "supervised": self.supervised,
        }


def _ensure_label_cols(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col, default in (
        ("abuse_label", 0),
        ("abuse_label_weak", 0),
        ("fraud_label", 0),
        ("fraud_label_source", ""),
        ("strong_fraud_label", 0),
        ("weak_policy_negative", 0),
    ):
        if col not in out.columns:
            out[col] = default
    out["fraud_label_source"] = (
        out["fraud_label_source"].astype("string").fillna("").astype(str).replace({"nan": "", "<NA>": ""})
    )
    return out


def _protected_mask(frame: pd.DataFrame) -> pd.Series:
    src = frame["fraud_label_source"].astype(str).str.lower()
    return src.eq("proven") | (frame["strong_fraud_label"].astype(float) >= 1)


def mint_uv_batch_discovery(
    frame: pd.DataFrame,
    history: pd.DataFrame | None,
    cfg: dict[str, Any] | None = None,
    *,
    edges: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Unsupervised UV bipartite on full history → weak discovery labels.

    Large-scale training path (batch, not per-order as-of). Never overwrites proven.
    Pass precomputed ``edges`` to avoid re-scoring history every multipass cycle.
    """
    out = _ensure_label_cols(frame)
    stats = {"uv_elevated_edges": 0, "uv_minted": 0, "enabled": True}
    if edges is None:
        if history is None or history.empty:
            stats["enabled"] = False
            return out, stats
        edges, _nodes = score_uv_bipartite(history, cfg)
    if edges is None or edges.empty or "elevated" not in edges.columns:
        return out, stats
    elev = edges[edges["elevated"]].copy()
    stats["uv_elevated_edges"] = int(len(elev))
    if elev.empty:
        return out, stats
    elev = elev.copy()
    elev["_uv"] = 1
    elev_key = elev.assign(
        _market=elev["market"].astype(str),
        _vertical=elev["vertical"].astype(str),
        _user_id=elev["user_id"].astype(str),
        _vendor_id=elev["vendor_id"].astype(str),
    )[["_market", "_vertical", "_user_id", "_vendor_id", "_uv"]].drop_duplicates()
    left = out.copy()
    left["_market"] = (
        left["market"].astype(str).str.upper().replace({"": "ALL"})
        if "market" in left.columns
        else "ALL"
    )
    left["_vertical"] = (
        left["vertical"].astype(str).str.lower().replace({"": "all"})
        if "vertical" in left.columns
        else "all"
    )
    left["_user_id"] = left["user_id"].astype(str) if "user_id" in left.columns else ""
    left["_vendor_id"] = left["vendor_id"].astype(str) if "vendor_id" in left.columns else ""
    merged = left.merge(elev_key, on=["_market", "_vertical", "_user_id", "_vendor_id"], how="left")
    protected = _protected_mask(out).to_numpy()
    hit = merged["_uv"].fillna(0).to_numpy() >= 1
    assign = hit & ~protected
    out.loc[assign, "abuse_label"] = 1
    out.loc[assign, "abuse_label_weak"] = 1
    out.loc[assign, "fraud_label"] = 1
    # Don't overwrite proven source (already excluded via protected).
    out.loc[assign, "fraud_label_source"] = "discovery"
    stats["uv_minted"] = int(assign.sum())
    return out, stats


def mint_isolation_forest_discovery(
    frame: pd.DataFrame,
    *,
    contamination: float = 0.05,
    random_state: int = 42,
    score_percentile: float = 95.0,
    max_mint_fraction: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Unsupervised IsolationForest on collusion/device subspace → discovery labels."""
    out = _ensure_label_cols(frame)
    cols = [c for c in _UNSUPERVISED_FEATURE_COLS if c in out.columns]
    stats: dict[str, Any] = {
        "iforest_features": cols,
        "iforest_minted": 0,
        "contamination": float(contamination),
        "max_mint_fraction": float(max_mint_fraction),
    }
    if len(cols) < 3 or len(out) < 50:
        stats["skipped"] = "insufficient_features_or_rows"
        return out, stats
    x = out[cols].astype(float).fillna(0.0).to_numpy()
    clf = IsolationForest(
        n_estimators=100,
        contamination=float(contamination),
        random_state=int(random_state),
        n_jobs=-1,
    )
    clf.fit(x)
    anom = -clf.decision_function(x)
    out["unsupervised_anomaly_score"] = anom
    thr = float(np.percentile(anom, float(score_percentile)))
    flagged = anom >= thr
    protected = _protected_mask(out).to_numpy()
    src = out["fraud_label_source"].astype(str).str.lower().to_numpy()
    assign = flagged & ~protected & (src != "proxy") & (src != "proven")
    # Hard cap IForest mint volume before global discovery cap.
    cap_n = max(0, int(len(out) * float(max_mint_fraction)))
    if assign.sum() > cap_n > 0:
        cand = np.where(assign)[0]
        order = cand[np.argsort(-anom[cand])][:cap_n]
        new_assign = np.zeros(len(out), dtype=bool)
        new_assign[order] = True
        assign = new_assign
    elif cap_n == 0:
        assign = np.zeros(len(out), dtype=bool)
    out.loc[assign, "abuse_label"] = 1
    out.loc[assign, "abuse_label_weak"] = 1
    out.loc[assign, "fraud_label"] = 1
    out.loc[assign, "fraud_label_source"] = "discovery"
    stats["iforest_minted"] = int(assign.sum())
    stats["anomaly_threshold"] = thr
    return out, stats


def mint_self_train_discovery(
    frame: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    min_decision_score: float = 80.0,
    max_fraction: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Semi-supervised: high decision_score unlabeled rows → weak discovery (capped)."""
    out = _ensure_label_cols(frame)
    stats = {
        "self_train_minted": 0,
        "min_decision_score": float(min_decision_score),
        "max_fraction": float(max_fraction),
    }
    if "decision_score" not in scored.columns:
        return out, stats
    ds = scored["decision_score"].astype(float).to_numpy()
    protected = _protected_mask(out).to_numpy()
    unlabeled = (
        (out["fraud_label"].astype(int).to_numpy() < 1)
        & (out["abuse_label"].astype(int).to_numpy() < 1)
        & ~protected
    )
    cand = unlabeled & (ds >= float(min_decision_score))
    idxs = np.where(cand)[0]
    if len(idxs) == 0:
        return out, stats
    cap = max(0, int(len(out) * float(max_fraction)))
    if cap <= 0:
        return out, stats
    order = idxs[np.argsort(-ds[idxs])][:cap]
    assign = np.zeros(len(out), dtype=bool)
    assign[order] = True
    out.loc[assign, "abuse_label"] = 1
    out.loc[assign, "abuse_label_weak"] = 1
    out.loc[assign, "fraud_label"] = 1
    out.loc[assign, "fraud_label_source"] = "discovery"
    stats["self_train_minted"] = int(assign.sum())
    return out, stats


def run_unsupervised_pass(
    frame: pd.DataFrame,
    *,
    history: pd.DataFrame | None = None,
    bipartite_cfg: dict[str, Any] | None = None,
    uv_edges: pd.DataFrame | None = None,
    label_weights: dict[str, Any] | None = None,
    prev_scored: pd.DataFrame | None = None,
    pass_idx: int = 1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """UV discovery (as-of daily by default) + IsolationForest + capped discovery."""
    bipartite_cfg = bipartite_cfg if bipartite_cfg is not None else load_bipartite_anomaly()
    dcfg = discovery_config(label_weights)
    mode = str(dcfg.get("mint_mode", "asof_daily")).lower()
    if mode == "batch":
        out, uv_stats = mint_uv_batch_discovery(
            frame, history, bipartite_cfg, edges=uv_edges
        )
        uv_stats = {**uv_stats, "mode": "batch"}
    else:
        out, uv_stats = mint_uv_asof_daily(frame, history, bipartite_cfg)

    contam = max(0.02, 0.05 - 0.002 * (pass_idx - 1))
    out, if_stats = mint_isolation_forest_discovery(
        out,
        contamination=contam,
        random_state=random_state + pass_idx,
        max_mint_fraction=float(dcfg["iforest_max_fraction"]),
    )
    self_stats: dict[str, Any] = {"self_train_minted": 0}
    if prev_scored is not None and pass_idx >= 2:
        out, self_stats = mint_self_train_discovery(
            out,
            prev_scored,
            max_fraction=float(dcfg["self_train_max_fraction"]),
        )
    out, cap_stats = cap_discovery_labels(
        out,
        max_fraction_of_rows=float(dcfg["max_fraction_of_rows"]),
        max_fraction_of_fraud_positives=float(dcfg["max_fraction_of_fraud_positives"]),
        random_state=random_state + pass_idx,
    )
    return out, {
        "uv": uv_stats,
        "isolation_forest": if_stats,
        "self_train": self_stats,
        "discovery_cap": cap_stats,
        "discovery_rate": float(
            (out["fraud_label_source"].astype(str).str.lower() == "discovery").mean()
        ),
    }


def run_supervised_pass(
    frame: pd.DataFrame,
    *,
    label_weights: dict[str, Any] | None = None,
    model_version: str = "0.6.2",
) -> tuple[TwoHeadModel, pd.DataFrame, dict[str, Any]]:
    """Proxy mint + two-head + OOF stacker fit."""
    label_weights = label_weights or load_label_weights()
    labeled = apply_proxy_fraud_labels(frame, label_weights)
    model = TwoHeadModel(model_version=model_version)
    model.fit(labeled, label_weights)
    scored = model.predict_proba(labeled)
    abuse_y = scored["abuse_label"].astype(int)
    fraud_y = scored["fraud_label"].astype(int)
    pattern = ((abuse_y >= 1) | (fraud_y >= 1)).astype(int)
    metrics = {
        "n": int(len(scored)),
        "stacker_fit_mode": getattr(model.decision_stacker, "fit_mode", None),
        "abuse_positive_rate": float(abuse_y.mean()) if len(abuse_y) else 0.0,
        "fraud_positive_rate": float(fraud_y.mean()) if len(fraud_y) else 0.0,
        "pattern_rate": float(pattern.mean()) if len(pattern) else 0.0,
        "decision_score_mean": float(scored["decision_score"].mean()),
        "decision_score_p90": float(scored["decision_score"].quantile(0.90)),
        "fraud_source_counts": scored["fraud_label_source"]
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict(),
    }
    return model, scored, metrics


def train_multipass(
    frame: pd.DataFrame,
    *,
    history: pd.DataFrame | None = None,
    n_passes: int = 10,
    label_weights: dict[str, Any] | None = None,
    bipartite_cfg: dict[str, Any] | None = None,
    model_version: str = "0.6.2",
    random_state: int = 42,
    early_stop_delta: float = 0.05,
    early_stop_patience: int = 2,
) -> tuple[TwoHeadModel, list[PassReport], pd.DataFrame]:
    """
    Alternate unsupervised label minting and supervised fitting for ``n_passes``.

    Stops early when ``decision_score_mean`` moves by less than ``early_stop_delta``
    for ``early_stop_patience`` consecutive passes (P1 plateau guard).

    Returns final model, per-pass reports, and last labeled frame.
    """
    if n_passes < 1:
        raise ValueError("n_passes must be >= 1")
    label_weights = label_weights or load_label_weights()
    bipartite_cfg = bipartite_cfg if bipartite_cfg is not None else load_bipartite_anomaly()
    working = _ensure_label_cols(frame)
    reports: list[PassReport] = []
    model: TwoHeadModel | None = None
    scored: pd.DataFrame | None = None
    dcfg = discovery_config(label_weights)
    uv_edges = None
    # Batch mode only: precompute full-history edges (leaky — opt-in via config).
    if (
        str(dcfg.get("mint_mode", "asof_daily")).lower() == "batch"
        and history is not None
        and not history.empty
    ):
        uv_edges, _nodes = score_uv_bipartite(history, bipartite_cfg)

    prev_mean: float | None = None
    stable = 0
    for p in range(1, int(n_passes) + 1):
        working, u_stats = run_unsupervised_pass(
            working,
            history=history,
            bipartite_cfg=bipartite_cfg,
            uv_edges=uv_edges,
            label_weights=label_weights,
            prev_scored=scored,
            pass_idx=p,
            random_state=random_state,
        )
        model, scored, s_stats = run_supervised_pass(
            working,
            label_weights=label_weights,
            model_version=f"{model_version}+p{p}",
        )
        mean = float(s_stats.get("decision_score_mean", 0.0))
        if prev_mean is not None and abs(mean - prev_mean) < float(early_stop_delta):
            stable += 1
        else:
            stable = 0
        s_stats["early_stop_stable"] = int(stable)
        reports.append(PassReport(pass_idx=p, unsupervised=u_stats, supervised=s_stats))
        if stable >= int(early_stop_patience) and p < int(n_passes):
            s_stats["early_stopped"] = True
            break
        prev_mean = mean

    assert model is not None
    return model, reports, working
