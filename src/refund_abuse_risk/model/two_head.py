from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from refund_abuse_risk.features.builders import (
    ABUSE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FRAUD_FEATURE_COLUMNS,
    abuse_model_feature_columns,
    fraud_model_feature_columns,
)

if TYPE_CHECKING:
    from refund_abuse_risk.scoring.decision import DecisionStacker


def apply_proxy_fraud_labels(
    frame: pd.DataFrame,
    label_weights: dict[str, Any],
) -> pd.DataFrame:
    """Mark high-confidence proxy fraud positives; preserve proven labels.

    Proxy uses graph concentration + device farm signals. Requires min support
    so a single refund cannot mint a fraud label.
    """
    out = frame.copy()
    rules = label_weights.get("proxy_rules") or {}
    min_cluster = float(rules.get("min_device_cluster_size", 4))
    min_accounts = float(rules.get("min_accounts_per_device", 3))
    min_lift = float(rules.get("min_uvd_refund_lift", 1.5))
    min_share = float(rules.get("min_uvd_refund_share", 0.5))
    min_rate = float(rules.get("min_coordinated_refund_rate", 0.35))
    min_orders = float(rules.get("min_user_orders_30d", 5))
    min_cooccur = float(rules.get("min_uvd_cooccur", 3))
    min_device_risk = float(rules.get("min_device_risk_score", 70))
    integrity_min_flags = int(rules.get("integrity_min_flags", 2))

    if "fraud_label" not in out.columns:
        out["fraud_label"] = 0
    if "fraud_label_source" not in out.columns:
        out["fraud_label_source"] = ""
    if "strong_fraud_label" not in out.columns:
        out["strong_fraud_label"] = 0
    for col in (
        "device_risk_score",
        "is_emulator",
        "is_cloned_app",
        "is_gps_spoof",
        "customer_courier_same_device",
    ):
        if col not in out.columns:
            out[col] = 0.0

    out["fraud_label"] = pd.to_numeric(out["fraud_label"], errors="coerce").fillna(0).astype(int)
    out["fraud_label_source"] = (
        out["fraud_label_source"].astype("string").fillna("").astype(str).replace({"nan": "", "<NA>": ""})
    )
    proven = out["fraud_label_source"].str.lower().eq("proven") | (
        out["strong_fraud_label"].astype(float) >= 1
    )
    discovery = out["fraud_label_source"].str.lower().eq("discovery")
    farm_proxy = (
        (out["device_cluster_size"].astype(float) >= min_cluster)
        & (out["accounts_per_device"].astype(float) >= min_accounts)
        & (out["uvd_refund_lift"].astype(float) >= min_lift)
        & (out["uvd_refund_share"].astype(float) >= min_share)
        & (out["uvd_cooccur"].astype(float) >= min_cooccur)
        & (out["user_refund_rate_30d"].astype(float) >= min_rate)
        & (out["user_orders_30d"].astype(float) >= min_orders)
    )
    integrity_flags = (
        out["is_emulator"].astype(float).clip(0, 1)
        + out["is_cloned_app"].astype(float).clip(0, 1)
        + out["is_gps_spoof"].astype(float).clip(0, 1)
        + out["customer_courier_same_device"].astype(float).clip(0, 1)
    )
    integrity_proxy = (
        (
            (out["device_risk_score"].astype(float) >= min_device_risk)
            | (integrity_flags >= float(integrity_min_flags))
        )
        & (out["uvd_cooccur"].astype(float) >= min_cooccur)
        & (out["user_orders_30d"].astype(float) >= min_orders)
        & (out["user_refund_rate_30d"].astype(float) >= min_rate)
    )
    proxy_mask = farm_proxy | integrity_proxy
    # Proven and discovery labels are protected from proxy overwrite.
    assign_proxy = proxy_mask & ~proven & ~discovery & (out["fraud_label"] < 1)
    out.loc[assign_proxy, "fraud_label"] = 1
    out.loc[assign_proxy, "fraud_label_source"] = "proxy"
    out.loc[proven, "fraud_label"] = 1
    out.loc[proven, "fraud_label_source"] = "proven"
    out.loc[discovery & ~proven, "fraud_label_source"] = "discovery"
    return out


def balance_class_weights(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Scale so total positive weight equals total negative weight (within-class ratios kept)."""
    y = np.asarray(y, dtype=int)
    w = np.asarray(weights, dtype=float).copy()
    pos = y >= 1
    neg = ~pos
    sum_pos = float(w[pos].sum()) if pos.any() else 0.0
    sum_neg = float(w[neg].sum()) if neg.any() else 0.0
    if sum_pos <= 0.0 or sum_neg <= 0.0:
        return w
    w[pos] *= sum_neg / sum_pos
    return w


def sample_weights_for_frame(
    frame: pd.DataFrame,
    label_weights: dict[str, Any],
    *,
    head: str,
) -> np.ndarray:
    proxy_w = float(label_weights.get("fraud_proxy_weight", 0.4))
    discovery_w = float(label_weights.get("fraud_discovery_weight", 0.25))
    proven_w = float(label_weights.get("fraud_proven_weight", 1.0))
    abuse_pos = float(label_weights.get("abuse_positive_weight", 1.0))
    abuse_weak = float(label_weights.get("abuse_weak_positive_weight", 0.6))
    weak_neg = float(label_weights.get("weak_policy_negative_weight", 0.3))
    clean_neg = float(label_weights.get("clean_negative_weight", 1.0))

    n = len(frame)
    weights = np.ones(n, dtype=float)
    if head == "fraud":
        source = frame.get("fraud_label_source", pd.Series([""] * n)).astype(str).str.lower().to_numpy()
        y = frame["fraud_label"].astype(float).to_numpy()
        weak_policy_neg = (
            frame.get("weak_policy_negative", pd.Series(np.zeros(n))).astype(float).to_numpy()
        )
        for i in range(n):
            if y[i] >= 1:
                if source[i] == "proven":
                    weights[i] = proven_w
                elif source[i] == "discovery":
                    weights[i] = discovery_w
                else:
                    weights[i] = proxy_w
            elif weak_policy_neg[i] >= 1:
                weights[i] = weak_neg
            else:
                weights[i] = clean_neg
        return weights

    y = frame["abuse_label"].astype(float).to_numpy()
    weak = frame.get("abuse_label_weak", pd.Series(np.zeros(n))).astype(float).to_numpy()
    weak_policy_neg = frame.get("weak_policy_negative", pd.Series(np.zeros(n))).astype(float).to_numpy()
    for i in range(n):
        if y[i] >= 1:
            weights[i] = abuse_weak if weak[i] >= 1 else abuse_pos
        elif weak_policy_neg[i] >= 1:
            weights[i] = weak_neg
        else:
            weights[i] = clean_neg
    return weights


def _make_head(kind: str = "abuse", hyperparams: dict[str, Any] | None = None) -> Pipeline:
    # Per-head configs from head_hyperparams.default.yaml (roadmap block 2).
    # ponytail: fixed per-head knobs; swap YAML instead of nested search on synth data.
    from refund_abuse_risk.config import load_head_hyperparams

    hp_all = hyperparams if hyperparams is not None else load_head_hyperparams()
    cfg = dict(hp_all.get(kind) or hp_all.get("abuse") or {})
    base = HistGradientBoostingClassifier(
        max_depth=int(cfg.get("max_depth", 4)),
        learning_rate=float(cfg.get("learning_rate", 0.08)),
        max_iter=int(cfg.get("max_iter", 120)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 5)),
        l2_regularization=float(cfg.get("l2_regularization", 1.0)),
        random_state=42,
    )
    method = str(cfg.get("calibrate_method", "sigmoid"))
    cv = int(cfg.get("cv", 3))
    clf = CalibratedClassifierCV(base, method=method, cv=cv)
    return Pipeline([("clf", clf)])


@dataclass
class TwoHeadModel:
    abuse_model: Any = None
    fraud_model: Any = None
    decision_stacker: DecisionStacker | None = None
    slice_calibrator: Any = None
    abuse_feature_columns: list[str] = field(default_factory=lambda: list(ABUSE_FEATURE_COLUMNS))
    fraud_feature_columns: list[str] = field(default_factory=lambda: list(FRAUD_FEATURE_COLUMNS))
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    model_version: str = "0.5.0"

    def fit(self, frame: pd.DataFrame, label_weights: dict[str, Any]) -> TwoHeadModel:
        labeled = apply_proxy_fraud_labels(frame, label_weights)
        abuse_y = labeled["abuse_label"].astype(int).to_numpy()
        fraud_y = labeled["fraud_label"].astype(int).to_numpy()
        abuse_w = sample_weights_for_frame(labeled, label_weights, head="abuse")
        fraud_w = sample_weights_for_frame(labeled, label_weights, head="fraud")
        if bool(label_weights.get("class_balance", False)):
            abuse_w = balance_class_weights(abuse_y, abuse_w)
            fraud_w = balance_class_weights(fraud_y, fraud_w)

        proxy_rules = label_weights.get("proxy_rules") or {}
        exclude_mint_fraud = bool(
            proxy_rules.get("exclude_mint_features_from_fraud_head", True)
        )
        exclude_mint_abuse = bool(
            proxy_rules.get("exclude_mint_features_from_abuse_head", True)
        )
        self.fraud_feature_columns = fraud_model_feature_columns(
            exclude_proxy_mint=exclude_mint_fraud
        )
        self.abuse_feature_columns = abuse_model_feature_columns(
            exclude_proxy_mint=exclude_mint_abuse
        )
        # Older synthetic frames may lack newly added columns.
        for col in (*self.abuse_feature_columns, *self.fraud_feature_columns):
            if col not in labeled.columns:
                labeled[col] = 0.0

        x_abuse = labeled[self.abuse_feature_columns].astype(float).to_numpy()
        x_fraud = labeled[self.fraud_feature_columns].astype(float).to_numpy()

        # OOF head probs for stacker (honest joint score); full-data heads for serve.
        oof_abuse, abuse_mode = self._oof_positive_proba(
            x_abuse, abuse_y, abuse_w, kind="abuse"
        )
        oof_fraud, fraud_mode = self._oof_positive_proba(
            x_fraud, fraud_y, fraud_w, kind="fraud"
        )
        stack_mode = (
            "oof" if abuse_mode == "oof" and fraud_mode == "oof" else "in_sample"
        )
        pattern_y = ((abuse_y >= 1) | (fraud_y >= 1)).astype(int)
        # Decision objective: proven fraud OR abuse labels — not proxy-minted fraud-only.
        source = (
            labeled["fraud_label_source"].astype(str).str.lower().to_numpy()
            if "fraud_label_source" in labeled.columns
            else np.array([""] * len(labeled))
        )
        strong = (
            labeled["strong_fraud_label"].astype(float).to_numpy()
            if "strong_fraud_label" in labeled.columns
            else np.zeros(len(labeled))
        )
        proven_y = ((source == "proven") | (strong >= 1)).astype(int)
        proxy_only = (source == "proxy") & (fraud_y >= 1) & (abuse_y < 1) & (proven_y < 1)
        stack_y = ((proven_y >= 1) | (abuse_y >= 1)).astype(int)
        # Drop proxy-only fraud from the joint target (keeps abuse + proven).
        stack_y = np.where(proxy_only, 0, stack_y).astype(int)
        min_stack_pos = int(label_weights.get("min_proven_for_stacker", 3))
        if int(stack_y.sum()) >= min_stack_pos and len(np.unique(stack_y)) > 1:
            stack_label = "proven_or_abuse"
        else:
            stack_y = pattern_y
            stack_label = "pattern_fallback"
        stack_w = np.maximum(abuse_w, fraud_w)
        # Lazy import avoids scoring.__init__ ↔ two_head cycle at module load.
        from refund_abuse_risk.config import load_head_hyperparams
        from refund_abuse_risk.scoring.decision import DecisionStacker
        from refund_abuse_risk.scoring.slice_calibrator import SliceCalibrator

        markets = labeled["market"].astype(str).to_numpy() if "market" in labeled.columns else None
        verticals = (
            labeled["vertical"].astype(str).to_numpy() if "vertical" in labeled.columns else None
        )
        self.decision_stacker = DecisionStacker().fit(
            oof_abuse * 100.0,
            oof_fraud * 100.0,
            stack_y,
            sample_weight=stack_w,
            fit_mode=stack_mode,
            markets=markets,
            verticals=verticals,
        )
        self.decision_stacker.stack_label = stack_label
        joint = self.decision_stacker.predict_scores(
            oof_abuse * 100.0, oof_fraud * 100.0, markets=markets, verticals=verticals
        )
        sc_cfg = (load_head_hyperparams().get("slice_calibrator") or {})
        self.slice_calibrator = SliceCalibrator(
            enabled=bool(sc_cfg.get("enabled", True)),
            min_slice_n=int(sc_cfg.get("min_slice_n", 40)),
            min_slice_positives=int(sc_cfg.get("min_slice_positives", 5)),
        ).fit(joint, stack_y, markets, verticals)

        self.abuse_model = _make_head("abuse")
        self.fraud_model = _make_head("fraud")
        self._fit_binary(self.abuse_model, x_abuse, abuse_y, abuse_w)
        self._fit_binary(self.fraud_model, x_fraud, fraud_y, fraud_w)
        return self

    @classmethod
    def _oof_positive_proba(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        *,
        kind: str = "abuse",
        n_splits: int = 3,
        random_state: int = 42,
    ) -> tuple[np.ndarray, str]:
        """Out-of-fold positive probabilities; falls back to in-sample if CV infeasible."""
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(y, dtype=int)
        n = len(y)
        classes, counts = np.unique(y, return_counts=True)
        if n < 4 or len(classes) < 2 or int(counts.min()) < 2:
            # Avoid CalibratedClassifierCV edge cases on tiny / one-class slices.
            base = cls._fit_histgbm_base(x, y, w)
            proba = base.predict_proba(x)
            classes_b = list(getattr(base, "classes_", [0, 1]))
            if 1 in classes_b:
                return proba[:, classes_b.index(1)], "in_sample"
            return np.zeros(n, dtype=float), "in_sample"

        splits = min(int(n_splits), int(counts.min()))
        if splits < 2:
            model = _make_head(kind)
            cls._fit_binary(model, x, y, w)
            return cls._positive_proba(model, x), "in_sample"

        oof = np.zeros(n, dtype=float)
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
        for train_idx, test_idx in skf.split(x, y):
            model = _make_head(kind)
            cls._fit_binary(model, x[train_idx], y[train_idx], w[train_idx])
            oof[test_idx] = cls._positive_proba(model, x[test_idx])
        return oof, "oof"

    @staticmethod
    def _fit_histgbm_base(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Any:
        base = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.1, max_iter=80, min_samples_leaf=3, random_state=42
        )
        try:
            base.fit(x, y, sample_weight=w)
        except TypeError:
            base.fit(x, y)
        return base

    @staticmethod
    def _set_clf_step(model: Pipeline, estimator: Any) -> None:
        # named_steps["clf"] = ... is a no-op on sklearn Pipeline; mutate steps.
        model.steps[-1] = ("clf", estimator)

    @classmethod
    def _fit_binary(cls, model: Pipeline, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            flip = 1 - int(classes[0]) if len(classes) == 1 else 1
            y = np.concatenate([y, np.array([flip], dtype=int)])
            x = np.vstack([x, x[0]])
            w = np.concatenate([w, np.array([1e-3])])
            classes, counts = np.unique(y, return_counts=True)

        # CalibratedClassifierCV needs >= cv samples per class; shrink cv if needed.
        min_count = int(counts.min())
        cv = min(3, min_count)
        if cv < 2:
            cls._set_clf_step(model, cls._fit_histgbm_base(x, y, w))
            return

        model.named_steps["clf"].set_params(cv=cv)
        try:
            try:
                model.fit(x, y, clf__sample_weight=w)
            except TypeError:
                model.fit(x, y)
        except (ValueError, RuntimeError):
            cls._set_clf_step(model, cls._fit_histgbm_base(x, y, w))
            return

        # Guard: CalibratedClassifierCV can appear "fit" to Pipeline yet be unusable.
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        step = model.named_steps["clf"]
        try:
            check_is_fitted(step)
        except (NotFittedError, ValueError, TypeError, AttributeError):
            cls._set_clf_step(model, cls._fit_histgbm_base(x, y, w))

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.abuse_model is None or self.fraud_model is None:
            raise RuntimeError("Model is not fitted")
        x_abuse = frame[self.abuse_feature_columns].astype(float).to_numpy()
        x_fraud = frame[self.fraud_feature_columns].astype(float).to_numpy()
        abuse_p = self._positive_proba(self.abuse_model, x_abuse)
        fraud_p = self._positive_proba(self.fraud_model, x_fraud)
        abuse_s = np.clip(abuse_p * 100.0, 0.0, 100.0)
        fraud_s = np.clip(fraud_p * 100.0, 0.0, 100.0)
        out = frame.copy()
        out["abuse_score"] = abuse_s
        out["fraud_score"] = fraud_s
        markets = frame["market"].astype(str).to_numpy() if "market" in frame.columns else None
        verticals = (
            frame["vertical"].astype(str).to_numpy() if "vertical" in frame.columns else None
        )
        if self.decision_stacker is None:
            decision = np.maximum(abuse_s, fraud_s)
        else:
            decision = self.decision_stacker.predict_scores(
                abuse_s, fraud_s, markets=markets, verticals=verticals
            )
        out["decision_score_raw"] = np.asarray(decision, dtype=float)
        if self.slice_calibrator is not None:
            decision = self.slice_calibrator.transform(decision, markets, verticals)
            out["slice_calibrator_applied"] = True
            n_slice = len(getattr(self.slice_calibrator, "models", {}) or {})
            out["slice_calibrator_n_models"] = int(n_slice)
        else:
            out["slice_calibrator_applied"] = False
            out["slice_calibrator_n_models"] = 0
        out["decision_score"] = decision
        return out

    @staticmethod
    def _positive_proba(model: Pipeline, x: np.ndarray) -> np.ndarray:
        # Prefer full Pipeline so StandardScaler runs when clf is still Calibrated*.
        step = model.named_steps["clf"]
        if isinstance(step, HistGradientBoostingClassifier):
            proba = step.predict_proba(x)
            classes = list(getattr(step, "classes_", [0, 1]))
        else:
            proba = model.predict_proba(x)
            classes = list(getattr(step, "classes_", [0, 1]))
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(x), dtype=float)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "abuse_model": self.abuse_model,
                "fraud_model": self.fraud_model,
                "decision_stacker": self.decision_stacker,
                "slice_calibrator": self.slice_calibrator,
                "abuse_feature_columns": self.abuse_feature_columns,
                "fraud_feature_columns": self.fraud_feature_columns,
                "feature_columns": self.feature_columns,
                "model_version": self.model_version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> TwoHeadModel:
        blob = joblib.load(path)
        return cls(
            abuse_model=blob["abuse_model"],
            fraud_model=blob["fraud_model"],
            decision_stacker=blob.get("decision_stacker"),
            slice_calibrator=blob.get("slice_calibrator"),
            abuse_feature_columns=list(blob.get("abuse_feature_columns", ABUSE_FEATURE_COLUMNS)),
            fraud_feature_columns=list(blob.get("fraud_feature_columns", FRAUD_FEATURE_COLUMNS)),
            feature_columns=list(blob.get("feature_columns", FEATURE_COLUMNS)),
            model_version=str(blob.get("model_version", "0.5.0")),
        )


def entity_prior_from_features(row: pd.Series | dict[str, Any]) -> float:
    """Standing prior from entity/link/device features (0-100)."""
    get = row.get if hasattr(row, "get") else lambda k, d=0: row[k] if k in row else d
    # Support-aware: thin history should not create extreme priors.
    orders = float(get("user_orders_30d", 0) or 0)
    support = min(orders / 5.0, 1.0)
    life_orders = float(get("user_lifetime_orders", 0) or 0)
    life_support = min(life_orders / 5.0, 1.0) if life_orders else 0.0
    parts = [
        float(get("user_refund_rate_30d", 0)) * 100.0 * support,
        float(get("user_refund_gmv_pct_30d", 0)) * 100.0 * support,
        min(float(get("user_refund_count_30d", 0)) * 5.0, 100.0) * support,
        float(get("user_refund_to_ltv_ratio", 0)) * 100.0 * life_support,
        min(float(get("user_lifetime_refund_count", 0)) * 3.0, 100.0) * life_support,
        float(get("driver_refund_rate_30d", 0)) * 80.0,
        float(get("vendor_refund_rate_30d", 0)) * 80.0,
        min(max(float(get("uvd_refund_lift", 1.0)) - 1.0, 0.0) * 35.0, 100.0),
        float(get("uvd_refund_share", 0)) * 90.0,
        min(max(float(get("device_cluster_size", 1.0)) - 1.0, 0.0) * 12.0, 100.0),
        min(max(float(get("accounts_per_device", 1.0)) - 1.0, 0.0) * 18.0, 100.0),
    ]
    return float(np.clip(max(parts), 0.0, 100.0))


def link_scores_from_features(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    get = row.get if hasattr(row, "get") else lambda k, d=0: row[k] if k in row else d

    def score(lift: float, cooccur: float, share: float) -> float:
        # Need support: lift alone with n=1 is not a ring.
        support = min(cooccur / 3.0, 1.0)
        base = min(max(lift - 1.0, 0.0) * 30.0, 70.0) * support
        conc = share * 40.0
        return float(np.clip(base + conc, 0.0, 100.0))

    return {
        "ud": score(
            float(get("ud_refund_lift", 1.0)),
            float(get("ud_cooccur", 0)),
            float(get("ud_refund_share", 0)),
        ),
        "uv": score(
            float(get("uv_refund_lift", 1.0)),
            float(get("uv_cooccur", 0)),
            float(get("uv_refund_share", 0)),
        ),
        "vd": score(
            float(get("vd_refund_lift", 1.0)),
            float(get("vd_cooccur", 0)),
            0.0,
        ),
        "uvd": score(
            float(get("uvd_refund_lift", 1.0)),
            float(get("uvd_cooccur", 0)),
            float(get("uvd_refund_share", 0)),
        ),
    }


def entity_scores_from_features(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    get = row.get if hasattr(row, "get") else lambda k, d=0: row[k] if k in row else d

    def entity_score(rate: float, gmv_pct: float, count: float, orders: float, count_scale: float) -> float:
        support = min(float(orders) / 5.0, 1.0) if orders else 0.0
        return float(
            np.clip(
                max(rate * 100.0 * support, gmv_pct * 100.0 * support, min(count * count_scale, 100.0)),
                0.0,
                100.0,
            )
        )

    return {
        "user": entity_score(
            float(get("user_refund_rate_30d", 0)),
            float(get("user_refund_gmv_pct_30d", 0)),
            float(get("user_refund_count_30d", 0)),
            float(get("user_orders_30d", 0)),
            6.0,
        ),
        "driver": entity_score(
            float(get("driver_refund_rate_30d", 0)),
            0.0,
            float(get("driver_refund_count_30d", 0)),
            float(get("driver_orders_30d", 0)),
            4.0,
        ),
        "vendor": entity_score(
            float(get("vendor_refund_rate_30d", 0)),
            float(get("vendor_refund_gmv_pct_30d", 0)),
            float(get("vendor_refund_count_30d", 0)),
            float(get("vendor_orders_30d", 0)),
            2.0,
        ),
    }


def device_cluster_score_from_features(row: pd.Series | dict[str, Any]) -> float:
    get = row.get if hasattr(row, "get") else lambda k, d=0: row[k] if k in row else d
    return float(
        np.clip(
            max(
                min(max(float(get("device_cluster_size", 1.0)) - 1.0, 0.0) * 12.0, 100.0),
                min(max(float(get("accounts_per_device", 1.0)) - 1.0, 0.0) * 20.0, 100.0),
                min(max(float(get("device_churn_30d", 0.0)) - 1.0, 0.0) * 10.0, 100.0),
            ),
            0.0,
            100.0,
        )
    )
