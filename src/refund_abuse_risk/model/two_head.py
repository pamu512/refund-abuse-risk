from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
)


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

    if "fraud_label" not in out.columns:
        out["fraud_label"] = 0
    if "fraud_label_source" not in out.columns:
        out["fraud_label_source"] = ""
    if "strong_fraud_label" not in out.columns:
        out["strong_fraud_label"] = 0

    out["fraud_label"] = pd.to_numeric(out["fraud_label"], errors="coerce").fillna(0).astype(int)
    out["fraud_label_source"] = (
        out["fraud_label_source"].astype("string").fillna("").astype(str).replace({"nan": "", "<NA>": ""})
    )
    proven = out["fraud_label_source"].str.lower().eq("proven") | (
        out["strong_fraud_label"].astype(float) >= 1
    )
    proxy_mask = (
        (out["device_cluster_size"].astype(float) >= min_cluster)
        & (out["accounts_per_device"].astype(float) >= min_accounts)
        & (out["uvd_refund_lift"].astype(float) >= min_lift)
        & (out["uvd_refund_share"].astype(float) >= min_share)
        & (out["uvd_cooccur"].astype(float) >= min_cooccur)
        & (out["user_refund_rate_30d"].astype(float) >= min_rate)
        & (out["user_orders_30d"].astype(float) >= min_orders)
    )
    assign_proxy = proxy_mask & ~proven & (out["fraud_label"] < 1)
    out.loc[assign_proxy, "fraud_label"] = 1
    out.loc[assign_proxy, "fraud_label_source"] = "proxy"
    out.loc[proven, "fraud_label"] = 1
    out.loc[proven, "fraud_label_source"] = "proven"
    return out


def sample_weights_for_frame(
    frame: pd.DataFrame,
    label_weights: dict[str, Any],
    *,
    head: str,
) -> np.ndarray:
    proxy_w = float(label_weights.get("fraud_proxy_weight", 0.4))
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
                weights[i] = proven_w if source[i] == "proven" else proxy_w
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


def _make_head() -> Pipeline:
    # HistGBM handles mixed scales; sigmoid calibration is stabler than isotonic on small N.
    # ponytail: no GNN — upgrade path is graph embeddings into these columns.
    base = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=120,
        min_samples_leaf=5,
        l2_regularization=1.0,
        random_state=42,
    )
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return Pipeline([("clf", clf)])


@dataclass
class TwoHeadModel:
    abuse_model: Any = None
    fraud_model: Any = None
    abuse_feature_columns: list[str] = field(default_factory=lambda: list(ABUSE_FEATURE_COLUMNS))
    fraud_feature_columns: list[str] = field(default_factory=lambda: list(FRAUD_FEATURE_COLUMNS))
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    model_version: str = "0.2.0"

    def fit(self, frame: pd.DataFrame, label_weights: dict[str, Any]) -> TwoHeadModel:
        labeled = apply_proxy_fraud_labels(frame, label_weights)
        abuse_y = labeled["abuse_label"].astype(int).to_numpy()
        fraud_y = labeled["fraud_label"].astype(int).to_numpy()
        abuse_w = sample_weights_for_frame(labeled, label_weights, head="abuse")
        fraud_w = sample_weights_for_frame(labeled, label_weights, head="fraud")

        x_abuse = labeled[self.abuse_feature_columns].astype(float).to_numpy()
        x_fraud = labeled[self.fraud_feature_columns].astype(float).to_numpy()

        self.abuse_model = _make_head()
        self.fraud_model = _make_head()
        self._fit_binary(self.abuse_model, x_abuse, abuse_y, abuse_w)
        self._fit_binary(self.fraud_model, x_fraud, fraud_y, fraud_w)
        return self

    @staticmethod
    def _fit_binary(model: Pipeline, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
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
            # Fall back to uncalibrated base estimator.
            base = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.1, max_iter=80, min_samples_leaf=3, random_state=42
            )
            base.fit(x, y, sample_weight=w)
            model.named_steps["clf"] = base
            return

        model.named_steps["clf"].set_params(cv=cv)
        try:
            model.fit(x, y, clf__sample_weight=w)
        except TypeError:
            model.fit(x, y)

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.abuse_model is None or self.fraud_model is None:
            raise RuntimeError("Model is not fitted")
        x_abuse = frame[self.abuse_feature_columns].astype(float).to_numpy()
        x_fraud = frame[self.fraud_feature_columns].astype(float).to_numpy()
        abuse_p = self._positive_proba(self.abuse_model, x_abuse)
        fraud_p = self._positive_proba(self.fraud_model, x_fraud)
        out = frame.copy()
        out["abuse_score"] = np.clip(abuse_p * 100.0, 0.0, 100.0)
        out["fraud_score"] = np.clip(fraud_p * 100.0, 0.0, 100.0)
        return out

    @staticmethod
    def _positive_proba(model: Pipeline, x: np.ndarray) -> np.ndarray:
        step = model.named_steps["clf"]
        proba = step.predict_proba(x)
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
            abuse_feature_columns=list(blob.get("abuse_feature_columns", ABUSE_FEATURE_COLUMNS)),
            fraud_feature_columns=list(blob.get("fraud_feature_columns", FRAUD_FEATURE_COLUMNS)),
            feature_columns=list(blob.get("feature_columns", FEATURE_COLUMNS)),
            model_version=str(blob.get("model_version", "0.2.0")),
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
