from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from refund_abuse_risk.features.builders import FEATURE_COLUMNS


def apply_proxy_fraud_labels(
    frame: pd.DataFrame,
    label_weights: dict[str, Any],
) -> pd.DataFrame:
    """Mark high-confidence proxy fraud positives; preserve proven labels."""
    out = frame.copy()
    rules = label_weights.get("proxy_rules") or {}
    min_cluster = float(rules.get("min_device_cluster_size", 4))
    min_accounts = float(rules.get("min_accounts_per_device", 3))
    min_lift = float(rules.get("min_uvd_refund_lift", 2.0))
    min_rate = float(rules.get("min_coordinated_refund_rate", 0.35))

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
        & (out["user_refund_rate_30d"].astype(float) >= min_rate)
    )
    # Proven wins; proxy only fills unlabeled / non-proven rows.
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

    weights = np.ones(len(frame), dtype=float)
    if head == "fraud":
        source = frame.get("fraud_label_source", pd.Series([""] * len(frame))).astype(str).str.lower()
        y = frame["fraud_label"].astype(float).to_numpy()
        for i, (label, src) in enumerate(zip(y, source, strict=True)):
            if label >= 1:
                weights[i] = proven_w if src == "proven" else proxy_w
            else:
                weights[i] = clean_neg
        return weights

    # abuse head
    y = frame["abuse_label"].astype(float).to_numpy()
    weak = frame.get("abuse_label_weak", pd.Series([0] * len(frame))).astype(float).to_numpy()
    weak_policy_neg = (
        frame.get("weak_policy_negative", pd.Series([0] * len(frame))).astype(float).to_numpy()
    )
    for i, (label, is_weak, is_weak_neg) in enumerate(zip(y, weak, weak_policy_neg, strict=True)):
        if label >= 1:
            weights[i] = abuse_weak if is_weak >= 1 else abuse_pos
        elif is_weak_neg >= 1:
            weights[i] = weak_neg
        else:
            weights[i] = clean_neg
    return weights


def _make_head(n_splits: int = 3) -> Pipeline:
    # ponytail: GBM + isotonic calibration; ceiling ~tabular collusion; upgrade: LightGBM/GNN.
    base = GradientBoostingClassifier(random_state=42)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=max(2, n_splits))
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


@dataclass
class TwoHeadModel:
    abuse_model: Any = None
    fraud_model: Any = None
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    model_version: str = "0.1.0"

    def fit(self, frame: pd.DataFrame, label_weights: dict[str, Any]) -> TwoHeadModel:
        labeled = apply_proxy_fraud_labels(frame, label_weights)
        x = labeled[self.feature_columns].astype(float).to_numpy()

        abuse_y = labeled["abuse_label"].astype(int).to_numpy()
        fraud_y = labeled["fraud_label"].astype(int).to_numpy()
        abuse_w = sample_weights_for_frame(labeled, label_weights, head="abuse")
        fraud_w = sample_weights_for_frame(labeled, label_weights, head="fraud")

        def _splits(y: np.ndarray) -> int:
            _, counts = np.unique(y, return_counts=True)
            return int(max(2, min(3, counts.min())))

        self.abuse_model = _make_head(_splits(abuse_y if len(np.unique(abuse_y)) > 1 else np.array([0, 1])))
        self.fraud_model = _make_head(_splits(fraud_y if len(np.unique(fraud_y)) > 1 else np.array([0, 1])))
        self._fit_binary(self.abuse_model, x, abuse_y, abuse_w)
        self._fit_binary(self.fraud_model, x, fraud_y, fraud_w)
        return self

    @staticmethod
    def _fit_binary(model: Pipeline, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        classes = np.unique(y)
        if len(classes) < 2:
            # Degenerate demo slice: synthesize opposite class with tiny weight.
            flip = 1 - int(classes[0]) if len(classes) == 1 else 1
            y = np.concatenate([y, np.array([flip], dtype=int)])
            x = np.vstack([x, x[0]])
            w = np.concatenate([w, np.array([1e-3])])
        try:
            model.fit(x, y, clf__sample_weight=w)
        except TypeError:
            # Older sklearn path: fit without sample weights.
            model.fit(x, y)

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.abuse_model is None or self.fraud_model is None:
            raise RuntimeError("Model is not fitted")
        x = frame[self.feature_columns].astype(float).to_numpy()
        abuse_p = self._positive_proba(self.abuse_model, x)
        fraud_p = self._positive_proba(self.fraud_model, x)
        out = frame.copy()
        out["abuse_score"] = np.clip(abuse_p * 100.0, 0.0, 100.0)
        out["fraud_score"] = np.clip(fraud_p * 100.0, 0.0, 100.0)
        return out

    @staticmethod
    def _positive_proba(model: Pipeline, x: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(x)
        classes = list(model.named_steps["clf"].classes_)
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
            feature_columns=list(blob["feature_columns"]),
            model_version=str(blob.get("model_version", "0.1.0")),
        )


def entity_prior_from_features(row: pd.Series | dict[str, Any]) -> float:
    """Standing prior from entity/link/device features (0-100), before order head scores."""
    get = row.get if hasattr(row, "get") else lambda k, d=0: row[k] if k in row else d
    parts = [
        float(get("user_refund_rate_30d", 0)) * 100.0,
        float(get("user_refund_gmv_pct_30d", 0)) * 100.0,
        min(float(get("user_refund_count_30d", 0)) * 5.0, 100.0),
        float(get("driver_refund_rate_30d", 0)) * 80.0,
        float(get("vendor_refund_rate_30d", 0)) * 80.0,
        min(float(get("uvd_refund_lift", 1.0)) * 20.0, 100.0),
        min(float(get("device_cluster_size", 1.0)) * 8.0, 100.0),
        min(float(get("accounts_per_device", 1.0)) * 15.0, 100.0),
    ]
    return float(np.clip(max(parts), 0.0, 100.0))


def link_scores_from_features(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    get = row.get if hasattr(row, "get") else lambda k, d=0: row[k] if k in row else d

    def lift_score(lift: float, cooccur: float) -> float:
        base = min(max(lift, 0.0) * 25.0, 100.0)
        boost = min(cooccur * 2.0, 20.0)
        return float(np.clip(base + boost, 0.0, 100.0))

    return {
        "ud": lift_score(float(get("ud_refund_lift", 1.0)), float(get("ud_cooccur", 0))),
        "uv": lift_score(float(get("uv_refund_lift", 1.0)), float(get("uv_cooccur", 0))),
        "vd": lift_score(float(get("vd_refund_lift", 1.0)), float(get("vd_cooccur", 0))),
        "uvd": lift_score(float(get("uvd_refund_lift", 1.0)), float(get("uvd_cooccur", 0))),
    }


def entity_scores_from_features(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    get = row.get if hasattr(row, "get") else lambda k, d=0: row[k] if k in row else d
    return {
        "user": float(
            np.clip(
                max(
                    float(get("user_refund_rate_30d", 0)) * 100.0,
                    float(get("user_refund_gmv_pct_30d", 0)) * 100.0,
                    min(float(get("user_refund_count_30d", 0)) * 6.0, 100.0),
                ),
                0.0,
                100.0,
            )
        ),
        "driver": float(
            np.clip(
                max(
                    float(get("driver_refund_rate_30d", 0)) * 100.0,
                    min(float(get("driver_refund_count_30d", 0)) * 4.0, 100.0),
                ),
                0.0,
                100.0,
            )
        ),
        "vendor": float(
            np.clip(
                max(
                    float(get("vendor_refund_rate_30d", 0)) * 100.0,
                    float(get("vendor_refund_gmv_pct_30d", 0)) * 100.0,
                    min(float(get("vendor_refund_count_30d", 0)) * 2.0, 100.0),
                ),
                0.0,
                100.0,
            )
        ),
    }


def device_cluster_score_from_features(row: pd.Series | dict[str, Any]) -> float:
    get = row.get if hasattr(row, "get") else lambda k, d=0: row[k] if k in row else d
    return float(
        np.clip(
            max(
                min(float(get("device_cluster_size", 1.0)) * 10.0, 100.0),
                min(float(get("accounts_per_device", 1.0)) * 18.0, 100.0),
                min(float(get("device_churn_30d", 0.0)) * 8.0, 100.0),
            ),
            0.0,
            100.0,
        )
    )
