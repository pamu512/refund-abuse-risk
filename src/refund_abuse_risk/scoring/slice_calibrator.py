"""Per market×vertical calibration of joint decision scores (post-stacker)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class SliceCalibrator:
    """
    Fit a tiny logistic calibrator per market×vertical when support clears floors.

    Falls back to global calibrator, then identity.
    """

    models: dict[str, Any] = field(default_factory=dict)
    global_model: Any = None
    min_slice_n: int = 40
    min_slice_positives: int = 5
    enabled: bool = True

    def fit(
        self,
        scores_0_100: np.ndarray,
        y: np.ndarray,
        markets: np.ndarray | None,
        verticals: np.ndarray | None,
    ) -> SliceCalibrator:
        if not self.enabled:
            return self
        s = np.asarray(scores_0_100, dtype=float).reshape(-1, 1) / 100.0
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            return self
        self.global_model = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        self.global_model.fit(s, y)
        if markets is None or verticals is None:
            return self
        m = np.asarray([str(x).upper() if x else "ALL" for x in markets])
        v = np.asarray([str(x).lower() if x else "all" for x in verticals])
        keys = np.array([f"{a}|{b}" for a, b in zip(m, v, strict=True)])
        self.models = {}
        for key in sorted(set(keys.tolist())):
            mask = keys == key
            if int(mask.sum()) < int(self.min_slice_n):
                continue
            if int(y[mask].sum()) < int(self.min_slice_positives):
                continue
            if len(np.unique(y[mask])) < 2:
                continue
            clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            clf.fit(s[mask], y[mask])
            self.models[key] = clf
        return self

    def transform(
        self,
        scores_0_100: np.ndarray,
        markets: np.ndarray | None = None,
        verticals: np.ndarray | None = None,
    ) -> np.ndarray:
        raw = np.asarray(scores_0_100, dtype=float)
        if not self.enabled or (self.global_model is None and not self.models):
            return raw
        s = (raw / 100.0).reshape(-1, 1)
        out = np.zeros(len(raw), dtype=float)
        if markets is None or verticals is None or not self.models:
            if self.global_model is None:
                return raw
            proba = self.global_model.predict_proba(s)
            classes = list(self.global_model.classes_)
            p = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(raw))
            return np.clip(p * 100.0, 0.0, 100.0)

        m = np.asarray([str(x).upper() if x else "ALL" for x in markets])
        v = np.asarray([str(x).lower() if x else "all" for x in verticals])
        for i in range(len(raw)):
            key = f"{m[i]}|{v[i]}"
            model = self.models.get(key) or self.global_model
            if model is None:
                out[i] = raw[i]
                continue
            proba = model.predict_proba(s[i : i + 1])
            classes = list(model.classes_)
            p = float(proba[0, classes.index(1)]) if 1 in classes else 0.0
            out[i] = p * 100.0
        return np.clip(out, 0.0, 100.0)
