"""Deprecated alias for refund_abuse_risk.graph.ridge_feature_decoder.

Historical Python module name. Prefer run_ridge_feature_decoder.
This is not a first-class public API and is not a GNN or GraphBEAN product.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from refund_abuse_risk.graph.ridge_feature_decoder import run_ridge_feature_decoder


def run_graphbean_lite(
    history: pd.DataFrame,
    *,
    bipartite_cfg: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deprecated alias for run_ridge_feature_decoder."""
    warnings.warn(
        "run_graphbean_lite is deprecated; use run_ridge_feature_decoder",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_ridge_feature_decoder(history, bipartite_cfg=bipartite_cfg, cfg=cfg)
