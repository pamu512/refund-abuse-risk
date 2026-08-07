"""Regression tests for third-party audit C1–C3 / H4 / H7 fixes."""

from __future__ import annotations

import numpy as np
import pytest

from refund_abuse_risk.integrations.feeds import _assert_http_uri_allowed, pull_http
from refund_abuse_risk.scoring.thresholds import threshold_at_recall_costed


def test_cost_infeasible_returns_none_not_fallback_cutoff() -> None:
    # Scores interleaved so any full-recall cutoff includes many FPs.
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=float)
    s = np.array([90, 89, 88, 87, 86, 10, 5, 1], dtype=float)
    t, info = threshold_at_recall_costed(
        y, s, target_recall=1.0, min_precision=0.95, max_fp_rate=0.01
    )
    assert t is None
    assert info["feasible"] is False
    assert info["soft_threshold_usable"] is False
    assert info.get("recall_only_threshold") is not None


def test_amounts_length_mismatch_raises() -> None:
    y = np.array([1, 0, 0], dtype=float)
    s = np.array([90, 20, 10], dtype=float)
    with pytest.raises(ValueError, match="amounts length"):
        threshold_at_recall_costed(
            y, s, target_recall=1.0, amounts=np.array([1.0, 2.0]), max_fp_refund_dollars_mean=5.0
        )


def test_http_requires_allowed_hosts() -> None:
    with pytest.raises(ValueError, match="allowed_hosts"):
        _assert_http_uri_allowed(
            "x", "https://evil.example/a.csv", {"uri": "https://evil.example/a.csv"}
        )
    with pytest.raises(ValueError, match="not in allowed_hosts"):
        _assert_http_uri_allowed(
            "x",
            "https://evil.example/a.csv",
            {"allowed_hosts": ["good.example"]},
        )
    _assert_http_uri_allowed(
        "x",
        "https://good.example/a.csv",
        {"allowed_hosts": ["good.example"]},
    )


def test_http_blocks_unlisted_loopback(tmp_path) -> None:
    with pytest.raises(ValueError, match="allowed_hosts|private|loopback|not in"):
        pull_http(
            "x",
            {"uri": "http://127.0.0.1:9/x", "filename": "x.csv"},
            root=tmp_path,
            stage_root=tmp_path / "s",
        )
