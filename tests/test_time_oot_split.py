from __future__ import annotations

import pandas as pd

from refund_abuse_risk.training.splits import time_based_order_split


def test_time_split_holds_out_last_days() -> None:
    rows = []
    for i in range(20):
        day = 1 + i  # 2026-07-01 .. 2026-07-20
        rows.append(
            {
                "order_id": f"O{i}",
                "event_ts": f"2026-07-{day:02d}T12:00:00Z",
                "abuse_label": int(i % 3 == 0),
            }
        )
    train, test, stats = time_based_order_split(
        pd.DataFrame(rows), holdout_days=7.0, min_train=5, min_test=3
    )
    assert stats["mode"] == "time_oot"
    assert stats["ok"]
    assert len(train) + len(test) == 20
    assert pd.to_datetime(train["event_ts"], utc=True).max() < pd.to_datetime(
        test["event_ts"], utc=True
    ).min()
    assert stats["test_n"] >= 3


def test_time_split_adapts_when_span_shorter_than_holdout() -> None:
    rows = [
        {"order_id": f"O{i}", "event_ts": f"2026-07-01T{i:02d}:00:00Z"}
        for i in range(10)
    ]
    train, test, stats = time_based_order_split(
        pd.DataFrame(rows), holdout_days=7.0, min_train=2, min_test=2
    )
    assert stats["adaptive"] is True
    assert len(train) >= 2 and len(test) >= 2
    assert pd.to_datetime(train["event_ts"], utc=True).max() <= pd.to_datetime(
        test["event_ts"], utc=True
    ).min()


def test_time_split_no_future_train_leak() -> None:
    frame = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d", "e", "f"],
            "event_ts": [
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00Z",
                "2026-01-10T00:00:00Z",
                "2026-01-11T00:00:00Z",
                "2026-01-12T00:00:00Z",
            ],
        }
    )
    train, test, _ = time_based_order_split(frame, holdout_days=3.0, min_train=1, min_test=1)
    assert set(train["order_id"]) == {"a", "b", "c"}
    assert set(test["order_id"]) == {"d", "e", "f"}
