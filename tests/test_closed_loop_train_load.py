from __future__ import annotations

from pathlib import Path

import pandas as pd

from refund_abuse_risk.training.closed_loop import load_training_orders


def test_load_training_orders_prefers_labeled(tmp_path: Path) -> None:
    orders = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "abuse_label": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "weak_policy_negative": 0,
                "event_ts": "2026-07-01T00:00:00Z",
            },
            {
                "order_id": "O2",
                "abuse_label": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "weak_policy_negative": 0,
                "event_ts": "2026-07-02T00:00:00Z",
            },
        ]
    )
    labeled = orders.copy()
    labeled.loc[labeled["order_id"] == "O2", "fraud_label"] = 1
    labeled.loc[labeled["order_id"] == "O2", "fraud_label_source"] = "proven"
    orders.to_csv(tmp_path / "orders.csv", index=False)
    labeled.to_csv(tmp_path / "orders.labeled.csv", index=False)
    sdk = orders.copy()
    sdk["device_risk_score"] = [0.0, 90.0]
    sdk.to_csv(tmp_path / "orders.sdk.csv", index=False)

    frame, stats = load_training_orders(tmp_path)
    assert "orders.labeled.csv" in stats["orders_source"]
    assert int(frame.loc[frame["order_id"] == "O2", "fraud_label"].iloc[0]) == 1
    assert float(frame.loc[frame["order_id"] == "O2", "device_risk_score"].iloc[0]) == 90.0
    assert stats["proven_rate"] == 0.5
