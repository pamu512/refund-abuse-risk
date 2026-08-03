from __future__ import annotations

import pandas as pd

from refund_abuse_risk.config import load_disposition_labels
from refund_abuse_risk.labels.dispositions import apply_dispositions_to_orders


def test_disposition_overlays_labels_prefer_latest() -> None:
    orders = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "event_ts": "2026-07-01T00:00:00Z",
                "abuse_label": 0,
                "abuse_label_weak": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
            },
            {
                "order_id": "O2",
                "event_ts": "2026-07-01T00:00:00Z",
                "abuse_label": 0,
                "abuse_label_weak": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
            },
        ]
    )
    dispositions = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "disposition": "weak_policy_auto_grant",
                "disposition_ts": "2026-07-10T00:00:00Z",
            },
            {
                "order_id": "O1",
                "disposition": "investigator_confirmed_fraud",
                "disposition_ts": "2026-07-12T00:00:00Z",
            },
            {
                "order_id": "O2",
                "disposition": "manual_denied_abuse",
                "disposition_ts": "2026-07-10T00:00:00Z",
            },
        ]
    )
    cfg = load_disposition_labels()
    cfg["lag_days"] = 7
    out = apply_dispositions_to_orders(orders, dispositions, cfg)
    o1 = out.loc[out["order_id"] == "O1"].iloc[0]
    assert int(o1["strong_fraud_label"]) == 1
    assert str(o1["fraud_label_source"]) == "proven"
    assert int(o1["abuse_label"]) == 1
    o2 = out.loc[out["order_id"] == "O2"].iloc[0]
    assert int(o2["abuse_label"]) == 1
    assert int(o2["fraud_label"]) == 0
    assert out.attrs["dispositions_applied"] == 2


def test_disposition_lag_skips_early_labels() -> None:
    orders = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "event_ts": "2026-07-01T00:00:00Z",
                "abuse_label": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
                "abuse_label_weak": 0,
            }
        ]
    )
    dispositions = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "disposition": "investigator_confirmed_fraud",
                "disposition_ts": "2026-07-03T00:00:00Z",  # < event + 7d
            }
        ]
    )
    cfg = load_disposition_labels()
    cfg["lag_days"] = 7
    out = apply_dispositions_to_orders(orders, dispositions, cfg)
    assert int(out.iloc[0]["fraud_label"]) == 0
    assert out.attrs["dispositions_skipped_lag"] == 1
    assert out.attrs["dispositions_applied"] == 0
