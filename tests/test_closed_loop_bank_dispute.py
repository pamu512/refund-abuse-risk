from __future__ import annotations

import numpy as np
import pandas as pd

from refund_abuse_risk.config import load_disposition_labels
from refund_abuse_risk.labels.dispositions import apply_dispositions_to_orders
from scripts.generate_closed_loop_labels import _synth_dispositions


def test_synth_mints_bank_dispute_and_chargeback() -> None:
    rng = np.random.default_rng(7)
    n = 400
    orders = pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(n)],
            "event_ts": pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC"),
            "abuse_label": [1] * n,
            "fraud_label": [1] * n,
            "fraud_label_source": ["proven"] * n,
            "weak_policy_negative": [0] * n,
        }
    )
    disp = _synth_dispositions(orders, rng)
    kinds = set(disp["disposition"].astype(str))
    assert "chargeback_lost" in kinds
    assert "bank_dispute_lost" in kinds
    assert "investigator_confirmed_fraud" in kinds

    cfg = load_disposition_labels()
    labeled = apply_dispositions_to_orders(orders, disp, cfg)
    bank = disp["disposition"].eq("bank_dispute_lost")
    assert bank.any()
    bank_ids = set(disp.loc[bank, "order_id"].astype(str))
    proven = labeled["fraud_label_source"].astype(str).str.lower().eq("proven")
    assert labeled.loc[labeled["order_id"].isin(bank_ids), "fraud_label"].eq(1).all()
    assert proven.loc[labeled["order_id"].isin(bank_ids)].all()
