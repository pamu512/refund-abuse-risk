from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from refund_abuse_risk.graph.entities import link_key
from refund_abuse_risk.pipeline.score import (
    claim_path_read,
    on_entity_risk_change,
    precompute_orders,
    refresh_order,
    train_two_head,
)
from refund_abuse_risk.schemas.models import LifecycleEvent

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "out.json"


def main() -> None:
    if not (DATA / "orders.csv").exists():
        import runpy

        runpy.run_path(str(ROOT / "scripts" / "generate_demo_data.py"), run_name="__main__")

    orders = pd.read_csv(DATA / "orders.csv")
    history = pd.read_csv(DATA / "history.csv")
    devices = pd.read_csv(DATA / "devices.csv")

    model = train_two_head(orders, history, devices)
    model_path = ROOT / "models" / "two_head.joblib"
    model.save(model_path)

    cache = precompute_orders(orders, history, devices, model)

    # Lifecycle refresh on one order.
    sample = orders.iloc[0].to_dict()
    refresh_order(sample, history, devices, model, cache, event=LifecycleEvent.DELIVERED)

    # Risk-change rescoring for fraud ring driver.
    key = link_key("driver", "DF0")
    on_entity_risk_change(
        {key},
        orders,
        history,
        devices,
        model,
        cache,
        new_scores={key: 95.0},
    )

    # Sync claim-path reads.
    payloads = []
    for order_id in ["O-CLEAN-0", "O-ABUSE-0", "O-FRAUD-0", "O-PROXY-0"]:
        snap = claim_path_read(order_id, cache)
        payloads.append(snap.model_dump(mode="json"))

    OUT.write_text(json.dumps(payloads, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(payloads)} snapshots to {OUT}")
    for p in payloads:
        print(
            f"{p['order_id']}: abuse={p['abuse_score']:.1f} fraud={p['fraud_score']:.1f} "
            f"tier={p['suggested_tier']} hard_gated={p['hard_gated']} reasons={p['reason_codes'][:3]}"
        )


if __name__ == "__main__":
    main()
