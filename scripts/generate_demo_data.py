#!/usr/bin/env python3
"""Generate synthetic orders/history/devices CSVs for the CSV demo."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    history_rows: list[dict] = []
    order_rows: list[dict] = []
    device_rows: list[dict] = []
    user_rows: list[dict] = []

    def add_user(uid: str, signup: datetime) -> None:
        user_rows.append({"user_id": uid, "signup_ts": signup.isoformat()})

    # Clean users — mature, high LTV, rare refunds
    for i in range(20):
        uid, did, vid, dev = f"U{i}", f"D{i%5}", f"V{i%4}", f"DEV{i}"
        add_user(uid, base - timedelta(days=180 + i))
        device_rows.append(
            {
                "user_id": uid,
                "device_id": dev,
                "cluster_id": f"C{i}",
                "last_seen_ts": (base + timedelta(days=20)).isoformat(),
            }
        )
        for j in range(12):
            ts = base + timedelta(days=j * 2, hours=i)
            # Keep baseline refund rate low so collusion lift stands out.
            is_refund = 1 if j == 11 and i % 10 == 0 else 0
            history_rows.append(
                {
                    "order_id": f"H-CLEAN-{i}-{j}",
                    "user_id": uid,
                    "driver_id": did,
                    "vendor_id": vid,
                    "device_id": dev,
                    "market": "SG",
                    "vertical": "food" if i % 2 == 0 else "qcommerce",
                    "amount": 20 + i,
                    "is_refund": is_refund,
                    "event_ts": ts.isoformat(),
                    "status": "delivered",
                    "claim_reason": "quality" if is_refund else "",
                }
            )
        order_rows.append(
            {
                "order_id": f"O-CLEAN-{i}",
                "user_id": uid,
                "driver_id": did,
                "vendor_id": vid,
                "device_id": dev,
                "market": "SG",
                "vertical": "food" if i % 2 == 0 else "qcommerce",
                "amount": 25 + i,
                "status": "delivered",
                "claim_reason": "",
                "event_ts": (base + timedelta(days=25, hours=i)).isoformat(),
                "abuse_label": 0,
                "abuse_label_weak": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
            }
        )

    # Serial claimant abuse — mature enough for count/rate gates + high LTV burn
    for i in range(15):
        uid, did, vid, dev = f"UA{i}", f"DA{i%3}", f"VA{i%3}", f"DEVA{i}"
        add_user(uid, base - timedelta(days=60 + i))
        device_rows.append(
            {
                "user_id": uid,
                "device_id": dev,
                "cluster_id": f"CA{i}",
                "last_seen_ts": (base + timedelta(days=22)).isoformat(),
            }
        )
        for j in range(10):
            ts = base + timedelta(days=j, hours=i)
            history_rows.append(
                {
                    "order_id": f"H-ABUSE-{i}-{j}",
                    "user_id": uid,
                    "driver_id": did,
                    "vendor_id": vid,
                    "device_id": dev,
                    "market": "SG",
                    "vertical": "food",
                    "amount": 40 + j,
                    "is_refund": 1 if j >= 3 else 0,
                    "event_ts": ts.isoformat(),
                    "status": "delivered",
                    "claim_reason": "missing_item" if j >= 3 else "",
                }
            )
        order_rows.append(
            {
                "order_id": f"O-ABUSE-{i}",
                "user_id": uid,
                "driver_id": did,
                "vendor_id": vid,
                "device_id": dev,
                "market": "SG",
                "vertical": "food",
                "amount": 55,
                "status": "delivered",
                "claim_reason": "missing_item",
                "event_ts": (base + timedelta(days=26, hours=i)).isoformat(),
                "abuse_label": 1,
                "abuse_label_weak": 1 if i % 2 == 0 else 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
            }
        )

    # Proven fraud collusion ring (shared device cluster + UVD)
    for i in range(12):
        uid = f"UF{i}"
        did, vid, dev = "DF0", "VF0", f"DEVF{i%2}"  # two devices, many accounts
        add_user(uid, base - timedelta(days=30 + i))
        device_rows.append(
            {
                "user_id": uid,
                "device_id": dev,
                "cluster_id": "CFARM",
                "last_seen_ts": (base + timedelta(days=24)).isoformat(),
            }
        )
        for j in range(6):
            ts = base + timedelta(days=j + 5, hours=i)
            history_rows.append(
                {
                    "order_id": f"H-FRAUD-{i}-{j}",
                    "user_id": uid,
                    "driver_id": did,
                    "vendor_id": vid,
                    "device_id": dev,
                    "market": "SG",
                    "vertical": "food",
                    "amount": 80,
                    "is_refund": 1,
                    "event_ts": ts.isoformat(),
                    "status": "delivered",
                    "claim_reason": "wrong_order",
                }
            )
        order_rows.append(
            {
                "order_id": f"O-FRAUD-{i}",
                "user_id": uid,
                "driver_id": did,
                "vendor_id": vid,
                "device_id": dev,
                "market": "SG",
                "vertical": "food",
                "amount": 90,
                "status": "delivered",
                "claim_reason": "wrong_order",
                "event_ts": (base + timedelta(days=27, hours=i)).isoformat(),
                "abuse_label": 1,
                "abuse_label_weak": 0,
                "fraud_label": 1,
                "fraud_label_source": "proven" if i < 8 else "proxy",
                "strong_fraud_label": 1 if i < 8 else 0,
                # Account-level prior (disposition), not same-order training label.
                "prior_strong_fraud": 1 if i < 8 else 0,
                "weak_policy_negative": 0,
            }
        )

    # Proxy-eligible ring (not marked proven; features should trigger proxy rule)
    for i in range(10):
        uid = f"UP{i}"
        did, vid, dev = "DP0", "VP0", f"DEVP{i%3}"
        add_user(uid, base - timedelta(days=45 + i))
        device_rows.append(
            {
                "user_id": uid,
                "device_id": dev,
                "cluster_id": "CPROXY",
                "last_seen_ts": (base + timedelta(days=23)).isoformat(),
            }
        )
        for j in range(5):
            ts = base + timedelta(days=j + 8, hours=i)
            history_rows.append(
                {
                    "order_id": f"H-PROXY-{i}-{j}",
                    "user_id": uid,
                    "driver_id": did,
                    "vendor_id": vid,
                    "device_id": dev,
                    "market": "ID",
                    "vertical": "qcommerce",
                    "amount": 30,
                    "is_refund": 1 if j >= 1 else 0,
                    "event_ts": ts.isoformat(),
                    "status": "delivered",
                    "claim_reason": "missing_item",
                }
            )
        order_rows.append(
            {
                "order_id": f"O-PROXY-{i}",
                "user_id": uid,
                "driver_id": did,
                "vendor_id": vid,
                "device_id": dev,
                "market": "ID",
                "vertical": "qcommerce",
                "amount": 35,
                "status": "delivered",
                "claim_reason": "missing_item",
                "event_ts": (base + timedelta(days=28, hours=i)).isoformat(),
                "abuse_label": 1,
                "abuse_label_weak": 1,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
            }
        )

    # Weak-policy negatives (approved refunds that shouldn't be clean negatives)
    for i in range(8):
        uid, did, vid, dev = f"UW{i}", f"DW{i%2}", f"VW{i%2}", f"DEVW{i}"
        add_user(uid, base - timedelta(days=90 + i))
        device_rows.append(
            {
                "user_id": uid,
                "device_id": dev,
                "cluster_id": f"CW{i}",
                "last_seen_ts": (base + timedelta(days=21)).isoformat(),
            }
        )
        for j in range(4):
            ts = base + timedelta(days=j + 2, hours=i)
            history_rows.append(
                {
                    "order_id": f"H-WEAK-{i}-{j}",
                    "user_id": uid,
                    "driver_id": did,
                    "vendor_id": vid,
                    "device_id": dev,
                    "market": "SG",
                    "vertical": "food",
                    "amount": 22,
                    "is_refund": 1 if j == 3 else 0,
                    "event_ts": ts.isoformat(),
                    "status": "delivered",
                    "claim_reason": "quality" if j == 3 else "",
                }
            )
        order_rows.append(
            {
                "order_id": f"O-WEAK-{i}",
                "user_id": uid,
                "driver_id": did,
                "vendor_id": vid,
                "device_id": dev,
                "market": "SG",
                "vertical": "food",
                "amount": 22,
                "status": "delivered",
                "claim_reason": "quality",
                "event_ts": (base + timedelta(days=24, hours=i)).isoformat(),
                "abuse_label": 0,
                "abuse_label_weak": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 1,
            }
        )

    # New user with 1-2 refunds, still positive LTV — isolated UVD, should NOT hard-gate
    for i in range(5):
        uid, did, vid, dev = f"UN{i}", f"DN{i}", f"VN{i}", f"DEVN{i}"
        add_user(uid, base + timedelta(days=20))  # signed up ~5 days before scoring
        device_rows.append(
            {
                "user_id": uid,
                "device_id": dev,
                "cluster_id": f"CN{i}",
                "last_seen_ts": (base + timedelta(days=24)).isoformat(),
            }
        )
        for j in range(4):
            ts = base + timedelta(days=22 + j, hours=i)
            history_rows.append(
                {
                    "order_id": f"H-NEW-{i}-{j}",
                    "user_id": uid,
                    "driver_id": did,
                    "vendor_id": vid,
                    "device_id": dev,
                    "market": "SG",
                    "vertical": "food",
                    "amount": 25,
                    # 1 refund / 4 orders → refund_to_ltv=0.25, below early-life burn 0.60
                    "is_refund": 1 if j == 3 else 0,
                    "event_ts": ts.isoformat(),
                    "status": "delivered",
                    "claim_reason": "quality" if j == 3 else "",
                }
            )
        order_rows.append(
            {
                "order_id": f"O-NEW-{i}",
                "user_id": uid,
                "driver_id": did,
                "vendor_id": vid,
                "device_id": dev,
                "market": "SG",
                "vertical": "food",
                "amount": 25,
                "status": "delivered",
                "claim_reason": "quality",
                "event_ts": (base + timedelta(days=26, hours=i)).isoformat(),
                "abuse_label": 0,
                "abuse_label_weak": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
            }
        )

    # New user attached to fraud UVD/device ring — early pass denied via related/combined
    for i in range(3):
        uid = f"UNR{i}"
        did, vid, dev = "DF0", "VF0", f"DEVF{i%2}"
        add_user(uid, base + timedelta(days=24))
        device_rows.append(
            {
                "user_id": uid,
                "device_id": dev,
                "cluster_id": "CFARM",
                "last_seen_ts": (base + timedelta(days=26)).isoformat(),
            }
        )
        for j in range(2):
            ts = base + timedelta(days=25 + j, hours=i)
            history_rows.append(
                {
                    "order_id": f"H-NEWRING-{i}-{j}",
                    "user_id": uid,
                    "driver_id": did,
                    "vendor_id": vid,
                    "device_id": dev,
                    "market": "SG",
                    "vertical": "food",
                    "amount": 30,
                    "is_refund": 1 if j == 1 else 0,
                    "event_ts": ts.isoformat(),
                    "status": "delivered",
                    "claim_reason": "missing_item" if j == 1 else "",
                }
            )
        order_rows.append(
            {
                "order_id": f"O-NEWRING-{i}",
                "user_id": uid,
                "driver_id": did,
                "vendor_id": vid,
                "device_id": dev,
                "market": "SG",
                "vertical": "food",
                "amount": 30,
                "status": "delivered",
                "claim_reason": "missing_item",
                "event_ts": (base + timedelta(days=27, hours=i)).isoformat(),
                "abuse_label": 1,
                "abuse_label_weak": 1,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
            }
        )

    # Early-life LTV burn — new account, almost all GMV refunded → hard gate
    for i in range(4):
        uid, did, vid, dev = f"UB{i}", f"D{i%5}", f"V{i%4}", f"DEVB{i}"
        add_user(uid, base + timedelta(days=22))
        device_rows.append(
            {
                "user_id": uid,
                "device_id": dev,
                "cluster_id": f"CB{i}",
                "last_seen_ts": (base + timedelta(days=25)).isoformat(),
            }
        )
        for j in range(3):
            ts = base + timedelta(days=23 + j, hours=i)
            history_rows.append(
                {
                    "order_id": f"H-BURN-{i}-{j}",
                    "user_id": uid,
                    "driver_id": did,
                    "vendor_id": vid,
                    "device_id": dev,
                    "market": "SG",
                    "vertical": "food",
                    "amount": 40,
                    "is_refund": 1,
                    "event_ts": ts.isoformat(),
                    "status": "delivered",
                    "claim_reason": "missing_item",
                }
            )
        order_rows.append(
            {
                "order_id": f"O-BURN-{i}",
                "user_id": uid,
                "driver_id": did,
                "vendor_id": vid,
                "device_id": dev,
                "market": "SG",
                "vertical": "food",
                "amount": 40,
                "status": "delivered",
                "claim_reason": "missing_item",
                "event_ts": (base + timedelta(days=27, hours=i)).isoformat(),
                "abuse_label": 1,
                "abuse_label_weak": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
            }
        )

    # Integrity / claim-proof cohort: cloned app + same device as courier + AI claim photo.
    for i in range(4):
        uid, did, vid, dev = f"UINT{i}", f"DINT{i}", f"VINT{i}", f"DEVINT{i}"
        add_user(uid, base - timedelta(days=60 + i))
        device_rows.append(
            {
                "user_id": uid,
                "device_id": dev,
                "cluster_id": f"CINT{i}",
                "last_seen_ts": (base + timedelta(days=26)).isoformat(),
                "device_risk_score": 88,
                "is_emulator": 0,
                "is_cloned_app": 1,
                "is_gps_spoof": 1,
                "is_tampered": 1,
            }
        )
        for j in range(6):
            ts = base + timedelta(days=j + 10, hours=i)
            history_rows.append(
                {
                    "order_id": f"H-INT-{i}-{j}",
                    "user_id": uid,
                    "driver_id": did,
                    "vendor_id": vid,
                    "device_id": dev,
                    "market": "SG",
                    "vertical": "food",
                    "amount": 45,
                    "is_refund": 1 if j >= 2 else 0,
                    "event_ts": ts.isoformat(),
                    "status": "delivered",
                    "claim_reason": "missing_item" if j >= 2 else "",
                }
            )
        order_rows.append(
            {
                "order_id": f"O-INT-{i}",
                "user_id": uid,
                "driver_id": did,
                "vendor_id": vid,
                "device_id": dev,
                "market": "SG",
                "vertical": "food",
                "amount": 45,
                "status": "delivered",
                "claim_reason": "missing_item",
                "event_ts": (base + timedelta(days=28, hours=i)).isoformat(),
                "abuse_label": 1,
                "abuse_label_weak": 0,
                "fraud_label": 0,
                "fraud_label_source": "",
                "strong_fraud_label": 0,
                "weak_policy_negative": 0,
                "customer_courier_same_device": 1,
                "claim_has_image": 1,
                "claim_image_ai_risk": 0.92,
                "claim_in_app_capture": 0,
                "pin_required": 1,
                "pin_verified": 0,
                "delivery_geofence_ok": 0,
            }
        )

    devices_df = pd.DataFrame(device_rows)
    for col, default in (
        ("device_risk_score", 0),
        ("is_emulator", 0),
        ("is_cloned_app", 0),
        ("is_gps_spoof", 0),
        ("is_tampered", 0),
    ):
        if col not in devices_df.columns:
            devices_df[col] = default
        devices_df[col] = devices_df[col].fillna(default)
    # Proxy farm devices look like high-risk / cloned.
    farm = devices_df["cluster_id"].astype(str).eq("CPROXY")
    devices_df.loc[farm, "device_risk_score"] = 80
    devices_df.loc[farm, "is_cloned_app"] = 1
    devices_df.loc[farm, "is_emulator"] = 1

    orders_df = pd.DataFrame(order_rows)
    _ets = pd.to_datetime(orders_df["event_ts"], utc=True)
    orders_df["delivered_ts"] = _ets
    _has_claim = orders_df["claim_reason"].astype(str).str.len().gt(0)
    orders_df["claim_ts"] = _ets + pd.to_timedelta(
        _has_claim.map({True: 6, False: 1}), unit="h"
    )
    for col, default in (
        ("customer_courier_same_device", 0),
        ("claim_has_image", 0),
        ("claim_image_ai_risk", 0.0),
        ("claim_in_app_capture", 0),
        ("pin_required", 0),
        ("pin_verified", 0),
        ("delivery_geofence_ok", 0),
        ("prior_strong_fraud", 0),
    ):
        if col not in orders_df.columns:
            orders_df[col] = default
        orders_df[col] = orders_df[col].fillna(default)
    # Clean deliveries: PIN verified + geofence ok; abuse claims often have external AI images.
    clean = orders_df["order_id"].astype(str).str.startswith("O-CLEAN-")
    orders_df.loc[clean, "pin_required"] = 1
    orders_df.loc[clean, "pin_verified"] = 1
    orders_df.loc[clean, "delivery_geofence_ok"] = 1
    abuse_like = orders_df["abuse_label"].astype(int) >= 1
    orders_df.loc[abuse_like & ~orders_df["order_id"].astype(str).str.startswith("O-INT-"), "claim_has_image"] = 1
    orders_df.loc[
        orders_df["order_id"].astype(str).str.startswith("O-BURN-"), "claim_image_ai_risk"
    ] = 0.85
    orders_df.loc[
        orders_df["order_id"].astype(str).str.startswith("O-BURN-"), "claim_in_app_capture"
    ] = 0

    # Closed-loop dispositions (subset). Timestamps honor lag_days (>= event + 7d).
    disposition_rows = [
        {
            "order_id": "O-FRAUD-0",
            "disposition": "investigator_confirmed_fraud",
            "disposition_ts": (base + timedelta(days=36)).isoformat(),
        },
        {
            "order_id": "O-WEAK-0",
            "disposition": "weak_policy_auto_grant",
            "disposition_ts": (base + timedelta(days=32)).isoformat(),
        },
        {
            "order_id": "O-BURN-0",
            "disposition": "manual_denied_abuse",
            "disposition_ts": (base + timedelta(days=35)).isoformat(),
        },
        {
            "order_id": "O-INT-0",
            "disposition": "manual_denied_fraud",
            "disposition_ts": (base + timedelta(days=36)).isoformat(),
        },
        {
            "order_id": "O-CLEAN-0",
            "disposition": "investigator_cleared",
            "disposition_ts": (base + timedelta(days=33)).isoformat(),
        },
        # Too early vs lag — should be skipped by ingest.
        {
            "order_id": "O-ABUSE-0",
            "disposition": "manual_denied_abuse",
            "disposition_ts": (base + timedelta(days=26, hours=2)).isoformat(),
        },
    ]

    pd.DataFrame(history_rows).to_csv(DATA / "history.csv", index=False)
    orders_df.to_csv(DATA / "orders.csv", index=False)
    devices_df.to_csv(DATA / "devices.csv", index=False)
    pd.DataFrame(user_rows).to_csv(DATA / "users.csv", index=False)
    pd.DataFrame(disposition_rows).to_csv(DATA / "dispositions.csv", index=False)
    print(
        f"Wrote {len(history_rows)} history, {len(orders_df)} orders, "
        f"{len(devices_df)} devices, {len(user_rows)} users, "
        f"{len(disposition_rows)} dispositions to {DATA}"
    )


if __name__ == "__main__":
    main()
