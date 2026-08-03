# refund-abuse-risk

Graph/ML toolkit for **refund abuse** and **fraud** scoring on food delivery and quick commerce platforms.

It turns order, device, entity-graph, and user tenure/LTV signals into:

- two calibrated scores (`abuse_score`, `fraud_score`)
- entity / link / device standing scores
- a **suggested** four-tier refund decision from **head thresholds**
- reason codes + an evidence pack for ops and investigators

**Design goal:** learn recurring behavioral patterns at ~**98%** recall on labeled pattern mass. Do not chase novel 2% edge cases with hard-coded rules. Downstream systems own enforcement.

---

## Abuse vs fraud

| Term | Meaning |
|---|---|
| **Abuse** | Behavioral / policy gaming (serial claimants, high refund rate vs LTV, weak-policy loopholes). Often not legally “proven”. |
| **Fraud** | Proveable collusion or bad behavior (investigator-confirmed), plus high-confidence **proxy** labels at reduced sample weight. |

The model never treats “refund approved under weak policy” as a clean negative without down-weighting.

---

## How it works

```text
orders + history + devices + users
        │
        ▼
 feature builders  ──► rolling rates, LTV/tenure, device cluster,
        │               UD/UV/VD/UVD link lift + related-ring aggregates
        ▼
 two-head model    ──► abuse_score + fraud_score (0–100)
        │
        ▼
 head thresholds   ──► soft / hold / deny from calibrated heads
        │               (only STRONG_FRAUD_LABEL hard-gates)
        ▼
 order snapshot cache  ◄── refresh on lifecycle events / entity risk change
        │
        ▼
 claim path (sync) ──► read precomputed snapshot only
```

### Scoring layers

1. **Features (point-in-time)**  
   History at/before the order timestamp; the scored order is excluded. Rates and link lifts are support-shrunk so 1–2 orders cannot look like serial abuse by themselves.

2. **Two-head ML**  
   - Abuse head: rates, GMV%, tenure/LTV, claim reasons, related/combined refunds.  
   - Fraud head: device/cluster, link lift/share, related-ring aggregates.  
   Fraud proxies need device farm + UVD concentration + support — not rate alone.

3. **Learning-primary tiers**  
   Suggested tier comes from `abuse_score` / `fraud_score` vs `head_thresholds` in `operating_point.default.yaml` (OR logic). Soft thresholds target ~98% recall on labeled abuse|fraud patterns. Entity prior is evidence only — it does **not** band or override heads.

4. **Hard gates (safety only)**  
   `STRONG_FRAUD_LABEL` forces `auto_deny`. Rate/LTV/ring caps are **features**, not hard decision rules.

5. **Suggested tier**  
   `auto_approve` → `soft_friction` → `hold_review` → `auto_deny` (advisory).

### Data you plug in

| Feed | Required fields (core) |
|---|---|
| **orders** | `order_id`, `user_id`, `driver_id`, `vendor_id`, `device_id`, `market`, `vertical`, `amount`, `status`, `event_ts`, optional claim/labels |
| **history** | Past orders with `is_refund`, `amount`, `event_ts`, same entity ids |
| **devices** | `user_id`, `device_id`, `cluster_id`, `last_seen_ts` |
| **users** | `user_id`, `signup_ts` (falls back to first history event if missing) |

Labels for training: `abuse_label`, `fraud_label`, `fraud_label_source` (`proven` / `proxy`), `strong_fraud_label`, `weak_policy_negative`, `abuse_label_weak`.

### Claim-path contract

Most claims are async (order already in flight or delivered). Prefetch/score on lifecycle events (`placed` → `picked_up` → `out_for_delivery` → `delivered` → `claim`) and when linked entity risk jumps. The claim API should only **read** the cached snapshot — no live multi-hop graph walk. Latency is not the modeling objective.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_demo_data.py
pytest -q
python -m examples.csv_demo
python scripts/backtest.py
python scripts/backtest.py --tune              # propose thresholds (HIL-aware)
python scripts/run_tuner.py                    # auto-step ≤ max_abs_delta; queue HIL if larger
python scripts/run_tuner.py --list-pending
python scripts/run_tuner.py --approve <id>     # human approves full move
```

Demo prints sample snapshots; JSON lands in `examples/csv_demo/out.json`.

---

## Configuration (start here for ops)

| File | Purpose |
|---|---|
| [`config/operating_point.default.yaml`](config/operating_point.default.yaml) | **Primary knobs:** `head_thresholds`, display blend, decision mode |
| [`config/policy_guardrails.default.yaml`](config/policy_guardrails.default.yaml) | Absolute bounds + `auto_apply.max_abs_delta` (HIL if larger) |
| [`config/policy.default.yaml`](config/policy.default.yaml) | Safety hard gate (`strong_fraud_label`) + evidence weights |
| [`config/label_weights.default.yaml`](config/label_weights.default.yaml) | Train-time proven/proxy weights and proxy minting rules |
| [`config/behavior_baselines.default.yaml`](config/behavior_baselines.default.yaml) | Entity/pair/device/cohort baselines + precision discount / trust credit |

**ML adjusts thresholds automatically within `max_abs_delta`; larger jumps need HIL approval** (`scripts/run_tuner.py --approve`). Head-threshold / policy YAML changes do not require retraining. Model / label-weight changes do.

### Behavior baselines (warehouse)

Head-score baselines (`abuse` for user/uv, `fraud` for driver/vendor/device/links) update on settled events only (idempotent). Clean actors who spike get a **head-attributed** precision discount (capped; larger relax needs HIL). **Cohort** peers (market × vertical × tenure) elevate even without a clean self-baseline. Clean user+device + low scores → **trust credit** (tighten thresholds). Lifts feed the model as features.

```bash
python scripts/export_behavior_baselines.py
# → data/warehouse/entity_behavior_baselines/as_of_date=YYYY-MM-DD/part.csv
# → data/warehouse/cohort_behavior_baselines/as_of_date=YYYY-MM-DD/part.csv

python scripts/run_bipartite_anomaly.py
# → data/warehouse/bipartite_uv_edges|nodes/as_of_date=YYYY-MM-DD/part.csv
```

Snapshots include `refund_effect` (`refund_auto_grant` → `refund_step_up` → `refund_manual_review` → `refund_block`) mapped from `suggested_tier` for downstream refund UX. Offline UV bipartite anomaly feeds `uv_edge_anomaly` / `*_bipartite_anomaly` features.

---

## Package layout

| Path | Role |
|---|---|
| `src/refund_abuse_risk/features/` | Feature builders (entity, link, device, LTV, related rings) |
| `src/refund_abuse_risk/model/` | Two-head trainer + standing score helpers |
| `src/refund_abuse_risk/scoring/` | Head tiers, safety gates, threshold tuner, evidence pack |
| `src/refund_abuse_risk/pipeline/` | Precompute cache, lifecycle / risk-change refresh, claim read |
| `src/refund_abuse_risk/schemas/` | Pydantic output schemas |
| `scripts/generate_demo_data.py` | Synthetic CSV generator |
| `scripts/backtest.py` | Holdout metrics + optional `--tune` for 98% recall |
| `examples/csv_demo/` | End-to-end offline demo |

---

## Documentation

| Doc | Audience |
|---|---|
| **[Ops & tuning manual](docs/MANUAL.md)** | Tuning head thresholds, claim path, retraining, monitoring |
| [Design spec](docs/superpowers/specs/2026-08-02-refund-abuse-risk-design.md) | Product/design decisions |

---

## What this is not

- Not a payment blocker or account-suspension system  
- Not a GNN / live graph database  
- Not a case-management UI  
- Not a legal “fraud proven” workflow (it **consumes** those labels)  
- Not an attempt to catch every novel 2% attack with YAML rules
