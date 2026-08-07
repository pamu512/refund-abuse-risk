# refund-abuse-risk

Graph/ML toolkit for **refund abuse** and **fraud** scoring on food delivery and quick commerce.

It turns order, history, device, and closed-loop label feeds into:

- calibrated `abuse_score` / `fraud_score` plus joint **`decision_score`**
- entity / link / device standing scores
- a **suggested** tier + `refund_effect` (advisory — Downstream owns enforcement)
- reason codes + evidence pack for ops / investigators

**Design goal:** catch recurring behavioral patterns at high recall on labeled pattern mass. Do not chase novel edge cases with hard-coded rules.

**Status (2026-08-06):** In-repo toolkit with prod-shaped OOT contracts, overlay promote/rollback, HTTP/S3 feed drivers, `ops.overnight.prod.yaml`, and minimal `serve_api.py`. Live warehouse URIs / real chargebacks remain Downstream — [`docs/CUTOVER.md`](docs/CUTOVER.md).

**Internal quality ratings are private** — do not publish letter grades or rubric scores in README/status/comms. See [`docs/GRADING.md`](docs/GRADING.md).

---

## Docs

| Doc | Use when |
|---|---|
| **[Ops manual](docs/MANUAL.md)** | How to run, minimum requirements, how to tune |
| **[Architecture](docs/ARCHITECTURE.md)** | Package map, train/serve paths, extension points |
| **[Ops runbook](docs/OPS_RUNBOOK.md)** | Overnight profile, promote tree, health checks |
| [CUTOVER.md](docs/CUTOVER.md) | Point feeds at live DBs / prod OOT packs |
| [GRADING.md](docs/GRADING.md) | Privacy policy for internal quality ratings |
| [Design spec](docs/superpowers/specs/2026-08-02-refund-abuse-risk-design.md) | Product/design decisions |

---

## Minimum requirements

| Need | Spec |
|---|---|
| **Python** | 3.11+ |
| **OS** | macOS / Linux (Windows via WSL ok) |
| **RAM** | ~4 GB for demo; 8–16 GB for large serve-path train |
| **Disk** | ~1 GB demo; tens of GB if you keep large history + models |
| **Core CSVs** | `orders`, `history`, `devices` (+ optional `users`) |
| **Labels** | Prefer proven fraud via dispositions; proxies/discovery down-weighted |
| **Not required** | GPU, Redis, Postgres, live HTTP API (library + scripts) |

Install:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                   # sanity: 100+ tests
```

---

## Quickstart (demo → train → tune)

```bash
# 1) Synthetic data
python scripts/generate_demo_data.py

# 2) Closed-loop labels + SDK synth (optional but recommended)
python scripts/generate_closed_loop_labels.py --apply --with-sdk

# 3) Serve-path train (train≈serve) + time-OOT
python scripts/train_model.py --feature-source serve --data-dir data \
  --max-rows 2000 --passes 3 --oot-days 7

# 4) Backtest + threshold proposal
python scripts/backtest.py
python scripts/backtest.py --tune

# 5) Auto-step thresholds (HIL if jump > max_abs_delta)
python scripts/run_tuner.py
python scripts/run_tuner.py --list-pending

# 6) Feeds / OOT honesty
python scripts/seed_feed_fixtures.py --with-warehouse
python scripts/pull_production_feeds.py --config config/feeds.warehouse.yaml --dry-run
python scripts/seed_oot_pack.py && python scripts/eval_oot_pack.py

# 7) Claim-path demo
python -m examples.csv_demo
# → examples/csv_demo/out.json
```

Full walkthrough, knobs, and promote rules: **[docs/MANUAL.md](docs/MANUAL.md)**.

---

## Abuse vs fraud

| Term | Meaning |
|---|---|
| **Abuse** | Behavioral / policy gaming (serial claimants, refund vs LTV, weak-policy loopholes). |
| **Fraud** | Investigator / chargeback / bank-dispute **proven** truth, plus down-weighted proxies / discovery. |

Weak-policy auto-grants are **not** clean negatives (`weak_policy_negative=1`).

---

## How scoring works

```text
feeds → orders (+ labeled / SDK) + history + devices
        │
        ▼
 serve-path features (PIT / as-of) + policy priors (market×vertical)
        │
        ▼
 two heads → OOF decision stacker → decision_score
        │
        ▼
 decision_thresholds (+ optional market×vertical overlays)
        │
        ▼
 suggested_tier → effect rules / budget → OrderRiskSnapshot
        │
        ▼
 claim path reads precomputed snapshot only
```

Default `decision_mode: decision_primary` — tiers cut on **`decision_score`**. Heads stay as evidence.

### Data you plug in

| Feed | Core fields |
|---|---|
| **orders** | `order_id`, entity ids, `market`, `vertical`, `amount`, `status`, `event_ts`; prefer `delivered_ts` / `claim_ts` |
| **history** | Past orders: `is_refund`, `amount`, `event_ts`, same entity ids |
| **devices** | `user_id`, `device_id`, `cluster_id`, `last_seen_ts` (+ optional integrity flags) |
| **users** | `user_id`, `signup_ts` |
| **dispositions** | `order_id`, `disposition`, `disposition_ts` → `orders.labeled.csv` |
| **sdk events** | JSONL envelopes → device/vision columns (confidence-gated) |
| **ops snapshot** | CS / hold / override / refund$ metrics for promote gates |

Labels: `abuse_label`, `fraud_label`, `fraud_label_source` (`proven` / `proxy` / `discovery`), `strong_fraud_label`, `weak_policy_negative`.

---

## Configuration (ops start here)

| File | Purpose |
|---|---|
| [`config/operating_point.default.yaml`](config/operating_point.default.yaml) | `decision_thresholds`, overlays, head knobs, monitoring / `$` FP ceilings |
| [`config/policy_guardrails.default.yaml`](config/policy_guardrails.default.yaml) | Bounds + `max_abs_delta` (HIL if larger) |
| [`config/label_weights.default.yaml`](config/label_weights.default.yaml) | Proven/proxy/discovery/weak weights (**retrain**) |
| [`config/vertical_policy.default.yaml`](config/vertical_policy.default.yaml) | Claim-window / photo / cash priors by vertical |
| [`config/feeds.warehouse.yaml`](config/feeds.warehouse.yaml) | Sqlite warehouse pull queries |
| [`config/oot_floors.default.yaml`](config/oot_floors.default.yaml) | Labeled OOT pack honesty floors |
| [`config/effect_rules.default.yaml`](config/effect_rules.default.yaml) | Shadow→live effect overrides + kill switch |

**Threshold / overlay YAML changes do not require retrain.** Feature code, label weights, and head hyperparams do.

---

## Package layout

| Path | Role |
|---|---|
| `src/refund_abuse_risk/features/` | PIT features + policy priors |
| `src/refund_abuse_risk/model/` | Two-head model + stacker |
| `src/refund_abuse_risk/scoring/` | Decision tiers, monitoring, budgets |
| `src/refund_abuse_risk/training/` | Serve-path frame, multipass, closed-loop load, OOT split |
| `src/refund_abuse_risk/integrations/` | SDK / ops / feeds runner |
| `src/refund_abuse_risk/pipeline/` | Precompute cache, lifecycle refresh, claim read |
| `scripts/` | Train, backtest, tuner, feeds, OOT packs, demo data |
| `config/` | All operating knobs |

---

## What this is not

- Not a payment blocker or account-suspension system  
- Not a live GNN / graph database  
- Not a case-management UI  
- Not a legal “fraud proven” workflow (it **consumes** those labels)  
- Not an attempt to catch every novel attack with YAML rules
