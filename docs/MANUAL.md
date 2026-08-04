# refund-abuse-risk — user & ops manual

How to **use** the repo, what you **need**, and how to **tune** the model.

Companion: [README.md](../README.md) · Live cutover: [CUTOVER.md](CUTOVER.md)

---

## 0. Minimum requirements

| Item | Minimum | Notes |
|---|---|---|
| Python | **3.11+** | `requires-python` in `pyproject.toml` |
| Packages | `pip install -e ".[dev]"` | pydantic, numpy, scikit-learn, pandas, pyyaml, joblib; pytest for tests |
| Hardware | 4 GB RAM / 2 CPU | Demo + small train. Large serve-path: 8–16 GB |
| Input data | `orders.csv`, `history.csv`, `devices.csv` | `users.csv` strongly recommended |
| Labels | At least some proven fraud | Via dispositions (`chargeback_lost`, `bank_dispute_lost`, investigator) |
| Optional | SDK events, ops snapshot | Improves features / promote gates |
| Not required | GPU, Redis, Postgres, HTTP server | This is a **library + scripts** toolkit |

Verify install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

---

## 1. How to use the repo (day-one path)

### 1.1 Demo end-to-end

```bash
python scripts/generate_demo_data.py
python scripts/generate_closed_loop_labels.py --apply --with-sdk
python scripts/train_model.py --feature-source serve --data-dir data \
  --max-rows 2000 --passes 3 --oot-days 7 \
  --model-out models/two_head_demo.joblib
python scripts/backtest.py
python -m examples.csv_demo
# → examples/csv_demo/out.json
```

### 1.2 Production-shaped loop

```text
1. Pull feeds     → dispositions / SDK / ops   (pull_production_feeds.py)
2. Train          → serve-path + closed-loop   (train_model.py)
3. Evaluate       → time-OOT + oot_pack floors (backtest / eval_oot_pack)
4. Tune           → thresholds / overlays      (backtest --tune / run_tuner)
5. Promote        → only if promote_ok         (HIL if Δ > max_abs_delta)
6. Serve          → precompute → claim_path_read
```

```bash
# Feeds (fixtures or sqlite warehouse)
python scripts/seed_feed_fixtures.py --with-warehouse
python scripts/pull_production_feeds.py --config config/feeds.warehouse.yaml

# Train (prefer labeled + SDK)
python scripts/train_model.py --feature-source serve --data-dir data \
  --max-rows 5000 --passes 4 --oot-days 7

# Honesty pack
python scripts/seed_oot_pack.py
python scripts/eval_oot_pack.py   # exit 1 if floors fail

# Thresholds
python scripts/backtest.py --tune
python scripts/backtest.py --write-slice-overlays examples/csv_demo/slice_overlays.yaml
python scripts/run_tuner.py
python scripts/run_tuner.py --list-pending
# python scripts/run_tuner.py --approve <id>
```

### 1.3 Integrate into your platform (library)

```python
from refund_abuse_risk.pipeline.score import (
    train_two_head,
    precompute_orders,
    refresh_order,
    on_entity_risk_change,
    claim_path_read,
)

model = train_two_head(train_orders, history, devices, users=users)
cache = precompute_orders(orders, history, devices, model, users=users)

refresh_order(order, history, devices, model, cache, users=users, event="delivered")
# Optional SDK at claim:
# refresh_order(..., device_sdk_event=..., vision_sdk_event=...)

snap = claim_path_read(order_id, cache)  # sync claim path — cache only
```

**Claim-path contract:** score async on lifecycle / entity-risk change; claim API only reads the snapshot. Do not walk the graph on the sync path.

### 1.4 When to refresh scores

| Trigger | Action |
|---|---|
| `placed` → `delivered` → `claim` | `refresh_order` |
| Linked entity score moves ≥ `entity_risk_change_delta` (default 10) | `on_entity_risk_change` |

---

## 2. What the model outputs

| Field | Role |
|---|---|
| `abuse_score` / `fraud_score` | Head evidence (0–100) |
| `decision_score` | **Primary** tier input (`decision_primary`) |
| `suggested_tier` | `auto_approve` → `soft_friction` → `hold_review` → `auto_deny` |
| `refund_effect` | UX hint: grant / step-up / manual review / block |
| `hard_gated` | Only `STRONG_FRAUD_LABEL` / prior strong fraud |
| `reason_codes` / `evidence_pack` | Ops explainability |
| `model_version` / `policy_version` | Audit provenance |

Tiers and effects are **advisory**. Downstream owns enforcement.

---

## 3. Decision logic (current default)

With `decision_mode: decision_primary` (default in `operating_point.default.yaml`):

1. Heads → OOF-trained stacker → `decision_score`.  
2. Optional `SliceCalibrator` adjusts by market×vertical.  
3. Tier from `decision_thresholds` (global) or first matching `decision_threshold_overlays`.  
4. Effect rules (`effect_rules.default.yaml`) may override `refund_effect` (shadow or live); `kill_switch` disables overrides.  
5. Budget floor may raise effect severity when account pressure is high.

| Tier | Typical meaning |
|---|---|
| `auto_approve` | Low friction |
| `soft_friction` | Extra proof (photo / PIN / cap) |
| `hold_review` | CS queue |
| `auto_deny` | No auto refund path |

Legacy `learning_primary` (OR of head thresholds) and `score_only` still exist; prefer `decision_primary`.

---

## 4. How to tune the model

Tune in this order. Change **one family** per cycle. Always re-run backtest / OOT before promote.

### 4.1 What needs retrain vs not

| Change | Retrain? |
|---|---|
| `decision_thresholds` / overlays | **No** |
| Effect rules / budget / kill switch | **No** |
| Monitoring ceilings (`max_decision_ece`, `$` FP, ops) | **No** |
| `label_weights` / proxy rules / discovery weights | **Yes** |
| Feature code / head hyperparams | **Yes** |
| New markets with different base rates | Yes (or overlays first) |

### 4.2 Threshold tuning (no retrain) — primary path

Edit `config/operating_point.default.yaml`:

| Knob | If you raise it | If you lower it |
|---|---|---|
| `decision_thresholds.soft_friction` | Fewer soft flags | Catch more pattern mass |
| `decision_thresholds.hold_review` | Fewer holds | More CS volume |
| `decision_thresholds.auto_deny` | Fewer auto-denies | More aggressive block |
| `head_thresholds.min_precision_at_soft` | Harder to promote soft | Easier promote (riskier) |
| `head_thresholds.max_fp_refund_dollars_mean` | Caps mean FP $ at soft | `null` = ungated |
| `decision_threshold_overlays` | Per market×vertical ladder | Empty = global only |

Automated path:

```bash
# Propose costed decision ladder + slice overlays
python scripts/backtest.py --tune
python scripts/backtest.py --write-slice-overlays path/to/overlays.yaml

# Auto-step within guardrails; queue HIL for large jumps
python scripts/run_tuner.py
python scripts/run_tuner.py --list-pending
python scripts/run_tuner.py --approve <id>   # or --reject <id>
```

Guardrails (`config/policy_guardrails.default.yaml`):

- `auto_apply.max_abs_delta` — max auto step (default 5 score points)  
- Deny keys often capped tighter (`max_abs_delta_by_key`)  
- Outside `bounds.*` → hard reject (not HIL)

**Promote only when** backtest reports `honesty.promote_ok` (costed `recommended.ok` ∧ `monitoring.ok` including slice ECE / ops / `$` gates).

### 4.3 Practical sequences

**Recall too low at soft**

1. `backtest.py --tune` — see recommended soft.  
2. If already near floor and recall still low → **retrain** (labels/features), don’t invent hard gates.  
3. Check proven vs proxy/discovery slices separately.

**Too many soft/holds on clean users**

1. Raise `decision_thresholds.soft_friction` / `hold_review` slightly.  
2. Inspect FPs — tighten proxy rules + retrain if proxies leak.  
3. Set / tighten `max_fp_refund_dollars_mean`.  
4. Do **not** reintroduce rate hard gates.

**One market/vertical miscalibrated**

1. Check `backtest` → `slices` / `recommended_overlays`.  
2. Write promote-eligible overlays (`--write-slice-overlays`).  
3. Adjust `vertical_policy.default.yaml` priors only if claim-window economics differ.

**Fraud head trusts proxies too much**

1. Lower `fraud_proxy_weight` / `fraud_discovery_weight` in `label_weights`.  
2. Retrain; evaluate **proven-only** AP as primary.

### 4.4 Retrain recipe (weights / features changed)

```bash
# Prefer closed-loop labeled orders + SDK
python scripts/generate_closed_loop_labels.py --apply --with-sdk   # or live feeds
python scripts/train_model.py --feature-source serve --data-dir data \
  --max-rows 5000 --passes 4 --oot-days 7 \
  --model-out models/two_head.joblib

# Primary metrics: OOT proven PR-AUC / precision@soft (in train summary + backtest)
python scripts/backtest.py
python scripts/eval_oot_pack.py
```

Important flags:

| Flag | Meaning |
|---|---|
| `--feature-source serve` | **Default.** Train≈serve (required for honesty) |
| `--feature-source frame` | Precomputed frame — demo only, not parity |
| `--no-closed-loop` | Skip labeled/SDK overlays (debug) |
| `--passes N` | Multipass UV→supervised (early-stops on flat decision mean) |
| `--oot-days 7` | Time holdout for primary metrics |

Bump `model_version` in operating point when promoting a new artifact.

### 4.5 Label quality (feeds the tuner)

| Source | Role |
|---|---|
| `chargeback_lost` / `bank_dispute_lost` / investigator confirmed | **Proven** hard truth |
| Proxy / discovery | Down-weighted; never overwrite proven |
| `weak_policy_auto_grant` | Weak negative (`weak_policy_negative=1`) |
| Disposition lag | Default `lag_days: 7` — same-day investigator labels don’t mint |

```bash
python scripts/ingest_dispositions.py --orders data/orders.csv --dispositions data/dispositions.csv
python scripts/ingest_sdk_signals.py --orders data/orders.csv --events data/sdk_events.jsonl
python scripts/pull_production_feeds.py --config config/feeds.warehouse.yaml
```

---

## 5. Config map (ops)

```text
config/
  operating_point.default.yaml   # decision ladder, overlays, monitoring, $ FP
  policy_guardrails.default.yaml # bounds + max_abs_delta / HIL
  policy.default.yaml            # strong_fraud hard gate + evidence weights
  label_weights.default.yaml     # train weights (retrain)
  vertical_policy.default.yaml   # claim-window / photo / cash priors
  head_hyperparams.default.yaml  # per-head model knobs (retrain)
  effect_rules.default.yaml      # shadow→live effects + kill_switch
  refund_budget.default.yaml     # advisory budget floor
  sdk_ingest.default.yaml        # SDK confidence gate
  disposition_labels.default.yaml# disposition → label mapping + lag
  feeds.default.yaml             # local_dir fixtures
  feeds.warehouse.yaml           # sqlite warehouse queries
  oot_floors.default.yaml        # labeled OOT pack floors
```

---

## 6. Input schema checklist

**orders / history**

- Ids: `order_id`, `user_id`, `driver_id`, `vendor_id`, `device_id`  
- Economics: `amount`; history `is_refund`  
- Time: `event_ts` (UTC); prefer `delivered_ts`, `claim_ts`  
- Context: `market`, `vertical` (`food` / `qcommerce` / `grocery`), `status`, `claim_reason`  
- Optional risk: device integrity, claim image, PIN, geofence  

**Point-in-time:** features use history with `event_ts ≤ order.event_ts` and exclude the scored `order_id`.

---

## 7. Monitoring & promote

| Gate | Where |
|---|---|
| Costed soft (`min_precision_at_soft`, optional `$` FP) | `recommended.ok` |
| ECE / PSI | `monitoring` in operating point |
| Ops (hold / override / refund$ / CS queue) | `ops_snapshot` + ceilings |
| Slice ECE (non-thin market×vertical) | backtest `monitoring.slices` |
| OOT pack floors | `eval_oot_pack.py` |

Promote checklist:

1. Diff YAML / model artifact.  
2. `backtest.py` — `honesty.promote_ok` true.  
3. `eval_oot_pack.py` green on your pack (strict floors for prod).  
4. Shadow one market; watch tier mix + proven precision.  
5. Bump `policy_version` / `model_version`; keep previous artifact for rollback.

---

## 8. Runbooks

### 8.1 Spike in false denies

1. Check `hard_gated` (should track strong-label rate only).  
2. If score-driven: raise `decision_thresholds.auto_deny` / `hold_review`.  
3. Sample FPs for proxy leakage — fix labels + retrain if needed.  
4. Do not add rate hard gates.

### 8.2 Collusion ring not caught

1. Confirm device cluster / UVD / SDK columns present.  
2. If scores high but tier approve → thresholds — `--tune` or lower soft/hold.  
3. If scores low on a recurring shape → label (chargeback/bank/investigator) + retrain.  
4. Confirm lifecycle + entity-risk refresh is running.

### 8.3 Claim path empty / stale

1. Confirm precompute wrote that `order_id`.  
2. Check lifecycle hooks for `delivered` / `claim`.  
3. Do **not** add live graph walks on the sync path.

### 8.4 Rollback

1. Restore previous `operating_point.default.yaml` (and effect rules if changed).  
2. Reload previous `*.joblib` + matching `model_version`.  
3. Rescore open orders if bad scores were written.

---

## 9. Security & versioning

- Treat snapshots / evidence as sensitive fraud data.  
- Restrict write access to proven fraud labels.  
- Log `model_version` + `policy_version` on every Downstream enforcement decision.

| Field | Bump when |
|---|---|
| `model_version` | New weights / feature set |
| `policy_version` | Threshold / safety / effect changes |

---

## 10. Script cheat sheet

| Script | Purpose |
|---|---|
| `generate_demo_data.py` | Small synthetic CSVs |
| `generate_large_demo_data.py` | Large demo + optional feature_frame |
| `generate_closed_loop_labels.py` | Dispositions / chargeback / bank dispute / QA / SDK |
| `train_model.py` | Serve-path multipass train + time-OOT |
| `refresh_train_bundle.py` | Generate closed-loop → train one-shot |
| `backtest.py` | Holdout metrics, `--tune`, `--write-slice-overlays` |
| `run_tuner.py` | Auto-step thresholds + HIL approve/reject |
| `pull_production_feeds.py` | Pull→stage→apply feeds |
| `seed_feed_fixtures.py` | Fixture + sqlite warehouse seed |
| `seed_oot_pack.py` / `eval_oot_pack.py` | Labeled OOT pack + floors |
| `ingest_dispositions.py` / `ingest_sdk_signals.py` / `ingest_ops_snapshot.py` | Single-feed ingest |

Design history: [`docs/superpowers/specs/`](superpowers/specs/).
