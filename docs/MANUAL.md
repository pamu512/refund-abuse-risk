# refund-abuse-risk — operations, tuning & maintenance manual

This manual is for fraud/risk ops, data scientists, and platform engineers who integrate, tune, or maintain the scorer.

Companion overview: [README.md](../README.md).

**Product goal:** a learning detector for recurring behavioral patterns (~**98%** recall on labeled abuse|fraud mass). Do not optimize for novel 2% attacks or sync latency.

---

## 1. Concepts

### 1.1 Outputs

Each scored order produces an `OrderRiskSnapshot`:

| Field | Description |
|---|---|
| `abuse_score` / `fraud_score` | Calibrated 0–100 heads (evidence / diagnostics) |
| `decision_score` | Joint stacker score — **primary tier input** under `decision_primary` |
| `entity_scores` | Standing user / driver / vendor risk |
| `link_scores` | UD / UV / VD / UVD standing risk |
| `device_cluster_score` | Shared-device / farm risk |
| `entity_prior` | Standing max-style prior for **evidence/monitoring only** |
| `combined_score` | Display blend of heads (not banded by prior) |
| `suggested_tier` | `auto_approve` \| `soft_friction` \| `hold_review` \| `auto_deny` |
| `refund_effect` | Progressive refund UX hint (tier + live effect rules) |
| `shadow_refund_effect` | Suggested override when rule mode is `shadow` |
| `effect_decision` | Matched rule id/mode, kill-switch flag, audit fields |
| `refund_budget` | Advisory account budget pressure (`ok` / `elevated` / `exhausted`) |
| `hard_gated` | `true` only for safety override (`STRONG_FRAUD_LABEL`) |
| `reason_codes` | Short codes for UI / routing |
| `evidence_pack` | Metric values, thresholds, entity ids, contributions |
| `model_version` / `policy_version` | Provenance for audits |

**Suggested tier / refund_effect are advisory.** Your refund/policy service decides what to enforce.

| `suggested_tier` | `refund_effect` | Downstream meaning |
|---|---|---|
| `auto_approve` | `refund_auto_grant` | Low-friction / auto refund |
| `soft_friction` | `refund_step_up` | Extra proof (photo, PIN, capped amount) |
| `hold_review` | `refund_manual_review` | CS queue |
| `auto_deny` | `refund_block` | No auto path |

### 1.1f Platform gates (Phase 3 → A++)

| Gate | Behavior |
|---|---|
| Device / vision adapters | Nested payloads + SDK ingest envelopes normalize to feature columns |
| SDK ingest | `ingest_sdk_signals.py` / `attach_sdk_signals` / `refresh_order(..._sdk_event=)` with `min_confidence` |
| Effect rules | `config/effect_rules.default.yaml` — ordered shadow→live overrides + global `kill_switch` |
| Refund budget | Pressure on snapshot; with `apply_as_effect_floor` floors `refund_effect` up when stricter |
| Discovery → weak labels | As-of UV edge mint (`event_ts`); `fraud_discovery_weight`; never overwrites proven; proxy cannot overwrite discovery |
| Monitoring | ECE + PSI under `monitoring`; promote requires `monitoring.ok` ∧ costed `recommended.ok` |
| Dual-run effects | Backtest `effects_dual_run`: base vs final vs shadow rates + budget floors |

### 1.1e Meaning gates (Phase 2 → A+)

| Gate | Behavior |
|---|---|
| `decision_primary` | Tiers from `decision_thresholds` on stacked `decision_score` |
| Slice ladders | Optional `decision_threshold_overlays` (market×vertical, first match) |
| OOF stacker | Joint score fit on out-of-fold head probs; serve uses full-data heads |
| Heads | Abuse/fraud remain for evidence + baseline under_threshold |
| Slice eval | Backtest reports market×vertical metrics + recommended slice ladders |
| Disposition lag | Labels only if `disposition_ts ≥ event_ts + lag_days` (default 7) |

### 1.1d Honesty gates (Phase 1 → A+)

| Gate | Behavior |
|---|---|
| As-of bipartite | UV anomaly uses only `event_ts ≤ order`; LOO market base rate |
| Proxy ≠ train features | Mint rules may label; fraud head excludes mint columns by default |
| Costed soft tune | Soft under recall **and** `min_precision_at_soft`; `recommended.ok` required to promote |
| Soft floors | Floors that rewrite soft → fail (hold/deny floor binds are warnings) |
| Baseline/cohort | Gate/evidence only — not head features |
| Hard gate | `prior_strong_fraud` only (same-order `strong_fraud_label` stripped at score) |

### 1.1c Platform risk feeds (device / claim / delivery)

Optional columns close the biggest detection gaps vs live platforms:

| Feed | Columns |
|---|---|
| devices | `device_risk_score`, `is_emulator`, `is_cloned_app`, `is_gps_spoof`, `is_tampered` |
| orders | `customer_courier_same_device`, `claim_has_image`, `claim_image_ai_risk`, `claim_in_app_capture`, `pin_required`, `pin_verified`, `delivery_geofence_ok` |

Vendor payloads may also be nested on the order as `device_intelligence` / `device_payload` and `claim_vision` / `vision_payload` — adapters in `integrations/device_vision.py` flatten them before feature build.

**SDK ingest** (batch or claim-path) accepts vendor-agnostic envelopes and joins them onto orders with confidence gating (`config/sdk_ingest.default.yaml`):

```bash
python scripts/ingest_sdk_signals.py --orders data/orders.csv --events data/sdk_events.jsonl
# → data/orders.sdk.csv
```

Envelope fields: `order_id`, `source` (`device` / `vision` or vendor aliases like `fingerprint` / `shield` / `incognia`), `payload`, optional `confidence` (0–1), `event_ts`. Claim-path can pass `device_sdk_event` / `vision_sdk_event` into `refresh_order`.

Missing columns default to 0. Integrity signals can mint **proxy fraud** labels (OR with classic device-farm proxy). Disposition feedback:

```bash
python scripts/ingest_dispositions.py --orders data/orders.csv --dispositions data/dispositions.csv
# → data/orders.labeled.csv
```

### 1.1b Offline UV bipartite anomaly

Batch job scores **user↔vendor** edges from history (lift vs market×vertical base rate × support). Elevated edges/nodes export to warehouse and join as features (`uv_edge_anomaly`, `user_bipartite_anomaly`, `vendor_bipartite_anomaly`). Not on claim-path latency.

```bash
python scripts/run_bipartite_anomaly.py --history data/history.csv
python scripts/run_bipartite_anomaly.py --history data/history.csv \
  --mint-weak-labels data/orders.csv
# → data/orders.discovery.csv
```

### 1.2 Abuse vs fraud heads

| Head | Train target | Typical drivers |
|---|---|---|
| Abuse | Broad behavioral labels (+ weak positives) | Refund rate/GMV%, LTV burn, reason repetition, tenure |
| Fraud | Proven labels + proxy labels (weight &lt; 1) | Device farms, UVD lift/share, related-ring aggregates |

Proxy fraud labels are minted only when **support + concentration** rules fire (see §4.3). They are not the same as proven fraud.

### 1.3 Related accounts (features, not gates)

Related accounts feed **features** the model learns from:

1. Users on the same `device_id` or device `cluster_id`
2. Users who share the same **driver and vendor** as the scored order (UVD triad peers)

Combined metrics = this user’s 30d refunds **plus** related users’ 30d refunds. Rings/LTV burn are learned patterns, not YAML hard gates.

### 1.4 Behavior baselines (entity / pair / combo / device / cohort)

Baselines track **head scores** (same 0–100 space as tier thresholds), not standing heuristics:

| Kind | Head |
|---|---|
| `user`, `uv` | `abuse_score` vs `abuse_soft_friction` |
| `driver`, `vendor`, `device`, `ud`, `vd`, `uvd` | `fraud_score` vs `fraud_soft_friction` |

Writes only on **settled** statuses (`delivered` / `claim` / …), **idempotent** per `order_id × entity`. Rolling window = `lookback_days` ∩ `lookback_events`. Clean is sticky but **revoked** if under_rate collapses. Pairs need `pair_min_cooccur`.

| State | Meaning |
|---|---|
| **Clean baseline** | Support + under_rate qualify (revocable) |
| **Elevated (self)** | Above own baseline for `min_elevated_streak` (needs clean) |
| **Elevated (cohort)** | Above market×vertical×tenure peer p50 — no clean self required (`COHORT_ELEVATED_*`) |
| **Precision discount** | **Head-attributed**: abuse kinds relax abuse thresholds only; fraud kinds relax fraud only |
| **Trust credit** | Clean user + clean device + low scores → tighten thresholds (`BASELINE_TRUST_CREDIT`) |
| **HIL relax** | Auto relax capped at `max_auto_relax_points`; remainder → `BASELINE_HIL_RELAX` |

Baseline lifts/streaks/cohort lifts are also **model features**. Warehouse daily snapshot:

```bash
python scripts/export_behavior_baselines.py
# data/warehouse/entity_behavior_baselines/as_of_date=YYYY-MM-DD/part.csv
# data/warehouse/cohort_behavior_baselines/as_of_date=YYYY-MM-DD/part.csv
```

---

## 2. End-to-end use

### 2.1 Offline / batch (demo)

```bash
pip install -e ".[dev]"
python scripts/generate_demo_data.py   # writes data/*.csv
python -m examples.csv_demo            # train → precompute → claim-path read
python scripts/backtest.py             # holdout + pattern-recall metrics
python scripts/backtest.py --tune      # propose + HIL-aware decision
python scripts/run_tuner.py            # auto-step; queue HIL if delta too large
pytest -q
```

### 2.2 Integrating with your platform

Recommended production shape:

1. **Async feature + score job** on order lifecycle events and entity-risk changes.  
2. **Write** `OrderRiskSnapshot` to a cache/table keyed by `order_id`.  
3. **Sync claim path** loads that row only (`claim_path_read` pattern).  
4. Policy service maps `suggested_tier` + `hard_gated` to product actions.

Python entry points (library):

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

# Lifecycle
refresh_order(order, history, devices, model, cache, users=users, event="delivered")

# Entity risk jumped
on_entity_risk_change({link_key}, open_orders, history, devices, model, cache,
                      new_scores={link_key: 95.0}, users=users)

# Claim time (sync)
snap = claim_path_read(order_id, cache)
```

### 2.3 When to refresh scores

| Trigger | Action |
|---|---|
| `placed` / `picked_up` / `out_for_delivery` / `delivered` / `claim` | `refresh_order` |
| Linked user/driver/vendor/UVD/device standing score moves by ≥ `entity_risk_change_delta` | `on_entity_risk_change` → rescore open orders |

Default delta: `10.0` in `operating_point.default.yaml`.

### 2.4 Input schema checklist

**orders / history**

- Identifiers: `order_id`, `user_id`, `driver_id`, `vendor_id`, `device_id`
- Economics: `amount`, `is_refund` (history)
- Time: `event_ts` (UTC ISO preferred)
- Context: `market`, `vertical` (`food` / `qcommerce`), `status`, `claim_reason`
- Labels (training rows): see README

**users**

- `user_id`, `signup_ts` — required for correct tenure / LTV features

**devices**

- `user_id`, `device_id`, `cluster_id`, `last_seen_ts`

Point-in-time rule: features use history with `event_ts ≤ order.event_ts` and **exclude** the scored `order_id`.

---

## 3. Decisioning logic (ops view)

### 3.1 Learning-primary path

With `decision_mode: learning_primary` (default):

1. Model emits calibrated `abuse_score` and `fraud_score` (0–100).  
2. Tier from **head thresholds** (OR logic):

| Tier | Rule |
|---|---|
| `auto_deny` | `abuse ≥ abuse_auto_deny` **or** `fraud ≥ fraud_auto_deny` |
| `hold_review` | else if either head ≥ hold threshold |
| `soft_friction` | else if either head ≥ soft threshold |
| `auto_approve` | else |

3. Soft thresholds are meant to catch ~`target_pattern_recall` (default **0.98**) of labeled abuse|fraud patterns.  
4. `combined_score` is a display blend of the two heads (prior is **not** used to band).  
5. Only `STRONG_FRAUD_LABEL` sets `hard_gated=true` and forces `auto_deny`.

Tune thresholds in `config/operating_point.default.yaml`, or let the tuner propose:

```bash
python scripts/run_tuner.py                 # ML auto-steps within max_abs_delta
python scripts/run_tuner.py --list-pending  # oversized moves waiting on HIL
python scripts/run_tuner.py --approve ID    # human approves full proposal
python scripts/run_tuner.py --reject ID
```

Same pattern as offline-cancel-risk: **auto-apply inside guardrails**, but if any key moves more than `auto_apply.max_abs_delta` (deny keys default 3), the cycle applies only a capped step and opens a **pending HIL** proposal for the remainder. Absolute bounds live in `config/policy_guardrails.default.yaml`.

### 3.2 What is intentionally *not* a hard gate

Rates, lifetime refunds, LTV burn, early-life windows, related-ring caps, and link/device score cutoffs are **features**. They appear in the evidence pack when elevated; they do **not** force deny. The model owns pattern detection.

### 3.3 Secondary mode

`decision_mode: score_only` maps `combined_score` through `tiers.*.max_score`. Prefer learning-primary unless you have a reason to score-band.

---

## 4. Tuning guide

Change one family of knobs at a time. Re-run `python scripts/backtest.py` and a labeled sample review before promoting.

### 4.1 `operating_point.default.yaml` — primary knobs

| Knob | Effect if ↑ | Effect if ↓ |
|---|---|---|
| `*_soft_friction` | Fewer soft flags (lower recall) | Catch more of the 98% mass |
| `*_hold_review` | Fewer holds | More review volume |
| `*_auto_deny` | Fewer auto-denies (higher precision) | More aggressive deny |
| `target_pattern_recall` | Soft-tier recall target for the tuner | — |

### 4.1b `policy_guardrails.default.yaml` — auto-apply vs HIL

| Knob | Role |
|---|---|
| `bounds.head_thresholds.*` | Absolute min/max (reject if outside) |
| `auto_apply.max_abs_delta` | Max score-point step auto-applied per cycle (default 5) |
| `auto_apply.max_abs_delta_by_key` | Tighter caps for deny thresholds (default 3) |
| `auto_apply.cooldown_minutes` | Min time between auto-applies |
| `auto_apply.min_pattern_recall_lift` | Require holdout pattern-recall lift to auto-step |

**Promotion tip:** let the tuner crawl; approve HIL only when the full jump is justified on a labeled holdout. Widen `hold_review` before tightening `auto_deny` when false-positive $ cost is high.

### 4.2 `policy.default.yaml` — safety only

| Knob | Role |
|---|---|
| `hard_gates.strong_fraud_label` | Force deny on investigator/strong label |
| `evidence_weights.*` | Contribution weights in the evidence pack |

Market overlays for rate/LTV gates are removed. Prefer market-specific models or threshold packs if base rates differ.

### 4.3 `label_weights.default.yaml` — training only

Requires **retrain** after changes.

| Knob | Role |
|---|---|
| `fraud_proxy_weight` | Down-weight proxy positives (default 0.4) |
| `fraud_proven_weight` | Weight for investigator-proven positives |
| `abuse_*` / `weak_policy_negative_weight` | Abuse head sample weights |
| `proxy_rules.*` | When unlabeled rows become proxy fraud |

Proxy rules require device farm size, multi-account device, UVD lift/share/cooccur, coordinated rate, and min user orders — by design, to reduce label leakage from “high rate alone”.

### 4.4 Practical tuning sequences

**Missing known patterns (recall &lt; 98% at soft)**

1. Run `python scripts/backtest.py --tune`.  
2. If recommended soft thresholds are already near floor and recall is still low → **retrain** (features/labels), do not invent hard gates.  
3. Check abuse vs fraud slice separately; fix the weak head.

**Too many soft/hold on clean users**

1. Raise soft/hold thresholds slightly (accept small recall tradeoff on the last %).  
2. Inspect false positives: are proxies too loose? Tighten `proxy_rules` + retrain.  
3. Do **not** reintroduce rate hard gates — that recreates the policy-engine failure mode.

**Fraud head overfits proxies**

1. Lower `fraud_proxy_weight`.  
2. Tighten `proxy_rules` (higher lift/share/cooccur).  
3. Retrain; evaluate **proven** and **proxy** slices separately.

**Mature serial claimants slipping through**

1. Confirm labels exist for that pattern; if not, label and retrain.  
2. Lower `abuse_hold_review` / `abuse_auto_deny` only after checking precision on high-LTV goods.

---

## 5. Reason codes cheat sheet

### Hard-gate (safety)

| Code | Meaning |
|---|---|
| `STRONG_FRAUD_LABEL` | Investigator / strong label on row → forced deny |

### Soft evidence (informational)

Examples: `ABUSE_HEAD`, `FRAUD_HEAD`, `USER_REFUND_TO_LTV_SOFT`, `COMBINED_REFUND_COUNT`, `RELATED_*`, `UVD_REFUND_LIFT`, `DEVICE_MULTI_ACCOUNT`, `REASON_REPEAT_RATE`.  
These explain the score; they do not set `hard_gated=true`.

---

## 6. Training & model maintenance

### 6.1 When to retrain

| Change | Retrain? |
|---|---|
| Head thresholds / evidence weights only | No |
| Feature code or label weights / proxy rules | Yes |
| New markets with different base rates | Yes (or market-specific models later) |
| Label taxonomy change | Yes |

Bump `model_version` in `operating_point.default.yaml` when promoting a new fit.

### 6.2 Training recipe

1. Build point-in-time feature frame with `build_order_feature_frame(orders, history, devices, users=users)`.  
2. Prefer **time-based** train/serve split in production (demo backtest uses a simple holdout).  
3. `train_two_head(...)` fits abuse + fraud heads (HistGradientBoosting + sigmoid calibration).  
4. Persist with `model.save(path)` / `TwoHeadModel.load(path)`.  
5. Run `scripts/backtest.py` and record:

   - Abuse / fraud ROC-AUC / AP  
   - Fraud **proven** + **proxy** slices  
   - `pattern_detection.recall_at_soft_or` (goal ≈ 0.98)  
   - Tier histogram; `hard_gated` should be rare (strong labels only)  
6. Run `python scripts/run_tuner.py` (or `backtest.py --tune --write-config`). Review `--list-pending` and `--approve` oversized moves.

### 6.3 Label quality rules

- Prefer proven fraud for the fraud head; keep proxies down-weighted.  
- Do not treat weak-policy approvals as clean negatives (`weak_policy_negative=1`).  
- Keep abuse and fraud labels conceptually separate in case tooling.  
- Refresh strong fraud labels from investigations on a fixed cadence (e.g. weekly).

### 6.4 Model artifact hygiene

- Store `model_version`, training date, git SHA, training row count, metrics JSON with the joblib artifact.  
- Canary: score a shadow traffic % with the new model; compare tier shifts and investigator agreement before 100% cutover.  
- Keep previous artifact for fast rollback.

---

## 7. Monitoring & SLOs

### 7.1 Online health

| Signal | Alert if |
|---|---|
| Claim-path cache miss rate | Spikes (scoring lag / lifecycle gaps) |
| `% hard_gated` | Should stay near strong-label rate; spike means label feed bug |
| Tier mix | Collapse to all `auto_deny` or all `auto_approve` |
| Pattern recall (offline) | Soft-tier recall on labeled set drops below target |
| Null/missing `signup_ts` | Tenure/LTV features degrade |

Latency is a platform concern for the cache read; it is **not** a modeling constraint.

### 7.2 Quality loops

| Loop | Cadence |
|---|---|
| Sample `hold_review` + `auto_deny` for human labels | Daily/weekly |
| Precision on proven-fraud queue | Weekly |
| False deny rate on high-LTV mature goods customers | Weekly |
| Proxy→proven conversion | Monthly |
| Re-tune soft thresholds vs target recall | After each retrain |

### 7.3 Config promotion checklist

1. Diff YAML; note threshold changes.  
2. Backtest + fixed regression orders (`O-NEW-*`, ring, burn, mature abuse, clean).  
3. Confirm `pattern_detection.recall_at_soft_or` ≈ target.  
4. Shadow in one market.  
5. Bump `policy_version`.  
6. Announce reason-code changes to ops UI owners.

---

## 8. Runbooks

### 8.1 Spike in false denies

1. Check whether `hard_gated` (should be strong labels only).  
2. If score-driven: raise `*_auto_deny` / `*_hold_review` slightly; review precision.  
3. Sample false positives for label errors / proxy leakage.  
4. Do not add rate hard gates.

### 8.2 Collusion ring not caught

1. Confirm device `cluster_id` and shared UVD appear in features (`related_account_count`, `combined_refund_*`, link scores).  
2. Check fraud score vs `fraud_soft_friction`; if scores are high but tier is approve, thresholds are wrong → `--tune` or lower soft/hold.  
3. If fraud score is low on a recurring ring shape → label + retrain (this is the 98% path).  
4. Ensure lifecycle + entity-risk refresh is running.

### 8.3 Claim path slow or empty

1. Confirm precompute ran for that `order_id`.  
2. Check lifecycle hooks for `delivered` / `claim`.  
3. Do **not** reintroduce live graph walks on the sync path — fix the async pipeline.

### 8.4 Rollback

1. Restore previous `operating_point.default.yaml` / `policy.default.yaml`.  
2. Or reload previous `two_head.joblib` + matching `model_version`.  
3. Recompute open orders if scores were already written with the bad version.

---

## 9. Security, privacy & compliance notes

- Snapshots contain entity ids and behavioral metrics — treat as sensitive fraud data.  
- Evidence packs may be shown to internal investigators; avoid putting raw PII beyond ids already in your fraud systems.  
- Proven fraud labels may be legally sensitive; restrict write access.  
- Log `model_version` + `policy_version` on every enforcement decision for auditability.

---

## 10. Versioning

| Version field | Where | Bump when |
|---|---|---|
| `model_version` | `operating_point.default.yaml` + artifact | New trained weights / feature set |
| `policy_version` | `operating_point.default.yaml` | Threshold / safety-gate changes |

Keep both on every snapshot and every downstream decision log line.

---

## 11. Quick reference — file map

```text
config/
  operating_point.default.yaml     # head_thresholds (ops, no retrain)
  policy_guardrails.default.yaml   # bounds + max_abs_delta / HIL
  policy.default.yaml              # strong_fraud_label + evidence weights
  label_weights.default.yaml       # training weights + proxy rules (retrain)
src/refund_abuse_risk/
  features/builders.py
  model/two_head.py
  scoring/policy.py
  scoring/thresholds.py
  control_plane/                   # tuner, audit, HIL proposals
  pipeline/score.py
scripts/
  generate_demo_data.py
  backtest.py
  run_tuner.py                     # auto-step + HIL approve/reject
docs/
  MANUAL.md
```

For design history and locked product decisions, see  
[`docs/superpowers/specs/2026-08-02-refund-abuse-risk-design.md`](superpowers/specs/2026-08-02-refund-abuse-risk-design.md).
