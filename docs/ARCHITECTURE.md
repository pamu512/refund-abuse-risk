# Architecture — refund-abuse-risk

Toolkit architecture for train ≈ serve honesty. Downstream owns the HTTP/API/queue runtime.

Companion: [MANUAL.md](MANUAL.md) · [OPS_RUNBOOK.md](OPS_RUNBOOK.md) · [CUTOVER.md](CUTOVER.md) · [GRADING.md](GRADING.md)

Internal quality ratings / letter grades are private — see [GRADING.md](GRADING.md). Score this repo by shipped contracts and gates, not public grade claims.

---

## 1. Intentional boundaries

| In this repo | Explicitly out (Downstream) |
|---|---|
| Library + CLI scripts + minimal score HTTP API | Multi-tenant gateway, queue workers, HA |
| In-process / file / jsonl claim cache | Redis / regional score store |
| SQLite HIL control plane | Postgres control plane HA |
| `local_dir` + `sqlite` + `http` + `s3` (https or boto3) feeds | Managed IAM / warehouse ownership |
| Advisory tier + `refund_effect` | Enforcement / payment rails |

Single-process claim cache and file OP are **design**, not unfinished scaffolding. Score architecture as a scoring toolkit, not a fraud platform.

---

## 2. Package map

| Package | Owns | Does not own |
|---|---|---|
| `schemas` | Snapshot / evidence / tier / effect types | Persistence |
| `features` | PIT order feature frame / row builders | Labels, model fit |
| `graph` | UV bipartite discovery, as-of + null | Serve-path scoring |
| `labels` | Disposition / weight contracts | Feed I/O |
| `model` | Two-head train / predict / class balance | Threshold promote |
| `scoring` | Policy, decision stacker, thresholds, Brier/adaptive ECE, Wilson+bootstrap precision CI, PSI, calibrator | Training loops |
| `pipeline` | Precompute → cache → claim-path read | Transport |
| `training` | Closed-loop order load, multipass helpers | Feed pull |
| `integrations` | Feeds pull/apply, SDK/ops/disposition ingest | Vendor SDKs |
| `baselines` | Behavior baseline export | Online serving |
| `control_plane` | HIL tuner proposals (SQLite) | Production change mgmt |
| `config` (module) | YAML loaders for `config/*.default.yaml` | Secrets |

Scripts under `scripts/` are thin CLIs over these packages. Prefer importing library functions in Downstream services.

---

## 3. Score path (runtime)

```text
lifecycle / entity-risk event
        │
        ▼
 build_order_feature_row (PIT history ≤ as_of)
        │
        ▼
 two_head.predict → abuse_score, fraud_score
        │
        ▼
 DecisionStacker → decision_score
        │
        ▼
 policy: hard gates + combine_scores (decision_primary)
        │
        ▼
 OrderRiskSnapshot → cache
        │
        ▼
 claim_path_read(order_id)  → Downstream enforcement
```

Rules:

1. **No live graph walks on the sync claim path** — discovery is batch / as-of.
2. **Heads explain; `decision_score` decides** when `decision_mode: decision_primary`.
3. **Hard gates** only for strong labels / policy caps — not rate heuristics.
4. Every Downstream decision should log `model_version` + `policy_version`.

---

## 4. Train path (honesty)

```text
feeds (dispositions / SDK / ops) ──► data/
        │
        ▼
 load_training_orders (closed-loop default)
        │
        ▼
 serve-path features (--feature-source serve)
   + discovery cap + as-of UV (+ null)
        │
        ▼
 multipass fit + OOF stacker
        │
        ▼
 time-OOT (proven-primary) + costed promote + ECE/PSI/ops gates
        │
        ▼
 promote_ok ? write OP / model : HIL or block
```

Train features must match serve builders. Proxy / discovery labels stay down-weighted; proven dispositions drive promote metrics.

---

## 5. Extension points

| Need | Extend here | Contract |
|---|---|---|
| New feed source | `integrations/feeds.py` source adapter | pull → stage → apply; no silent schema drift |
| New label type | `config/disposition_labels.default.yaml` + labels module | proven vs proxy weight |
| Market×vertical ladder | `decision_threshold_overlays` (demo-seeded) + `promote_overlays.py` | serve resolves most-specific overlay; sidecar via `DECISION_OVERLAYS_PATH` |
| Slice calibrator | fitted in `TwoHeadModel`, applied in `predict_proba` | `decision_score_raw` vs calibrated; joblib round-trips |
| PIT replay | `scripts/check_pit_replay.py` + CI | future history must not change as-of features |
| Ops ceilings | sidecar `ops_snapshot` + ingest | `max_ops_snapshot_age_hours: 24` on default OP; missing/stale metrics fail-closed once ceilings set |
| Prod-shaped OOT | `oot/` validate (`min_pack_n`) + `eval_oot_pack` (`min_holdout_n`) | schema CI ≠ prod AP / lift |
| Vendor SDK events | `integrations` SDK ingest → feature columns | claim refresh may re-read cache |
| New head features | `features/builders.py` + FEATURE column lists | PIT only; add leakage test |

---

## 6. Config surface

All defaults live in `config/*.default.yaml` (15 files). Loaders in `refund_abuse_risk.config`. Overnight job order: `config/ops.overnight.yaml`.

Critical honesty knobs: `operating_point.default.yaml` (thresholds, monitoring, overlays), `label_weights.default.yaml`, `oot_floors.default.yaml`, `feeds*.yaml`.

---

## 7. Versioning

| Artifact | Bump when |
|---|---|
| `model_version` | Weights or feature set change |
| `policy_version` | Thresholds, gates, effect rules, overlays |
| joblib path | Keep previous artifact for rollback (see MANUAL §8.4) |
