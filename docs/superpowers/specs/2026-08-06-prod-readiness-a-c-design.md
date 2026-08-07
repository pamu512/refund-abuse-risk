# Prod-shaped contracts + overlay promote — design

Date: 2026-08-06  
Status: approved + implemented (2026-08-06)  
Tracks: **live-shaped data path** + **promote-to-serve overlays**  
Acceptance bar: **contracts + prod-shaped fixture pack** (not live warehouse)

> Privacy: historical filename retained for links. Internal quality ratings are private — see [docs/GRADING.md](../../GRADING.md).

Companion: [CUTOVER.md](../../CUTOVER.md) · [OPS_RUNBOOK.md](../../OPS_RUNBOOK.md) · [ARCHITECTURE.md](../../ARCHITECTURE.md)

---

## 1. Goal

Ship in-repo production-shaped contracts without claiming live loss reduction:

1. **Strict prod floors** + a **prod-shaped** OOT pack (disposition/chargeback schema).
2. **Train gate** that refuses synth-only when required.
3. **Ops snapshot freshness** gate.
4. **Overlay promote / rollback** into operating point used at serve.

Explicitly **not** in this change: live warehouse URIs, HTTP serve API, GraphBEAN, vendor SDK stream.

---

## 2. Approach

**Thin CLI layer** over existing modules (`load_training_orders`, `eval_oot_pack`, `recommended_overlays_from_slices`, `resolve_decision_thresholds`, monitoring gates). No new `prod_readiness` package.

---

## 3. Track — live-shaped data path

### 3.1 Strict floors

Add `config/oot_floors.prod.yaml` stricter than `oot_floors.default.yaml`:

| Key | Default (demo) | Prod (target) |
|---|---|---|
| `min_holdout_n` | 30 | ≥ 100 |
| `min_proven_positives` | 3 | ≥ 15 |
| `min_fraud_proven_average_precision` | 0.20 | ≥ 0.35 |
| `min_fraud_proven_precision_at_soft` | 0.10 | ≥ 0.20 |
| `max_decision_ece` | 0.35 | ≤ 0.20 |
| `require_serve_path` | true | true |
| `max_ops_snapshot_age_hours` | (unset) | 24 |

`eval_oot_pack.py` continues to merge `manifest.floors` over the chosen floors file.

### 3.2 Prod-shaped fixture pack

Path: `data/oot_packs/prod_shaped_v1/`

Required files:

- `manifest.yaml` — `profile: prod_shaped`, `floors_file` or inline strict floors, `require_dispositions: true`, `holdout_days`
- `orders.csv` — includes market/vertical/amount/event_ts; joinable to dispositions
- `dispositions.csv` — proven outcomes (`chargeback_lost` and/or `bank_dispute_lost` and investigator labels per `disposition_labels.default.yaml`)
- `history.csv`, `devices.csv`, optional `users.csv`

**Schema validation** (new): `scripts/validate_oot_pack.py` (or `--schema-only` on eval):

- Required columns present
- At least `min_proven_positives` disposition-proven rows (by config mapping)
- Manifest profile ∈ {`demo`, `prod_shaped`}

**CI policy:**

- Schema + disposition gates: **must pass**
- Full `eval_oot_pack` against prod floors: **may exit 1** on synth AP — documented; not a green-metric claim

Seed helper: extend `seed_oot_pack.py` with `--profile prod_shaped` or a dedicated seed that writes the pack from closed-loop labels (fixture, not lift).

### 3.3 Train refuses synth-only

- Flag: `--require-dispositions` on `train_model.py` (and overnight prod path).
- Behavior via `load_training_orders` stats:
  - Fail if neither `orders.labeled.csv` with proven mass nor `dispositions.csv` applied
  - Fail if proven rate / proven count below configurable floor (default: ≥ 1 proven row and dispositions path present, or labeled pre-baked with proven count ≥ `min_train_proven`)
- Demo default: flag **off**. Overnight / CUTOVER: flag **on** for prod.

### 3.4 Ops snapshot freshness

- Snapshots already carry `as_of` (normalize if missing → treat as stale when freshness enforced).
- Config: `monitoring.max_ops_snapshot_age_hours` (prod floors / OP monitoring).
- `evaluate_monitoring_gates`: if max age set and snapshot `as_of` older than cutoff → `ok=False` with reason.
- Overnight: after pull feeds, freshness checked when ceilings / age configured.

---

## 4. Track — promote-to-serve overlays

### 4.1 Promote CLI

`scripts/promote_overlays.py`

| Flag | Meaning |
|---|---|
| `--overlays PATH` | YAML with `decision_threshold_overlays` (from `--write-slice-overlays`) |
| `--op PATH` | Target operating point (default `config/operating_point.default.yaml`) |
| `--dry-run` | Print merge; no write |
| `--require-non-empty` | Exit 1 if overlay list empty |
| `--rollback` | Restore newest backup under `config/backups/` |
| `--backup-keep N` | Default 5 |

Steps (promote):

1. Load overlays; optionally filter to promote-eligible only if metadata present; otherwise trust fragment from `recommended_overlays_from_slices` (already filtered).
2. Backup current OP → `config/backups/operating_point.<utc>.yaml`.
3. Write OP with replaced `decision_threshold_overlays`; bump `policy_version` when present (append or patch — implement as: if string endswith digits, increment; else append `.overlaysN`).
4. Print summary (n overlays, backup path).

Rollback: copy newest backup over OP path.

### 4.2 Serve path

No new runtime. `resolve_decision_thresholds` first-match stays. Tests confirm market×vertical after promote changes tier boundaries.

### 4.3 Overnight wiring

- After backtest: if env `PROMOTE_OVERLAYS=1` (or profile flag) and overlays file non-empty and `require_promote` path green → run promote.
- Demo overnight: leave `PROMOTE_OVERLAYS` unset so empty overlays do not fail.

### 4.4 HIL interaction

Promote overlays is **separate** from head-threshold HIL. Large global ladder changes still go through existing tuner HIL. Overlay promote is gated by costed `promote_eligible` + optional `--require-promote` overnight.

---

## 5. Tests

| Test | Asserts |
|---|---|
| Pack schema | `prod_shaped_v1` validates; missing disposition column fails |
| Require dispositions | train/load fails without; passes with dispositions |
| Ops freshness | stale `as_of` fails monitoring gate; fresh passes |
| Promote / rollback | backup written; resolve differs by market; rollback restores |
| Thin / empty | `--require-non-empty` fails on `[]` |

---

## 6. Docs

- Update CUTOVER, OPS_RUNBOOK, MANUAL cheat sheet, README status (capability language only; no public grade claims).

---

## 7. Non-goals

- S3/HTTP feed drivers
- Serve HTTP API / auth / queue
- GraphBEAN
- Claiming synth pack clears prod AP floors
- Auto-HIL approve

---

## 8. Implementation order

1. Floors + pack seed + `validate_oot_pack` (schema)
2. `--require-dispositions` + freshness gate
3. `promote_overlays.py` + tests
4. Overnight / docs

---

## Spec self-review

- [x] No placeholders / TBD for required behavior
- [x] No contradiction with ARCHITECTURE (toolkit, Downstream owns API)
- [x] Scope limited to contracts + prod-shaped fixture pack acceptance bar
- [x] Success criteria testable (schema CI vs optional full eval)
- [x] Synth AP failure on prod floors is explicit, not a silent pass
