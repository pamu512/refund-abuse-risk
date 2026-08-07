# Prod-shaped contracts + overlay promote — implementation plan

> Privacy: historical filename retained for links. Internal quality ratings are private — see [docs/GRADING.md](../../GRADING.md).

> **For agentic workers:** Execute task-by-task. Steps use checkbox syntax.

**Goal:** Ship live-shaped data contracts (strict floors, prod_shaped pack, require-dispositions, ops freshness) and overlay promote/rollback without claiming live lift.

**Architecture:** Thin CLIs + small helpers over existing `load_training_orders`, monitoring gates, `recommended_overlays_from_slices`, `resolve_decision_thresholds`.

**Tech Stack:** Python 3.11+, pandas, PyYAML, pytest.

## Global Constraints

- Acceptance bar: contracts + prod-shaped fixture pack (spec § acceptance **2**)
- Synth AP may fail prod floors — CI gates schema/dispositions, not prod AP green
- No HTTP API / S3 / GraphBEAN
- Do not commit unless user asks

---

## File map

| File | Role |
|---|---|
| `config/oot_floors.prod.yaml` | Strict floors |
| `src/refund_abuse_risk/oot/validate.py` | Pack schema validation |
| `scripts/validate_oot_pack.py` | CLI for schema |
| `scripts/seed_oot_pack.py` | `--profile prod_shaped` |
| `data/oot_packs/prod_shaped_v1/` | Fixture pack |
| `src/.../training/closed_loop.py` | `require_dispositions` / proven mass |
| `scripts/train_model.py` | `--require-dispositions` |
| `src/.../scoring/monitoring.py` | max ops snapshot age |
| `scripts/promote_overlays.py` | promote / rollback |
| `config/ops.overnight.yaml` | optional promote step |
| tests + docs | coverage / CUTOVER / OPS_RUNBOOK |

---

### Task 1: Floors + pack schema validator
- [x] Add `config/oot_floors.prod.yaml`
- [x] Add `refund_abuse_risk.oot.validate` + `scripts/validate_oot_pack.py`
- [x] Tests: valid shape / missing disposition fails
- [x] Verify: pytest targeted

### Task 2: Seed prod_shaped pack
- [x] Extend `seed_oot_pack.py --profile prod_shaped`
- [x] Generate pack under `data/oot_packs/prod_shaped_v1/` (gitignore exception)
- [x] Schema validation passes; full eval may fail AP (documented)
- [x] Verify: validate_oot_pack exit 0

### Task 3: require-dispositions + ops freshness
- [x] `load_training_orders` / train flag
- [x] `evaluate_monitoring_gates` age check
- [x] Tests for both
- [x] Verify: pytest targeted

### Task 4: promote_overlays + overnight + docs
- [x] `scripts/promote_overlays.py` backup/apply/rollback
- [x] Tests promote/rollback/resolve
- [x] Wire overnight env `PROMOTE_OVERLAYS`
- [x] Docs (capability language; no public grade claims)
- [x] Verify: pytest subset green
