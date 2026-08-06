# Ops runbook — overnight, promote, troubleshoot

Day-to-day ops for the toolkit. Requirements / tuning depth: [MANUAL.md](MANUAL.md). Live wiring: [CUTOVER.md](CUTOVER.md). Package map: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Overnight profile

Declarative job order: [`config/ops.overnight.yaml`](../config/ops.overnight.yaml).

Runner (demo-safe defaults; override env for prod paths):

```bash
./scripts/ops_overnight.sh
# or: python scripts/ops_overnight.py --dry-run
#      python scripts/ops_overnight.py
```

| Step | Script | Fail behavior |
|---|---|---|
| 1 Pull feeds | `pull_production_feeds.py` | abort |
| 2 Train | `train_model.py --feature-source serve …` | abort |
| 3 OOT pack eval | `eval_oot_pack.py` | abort (exit 1 on floor breach) |
| 4 Backtest / tune propose | `backtest.py` (+ optional `--tune`) | with `--require-promote`: abort unless `examples/csv_demo/backtest_metrics.json` has `honesty.promote_ok: true` |
| 5 HIL list | `run_tuner.py --list-pending` | never auto-approve |

**Prod overrides (Downstream):** set `FEEDS_CONFIG`, `DATA_DIR`, `OOT_PACK`, `MODEL_OUT`, and point warehouse URIs at live DBs. Do not run overnight on synth packs and treat metrics as lift.

Suggested cron (example):

```cron
15 2 * * * cd /opt/refund-abuse-risk && ./scripts/ops_overnight.sh >> /var/log/rar-overnight.log 2>&1
```

---

## 2. Promote decision tree

```text
backtest honesty.promote_ok?
        │
   no ──┴──► block write; file HIL or fix floors / ECE / `$` / ops
        │
       yes
        │
eval_oot_pack (strict floors for prod) green?
        │
   no ──┴──► block; inspect pack metrics vs oot_floors
        │
       yes
        │
|Δ threshold| > max_abs_delta?
        │
   yes ─┴──► HIL approve via run_tuner.py --approve <id>
        │
       no / approved
        │
shadow one market → watch tier mix + proven precision → bump versions
```

Never promote when:

- `soft_floor_would_bind` on soft friction (costed contract rewritten)
- thin-slice overlays marked `promote_eligible: false`
- ECE / PSI / ops_snapshot ceilings breached
- OOT pack floors fail

Missing ECE/PSI → gate does not block (unknown). Once ceilings are set in OP, live `ops_snapshot` values must stay under them.

---

## 3. Config inventory (ops-owned)

| File | Owner concern |
|---|---|
| `operating_point.default.yaml` | Ladders, overlays, monitoring ceilings, decision_mode |
| `policy.default.yaml` / `policy_guardrails.default.yaml` | Caps, hard gates |
| `label_weights.default.yaml` | Proven vs proxy mass |
| `disposition_labels.default.yaml` | Which dispositions count as proven |
| `oot_floors.default.yaml` | Pack promote floors (strict for prod) |
| `feeds.warehouse.yaml` | Live pull URIs / queries |
| `ops.overnight.yaml` | Job order + CLI flags |
| `vertical_policy.default.yaml` | Market×vertical priors |
| `effect_rules.default.yaml` | Advisory refund_effect mapping |
| `sdk_ingest.default.yaml` | SDK event → feature map |
| `head_hyperparams.default.yaml` | Per-head tree depth / etc. |

Diff every promote. Keep previous OP + joblib for rollback (MANUAL §8.4).

---

## 4. Daily health checks

```bash
# Feeds applied?
ls -lt data/feeds_stage 2>/dev/null | head
# Last train artifact
ls -lt models/*.joblib | head
# Pending HIL
python scripts/run_tuner.py --list-pending
# OOT (prod pack id)
python scripts/eval_oot_pack.py --pack data/oot_packs/<prod_id>
# Ops snapshot freshness
python scripts/ingest_ops_snapshot.py --help   # then apply latest JSON
```

| Signal | Healthy | Action if bad |
|---|---|---|
| `honesty.promote_ok` | true on last backtest | see §2 |
| OOT pack | exit 0 | inspect AP/ECE vs floors |
| Hold rate vs ceiling | under `max_hold_rate` | raise soft or fix labels |
| Live override rate | under ceiling | shadow vs live mismatch |
| Hard-gate rate | ≈ strong-label rate | remove rate heuristics |

---

## 5. Incident snippets

| Symptom | First checks |
|---|---|
| Spike false denies | MANUAL §8.1 — hard_gated vs score-driven |
| Ring missed | MANUAL §8.2 — devices / labels / thresholds |
| Stale claim scores | MANUAL §8.3 — precompute + lifecycle |
| Overnight abort at feeds | warehouse URI / sqlite path / schema |
| Overnight abort at OOT | pack labels empty or floors too strict for demo |
| Promote blocked on `$` FP | `max_fp_refund_dollars_mean` vs amount column |

---

## 6. SLOs (toolkit ops — not loss $)

These are **process** SLOs for the scoring toolkit. Fraud-loss SLOs are Downstream.

| SLO | Target |
|---|---|
| Overnight job completes or pages | daily |
| Promote only with `promote_ok` + green OOT | 100% |
| Rollback artifact retained | last N=2 model+OP pairs |
| Ops snapshot age | < 24h when `max_ops_snapshot_age_hours` set |
| Synth/demo packs | never used for prod promote |
| Prod-shaped schema pack | `validate_oot_pack.py` green in CI |

---

## 7. Prod-shaped pack + overlay promote (A+C)

```bash
# Schema contract (CI) — not lift
python scripts/seed_oot_pack.py --profile prod_shaped   # if regenerating
python scripts/validate_oot_pack.py --pack-dir data/oot_packs/prod_shaped_v1

# Train refuses synth-only
python scripts/train_model.py --feature-source serve --require-dispositions

# Overlays → serve OP
python scripts/backtest.py --write-slice-overlays examples/csv_demo/slice_overlays.yaml
python scripts/promote_overlays.py --overlays examples/csv_demo/slice_overlays.yaml --dry-run
python scripts/promote_overlays.py --overlays examples/csv_demo/slice_overlays.yaml --require-non-empty
# python scripts/promote_overlays.py --rollback
```

Full `eval_oot_pack` against `oot_floors.prod.yaml` may fail AP on the synth fixture — expected. Live packs must clear those floors.

---

## 8. Prod overnight + serve API + HTTP feeds

```bash
# Prod profile (require-dispositions, prod pack schema, require_promote)
python scripts/ops_overnight.py --profile config/ops.overnight.prod.yaml --dry-run
OPS_OVERNIGHT_PROFILE=config/ops.overnight.prod.yaml ./scripts/ops_overnight.sh --require-promote

# Overlay promote after non-empty write
PROMOTE_OVERLAYS=1 python scripts/ops_overnight.py --profile config/ops.overnight.prod.yaml

# HTTP / pre-signed S3 feeds — see config/feeds.http.example.yaml
python scripts/pull_production_feeds.py --config config/feeds.http.example.yaml

# Minimal claim-path API
SCORE_API_TOKEN=dev-token python scripts/serve_api.py --cache-jsonl data/score_cache.jsonl
# GET /health
# GET /v1/orders/{order_id}/risk  (Authorization: Bearer …)
```

Set `monitoring.max_ops_snapshot_age_hours: 24` on the live operating point.
