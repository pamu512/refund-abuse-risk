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

## 7. Prod-shaped contracts + overlay promote

```bash
# Schema contract (CI) — not lift
python scripts/seed_oot_pack.py --profile prod_shaped   # if regenerating
python scripts/validate_oot_pack.py --pack-dir data/oot_packs/prod_shaped_v1

# Train refuses synth-only
python scripts/train_model.py --feature-source serve --require-dispositions

# Fresh ops sidecar required before promote (default OP age gate = 24h; no baked snapshot)
python scripts/seed_feed_fixtures.py   # demo: stamps as_of=now
python scripts/pull_production_feeds.py --config config/feeds.default.yaml

# Overlays → serve OP (default OP ships demo SG/ID|food ladders; replace for prod)
python scripts/backtest.py --write-slice-overlays examples/csv_demo/slice_overlays.yaml
python scripts/promote_overlays.py --overlays examples/csv_demo/slice_overlays.yaml --dry-run
python scripts/promote_overlays.py --overlays examples/csv_demo/slice_overlays.yaml --require-non-empty
# python scripts/promote_overlays.py --rollback
# Optional sidecar: DECISION_OVERLAYS_PATH=config/decision_threshold_overlays.demo.yaml

# PIT leakage gate (also in GitHub Actions CI)
python scripts/check_pit_replay.py

# P0 — segment refund anomalies → proposed rules (shadow only; never auto-merge)
python scripts/detect_segment_anomalies.py --orders data/orders.csv
# → data/proposed_rules/segment_anomalies.proposed.yaml

# Ingest proposed rules into OP overlays / effect_rules (default config: disabled)
python scripts/ingest_proposed_rules.py --config config/rule_ingest.default.yaml --dry-run
# Apply (requires rule_ingest.enabled: true): merge-tighten overlays; append new effect/challenge ids
# python scripts/ingest_proposed_rules.py --config config/rule_ingest.default.yaml
# Allow mode: live from proposals for this process:
# INGEST_RULES_LIVE=1 python scripts/ingest_proposed_rules.py --config config/rule_ingest.default.yaml
# Overnight step (also needs enabled: true in rule_ingest config):
# INGEST_PROPOSED_RULES=1 python scripts/ops_overnight.py

# P1a — decision archive (set on serve / score path)
export DECISION_ARCHIVE_PATH=data/decision_archive.db
python scripts/query_decision_archive.py --limit 20

# P1b — weak LF labels (proven rows untouched)
python scripts/mint_weak_labels.py --with-features

# P2a — GraphBEAN-lite UV recon → proposed actions (shadow only; never auto-merge)
python scripts/run_graphbean_lite.py --history data/history.csv
# → data/proposed_rules/graphbean_lite.proposed.yaml
# Unknown kinds (e.g. graphbean_edge) are skipped by ingest until mapped to overlay/effect/challenge.

# P2b — risk challenges live under config/effect_rules.default.yaml (challenge_rules);
# score path fills risk_challenge / shadow_risk_challenge on OrderRiskSnapshot.
```

**CI green ≠ production lift.** `validate_oot_pack` / pytest only prove schema + disposition
contracts (`min_pack_n`, proven counts). Proven AP / ECE / `temporal_ok` require a **live**
OOT pack under `eval_oot_pack` + `oot_floors.prod.yaml`. Synth `prod_shaped_v1` may fail
those floors — expected.

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
