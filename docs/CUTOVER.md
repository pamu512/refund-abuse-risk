# Cutover checklist — live Downstream wiring

In-repo ladder is closed. Remaining work is Downstream wiring, not more synth multipass.

## Already shipped (this repo)

- [x] Serve-path train ≈ serve; discovery cap; as-of UV + null
- [x] Proven-primary time-OOT; costed promote; slice ECE; `$` FP ceiling
- [x] Closed-loop train load (`load_training_orders`)
- [x] Market×vertical policy features + overlay recommend/write
- [x] Hybrid feeds: `local_dir` + `sqlite` (`feeds.default.yaml` / `feeds.warehouse.yaml`)
- [x] Labeled OOT packs + floors (`eval_oot_pack.py`)
- [x] Ops snapshot + SDK / disposition ingest paths
- [x] Prod-shaped contracts + overlay promote: `oot_floors.prod.yaml`, `data/oot_packs/prod_shaped_v1/`, `validate_oot_pack.py`, `--require-dispositions`, ops snapshot max-age gate, `promote_overlays.py` (+ rollback)
- [x] HTTP + S3 feed drivers (`http` / `s3` https or boto3); `feeds.http.example.yaml`
- [x] `config/ops.overnight.prod.yaml` (require-dispositions, prod pack schema, require_promote)
- [x] Minimal claim-path HTTP API (`scripts/serve_api.py`, Bearer / X-Api-Token)
- [x] Audit C/H remediation: fail-closed promote, costed None, proven∪abuse stacker,
      proven ECE gates, class_balance off, mint-feature holdout, temporal_ok promote,
      HTTP allowed_hosts, serve audit/rate-limit/optional TLS

## Downstream (outside this repo)

1. Point `config/feeds.warehouse.yaml` `uri` / `query` at live investigator + ops DBs (or export cron → sqlite/CSV).
2. Replace / augment `prod_shaped_v1` with a **live** pack; keep `config/oot_floors.prod.yaml` (schema CI uses the fixture; live AP must clear prod floors).
3. Wire vendor SDK / vision event stream into `sdk_events` fixture path or claim-path `refresh_order`.
4. Cron: `./scripts/ops_overnight.sh` with `FEEDS_CONFIG` / `OOT_PACK` / `--require-promote`; set `PROMOTE_OVERLAYS=1` when overlays file is non-empty — see [OPS_RUNBOOK.md](OPS_RUNBOOK.md).
5. Refresh `data/ops_snapshot.json` daily (default OP already sets `max_ops_snapshot_age_hours: 24`; no baked demo snapshot).
6. Pass `delivered_ts` / `claim_ts` on score requests so policy window features fire.
7. Keep Architecture boundaries: no claim-path live graph walks; Downstream owns API/queue ([ARCHITECTURE.md](ARCHITECTURE.md)).

## Explicitly deferred (P2)

- GraphBEAN discovery, Postgres control plane, multi-region score store.

## Prod overnight (in-repo)

```bash
python scripts/ops_overnight.py --profile config/ops.overnight.prod.yaml --dry-run
# Live:
# SCORE_API_TOKEN=... PROMOTE_OVERLAYS=1 \
#   python scripts/ops_overnight.py --profile config/ops.overnight.prod.yaml --require-promote
# Claim-path API (precomputed jsonl cache):
# SCORE_API_TOKEN=... python scripts/serve_api.py --cache-jsonl data/score_cache.jsonl
```
