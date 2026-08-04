# Cutover checklist — A++ thin → live

In-repo ladder is closed. Remaining work is Downstream wiring, not more synth multipass.

## Already shipped (this repo)

- [x] Serve-path train ≈ serve; discovery cap; as-of UV + null
- [x] Proven-primary time-OOT; costed promote; slice ECE; `$` FP ceiling
- [x] Closed-loop train load (`load_training_orders`)
- [x] Market×vertical policy features + overlay recommend/write
- [x] Hybrid feeds: `local_dir` + `sqlite` (`feeds.default.yaml` / `feeds.warehouse.yaml`)
- [x] Labeled OOT packs + floors (`eval_oot_pack.py`)
- [x] Ops snapshot + SDK / disposition ingest paths

## Downstream (outside this repo)

1. Point `config/feeds.warehouse.yaml` `uri` / `query` at live investigator + ops DBs (or export cron → sqlite/CSV).
2. Drop production packs under `data/oot_packs/<id>/` with **strict** floors (not `demo_*` lenient floors).
3. Wire vendor SDK / vision event stream into `sdk_events` fixture path or claim-path `refresh_order`.
4. Cron: `pull_production_feeds.py` → train/backtest → HIL promote only when `promote_ok`.
5. Pass `delivered_ts` / `claim_ts` on score requests so policy window features fire.

## Explicitly deferred (P2)

- S3/HTTP feed drivers, GraphBEAN discovery, serve HTTP API, Postgres control plane.
