# Production feeds (hybrid) — design

Approved earlier; implementing 2026-08-04 with honesty knobs.

## Scope
- `local_dir` sources → stage → apply (default); `--stage-only` / `--dry-run`
- Feeds: dispositions, sdk_events, ops_snapshot
- Fixtures under `data/feeds/fixtures/`; prod = change `uri` in YAML
- Out of scope: real SQL/S3 drivers (clear error if selected)
