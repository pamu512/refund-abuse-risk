# Proposed-rule ingest — implementation plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Design is approved — do not reopen Approach A vs portal.

**Goal:** Ingest `data/proposed_rules/*.yaml` into OP overlays + effect/challenge rules with shadow-by-default, merge-tighten, and env-gated overnight — no analyst portal.

**Architecture:** Thin CLI `scripts/ingest_proposed_rules.py` over `refund_abuse_risk.ops.rule_ingest` helpers. Reuse `promote_overlays.backup_op` / `bump_policy_version` (generalize backup name prefix). Overnight step `when_env: INGEST_PROPOSED_RULES` after proposal producers, before `promote_overlays`.

**Tech Stack:** Python 3.11+, PyYAML, pytest. No new dependencies.

## Global Constraints

- Follow [design](../specs/2026-08-06-rule-ingest-design.md) Approach A exactly
- Never auto_enforce at discovery; ingest is the only writer into live configs from proposals
- Merge-tighten only (never raise soft/hold/deny thresholds); map proposal `deny` → OP `auto_deny`
- No analyst portal; no letter grades / project ratings ([GRADING.md](../../GRADING.md))
- Do not commit unless user asks

---

## File map

| File | Role |
|---|---|
| `config/rule_ingest.default.yaml` | Gates, sources, targets, live switch (enabled/live_enabled false) |
| `src/refund_abuse_risk/config.py` | `load_rule_ingest` |
| `src/refund_abuse_risk/ops/rule_ingest.py` | Parse, gates, live policy, merge-tighten, skip-if-id, run |
| `scripts/ingest_proposed_rules.py` | CLI: dry-run / apply / JSON summary |
| `scripts/promote_overlays.py` | Export backup helper with optional filename prefix |
| `config/ops.overnight.yaml` + `.prod.yaml` | Optional ingest step + path placeholder |
| `docs/OPS_RUNBOOK.md` / `docs/MANUAL.md` | Ingest + env flags |
| `src/.../ops/segment_anomaly.py` | Notes → ingest path (not “analyst edit”) |
| `tests/test_rule_ingest.py` | Disabled, shadow, live, merge-tighten, skip id, gates |

---

### Task 1: Config + loader + backup helper

**Files:**
- Create: `config/rule_ingest.default.yaml` (schema from design §4; also `max_overlay_proposals` / `max_challenge_proposals` if design max_proposals is the only cap — use design’s `gates.max_proposals: 50` as the hard cap; deliverable “max overlay/challenge proposals” = same gate or split under gates — **use design keys**: `max_proposals` only)
- Modify: `src/refund_abuse_risk/config.py` — add `load_rule_ingest`
- Modify: `scripts/promote_overlays.py` — `backup_op(..., prefix="operating_point")` so effect_rules can reuse with `prefix="effect_rules"`

- [ ] **Step 1:** Add default YAML + loader
- [ ] **Step 2:** Generalize `backup_op` keep-glob to `{prefix}.*.yaml`
- [ ] **Step 3:** Verify loader returns mapping with `enabled is False`

---

### Task 2: Core library + unit tests (TDD)

**Files:**
- Create: `src/refund_abuse_risk/ops/rule_ingest.py`
- Create: `tests/test_rule_ingest.py`
- Modify: `src/refund_abuse_risk/ops/__init__.py` only if needed (prefer not)

**Interfaces:**
- `live_allowed(cfg, env) -> bool` — `cfg.live_enabled` or `INGEST_RULES_LIVE=1`
- `discover_proposal_files(sources, root) -> list[Path]`
- `parse_proposals(doc, source_path) -> list[dict]` — flatten `proposed_rules` / typed lists / kind objects
- `apply_gates(proposals, gates) -> (kept, skipped_reasons)`
- `coerce_live_policy(proposals, live_ok) -> list` — force shadow + clear `auto_enforce`
- `merge_tighten_overlay(op, proposal) -> (op, accepted|rejected_keys)`
- `append_rule_if_new(rules, proposal) -> (rules, appended: bool)`
- `run_ingest(*, cfg, root, dry_run, strict) -> summary dict`

- [ ] **Step 1: Write failing tests** — disabled no-op; shadow write; live flag; merge-tighten (40+50→40, 40+30→30); skip existing id; gates (low z / max_proposals)
- [ ] **Step 2: Implement `rule_ingest.py`**
  - Overlay keys: `soft_friction`, `hold_review`, `auto_deny` (accept `deny` alias)
  - Deltas from `suggested.*_delta`; positive loosen → reject key
  - Backup OP / effect_rules before mutate; bump `policy_version` on OP overlay change
  - Dry-run: no writes / no backups
- [ ] **Step 3:** `pytest tests/test_rule_ingest.py -v` PASS

---

### Task 3: CLI

**Files:**
- Create: `scripts/ingest_proposed_rules.py`

```bash
python scripts/ingest_proposed_rules.py --config config/rule_ingest.default.yaml [--dry-run] [--strict] [--summary-out PATH]
```

- [ ] Exit 0 on disabled / success; 1 on IO/parse or `--strict` soft failures
- [ ] Print JSON summary to stdout

---

### Task 4: Overnight + docs + segment notes

**Files:**
- Modify: `config/ops.overnight.yaml` — path `rule_ingest_config`; step after `graphbean_lite`, before `promote_overlays`, `optional: true`, `when_env: INGEST_PROPOSED_RULES`
- Modify: `config/ops.overnight.prod.yaml` — same step before `promote_overlays` (+ path)
- Modify: `docs/OPS_RUNBOOK.md` §7 — ingest dry-run/apply + env flags
- Modify: `docs/MANUAL.md` — config map + script row
- Modify: `src/refund_abuse_risk/ops/segment_anomaly.py` — proposal note / YAML notes → `scripts/ingest_proposed_rules.py` (not analyst edit)
- Modify: `tests/test_architecture_and_ops_docs.py` if overnight id assertions need `ingest_proposed_rules` / loader list

- [ ] Overnight dry-run still exit 0; placeholder keys resolve
- [ ] Focused pytest then `.venv/bin/python -m pytest`

---

## Verify (success criteria from design §10)

1. Default config → `{skipped: true, reason: disabled}` or shadow-only writes
2. Merge never loosens; duplicate effect/challenge ids skipped
3. Backup before mutate; dry-run creates none
4. Gates + summary JSON
5. Overnight step only when `INGEST_PROPOSED_RULES` set
6. Docs mention ingest; no grade language

## How to run (operators)

```bash
# Dry-run (enable in a temp config or flip enabled:true first)
python scripts/ingest_proposed_rules.py --config config/rule_ingest.default.yaml --dry-run

# Apply shadow merges (requires enabled: true in config)
python scripts/ingest_proposed_rules.py --config config/rule_ingest.default.yaml

# Allow mode: live from proposals
INGEST_RULES_LIVE=1 python scripts/ingest_proposed_rules.py --config …

# Overnight
INGEST_PROPOSED_RULES=1 python scripts/ops_overnight.py --profile config/ops.overnight.yaml
```
