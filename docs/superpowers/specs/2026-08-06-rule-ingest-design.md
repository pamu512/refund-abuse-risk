# Proposed-rule ingest (no analyst portal) — design

Date: 2026-08-06  
Status: approved for implementation (Approach A)  
Companion: [platform P0/P1 design](2026-08-06-platform-p0-p1-design.md) · [OPS_RUNBOOK.md](../../OPS_RUNBOOK.md) · [ARCHITECTURE.md](../../ARCHITECTURE.md) · [GRADING.md](../../GRADING.md)

---

## 1. Goal

Ingest machine-proposed rules from `data/proposed_rules/*.yaml` into the toolkit’s config surfaces **without** an analyst portal:

| Target | Destination |
|---|---|
| Decision threshold overlays | `operating_point` `decision_threshold_overlays` (or sidecar OP) |
| Effect rules | `config/effect_rules.default.yaml` `rules` |
| Challenge rules | `config/effect_rules.default.yaml` `challenge_rules` |

Default path is **shadow only**. Live writes require an explicit config flag or env override. Overnight may call ingest only behind an env gate (same pattern as `PROMOTE_OVERLAYS`).

---

## 2. Non-goals

- Analyst / review portal UI
- Auto overnight ingest without an env flag
- Full GraphBEAN (neural) — GraphBEAN-lite proposals may be sources; ingest treats them as YAML proposals only
- Loosening thresholds (merge never raises soft/hold cutoffs; only tighten)
- Claiming loss reduction from ingested rules
- Replacing HIL tuner for global head-threshold ladder jumps

---

## 3. Approach

**Approach A (approved):**

1. `config/rule_ingest.default.yaml` — gates, paths, live switch
2. `scripts/ingest_proposed_rules.py` — thin CLI over merge helpers

Reuse existing patterns from `promote_overlays.py` (OP backup, dry-run, JSON summary) and overnight `when_env` optional steps. Prefer library helpers under `refund_abuse_risk` only if the CLI would otherwise grow past a thin wrapper; no new package required if helpers stay in the script or a small module next to segment anomaly / control plane.

---

## 4. Config schema

`config/rule_ingest.default.yaml`:

```yaml
version: 1
enabled: false                 # master switch; false = no-op even if CLI run
live_enabled: false            # false → force all ingested rules to mode: shadow
                               # true OR env INGEST_RULES_LIVE=1 → allow mode: live from proposal

# Source globs (relative to repo root unless absolute)
sources:
  - data/proposed_rules/*.proposed.yaml
  - data/proposed_rules/*.yaml

# Gates applied to each proposal before merge
gates:
  min_z: 3.0                   # segment / graphbean z-like fields; skip if present and below
  min_recon_z: 0.0             # GraphBEAN-lite recon residual; 0 = off
  min_edge_orders: 0           # skip overlay/effect proposals with edge_n / n_test_day below
  max_proposals: 50            # hard cap after filtering; excess dropped (logged)

# Targets
targets:
  operating_point: config/operating_point.default.yaml
  effect_rules: config/effect_rules.default.yaml
  backup_dir: config/backups
  backup_keep: 5

# Overnight (profile may reference; still needs when_env)
overnight:
  step_id: ingest_proposed_rules
  when_env: INGEST_PROPOSED_RULES
```

Notes:

- `enabled: false` by default (demo-safe).
- `live_enabled` is independent of `enabled`: both must allow live for any `mode: live` write; otherwise coerce to shadow.
- Env `INGEST_RULES_LIVE=1` overrides `live_enabled: false` for that run (same honesty pattern as other ops env gates).
- Missing optional proposal fields used by a gate → gate does not fail the proposal (skip that check).

---

## 5. Proposal document shape

Ingest accepts YAML documents that are either:

1. A mapping with `proposed_rules: [...]` (segment anomaly / GraphBEAN-lite style), or
2. A flat list under a recognized key (`decision_threshold_overlays`, `rules`, `challenge_rules`), or
3. Individual proposal objects with a `kind` field.

Recognized `kind` values:

| `kind` | Writes to |
|---|---|
| `decision_threshold_overlay` | OP `decision_threshold_overlays` |
| `effect_rule` | effect_rules `rules` |
| `challenge_rule` | effect_rules `challenge_rules` |

Minimum fields:

**Overlay**

- `market`, `vertical` (strings)
- Optional: `soft_friction`, `hold_review`, `deny` absolute thresholds **or** `suggested.soft_friction_delta` / `hold_review_delta` (applied relative to current resolved / global ladder at ingest time — deltas only tighten)
- `mode`: `shadow` \| `live` (coerced per live policy)
- `id` optional; if absent, derive stable id from `source|market|vertical|kind`

**Effect / challenge rule**

- `id` (required for skip-if-exists)
- `when` mapping, `effect` or `challenge`, `reason_code`
- `mode`: `shadow` \| `live`

Unknown `kind` → skip + warn in summary (do not fail the whole run unless `--strict`).

---

## 6. Ingest behavior

| Step | Behavior |
|---|---|
| Load config | Fail if YAML invalid; if `enabled: false`, exit 0 with summary `{skipped: true, reason: disabled}` |
| Discover sources | Expand globs; de-dupe paths; skip missing globs quietly |
| Parse proposals | Flatten `proposed_rules` / typed lists; attach `source_path` |
| Apply gates | Drop proposals failing `min_z` / `min_recon_z` / `min_edge_orders`; truncate to `max_proposals` |
| Live policy | If not live-allowed → set every proposal `mode: shadow` and clear any `auto_enforce: true` |
| Overlay merge | Merge-tighten into OP (see §7); backup OP first if any overlay mutates |
| Effect/challenge merge | Append only when `id` absent in target list (see §7) |
| Dry-run | Compute summary; write nothing |
| Write | Persist OP and/or effect_rules YAML; bump `policy_version` on OP when overlays change |
| Summary | Print JSON to stdout (and optional `--summary-out PATH`) |

CLI sketch:

```bash
python scripts/ingest_proposed_rules.py \
  --config config/rule_ingest.default.yaml \
  [--dry-run] [--strict] [--summary-out PATH]
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Success (including disabled / nothing to do) |
| 1 | Config/IO/parse failure, or `--strict` with skipped unknown kinds / gate drops that operator wants fatal |

---

## 7. Conflict policy

### Overlays — merge-tighten (stricter wins)

For each proposal overlay matching `(market, vertical)`:

1. Find existing OP overlay with same market×vertical (case-normalize market upper / vertical lower to match serve resolve).
2. If none: append a new overlay entry with absolute thresholds derived from current global/overlay base **minus** only the tightening deltas (or use absolute fields if provided and stricter).
3. If exists: for each ladder key present (`soft_friction`, `hold_review`, `deny`), keep `min(existing, proposed)` (lower threshold = stricter / earlier friction). Never raise a threshold.
4. Preserve unrelated overlay keys; do not delete overlays not mentioned in proposals.

Delta application: if only deltas are present, resolve base = current overlay thresholds if present else global `decision_thresholds`, then `new = base + delta` with the constraint that resulting values must be ≤ base for tighten-only deltas (negative deltas expected). Positive deltas that would loosen → **reject that key** (log; do not apply).

### Effect / challenge — skip-if-id-exists

- If proposal `id` already exists in `rules` / `challenge_rules` → skip (no overwrite, no reorder).
- If new → append at end (first-match-wins semantics of existing engine unchanged for prior rules).
- Never delete or reorder existing rules.

### Dual targets in one run

A single proposal file may mix kinds. Overlay mutate triggers OP backup; effect_rules mutate should also backup `effect_rules.default.yaml` to `backup_dir` before write (mirror promote_overlays keep policy).

---

## 8. Flags and env

| Control | Effect |
|---|---|
| `rule_ingest.enabled` | Master on/off |
| `rule_ingest.live_enabled` | Allow `mode: live` from proposals |
| `INGEST_RULES_LIVE=1` | Force live-allowed for this process |
| `INGEST_PROPOSED_RULES=1` | Overnight `when_env` to run ingest step |
| `--dry-run` | No file writes |
| `--strict` | Non-zero exit on unknown kinds / soft gate drops |
| `--config PATH` | Alternate ingest config |

Shadow-by-default invariant: with default config and no env, every written rule has `mode: shadow` (or is not written).

---

## 9. Overnight wiring

Add optional step to `config/ops.overnight.yaml` / `ops.overnight.prod.yaml` **after** proposal producers (`segment_anomalies`, `graphbean_lite`) and **before** or **after** `promote_overlays` as follows:

Recommended order:

1. `segment_anomalies` (optional)
2. `graphbean_lite` (optional)
3. `ingest_proposed_rules` — `optional: true`, `when_env: INGEST_PROPOSED_RULES`
4. `promote_overlays` — unchanged (`when_env: PROMOTE_OVERLAYS`)

Rationale: ingest can merge proposal overlays into OP in shadow form; `promote_overlays` remains the backtest-driven path for slice-recommended overlays. Operators may enable one, both, or neither. Ingest does **not** imply promote eligibility.

Demo profile: leave `INGEST_PROPOSED_RULES` unset. Document in OPS_RUNBOOK §7 next to proposed-rules commands.

---

## 10. Success criteria

- [ ] Default config: ingest is a no-op or shadow-only; no live rule appears without `live_enabled` or `INGEST_RULES_LIVE=1`
- [ ] Overlay merge never loosens a threshold (unit tests: existing soft 40 + proposal 50 → stays 40; proposal 30 → becomes 30)
- [ ] Effect/challenge with duplicate `id` skipped; new `id` appended
- [ ] OP / effect_rules backup created before mutate; dry-run creates none
- [ ] Gates drop low-z / over-cap proposals; summary JSON reports accepted / skipped / reasons
- [ ] Overnight step runs only when `INGEST_PROPOSED_RULES` set
- [ ] Docs: MANUAL/OPS_RUNBOOK mention ingest + env flags; no public grade/rating language

---

## 11. Spec self-review

- [x] No placeholders for required behavior
- [x] No portal / auto-overnight-without-flag / full GraphBEAN / loosening in scope
- [x] Aligns with ARCHITECTURE (toolkit CLIs; Downstream owns enforcement)
- [x] Conflict policy and live/shadow split are explicit and testable
- [x] No letter grades or numeric project ratings in this spec
