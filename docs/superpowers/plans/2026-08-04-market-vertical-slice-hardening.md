# Market × vertical slice hardening — Implementation Plan

> **For agentic workers:** Implement tasks in order; each ends with a runnable check.

**Goal:** Policy prior features + proven slice eval/ECE promote gate + overlay suggest/write.

**Architecture:** YAML policy priors → feature builder; backtest enriches `slices` and folds slice ECE into `monitoring.ok`; recommend_by_slice → overlay list / optional YAML write.

**Tech stack:** existing pandas / sklearn / pytest / PyYAML.

## File map

- `config/vertical_policy.default.yaml` (new)
- `src/refund_abuse_risk/config.py` — loader
- `src/refund_abuse_risk/features/policy_priors.py` (new) — lookup + feature dict
- `src/refund_abuse_risk/features/builders.py` — wire columns
- `scripts/backtest.py` — slice proven/ECE, promote, `--write-slice-overlays`
- `tests/test_policy_priors_and_slices.py` (new)

## Tasks

### Task 1: Policy priors + features
- Add YAML + `load_vertical_policy` + `policy_prior_features(order, cfg)`
- Wire into FEATURE/ABUSE/FRAUD columns and `build_order_features`
- Test: food vs qcommerce windows; remaining_frac clamp

### Task 2: Backtest slice honesty + overlays
- Proven metrics + ECE per slice; thin skip; fold into monitoring.ok
- Emit `recommended_overlays`; CLI write flag
- Test: thin slice does not fail gate; overlay shape

### Task 3: Docs touch
- README one-liner on vertical_policy + write-slice-overlays
