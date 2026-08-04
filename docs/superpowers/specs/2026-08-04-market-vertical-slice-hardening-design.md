# Market × vertical slice hardening — design

Approved 2026-08-04. Option 2: eval + seeded overlays + policy features. No per-slice models.

## Goals

1. Encode vertical/market **policy priors** as features (claim window, photo prior, cash bias, returns).
2. Backtest reports **proven** slice metrics + per-slice ECE; non-thin slices gate promote on ECE.
3. Emit / optionally write **decision_threshold_overlays** from costed per-slice recommend.

## Changes

### Config
- `config/vertical_policy.default.yaml` — defaults by vertical; optional market overlays.
- `load_vertical_policy()` in `config.py`.

### Features
- `policy_claim_window_hours`, `policy_photo_prior`, `policy_remedy_cash_bias`, `policy_returns_allowed`
- `hours_since_delivery`, `claim_window_remaining_frac` (neutral 0 if no delivery/event ts)
- Lookup: market overlay wins else vertical default else food defaults.

### Backtest
- `slices[key]`: add proven AP / precision@soft / n_proven / decision_ece
- `slices_monitoring.ok` = all non-thin slices with n≥min clear `max_decision_ece`
- Fold into `monitoring.ok`
- Always emit `recommended_overlays` list; `--write-slice-overlays PATH` writes YAML fragment

### Out of scope
- Separate models per slice; production feed runner; committing demo data churn.
