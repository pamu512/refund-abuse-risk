# Promote / monitoring / effect hardening — design

> Privacy: historical filename retained for links. Internal quality ratings are private — see [docs/GRADING.md](../../GRADING.md).

Scope: tighten Phase 1–3 surfaces already shipped. No GraphBEAN, no live SDK.

## Goals

1. Discovery labels are PIT-honest and down-weighted; never count as proven.
2. ECE/PSI breaches block promote the same way costed soft floors do.
3. Budget can floor effect severity; shadow rules report dual-run rates in backtest.
4. Kill switch remains the only global escape for effect overrides.

## Changes

### Discovery mint
- Score UV edges from history **as-of each order’s `event_ts`**, excluding the order itself.
- Mint only when that as-of edge is elevated.
- Set `fraud_label_source=discovery`, `abuse_label_weak=1`.
- Never overwrite `proven`.
- `fraud_discovery_weight` (default 0.25) in `label_weights`; fraud head uses it for discovery source.
- Backtest: `fraud_discovery` slice; pattern/promote primary stays proven + cost + monitoring.

### Monitoring gates
- `operating_point.monitoring`: `max_decision_ece`, `max_psi_train_test`.
- Backtest sets `monitoring.ok` and folds into promote (`recommended.ok` ∧ monitoring.ok).

### Budget floor
- When `apply_as_effect_floor: true` (default), if budget `suggested_effect` is stricter than final effect, raise `refund_effect` and add `BUDGET_FLOOR`.
- Kill switch still disables effect-rule overrides; budget floor remains unless `budget.kill_switch` / disabled.

### Effect dual-run
- Backtest tallies base vs final vs shadow effect rates from snapshots under `effects_dual_run`.

## Follow-ups
- Per-slice decision ladders → `decision_threshold_overlays` + backtest recommend/promote.
- OOF stacker refit → stacker fit on OOF head probs; `fit_mode` audited in backtest.
- SDK ingest → `integrations/sdk_ingest.py` + `scripts/ingest_sdk_signals.py` (vendor-agnostic envelopes, confidence gate, claim-path `refresh_order` hooks). No vendor SDK dependency.
