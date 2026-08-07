# Platform leverage ranking + P0/P1 design

## Rank (by leverage for *this* toolkit)

| Rank | Item | Why this order |
|---|---|---|
| **P0** | Segment refund anomaly → **proposed rules only** | Highest $/week leverage: catches new MOs without GraphBEAN; matches DoorDash/RADAR; safe (no auto-enforce) |
| **P1a** | Append-only **decision archive** | Unlocks CS/DS feedback, promote debugging, offline LF mining (Grab Archivist); small code, permanent compounding value |
| **P1b** | **Weak-label factory** (LF → proxy labels) | Multiplies training signal when proven is thin (Swiggy); must stay below proven on promote metrics |
| **P2a** | GraphBEAN-lite unsupervised | **Shipped (lite):** Ridge recon + structure residual on UV edges; proposals only. Full GraphBEAN/RGCN still deferred |
| **P2b** | Risk-challenge UX effects | **Shipped:** `RiskChallenge` + `challenge_rules` on score path (orthogonal to `RefundEffect`) |
| **Cutover** | Chargeback maturity / ORC process | Downstream; not an in-repo feature |

## P0 — Segment anomaly (propose only)

- Input: orders with `event_ts`, `market`, `vertical`, refund/claim signal
- Metric: daily refund rate per `market|vertical` (optional claim_reason)
- Detector: moving-window z-score with baseline / gap / test day (DoorDash-shaped)
- Output: `data/proposed_rules/*.yaml` + JSON report — **never** written into live OP or effect rules automatically
- Overnight: optional step after backtest

## P1a — Decision archive

- SQLite append-only: order_id, scores, tier, effect, overlay, model/policy versions, reason_codes
- Hook: `score_feature_row` when `DECISION_ARCHIVE_PATH` set (or explicit archive arg)
- Query helper for CS/DS

## P1b — Weak-label factory

- Pure-Python labeling functions (no Snorkel dep): abstain / 0 / 1
- Combine via majority of non-abstain; write `abuse_label_weak` / discovery source
- **Never** overwrite `fraud_label_source=proven`
- Proven-primary metrics unchanged

## P2a — GraphBEAN-lite

- Input: history with user/vendor/market/vertical/`is_refund`
- Model: Ridge feature decoder + degree structure residual on `score_uv_bipartite` edges
- Output: edge/node scores, MO tags, `proposed_actions` with `auto_enforce: false`
- Script: `scripts/run_graphbean_lite.py` → `data/proposed_rules/graphbean_lite.*`
- Ceiling: linear recon; upgrade = Grab GraphBEAN / RGCN on same edge schema

## P2b — Risk challenges

- Enum: `none` / `payment_verify` / `identity_verify` / `in_app_capture`
- Config: `challenge_rules` + `challenges_enabled` in effect_rules YAML
- Wired in `score_feature_row` → `risk_challenge` / `shadow_risk_challenge`
- Orthogonal to `RefundEffect` (Uber penny-drop-shaped friction on soft/hold)

## Non-goals

- Auto-promoting proposed rules / GraphBEAN actions
- Full neural GraphBEAN
- Snorkel as a dependency
