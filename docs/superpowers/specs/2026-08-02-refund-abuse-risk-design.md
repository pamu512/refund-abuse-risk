# Refund Abuse / Fraud Risk — Design

Date: 2026-08-02  
Status: implemented (v0.1)

## Problem

Food delivery and quick commerce refund paths leak money through serial claiming, device farms, and collusion rings (user–driver–vendor). Weak historical policy means approved refunds are not clean negatives. Ops need tunable policy-as-code plus ML that separates **abuse** (behavioral gaming) from **fraud** (proven or proveable collusion/behavior).

## Goals

1. Two scores: `abuse_score`, `fraud_score` (0–100), shared graph/device/order features.
2. Standing ranks for user, driver, vendor, UD/UV/VD/UVD links, device clusters.
3. Precomputed order snapshots for async claims; sync path is cache read only.
4. Suggested 4-tier decision + reason codes + evidence pack; enforcement is downstream.
5. Ops-tunable YAML policy scoped by market × vertical × entity type.

## Non-goals

- Payment blocking / account suspension execution
- GNN / streaming graph DB
- Legal case workflow (consumes proven labels only)

## Score semantics

| Term | Meaning |
|---|---|
| Abuse | Behavioral pattern / policy gaming; may lack legal proof |
| Fraud | Investigator-confirmed collusion/behavior, or high-confidence proxy (sample weight &lt; 1) |

Fraud proxies (configurable): large device cluster ∧ multi-account device ∧ elevated UVD refund lift ∧ high refund rate.

## Architecture

1. **Feature builders** — rolling refund counts/rates/%GMV; device cluster stats; link co-occurrence + refund lift; order/claim context.
2. **Two-head model** — calibrated gradient boosting per head; proxy sample weights on fraud head.
3. **Entity prior band** — entity/link/device prior sets floor/ceiling; order head scores move within band.
4. **Policy overlay** — hard gates from strong fraud labels, rolling caps, high link/device scores → force `auto_deny`.
5. **Cache** — precompute on lifecycle events; rescore open orders when entity/link risk moves by ≥ configured delta.

## Output contract

`OrderRiskSnapshot`: scores, entity/link/device scores, `suggested_tier`, `reason_codes`, `evidence_pack`, `hard_gated`, versions, `scored_at`.

## Verification

- `pytest` unit/integration tests
- `python -m examples.csv_demo` writes scored snapshots
- `python scripts/backtest.py` reports abuse/fraud metrics and proven vs proxy slices
