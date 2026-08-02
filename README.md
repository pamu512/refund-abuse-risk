# refund-abuse-risk

Graph/ML toolkit for **refund abuse** and **fraud** scoring on food delivery and quick commerce.

Abuse ≠ fraud: abuse is behavioral / policy gaming; fraud is proven collusion or proveable bad behavior (plus down-weighted high-confidence proxies).

## What it emits

For each order (precomputed for async claims):

- `abuse_score` / `fraud_score` (0–100)
- entity + link scores (user, driver, vendor, UD/UV/VD/UVD) and device cluster score
- **suggested** 4-tier decision (`auto_approve` / `soft_friction` / `hold_review` / `auto_deny`)
- reason codes + full evidence pack

Downstream policy services own enforcement. This package scores and advises.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_demo_data.py
pytest -q
python -m examples.csv_demo
python scripts/backtest.py
```

## Layout

| Path | Role |
|---|---|
| `config/policy.default.yaml` | Ops-tunable hard gates / evidence weights (market × vertical × entity) |
| `config/label_weights.default.yaml` | Proven vs proxy sample weights + proxy rules |
| `config/operating_point.default.yaml` | Tier bands + entity prior bands |
| `src/refund_abuse_risk/features/` | Entity / link / device / order feature builders |
| `src/refund_abuse_risk/model/` | Two-head trainer (abuse + fraud) |
| `src/refund_abuse_risk/scoring/` | Policy overlay, hard gates, evidence pack |
| `src/refund_abuse_risk/pipeline/` | Precompute cache, lifecycle + risk-change refresh, claim-path read |

## Design

See [`docs/superpowers/specs/2026-08-02-refund-abuse-risk-design.md`](docs/superpowers/specs/2026-08-02-refund-abuse-risk-design.md).
