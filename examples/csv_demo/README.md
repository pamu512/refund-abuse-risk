# CSV demo

Zero-network end-to-end on synthetic CSVs under `data/`.

```bash
pip install -e ".[dev]"
python scripts/generate_demo_data.py
python -m examples.csv_demo
python scripts/backtest.py
```

Outputs:

- `examples/csv_demo/out.json` — scored order snapshots (abuse/fraud, tier, evidence)
- `examples/csv_demo/backtest_metrics.json` — holdout metrics by head and proven/proxy fraud slices
