# CSV demo

Zero-network end-to-end on synthetic CSVs under `data/`.

```bash
pip install -e ".[dev]"
python scripts/generate_demo_data.py
python -m examples.csv_demo
```

Writes `examples/csv_demo/out.json` with order snapshots (abuse/fraud scores, suggested tier, reason codes, evidence pack).

See the repo [README](../../README.md) for how scoring works, and [docs/MANUAL.md](../../docs/MANUAL.md) for tuning and operations.
