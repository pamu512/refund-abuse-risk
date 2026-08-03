#!/usr/bin/env python3
"""Export entity/pair/combo behavior baselines for warehouse ETL (daily snapshot)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from refund_abuse_risk.baselines.cohort import CohortBaselineStore
from refund_abuse_risk.baselines.store import BehaviorBaselineStore
from refund_abuse_risk.config import load_behavior_baselines

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "behavior_baselines.db"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: data/warehouse/entity_behavior_baselines/as_of_date=YYYY-MM-DD/part.csv",
    )
    parser.add_argument("--as-of-date", default=None, help="Partition date (UTC), default today")
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--json-summary", action="store_true")
    args = parser.parse_args()

    cfg = load_behavior_baselines()
    version = str(cfg.get("baseline_version", "0.3.0"))
    as_of = args.as_of_date or datetime.now(timezone.utc).date().isoformat()
    out = args.out or (
        ROOT / "data" / "warehouse" / "entity_behavior_baselines" / f"as_of_date={as_of}" / "part.csv"
    )
    cohort_out = (
        ROOT
        / "data"
        / "warehouse"
        / str((cfg.get("warehouse") or {}).get("cohort_table", "cohort_behavior_baselines"))
        / f"as_of_date={as_of}"
        / "part.csv"
    )

    store = BehaviorBaselineStore(args.db)
    cohort = CohortBaselineStore(args.db)
    if args.clean_only:
        import pandas as pd
        from dataclasses import asdict

        rows = store.list_clean_baselines()
        frame = pd.DataFrame([asdict(r) for r in rows]) if rows else store.export_dataframe().iloc[0:0]
        if not frame.empty:
            frame.insert(0, "as_of_date", as_of)
            frame.insert(1, "baseline_version", version)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        n_rows = int(len(frame))
    else:
        store.export_csv(out, as_of_date=as_of, baseline_version=version)
        n_rows = int(store.export_dataframe().shape[0])
    cohort.export_csv(cohort_out, as_of_date=as_of, baseline_version=version)

    table = (cfg.get("warehouse") or {}).get("table", "entity_behavior_baselines")
    summary = {
        "table": table,
        "path": str(out),
        "cohort_table": (cfg.get("warehouse") or {}).get("cohort_table"),
        "cohort_path": str(cohort_out),
        "as_of_date": as_of,
        "baseline_version": version,
        "n_rows": n_rows,
        "n_clean": len(store.list_clean_baselines()),
        "n_cohorts": int(cohort.export_dataframe().shape[0]),
        "partition_keys": (cfg.get("warehouse") or {}).get(
            "partition_keys", ["as_of_date", "market", "vertical", "entity_kind"]
        ),
    }
    if args.json_summary:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Wrote {summary['n_rows']} entity rows ({summary['n_clean']} clean), "
            f"{summary['n_cohorts']} cohorts as_of={as_of}"
        )


if __name__ == "__main__":
    main()
