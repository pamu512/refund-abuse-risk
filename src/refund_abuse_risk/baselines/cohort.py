"""Market × vertical × tenure cohort baselines (Stripe-style peer reference)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def tenure_bucket(days_since_signup: float, buckets: list[dict[str, Any]] | None = None) -> str:
    """Map tenure days to a coarse bucket (new / mid / mature by default)."""
    days = max(0.0, float(days_since_signup or 0.0))
    rules = buckets or [
        {"name": "new", "max_days": 14},
        {"name": "mid", "max_days": 90},
        {"name": "mature", "max_days": None},
    ]
    for rule in rules:
        max_days = rule.get("max_days")
        if max_days is None or days < float(max_days):
            return str(rule.get("name", "unknown"))
    return str(rules[-1].get("name", "mature"))


def cohort_key(market: str, vertical: str, bucket: str, score_head: str) -> str:
    return "|".join(
        [
            (market or "").strip().upper() or "ALL",
            (vertical or "").strip().lower() or "all",
            str(bucket),
            str(score_head),
        ]
    )


@dataclass
class CohortRow:
    cohort_key: str
    market: str
    vertical: str
    tenure_bucket: str
    score_head: str
    n_observations: int
    score_mean: float
    score_p50: float
    score_p90: float
    updated_at: str


class CohortBaselineStore:
    """Rolling peer baselines for head scores within market × vertical × tenure."""

    def __init__(self, sqlite_path: Path | str) -> None:
        self._path = Path(sqlite_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cohort_baselines (
                  cohort_key TEXT PRIMARY KEY,
                  market TEXT NOT NULL,
                  vertical TEXT NOT NULL,
                  tenure_bucket TEXT NOT NULL,
                  score_head TEXT NOT NULL,
                  n_observations INTEGER NOT NULL DEFAULT 0,
                  score_mean REAL NOT NULL DEFAULT 0,
                  score_p50 REAL NOT NULL DEFAULT 0,
                  score_p90 REAL NOT NULL DEFAULT 0,
                  score_history_json TEXT NOT NULL DEFAULT '[]',
                  updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, key: str) -> CohortRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cohort_baselines WHERE cohort_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return self._to_row(row)

    def update_score(
        self,
        *,
        market: str,
        vertical: str,
        tenure_bucket_name: str,
        score_head: str,
        score: float,
        lookback_events: int = 500,
    ) -> CohortRow:
        key = cohort_key(market, vertical, tenure_bucket_name, score_head)
        score = float(score)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM cohort_baselines WHERE cohort_key = ?", (key,)
            ).fetchone()
            history: list[float]
            if existing is None:
                history = []
            else:
                history = [float(x) for x in json.loads(existing["score_history_json"] or "[]")]
            history.append(score)
            history = history[-int(lookback_events) :]
            n = len(history)
            mean = float(sum(history) / n) if n else 0.0
            sorted_h = sorted(history)
            p50 = float(sorted_h[len(sorted_h) // 2]) if n else 0.0
            p90 = float(sorted_h[max(0, int(n * 0.9) - 1)]) if n else 0.0
            updated_at = _utc_now_iso()
            conn.execute(
                """
                INSERT INTO cohort_baselines(
                  cohort_key, market, vertical, tenure_bucket, score_head,
                  n_observations, score_mean, score_p50, score_p90,
                  score_history_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cohort_key) DO UPDATE SET
                  n_observations=excluded.n_observations,
                  score_mean=excluded.score_mean,
                  score_p50=excluded.score_p50,
                  score_p90=excluded.score_p90,
                  score_history_json=excluded.score_history_json,
                  updated_at=excluded.updated_at
                """,
                (
                    key,
                    (market or "").strip().upper() or "ALL",
                    (vertical or "").strip().lower() or "all",
                    tenure_bucket_name,
                    score_head,
                    n,
                    mean,
                    p50,
                    p90,
                    json.dumps(history),
                    updated_at,
                ),
            )
            conn.commit()
        row = self.get(key)
        assert row is not None
        return row

    def lift(
        self,
        score: float,
        *,
        market: str,
        vertical: str,
        tenure_bucket_name: str,
        score_head: str,
        min_support: int = 30,
    ) -> tuple[float, float, int]:
        """Return (lift_vs_p50, cohort_p50, n). Lift=1.0 if under-supported."""
        row = self.get(cohort_key(market, vertical, tenure_bucket_name, score_head))
        if row is None or row.n_observations < int(min_support):
            return 1.0, 0.0, int(row.n_observations if row else 0)
        eps = 1e-6
        p50 = max(float(row.score_p50), eps)
        return float(score) / p50, float(row.score_p50), int(row.n_observations)

    def export_dataframe(self, *, as_of_date: str | None = None, baseline_version: str = "") -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM cohort_baselines").fetchall()
        as_of = as_of_date or datetime.now(timezone.utc).date().isoformat()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "as_of_date",
                    "baseline_version",
                    "cohort_key",
                    "market",
                    "vertical",
                    "tenure_bucket",
                    "score_head",
                    "n_observations",
                    "score_mean",
                    "score_p50",
                    "score_p90",
                    "updated_at",
                ]
            )
        frame = pd.DataFrame([asdict(self._to_row(r)) for r in rows])
        frame.insert(0, "as_of_date", as_of)
        frame.insert(1, "baseline_version", baseline_version)
        return frame

    def export_csv(self, path: Path | str, *, as_of_date: str | None = None, baseline_version: str = "") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.export_dataframe(as_of_date=as_of_date, baseline_version=baseline_version).to_csv(
            path, index=False
        )
        return path

    @staticmethod
    def _to_row(row: sqlite3.Row) -> CohortRow:
        return CohortRow(
            cohort_key=str(row["cohort_key"]),
            market=str(row["market"]),
            vertical=str(row["vertical"]),
            tenure_bucket=str(row["tenure_bucket"]),
            score_head=str(row["score_head"]),
            n_observations=int(row["n_observations"]),
            score_mean=float(row["score_mean"]),
            score_p50=float(row["score_p50"]),
            score_p90=float(row["score_p90"]),
            updated_at=str(row["updated_at"]),
        )
