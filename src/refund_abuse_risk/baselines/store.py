"""Persistent entity/pair/combo behavioral baselines (head-score space)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from refund_abuse_risk.graph.entities import link_key

BASELINE_FEATURE_KINDS = ("user", "driver", "vendor", "device", "ud", "uv", "vd", "uvd")


@dataclass
class BaselineRow:
    entity_key: str
    entity_kind: str
    entity_id: str
    score_head: str
    market: str
    vertical: str
    n_observations: int
    n_under: int
    under_rate: float
    baseline_mean: float
    baseline_p50: float
    baseline_p90: float
    last_score: float
    elevated_streak: int
    under_streak: int
    is_clean_baseline: bool
    updated_at: str
    user_id: str = ""
    driver_id: str = ""
    vendor_id: str = ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _parse_ts(value: Any) -> datetime:
    if value is None or value == "":
        return _utc_now()
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def entity_keys_from_order(
    order: dict[str, Any],
    *,
    kinds: Iterable[str],
    kind_heads: dict[str, str],
    abuse_score: float,
    fraud_score: float,
) -> list[dict[str, Any]]:
    """Build observation payloads: key, kind, id, head score, id parts, support."""
    user_id = str(order.get("user_id", "") or "")
    driver_id = str(order.get("driver_id", "") or "")
    vendor_id = str(order.get("vendor_id", "") or "")
    device_id = str(order.get("device_id", "") or "")
    kind_set = {str(k) for k in kinds}
    get = order.get

    def head_score(kind: str) -> float:
        head = str(kind_heads.get(kind, "abuse"))
        return float(fraud_score if head == "fraud" else abuse_score)

    singles = {
        "user": (user_id, {"user_id": user_id}, 1.0),
        "driver": (driver_id, {"driver_id": driver_id}, 1.0),
        "vendor": (vendor_id, {"vendor_id": vendor_id}, 1.0),
        "device": (device_id, {"device_id": device_id, "user_id": user_id}, 1.0),
    }
    pairs = {
        "ud": (
            f"{user_id}|{driver_id}",
            {"user_id": user_id, "driver_id": driver_id},
            float(get("ud_cooccur", 0) or 0),
        ),
        "uv": (
            f"{user_id}|{vendor_id}",
            {"user_id": user_id, "vendor_id": vendor_id},
            float(get("uv_cooccur", 0) or 0),
        ),
        "vd": (
            f"{vendor_id}|{driver_id}",
            {"vendor_id": vendor_id, "driver_id": driver_id},
            float(get("vd_cooccur", 0) or 0),
        ),
        "uvd": (
            f"{user_id}|{vendor_id}|{driver_id}",
            {"user_id": user_id, "vendor_id": vendor_id, "driver_id": driver_id},
            float(get("uvd_cooccur", 0) or 0),
        ),
    }
    out: list[dict[str, Any]] = []
    for kind, (eid, parts, support) in {**singles, **pairs}.items():
        if kind not in kind_set or not eid or eid.startswith("|") or eid.endswith("|"):
            continue
        if kind in singles:
            key = link_key(kind, eid)
        elif kind == "ud":
            key = link_key("ud", user_id, driver_id)
        elif kind == "uv":
            key = link_key("uv", user_id, vendor_id)
        elif kind == "vd":
            key = link_key("vd", vendor_id, driver_id)
        else:
            key = link_key("uvd", user_id, vendor_id, driver_id)
        out.append(
            {
                "entity_key": key,
                "entity_kind": kind,
                "entity_id": eid,
                "score": head_score(kind),
                "score_head": str(kind_heads.get(kind, "abuse")),
                "parts": parts,
                "support": float(support),
            }
        )
    return out


class BehaviorBaselineStore:
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
                CREATE TABLE IF NOT EXISTS behavior_baselines (
                  entity_key TEXT NOT NULL,
                  entity_kind TEXT NOT NULL,
                  entity_id TEXT NOT NULL,
                  score_head TEXT NOT NULL DEFAULT 'abuse',
                  market TEXT NOT NULL DEFAULT '',
                  vertical TEXT NOT NULL DEFAULT '',
                  user_id TEXT NOT NULL DEFAULT '',
                  driver_id TEXT NOT NULL DEFAULT '',
                  vendor_id TEXT NOT NULL DEFAULT '',
                  n_observations INTEGER NOT NULL DEFAULT 0,
                  n_under INTEGER NOT NULL DEFAULT 0,
                  under_rate REAL NOT NULL DEFAULT 0,
                  baseline_mean REAL NOT NULL DEFAULT 0,
                  baseline_p50 REAL NOT NULL DEFAULT 0,
                  baseline_p90 REAL NOT NULL DEFAULT 0,
                  last_score REAL NOT NULL DEFAULT 0,
                  elevated_streak INTEGER NOT NULL DEFAULT 0,
                  under_streak INTEGER NOT NULL DEFAULT 0,
                  is_clean_baseline INTEGER NOT NULL DEFAULT 0,
                  score_history_json TEXT NOT NULL DEFAULT '[]',
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (entity_key, market, vertical)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS baseline_obs_dedupe (
                  order_id TEXT NOT NULL,
                  entity_key TEXT NOT NULL,
                  market TEXT NOT NULL DEFAULT '',
                  vertical TEXT NOT NULL DEFAULT '',
                  observed_at TEXT NOT NULL,
                  PRIMARY KEY (order_id, entity_key, market, vertical)
                )
                """
            )
            # Lightweight migration for older DBs missing score_head.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(behavior_baselines)").fetchall()}
            if "score_head" not in cols:
                conn.execute(
                    "ALTER TABLE behavior_baselines ADD COLUMN score_head TEXT NOT NULL DEFAULT 'abuse'"
                )
            conn.commit()

    def has_observation(
        self,
        order_id: str,
        entity_key: str,
        *,
        market: str = "",
        vertical: str = "",
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM baseline_obs_dedupe
                WHERE order_id = ? AND entity_key = ? AND market = ? AND vertical = ?
                """,
                (str(order_id), entity_key, market or "", vertical or ""),
            ).fetchone()
        return row is not None

    def get(
        self,
        entity_key: str,
        *,
        market: str = "",
        vertical: str = "",
    ) -> BaselineRow | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM behavior_baselines
                WHERE entity_key = ? AND market = ? AND vertical = ?
                """,
                (entity_key, market or "", vertical or ""),
            ).fetchone()
        if row is None:
            return None
        return self._to_row(row)

    def update_observation(
        self,
        *,
        order_id: str,
        entity_key: str,
        entity_kind: str,
        entity_id: str,
        score: float,
        score_head: str,
        under_threshold: float,
        event_ts: Any = None,
        market: str = "",
        vertical: str = "",
        user_id: str = "",
        driver_id: str = "",
        vendor_id: str = "",
        lookback_events: int = 50,
        lookback_days: int = 30,
        min_observations: int = 10,
        min_under_rate: float = 0.80,
        revoke_under_rate: float = 0.50,
        elevated_min_delta: float = 15.0,
        elevated_min_lift: float = 1.5,
        pair_min_cooccur: float = 0.0,
        support: float = 1.0,
        skip_if_duplicate: bool = True,
    ) -> BaselineRow | None:
        """
        Append one settled observation. Returns None if duplicate order×entity.
        """
        market = market or ""
        vertical = vertical or ""
        order_id = str(order_id or "")
        if not order_id:
            return None
        if skip_if_duplicate and self.has_observation(
            order_id, entity_key, market=market, vertical=vertical
        ):
            return self.get(entity_key, market=market, vertical=vertical)

        # Pairs without support do not qualify / update clean path — still record score.
        pair_kinds = {"ud", "uv", "vd", "uvd"}
        support_ok = (entity_kind not in pair_kinds) or (
            float(support) >= float(pair_min_cooccur)
        )

        ts = _parse_ts(event_ts)
        ts_iso = ts.isoformat().replace("+00:00", "Z")
        score = float(score)

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM behavior_baselines
                WHERE entity_key = ? AND market = ? AND vertical = ?
                """,
                (entity_key, market, vertical),
            ).fetchone()
            was_clean = False
            if existing is None:
                history: list[dict[str, Any]] = []
                under_streak = 0
                elevated_streak = 0
            else:
                history = json.loads(existing["score_history_json"] or "[]")
                under_streak = int(existing["under_streak"])
                elevated_streak = int(existing["elevated_streak"])
                was_clean = bool(existing["is_clean_baseline"])

            history.append({"ts": ts_iso, "score": score})
            # Normalize legacy float-only histories from older schema.
            normalized: list[dict[str, Any]] = []
            for item in history:
                if isinstance(item, dict):
                    normalized.append(
                        {"ts": str(item.get("ts") or ts_iso), "score": float(item.get("score", 0))}
                    )
                else:
                    normalized.append({"ts": ts_iso, "score": float(item)})
            history = normalized
            cutoff = ts - timedelta(days=int(lookback_days))
            trimmed: list[dict[str, Any]] = []
            for item in history:
                item_ts = _parse_ts(item.get("ts"))
                if item_ts >= cutoff:
                    trimmed.append(item)
            history = trimmed[-int(lookback_events) :]

            scores = [float(h["score"]) for h in history]
            n_obs = len(scores)
            n_under = sum(1 for s in scores if s < float(under_threshold))
            under_rate = float(n_under / n_obs) if n_obs else 0.0
            under = score < float(under_threshold)
            under_streak = under_streak + 1 if under else 0

            under_scores = [s for s in scores if s < float(under_threshold)]
            if under_scores:
                baseline_mean = float(sum(under_scores) / len(under_scores))
                sorted_u = sorted(under_scores)
                baseline_p50 = float(sorted_u[len(sorted_u) // 2])
                baseline_p90 = float(sorted_u[max(0, int(len(sorted_u) * 0.9) - 1)])
            else:
                baseline_mean = float(sum(scores) / len(scores)) if scores else 0.0
                sorted_h = sorted(scores) if scores else [0.0]
                baseline_p50 = float(sorted_h[len(sorted_h) // 2])
                baseline_p90 = float(sorted_h[max(0, int(len(sorted_h) * 0.9) - 1)])

            eps = 1e-6
            lift = score / max(baseline_mean, eps)
            elevated = (score - baseline_mean) >= float(elevated_min_delta) or lift >= float(
                elevated_min_lift
            )
            elevated_streak = elevated_streak + 1 if elevated and not under else 0

            earned = (
                support_ok
                and n_obs >= int(min_observations)
                and under_rate >= float(min_under_rate)
            )
            if was_clean and under_rate < float(revoke_under_rate):
                is_clean = False
            else:
                is_clean = was_clean or earned

            updated_at = _utc_now_iso()
            conn.execute(
                """
                INSERT INTO behavior_baselines(
                  entity_key, entity_kind, entity_id, score_head, market, vertical,
                  user_id, driver_id, vendor_id,
                  n_observations, n_under, under_rate,
                  baseline_mean, baseline_p50, baseline_p90,
                  last_score, elevated_streak, under_streak,
                  is_clean_baseline, score_history_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_key, market, vertical) DO UPDATE SET
                  entity_kind=excluded.entity_kind,
                  entity_id=excluded.entity_id,
                  score_head=excluded.score_head,
                  user_id=excluded.user_id,
                  driver_id=excluded.driver_id,
                  vendor_id=excluded.vendor_id,
                  n_observations=excluded.n_observations,
                  n_under=excluded.n_under,
                  under_rate=excluded.under_rate,
                  baseline_mean=excluded.baseline_mean,
                  baseline_p50=excluded.baseline_p50,
                  baseline_p90=excluded.baseline_p90,
                  last_score=excluded.last_score,
                  elevated_streak=excluded.elevated_streak,
                  under_streak=excluded.under_streak,
                  is_clean_baseline=excluded.is_clean_baseline,
                  score_history_json=excluded.score_history_json,
                  updated_at=excluded.updated_at
                """,
                (
                    entity_key,
                    entity_kind,
                    entity_id,
                    score_head,
                    market,
                    vertical,
                    user_id,
                    driver_id,
                    vendor_id,
                    n_obs,
                    n_under,
                    under_rate,
                    baseline_mean,
                    baseline_p50,
                    baseline_p90,
                    score,
                    elevated_streak,
                    under_streak,
                    1 if is_clean else 0,
                    json.dumps(history),
                    updated_at,
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO baseline_obs_dedupe(
                  order_id, entity_key, market, vertical, observed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (order_id, entity_key, market, vertical, updated_at),
            )
            conn.commit()

        return self.get(entity_key, market=market, vertical=vertical)

    def list_all(self) -> list[BaselineRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM behavior_baselines ORDER BY updated_at DESC"
            ).fetchall()
        return [self._to_row(r) for r in rows]

    def list_clean_baselines(self) -> list[BaselineRow]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM behavior_baselines
                WHERE is_clean_baseline = 1
                ORDER BY entity_kind, entity_id
                """
            ).fetchall()
        return [self._to_row(r) for r in rows]

    def export_dataframe(self, *, as_of_date: str | None = None, baseline_version: str = "") -> pd.DataFrame:
        rows = self.list_all()
        as_of = as_of_date or _utc_now().date().isoformat()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "as_of_date",
                    "baseline_version",
                    "entity_key",
                    "entity_kind",
                    "entity_id",
                    "score_head",
                    "market",
                    "vertical",
                    "user_id",
                    "driver_id",
                    "vendor_id",
                    "n_observations",
                    "n_under",
                    "under_rate",
                    "baseline_mean",
                    "baseline_p50",
                    "baseline_p90",
                    "last_score",
                    "elevated_streak",
                    "under_streak",
                    "is_clean_baseline",
                    "updated_at",
                ]
            )
        frame = pd.DataFrame([asdict(r) for r in rows])
        frame.insert(0, "as_of_date", as_of)
        frame.insert(1, "baseline_version", baseline_version)
        return frame

    def export_csv(
        self,
        path: Path | str,
        *,
        as_of_date: str | None = None,
        baseline_version: str = "",
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.export_dataframe(as_of_date=as_of_date, baseline_version=baseline_version).to_csv(
            path, index=False
        )
        return path

    def feature_map_for_order(
        self,
        order: dict[str, Any],
        *,
        kinds: Iterable[str] | None = None,
    ) -> dict[str, float]:
        """Read-only baseline features for model input (defaults 0 / lift 1)."""
        kinds = list(kinds or BASELINE_FEATURE_KINDS)
        market = str(order.get("market", "") or "")
        vertical = str(order.get("vertical", "") or "")
        abuse = float(order.get("abuse_score", 0) or 0)
        fraud = float(order.get("fraud_score", 0) or 0)
        # Use placeholder scores only for key construction; features come from store.
        payloads = entity_keys_from_order(
            order,
            kinds=kinds,
            kind_heads={k: "abuse" for k in kinds},
            abuse_score=abuse,
            fraud_score=fraud,
        )
        # Rebuild keys without depending on kind_heads for lookup
        user_id = str(order.get("user_id", "") or "")
        driver_id = str(order.get("driver_id", "") or "")
        vendor_id = str(order.get("vendor_id", "") or "")
        device_id = str(order.get("device_id", "") or "")
        key_by_kind = {
            "user": link_key("user", user_id) if user_id else "",
            "driver": link_key("driver", driver_id) if driver_id else "",
            "vendor": link_key("vendor", vendor_id) if vendor_id else "",
            "device": link_key("device", device_id) if device_id else "",
            "ud": link_key("ud", user_id, driver_id) if user_id and driver_id else "",
            "uv": link_key("uv", user_id, vendor_id) if user_id and vendor_id else "",
            "vd": link_key("vd", vendor_id, driver_id) if vendor_id and driver_id else "",
            "uvd": link_key("uvd", user_id, vendor_id, driver_id)
            if user_id and vendor_id and driver_id
            else "",
        }
        out: dict[str, float] = {}
        for kind in BASELINE_FEATURE_KINDS:
            prefix = f"{kind}_baseline"
            out[f"{prefix}_lift"] = 1.0
            out[f"{prefix}_under_rate"] = 0.0
            out[f"{prefix}_elevated_streak"] = 0.0
            out[f"{kind}_is_clean_baseline"] = 0.0
            key = key_by_kind.get(kind, "")
            if not key:
                continue
            row = self.get(key, market=market, vertical=vertical)
            if row is None:
                continue
            eps = 1e-6
            # Lift of last known score vs baseline (feature for next decision).
            lift = float(row.last_score) / max(float(row.baseline_mean), eps)
            out[f"{prefix}_lift"] = float(lift)
            out[f"{prefix}_under_rate"] = float(row.under_rate)
            out[f"{prefix}_elevated_streak"] = float(row.elevated_streak)
            out[f"{kind}_is_clean_baseline"] = 1.0 if row.is_clean_baseline else 0.0
        _ = payloads
        return out

    @staticmethod
    def _to_row(row: sqlite3.Row) -> BaselineRow:
        keys = row.keys()
        return BaselineRow(
            entity_key=str(row["entity_key"]),
            entity_kind=str(row["entity_kind"]),
            entity_id=str(row["entity_id"]),
            score_head=str(row["score_head"] if "score_head" in keys else "abuse"),
            market=str(row["market"]),
            vertical=str(row["vertical"]),
            n_observations=int(row["n_observations"]),
            n_under=int(row["n_under"]),
            under_rate=float(row["under_rate"]),
            baseline_mean=float(row["baseline_mean"]),
            baseline_p50=float(row["baseline_p50"]),
            baseline_p90=float(row["baseline_p90"]),
            last_score=float(row["last_score"]),
            elevated_streak=int(row["elevated_streak"]),
            under_streak=int(row["under_streak"]),
            is_clean_baseline=bool(row["is_clean_baseline"]),
            updated_at=str(row["updated_at"]),
            user_id=str(row["user_id"] or ""),
            driver_id=str(row["driver_id"] or ""),
            vendor_id=str(row["vendor_id"] or ""),
        )


def zero_baseline_features() -> dict[str, float]:
    out: dict[str, float] = {}
    for kind in BASELINE_FEATURE_KINDS:
        out[f"{kind}_baseline_lift"] = 1.0
        out[f"{kind}_baseline_under_rate"] = 0.0
        out[f"{kind}_baseline_elevated_streak"] = 0.0
        out[f"{kind}_is_clean_baseline"] = 0.0
    out["user_cohort_lift"] = 1.0
    out["driver_cohort_lift"] = 1.0
    out["vendor_cohort_lift"] = 1.0
    out["device_cohort_lift"] = 1.0
    out["tenure_bucket_code"] = 1.0
    return out
