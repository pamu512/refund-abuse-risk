"""Append-only decision / score archive (Grab Archivist-shaped, SQLite)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class DecisionArchive:
    """Persist every score decision for CS/DS feedback and offline mining."""

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
                CREATE TABLE IF NOT EXISTS decision_archive (
                  archive_id TEXT PRIMARY KEY,
                  ts TEXT NOT NULL,
                  order_id TEXT NOT NULL,
                  market TEXT NOT NULL DEFAULT '',
                  vertical TEXT NOT NULL DEFAULT '',
                  abuse_score REAL,
                  fraud_score REAL,
                  decision_score REAL,
                  suggested_tier TEXT,
                  refund_effect TEXT,
                  shadow_refund_effect TEXT,
                  hard_gated INTEGER NOT NULL DEFAULT 0,
                  model_version TEXT,
                  policy_version TEXT,
                  reason_codes_json TEXT,
                  effect_decision_json TEXT,
                  extras_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decision_archive_order "
                "ON decision_archive(order_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decision_archive_ts "
                "ON decision_archive(ts)"
            )
            conn.commit()

    def append(
        self,
        *,
        order_id: str,
        market: str = "",
        vertical: str = "",
        abuse_score: float | None = None,
        fraud_score: float | None = None,
        decision_score: float | None = None,
        suggested_tier: str = "",
        refund_effect: str = "",
        shadow_refund_effect: str | None = None,
        hard_gated: bool = False,
        model_version: str = "",
        policy_version: str = "",
        reason_codes: list[str] | None = None,
        effect_decision: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
        ts: str | None = None,
    ) -> str:
        archive_id = uuid4().hex
        stamped = ts or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decision_archive(
                  archive_id, ts, order_id, market, vertical,
                  abuse_score, fraud_score, decision_score,
                  suggested_tier, refund_effect, shadow_refund_effect,
                  hard_gated, model_version, policy_version,
                  reason_codes_json, effect_decision_json, extras_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    archive_id,
                    stamped,
                    str(order_id or ""),
                    market or "",
                    vertical or "",
                    abuse_score,
                    fraud_score,
                    decision_score,
                    suggested_tier or "",
                    refund_effect or "",
                    shadow_refund_effect,
                    1 if hard_gated else 0,
                    model_version or "",
                    policy_version or "",
                    json.dumps(list(reason_codes or [])),
                    json.dumps(effect_decision or {}),
                    json.dumps(extras or {}),
                ),
            )
            conn.commit()
        return archive_id

    def append_snapshot(self, snapshot: Any) -> str:
        """Accept OrderRiskSnapshot or a mapping with the same fields."""

        def _get(key: str, default: Any = None) -> Any:
            if isinstance(snapshot, dict):
                return snapshot.get(key, default)
            return getattr(snapshot, key, default)

        tier = _get("suggested_tier")
        tier_s = tier.value if hasattr(tier, "value") else str(tier or "")
        effect = _get("refund_effect")
        effect_s = effect.value if hasattr(effect, "value") else str(effect or "")
        shadow = _get("shadow_refund_effect")
        if shadow is not None and hasattr(shadow, "value"):
            shadow_s: str | None = shadow.value
        elif shadow:
            shadow_s = str(shadow)
        else:
            shadow_s = None
        scored_at = _get("scored_at")
        ts = None
        if scored_at is not None:
            ts = (
                scored_at.isoformat().replace("+00:00", "Z")
                if hasattr(scored_at, "isoformat")
                else str(scored_at)
            )
        return self.append(
            order_id=str(_get("order_id") or ""),
            market=str(_get("market") or ""),
            vertical=str(_get("vertical") or ""),
            abuse_score=float(_get("abuse_score") or 0.0),
            fraud_score=float(_get("fraud_score") or 0.0),
            decision_score=float(_get("decision_score") or 0.0),
            suggested_tier=tier_s,
            refund_effect=effect_s,
            shadow_refund_effect=shadow_s,
            hard_gated=bool(_get("hard_gated")),
            model_version=str(_get("model_version") or ""),
            policy_version=str(_get("policy_version") or ""),
            reason_codes=list(_get("reason_codes") or []),
            effect_decision=dict(_get("effect_decision") or {}),
            extras={
                "combined_score": _get("combined_score"),
                "precision_discount": _get("precision_discount"),
            },
            ts=ts,
        )

    def list_for_order(self, order_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM decision_archive
                WHERE order_id = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (str(order_id), int(limit)),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM decision_archive
                ORDER BY ts DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key, nice in (
            ("reason_codes_json", "reason_codes"),
            ("effect_decision_json", "effect_decision"),
            ("extras_json", "extras"),
        ):
            raw = item.pop(key, None)
            item[nice] = json.loads(raw) if raw else ([] if nice == "reason_codes" else {})
        item["hard_gated"] = bool(item.get("hard_gated"))
        return item
