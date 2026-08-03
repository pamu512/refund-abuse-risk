"""Append-only tuning audit log (SQLite)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class PolicyAuditLog:
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
                CREATE TABLE IF NOT EXISTS policy_audit_log (
                  audit_id TEXT PRIMARY KEY,
                  ts TEXT NOT NULL,
                  actor TEXT NOT NULL,
                  action TEXT NOT NULL,
                  market TEXT NOT NULL DEFAULT '',
                  vertical TEXT NOT NULL DEFAULT '',
                  before_json TEXT,
                  after_json TEXT,
                  metrics_before_json TEXT,
                  metrics_after_json TEXT,
                  constraints_json TEXT,
                  decision TEXT NOT NULL,
                  reason TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def append(
        self,
        *,
        actor: str,
        action: str,
        market: str = "",
        vertical: str = "",
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        metrics_before: dict[str, Any] | None = None,
        metrics_after: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        decision: str = "recorded",
        reason: str = "",
    ) -> str:
        audit_id = uuid4().hex
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO policy_audit_log(
                  audit_id, ts, actor, action, market, vertical,
                  before_json, after_json, metrics_before_json, metrics_after_json,
                  constraints_json, decision, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    ts,
                    actor,
                    action,
                    market or "",
                    vertical or "",
                    json.dumps(before) if before is not None else None,
                    json.dumps(after) if after is not None else None,
                    json.dumps(metrics_before) if metrics_before is not None else None,
                    json.dumps(metrics_after) if metrics_after is not None else None,
                    json.dumps(constraints) if constraints is not None else None,
                    decision,
                    reason,
                ),
            )
            conn.commit()
        return audit_id

    def list_recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM policy_audit_log
                ORDER BY ts DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in (
                "before_json",
                "after_json",
                "metrics_before_json",
                "metrics_after_json",
                "constraints_json",
            ):
                raw = item.pop(key, None)
                nice = key.replace("_json", "")
                item[nice] = json.loads(raw) if raw else None
            out.append(item)
        return out

    def last_apply_at(self, *, market: str = "", vertical: str = "") -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT ts FROM policy_audit_log
                WHERE action = 'apply' AND decision = 'accepted'
                  AND market = ? AND vertical = ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (market or "", vertical or ""),
            ).fetchone()
        return None if row is None else str(row["ts"])
