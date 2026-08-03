"""Pending human-in-the-loop (HIL) proposals for oversized threshold moves."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class HilProposalStore:
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
                CREATE TABLE IF NOT EXISTS hil_proposals (
                  proposal_id TEXT PRIMARY KEY,
                  ts TEXT NOT NULL,
                  status TEXT NOT NULL,
                  market TEXT NOT NULL DEFAULT '',
                  vertical TEXT NOT NULL DEFAULT '',
                  current_json TEXT NOT NULL,
                  proposed_json TEXT NOT NULL,
                  auto_step_json TEXT,
                  metrics_json TEXT,
                  reason TEXT NOT NULL DEFAULT '',
                  decided_at TEXT,
                  decided_by TEXT
                )
                """
            )
            conn.commit()

    def create(
        self,
        *,
        current: dict[str, Any],
        proposed: dict[str, Any],
        auto_step: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        market: str = "",
        vertical: str = "",
        reason: str = "delta_exceeds_max_auto",
    ) -> str:
        proposal_id = uuid4().hex
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO hil_proposals(
                  proposal_id, ts, status, market, vertical,
                  current_json, proposed_json, auto_step_json, metrics_json, reason
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    ts,
                    market or "",
                    vertical or "",
                    json.dumps(current),
                    json.dumps(proposed),
                    json.dumps(auto_step) if auto_step is not None else None,
                    json.dumps(metrics) if metrics is not None else None,
                    reason,
                ),
            )
            conn.commit()
        return proposal_id

    def get(self, proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM hil_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row(row)

    def list_pending(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM hil_proposals
                WHERE status = 'pending'
                ORDER BY ts DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [self._row(r) for r in rows]

    def mark(
        self,
        proposal_id: str,
        *,
        status: str,
        decided_by: str,
    ) -> dict[str, Any]:
        if status not in {"approved", "rejected"}:
            raise ValueError("status must be approved or rejected")
        row = self.get(proposal_id)
        if row is None:
            raise KeyError(f"unknown proposal_id={proposal_id}")
        if row["status"] != "pending":
            raise ValueError(f"proposal already {row['status']}")
        decided_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE hil_proposals
                SET status = ?, decided_at = ?, decided_by = ?
                WHERE proposal_id = ?
                """,
                (status, decided_at, decided_by, proposal_id),
            )
            conn.commit()
        out = self.get(proposal_id)
        assert out is not None
        return out

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["current"] = json.loads(item.pop("current_json"))
        item["proposed"] = json.loads(item.pop("proposed_json"))
        auto = item.pop("auto_step_json")
        item["auto_step"] = json.loads(auto) if auto else None
        metrics = item.pop("metrics_json")
        item["metrics"] = json.loads(metrics) if metrics else None
        return item
