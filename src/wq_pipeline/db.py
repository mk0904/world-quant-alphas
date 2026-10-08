"""SQLite results store for the alpha pipeline.

Schema: one row per simulation (grid cell). UNIQUE on the dedup tuple so re-runs
upsert idempotently. WAL mode + busy_timeout for safe concurrent reads while a
single writer (the runner main thread) does all inserts.

Each row stores:
  - the rendered expression and its settings (denormalized for fast SQL filtering)
  - status lifecycle (QUEUED -> RUNNING -> COMPLETE/FAILED/TIMEOUT)
  - all IS metrics as typed columns (Sharpe, fitness, etc.)
  - each named IS check as result/value/limit triplet (LOW_SHARPE, LOW_FITNESS, ...)
  - raw alpha payload as JSON for escape-hatch / future check additions
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterable
from typing import Any

from .models import GridItem

logger = logging.getLogger(__name__)

STATUS_QUEUED = "QUEUED"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETE = "COMPLETE"
STATUS_FAILED = "FAILED"
STATUS_TIMEOUT = "TIMEOUT"

# IS checks we flatten into typed columns. Anything else stays in raw JSON.
KNOWN_CHECKS = (
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "SELF_CORRELATION",
    "MATCHES_COMPETITION",
    "UNITS",
)


def _check_columns() -> list[str]:
    cols: list[str] = []
    for name in KNOWN_CHECKS:
        base = f"chk_{name.lower()}"
        cols.append(f"{base}_result")
        cols.append(f"{base}_value")
        cols.append(f"{base}_limit")
    return cols


SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS alphas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_name TEXT NOT NULL,
    field_id TEXT NOT NULL,
    expression TEXT NOT NULL,
    expression_hash TEXT NOT NULL,
    alpha_id TEXT,
    sim_location TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    region TEXT NOT NULL,
    universe TEXT NOT NULL,
    delay INTEGER NOT NULL,
    decay INTEGER NOT NULL,
    neutralization TEXT NOT NULL,
    truncation REAL NOT NULL,
    pasteurization TEXT,
    unit_handling TEXT,
    nan_handling TEXT,
    language TEXT,
    is_sharpe REAL,
    is_fitness REAL,
    is_turnover REAL,
    is_returns REAL,
    is_drawdown REAL,
    is_margin REAL,
    is_pnl REAL,
    is_book_size REAL,
    is_long_count INTEGER,
    is_short_count INTEGER,
    {", ".join(f"{c} {'REAL' if c.endswith(('_value', '_limit')) else 'TEXT'}" for c in _check_columns())},
    raw_alpha_json TEXT,
    submitted_at INTEGER,
    completed_at INTEGER,
    duration_s REAL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE (expression_hash, region, universe, delay, decay, neutralization, truncation)
);
CREATE INDEX IF NOT EXISTS idx_alphas_status ON alphas(status);
CREATE INDEX IF NOT EXISTS idx_alphas_template ON alphas(template_name);
CREATE INDEX IF NOT EXISTS idx_alphas_sharpe ON alphas(is_sharpe);
CREATE INDEX IF NOT EXISTS idx_alphas_alpha_id ON alphas(alpha_id);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open a connection with WAL + busy_timeout. Caller owns close."""
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def upsert_pending(conn: sqlite3.Connection, item: GridItem) -> int:
    """Insert a QUEUED row if no matching dedup_key exists. Return rowid."""
    s = item.settings
    expr_hash = item.expression_hash()
    cur = conn.execute(
        """
        SELECT id, status FROM alphas
        WHERE expression_hash = ? AND region = ? AND universe = ?
          AND delay = ? AND decay = ? AND neutralization = ? AND truncation = ?
        """,
        (expr_hash, s.region, s.universe, s.delay, s.decay, s.neutralization, s.truncation),
    )
    row = cur.fetchone()
    if row is not None:
        return int(row["id"])

    cur = conn.execute(
        """
        INSERT INTO alphas (
            template_name, field_id, expression, expression_hash, status,
            region, universe, delay, decay, neutralization, truncation,
            pasteurization, unit_handling, nan_handling, language
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.template_name,
            item.field_id,
            item.expression,
            expr_hash,
            STATUS_QUEUED,
            s.region,
            s.universe,
            s.delay,
            s.decay,
            s.neutralization,
            s.truncation,
            s.pasteurization,
            s.unit_handling,
            s.nan_handling,
            s.language,
        ),
    )
    return int(cur.lastrowid)


def mark_running(conn: sqlite3.Connection, row_id: int, sim_location: str) -> None:
    conn.execute(
        "UPDATE alphas SET status = ?, sim_location = ?, submitted_at = ? WHERE id = ?",
        (STATUS_RUNNING, sim_location, int(time.time()), row_id),
    )


def mark_failed(conn: sqlite3.Connection, row_id: int, error_message: str) -> None:
    conn.execute(
        "UPDATE alphas SET status = ?, error_message = ?, completed_at = ? WHERE id = ?",
        (STATUS_FAILED, error_message[:1000], int(time.time()), row_id),
    )


def mark_timeout(conn: sqlite3.Connection, row_id: int, error_message: str = "") -> None:
    conn.execute(
        "UPDATE alphas SET status = ?, error_message = ?, completed_at = ? WHERE id = ?",
        (STATUS_TIMEOUT, error_message[:1000] or "watchdog timeout", int(time.time()), row_id),
    )


def _extract_check(checks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for c in checks:
        if c.get("name") == name:
            return c
    return {}


def mark_complete(
    conn: sqlite3.Connection,
    row_id: int,
    alpha_id: str | None,
    alpha_payload: dict[str, Any] | None,
    duration_s: float | None,
) -> None:
    """Write IS metrics + flattened checks + raw JSON for a finished sim."""
    is_block: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    if alpha_payload:
        is_block = alpha_payload.get("is") or {}
        checks = list(is_block.get("checks") or [])

    cols = [
        "status",
        "alpha_id",
        "completed_at",
        "duration_s",
        "is_sharpe",
        "is_fitness",
        "is_turnover",
        "is_returns",
        "is_drawdown",
        "is_margin",
        "is_pnl",
        "is_book_size",
        "is_long_count",
        "is_short_count",
    ]
    vals: list[Any] = [
        STATUS_COMPLETE,
        alpha_id,
        int(time.time()),
        duration_s,
        is_block.get("sharpe"),
        is_block.get("fitness"),
        is_block.get("turnover"),
        is_block.get("returns"),
        is_block.get("drawdown"),
        is_block.get("margin"),
        is_block.get("pnl"),
        is_block.get("bookSize"),
        is_block.get("longCount"),
        is_block.get("shortCount"),
    ]

    for name in KNOWN_CHECKS:
        c = _extract_check(checks, name)
        base = f"chk_{name.lower()}"
        cols.append(f"{base}_result")
        cols.append(f"{base}_value")
        cols.append(f"{base}_limit")
        vals.append(c.get("result"))
        vals.append(_as_float(c.get("value")))
        vals.append(_as_float(c.get("limit")))

    cols.append("raw_alpha_json")
    vals.append(json.dumps(alpha_payload) if alpha_payload is not None else None)

    set_clause = ", ".join(f"{c} = ?" for c in cols)
    vals.append(row_id)
    conn.execute(f"UPDATE alphas SET {set_clause} WHERE id = ?", vals)


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def done_dedup_keys(conn: sqlite3.Connection) -> set[tuple]:
    """Tuples for rows already finished (any terminal status). Drives resume."""
    cur = conn.execute(
        """
        SELECT expression_hash, region, universe, delay, decay, neutralization, truncation
        FROM alphas
        WHERE status IN (?, ?, ?)
        """,
        (STATUS_COMPLETE, STATUS_FAILED, STATUS_TIMEOUT),
    )
    return {tuple(row) for row in cur.fetchall()}


def count_by_status(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.execute("SELECT status, COUNT(*) AS n FROM alphas GROUP BY status")
    return {str(r["status"]): int(r["n"]) for r in cur.fetchall()}


def reset_running_to_queued(conn: sqlite3.Connection) -> int:
    """Recover from a crash mid-run: re-queue any RUNNING rows."""
    cur = conn.execute(
        "UPDATE alphas SET status = ?, sim_location = NULL WHERE status = ?",
        (STATUS_QUEUED, STATUS_RUNNING),
    )
    return cur.rowcount or 0


def reset_failed_and_timeout(conn: sqlite3.Connection) -> int:
    """Re-queue any FAILED or TIMEOUT rows. Useful after a network blip blew out
    a batch of in-flight sims. Caller should know whether the failures are
    transient (network/wifi drop) or persistent (bad expression, no perms).
    """
    cur = conn.execute(
        """
        UPDATE alphas
        SET status = ?, error_message = NULL, sim_location = NULL,
            submitted_at = NULL, completed_at = NULL
        WHERE status IN (?, ?)
        """,
        (STATUS_QUEUED, STATUS_FAILED, STATUS_TIMEOUT),
    )
    return cur.rowcount or 0


def fetch_pending(
    conn: sqlite3.Connection, statuses: Iterable[str] = (STATUS_QUEUED,)
) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in statuses)
    cur = conn.execute(
        f"SELECT * FROM alphas WHERE status IN ({placeholders}) ORDER BY id",
        tuple(statuses),
    )
    return list(cur.fetchall())
