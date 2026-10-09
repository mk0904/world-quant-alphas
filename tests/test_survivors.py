"""End-to-end SQL-semantics tests for the survivor filter.

Exercises the actual SQL against an in-memory DB so we catch regressions like
NULL-handling bugs (NULL != 'FAIL' is NULL in SQL three-valued logic — caught
this one in code review and the tests guard against re-introduction).
"""

import pytest

from wq_pipeline import db
from wq_pipeline.client import SimulationSettings
from wq_pipeline.models import GridItem
from wq_pipeline.survivors import Thresholds, build_query


def _payload(sharpe: float, fitness: float, *, units_check: str | None = None):
    checks = [
        {"name": "CONCENTRATED_WEIGHT", "result": "PASS", "value": None, "limit": None},
        {"name": "MATCHES_COMPETITION", "result": "PASS", "value": None, "limit": None},
        {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS", "value": -0.1, "limit": -0.27},
    ]
    if units_check is not None:
        checks.append({"name": "UNITS", "result": units_check, "value": None, "limit": None})
    return {
        "is": {
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": 0.01,
            "returns": 0.05,
            "drawdown": 0.20,
            "margin": 0.0060,
            "pnl": 100_000,
            "bookSize": 20_000_000,
            "longCount": 100,
            "shortCount": 100,
            "checks": checks,
        }
    }


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.init_schema(c)
    yield c
    c.close()


def _seed(conn, *, expr: str, sharpe: float, fitness: float, units_check: str | None = None):
    gi = GridItem("t", "fnd6_x", expr, SimulationSettings(neutralization="MARKET"))
    rid = db.upsert_pending(conn, gi)
    db.mark_complete(
        conn, rid, alpha_id=f"A_{expr}", alpha_payload=_payload(sharpe, fitness, units_check=units_check), duration_s=10.0
    )
    return rid


def test_default_filter_rejects_below_threshold(conn):
    _seed(conn, expr="weak", sharpe=0.6, fitness=0.4)
    sql, params = build_query(Thresholds())
    rows = conn.execute(sql, params).fetchall()
    assert rows == []


def test_default_filter_accepts_passing_alpha(conn):
    _seed(conn, expr="strong", sharpe=1.5, fitness=1.4)
    sql, params = build_query(Thresholds())
    rows = conn.execute(sql, params).fetchall()
    assert len(rows) == 1
    assert rows[0]["alpha_id"] == "A_strong"


def test_null_check_does_not_silently_exclude(conn):
    """Regression: NULL != 'FAIL' is NULL in SQL three-valued logic. The query
    uses COALESCE so a row with no UNITS check still passes the structural filter."""
    _seed(conn, expr="no_units_check", sharpe=1.5, fitness=1.4, units_check=None)
    sql, params = build_query(Thresholds())
    rows = conn.execute(sql, params).fetchall()
    assert len(rows) == 1


def test_explicit_units_fail_excludes(conn):
    _seed(conn, expr="units_failed", sharpe=1.5, fitness=1.4, units_check="FAIL")
    sql, params = build_query(Thresholds())
    rows = conn.execute(sql, params).fetchall()
    assert rows == []


def test_allow_inverse_accepts_negative_sharpe(conn):
    _seed(conn, expr="inverse", sharpe=-1.5, fitness=-1.4)
    # default mode rejects
    sql, params = build_query(Thresholds())
    assert conn.execute(sql, params).fetchall() == []
    # allow_inverse accepts
    sql, params = build_query(Thresholds(allow_inverse=True))
    rows = conn.execute(sql, params).fetchall()
    assert len(rows) == 1
