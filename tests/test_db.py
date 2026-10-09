import pytest

from wq_pipeline import db
from wq_pipeline.client import SimulationSettings
from wq_pipeline.models import GridItem


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    c = db.connect(path)
    db.init_schema(c)
    yield c
    c.close()


@pytest.fixture
def grid_item():
    return GridItem(
        template_name="ts_mean_10",
        field_id="fnd6_pnrsho",
        expression="ts_mean(fnd6_pnrsho, 10)",
        settings=SimulationSettings(neutralization="MARKET", truncation=0.01),
    )


def _fake_alpha_payload(*, sharpe: float, fitness: float, low_sharpe_result: str = "FAIL"):
    return {
        "is": {
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": 0.01,
            "returns": 0.05,
            "drawdown": 0.30,
            "margin": 0.0050,
            "pnl": 1_000_000,
            "bookSize": 20_000_000,
            "longCount": 100,
            "shortCount": 100,
            "checks": [
                {"name": "LOW_SHARPE", "result": low_sharpe_result, "value": sharpe, "limit": 1.25},
                {"name": "CONCENTRATED_WEIGHT", "result": "PASS", "value": None, "limit": None},
                {"name": "MATCHES_COMPETITION", "result": "PASS", "value": None, "limit": None},
            ],
        }
    }


def test_upsert_pending_is_idempotent(conn, grid_item):
    rid1 = db.upsert_pending(conn, grid_item)
    rid2 = db.upsert_pending(conn, grid_item)
    assert rid1 == rid2


def test_lifecycle_running_to_complete_writes_metrics(conn, grid_item):
    rid = db.upsert_pending(conn, grid_item)
    db.mark_running(conn, rid, sim_location="https://api.worldquantbrain.com/simulations/abc")
    payload = _fake_alpha_payload(sharpe=1.5, fitness=1.4)
    db.mark_complete(conn, rid, alpha_id="ALPHA1", alpha_payload=payload, duration_s=120.0)

    row = dict(conn.execute("SELECT * FROM alphas WHERE id = ?", (rid,)).fetchone())
    assert row["status"] == db.STATUS_COMPLETE
    assert row["alpha_id"] == "ALPHA1"
    assert row["is_sharpe"] == 1.5
    assert row["is_fitness"] == 1.4
    assert row["chk_low_sharpe_result"] == "FAIL"  # 1.5 still fails the check object's stored result
    assert row["chk_concentrated_weight_result"] == "PASS"


def test_done_dedup_keys_includes_terminal_states(conn, grid_item):
    rid = db.upsert_pending(conn, grid_item)
    db.mark_complete(conn, rid, alpha_id="A", alpha_payload=_fake_alpha_payload(sharpe=1.0, fitness=1.0), duration_s=10.0)
    keys = db.done_dedup_keys(conn)
    assert grid_item.dedup_key() in keys


def test_reset_running_to_queued(conn, grid_item):
    rid = db.upsert_pending(conn, grid_item)
    db.mark_running(conn, rid, sim_location="x")
    n = db.reset_running_to_queued(conn)
    assert n == 1
    status = conn.execute("SELECT status FROM alphas WHERE id = ?", (rid,)).fetchone()["status"]
    assert status == db.STATUS_QUEUED


def test_reset_failed_and_timeout_re_queues(conn, grid_item):
    rid_a = db.upsert_pending(conn, grid_item)
    db.mark_failed(conn, rid_a, "network blip")
    rid_b = db.upsert_pending(
        conn,
        GridItem(
            template_name="ts_mean_10",
            field_id="fnd6_other",
            expression="ts_mean(fnd6_other, 10)",
            settings=SimulationSettings(neutralization="SUBINDUSTRY"),
        ),
    )
    db.mark_timeout(conn, rid_b, "watchdog")

    n = db.reset_failed_and_timeout(conn)
    assert n == 2
    statuses = {r["status"] for r in conn.execute("SELECT status FROM alphas").fetchall()}
    assert statuses == {db.STATUS_QUEUED}
