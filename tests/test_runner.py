import pytest

from wq_pipeline import db
from wq_pipeline import runner as runner_module
from wq_pipeline.client import SimulationSettings
from wq_pipeline.models import GridItem
from wq_pipeline.runner import Runner, RunnerConfig, _alpha_dedup_key


class UnusedClient:
    def get_alpha(self, alpha_id):
        raise AssertionError(f"unexpected detail fetch for {alpha_id}")


def test_active_alpha_dedup_key_matches_expression_and_settings():
    settings = SimulationSettings(neutralization="MARKET", truncation=0.01)
    item = GridItem("template", "fnd6_x", "rank(fnd6_x)", settings)
    alpha = {
        "id": "A1",
        "status": "ACTIVE",
        "regular": item.expression,
        "settings": settings.to_payload(),
    }

    assert _alpha_dedup_key(alpha, UnusedClient()) == item.dedup_key()


def test_enqueue_skips_active_alpha_without_persisting_it(tmp_path):
    conn = db.connect(str(tmp_path / "alphas.db"))
    db.init_schema(conn)
    try:
        settings = SimulationSettings(neutralization="MARKET", truncation=0.01)
        item = GridItem("template", "fnd6_x", "rank(fnd6_x)", settings)
        runner = Runner("email", "password", conn, RunnerConfig())

        pending = runner._enqueue_grid([item], {item.dedup_key()})

        assert pending == []
        assert conn.execute("SELECT COUNT(*) FROM alphas").fetchone()[0] == 0
    finally:
        conn.close()


def test_active_alpha_without_list_details_fetches_alpha_record():
    settings = SimulationSettings(neutralization="MARKET", truncation=0.01)
    item = GridItem("template", "fnd6_x", "rank(fnd6_x)", settings)

    class DetailClient:
        def get_alpha(self, alpha_id):
            assert alpha_id == "A1"
            return {"regular": item.expression, "settings": settings.to_payload()}

    assert _alpha_dedup_key({"id": "A1", "status": "ACTIVE"}, DetailClient()) == item.dedup_key()


def test_non_active_alpha_is_not_used_for_deduplication():
    assert _alpha_dedup_key({"id": "A1", "status": "INACTIVE"}, UnusedClient()) is None


def test_run_fetches_active_alphas_before_enqueueing(tmp_path, monkeypatch):
    conn = db.connect(str(tmp_path / "alphas.db"))
    db.init_schema(conn)
    settings = SimulationSettings(neutralization="MARKET", truncation=0.01)
    item = GridItem("template", "fnd6_x", "rank(fnd6_x)", settings)

    class AccountClient:
        def __init__(self, email, password):
            assert (email, password) == ("email", "password")

        def get_user_alphas(self, *, status):
            assert status == "ACTIVE"
            return [
                {
                    "id": "A1",
                    "status": "ACTIVE",
                    "regular": item.expression,
                    "settings": settings.to_payload(),
                }
            ]

    monkeypatch.setattr(runner_module, "WQClient", AccountClient)
    runner = Runner("email", "password", conn, RunnerConfig(dry_run=True))
    try:
        runner.run([item])
        assert conn.execute("SELECT COUNT(*) FROM alphas").fetchone()[0] == 0
    finally:
        conn.close()


def test_run_does_not_enqueue_if_active_alpha_lookup_fails(tmp_path, monkeypatch):
    conn = db.connect(str(tmp_path / "alphas.db"))
    db.init_schema(conn)
    item = GridItem(
        "template",
        "fnd6_x",
        "rank(fnd6_x)",
        SimulationSettings(neutralization="MARKET", truncation=0.01),
    )

    class AccountClient:
        def __init__(self, email, password):
            pass

        def get_user_alphas(self, *, status):
            raise RuntimeError("account lookup failed")

    monkeypatch.setattr(runner_module, "WQClient", AccountClient)
    runner = Runner("email", "password", conn, RunnerConfig(dry_run=True))
    try:
        with pytest.raises(RuntimeError, match="account lookup failed"):
            runner.run([item])
        assert conn.execute("SELECT COUNT(*) FROM alphas").fetchone()[0] == 0
    finally:
        conn.close()
