"""Step 3: template grid runner.

Cross 6 templates × N fields × universes × neutralizations, submit to WQ BRAIN
3 concurrent, write all results to SQLite. Resumable: a re-run skips any cell
already terminal (COMPLETE/FAILED/TIMEOUT). RUNNING rows from a prior crash get
re-queued on startup.
"""
from __future__ import annotations

import argparse
import csv
import errno
import fcntl
import hashlib
import logging
import os
import sqlite3
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from . import db
from .client import SimulationSettings, WQClient
from .models import GridItem, Template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("runner")

# Templates. rank_neg dropped after first 222-sim batch (avg Sharpe -0.27, dead weight).
# Three new variants added based on top performers in that batch:
#   ts_delta_5_rank (max 0.98) and ts_zscore_63_neg (max 0.91) were the winners.
TEMPLATES: tuple[Template, ...] = (
    Template(name="ts_mean_10", expression="ts_mean({f}, 10)", decay=4, truncation=0.01),
    Template(name="ts_zscore_63_neg", expression="-ts_zscore({f}, 63)", decay=4, truncation=0.01),
    Template(name="ts_rank_22_neg", expression="-ts_rank({f}, 22)", decay=4, truncation=0.01),
    Template(name="ts_delta_5_rank", expression="rank(ts_delta({f}, 5))", decay=4, truncation=0.01),
    Template(name="ts_zscore_22", expression="ts_zscore({f}, 22)", decay=4, truncation=0.01),
    # winner variants:
    Template(name="ts_delta_5_rank_neg", expression="-rank(ts_delta({f}, 5))", decay=4, truncation=0.01),
    Template(name="ts_delta_22_rank", expression="rank(ts_delta({f}, 22))", decay=4, truncation=0.01),
    Template(name="ts_zscore_252_neg", expression="-ts_zscore({f}, 252)", decay=4, truncation=0.01),
)

# Prefixes the consultant tier can simulate (per user spec). Discovery sees more,
# but submitting outside this set will burn API calls on permission errors.
ALLOWED_PREFIXES: tuple[str, ...] = ("fnd6_", "pv13_")


@contextmanager
def _exclusive_lock(db_path: str):
    """File lock so two runner processes can't race on the same DB.

    Uses flock on a sibling .lock file. If another runner already holds it,
    this one exits cleanly with a clear message — no DB corruption, no double
    submissions.
    """
    lock_path = f"{db_path}.lock"
    # Held-open across the yield by design (flock is released on close); not a
    # context-manager candidate.
    fh = open(lock_path, "w")  # noqa: SIM115
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            if getattr(exc, "errno", None) in (errno.EAGAIN, errno.EWOULDBLOCK):
                fh.close()
                raise SystemExit(
                    f"another runner is already using {db_path} (lock at {lock_path}). "
                    f"kill it (pkill -f runner.py) before starting a new one."
                ) from None
            raise
        fh.write(f"pid={os.getpid()}\n")
        fh.flush()
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


@dataclass(frozen=True)
class RunnerConfig:
    db_path: str = "alphas.db"
    fields_csv: str = "fields.csv"
    universes: tuple[str, ...] = ("TOP1000",)
    neutralizations: tuple[str, ...] = ("MARKET",)
    max_workers: int = 3
    sim_timeout_s: int = 540  # passed to poll_simulation
    watchdog_s: int = 660     # main-loop deadline; > sim_timeout_s for headroom
    max_fields: int = 0       # 0 = no cap
    field_prefixes: tuple[str, ...] = ALLOWED_PREFIXES
    dry_run: bool = False     # build grid, write QUEUED rows, don't dispatch


# ---------- field universe ----------


def load_field_ids(
    csv_path: str,
    prefixes: tuple[str, ...],
    max_fields: int = 0,
) -> list[str]:
    """Read fields.csv and return ids matching any prefix, sorted by user popularity desc."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"missing {csv_path} — run discover_fields.py first")

    rows: list[tuple[str, int]] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            fid = (r.get("id") or "").strip()
            if not fid or not any(fid.startswith(p) for p in prefixes):
                continue
            try:
                pop = int(r.get("userCount") or 0)
            except ValueError:
                pop = 0
            rows.append((fid, pop))

    rows.sort(key=lambda x: -x[1])
    ids = [r[0] for r in rows]
    if max_fields > 0:
        ids = ids[:max_fields]
    logger.info(
        "loaded %d field ids from %s (prefixes=%s, cap=%s)",
        len(ids), csv_path, prefixes, max_fields or "-",
    )
    return ids


# ---------- grid ----------


def build_grid(
    templates: Iterable[Template],
    field_ids: Iterable[str],
    universes: Iterable[str],
    neutralizations: Iterable[str],
    base_settings: SimulationSettings | None = None,
) -> list[GridItem]:
    base = base_settings or SimulationSettings()
    items: list[GridItem] = []
    for tpl in templates:
        for fid in field_ids:
            expr = tpl.render(fid)
            for univ in universes:
                for neut in neutralizations:
                    settings = SimulationSettings(
                        region=base.region,
                        universe=univ,
                        delay=base.delay,
                        decay=tpl.decay,
                        neutralization=neut,
                        truncation=tpl.truncation,
                        pasteurization=base.pasteurization,
                        unit_handling=base.unit_handling,
                        nan_handling=base.nan_handling,
                        language=base.language,
                        instrument_type=base.instrument_type,
                        visualization=base.visualization,
                    )
                    items.append(
                        GridItem(
                            template_name=tpl.name,
                            field_id=fid,
                            expression=expr,
                            settings=settings,
                        )
                    )
    return items


# ---------- worker ----------

# requests.Session is not thread-safe. Each worker thread gets its own WQClient
# (and its own Session) via thread-local storage. Authenticated lazily on first use.
_thread_local = threading.local()


def _get_thread_client(email: str, password: str) -> WQClient:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = WQClient(email, password)
        _thread_local.client = client
    return client


def _run_one(email: str, password: str, item: GridItem, sim_timeout_s: int) -> dict:
    start = time.time()
    try:
        client = _get_thread_client(email, password)
        result = client.run_simulation(
            item.expression,
            item.settings,
        )
        return {
            "ok": True,
            "sim_location": result.get("location"),
            "alpha_id": result.get("alpha_id"),
            "alpha_payload": result.get("alpha"),
            "sim_status": (result.get("simulation") or {}).get("status"),
            "sim_message": (result.get("simulation") or {}).get("message"),
            "duration_s": time.time() - start,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_s": time.time() - start,
        }


def _alpha_dedup_key(
    alpha: dict[str, Any], client: WQClient
) -> tuple[str, str, str, int, int, str, float] | None:
    if str(alpha.get("status", "ACTIVE")).upper() != "ACTIVE":
        return None

    record = alpha
    settings = record.get("settings")
    expression = record.get("regular")
    if isinstance(expression, dict):
        expression = expression.get("code") or expression.get("expression")
    required_settings = (
        "region",
        "universe",
        "delay",
        "decay",
        "neutralization",
        "truncation",
    )
    if (
        not isinstance(expression, str)
        or not isinstance(settings, dict)
        or any(name not in settings for name in required_settings)
    ):
        alpha_id = alpha.get("id")
        if not alpha_id:
            raise RuntimeError("active alpha record is missing its expression or settings")
        record = client.get_alpha(str(alpha_id))
        settings = record.get("settings")
        expression = record.get("regular")
        if isinstance(expression, dict):
            expression = expression.get("code") or expression.get("expression")

    if (
        not isinstance(expression, str)
        or not isinstance(settings, dict)
        or any(name not in settings for name in required_settings)
    ):
        raise RuntimeError(
            f"active alpha {alpha.get('id', '<unknown>')} is missing its expression or settings"
        )

    return (
        hashlib.sha1(expression.encode("utf-8")).hexdigest(),
        settings["region"],
        settings["universe"],
        int(settings["delay"]),
        int(settings["decay"]),
        settings["neutralization"],
        float(settings["truncation"]),
    )


# ---------- dispatcher ----------


class Runner:
    def __init__(
        self,
        email: str,
        password: str,
        conn: sqlite3.Connection,
        config: RunnerConfig,
    ) -> None:
        self._email = email
        self._password = password
        self._conn = conn
        self._cfg = config

    def _enqueue_grid(
        self,
        grid: list[GridItem],
        active_alpha_keys: set[tuple[str, str, str, int, int, str, float]],
    ) -> list[tuple[int, GridItem]]:
        """Upsert all items to DB. Return [(row_id, item), ...] for not-yet-done items."""
        done = db.done_dedup_keys(self._conn)
        pending: list[tuple[int, GridItem]] = []
        skipped = 0
        skipped_active = 0
        for item in grid:
            if item.dedup_key() in active_alpha_keys:
                skipped_active += 1
                continue
            if item.dedup_key() in done:
                skipped += 1
                continue
            row_id = db.upsert_pending(self._conn, item)
            pending.append((row_id, item))
        logger.info(
            "grid: %d total, %d active on account, %d already done, %d to run",
            len(grid), skipped_active, skipped, len(pending),
        )
        return pending

    def run(self, grid: list[GridItem], retry_failed: bool = False) -> None:
        cfg = self._cfg
        account = WQClient(self._email, self._password)
        active_alphas = account.get_user_alphas(status="ACTIVE")
        active_alpha_keys: set[tuple[str, str, str, int, int, str, float]] = {
            key
            for alpha in active_alphas
            if (key := _alpha_dedup_key(alpha, account)) is not None
        }
        logger.info("loaded %d active alphas from BRAIN", len(active_alpha_keys))

        recovered = db.reset_running_to_queued(self._conn)
        if recovered:
            logger.info("recovered %d RUNNING rows from prior crash -> QUEUED", recovered)
        if retry_failed:
            n = db.reset_failed_and_timeout(self._conn)
            if n:
                logger.info("re-queued %d FAILED/TIMEOUT rows for retry", n)

        pending = self._enqueue_grid(grid, active_alpha_keys)
        if cfg.dry_run:
            logger.info("dry-run: not dispatching")
            return
        if not pending:
            logger.info("nothing to do")
            return

        executor = ThreadPoolExecutor(max_workers=cfg.max_workers)
        active: dict[int, tuple[Future, float, GridItem]] = {}  # row_id -> (future, start, item)
        idx = 0
        completed = 0
        failed = 0
        timed_out = 0
        last_progress = time.time()

        try:
            while idx < len(pending) or active:
                while len(active) < cfg.max_workers and idx < len(pending):
                    row_id, item = pending[idx]
                    # mark RUNNING BEFORE submit: closes the TOCTOU window where a
                    # crash between submit() and mark_running() would leave the row
                    # QUEUED while a worker is already (or about to be) hitting WQ.
                    # On a crash recovery, reset_running_to_queued resets cleanly.
                    db.mark_running(self._conn, row_id, sim_location="")
                    fut = executor.submit(
                        _run_one,
                        self._email,
                        self._password,
                        item,
                        cfg.sim_timeout_s,
                    )
                    active[row_id] = (fut, time.time(), item)
                    idx += 1

                done_ids: list[int] = []
                for row_id, (fut, started, item) in list(active.items()):
                    if fut.done():
                        done_ids.append(row_id)
                        continue
                    if time.time() - started > cfg.watchdog_s:
                        logger.warning(
                            "watchdog kill row=%d expr=%s after %.1fs",
                            row_id, item.expression, time.time() - started,
                        )
                        db.mark_timeout(self._conn, row_id, "watchdog timeout")
                        timed_out += 1
                        # we cannot cancel a running thread; we stop tracking it.
                        # poll_simulation has its own timeout so the thread will exit.
                        done_ids.append(row_id)

                for row_id in done_ids:
                    fut, started, item = active.pop(row_id)
                    if fut.done():
                        try:
                            result = fut.result(timeout=1.0)
                        except Exception as exc:  # noqa: BLE001
                            db.mark_failed(self._conn, row_id, f"future raised: {exc}")
                            failed += 1
                            continue
                        if result.get("ok"):
                            sim_status = result.get("sim_status")
                            if sim_status and sim_status != "COMPLETE":
                                msg = result.get("sim_message") or ""
                                db.mark_failed(
                                    self._conn,
                                    row_id,
                                    f"sim_status={sim_status} {msg}".strip(),
                                )
                                failed += 1
                            else:
                                # mark_running updates location after the fact
                                if result.get("sim_location"):
                                    self._conn.execute(
                                        "UPDATE alphas SET sim_location = ? WHERE id = ?",
                                        (result["sim_location"], row_id),
                                    )
                                db.mark_complete(
                                    self._conn,
                                    row_id,
                                    alpha_id=result.get("alpha_id"),
                                    alpha_payload=result.get("alpha_payload"),
                                    duration_s=result.get("duration_s"),
                                )
                                completed += 1
                        else:
                            db.mark_failed(self._conn, row_id, result.get("error") or "unknown")
                            failed += 1

                if time.time() - last_progress > 30 or (
                    not active and idx >= len(pending)
                ):
                    last_progress = time.time()
                    logger.info(
                        "progress: dispatched=%d/%d active=%d ok=%d fail=%d timeout=%d",
                        idx, len(pending), len(active), completed, failed, timed_out,
                    )

                if active and not done_ids:
                    time.sleep(1.0)
        finally:
            # wait=True so any zombie threads that watchdog gave up tracking finish
            # cleanly (their inner poll_simulation will time out shortly). Avoids a
            # hung process at exit.
            executor.shutdown(wait=True, cancel_futures=True)

        logger.info(
            "run done: ok=%d fail=%d timeout=%d total=%d",
            completed, failed, timed_out, len(pending),
        )


# ---------- entrypoint ----------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="alphas.db")
    p.add_argument("--fields-csv", default="fields.csv")
    p.add_argument("--universes", default="TOP1000",
                   help="comma-separated, e.g. TOP1000,TOP3000")
    p.add_argument("--neutralizations", default="MARKET",
                   help="comma-separated, e.g. MARKET,SUBINDUSTRY")
    p.add_argument("--max-fields", type=int, default=10,
                   help="cap N most-popular fields per prefix; 0 = no cap")
    p.add_argument("--field-prefixes", default=",".join(ALLOWED_PREFIXES))
    p.add_argument("--max-workers", type=int, default=3)
    p.add_argument("--sim-timeout", type=int, default=540)
    p.add_argument("--watchdog", type=int, default=660)
    p.add_argument("--templates", default="",
                   help="comma-separated subset of template names; default = all 6")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-queue FAILED/TIMEOUT rows at startup (use after network blip)")
    return p.parse_args(argv)


def _select_templates(names_csv: str) -> tuple[Template, ...]:
    if not names_csv.strip():
        return TEMPLATES
    wanted = {n.strip() for n in names_csv.split(",") if n.strip()}
    chosen = tuple(t for t in TEMPLATES if t.name in wanted)
    if not chosen:
        raise SystemExit(f"no template names matched. available: {[t.name for t in TEMPLATES]}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    email = os.environ.get("WQ_EMAIL")
    password = os.environ.get("WQ_PASS")
    if not email or not password:
        print("ERROR: set WQ_EMAIL and WQ_PASS in .env", file=sys.stderr)
        return 1

    cfg = RunnerConfig(
        db_path=args.db,
        fields_csv=args.fields_csv,
        universes=tuple(u.strip() for u in args.universes.split(",") if u.strip()),
        neutralizations=tuple(n.strip() for n in args.neutralizations.split(",") if n.strip()),
        max_workers=args.max_workers,
        sim_timeout_s=args.sim_timeout,
        watchdog_s=args.watchdog,
        max_fields=args.max_fields,
        field_prefixes=tuple(p.strip() for p in args.field_prefixes.split(",") if p.strip()),
        dry_run=args.dry_run,
    )

    templates = _select_templates(args.templates)
    field_ids = load_field_ids(cfg.fields_csv, cfg.field_prefixes, cfg.max_fields)
    if not field_ids:
        print("ERROR: no fields matched the prefix filter", file=sys.stderr)
        return 2

    grid = build_grid(templates, field_ids, cfg.universes, cfg.neutralizations)
    logger.info(
        "grid size: %d (%d templates x %d fields x %d universes x %d neuts)",
        len(grid), len(templates), len(field_ids), len(cfg.universes), len(cfg.neutralizations),
    )

    with _exclusive_lock(cfg.db_path):
        conn = db.connect(cfg.db_path)
        try:
            db.init_schema(conn)
            runner = Runner(email, password, conn, cfg)
            runner.run(grid, retry_failed=args.retry_failed)
            counts = db.count_by_status(conn)
            print("\nfinal counts:", counts)
        finally:
            conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
# runner active alpha deduplication verified
