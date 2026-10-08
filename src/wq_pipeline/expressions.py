"""Bulk-grid raw alpha expressions through the same runner/DB.

Feed a file of literal Fast Expression strings (one per line, # comments OK) and
this crosses each with the chosen universes / neutralizations / decay / trunc,
then dispatches through the same Runner + SQLite DB as runner.py. Resume,
dedup, concurrency, watchdog all reused.

Use cases:
- crowd-sourced alpha ideas you found online
- your own hand-written candidates
- output of an LLM that produced full expressions (not templates)
- variants you want to A/B (e.g. same expression at decay 4 vs 8)

Each expression is recorded in the DB with template_name='manual' and
field_id parsed best-effort from the expression (first fnd6_/pv13_ token), so
survivors.py / correlation.py work unchanged.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections.abc import Iterable

from dotenv import load_dotenv

from . import db
from .client import SimulationSettings
from .models import GridItem
from .runner import ALLOWED_PREFIXES, Runner, RunnerConfig, _exclusive_lock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("submit_expr")


_FIELD_TOKEN = re.compile(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b")


def parse_expressions_file(path: str) -> list[tuple[str, str]]:
    """Read file. Each line is either:
      - bare expression: rank(close - ts_mean(close, 20))
      - labelled: my_idea_1 :: rank(close - ts_mean(close, 20))
    Blank lines and # comments skipped. Returns [(label, expression), ...].
    """
    out: list[tuple[str, str]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing expressions file: {path}")
    with open(path) as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "::" in line:
                label, _, expr = line.partition("::")
                label = label.strip() or f"line{lineno}"
                expr = expr.strip()
            else:
                label = f"line{lineno}"
                expr = line
            if not expr:
                continue
            out.append((label, expr))
    return out


def _guess_field_id(expression: str) -> str:
    """Best-effort: first token that looks like a data field. '' if none."""
    for match in _FIELD_TOKEN.finditer(expression):
        token = match.group(1)
        if any(token.startswith(p) for p in ALLOWED_PREFIXES) or token.startswith(
            ("anl4_", "fnd2_", "fnd17_", "mdl", "pv1", "pv2", "snt1_")
        ):
            return token
    return ""


def build_expression_grid(
    expressions: Iterable[tuple[str, str]],
    universes: Iterable[str],
    neutralizations: Iterable[str],
    decay: int,
    truncation: float,
    base: SimulationSettings | None = None,
) -> list[GridItem]:
    base = base or SimulationSettings()
    items: list[GridItem] = []
    for label, expr in expressions:
        for univ in universes:
            for neut in neutralizations:
                settings = SimulationSettings(
                    region=base.region,
                    universe=univ,
                    delay=base.delay,
                    decay=decay,
                    neutralization=neut,
                    truncation=truncation,
                    pasteurization=base.pasteurization,
                    unit_handling=base.unit_handling,
                    nan_handling=base.nan_handling,
                    language=base.language,
                    instrument_type=base.instrument_type,
                    visualization=base.visualization,
                )
                items.append(
                    GridItem(
                        template_name=f"manual:{label}",
                        field_id=_guess_field_id(expr),
                        expression=expr,
                        settings=settings,
                    )
                )
    return items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="path to expressions file")
    p.add_argument("--db", default="alphas.db")
    p.add_argument("--universes", default="TOP1000")
    p.add_argument("--neutralizations", default="MARKET")
    p.add_argument("--decay", type=int, default=4)
    p.add_argument("--truncation", type=float, default=0.01)
    p.add_argument("--max-workers", type=int, default=3)
    p.add_argument("--sim-timeout", type=int, default=540)
    p.add_argument("--watchdog", type=int, default=660)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-queue FAILED/TIMEOUT rows at startup (use after network blip)")
    args = p.parse_args(argv)

    load_dotenv()
    email = os.environ.get("WQ_EMAIL")
    password = os.environ.get("WQ_PASS")
    if not email or not password:
        print("ERROR: set WQ_EMAIL and WQ_PASS in .env", file=sys.stderr)
        return 1

    exprs = parse_expressions_file(args.file)
    if not exprs:
        print("ERROR: no expressions parsed (check file format)", file=sys.stderr)
        return 2
    logger.info("parsed %d expressions from %s", len(exprs), args.file)

    universes = tuple(u.strip() for u in args.universes.split(",") if u.strip())
    neuts = tuple(n.strip() for n in args.neutralizations.split(",") if n.strip())

    grid = build_expression_grid(exprs, universes, neuts, args.decay, args.truncation)
    logger.info(
        "grid: %d cells (%d expr x %d univ x %d neut)",
        len(grid), len(exprs), len(universes), len(neuts),
    )

    cfg = RunnerConfig(
        db_path=args.db,
        max_workers=args.max_workers,
        sim_timeout_s=args.sim_timeout,
        watchdog_s=args.watchdog,
        dry_run=args.dry_run,
    )

    with _exclusive_lock(args.db):
        conn = db.connect(args.db)
        try:
            db.init_schema(conn)
            runner = Runner(email, password, conn, cfg)
            runner.run(grid, retry_failed=args.retry_failed)
            print("\nfinal counts:", db.count_by_status(conn))
        finally:
            conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
