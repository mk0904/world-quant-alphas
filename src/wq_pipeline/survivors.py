"""Step 5: filter the alphas table for IS-passing candidates.

Default thresholds match the brief: Sharpe >= 1.25, fitness >= 1.3, |margin| >= 50bp.
All non-pending IS checks must PASS (SELF_CORRELATION often stays PENDING — opt-in).

Exports survivors to CSV for review and stages them for Step 6 correlation pruning.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass

from . import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("survivors")

# Structural IS checks survivors must not FAIL. Note: LOW_SHARPE / LOW_FITNESS / LOW_TURNOVER
# / HIGH_TURNOVER are deliberately excluded — they duplicate the explicit float thresholds
# in Thresholds, and using them would re-impose WQ's hardcoded 1.25 / 1.0 / 0.01 / 0.7 limits.
REQUIRED_CHECKS = (
    "concentrated_weight",
    "low_sub_universe_sharpe",
    "matches_competition",
    "units",
)

# REQUIRED_CHECKS values are interpolated into SQL column names. Belt-and-braces
# guard against any future change that takes them from user input.
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
for _name in REQUIRED_CHECKS:
    if not _SAFE_NAME.match(_name):
        raise ValueError(f"unsafe check name {_name!r} — must match {_SAFE_NAME.pattern}")


@dataclass(frozen=True)
class Thresholds:
    min_sharpe: float = 1.25
    min_fitness: float = 1.3
    min_abs_margin: float = 0.0050  # 50 bp
    max_turnover: float = 0.7
    min_turnover: float = 0.01
    require_self_correlation_pass: bool = False
    # Allow negative-Sharpe alphas via |Sharpe| filter. Useful for alpha mining
    # because the inverse expression of a -1.5-Sharpe alpha is a +1.5-Sharpe alpha.
    # OFF by default — competition submissions need positive Sharpe directly.
    allow_inverse: bool = False


def build_query(thr: Thresholds) -> tuple[str, list]:
    where: list[str] = ["status = 'COMPLETE'", "is_sharpe IS NOT NULL"]
    params: list = []

    if thr.allow_inverse:
        where.append("ABS(is_sharpe) >= ?")
        params.append(thr.min_sharpe)
        where.append("ABS(is_fitness) >= ?")
        params.append(thr.min_fitness)
        where.append("ABS(is_margin) >= ?")
        params.append(thr.min_abs_margin)
    else:
        where.append("is_sharpe >= ?")
        params.append(thr.min_sharpe)
        where.append("is_fitness >= ?")
        params.append(thr.min_fitness)
        where.append("ABS(is_margin) >= ?")
        params.append(thr.min_abs_margin)
    where.append("is_turnover BETWEEN ? AND ?")
    params.extend([thr.min_turnover, thr.max_turnover])

    for chk in REQUIRED_CHECKS:
        # accept NULL (check absent on row), PASS, or PENDING; reject FAIL.
        # NULL != 'FAIL' is NULL in SQL three-valued logic so we coalesce.
        where.append(f"COALESCE(chk_{chk}_result, 'PASS') != 'FAIL'")

    if thr.require_self_correlation_pass:
        where.append("chk_self_correlation_result = 'PASS'")

    order_by = "ABS(is_sharpe) DESC" if thr.allow_inverse else "is_sharpe DESC"
    sql = f"""
    SELECT id, alpha_id, template_name, field_id, expression,
           region, universe, delay, decay, neutralization, truncation,
           is_sharpe, is_fitness, is_turnover, is_returns, is_drawdown, is_margin,
           is_long_count, is_short_count,
           chk_self_correlation_result, chk_self_correlation_value,
           submitted_at, completed_at, duration_s
    FROM alphas
    WHERE {' AND '.join(where)}
    ORDER BY {order_by}
    """
    return sql, params


EXPORT_COLS = (
    "id", "alpha_id", "template_name", "field_id", "expression",
    "region", "universe", "delay", "decay", "neutralization", "truncation",
    "is_sharpe", "is_fitness", "is_turnover", "is_returns", "is_drawdown", "is_margin",
    "is_long_count", "is_short_count",
    "chk_self_correlation_result", "chk_self_correlation_value",
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="alphas.db")
    p.add_argument("--out-csv", default="survivors.csv")
    p.add_argument("--min-sharpe", type=float, default=1.25)
    p.add_argument("--min-fitness", type=float, default=1.3)
    p.add_argument("--min-abs-margin", type=float, default=0.0050)
    p.add_argument("--min-turnover", type=float, default=0.01)
    p.add_argument("--max-turnover", type=float, default=0.7)
    p.add_argument("--require-self-corr-pass", action="store_true")
    p.add_argument("--allow-inverse", action="store_true",
                   help="filter on |Sharpe| / |fitness| (accepts negative-Sharpe alphas; "
                        "submit them with sign flipped). Off by default.")
    p.add_argument("--limit", type=int, default=0, help="cap output (0 = all)")
    args = p.parse_args(argv)

    thr = Thresholds(
        min_sharpe=args.min_sharpe,
        min_fitness=args.min_fitness,
        min_abs_margin=args.min_abs_margin,
        max_turnover=args.max_turnover,
        min_turnover=args.min_turnover,
        require_self_correlation_pass=args.require_self_corr_pass,
        allow_inverse=args.allow_inverse,
    )
    logger.info("thresholds: %s", thr)

    conn = db.connect(args.db)
    try:
        sql, params = build_query(thr)
        cur = conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        if args.limit > 0:
            rows = rows[: args.limit]
    finally:
        conn.close()

    if not rows:
        logger.info("0 survivors out of completed sims")
        with open(args.out_csv, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=EXPORT_COLS).writeheader()
        return 0

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPORT_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    logger.info("wrote %d survivors -> %s", len(rows), args.out_csv)
    print("\n--- top 10 survivors by |sharpe| ---")
    print(f"{'id':<5} {'alpha':<10} {'template':<18} {'field':<14} "
          f"{'sharpe':>7} {'fitness':>8} {'turn':>7} {'margin':>9}")
    for r in rows[:10]:
        print(
            f"{r['id']:<5} {(r['alpha_id'] or '-'):<10} "
            f"{r['template_name']:<18} {r['field_id']:<14} "
            f"{r['is_sharpe']:>7.3f} {r['is_fitness']:>8.3f} "
            f"{r['is_turnover']:>7.4f} {r['is_margin']:>9.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
