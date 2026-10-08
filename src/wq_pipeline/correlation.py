"""Step 6: correlation pruning.

Take the survivors set (Step 5 logic), pull PnL series for each alpha, compute
pairwise Pearson correlation on daily returns, then greedy-select an
uncorrelated basket: sort by |Sharpe| descending; keep an alpha only if its
max |corr| to the already-kept set is below the threshold.

PnL fetches are cached to disk (one .npz per alpha) so re-runs with different
correlation thresholds don't re-hit the API.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv

from . import db
from .client import WQClient
from .survivors import Thresholds, build_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("correlation")

CACHE_DIR = ".pnl_cache"


@dataclass(frozen=True)
class Survivor:
    alpha_id: str
    sharpe: float
    fitness: float
    template_name: str
    field_id: str
    expression: str


# ---------- PnL fetch with disk cache ----------


def _cache_path(alpha_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{alpha_id}.npz")


def _records_to_returns(records: Sequence[Sequence]) -> np.ndarray:
    """Convert [[date, pnl], ...] to a daily-returns numpy array.

    The PnL series WQ returns is cumulative ($-denominated). We diff to get daily
    PnL changes, which is what we want for correlation (correlation of cumulative
    series is meaningless).
    """
    vals: list[float] = []
    for rec in records:
        if not rec or len(rec) < 2:
            continue
        try:
            vals.append(float(rec[1]))
        except (TypeError, ValueError):
            vals.append(np.nan)
    arr = np.array(vals, dtype=float)
    if arr.size < 2:
        return np.array([], dtype=float)
    diffs = np.diff(arr)
    return diffs


def fetch_pnl(client: WQClient, alpha_id: str, force: bool = False) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(alpha_id)
    if not force and os.path.exists(path):
        try:
            with np.load(path) as data:
                return data["returns"]
        except Exception:  # noqa: BLE001 — corrupt cache, refetch
            logger.warning("cache corrupt for %s; refetching", alpha_id)

    payload = client.get_alpha_pnl(alpha_id)
    records = payload.get("records") or []
    returns = _records_to_returns(records)
    np.savez_compressed(path, returns=returns)
    logger.info("fetched pnl alpha=%s len=%d -> %s", alpha_id, returns.size, path)
    return returns


# ---------- correlation + greedy de-dup ----------


def align_returns(series_by_alpha: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Stack returns into a 2-D matrix (n_alphas x n_days). Drop alphas with mismatched length."""
    if not series_by_alpha:
        return [], np.empty((0, 0))
    target_len = max(s.size for s in series_by_alpha.values())
    keep_ids: list[str] = []
    rows: list[np.ndarray] = []
    for aid, s in series_by_alpha.items():
        if s.size != target_len or s.size == 0:
            logger.warning("dropping %s: returns len=%d (target=%d)", aid, s.size, target_len)
            continue
        keep_ids.append(aid)
        rows.append(s)
    if not rows:
        return [], np.empty((0, 0))
    return keep_ids, np.vstack(rows)


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation across rows. NaNs replaced by 0 first (correlation undefined otherwise)."""
    if matrix.size == 0:
        return np.empty((0, 0))
    safe = np.where(np.isnan(matrix), 0.0, matrix)
    corr = np.corrcoef(safe)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    return corr


def greedy_dedup(
    survivors: list[Survivor],
    corr: np.ndarray,
    alpha_order: list[str],
    threshold: float,
) -> list[Survivor]:
    """Iterate survivors by descending |Sharpe|; keep if max |corr| to kept < threshold."""
    rank_by_id = {aid: i for i, aid in enumerate(alpha_order)}
    sorted_survivors = sorted(survivors, key=lambda s: -abs(s.sharpe))
    kept: list[Survivor] = []
    kept_idx: list[int] = []
    skipped: list[tuple[Survivor, str, float]] = []

    for s in sorted_survivors:
        if s.alpha_id not in rank_by_id:
            continue
        idx = rank_by_id[s.alpha_id]
        if not kept_idx:
            kept.append(s)
            kept_idx.append(idx)
            continue
        max_corr = float(np.max(np.abs(corr[idx, kept_idx])))
        worst_against = kept[int(np.argmax(np.abs(corr[idx, kept_idx])))].alpha_id
        if max_corr < threshold:
            kept.append(s)
            kept_idx.append(idx)
        else:
            skipped.append((s, worst_against, max_corr))

    if skipped:
        logger.info("skipped %d for correlation", len(skipped))
        for s, against, c in skipped[:5]:
            logger.info(
                "  %s (sharpe=%.2f) corr=%.3f vs %s", s.alpha_id, s.sharpe, c, against
            )
    return kept


# ---------- entrypoint ----------


def load_survivors(conn, thr: Thresholds) -> list[Survivor]:
    sql, params = build_query(thr)
    cur = conn.execute(sql, params)
    out: list[Survivor] = []
    for r in cur.fetchall():
        if not r["alpha_id"]:
            continue
        out.append(
            Survivor(
                alpha_id=r["alpha_id"],
                sharpe=float(r["is_sharpe"]),
                fitness=float(r["is_fitness"]) if r["is_fitness"] is not None else 0.0,
                template_name=r["template_name"],
                field_id=r["field_id"],
                expression=r["expression"],
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="alphas.db")
    p.add_argument("--out-csv", default="basket.csv")
    p.add_argument("--corr-threshold", type=float, default=0.7)
    p.add_argument("--min-sharpe", type=float, default=1.25)
    p.add_argument("--min-fitness", type=float, default=1.3)
    p.add_argument("--min-abs-margin", type=float, default=0.0050)
    p.add_argument("--min-turnover", type=float, default=0.01)
    p.add_argument("--max-turnover", type=float, default=0.7)
    p.add_argument("--require-self-corr-pass", action="store_true")
    p.add_argument("--force-refetch", action="store_true")
    args = p.parse_args(argv)

    load_dotenv()
    email = os.environ.get("WQ_EMAIL")
    password = os.environ.get("WQ_PASS")
    if not email or not password:
        print("ERROR: set WQ_EMAIL and WQ_PASS in .env", file=sys.stderr)
        return 1

    thr = Thresholds(
        min_sharpe=args.min_sharpe,
        min_fitness=args.min_fitness,
        min_abs_margin=args.min_abs_margin,
        max_turnover=args.max_turnover,
        min_turnover=args.min_turnover,
        require_self_correlation_pass=args.require_self_corr_pass,
    )

    conn = db.connect(args.db)
    try:
        survivors = load_survivors(conn, thr)
    finally:
        conn.close()

    if not survivors:
        logger.info("no survivors at thresholds %s", thr)
        return 0
    logger.info("loaded %d survivors", len(survivors))

    client = WQClient(email, password)
    series: dict[str, np.ndarray] = {}
    for s in survivors:
        try:
            series[s.alpha_id] = fetch_pnl(client, s.alpha_id, force=args.force_refetch)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pnl fetch failed for %s: %s", s.alpha_id, exc)

    alpha_order, matrix = align_returns(series)
    if not alpha_order:
        logger.error("no usable PnL series after alignment")
        return 2
    logger.info("aligned %d series x %d days", len(alpha_order), matrix.shape[1])

    corr = correlation_matrix(matrix)
    kept = greedy_dedup(survivors, corr, alpha_order, args.corr_threshold)
    logger.info("kept %d / %d survivors at corr threshold %.2f",
                len(kept), len(survivors), args.corr_threshold)

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["alpha_id", "sharpe", "fitness", "template_name", "field_id", "expression"])
        for s in kept:
            writer.writerow([s.alpha_id, s.sharpe, s.fitness, s.template_name, s.field_id, s.expression])

    print(f"\n--- basket ({len(kept)} alphas) ---")
    for s in kept:
        print(f"  {s.alpha_id:<10} sharpe={s.sharpe:>6.3f}  {s.template_name:<18} {s.field_id:<14} {s.expression}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
