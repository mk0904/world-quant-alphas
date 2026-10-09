import numpy as np

from wq_pipeline.correlation import (
    Survivor,
    _records_to_returns,
    align_returns,
    correlation_matrix,
    greedy_dedup,
)


def test_records_to_returns_diffs_cumulative_pnl():
    records = [["2025-01-01", 100.0], ["2025-01-02", 150.0], ["2025-01-03", 130.0]]
    rets = _records_to_returns(records)
    np.testing.assert_array_equal(rets, np.array([50.0, -20.0]))


def test_records_to_returns_handles_short_series():
    assert _records_to_returns([["2025-01-01", 100.0]]).size == 0
    assert _records_to_returns([]).size == 0


def test_align_returns_drops_mismatched_lengths():
    series = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([1.0, 2.0]), "C": np.array([4.0, 5.0, 6.0])}
    ids, mat = align_returns(series)
    assert ids == ["A", "C"]
    assert mat.shape == (2, 3)


def test_correlation_matrix_diagonal_ones():
    rng = np.random.default_rng(seed=42)
    mat = rng.standard_normal((3, 200))
    corr = correlation_matrix(mat)
    np.testing.assert_allclose(np.diag(corr), [1.0, 1.0, 1.0])
    assert corr.shape == (3, 3)


def test_greedy_dedup_keeps_top_then_drops_correlated():
    """Three alphas: A and B are highly correlated; C is independent.
    Sorted by |Sharpe|: A (best), B, C. A is kept first, B drops (corr≈1 with A), C kept.
    """
    n = 500
    rng = np.random.default_rng(seed=7)
    a = rng.standard_normal(n)
    b = a + rng.standard_normal(n) * 0.05  # near-duplicate of A
    c = rng.standard_normal(n)             # independent
    matrix = np.vstack([a, b, c])
    corr = correlation_matrix(matrix)

    survivors = [
        Survivor(alpha_id="A", sharpe=1.5, fitness=1.0, template_name="t", field_id="x", expression="A"),
        Survivor(alpha_id="B", sharpe=1.4, fitness=1.0, template_name="t", field_id="y", expression="B"),
        Survivor(alpha_id="C", sharpe=1.3, fitness=1.0, template_name="t", field_id="z", expression="C"),
    ]
    kept = greedy_dedup(survivors, corr, ["A", "B", "C"], threshold=0.7)
    kept_ids = {s.alpha_id for s in kept}
    assert "A" in kept_ids and "C" in kept_ids
    assert "B" not in kept_ids
