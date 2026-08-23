"""In-env unit parity for the Rust gene-score window-sum kernel.

Runs entirely inside $RUST_TEST_ENV against scipy directly, so it is fast
and needs no cross-env dump. The cross-env manifest gate in
test_exact_match.py is the authoritative check; this file catches
semantic regressions in seconds.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

_rust = pytest.importorskip("schicluster_rs._rust")


def _call(D, row_start, row_end, col_start, col_end):
    D = D.tocsr()
    D.sort_indices()
    return np.asarray(_rust.py_gene_score_chrom(
        np.ascontiguousarray(D.indptr, dtype=np.int64),
        np.ascontiguousarray(D.indices, dtype=np.int64),
        np.ascontiguousarray(D.data, dtype=np.float64),
        int(D.shape[0]), int(D.shape[1]),
        np.ascontiguousarray(row_start, dtype=np.int64),
        np.ascontiguousarray(row_end, dtype=np.int64),
        np.ascontiguousarray(col_start, dtype=np.int64),
        np.ascontiguousarray(col_end, dtype=np.int64),
    ))


def _scipy_ref(D, row_start, row_end, col_start, col_end):
    return np.asarray([
        D[r0:r1, c0:c1].sum()
        for r0, r1, c0, c1 in zip(row_start, row_end, col_start, col_end)
    ], dtype=np.float64)


@pytest.fixture
def dense_csr():
    n = 10
    return csr_matrix(np.arange(n * n).reshape(n, n).astype(np.float64))


def test_negative_row_start_resolves_to_n_minus_1(dense_csr):
    """xx=0 makes the impute window's row start -1; scipy reads that as n-1,
    yielding an empty window. The kernel must reproduce that, not 'fix' it."""
    got = _call(dense_csr, [-1], [4], [0], [5])
    assert got[0] == 0.0
    assert _scipy_ref(dense_csr, [-1], [4], [0], [5])[0] == 0.0


def test_column_overrun_clips(dense_csr):
    got = _call(dense_csr, [7], [10], [8], [11])
    assert got[0] == pytest.approx(_scipy_ref(dense_csr, [7], [10], [8], [11])[0])


def test_empty_and_inverted_windows_are_zero(dense_csr):
    got = _call(dense_csr, [5, 8], [5, 3], [0, 0], [10, 10])
    assert list(got) == [0.0, 0.0]


def test_matches_scipy_on_random_sparse_windows():
    rng = np.random.default_rng(7)
    n = 120
    dense = rng.exponential(1.0, (n, n))
    dense[dense < 1.2] = 0.0
    D = csr_matrix(np.triu(dense, k=1))
    n_g = 300
    r0 = rng.integers(-1, n, n_g)
    r1 = r0 + rng.integers(0, 20, n_g)
    c0 = rng.integers(0, n, n_g)
    c1 = c0 + rng.integers(0, 20, n_g)
    got = _call(D, r0, r1, c0, c1)
    ref = _scipy_ref(D, r0, r1, c0, c1)
    assert np.max(np.abs(got - ref)) < 1e-6


def test_no_genes_returns_empty(dense_csr):
    assert _call(dense_csr, [], [], [], []).size == 0
