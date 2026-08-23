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
        bool(D.dtype == np.float32),
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


def test_wrapper_matches_upstream_on_the_fixture():
    """End-to-end within this env: our impute wrapper vs a literal transcription
    of upstream's loop, on the committed fixture."""
    cooler = pytest.importorskip("cooler")
    import pandas as pd
    from scipy.sparse import triu
    from schicluster_rs.gene_score import gene_score_impute

    pack = np.load("data/fixtures/gene_score_small.npz", allow_pickle=False)
    chrom_sizes = pd.Series(pack["gene_score.chrom_size"],
                            index=[str(c) for c in pack["gene_score.chrom"]])
    gene_meta = pd.DataFrame(
        {0: [str(pack["gene_score.chrom"][0])] * pack["gene_score.gene_id"].size,
         1: pack["gene_score.gene_start_bin"],
         2: pack["gene_score.gene_end_bin"]},
        index=[str(g) for g in pack["gene_score.gene_id"]],
    )
    cool_path = "data/fixtures/gene_score_small.cool"

    def upstream(cell_path, chrom_sizes, gene_meta):
        cool = cooler.Cooler(cell_path)
        result = []
        for chrom in chrom_sizes.index:
            D = triu(cool.matrix(balance=False, sparse=True).fetch(chrom), k=1).tocsr()
            gene = gene_meta.loc[gene_meta[0] == chrom, [1, 2]].values
            for xx, yy in gene:
                result.append(D[(xx - 1):(yy + 1), xx:(yy + 2)].sum())
        return result

    ref = np.asarray(upstream(cool_path, chrom_sizes, gene_meta), dtype=np.float64)
    got = np.asarray(gene_score_impute(cool_path, chrom_sizes, gene_meta), dtype=np.float64)
    assert got.shape == ref.shape
    # Bit-exact, not merely within the 1e-6 gate: the kernel reproduces scipy's
    # matvec-then-pairwise reduction in the source dtype. A regression to a
    # plain f64 accumulation would show up here as ~3.8e-6 of drift.
    assert got.tolist() == ref.tolist()
    # the bin-0 gene must be zero in BOTH
    assert ref[0] == 0.0 and got[0] == 0.0


def test_bit_exact_vs_scipy_on_random_float32_windows():
    """scipy's csr.sum(axis=None) is a ones-matvec (serial per row) followed by
    np.add.reduce (pairwise) over the row sums, all in the source dtype. On f32
    input a naive f64 accumulation drifts ~1e-6, so assert exactness."""
    rng = np.random.default_rng(11)
    n = 60
    for _ in range(50):
        dense = rng.exponential(1.0, (n, n)).astype(np.float32)
        dense[dense < 0.8] = 0.0
        D = csr_matrix(np.triu(dense, k=1))
        assert D.dtype == np.float32
        r0 = rng.integers(0, n - 2, 40)
        c0 = rng.integers(0, n - 2, 40)
        r1 = r0 + rng.integers(1, 25, 40)
        c1 = c0 + rng.integers(1, 25, 40)
        got = _call(D, r0, r1, c0, c1)
        ref = _scipy_ref(D, r0, r1, c0, c1)
        assert got.tolist() == ref.tolist()
