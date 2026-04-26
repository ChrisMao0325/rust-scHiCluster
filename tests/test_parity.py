"""Parity test: Rust random_walk_cpu must match scipy / schicluster bit-for-bit.

We seed a deterministic sparse contact matrix that mimics the post-Gaussian
state inside `impute_chromosome` (sparse, symmetric, ~5 % density on a small
chromosome), row-normalize it, and run both kernels with the published
defaults (rp=0.5, tol=0.01, n_iter=30).

Bit-identical isn't quite achievable across libraries because both implement
sparse matmul slightly differently in operation order; we accept ‖A − B‖_∞
< 1e-5 (relative) which is the same float-32 rounding budget cooler / scipy
tolerate internally.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm as sparse_norm


def _make_test_csr(n: int = 200, density: float = 0.05, seed: int = 0):
    """Synthetic post-Gaussian contact CSR — symmetric, row-stochastic."""
    rng = np.random.default_rng(seed)
    nnz = int(density * n * n)
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    vals = rng.exponential(1.0, size=nnz).astype(np.float32)
    M = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    M = M + M.T  # symmetric
    # Row-normalize (replicates the (d.dot(B)) step in impute_chromosome).
    row_sum = np.asarray(M.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    P = diags(1.0 / row_sum) @ M
    return P.astype(np.float32)


@pytest.mark.parametrize("n", [50, 200, 500])
@pytest.mark.parametrize("rp", [0.05, 0.5, 0.9])
def test_random_walk_parity(n: int, rp: float):
    """Rust vs scipy: ‖Q_rust − Q_py‖_∞ / ‖Q_py‖_∞ < 1e-4."""
    schicluster_rs = pytest.importorskip("schicluster_rs")
    if not schicluster_rs._RUST_AVAILABLE:
        pytest.skip("rust extension not built — run `maturin develop --release`")

    pytest.importorskip("schicluster")
    from schicluster.impute import impute_chromosome as schicluster_imp
    P = _make_test_csr(n=n, seed=hash((n, rp)) % 2**31)

    Q_py = schicluster_imp.random_walk_cpu(P, rp, tol=0.01).toarray()
    Q_rs = schicluster_rs.random_walk_cpu(P, rp=rp, tol=0.01, n_iter=30).toarray()

    rel_err = np.max(np.abs(Q_py - Q_rs)) / max(np.max(np.abs(Q_py)), 1e-9)
    assert rel_err < 1e-4, (
        f"n={n} rp={rp}: rust deviates from scipy by {rel_err:.2e} "
        f"(>{1e-4:.0e}). Sample diff matrix abs.max="
        f"{np.max(np.abs(Q_py - Q_rs)):.3e}"
    )


@pytest.mark.parametrize("rp", [1.0])
def test_random_walk_rp_one_returns_p(rp: float):
    """rp=1 must early-out to P unchanged (matches upstream)."""
    schicluster_rs = pytest.importorskip("schicluster_rs")
    if not schicluster_rs._RUST_AVAILABLE:
        pytest.skip("rust extension not built")
    P = _make_test_csr(n=64, seed=0)
    Q = schicluster_rs.random_walk_cpu(P, rp=rp, tol=0.01).toarray()
    assert np.allclose(Q, P.toarray(), atol=0)


def test_patch_schicluster():
    """patch_schicluster() should rebind upstream module's reference."""
    schicluster_rs = pytest.importorskip("schicluster_rs")
    if not schicluster_rs._RUST_AVAILABLE:
        pytest.skip("rust extension not built")
    pytest.importorskip("schicluster")
    from schicluster.impute import impute_chromosome as schicluster_imp
    original = schicluster_imp.random_walk_cpu
    try:
        ok = schicluster_rs.patch_schicluster()
        assert ok
        assert schicluster_imp.random_walk_cpu is schicluster_rs.random_walk_cpu
    finally:
        schicluster_imp.random_walk_cpu = original
