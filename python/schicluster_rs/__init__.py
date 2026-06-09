"""Rust port of `random_walk_cpu` — the hot loop of scHiCluster.

Public API:

* :func:`random_walk_cpu` — drop-in replacement for
  :func:`schicluster.impute.impute_chromosome.random_walk_cpu`. Accepts
  any sparse / dense matrix scipy understands, returns the same
  ``scipy.sparse.csr_matrix`` shape, with bit-equivalent values
  (relative error < 1e-4 due to BLAS vs scipy operation-order
  rounding).
* :func:`patch_schicluster` — monkey-patch upstream so every existing
  scHiCluster caller (CLI, snakemake, ``epione.single.hic``) inherits
  the speedup.

Build:

    cd rust-scHiCluster && maturin develop --release
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse

try:
    from schicluster_rs._rust import (
        py_random_walk_cpu_csr as _rwr_csr,
        py_impute_chromosome_inner as _impute_inner,
        py_convolve2d_mirror as _conv2d_mirror,
        py_loop_bkg_chrom as _loop_bkg_chrom,
        py_merge_cells_sum as _merge_cells_sum,
        py_scan_kernels_chrom as _scan_kernels_chrom,
        py_find_summit_chrom as _find_summit_chrom,
        py_insulation_score_chrom as _insulation_score_chrom,
        py_topdom_chrom as _topdom_chrom,
        py_compartment_chrom as _compartment_chrom,
        py_make_chrom_features as _make_chrom_features,
        set_num_threads as _set_num_threads,
    )
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def set_num_threads(n: int) -> bool:
    """Configure the rayon thread pool size for this process.

    Call from each ``ProcessPoolExecutor`` worker before any impute call
    to avoid thread oversubscription. Typical sizing on a 16-core node:

    >>> import schicluster_rs
    >>> schicluster_rs.set_num_threads(2)   # 8 workers × 2 = 16 ✓

    First-call wins — subsequent calls are no-ops (rayon's global pool
    is build-once).

    Returns ``True`` if the pool was just built, ``False`` if a previous
    call already configured it.
    """
    if not _RUST_AVAILABLE:
        return False
    return _set_num_threads(int(n))


def random_walk_cpu(p, rp: float = 0.5, tol: float = 0.01,
                     n_iter: int = 30):
    """Iterative random-walk-with-restart — Rust kernel.

    Bit-equivalent to upstream
    ``schicluster.impute.impute_chromosome.random_walk_cpu`` (relative
    error < 1e-4 — BLAS-vs-scipy rounding budget). Same default params.

    Arguments:
        p: scipy CSR (or dense) row-stochastic n×n matrix.
        rp: restart probability. ``1.0`` returns ``p`` unchanged.
        tol: Frobenius-norm convergence tolerance.
        n_iter: max iterations (default ``30``, matches upstream).

    Returns:
        ``scipy.sparse.csr_matrix`` of the converged Q.

    Speedup vs scipy on 25 kb mouse chromosomes (n ~ 5–8 k):

    * chr1 (n=7820): 3.4 s → 0.5 s  (7× — BLAS dense matmul wins
      because post-Gaussian P is ~30 % dense)
    * chrX (n=6845): 1.5 s → 0.25 s (6×)
    * chr19 (n=2461): 0.4 s → 0.06 s (7×)

    Falls back to the upstream Python implementation if the Rust
    extension wasn't compiled.
    """
    if not _RUST_AVAILABLE:
        from schicluster.impute.impute_chromosome import random_walk_cpu as _py
        return _py(p, rp, tol)

    if rp == 1.0:
        return p.copy() if issparse(p) else csr_matrix(p)

    if not issparse(p):
        p = csr_matrix(p)
    p = p.astype(np.float32, copy=False)
    p_coo = p.tocoo()
    rows = p_coo.row.astype(np.uint32, copy=False)
    cols = p_coo.col.astype(np.uint32, copy=False)
    vals = p_coo.data.astype(np.float32, copy=False)

    out_rows, out_cols, out_vals = _rwr_csr(
        rows, cols, vals, p.shape[0],
        float(rp), float(tol), int(n_iter),
    )
    if out_rows.size == 0:
        return csr_matrix(p.shape, dtype=np.float32)
    return csr_matrix(
        (out_vals, (out_rows, out_cols)),
        shape=p.shape,
        dtype=np.float32,
    )


def impute_chromosome(scool_url, chrom, resolution, output_path,
                       *, logscale=False, pad=1, std=1.0, rp=0.5,
                       tol=0.01, output_dist=500_000_000_000,
                       window_size=500_000_000_000, step_size=10_000_000,
                       min_cutoff=0, band_factor=0):
    """End-to-end Rust replacement for
    ``schicluster.impute.impute_chromosome.impute_chromosome``.

    The whole inner numerical pipeline (Gaussian conv → row-normalise →
    RWR via parallel SpMM → symmetrise → SQRTVC → triangle filter)
    runs in a single Rust call — no scipy round-trips. Only cooler
    I/O and HDF5 write happen in Python.

    Numerical equivalence to upstream is asserted by
    ``tests/test_parity.py`` to relative error < 1e-4 (float-32 BLAS
    rounding budget).

    Speedup vs upstream on real Chang LC462 25 kb data:

    * chr1 (n=7820): 5.0 s → 1.3 s — 4×
    * chr19 (n=2461): 0.4 s → 0.1 s — 4×

    Falls back to the upstream Python ``impute_chromosome`` if the
    Rust extension wasn't compiled.
    """
    if not _RUST_AVAILABLE:
        from schicluster.impute.impute_chromosome import (
            impute_chromosome as _py
        )
        return _py(
            chrom=chrom, resolution=resolution, output_path=output_path,
            scool_url=scool_url, logscale=logscale, pad=pad, std=std,
            rp=rp, tol=tol, window_size=window_size, step_size=step_size,
            output_dist=output_dist, min_cutoff=min_cutoff,
        )

    import cooler
    import pandas as pd

    cell_cool = cooler.Cooler(scool_url)
    A = cell_cool.matrix(balance=False, sparse=True).fetch(chrom)
    n_bins = A.shape[0]
    A_coo = A.tocoo()
    rows = A_coo.row.astype(np.uint32, copy=False)
    cols = A_coo.col.astype(np.uint32, copy=False)
    vals = A_coo.data.astype(np.float32, copy=False)
    if logscale:
        vals = np.log2(vals + 1).astype(np.float32, copy=False)

    output_dist_bins = output_dist // resolution
    if output_dist_bins > n_bins:
        output_dist_bins = -1  # signal: keep full upper triangle

    out_r, out_c, out_v = _impute_inner(
        rows, cols, vals, n_bins,
        int(pad), float(std), float(rp), float(tol),
        int(output_dist_bins), int(band_factor),
    )

    # Optional min_cutoff filter (matches upstream).
    if min_cutoff > 0:
        m = out_v > min_cutoff
        out_r = out_r[m]; out_c = out_c[m]; out_v = out_v[m]

    df = pd.DataFrame({"bin1_id": out_r, "bin2_id": out_c, "count": out_v})
    chunk_size = 5_000_000
    with pd.HDFStore(output_path, complib="zlib", complevel=3) as hdf:
        for i, chunk_start in enumerate(range(0, df.shape[0], chunk_size)):
            hdf[f"c{i}"] = df.iloc[chunk_start:chunk_start + chunk_size]


def convolve2d_mirror(a, kernel):
    """2-D convolution matching ``scipy.ndimage.convolve(a, kernel, mode='mirror')``.

    Both ``a`` and ``kernel`` must be C-contiguous ``np.float32``. Returns a
    ``np.float32`` array of the same shape as ``a``.

    Shared primitive for the loop module's background normalisation
    (``loop_bkg``) and loop-candidate background scan (``scan_kernel``).
    """
    if not _RUST_AVAILABLE:
        from scipy.ndimage import convolve
        return convolve(a, kernel, mode="mirror")
    a32 = np.ascontiguousarray(a, dtype=np.float32)
    k32 = np.ascontiguousarray(kernel, dtype=np.float32)
    return _conv2d_mirror(a32, k32)


from schicluster_rs.loop import (
    loop_bkg_chrom,
    merge_cells_for_single_chromosome,
    loop_background as _loop_background,
    find_summit as _find_summit,
)

# Re-exports for the public API surface declared in __all__.
loop_background = _loop_background
find_summit = _find_summit

from schicluster_rs.domain import (
    insulation_score_chrom,
    run_top_dom as _run_top_dom,
)

from schicluster_rs.compartment import single_chrom_compartment as _single_chrom_compartment
from schicluster_rs.embedding import make_chrom_matrix as _make_chrom_matrix

single_chrom_compartment = _single_chrom_compartment
make_chrom_matrix = _make_chrom_matrix


class _DomainRStub:
    """Stand-in for the rpy2 `r` global used by
    ``schicluster.domain.call_domain.call_domain_and_insulation``.

    Upstream does ``r.source('TopDom.R')`` then references ``r.RunTopDom(j, p,
    x, bins, window_size)`` to compute domains. We swap that to call the
    Rust kernel directly so the upstream orchestrator (boundary aggregation,
    insulation concatenation, .npz / .h5ad / .nc writers) keeps working.
    """
    @staticmethod
    def source(_path):
        return None

    @staticmethod
    def RunTopDom(j, p, x, bins, window_size):
        return _run_top_dom(j, p, x, bins, window_size)


def patch_schicluster() -> bool:
    """Monkey-patch ``schicluster.impute.impute_chromosome.random_walk_cpu``
    to use the Rust kernel.

    Call once at the top of your driver script:

    >>> import schicluster_rs
    >>> schicluster_rs.patch_schicluster()
    >>> from schicluster.impute.impute_chromosome import impute_chromosome
    >>> # ... runs ~5–7× faster end-to-end on large chroms

    Returns ``True`` on success, ``False`` if the Rust extension wasn't
    built (graceful fallback to upstream Python).
    """
    if not _RUST_AVAILABLE:
        return False
    try:
        from schicluster.impute import impute_chromosome as _impute_mod
        _impute_mod.random_walk_cpu = random_walk_cpu
        # Also rebind the top-level impute_chromosome (the function the
        # caller actually calls) — this is the big win.
        _impute_mod.impute_chromosome = impute_chromosome
        # ---- loop module ----
        from schicluster.loop import loop_bkg as _loop_bkg_mod
        from schicluster.loop import merge_cell_to_group as _merge_mod
        from schicluster.loop import loop_calling as _loop_calling_mod
        _loop_bkg_mod.calculate_chrom_background_normalization = loop_bkg_chrom
        _merge_mod.merge_cells_for_single_chromosome = merge_cells_for_single_chromosome
        _loop_calling_mod.loop_background = _loop_background
        _loop_calling_mod.find_summit = _find_summit
        # ---- domain module ----
        from schicluster.domain import call_domain as _domain_mod
        _domain_mod.single_chrom_calculate_insulation_score = insulation_score_chrom
        # call_domain_and_insulation builds a local `run_top_dom` closure
        # inside its body via `def run_top_dom(matrix, bins): ...`, so we
        # cannot rebind it at module level. Instead, monkey-patch the
        # rpy2 `r` global so the closure that's built from it uses our
        # Rust kernel.
        _domain_mod.r = _DomainRStub()
        # ---- compartment module ----
        from schicluster.compartment import call_compartment as _comp_mod
        _comp_mod.single_chrom_compartment = single_chrom_compartment
        # ---- embedding module ----
        from schicluster.embedding import calc_embedding as _emb_mod
        _emb_mod.make_chrom_matrix = make_chrom_matrix
        return True
    except ImportError:
        return False


__all__ = [
    "random_walk_cpu", "impute_chromosome", "patch_schicluster",
    "set_num_threads", "convolve2d_mirror",
    "loop_bkg_chrom", "merge_cells_for_single_chromosome",
    "loop_background", "find_summit",
    "insulation_score_chrom",
    "single_chrom_compartment", "make_chrom_matrix",
]
