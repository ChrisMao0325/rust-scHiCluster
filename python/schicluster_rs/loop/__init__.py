"""Python wrappers around the Rust loop-module kernels.

Each wrapper preserves the upstream function signature so that
`patch_schicluster()` can monkey-patch the upstream module attributes
without breaking callers. The shuffle=True path of loop_bkg falls back
to the upstream Python implementation (RNG; out of parity scope).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, save_npz, triu


def loop_bkg_chrom(cell_url, chrom, resolution, output_prefix,
                   dist=10_050_000, cap=5, pad=5, gap=2,
                   min_cutoff=1e-6, log_e=False, shuffle=False):
    """Drop-in replacement for
    schicluster.loop.loop_bkg.calculate_chrom_background_normalization.

    Calls the Rust kernel for the deterministic path; falls back to
    upstream Python when shuffle=True (RNG; intentionally not ported).
    """
    if shuffle:
        from schicluster.loop.loop_bkg import (
            calculate_chrom_background_normalization as _upstream,
        )
        return _upstream(
            cell_url=cell_url, chrom=chrom, resolution=resolution,
            output_prefix=output_prefix, dist=dist, cap=cap, pad=pad, gap=gap,
            min_cutoff=min_cutoff, log_e=log_e, shuffle=True,
        )

    try:
        from schicluster_rs._rust import py_loop_bkg_chrom as _rust_loop_bkg
    except ImportError:
        from schicluster.loop.loop_bkg import (
            calculate_chrom_background_normalization as _upstream,
        )
        return _upstream(
            cell_url=cell_url, chrom=chrom, resolution=resolution,
            output_prefix=output_prefix, dist=dist, cap=cap, pad=pad, gap=gap,
            min_cutoff=min_cutoff, log_e=log_e, shuffle=False,
        )

    import cooler
    cool = cooler.Cooler(cell_url)
    matrix = triu(cool.matrix(balance=False, sparse=True).fetch(chrom)).astype(np.float32)
    n = matrix.shape[0]
    coo = matrix.tocoo()
    rows = coo.row.astype(np.uint32, copy=False)
    cols = coo.col.astype(np.uint32, copy=False)
    vals = coo.data.astype(np.float32, copy=False)

    dist_bins = int(dist // resolution)
    (er, ec, ev), (tr, tc, tv) = _rust_loop_bkg(
        rows, cols, vals, int(n), int(dist_bins),
        float(cap), int(pad), int(gap), float(min_cutoff), bool(log_e),
    )
    e_sparse = csr_matrix((ev, (er, ec)), shape=(n, n), dtype=np.float32)
    t_sparse = csr_matrix((tv, (tr, tc)), shape=(n, n), dtype=np.float32)
    save_npz(f"{output_prefix}.E.npz", e_sparse)
    save_npz(f"{output_prefix}.T.npz", t_sparse)
    return


def merge_cells_for_single_chromosome(output_dir, output_prefix, merge_type="E"):
    """Drop-in replacement for
    schicluster.loop.merge_cell_to_group.merge_cells_for_single_chromosome.

    Loads every *.<merge_type>.npz CSR matrix in output_dir, sums via Rust,
    writes the two HDF outputs the upstream emits.
    """
    try:
        from schicluster_rs._rust import py_merge_cells_sum as _rust_merge
    except ImportError:
        from schicluster.loop.merge_cell_to_group import (
            merge_cells_for_single_chromosome as _upstream,
        )
        return _upstream(output_dir=output_dir, output_prefix=output_prefix,
                         merge_type=merge_type)

    import pathlib
    import pandas as pd
    from scipy.sparse import load_npz

    def _write_coo(path, matrix):
        """Write a sparse matrix as a pixel DataFrame in HDF5 (no chunk split)."""
        coo = matrix.tocoo(copy=False)
        df = pd.DataFrame({
            "bin1_id": coo.row.astype(np.int64),
            "bin2_id": coo.col.astype(np.int64),
            "count": coo.data.astype(np.float32),
        })
        with pd.HDFStore(path, mode="w", complib="zlib", complevel=3) as hdf:
            hdf["c0"] = df

    cell_paths = sorted(str(p) for p in pathlib.Path(output_dir).glob(f"*.{merge_type}.npz"))
    if not cell_paths:
        raise FileNotFoundError(f"no *.{merge_type}.npz under {output_dir}")

    cell_ids_list, rows_list, cols_list, vals_list = [], [], [], []
    n_dims = None
    for cell_idx, path in enumerate(cell_paths):
        m = load_npz(path).tocoo()
        if n_dims is None:
            n_dims = m.shape[0]
        if m.nnz == 0:
            continue
        cell_ids_list.append(np.full(m.nnz, cell_idx, dtype=np.uint32))
        rows_list.append(m.row.astype(np.uint32, copy=False))
        cols_list.append(m.col.astype(np.uint32, copy=False))
        vals_list.append(m.data.astype(np.float32, copy=False))

    cell_ids = np.concatenate(cell_ids_list) if cell_ids_list else np.empty(0, np.uint32)
    rows = np.concatenate(rows_list) if rows_list else np.empty(0, np.uint32)
    cols = np.concatenate(cols_list) if cols_list else np.empty(0, np.uint32)
    vals = np.concatenate(vals_list) if vals_list else np.empty(0, np.float32)

    (er, ec, ev), (e2r, e2c, e2v) = _rust_merge(cell_ids, rows, cols, vals)
    e_sum = csr_matrix((ev, (er, ec)), shape=(n_dims, n_dims), dtype=np.float32)
    e2_sum = csr_matrix((e2v, (e2r, e2c)), shape=(n_dims, n_dims), dtype=np.float32)
    _write_coo(f"{output_prefix}.{merge_type}.hdf", e_sum)
    _write_coo(f"{output_prefix}.{merge_type}2.hdf", e2_sum)
    return


def loop_background(E, pad, gap, loop):
    """Drop-in replacement for
    schicluster.loop.loop_calling.loop_background.

    Returns 4 arrays (loop_bl, loop_donut, loop_h, loop_v).
    """
    try:
        from schicluster_rs._rust import py_scan_kernels_chrom as _rust_scan
    except ImportError:
        from schicluster.loop.loop_calling import loop_background as _upstream
        return _upstream(E=E, pad=pad, gap=gap, loop=loop)

    e32 = np.ascontiguousarray(E, dtype=np.float32)
    xs = np.ascontiguousarray(loop[0], dtype=np.uint32)
    ys = np.ascontiguousarray(loop[1], dtype=np.uint32)
    bl, donut, h, v = _rust_scan(e32, int(pad), int(gap), xs, ys)
    return bl, donut, h, v


def find_summit(loop, res, dist_thres):
    """Drop-in replacement for schicluster.loop.loop_calling.find_summit.

    Returns the subset of `loop` rows that are peaks, with an added 'size' column.
    """
    try:
        from schicluster_rs._rust import py_find_summit_chrom as _rust_summit
    except ImportError:
        from schicluster.loop.loop_calling import find_summit as _upstream
        return _upstream(loop=loop, res=res, dist_thres=dist_thres)

    loop = loop.copy()
    cord = (loop[["x1", "y1"]].values // res).astype(np.uint32)
    xs = np.ascontiguousarray(cord[:, 0])
    ys = np.ascontiguousarray(cord[:, 1])
    es = np.ascontiguousarray(loop["E"].values, dtype=np.float32)
    idx, sizes = _rust_summit(xs, ys, es, int(dist_thres))
    out = loop.iloc[np.asarray(idx, dtype=np.int64)].copy()
    out["size"] = np.asarray(sizes, dtype=np.int64)
    return out


__all__ = [
    "loop_bkg_chrom",
    "merge_cells_for_single_chromosome",
    "loop_background",
    "find_summit",
]
