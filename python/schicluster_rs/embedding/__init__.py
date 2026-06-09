"""Python wrapper around the Rust embedding cell-by-feature kernel.

Drop-in replacement for
schicluster.embedding.calc_embedding.make_chrom_matrix.

The upstream function reads .cool files per cell; we preserve that surface
(reading cool stays Python) and only the in-memory extraction goes through
Rust. SVD is intentionally untouched and continues to use sklearn.
"""
from __future__ import annotations

import numpy as np


def make_chrom_matrix(cell_table, chrom, nbins, output_path,
                      scale_factor, dist, resolution):
    """Reads cool files per cell, calls Rust extraction kernel, writes npz.

    Output array shape matches upstream's: (n_cells, n_features) float32.
    """
    try:
        from schicluster_rs._rust import py_make_chrom_features as _rust_features
    except ImportError:
        from schicluster.embedding.calc_embedding import (
            make_chrom_matrix as _upstream,
        )
        return _upstream(
            cell_table=cell_table, chrom=chrom, nbins=nbins,
            output_path=output_path, scale_factor=scale_factor,
            dist=dist, resolution=resolution,
        )

    import cooler
    n_cells = cell_table.size
    cells = np.zeros((n_cells, nbins, nbins), dtype=np.float32)
    for i, (_, cell_url) in enumerate(cell_table.items()):
        cool = cooler.Cooler(cell_url)
        cells[i, :, :] = np.ascontiguousarray(
            cool.matrix(balance=False, sparse=False).fetch(chrom),
            dtype=np.float32,
        )
    dist_bins_plus_1 = int(dist / resolution + 1)
    out = _rust_features(cells, dist_bins_plus_1, float(scale_factor))
    np.savez(output_path, np.asarray(out, dtype=np.float32))
    return


__all__ = ["make_chrom_matrix"]
