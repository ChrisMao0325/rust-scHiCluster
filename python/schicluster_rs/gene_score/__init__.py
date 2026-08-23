"""Rust-backed replacements for scHiCluster's per-cell gene-score workers.

Drop-in for schicluster.draft.gene_score.gene_score_impute / gene_score_raw.
Cooler reads (impute mode) and the pandas groupby matrix build (raw mode) stay
in Python; only the per-gene window-sum loop crosses into Rust.

Upstream quirk preserved deliberately: gene_score_impute's window is
D[(xx-1):(yy+1), xx:(yy+2)]. When xx == 0 the row start is -1, which scipy
resolves to n-1, so the window is empty and the gene scores 0.0. Porting that
"correctly" would silently change every first-bin gene's score, so the Rust
kernel reproduces the slice semantics instead. See tutorial/gene_score.md.
"""
from __future__ import annotations

import numpy as np


def _window_sums(matrix, row_start, row_end, col_start, col_end):
    """Sum `matrix` over each rectangular window, via the Rust kernel."""
    from schicluster_rs._rust import py_gene_score_chrom

    csr = matrix.tocsr()
    csr.sort_indices()
    # Reduce in the CSR's own dtype: upstream's .sum() accumulates in float32
    # for f32 cools, and its result carries that rounding. See gene_score.rs.
    input_f32 = (csr.dtype == np.float32)
    return np.asarray(py_gene_score_chrom(
        np.ascontiguousarray(csr.indptr, dtype=np.int64),
        np.ascontiguousarray(csr.indices, dtype=np.int64),
        np.ascontiguousarray(csr.data, dtype=np.float64),
        int(csr.shape[0]), int(csr.shape[1]),
        np.ascontiguousarray(row_start, dtype=np.int64),
        np.ascontiguousarray(row_end, dtype=np.int64),
        np.ascontiguousarray(col_start, dtype=np.int64),
        np.ascontiguousarray(col_end, dtype=np.int64),
        bool(input_f32),
    ))


def gene_score_impute(cell_path, chrom_sizes, gene_meta):
    """Per-cell gene scores from an imputed .cool. Signature matches upstream."""
    import cooler
    from scipy.sparse import triu

    cool = cooler.Cooler(cell_path)
    result = []
    for chrom in chrom_sizes.index:
        matrix = triu(cool.matrix(balance=False, sparse=True).fetch(chrom), k=1).tocsr()
        gene = gene_meta.loc[gene_meta[0] == chrom, [1, 2]].values
        if gene.size == 0:
            continue
        xx = gene[:, 0].astype(np.int64)
        yy = gene[:, 1].astype(np.int64)
        result.extend(_window_sums(matrix, xx - 1, yy + 1, xx, yy + 2).tolist())
    return result


def gene_score_raw(cell_path, chrom_sizes, gene_meta, resolution,
                   chrom1, pos1, chrom2, pos2):
    """Per-cell gene scores straight from a contact file. Signature matches upstream.

    The matrix build below is a literal transcription of upstream so that the
    only behavioural difference is the window-sum kernel.
    """
    import pandas as pd
    from scipy.sparse import csr_matrix

    data = pd.read_csv(cell_path, sep='\t', index_col=None, header=None, comment='#')
    data = data.loc[(data[chrom1] == data[chrom2]) & data[chrom1].isin(chrom_sizes.index)]
    result = []
    for chrom in chrom_sizes.index:
        n_bins = (chrom_sizes.loc[chrom] // resolution) + 1
        chrfilter = (data[chrom1] == chrom)
        if chrfilter.sum() == 0:
            matrix = csr_matrix((n_bins, n_bins))
        else:
            D = data.loc[chrfilter].copy()
            D[[pos1, pos2]] = (D[[pos1, pos2]] - 1) // resolution
            D = D.groupby(by=[pos1, pos2])[chrom1].count().reset_index()
            matrix = csr_matrix((D[chrom1].astype(np.int32), (D[pos1], D[pos2])),
                                (n_bins, n_bins))
        gene = gene_meta.loc[gene_meta[0] == chrom, [1, 2]].values
        if gene.size == 0:
            continue
        xx = gene[:, 0].astype(np.int64)
        yy = gene[:, 1].astype(np.int64)
        # Raw mode carries int32 counts; the sums are exact integers, and
        # upstream's .sum() returns numpy.int64, so cast to match its dtype.
        sums = _window_sums(matrix, xx, yy + 1, xx, yy + 1)
        result.extend(np.rint(sums).astype(np.int64).tolist())
    return result


__all__ = ["gene_score_impute", "gene_score_raw"]
