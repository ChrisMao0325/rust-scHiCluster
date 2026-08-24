"""Python wrappers around the Rust domain-module kernels.

Two upstream targets:

* ``single_chrom_calculate_insulation_score`` — sliding-window insulation
  score per bin. Direct one-to-one with the Rust kernel.
* The rpy2 closure ``run_top_dom`` inside
  ``schicluster.domain.call_domain.call_domain_and_insulation`` — the
  pure-Rust TopDom returns the same 4-column DataFrame so the upstream
  orchestrator (chrom labelling, blacklist boundary aggregation, etc.)
  keeps working unchanged.
"""
from __future__ import annotations

import numpy as np


TAG_GAP = 0
TAG_DOMAIN = 1
TAG_BOUNDARY = 2
TAG_TO_NAME = {TAG_GAP: "gap", TAG_DOMAIN: "domain", TAG_BOUNDARY: "boundary"}


def insulation_score_chrom(matrix, window_size=10, save_count=False):
    """Drop-in replacement for
    schicluster.domain.call_domain.single_chrom_calculate_insulation_score.

    ``matrix`` can be a scipy.sparse matrix or a numpy array; we densify to
    f32 contiguous before passing to Rust (the fixture sizes are tiny so
    densification cost is irrelevant; production single-chrom matrices fit
    in memory at 25 kb resolution).
    """
    try:
        from schicluster_rs._rust import py_insulation_score_chrom as _rust_ins
    except ImportError:
        from schicluster.domain.call_domain import (
            single_chrom_calculate_insulation_score as _upstream,
        )
        return _upstream(matrix=matrix, window_size=window_size, save_count=save_count)

    if hasattr(matrix, "toarray"):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    dense_f32 = np.ascontiguousarray(dense, dtype=np.float32)
    return _rust_ins(dense_f32, int(window_size), bool(save_count))


def _topdom_chrom_to_df(matrix, bins, window_size, stat_filter=True):
    """Rust TopDom -> pandas.DataFrame with the same 4 columns the rpy2 closure emits.

    `bins` must be a DataFrame with the upstream columns
    ``['chr', 'from.coord', 'to.coord']`` — matches what
    ``call_domain_and_insulation`` constructs before calling ``run_top_dom``.
    Returns columns: ``chrom``, ``chromStart``, ``chromEnd``, ``name``.
    """
    import pandas as pd

    try:
        from schicluster_rs._rust import py_topdom_chrom as _rust_topdom
    except ImportError as e:
        raise ImportError("schicluster_rs Rust extension not available") from e

    if hasattr(matrix, "toarray"):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    dense_f32 = np.ascontiguousarray(dense, dtype=np.float32)
    rows = _rust_topdom(dense_f32, int(window_size), bool(stat_filter))

    if not rows:
        return pd.DataFrame([], columns=["chrom", "chromStart", "chromEnd", "name"])

    from_coord = bins["from.coord"].to_numpy()
    to_coord = bins["to.coord"].to_numpy()
    chrom_name = str(bins["chr"].iloc[0])

    n = len(from_coord)
    records = []
    for from_id, to_id, tag in rows:
        # R's Convert.Bin.To.Domain.TMP sets to.coord as from.coord of the next row,
        # which equals to_coord[to_id + 1] for non-terminal bins.  For the last bin
        # (to_id == n-1) the sentinel value to_coord[n-1] is already correct.
        end_bin = min(int(to_id) + 1, n - 1)
        records.append({
            "chrom": chrom_name,
            "chromStart": int(from_coord[from_id]),
            "chromEnd": int(to_coord[end_bin]),
            "name": TAG_TO_NAME[int(tag)],
        })
    return pd.DataFrame.from_records(records,
                                     columns=["chrom", "chromStart", "chromEnd", "name"])


def run_top_dom(j, p, x, bins, window_size):
    """Drop-in replacement for the rpy2 closure ``RunTopDom`` inside
    upstream ``schicluster/domain/TopDom.R``.

    ``j`` / ``p`` / ``x`` come from a scipy CSC matrix the upstream Python
    builds (matrix.indices+1, matrix.indptr, matrix.data). We rebuild the
    dense symmetric matrix here and hand it to the Rust kernel.
    """
    import pandas as pd

    from scipy.sparse import csc_matrix
    j = np.asarray(j).astype(np.int64) - 1  # back to 0-indexed
    p = np.asarray(p).astype(np.int64)
    x = np.asarray(x).astype(np.float32)
    n = len(bins)
    csc = csc_matrix((x, j, p), shape=(n, n))
    dense = np.asarray(csc.todense(), dtype=np.float32)
    bins_df = pd.DataFrame(bins) if not isinstance(bins, pd.DataFrame) else bins
    return _topdom_chrom_to_df(dense, bins_df, window_size, stat_filter=True)


__all__ = [
    "insulation_score_chrom",
    "run_top_dom",
    "TAG_GAP", "TAG_DOMAIN", "TAG_BOUNDARY",
]
