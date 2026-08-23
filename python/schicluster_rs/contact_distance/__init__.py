"""Rust-backed replacement for scHiCluster's per-cell contact-distance worker.

Drop-in for schicluster.cool.contact_distance.compute_decay. Returns the same
[sparsity_frame, decay_frame] pair the upstream orchestrator concatenates, so
pd.concat / to_hdf downstream are untouched.

The bin edges are computed by numpy in the orchestrator and passed straight
through, so Rust never recomputes exp2 and there is no ULP drift to reason
about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_decay(cell_name, contact_path, bins, chrom_sizes, resolution,
                  chrom1=1, chrom2=5, pos1=2, pos2=6):
    """Distance-decay histogram + per-chrom sparsity for one cell."""
    from schicluster_rs._rust import py_contact_decay_cell

    hist, sparsity = py_contact_decay_cell(
        str(contact_path),
        [str(c) for c in chrom_sizes.index],
        np.ascontiguousarray(bins, dtype=np.float64),
        int(resolution),
        int(chrom1), int(pos1), int(chrom2), int(pos2),
    )
    # Upstream builds this from value_counts(), so chroms with no surviving
    # off-diagonal pair are simply absent. Preserve that.
    sparsity_series = pd.Series(
        {str(k): int(v) for k, v in sparsity},
        dtype='int64',
    )
    return [
        pd.DataFrame(sparsity_series).set_axis([cell_name], axis=1),
        pd.DataFrame(np.asarray(hist, dtype=np.int64), columns=[cell_name]),
    ]


__all__ = ["compute_decay"]
