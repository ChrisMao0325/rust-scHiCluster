"""Python wrapper around the Rust compartment kernel.

Drop-in replacement for
schicluster.compartment.call_compartment.single_chrom_compartment.
"""
from __future__ import annotations

import numpy as np


def single_chrom_compartment(matrix, cpg_ratio, calc_strength=False):
    """Match upstream signature: (matrix, cpg_ratio) -> (comp, scores).

    Falls back to upstream when the Rust extension is unavailable.
    """
    try:
        from schicluster_rs._rust import py_compartment_chrom as _rust_compartment
    except ImportError:
        from schicluster.compartment.call_compartment import (
            single_chrom_compartment as _upstream,
        )
        return _upstream(matrix=matrix, cpg_ratio=cpg_ratio, calc_strength=calc_strength)

    if hasattr(matrix, "toarray"):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    dense_f32 = np.ascontiguousarray(dense, dtype=np.float32)

    # cpg_ratio may be a pandas Series (upstream's call site does that); coerce.
    cpg_arr = cpg_ratio.values if hasattr(cpg_ratio, "values") else np.asarray(cpg_ratio)
    cpg_f32 = np.ascontiguousarray(cpg_arr, dtype=np.float32)

    comp, strength = _rust_compartment(dense_f32, cpg_f32, bool(calc_strength))
    comp_arr = np.asarray(comp)
    if calc_strength:
        return comp_arr, np.asarray(strength)
    else:
        return comp_arr, None


__all__ = ["single_chrom_compartment"]
