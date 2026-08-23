"""In-env unit parity for the Rust contact-distance reader.

Compares against a literal pandas/numpy transcription of upstream's
compute_decay, on the committed fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_rust = pytest.importorskip("schicluster_rs._rust")

FIXTURE = "data/fixtures/contact_distance_small.tsv.gz"
PACK = np.load("data/fixtures/contact_distance_small.npz", allow_pickle=False)
CHROMS = [str(c) for c in PACK["contact_distance.chroms"]]
EDGES = PACK["contact_distance.bin_edges"]
RESOLUTION = int(PACK["contact_distance.resolution"])
C1, P1, C2, P2 = (int(x) for x in PACK["contact_distance.cols"])


def _upstream_reference():
    chrom_sizes = pd.Series(PACK["contact_distance.chrom_sizes"], index=CHROMS)
    data = pd.read_csv(FIXTURE, sep='\t', header=None, index_col=None)
    data = data.loc[(data[C1] == data[C2]) & data[C1].isin(chrom_sizes.index)]
    hist = np.histogram(np.abs(data[P2] - data[P1]), EDGES)[0]
    data[[P1, P2]] = data[[P1, P2]] // RESOLUTION
    grouped = data.groupby(by=[C1, P1, P2])[C2].count().reset_index()
    sparsity = grouped.loc[grouped[P1] != grouped[P2], C1].value_counts()
    return hist, sparsity


def _rust_call():
    hist, sparsity = _rust.py_contact_decay_cell(
        FIXTURE, CHROMS, np.ascontiguousarray(EDGES, dtype=np.float64),
        RESOLUTION, C1, P1, C2, P2,
    )
    return np.asarray(hist, dtype=np.int64), dict(sparsity)


def test_decay_histogram_matches_numpy():
    ref_hist, _ = _upstream_reference()
    got_hist, _ = _rust_call()
    assert got_hist.tolist() == ref_hist.tolist()


def test_sparsity_matches_pandas_groupby():
    _, ref_sparsity = _upstream_reference()
    _, got_sparsity = _rust_call()
    assert got_sparsity == {str(k): int(v) for k, v in ref_sparsity.items()}


def test_out_of_range_distances_are_dropped():
    """The fixture emits one contact at exactly edges[-1] (kept, final bin) and
    one beyond it (dropped), plus 20 below edges[0] (dropped)."""
    ref_hist, _ = _upstream_reference()
    got_hist, _ = _rust_call()
    assert got_hist[-1] == ref_hist[-1] >= 1


def test_unknown_chrom_and_trans_are_excluded():
    _, got_sparsity = _rust_call()
    assert "chrUn" not in got_sparsity
    assert set(got_sparsity).issubset(set(CHROMS))


def test_missing_file_raises():
    with pytest.raises(Exception):
        _rust.py_contact_decay_cell(
            "data/fixtures/does_not_exist.tsv.gz", CHROMS,
            np.ascontiguousarray(EDGES, dtype=np.float64),
            RESOLUTION, C1, P1, C2, P2,
        )


def test_wrapper_returns_upstream_frame_shapes():
    from schicluster_rs.contact_distance import compute_decay

    chrom_sizes = pd.DataFrame(PACK["contact_distance.chrom_sizes"], index=CHROMS)
    sparsity_df, decay_df = compute_decay(
        cell_name="cell_A", contact_path=FIXTURE, bins=EDGES,
        chrom_sizes=chrom_sizes, resolution=RESOLUTION,
        chrom1=C1, pos1=P1, chrom2=C2, pos2=P2,
    )
    ref_hist, ref_sparsity = _upstream_reference()
    assert list(decay_df.columns) == ["cell_A"]
    assert list(sparsity_df.columns) == ["cell_A"]
    assert decay_df["cell_A"].tolist() == ref_hist.tolist()
    for chrom, count in ref_sparsity.items():
        assert int(sparsity_df.loc[str(chrom), "cell_A"]) == int(count)
