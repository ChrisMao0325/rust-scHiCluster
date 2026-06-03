"""Run upstream Python references and dump JSON.

Invoked under $PYTHON_REF_ENV (schicluster env, py 3.6). Emits one JSON
block per manifest output that has a reference for it; later phases
extend this driver.

py3.6 NOTE: do NOT add `from __future__ import annotations` (added in
3.7) or PEP 585 subscripted generics -- schicluster env is Python 3.6.

Usage (from repo root):
    python tests/py_reference_driver.py
"""
import json
import pathlib
import shutil
import tempfile

import numpy as np
import pandas as pd

# Upstream imports (validated in $PYTHON_REF_ENV).
from scipy.ndimage import convolve
from scipy.sparse import csr_matrix, load_npz, save_npz, triu
from schicluster.loop.loop_bkg import calculate_chrom_background_normalization
from schicluster.loop.merge_cell_to_group import (
    merge_cells_for_single_chromosome as upstream_merge,
)
from schicluster.loop.loop_calling import loop_background as upstream_loop_background
from schicluster.loop.loop_calling import find_summit as upstream_find_summit


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONV_FIXTURE = REPO_ROOT / "data" / "fixtures" / "conv_small.npz"
LOOP_FIXTURE = REPO_ROOT / "data" / "fixtures" / "loop_small.npz"
LOOP_COOL = REPO_ROOT / "data" / "fixtures" / "loop_small.cool"
OUT = REPO_ROOT / "data" / "fixtures" / "reference_output.json"

# Phase 1 fixture constants (kept in sync with data/fixtures/synthesize.py).
LOOP_N_BINS = 200
LOOP_RESOLUTION = 10_000
LOOP_DIST = 20 * LOOP_RESOLUTION
LOOP_PAD = 5
LOOP_GAP = 2
LOOP_CAP = 5.0
LOOP_MIN_CUTOFF = 1e-6
LOOP_DIST_THRES_BP = 30_000
LOOP_SUMMIT_DIST_BINS = LOOP_DIST_THRES_BP // LOOP_RESOLUTION
LOOP_CHROM = "chr1"


def _load_npz(path):
    with np.load(str(path)) as z:
        return {k: z[k] for k in z.files}


def ref_conv_convolved(conv_pack):
    a = conv_pack["input"]
    k = conv_pack["kernel"]
    return convolve(a, k, mode="mirror").astype(np.float32).tolist()


def ref_loop_bkg(temp_dir):
    out_prefix = str(temp_dir / "loop_small")
    calculate_chrom_background_normalization(
        cell_url=str(LOOP_COOL),
        chrom=LOOP_CHROM,
        resolution=LOOP_RESOLUTION,
        output_prefix=out_prefix,
        dist=LOOP_DIST,
        cap=LOOP_CAP,
        pad=LOOP_PAD,
        gap=LOOP_GAP,
        min_cutoff=LOOP_MIN_CUTOFF,
        log_e=False,
        shuffle=False,
    )
    e_sparse = load_npz(out_prefix + ".E.npz").tocoo()
    t_sparse = load_npz(out_prefix + ".T.npz").tocoo()
    return (
        {
            "rows": e_sparse.row.astype(np.uint32).tolist(),
            "cols": e_sparse.col.astype(np.uint32).tolist(),
            "vals": e_sparse.data.astype(np.float32).tolist(),
        },
        {
            "rows": t_sparse.row.astype(np.uint32).tolist(),
            "cols": t_sparse.col.astype(np.uint32).tolist(),
            "vals": t_sparse.data.astype(np.float32).tolist(),
        },
    )


def ref_merge(loop_pack, temp_dir):
    cell_ids = loop_pack["merge.cell_ids"]
    rows = loop_pack["merge.input.rows"]
    cols = loop_pack["merge.input.cols"]
    vals = loop_pack["merge.input.vals"]
    n = LOOP_N_BINS
    unique_cells = sorted(np.unique(cell_ids).tolist())
    for cid in unique_cells:
        mask = cell_ids == cid
        m = csr_matrix(
            (vals[mask], (rows[mask], cols[mask])), shape=(n, n), dtype=np.float32
        )
        save_npz(str(temp_dir / "cell{}.E.npz".format(cid)), m)
    out_prefix = str(temp_dir / "merge_small")
    upstream_merge(output_dir=str(temp_dir), output_prefix=out_prefix, merge_type="E")
    e_sum_df = pd.read_hdf(out_prefix + ".E.hdf").reset_index(drop=True)
    e2_sum_df = pd.read_hdf(out_prefix + ".E2.hdf").reset_index(drop=True)
    return (
        {
            "rows": e_sum_df["bin1_id"].astype(np.uint32).tolist(),
            "cols": e_sum_df["bin2_id"].astype(np.uint32).tolist(),
            "vals": e_sum_df["count"].astype(np.float32).tolist(),
        },
        {
            "rows": e2_sum_df["bin1_id"].astype(np.uint32).tolist(),
            "cols": e2_sum_df["bin2_id"].astype(np.uint32).tolist(),
            "vals": e2_sum_df["count"].astype(np.float32).tolist(),
        },
    )


def ref_scan_kernels(loop_pack):
    e = loop_pack["scan.E_dense"]
    xs = loop_pack["scan.loop_xs"]
    ys = loop_pack["scan.loop_ys"]
    loop_bl, loop_donut, loop_h, loop_v = upstream_loop_background(
        E=e, pad=LOOP_PAD, gap=LOOP_GAP, loop=(xs, ys)
    )
    return {
        "bl": np.asarray(loop_bl, dtype=np.float32).tolist(),
        "donut": np.asarray(loop_donut, dtype=np.float32).tolist(),
        "h": np.asarray(loop_h, dtype=np.float32).tolist(),
        "v": np.asarray(loop_v, dtype=np.float32).tolist(),
    }


def ref_find_summit(loop_pack):
    x1 = loop_pack["summit.x1"]
    y1 = loop_pack["summit.y1"]
    es = loop_pack["summit.E"]
    df = pd.DataFrame({"x1": x1, "y1": y1, "E": es})
    summit_df = upstream_find_summit(loop=df, res=LOOP_RESOLUTION,
                                     dist_thres=LOOP_SUMMIT_DIST_BINS)
    selected = summit_df.index.to_numpy().astype(np.uint32)
    sizes = summit_df["size"].to_numpy().astype(np.uint32)
    return {"idx": selected.tolist(), "sizes": sizes.tolist()}


def main():
    conv_pack = _load_npz(CONV_FIXTURE)
    loop_pack = _load_npz(LOOP_FIXTURE)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    payload = {"convolved": ref_conv_convolved(conv_pack)}

    # Use separate temp dirs so that *.E.npz globs do not cross-contaminate
    # loop_bkg output (loop_small.E.npz) with merge cell inputs.
    bkg_dir = pathlib.Path(tempfile.mkdtemp(prefix="schicluster_rs_ref_bkg_"))
    merge_dir = pathlib.Path(tempfile.mkdtemp(prefix="schicluster_rs_ref_merge_"))
    try:
        e, t = ref_loop_bkg(bkg_dir)
        payload["loop_bkg"] = {"E": e, "T": t}

        e_sum, e2_sum = ref_merge(loop_pack, merge_dir)
        payload["merge"] = {"e_sum": e_sum, "e2_sum": e2_sum}

        payload["scan_kernels"] = ref_scan_kernels(loop_pack)
        payload["find_summit"] = ref_find_summit(loop_pack)
    finally:
        shutil.rmtree(str(bkg_dir), ignore_errors=True)
        shutil.rmtree(str(merge_dir), ignore_errors=True)

    OUT.write_text(json.dumps(payload))
    print("wrote {} (top-level keys: {})".format(OUT, sorted(payload.keys())))


if __name__ == "__main__":
    main()
