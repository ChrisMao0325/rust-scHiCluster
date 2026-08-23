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
from schicluster.domain.call_domain import (
    single_chrom_calculate_insulation_score as upstream_insulation,
)
from schicluster.compartment.call_compartment import (
    single_chrom_compartment as upstream_compartment,
)
from schicluster.embedding.calc_embedding import make_idx as upstream_make_idx
from schicluster.draft.gene_score import (
    gene_score_impute as upstream_gene_score_impute,
    gene_score_raw as upstream_gene_score_raw,
)
from schicluster.cool.contact_distance import compute_decay as upstream_compute_decay
from rpy2.robjects import r as _rpy2_r, pandas2ri as _rpy2_pandas2ri, numpy2ri as _rpy2_numpy2ri
import schicluster as _schicluster
import pathlib as _pathlib_for_topdom
_rpy2_pandas2ri.activate()
_rpy2_numpy2ri.activate()
_rpy2_r.source(str(_pathlib_for_topdom.Path(_schicluster.__path__[0]) / 'domain/TopDom.R'))


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

DOMAIN_FIXTURE = REPO_ROOT / "data" / "fixtures" / "domain_small.npz"
DOMAIN_WINDOW_SIZE = 5
DOMAIN_BIN_RESOLUTION = 10_000
DOMAIN_CHROM = "chr1"

COMPARTMENT_FIXTURE = REPO_ROOT / "data" / "fixtures" / "compartment_small.npz"
EMBEDDING_FIXTURE = REPO_ROOT / "data" / "fixtures" / "embedding_small.npz"
GENE_SCORE_FIXTURE = REPO_ROOT / "data" / "fixtures" / "gene_score_small.npz"
GENE_SCORE_COOL = REPO_ROOT / "data" / "fixtures" / "gene_score_small.cool"
GENE_SCORE_CONTACTS = REPO_ROOT / "data" / "fixtures" / "gene_score_small.contact.tsv.gz"
CONTACT_DISTANCE_FIXTURE = REPO_ROOT / "data" / "fixtures" / "contact_distance_small.npz"
CONTACT_DISTANCE_TSV = REPO_ROOT / "data" / "fixtures" / "contact_distance_small.tsv.gz"


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


def ref_insulation_score(domain_pack):
    from scipy.sparse import csc_matrix
    m = domain_pack["topdom.matrix"]
    csc = csc_matrix(m.astype(np.float32))
    w = int(domain_pack["insulation.window_size"])
    score = upstream_insulation(csc, window_size=w, save_count=False)
    return np.asarray(score, dtype=np.float32).tolist()


def ref_compartment(comp_pack):
    from scipy.sparse import csr_matrix
    import pandas as pd
    m = csr_matrix(comp_pack["compartment.matrix"].astype(np.float32))
    cpg_series = pd.Series(comp_pack["compartment.cpg_ratio"].astype(np.float32))
    comp, strength = upstream_compartment(m, cpg_series, calc_strength=True)
    return (
        np.asarray(comp, dtype=np.float64).tolist(),
        np.asarray(strength, dtype=np.float64).tolist(),
    )


def ref_embedding_features(emb_pack):
    cells = emb_pack["embedding.cells"]
    n_cells = cells.shape[0]
    n_bins = int(emb_pack["embedding.n_bins"])
    dist = int(emb_pack["embedding.dist"])
    resolution = int(emb_pack["embedding.resolution"])
    scale = float(emb_pack["embedding.scale_factor"])
    idx = upstream_make_idx(n_bins, dist, resolution)
    out = np.zeros((n_cells, idx[0].size), dtype=np.float32)
    for i in range(n_cells):
        out[i, :] = cells[i][idx].ravel()
    out *= scale
    return out.tolist()


def ref_topdom_bed(domain_pack):
    from scipy.sparse import csc_matrix
    m = domain_pack["topdom.matrix"]
    w = int(domain_pack["topdom.window_size"])
    n = m.shape[0]
    csc = csc_matrix(m.astype(np.float32))
    bins = pd.DataFrame({
        "chr": [DOMAIN_CHROM] * n,
        "from.coord": [i * DOMAIN_BIN_RESOLUTION for i in range(n)],
        "to.coord": [(i + 1) * DOMAIN_BIN_RESOLUTION for i in range(n)],
    })
    result = _rpy2_r.RunTopDom(csc.indices + 1, csc.indptr, csc.data, bins, w)
    df = pd.DataFrame(result)
    if df.shape[0] == 0 or df.shape[1] == 0:
        return []
    if df.shape[1] != 4 and df.shape[0] == 4:
        df = df.T
    df.columns = ["chrom", "chromStart", "chromEnd", "name"]
    return df.to_dict(orient="records")


def _gene_meta_from_pack(pack):
    """Rebuild the gene_meta frame the orchestrator hands its workers.

    Bins are already floor-divided by resolution in the fixture, matching what
    gene_score() does before submitting.
    """
    chrom = str(pack["gene_score.chrom"][0])
    ids = [str(g) for g in pack["gene_score.gene_id"]]
    return pd.DataFrame(
        {0: [chrom] * len(ids),
         1: pack["gene_score.gene_start_bin"],
         2: pack["gene_score.gene_end_bin"]},
        index=ids,
    )


def _chrom_sizes_from_pack(pack):
    return pd.Series(
        pack["gene_score.chrom_size"],
        index=[str(c) for c in pack["gene_score.chrom"]],
    )


def ref_gene_score_impute(pack):
    return [float(v) for v in upstream_gene_score_impute(
        cell_path=str(GENE_SCORE_COOL),
        chrom_sizes=_chrom_sizes_from_pack(pack),
        gene_meta=_gene_meta_from_pack(pack),
    )]


def ref_gene_score_raw(pack):
    return [int(v) for v in upstream_gene_score_raw(
        cell_path=str(GENE_SCORE_CONTACTS),
        chrom_sizes=_chrom_sizes_from_pack(pack),
        gene_meta=_gene_meta_from_pack(pack),
        resolution=int(pack["gene_score.resolution"]),
        chrom1=0, pos1=1, chrom2=2, pos2=3,
    )]


def ref_contact_distance(pack):
    """Returns (decay_counts, sparsity_counts) with sparsity sorted by chrom name."""
    chroms = [str(c) for c in pack["contact_distance.chroms"]]
    chrom_sizes = pd.DataFrame(pack["contact_distance.chrom_sizes"], index=chroms)
    c1, p1, c2, p2 = [int(x) for x in pack["contact_distance.cols"]]
    sparsity_df, decay_df = upstream_compute_decay(
        cell_name="fixture_cell",
        contact_path=str(CONTACT_DISTANCE_TSV),
        bins=pack["contact_distance.bin_edges"],
        chrom_sizes=chrom_sizes,
        resolution=int(pack["contact_distance.resolution"]),
        chrom1=c1, pos1=p1, chrom2=c2, pos2=p2,
    )
    decay = [int(v) for v in decay_df["fixture_cell"].values]
    series = sparsity_df["fixture_cell"]
    sparsity = [int(series.loc[k]) for k in sorted(str(x) for x in series.index)]
    return decay, sparsity


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

        domain_pack = _load_npz(DOMAIN_FIXTURE)
        payload["insulation"] = {"score": ref_insulation_score(domain_pack)}
        payload["topdom"] = {"bed": ref_topdom_bed(domain_pack)}

        comp_pack = _load_npz(COMPARTMENT_FIXTURE)
        comp_vals, strength_vals = ref_compartment(comp_pack)
        payload["compartment"] = {"comp": comp_vals, "strength": strength_vals}

        emb_pack = _load_npz(EMBEDDING_FIXTURE)
        payload["embedding"] = {"cell_by_feature": ref_embedding_features(emb_pack)}

        gs_pack = _load_npz(GENE_SCORE_FIXTURE)
        payload["gene_score"] = {
            "impute": ref_gene_score_impute(gs_pack),
            "raw": ref_gene_score_raw(gs_pack),
        }

        cd_pack = _load_npz(CONTACT_DISTANCE_FIXTURE)
        cd_decay, cd_sparsity = ref_contact_distance(cd_pack)
        payload["contact_distance"] = {"decay": cd_decay, "sparsity": cd_sparsity}
    finally:
        shutil.rmtree(str(bkg_dir), ignore_errors=True)
        shutil.rmtree(str(merge_dir), ignore_errors=True)

    OUT.write_text(json.dumps(payload))
    print("wrote {} (top-level keys: {})".format(OUT, sorted(payload.keys())))


if __name__ == "__main__":
    main()
