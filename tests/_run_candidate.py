"""Run schicluster_rs Rust candidate and dump JSON.

Invoked under $RUST_TEST_ENV (rebuild-rust env, py 3.10) after
`maturin develop --release`. Output keys mirror py_reference_driver.py.

Usage (from repo root, after the Rust build):
    python tests/_run_candidate.py
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

import schicluster_rs


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONV_FIXTURE = REPO_ROOT / "data" / "fixtures" / "conv_small.npz"
LOOP_FIXTURE = REPO_ROOT / "data" / "fixtures" / "loop_small.npz"
LOOP_COOL = REPO_ROOT / "data" / "fixtures" / "loop_small.cool"
OUT = REPO_ROOT / "data" / "fixtures" / "candidate_output.json"

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


def _load_npz(path: pathlib.Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def cand_conv_convolved(conv: dict) -> list:
    a = np.ascontiguousarray(conv["input"], dtype=np.float32)
    k = np.ascontiguousarray(conv["kernel"], dtype=np.float32)
    return np.asarray(schicluster_rs.convolve2d_mirror(a, k), dtype=np.float32).tolist()


def cand_loop_bkg(temp_dir: pathlib.Path):
    out_prefix = str(temp_dir / "loop_small")
    schicluster_rs.loop_bkg_chrom(
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


def cand_merge(loop_pack: dict, temp_dir: pathlib.Path):
    """Call the Rust merge kernel directly, avoiding HDF5 I/O (pytables not
    available in rebuild-rust env).
    """
    from schicluster_rs._rust import py_merge_cells_sum as _rust_merge

    cell_ids = np.ascontiguousarray(loop_pack["merge.cell_ids"], dtype=np.uint32)
    rows = np.ascontiguousarray(loop_pack["merge.input.rows"], dtype=np.uint32)
    cols = np.ascontiguousarray(loop_pack["merge.input.cols"], dtype=np.uint32)
    vals = np.ascontiguousarray(loop_pack["merge.input.vals"], dtype=np.float32)

    (er, ec, ev), (e2r, e2c, e2v) = _rust_merge(cell_ids, rows, cols, vals)
    return (
        {
            "rows": np.asarray(er, dtype=np.uint32).tolist(),
            "cols": np.asarray(ec, dtype=np.uint32).tolist(),
            "vals": np.asarray(ev, dtype=np.float32).tolist(),
        },
        {
            "rows": np.asarray(e2r, dtype=np.uint32).tolist(),
            "cols": np.asarray(e2c, dtype=np.uint32).tolist(),
            "vals": np.asarray(e2v, dtype=np.float32).tolist(),
        },
    )


def cand_scan_kernels(loop_pack: dict) -> dict:
    e = np.ascontiguousarray(loop_pack["scan.E_dense"], dtype=np.float32)
    xs = np.ascontiguousarray(loop_pack["scan.loop_xs"], dtype=np.uint32)
    ys = np.ascontiguousarray(loop_pack["scan.loop_ys"], dtype=np.uint32)
    bl, donut, h, v = schicluster_rs.loop_background(e, LOOP_PAD, LOOP_GAP, (xs, ys))
    return {
        "bl": np.asarray(bl, dtype=np.float32).tolist(),
        "donut": np.asarray(donut, dtype=np.float32).tolist(),
        "h": np.asarray(h, dtype=np.float32).tolist(),
        "v": np.asarray(v, dtype=np.float32).tolist(),
    }


def cand_find_summit(loop_pack: dict) -> dict:
    df = pd.DataFrame({
        "x1": loop_pack["summit.x1"],
        "y1": loop_pack["summit.y1"],
        "E": loop_pack["summit.E"],
    })
    summit_df = schicluster_rs.find_summit(df, res=LOOP_RESOLUTION,
                                            dist_thres=LOOP_SUMMIT_DIST_BINS)
    selected = summit_df.index.to_numpy().astype(np.uint32)
    sizes = summit_df["size"].to_numpy().astype(np.uint32)
    return {"idx": selected.tolist(), "sizes": sizes.tolist()}


def cand_insulation_score(domain_pack: dict) -> list:
    m = np.ascontiguousarray(domain_pack["topdom.matrix"], dtype=np.float32)
    w = int(domain_pack["insulation.window_size"])
    score = schicluster_rs.insulation_score_chrom(m, window_size=w, save_count=False)
    return np.asarray(score, dtype=np.float32).tolist()


def cand_compartment(comp_pack: dict):
    m = np.ascontiguousarray(comp_pack["compartment.matrix"], dtype=np.float32)
    cpg = np.ascontiguousarray(comp_pack["compartment.cpg_ratio"], dtype=np.float32)
    comp, strength = schicluster_rs.single_chrom_compartment(m, cpg, calc_strength=True)
    return (
        np.asarray(comp, dtype=np.float64).tolist(),
        np.asarray(strength, dtype=np.float64).tolist(),
    )


def cand_embedding_features(emb_pack: dict) -> list:
    from schicluster_rs._rust import py_make_chrom_features
    cells = np.ascontiguousarray(emb_pack["embedding.cells"], dtype=np.float32)
    dist = int(emb_pack["embedding.dist"])
    resolution = int(emb_pack["embedding.resolution"])
    dist_bins_plus_1 = dist // resolution + 1
    scale = float(emb_pack["embedding.scale_factor"])
    out = py_make_chrom_features(cells, dist_bins_plus_1, scale)
    return np.asarray(out, dtype=np.float32).tolist()


def cand_topdom_bed(domain_pack: dict) -> list:
    from schicluster_rs.domain import _topdom_chrom_to_df
    m = np.ascontiguousarray(domain_pack["topdom.matrix"], dtype=np.float32)
    w = int(domain_pack["topdom.window_size"])
    n = m.shape[0]
    bins = pd.DataFrame({
        "chr": [DOMAIN_CHROM] * n,
        "from.coord": [i * DOMAIN_BIN_RESOLUTION for i in range(n)],
        "to.coord": [(i + 1) * DOMAIN_BIN_RESOLUTION for i in range(n)],
    })
    df = _topdom_chrom_to_df(m, bins, w, stat_filter=True)
    return df.to_dict(orient="records")


def _gene_meta_from_pack(pack):
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


def cand_gene_score_impute(pack):
    from schicluster_rs.gene_score import gene_score_impute
    return [float(v) for v in gene_score_impute(
        cell_path=str(GENE_SCORE_COOL),
        chrom_sizes=_chrom_sizes_from_pack(pack),
        gene_meta=_gene_meta_from_pack(pack),
    )]


def cand_gene_score_raw(pack):
    from schicluster_rs.gene_score import gene_score_raw
    return [int(v) for v in gene_score_raw(
        cell_path=str(GENE_SCORE_CONTACTS),
        chrom_sizes=_chrom_sizes_from_pack(pack),
        gene_meta=_gene_meta_from_pack(pack),
        resolution=int(pack["gene_score.resolution"]),
        chrom1=0, pos1=1, chrom2=2, pos2=3,
    )]


def cand_contact_distance(pack):
    from schicluster_rs.contact_distance import compute_decay
    chroms = [str(c) for c in pack["contact_distance.chroms"]]
    chrom_sizes = pd.DataFrame(pack["contact_distance.chrom_sizes"], index=chroms)
    c1, p1, c2, p2 = [int(x) for x in pack["contact_distance.cols"]]
    sparsity_df, decay_df = compute_decay(
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


def main() -> None:
    conv = _load_npz(CONV_FIXTURE)
    loop_pack = _load_npz(LOOP_FIXTURE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"convolved": cand_conv_convolved(conv)}
    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="schicluster_rs_cand_"))
    try:
        e, t = cand_loop_bkg(temp_dir)
        payload["loop_bkg"] = {"E": e, "T": t}

        e_sum, e2_sum = cand_merge(loop_pack, temp_dir)
        payload["merge"] = {"e_sum": e_sum, "e2_sum": e2_sum}

        payload["scan_kernels"] = cand_scan_kernels(loop_pack)
        payload["find_summit"] = cand_find_summit(loop_pack)

        domain_pack = _load_npz(DOMAIN_FIXTURE)
        payload["insulation"] = {"score": cand_insulation_score(domain_pack)}
        payload["topdom"] = {"bed": cand_topdom_bed(domain_pack)}

        comp_pack = _load_npz(COMPARTMENT_FIXTURE)
        comp_vals, strength_vals = cand_compartment(comp_pack)
        payload["compartment"] = {"comp": comp_vals, "strength": strength_vals}

        emb_pack = _load_npz(EMBEDDING_FIXTURE)
        payload["embedding"] = {"cell_by_feature": cand_embedding_features(emb_pack)}

        gs_pack = _load_npz(GENE_SCORE_FIXTURE)
        payload["gene_score"] = {
            "impute": cand_gene_score_impute(gs_pack),
            "raw": cand_gene_score_raw(gs_pack),
        }

        cd_pack = _load_npz(CONTACT_DISTANCE_FIXTURE)
        cd_decay, cd_sparsity = cand_contact_distance(cd_pack)
        payload["contact_distance"] = {"decay": cd_decay, "sparsity": cd_sparsity}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT} (top-level keys: {sorted(payload.keys())})")


if __name__ == "__main__":
    main()
