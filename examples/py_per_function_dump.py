"""Per-function upstream reference dump for Notebook 3.

Runs under $PYTHON_REF_ENV (conda env `schicluster`, **Python 3.6**), where
upstream scHiCluster and rpy2/R live. schicluster_rs is abi3-py39 and cannot be
imported there, so the notebook compares against this JSON instead.

py3.6 NOTE: no f-strings, no `from __future__ import annotations`, no PEP 585.

Usage (from repo root):
    conda run -n schicluster python examples/py_per_function_dump.py
"""
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))
os.chdir(str(REPO_ROOT))

import numpy as np

# Reuse the driver's fixture loading and reference calls rather than
# duplicating them — this file exists to slice the same work per function.
import py_reference_driver as drv

OUT = REPO_ROOT / "data" / "fixtures" / "per_function_reference.json"


def main():
    payload = {}

    conv_pack = drv._load_npz(drv.CONV_FIXTURE)
    payload["convolve2d_mirror"] = {
        "convolved": drv.ref_conv_convolved(conv_pack),
        "note": "scipy.ndimage.convolve(a, kernel, mode='mirror')",
    }

    domain_pack = drv._load_npz(drv.DOMAIN_FIXTURE)
    payload["insulation_score_chrom"] = {
        "score": drv.ref_insulation_score(domain_pack),
        "note": "single_chrom_calculate_insulation_score",
    }
    payload["topdom_chrom"] = {
        "bed": drv.ref_topdom_bed(domain_pack),
        "note": "TopDom.R::RunTopDom via rpy2",
    }

    comp_pack = drv._load_npz(drv.COMPARTMENT_FIXTURE)
    comp_vals, strength_vals = drv.ref_compartment(comp_pack)
    payload["single_chrom_compartment"] = {
        "comp": comp_vals,
        "strength": strength_vals,
        "note": "single_chrom_compartment(..., calc_strength=True)",
    }

    emb_pack = drv._load_npz(drv.EMBEDDING_FIXTURE)
    payload["make_chrom_matrix"] = {
        "cell_by_feature": drv.ref_embedding_features(emb_pack),
        "note": "make_chrom_matrix extraction, pre-SVD",
    }

    gs_pack = drv._load_npz(drv.GENE_SCORE_FIXTURE)
    payload["gene_score_impute"] = {
        "scores": drv.ref_gene_score_impute(gs_pack),
        "gene_ids": [str(g) for g in gs_pack["gene_score.gene_id"]],
        "note": "gene_score_impute on gene_score_small.cool",
    }
    payload["gene_score_raw"] = {
        "scores": drv.ref_gene_score_raw(gs_pack),
        "gene_ids": [str(g) for g in gs_pack["gene_score.gene_id"]],
        "note": "gene_score_raw on gene_score_small.contact.tsv.gz",
    }

    cd_pack = drv._load_npz(drv.CONTACT_DISTANCE_FIXTURE)
    decay, sparsity = drv.ref_contact_distance(cd_pack)
    payload["compute_decay"] = {
        "decay": decay,
        "sparsity": sparsity,
        "chroms": sorted(str(c) for c in cd_pack["contact_distance.chroms"]),
        "note": "compute_decay on contact_distance_small.tsv.gz",
    }

    # loop module: reuse the driver's own reference calls
    loop_pack = drv._load_npz(drv.LOOP_FIXTURE)
    payload["loop_background"] = {
        "kernels": drv.ref_scan_kernels(loop_pack),
        "note": "loop_background -> (bl, donut, h, v)",
    }
    payload["find_summit"] = {
        "result": drv.ref_find_summit(loop_pack),
        "note": "find_summit -> (idx, sizes)",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(payload, fh)
    print("wrote {} ({} functions: {})".format(OUT, len(payload), sorted(payload)))


if __name__ == "__main__":
    main()
