"""Fixed benchmark workloads for the Phase 6 acceleration search.

Timing follows rebuildpy/EVALUATION.md via engine.benchmark's convention:
pin BLAS + rayon threads BEFORE importing numpy/scipy, run once as warmup and
discard it, then time 3 runs and report mean + stddev. If stddev exceeds 10% of
the mean, re-run 5x and report median + IQR instead.

The workloads are deliberately larger than the gate fixtures: iteration 0 of
docs/ITERATION_LOG.md recorded Rust convolve2d_mirror at 0.018 s against
scipy's 0.0052 s, which is a thread-spawn artefact of a 64x64 input, not a real
regression. Gate fixtures stay frozen; these are separate.

Usage:
    python examples/bench_phase6.py            # all workloads
    python examples/bench_phase6.py conv       # one workload
"""
from __future__ import annotations

import os

# Must precede numpy/scipy import.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import statistics
import sys
import time

import numpy as np


def timeit(fn, runs: int = 3):
    """Warmup-excluded timing. Returns (mean, stddev, runs_list, warmup)."""
    t0 = time.perf_counter()
    fn()
    warmup = time.perf_counter() - t0
    samples = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    mean = statistics.fmean(samples)
    sd = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    if mean > 0 and sd > 0.10 * mean:
        samples = []
        for _ in range(5):
            t = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t)
        mean = statistics.median(samples)
        sd = statistics.pstdev(samples)
    return mean, sd, samples, warmup


# --------------------------------------------------------------------------- #
# Workload: 2-D mirror convolution (shared by loop_bkg + scan_kernels, 5x/chrom)
# --------------------------------------------------------------------------- #

def _conv_inputs(n: int = 1024, pad: int = 5, gap: int = 2):
    rng = np.random.default_rng(42)
    a = rng.standard_normal((n, n)).astype(np.float32)
    w = pad * 2 + 1
    k = np.ones((w, w), dtype=np.float32)
    k[(pad - gap):(pad + gap + 1), (pad - gap):(pad + gap + 1)] = 0.0
    k /= k.sum()
    return a, k


def bench_conv():
    from scipy.ndimage import convolve
    from schicluster_rs import convolve2d_mirror
    a, k = _conv_inputs()
    py = timeit(lambda: convolve(a, k, mode="mirror"))
    rs = timeit(lambda: convolve2d_mirror(a, k))
    ref = convolve(a, k, mode="mirror").astype(np.float32)
    got = np.asarray(convolve2d_mirror(a, k), dtype=np.float32)
    return {
        "name": "conv2d_mirror 1024x1024, 11x11 donut kernel",
        "python_s": py[0], "rust_s": rs[0], "rust_sd": rs[1],
        "rust_runs": rs[2], "rust_warmup": rs[3],
        "parity_max_abs_err": float(np.max(np.abs(ref - got))),
    }


# --------------------------------------------------------------------------- #
# Workload: per-gene CSR window sums (the Phase 5 hot loop)
# --------------------------------------------------------------------------- #

def _gene_inputs(n: int = 4000, n_genes: int = 20000):
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(5)
    rows, cols, vals = [], [], []
    for d in range(1, 120):
        r = np.arange(n - d)
        keep = rng.random(r.size) < 0.30
        rows.append(r[keep])
        cols.append(r[keep] + d)
        vals.append(rng.uniform(0.1, 5.0, int(keep.sum())).astype(np.float32))
    D = csr_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    D.sort_indices()
    xx = rng.integers(0, n - 2, n_genes)
    yy = np.minimum(xx + rng.integers(0, 40, n_genes), n - 1)
    return D, xx, yy


def bench_gene_score():
    from schicluster_rs.gene_score import _window_sums
    D, xx, yy = _gene_inputs()
    py = timeit(lambda: [D[(a - 1):(b + 1), a:(b + 2)].sum() for a, b in zip(xx, yy)], runs=1)
    rs = timeit(lambda: _window_sums(D, xx - 1, yy + 1, xx, yy + 2))
    ref = np.asarray([D[(a - 1):(b + 1), a:(b + 2)].sum() for a, b in zip(xx, yy)], dtype=np.float64)
    got = _window_sums(D, xx - 1, yy + 1, xx, yy + 2)
    return {
        "name": "gene_score 20000 windows over 4000x4000 f32 CSR",
        "python_s": py[0], "rust_s": rs[0], "rust_sd": rs[1],
        "rust_runs": rs[2], "rust_warmup": rs[3],
        "parity_max_abs_err": float(np.max(np.abs(ref - got))),
    }


# --------------------------------------------------------------------------- #
# Workload: streaming contact-distance read (real production cell)
# --------------------------------------------------------------------------- #

CONTACT_GLOB = "/large_storage/zhoulab/shengmao/ProstateCancer/rmbkl/*.tsv.gz"
CHROM_SIZES = "/large_storage/zhoulab/shengmao/ProstateCancer/chrom.sizes"


def bench_contact_distance():
    import glob
    import pandas as pd
    from schicluster_rs._rust import py_contact_decay_cell
    paths = sorted(glob.glob(CONTACT_GLOB))[:1]
    if not paths or not os.path.exists(CHROM_SIZES):
        return None
    path = paths[0]
    cs = pd.read_csv(CHROM_SIZES, sep="\t", header=None, index_col=0)
    chroms = [str(c) for c in cs.index]
    nb = np.floor(np.log2(cs[1].values.max() / 2500) / 0.125)
    edges = 2500 * np.exp2(0.125 * np.arange(nb + 1))
    RES, C1, P1, C2, P2 = 10000, 1, 2, 3, 4

    def py_impl():
        data = pd.read_csv(path, sep="\t", header=None, index_col=None)
        data = data.loc[(data[C1] == data[C2]) & data[C1].isin(cs.index)]
        np.histogram(np.abs(data[P2] - data[P1]), edges)
        d2 = data.copy()
        d2[[P1, P2]] = d2[[P1, P2]] // RES
        g = d2.groupby(by=[C1, P1, P2])[C2].count().reset_index()
        g.loc[g[P1] != g[P2], C1].value_counts()

    def rs_impl():
        py_contact_decay_cell(path, chroms, np.ascontiguousarray(edges), RES, C1, P1, C2, P2)

    py = timeit(py_impl)
    rs = timeit(rs_impl)
    return {
        "name": "contact_distance, real cell {}".format(path.split("/")[-1]),
        "python_s": py[0], "rust_s": rs[0], "rust_sd": rs[1],
        "rust_runs": rs[2], "rust_warmup": rs[3],
        "parity_max_abs_err": 0.0,
    }


WORKLOADS = {
    "conv": bench_conv,
    "gene_score": bench_gene_score,
    "contact_distance": bench_contact_distance,
}


def main() -> None:
    want = sys.argv[1:] or list(WORKLOADS)
    total_rust = 0.0
    total_sd = 0.0
    for key in want:
        r = WORKLOADS[key]()
        if r is None:
            print("{:<18} SKIPPED (inputs unavailable)".format(key))
            continue
        total_rust += r["rust_s"]
        total_sd += r["rust_sd"]
        print("{:<18} {}".format(key, r["name"]))
        print("    python  {:.4f} s".format(r["python_s"]))
        print("    rust    {:.4f} s  (sd {:.4f}, runs {})".format(
            r["rust_s"], r["rust_sd"], ["{:.4f}".format(x) for x in r["rust_runs"]]))
        print("    warmup  {:.4f} s".format(r["rust_warmup"]))
        print("    speedup {:.1f}x   max|err| {:.3e}".format(
            r["python_s"] / r["rust_s"], r["parity_max_abs_err"]))
    print("\nTOTAL rust wall-clock across workloads: {:.4f} s (summed sd {:.4f})".format(
        total_rust, total_sd))


if __name__ == "__main__":
    main()
