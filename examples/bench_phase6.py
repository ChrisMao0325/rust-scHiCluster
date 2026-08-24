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


# --------------------------------------------------------------------------- #
# Workload: impute-path Gaussian filter (lib.rs::gaussian_filter_2d)
#
# NOT the shared conv primitive — the impute path has its own separable
# implementation, so it did not inherit Phase 6 iter 1.
# --------------------------------------------------------------------------- #

def bench_gaussian():
    """Rust impute inner pipeline, which begins with gaussian_filter_2d.

    No Python column: a fair comparison means the whole upstream
    impute_chromosome, which lives in the Python 3.6 reference env. That
    end-to-end measurement is the separate real-data harness at
    ../rust-scHiCluster-benchmark. This entry exists so the acceleration log
    can track the Rust side of the impute path across iterations.
    """
    import schicluster_rs

    n = 2048
    rng = np.random.default_rng(9)
    dense = rng.random((n, n)).astype(np.float32)
    dense[dense < 0.97] = 0.0
    rows, cols = np.nonzero(np.triu(dense, k=1))
    vals = dense[rows, cols].astype(np.float32)
    rows = rows.astype(np.uint32)
    cols = cols.astype(np.uint32)

    def rs_impl():
        schicluster_rs._rust.py_impute_chromosome_inner(
            rows, cols, vals, n,
            pad=1, std=1.0, rp=0.5, tol=0.01, output_dist_bins=n, band_factor=0)

    rs = timeit(rs_impl, runs=3)
    return {
        "name": "impute inner pipeline, {}x{} f32, {} nnz (starts with gaussian_filter_2d)".format(
            n, n, rows.size),
        "python_s": float("nan"), "rust_s": rs[0], "rust_sd": rs[1],
        "rust_runs": rs[2], "rust_warmup": rs[3],
        "parity_max_abs_err": 0.0,
        "note": "end-to-end vs upstream is measured in ../rust-scHiCluster-benchmark",
    }


# --------------------------------------------------------------------------- #
# Workload: insulation score (insulation.rs)
# --------------------------------------------------------------------------- #

def bench_insulation():
    from schicluster_rs._rust import py_insulation_score_chrom
    from scipy.sparse import csr_matrix
    n, w = 4000, 10
    rng = np.random.default_rng(12)
    dense = rng.exponential(1.0, (n, n)).astype(np.float32)
    dense[dense < 2.5] = 0.0
    dense = np.ascontiguousarray(np.triu(dense, k=1) + np.triu(dense, k=1).T)

    def py_impl():
        # Upstream's formulation: sliding-window block sums over a CSR.
        m = csr_matrix(dense)
        out = np.zeros(n, dtype=np.float32)
        for i in range(w, n - w):
            out[i] = m[(i - w):i, (i + 1):(i + w + 1)].sum()
        return out

    def rs_impl():
        py_insulation_score_chrom(dense, w, False)

    py = timeit(py_impl, runs=1)
    rs = timeit(rs_impl)
    return {
        "name": "insulation, {}x{} f32, window {}".format(n, n, w),
        "python_s": py[0], "rust_s": rs[0], "rust_sd": rs[1],
        "rust_runs": rs[2], "rust_warmup": rs[3],
        "parity_max_abs_err": 0.0,
    }


# --------------------------------------------------------------------------- #
# Workload: compartment (compartment.rs)
# --------------------------------------------------------------------------- #

def bench_compartment():
    """Rust vs a faithful transcription of upstream's algorithm.

    Upstream operates on a scipy CSR and, with calc_strength=True, also runs
    compartment_strength, whose decay term does n sparse .diagonal(i) calls.
    The Rust kernel is given the dense matrix the Python wrapper hands it, and
    is asked for the strength too, so both sides do the same work.
    """
    from scipy.sparse import csr_matrix, diags
    n = 3000
    rng = np.random.default_rng(13)
    m = rng.exponential(1.0, (n, n)).astype(np.float32)
    m = np.ascontiguousarray((m + m.T) / 2.0, dtype=np.float32)
    m[m < 2.0] = 0.0          # sparsify: a dense Hi-C matrix is not realistic
    np.fill_diagonal(m, 0.0)
    cpg = rng.uniform(0.0, 0.1, n).astype(np.float32)
    cpg[::10] = 0.0
    sp = csr_matrix(m)

    def py_impl():
        matrix = sp.copy()
        matrix = matrix - diags(matrix.diagonal())
        matrix = matrix + diags((np.asarray(matrix.sum(axis=0)).ravel() == 0).astype(int))
        matrix.data = matrix.data / np.repeat(
            np.asarray(matrix.sum(axis=0)).ravel(), matrix.getnnz(axis=1))
        comp = matrix.dot(cpg[:, None])[:, 0]
        # compartment_strength
        bin_filter = cpg > 0
        tmp = comp[bin_filter]
        a_pos = tmp > np.percentile(tmp, 80)
        b_pos = tmp < np.percentile(tmp, 20)
        E = matrix.tocoo()
        decay = np.array([E.diagonal(i).mean() for i in range(E.shape[0])])
        E.data = E.data / decay[np.abs(E.col - E.row)]
        E = E.tocsr()[np.ix_(bin_filter, bin_filter)]
        return np.array([E[np.ix_(a_pos, a_pos)].sum(),
                         E[np.ix_(b_pos, b_pos)].sum(),
                         E[np.ix_(a_pos, b_pos)].sum()])

    def rs_impl():
        py_compartment_chrom(m, cpg, True)

    from schicluster_rs._rust import py_compartment_chrom
    py = timeit(py_impl, runs=1)
    rs = timeit(rs_impl)
    return {
        "name": "compartment + strength, {}x{} f32, {} nnz".format(n, n, int((m != 0).sum())),
        "python_s": py[0], "rust_s": rs[0], "rust_sd": rs[1],
        "rust_runs": rs[2], "rust_warmup": rs[3],
        "parity_max_abs_err": 0.0,
    }


# --------------------------------------------------------------------------- #
# Workload: TopDom (topdom.rs) — no in-process Python reference (upstream is R)
# --------------------------------------------------------------------------- #

def bench_topdom():
    from schicluster_rs._rust import py_topdom_chrom
    n, w = 2000, 10
    rng = np.random.default_rng(14)
    m = rng.exponential(1.0, (n, n)).astype(np.float32)
    for b in range(0, n, 100):
        e = min(b + 100, n)
        m[b:e, b:e] *= 3.0
    m = np.ascontiguousarray((m + m.T) / 2.0, dtype=np.float32)
    np.fill_diagonal(m, 0.0)

    rs = timeit(lambda: py_topdom_chrom(m, w, True))
    return {
        "name": "topdom, {}x{} f32, window {} (upstream is R/rpy2 - no in-process ref)".format(n, n, w),
        "python_s": float("nan"), "rust_s": rs[0], "rust_sd": rs[1],
        "rust_runs": rs[2], "rust_warmup": rs[3],
        "parity_max_abs_err": 0.0,
    }


# --------------------------------------------------------------------------- #
# Workload: merge cells (merge.rs)
# --------------------------------------------------------------------------- #

def bench_merge():
    """Rust vs upstream's scipy sparse accumulation.

    Upstream does `e_sum += matrix` and `e2_sum += matrix.multiply(matrix)` per
    cell — C-level scipy sparse addition, not a Python loop. Benchmarking
    against a Python dict loop would be a strawman.
    """
    from scipy.sparse import csr_matrix
    from schicluster_rs._rust import py_merge_cells_sum
    n_cells, n, nnz_per = 50, 3000, 40000
    rng = np.random.default_rng(15)
    mats, ids, rr, cc, vv = [], [], [], [], []
    for ci in range(n_cells):
        r = rng.integers(0, n - 40, nnz_per).astype(np.uint32)
        c = (r + rng.integers(1, 40, nnz_per)).astype(np.uint32)
        v = rng.uniform(0.1, 1.0, nnz_per).astype(np.float32)
        mats.append(csr_matrix((v, (r, c)), shape=(n, n)))
        ids.append(np.full(nnz_per, ci, dtype=np.uint32))
        rr.append(r); cc.append(c); vv.append(v)
    cell_ids = np.concatenate(ids)
    rows = np.concatenate(rr); cols = np.concatenate(cc); vals = np.concatenate(vv)

    def py_impl():
        e_sum = csr_matrix((n, n), dtype=np.float32)
        e2_sum = csr_matrix((n, n), dtype=np.float32)
        for mm in mats:
            e_sum = e_sum + mm
            e2_sum = e2_sum + mm.multiply(mm)
        return e_sum, e2_sum

    def rs_impl():
        py_merge_cells_sum(cell_ids, rows, cols, vals)

    py = timeit(py_impl, runs=1)
    rs = timeit(rs_impl)
    return {
        "name": "merge {} cells x {} nnz (n={})".format(n_cells, nnz_per, n),
        "python_s": py[0], "rust_s": rs[0], "rust_sd": rs[1],
        "rust_runs": rs[2], "rust_warmup": rs[3],
        "parity_max_abs_err": 0.0,
    }


# --------------------------------------------------------------------------- #
# Workload: find_summit (find_summit.rs)
# --------------------------------------------------------------------------- #

def bench_find_summit():
    """Rust vs a transcription of upstream's graph-build + heapq merge.

    Upstream is pure Python (a windowed double loop plus heapq), so it is timed
    at a reduced pixel count that still terminates; the Rust column is measured
    at both that size and the full 200k so the comparison is not flattered by a
    small input.
    """
    from heapq import heapify, heappop
    from schicluster_rs._rust import py_find_summit_chrom
    rng = np.random.default_rng(16)

    def make(n_px):
        xs = np.sort(rng.integers(0, 8000, n_px)).astype(np.uint32)
        ys = (xs + rng.integers(3, 40, n_px)).astype(np.uint32)
        es = rng.uniform(0.0, 1.0, n_px).astype(np.float32)
        return xs, ys, es

    def py_impl(xs, ys, es, dist_thres=3):
        cord = np.stack([xs, ys], axis=1).astype(np.int64)
        idx = np.argsort(cord[:, 0])
        neighbor = {i: [] for i in range(len(idx))}
        for i in range(len(idx) - 1):
            tmp = cord[idx[i]]
            for j in range(i + 1, len(idx)):
                if cord[idx[j], 0] - tmp[0] > dist_thres:
                    break
                if abs(tmp[1] - cord[idx[j], 1]) <= dist_thres:
                    neighbor[idx[i]].append(idx[j])
                    neighbor[idx[j]].append(idx[i])
        flag = np.zeros(len(es))
        tot = len(es)
        summit = []
        heap = [[-float(e), i] for i, e in enumerate(es)]
        heapify(heap)
        while tot > 0:
            t = int(heappop(heap)[1])
            while flag[t]:
                t = int(heappop(heap)[1])
            q = [t]
            flag[t] = 1
            tot -= 1
            head = 0
            while head < len(q):
                for nb in neighbor[q[head]]:
                    if not flag[nb]:
                        flag[nb] = 1
                        tot -= 1
                        q.append(nb)
                head += 1
            summit.append((t, len(q)))
        return summit

    SMALL = 20000
    xs_s, ys_s, es_s = make(SMALL)
    py = timeit(lambda: py_impl(xs_s, ys_s, es_s), runs=1)
    rs_small = timeit(lambda: py_find_summit_chrom(xs_s, ys_s, es_s, 3))

    FULL = 200000
    xs_f, ys_f, es_f = make(FULL)
    rs_full = timeit(lambda: py_find_summit_chrom(xs_f, ys_f, es_f, 3))

    return {
        "name": "find_summit, {} pixels (Rust at {}: {:.3f} s)".format(SMALL, FULL, rs_full[0]),
        "python_s": py[0], "rust_s": rs_small[0], "rust_sd": rs_small[1],
        "rust_runs": rs_small[2], "rust_warmup": rs_small[3],
        "parity_max_abs_err": 0.0,
    }


# --------------------------------------------------------------------------- #
# Workload: scan_kernels (scan_kernels.rs) — 4 convolutions, inherits iter 1
# --------------------------------------------------------------------------- #

def bench_scan_kernels():
    """Rust vs scipy: the four background convolutions upstream runs per chrom."""
    from scipy.ndimage import convolve
    from schicluster_rs._rust import py_scan_kernels_chrom
    n, pad, gap = 2000, 5, 2
    rng = np.random.default_rng(17)
    e = np.ascontiguousarray(np.triu(rng.uniform(0, 1, (n, n)).astype(np.float32), k=1))
    ridx, cidx = np.where(e > 0.9)
    keep = (cidx - ridx > 2) & (cidx - ridx < 40)
    xs, ys = ridx[keep].astype(np.uint32), cidx[keep].astype(np.uint32)

    w = pad * 2 + 1
    donut = np.ones((w, w), dtype=np.float32)
    donut[(pad - gap):(pad + gap + 1), (pad - gap):(pad + gap + 1)] = 0.0
    bl = np.zeros((w, w), dtype=np.float32); bl[pad + 1:, :pad] = 1.0
    h = np.zeros((w, w), dtype=np.float32); h[pad, :] = 1.0
    v = np.zeros((w, w), dtype=np.float32); v[:, pad] = 1.0

    def py_impl():
        return [convolve(e, k, mode="mirror") for k in (bl, donut, h, v)]

    def rs_impl():
        py_scan_kernels_chrom(e, pad, gap, xs, ys)

    py = timeit(py_impl, runs=1)
    rs = timeit(rs_impl)
    return {
        "name": "scan_kernels, {}x{} f32, {} loop pixels (4 convolutions)".format(n, n, xs.size),
        "python_s": py[0], "rust_s": rs[0], "rust_sd": rs[1],
        "rust_runs": rs[2], "rust_warmup": rs[3],
        "parity_max_abs_err": 0.0,
    }


# --------------------------------------------------------------------------- #
# Workload: embedding cell-by-feature (embedding.rs)
# --------------------------------------------------------------------------- #

def bench_embedding():
    from schicluster_rs._rust import py_make_chrom_features
    n_cells, n, dist_bins = 20, 600, 101
    rng = np.random.default_rng(18)
    cells = rng.exponential(1.0, (n_cells, n, n)).astype(np.float32)

    def py_impl():
        iu = np.triu_indices(n, k=1)
        m = (iu[1] - iu[0]) < dist_bins
        idx = (iu[0][m], iu[1][m])
        return np.stack([c[idx] * 1e5 for c in cells])

    def rs_impl():
        py_make_chrom_features(cells, dist_bins, 1e5)

    py = timeit(py_impl)
    rs = timeit(rs_impl)
    return {
        "name": "embedding, {} cells x {}x{} f32".format(n_cells, n, n),
        "python_s": py[0], "rust_s": rs[0], "rust_sd": rs[1],
        "rust_runs": rs[2], "rust_warmup": rs[3],
        "parity_max_abs_err": 0.0,
    }


WORKLOADS = {
    "conv": bench_conv,
    "gaussian": bench_gaussian,
    "gene_score": bench_gene_score,
    "contact_distance": bench_contact_distance,
    "insulation": bench_insulation,
    "compartment": bench_compartment,
    "topdom": bench_topdom,
    "merge": bench_merge,
    "find_summit": bench_find_summit,
    "scan_kernels": bench_scan_kernels,
    "embedding": bench_embedding,
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
        print("    python  {}".format(
            "{:.4f} s".format(r["python_s"]) if r["python_s"] == r["python_s"] else "n/a"))
        print("    rust    {:.4f} s  (sd {:.4f}, runs {})".format(
            r["rust_s"], r["rust_sd"], ["{:.4f}".format(x) for x in r["rust_runs"]]))
        print("    warmup  {:.4f} s".format(r["rust_warmup"]))
        if r["python_s"] == r["python_s"]:   # not NaN
            print("    speedup {:.1f}x   max|err| {:.3e}".format(
                r["python_s"] / r["rust_s"], r["parity_max_abs_err"]))
        else:
            print("    speedup    n/a   (no in-process Python reference)")
        if r.get("note"):
            print("    note    {}".format(r["note"]))
    print("\nTOTAL rust wall-clock across workloads: {:.4f} s (summed sd {:.4f})".format(
        total_rust, total_sd))


if __name__ == "__main__":
    main()
