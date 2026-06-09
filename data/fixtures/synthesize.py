"""Generate deterministic synthetic fixtures for schicluster-rs parity tests.

Run under any env with numpy + scipy. Output is bit-identical across envs
because every RNG call is seeded and every op is float32.

Usage:
    python data/fixtures/synthesize.py
"""
from __future__ import annotations

import pathlib

import cooler
import numpy as np
import pandas as pd
from scipy.ndimage import convolve


FIXTURE_DIR = pathlib.Path(__file__).resolve().parent


def conv_small_fixture(seed: int = 42, n: int = 64, pad: int = 3, gap: int = 1) -> dict:
    """64x64 f32 input + 7x7 donut-shaped f32 kernel + scipy mirror-convolve reference.

    The kernel matches the shape of the "donut" mask used in loop_bkg.py
    so this fixture exercises the same code path the loop module will hit.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)).astype(np.float32)

    w = pad * 2 + 1
    k = np.ones((w, w), dtype=np.float32)
    k[(pad - gap):(pad + gap + 1), (pad - gap):(pad + gap + 1)] = 0.0
    k = k / k.sum()

    ref = convolve(a, k, mode="mirror").astype(np.float32)
    return {"input": a, "kernel": k, "convolved": ref}


# Phase 1 fixture parameters (constants — read by both reference and candidate drivers).
LOOP_N_BINS = 200
LOOP_RESOLUTION = 10_000
LOOP_DIST = 20 * LOOP_RESOLUTION         # 20 bins ~= 200 kb window
LOOP_PAD = 5
LOOP_GAP = 2
LOOP_CAP = 5.0
LOOP_MIN_CUTOFF = 1e-6
LOOP_N_CELLS = 5
LOOP_DIST_THRES_BP = 30_000              # for find_summit (3 bins)
LOOP_SUMMIT_DIST_BINS = LOOP_DIST_THRES_BP // LOOP_RESOLUTION
LOOP_CHROM = "chr1"


def _upper_tri_synthetic(seed, n, density, max_diag):
    """Random non-negative sparse upper-tri matrix on diagonals 1..max_diag."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for d in range(1, max_diag + 1):
        length = n - d
        nnz = max(1, int(length * density))
        chosen = rng.choice(length, size=nnz, replace=False)
        for r in chosen:
            rows.append(int(r))
            cols.append(int(r + d))
            vals.append(float(rng.uniform(0.1, 1.0)))
    rows = np.asarray(rows, dtype=np.uint32)
    cols = np.asarray(cols, dtype=np.uint32)
    vals = np.asarray(vals, dtype=np.float32)
    order = np.lexsort((cols, rows))
    return rows[order], cols[order], vals[order]


def _bins_df(n_bins, resolution, chrom):
    starts = np.arange(n_bins, dtype=np.int64) * resolution
    return pd.DataFrame({
        "chrom": [chrom] * n_bins,
        "start": starts,
        "end": starts + resolution,
    })


def merge_small_fixture(seed=43):
    """Five per-cell sparse upper-tri matrices for merge_cells_sum.

    Triplets packed flat with a cell-id column so the driver code can split them.
    """
    cells = []
    for i in range(LOOP_N_CELLS):
        r, c, v = _upper_tri_synthetic(seed + i + 1, LOOP_N_BINS, density=0.05, max_diag=20)
        cells.append((np.full(r.shape, i, dtype=np.uint32), r, c, v))
    cell_ids = np.concatenate([x[0] for x in cells])
    rows = np.concatenate([x[1] for x in cells])
    cols = np.concatenate([x[2] for x in cells])
    vals = np.concatenate([x[3] for x in cells])
    return {
        "merge.cell_ids": cell_ids,
        "merge.input.rows": rows,
        "merge.input.cols": cols,
        "merge.input.vals": vals,
    }


def scan_summit_small_fixture(seed=44):
    """Dense E + loop pixel coordinates for scan_kernels + find_summit.

    Loop pixels = positions where E_upper > 0.5 AND (y - x) in (2, 20).
    """
    rng = np.random.default_rng(seed)
    e = rng.uniform(0.0, 1.0, size=(LOOP_N_BINS, LOOP_N_BINS)).astype(np.float32)
    iu = np.triu_indices(LOOP_N_BINS, k=1)
    e_upper = np.zeros((LOOP_N_BINS, LOOP_N_BINS), dtype=np.float32)
    e_upper[iu] = e[iu]
    row_idx, col_idx = np.where(e_upper > 0.5)
    # Upstream loop_calling indexing: loop = np.where(E > 0). For an upper-tri E,
    # np.where returns (row_idx, col_idx); diff = col - row. We mirror that here.
    diff = col_idx - row_idx
    mask = (diff > 2) & (diff < 20)
    xs = row_idx[mask].astype(np.uint32)
    ys = col_idx[mask].astype(np.uint32)
    e_vals_at_loop = e_upper[xs, ys].astype(np.float32)
    return {
        "scan.E_dense": e_upper,
        "scan.loop_xs": xs,
        "scan.loop_ys": ys,
        "summit.x1": (xs.astype(np.int64)) * LOOP_RESOLUTION,
        "summit.y1": (ys.astype(np.int64)) * LOOP_RESOLUTION,
        "summit.E": e_vals_at_loop,
    }


def write_loop_small_cool():
    """Write a tiny synthetic single-cell cooler used by the upstream loop_bkg reference.

    Returns the triplets so the same matrix can be re-emitted to the npz under
    'loop_bkg.input.*' keys for the candidate driver.
    """
    rows, cols, vals = _upper_tri_synthetic(45, LOOP_N_BINS, density=0.15, max_diag=20)
    bins = _bins_df(LOOP_N_BINS, LOOP_RESOLUTION, LOOP_CHROM)
    pixels = pd.DataFrame({
        "bin1_id": rows.astype(np.int64),
        "bin2_id": cols.astype(np.int64),
        "count": vals.astype(np.float32),
    })
    out = FIXTURE_DIR / "loop_small.cool"
    if out.exists():
        out.unlink()
    cooler.create_cooler(
        cool_uri=str(out),
        bins=bins,
        pixels=pixels,
        ordered=True,
        dtypes={"count": np.float32},
    )
    print(f"wrote {out} ({LOOP_N_BINS} bins, {len(pixels)} nnz)")
    return {
        "loop_bkg.input.rows": rows,
        "loop_bkg.input.cols": cols,
        "loop_bkg.input.vals": vals,
    }


def loop_small_packed_fixture():
    """Pack every Phase-1 array into one npz keyed by output names from manifest."""
    bkg = write_loop_small_cool()
    merge = merge_small_fixture()
    scan_summit = scan_summit_small_fixture()
    return {**bkg, **merge, **scan_summit}


# ---- Phase 2 fixture parameters ----
DOMAIN_N_BINS = 80
DOMAIN_BLOCK_SIZE = 20
DOMAIN_WINDOW_SIZE = 5      # insulation + topdom window


def domain_small_fixture(seed: int = 46) -> dict:
    """Synthetic dense Hi-C matrix with 4 planted 20-bin blocks at n=80.

    Both TopDom and insulation_score consume the same matrix. Block-diagonal
    structure gives TopDom enough signal to find domain boundaries; the small
    n keeps the test fast.
    """
    rng = np.random.default_rng(seed)
    n = DOMAIN_N_BINS
    block_size = DOMAIN_BLOCK_SIZE
    # base contact distribution
    matrix = rng.exponential(1.0, size=(n, n)).astype(np.float64)
    # amplify intra-block contacts so domains stick out
    for b in range(0, n, block_size):
        end = min(b + block_size, n)
        matrix[b:end, b:end] *= 3.0
    # symmetrise + zero diagonal (TopDom assumes symmetric, dense, diagonal-free)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return {
        "topdom.matrix": matrix.astype(np.float32),
        "topdom.window_size": np.asarray(DOMAIN_WINDOW_SIZE, dtype=np.int32),
        "insulation.window_size": np.asarray(DOMAIN_WINDOW_SIZE, dtype=np.int32),
    }


# ---- Phase 3 fixture parameters ----
COMPARTMENT_N_BINS = 100


def compartment_small_fixture(seed: int = 47) -> dict:
    """100×100 symmetric Hi-C matrix + CpG-ratio vector for compartment calling.

    A handful of zeroed CpG bins exercises the `bin_filter = cpg_ratio > 0`
    branch in compartment_strength; the remaining bins span the
    20th/80th percentile A/B partition.
    """
    rng = np.random.default_rng(seed)
    n = COMPARTMENT_N_BINS
    m = rng.exponential(1.0, (n, n)).astype(np.float64)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 0.0)
    cpg = rng.uniform(0.0, 0.1, n).astype(np.float32)
    cpg[::10] = 0.0  # 10 bins zeroed for the bin_filter branch
    return {
        "compartment.matrix": m.astype(np.float32),
        "compartment.cpg_ratio": cpg,
    }


# ---- Phase 4 fixture parameters ----
EMBEDDING_N_CELLS = 5
EMBEDDING_N_BINS = 50
EMBEDDING_DIST = 200_000
EMBEDDING_RESOLUTION = 10_000
EMBEDDING_SCALE_FACTOR = 100_000


def embedding_small_fixture(seed: int = 48) -> dict:
    """5 cells × 50×50 dense Hi-C matrices for embedding cell-by-feature extraction.

    Bypasses cooler I/O — the upstream's `make_chrom_matrix` reads .cool files
    per cell; our driver code calls the same logic but with the cells already
    in memory so the parity gate exercises only the Rust kernel.
    """
    rng = np.random.default_rng(seed)
    cells = np.stack([
        rng.exponential(1.0, (EMBEDDING_N_BINS, EMBEDDING_N_BINS)).astype(np.float32)
        for _ in range(EMBEDDING_N_CELLS)
    ])
    return {
        "embedding.cells": cells,
        "embedding.n_bins": np.asarray(EMBEDDING_N_BINS, dtype=np.int32),
        "embedding.dist": np.asarray(EMBEDDING_DIST, dtype=np.int32),
        "embedding.resolution": np.asarray(EMBEDDING_RESOLUTION, dtype=np.int32),
        "embedding.scale_factor": np.asarray(EMBEDDING_SCALE_FACTOR, dtype=np.int32),
    }


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    # ---- conv_small (Phase 0) ----
    fixture = conv_small_fixture()
    np.savez(FIXTURE_DIR / "conv_small.npz", **fixture)
    print(f"wrote {FIXTURE_DIR / 'conv_small.npz'}")
    print(f"  input.shape    = {fixture['input'].shape}, dtype = {fixture['input'].dtype}")
    print(f"  kernel.shape   = {fixture['kernel'].shape}, sum  = {fixture['kernel'].sum():.6f}")
    print(f"  convolved.shape= {fixture['convolved'].shape}, mean = {fixture['convolved'].mean():.6e}")
    # ---- loop_small (Phase 1) ----
    loop_pack = loop_small_packed_fixture()
    np.savez(FIXTURE_DIR / "loop_small.npz", **loop_pack)
    print(f"wrote {FIXTURE_DIR / 'loop_small.npz'} ({len(loop_pack)} keys)")
    print(f"  n_cells  = {LOOP_N_CELLS}, n_bins = {LOOP_N_BINS}")
    print(f"  scan loop pixels = {loop_pack['scan.loop_xs'].size}")
    # ---- domain_small (Phase 2) ----
    domain_pack = domain_small_fixture()
    np.savez(FIXTURE_DIR / "domain_small.npz", **domain_pack)
    print(f"wrote {FIXTURE_DIR / 'domain_small.npz'} ({len(domain_pack)} keys)")
    print(f"  matrix.shape    = {domain_pack['topdom.matrix'].shape}, "
          f"dtype = {domain_pack['topdom.matrix'].dtype}")
    print(f"  window_size     = {int(domain_pack['topdom.window_size'])}")
    # ---- compartment_small (Phase 3) ----
    comp_pack = compartment_small_fixture()
    np.savez(FIXTURE_DIR / "compartment_small.npz", **comp_pack)
    print(f"wrote {FIXTURE_DIR / 'compartment_small.npz'} ({len(comp_pack)} keys)")
    print(f"  matrix.shape    = {comp_pack['compartment.matrix'].shape}")
    print(f"  cpg.nnz         = {int((comp_pack['compartment.cpg_ratio'] > 0).sum())}")
    # ---- embedding_small (Phase 4) ----
    emb_pack = embedding_small_fixture()
    np.savez(FIXTURE_DIR / "embedding_small.npz", **emb_pack)
    print(f"wrote {FIXTURE_DIR / 'embedding_small.npz'} ({len(emb_pack)} keys)")
    print(f"  cells.shape     = {emb_pack['embedding.cells'].shape}")
    print(f"  dist_bins       = {int(emb_pack['embedding.dist']) // int(emb_pack['embedding.resolution'])}")


if __name__ == "__main__":
    main()
