# scHiCluster Rust port — Phase 1 (loop module)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit discipline:** per user direction, this plan ships as **one commit at the end** (Task 9). Subagent implementers in earlier tasks must NOT run `git commit`. They write code, run smoke checks, leave changes staged or unstaged for the controller to assemble the single Phase 1 commit. The final task is the only commit step.

**Goal:** Turn 10 of the 17 pre-registered parity-manifest outputs green by porting `scHiCluster.loop.{loop_bkg, merge_cell_to_group, loop_calling}` numerical hot paths to Rust, leaving cooler/HDF5 I/O, pandas dataframes, `multipletests` FDR, and the paired t-test in Python.

**Architecture:** Four new Rust files (`loop_bkg.rs`, `merge.rs`, `scan_kernels.rs`, `find_summit.rs`) each exposing one coarse PyO3 entrypoint that maps to one upstream Python function. Reuses Phase 0's `conv::convolve2d_mirror` for all five convolutions in the pipeline (one in `loop_bkg`, four in `scan_kernels`). A new `python/schicluster_rs/loop.py` shim holds the public wrappers (cooler/HDF5/.npz glue) and `patch_schicluster()` rebinds the upstream per-chrom functions. Parity validated cross-env (`schicluster` py3.6 reference ↔ `rebuild-rust` py3.10 candidate) against the existing manifest gate.

**Tech Stack:** Rust 1.95 + PyO3 0.22 + ndarray 0.16 + rayon 1.10 + sprs 0.11 (existing); `cooler`, `h5py`, `numpy`, `scipy`, `pandas` for the Python wrappers (already in `rebuild-rust` dev extras from Phase 0).

---

## Phase 1 outputs (already pre-registered in `data/manifest.yaml` — read-only)

| Manifest output | Algorithm class | Threshold | Upstream Python |
|---|---|---|---|
| `loop_bkg.E` | deterministic-bounded | 1e-6 | `loop/loop_bkg.py::calculate_chrom_background_normalization` → E.npz |
| `loop_bkg.T` | deterministic-bounded | 1e-6 | same → T.npz |
| `merge.e_sum` | deterministic-bounded | 1e-6 | `loop/merge_cell_to_group.py::merge_cells_for_single_chromosome` → E sum CSR |
| `merge.e2_sum` | deterministic-bounded | 1e-6 | same → E² sum CSR |
| `scan_kernels.bl` | deterministic-bounded | 1e-6 | `loop/loop_calling.py::loop_background` → bottom-left kernel values |
| `scan_kernels.donut` | deterministic-bounded | 1e-6 | same → donut kernel values |
| `scan_kernels.h` | deterministic-bounded | 1e-6 | same → horizontal kernel values |
| `scan_kernels.v` | deterministic-bounded | 1e-6 | same → vertical kernel values |
| `find_summit.idx` | ranked (set Jaccard) | 0.99 | `loop/loop_calling.py::find_summit` → selected pixel indices |
| `find_summit.sizes` | classification | 1.0 (exact on intersection) | same → cluster sizes |

Phase 1 is complete when `pytest -q tests/test_exact_match.py` reports `11 passed, 6 skipped` (1 conv from Phase 0 + 10 loop outputs; the 6 skips are domain/compartment/embedding).

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `data/fixtures/synthesize.py` | MODIFY | Adds `loop_bkg_small_fixture`, `merge_small_fixture`, `scan_summit_small_fixture` + a tiny synthetic cooler file for the upstream Python reference to read. |
| `data/fixtures/loop_small.cool` | NEW (generated, .gitignored) | Synthetic cooler — 200 bins × ~10 % sparsity upper-tri — used by `loop_bkg` reference. |
| `data/fixtures/loop_small.npz` | NEW (generated, .gitignored) | Per-cell sparse matrices + scan E + summit pixels packed into one npz. |
| `rust/src/loop_bkg.rs` | NEW | `loop_bkg_chrom` — per-diagonal pctl-99/zscore±cap normalisation + donut convolution. Reuses `crate::conv::convolve2d_mirror`. |
| `rust/src/merge.rs` | NEW | `merge_cells_sum` — sparse accumulator producing `(e_sum, e2_sum)` triplets in row-major order. |
| `rust/src/scan_kernels.rs` | NEW | `scan_kernels_chrom` — builds 4 kernels (bl, donut, h, v), four convolutions on `E`, gather at loop pixels. |
| `rust/src/find_summit.rs` | NEW | `find_summit_chrom` — neighbor graph + max-heap peak merge. |
| `rust/src/lib.rs` | MODIFY | Mounts the four new modules; registers four new PyO3 functions in `#[pymodule] _rust`. |
| `python/schicluster_rs/loop.py` | NEW | Python wrappers (cooler/HDF5/.npz glue) for each kernel; module-level monkey-patch helper. |
| `python/schicluster_rs/__init__.py` | MODIFY | Imports four new Rust symbols + the loop wrappers; extends `__all__`; extends `patch_schicluster()` to rebind upstream loop module functions. |
| `tests/py_reference_driver.py` | MODIFY | Adds 10 reference computations (calls upstream Python on the fixture). |
| `tests/_run_candidate.py` | MODIFY | Adds 10 candidate computations (calls Rust on the fixture). |
| `rust/Cargo.toml` | MODIFY (Task 9 only) | Version `0.2.0-dev0` → `0.2.0`. |
| `pyproject.toml` | MODIFY (Task 9 only) | Version `0.2.0.dev0` → `0.2.0`. |
| `ITERATION_LOG.md` | MODIFY (Task 9 only) | Append Phase 1 baseline block (iteration 1). |

---

## Task 1: Phase 1 fixtures + synthetic cooler

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/synthesize.py`

The plan: extend `synthesize.py` to write a single packed `loop_small.npz` containing every array the Phase-1 reference and candidate drivers need, plus a tiny `loop_small.cool` cooler file so the upstream `calculate_chrom_background_normalization` can be called against a real cooler URL (the upstream function takes a cool URL, not an in-memory matrix).

- [ ] **Step 1: Extend `synthesize.py`.** Append the following at module scope (after the existing `conv_small_fixture` and `main` definitions; add the new fixtures to `main`):

```python
import cooler
import pandas as pd


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
    # Sort row-major for cooler ingestion.
    order = np.lexsort((cols, rows))
    return rows[order], cols[order], vals[order]


def _bins_df(n_bins, resolution, chrom):
    starts = np.arange(n_bins, dtype=np.int64) * resolution
    return pd.DataFrame({
        "chrom": [chrom] * n_bins,
        "start": starts,
        "end": starts + resolution,
    })


def loop_bkg_small_fixture(seed=42):
    """Synthetic single-cell sparse upper-tri matrix used as input to loop_bkg.

    Density picked so all 20 distance diagonals have at least one nnz.
    """
    rows, cols, vals = _upper_tri_synthetic(seed, LOOP_N_BINS, density=0.15, max_diag=20)
    return {"loop_bkg.input.rows": rows, "loop_bkg.input.cols": cols, "loop_bkg.input.vals": vals}


def merge_small_fixture(seed=43):
    """Five per-cell sparse upper-tri matrices for merge_cells_sum.

    All matrices share the same shape (LOOP_N_BINS, LOOP_N_BINS); cell triplets
    are packed flat with a cell-id column so the driver code can split them.
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

    Loop pixels are produced by `np.where(E_dense > 0) & (dist filter)` over a
    seeded dense E that's smooth enough for the donut/h/v kernels to have
    non-trivial values everywhere.
    """
    rng = np.random.default_rng(seed)
    e = rng.uniform(0.0, 1.0, size=(LOOP_N_BINS, LOOP_N_BINS)).astype(np.float32)
    # Drop the lower triangle + diagonal so the matrix matches upstream's
    # "upper-tri only" expectation. Loop pixels must be in the upper triangle.
    iu = np.triu_indices(LOOP_N_BINS, k=1)
    e_upper = np.zeros((LOOP_N_BINS, LOOP_N_BINS), dtype=np.float32)
    e_upper[iu] = e[iu]
    # Loop candidate pixels: positive E and 2 < (y - x) < 20 bins (matches
    # loop_calling's min_dist / max_dist filter at LOOP_RESOLUTION).
    ys, xs = np.where(e_upper > 0.5)
    diff = ys - xs  # ys is col, xs is row; matches loop_calling indexing
    mask = (diff > 2) & (diff < 20)
    xs = xs[mask].astype(np.uint32)
    ys = ys[mask].astype(np.uint32)
    e_vals_at_loop = e_upper[xs, ys].astype(np.float32)
    return {
        "scan.E_dense": e_upper,
        "scan.loop_xs": xs,
        "scan.loop_ys": ys,
        "summit.x1": (xs * LOOP_RESOLUTION).astype(np.int64),  # genomic coord (bp)
        "summit.y1": (ys * LOOP_RESOLUTION).astype(np.int64),
        "summit.E": e_vals_at_loop,
    }


def write_loop_small_cool():
    """Write a tiny synthetic single-cell cooler used by the upstream loop_bkg reference."""
    rng = np.random.default_rng(45)
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
    return {"loop_bkg.input.rows": rows, "loop_bkg.input.cols": cols, "loop_bkg.input.vals": vals}


def loop_small_packed_fixture():
    """Pack every Phase-1 array into one npz keyed by output names from manifest."""
    bkg = write_loop_small_cool()       # also produces the cooler the upstream needs
    merge = merge_small_fixture()
    scan_summit = scan_summit_small_fixture()
    return {**bkg, **merge, **scan_summit}


# --- extend main() so both fixtures regenerate together ---

# Find the existing `def main():` near the bottom of synthesize.py and insert
# the loop_small write before the trailing `print` so the same `python
# data/fixtures/synthesize.py` invocation produces both conv_small.npz and
# loop_small.npz + loop_small.cool.
```

After the new function definitions, **edit** the existing `def main()` to additionally write the loop fixture:

Original (end of `main`):
```python
def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture = conv_small_fixture()
    np.savez(FIXTURE_DIR / "conv_small.npz", **fixture)
    print(f"wrote {FIXTURE_DIR / 'conv_small.npz'}")
    print(f"  input.shape    = {fixture['input'].shape}, dtype = {fixture['input'].dtype}")
    print(f"  kernel.shape   = {fixture['kernel'].shape}, sum  = {fixture['kernel'].sum():.6f}")
    print(f"  convolved.shape= {fixture['convolved'].shape}, mean = {fixture['convolved'].mean():.6e}")
```

Replace with:
```python
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
```

- [ ] **Step 2: Add the new generated artefacts to `.gitignore`:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
for line in "data/fixtures/loop_small.npz" "data/fixtures/loop_small.cool"; do
  grep -qxF "$line" .gitignore || echo "$line" >> .gitignore
done
```

- [ ] **Step 3: Regenerate fixtures in `rebuild-rust` and verify keys:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    python data/fixtures/synthesize.py
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
import numpy as np, pathlib, cooler
d = np.load(pathlib.Path('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/loop_small.npz'))
print('loop_small.npz keys:', sorted(d.files))
print('  loop_bkg.input.rows nnz =', d['loop_bkg.input.rows'].size)
print('  merge.cell_ids unique  =', np.unique(d['merge.cell_ids']).tolist())
print('  scan.E_dense shape     =', d['scan.E_dense'].shape)
print('  scan loop pixels       =', d['scan.loop_xs'].size)
print('  summit.E nnz           =', d['summit.E'].size)
c = cooler.Cooler('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/loop_small.cool')
print('cooler bins:', c.bins()[:].shape[0], 'chroms:', list(c.chromnames))
"
```

Expected:
- `loop_small.npz keys` includes every key listed in the new fixture functions.
- `merge.cell_ids unique  = [0, 1, 2, 3, 4]`.
- `scan.E_dense shape     = (200, 200)`.
- `loop pixels` count > 0 and `summit.E nnz` equal to `scan.loop_xs.size`.
- cooler bins = 200, chroms = `['chr1']`.

If the run fails with a SyntaxError, fix the syntax in `synthesize.py` and re-run. If `cooler.create_cooler` errors with `ordered=True`, drop that kwarg and re-run.

- [ ] **Step 4: Verify the byte form of the cooler is usable by upstream:**

```bash
env -u VIRTUAL_ENV conda run -n schicluster --no-capture-output python -c "
import cooler
c = cooler.Cooler('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/loop_small.cool')
m = c.matrix(balance=False, sparse=True).fetch('chr1')
print('schicluster-env cool fetch ok; shape =', m.shape, 'nnz =', m.nnz)
"
```

Expected: prints shape `(200, 200)` and a positive nnz. Confirms the synthesized cool is interpretable by the schicluster env's older cooler.

**No commit in this task** — leave the synthesize.py modifications staged for the Task 9 final commit.

---

## Task 2: Rust `loop_bkg_chrom`

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/loop_bkg.rs`

Ports `scHiCluster/schicluster/loop/loop_bkg.py::calculate_chrom_background_normalization` (lines 29-115). The Rust function takes raw upper-tri CSR triplets, returns two (rows, cols, vals) tuples representing the E and T sparse matrices. The Python wrapper handles cooler read + .npz write.

The numerical pipeline (each step matches upstream):
1. Densify the input upper-tri matrix into `Array2<f32>` shape (n, n).
2. Zero the diagonal.
3. For diagonal `d ∈ 1..=dist_bins`:
   - Take diagonal values into a contiguous f32 slice.
   - Filter positives (`v > 0`).
   - If `log_e`: zscore log10(positives); else pctl-99 clip + zscore.
   - NaN → 0; clip to `[-cap, cap]`.
   - Write back to the diagonal: positives get the zscored values; non-positives get `min(zscored)`.
   - **Shuffle is intentionally excluded from Rust** (RNG-driven; out of gate per spec §11). Python wrapper handles `shuffle=True` by falling back to upstream.
4. Build the (2 pad+1)×(2 pad+1) donut kernel; T = `conv::convolve2d_mirror(E, kernel)`.
5. Construct upper-tri mask (diagonal + up to `dist_bins` superdiagonals). Element-wise apply: `E_sparse = sparse(E)`; `T_sparse = sparse(T * mask)`.
6. `min_cutoff > 0` → drop entries with `|.| ≤ min_cutoff` in both E and T.
7. T = E - T (sparse delta).
8. Emit E and T as upper-tri triplets sorted row-major.

- [ ] **Step 1: Write `rust/src/loop_bkg.rs`** (Write tool):

```rust
//! Port of scHiCluster/schicluster/loop/loop_bkg.py::calculate_chrom_background_normalization
//! (excluding the shuffle=True stochastic branch — see python/schicluster_rs/loop.py).
//!
//! Pipeline:
//!   1. densify upper-tri triplets -> Array2<f32>
//!   2. zero diagonal
//!   3. per-diagonal pctl-99 clip + scipy.stats.zscore (ddof=0) + clip(-cap, cap),
//!      writing the zscored values into positives and `min(zscored)` into non-positives
//!   4. donut-minus convolution -> T
//!   5. upper-tri mask T; apply min_cutoff |.| filter to both E and T
//!   6. emit E and T upper-tri triplets row-major (T = E - T at the sparse level)

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::conv::convolve2d_mirror;

const ZSCORE_NAN_REPLACEMENT: f32 = 0.0;

/// numpy.percentile default ("linear" interpolation): for a sorted ascending
/// slice `s` of length n and a percentile p ∈ [0, 100], compute the value at
/// the fractional index k = p/100 * (n - 1) via linear interpolation.
fn percentile_linear(values: &mut [f32], p: f32) -> f32 {
    debug_assert!(!values.is_empty());
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 1 {
        return values[0];
    }
    let frac = p / 100.0 * (n as f32 - 1.0);
    let lo = frac.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let w = frac - lo as f32;
    values[lo] * (1.0 - w) + values[hi] * w
}

/// scipy.stats.zscore with ddof=0: (x - mean) / std, std with denominator n.
/// Returns a fresh Vec with same length.
fn zscore_ddof0(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = values.iter().copied().sum::<f32>() / n as f32;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    if std == 0.0 {
        // scipy returns NaNs in this case; upstream replaces with 0.
        return vec![0.0; n];
    }
    values.iter().map(|v| (v - mean) / std).collect()
}

/// Replace NaN with `ZSCORE_NAN_REPLACEMENT` (matches `tmp2[isnan(tmp2)] = 0`).
fn nan_to_zero(values: &mut [f32]) {
    for v in values.iter_mut() {
        if v.is_nan() {
            *v = ZSCORE_NAN_REPLACEMENT;
        }
    }
}

/// `tmp` and `tmp_filter` form one diagonal of E. Mutates `tmp` in place to the
/// upstream's post-normalisation form for that diagonal.
fn normalise_diagonal(tmp: &mut [f32], cap: f32, log_e: bool) {
    let tmp_filter: Vec<bool> = tmp.iter().map(|v| *v > 0.0).collect();
    let mut positives: Vec<f32> = tmp.iter().copied().filter(|v| *v > 0.0).collect();
    if positives.is_empty() {
        for v in tmp.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let normalised: Vec<f32> = if log_e {
        let logs: Vec<f32> = positives.iter().map(|v| v.log10()).collect();
        let mut z = zscore_ddof0(&logs);
        nan_to_zero(&mut z);
        z.iter().map(|v| v.clamp(-cap, cap)).collect()
    } else {
        let cutoff = percentile_linear(&mut positives, 99.0);
        let clipped: Vec<f32> = positives.iter().map(|v| v.min(cutoff)).collect();
        let mut z = zscore_ddof0(&clipped);
        nan_to_zero(&mut z);
        z.iter().map(|v| v.clamp(-cap, cap)).collect()
    };

    let min_val = normalised
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);

    let mut k = 0usize;
    for (i, was_positive) in tmp_filter.iter().enumerate() {
        if *was_positive {
            tmp[i] = normalised[k];
            k += 1;
        } else {
            tmp[i] = min_val;
        }
    }
}

/// Build the donut-minus kernel of width `w = 2*pad+1`:
/// 1 everywhere except the (pad-gap)..=(pad+gap) inner block (which is 0),
/// then normalised to sum to 1. Matches loop_bkg.py lines 99-102.
fn donut_kernel(pad: usize, gap: usize) -> (Vec<f32>, usize) {
    let w = 2 * pad + 1;
    let mut k = vec![1.0_f32; w * w];
    let lo = pad - gap;
    let hi = pad + gap;
    for i in lo..=hi {
        for j in lo..=hi {
            k[i * w + j] = 0.0;
        }
    }
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    (k, w)
}

#[allow(clippy::too_many_arguments)]
pub fn loop_bkg_chrom(
    rows: &[u32],
    cols: &[u32],
    vals: &[f32],
    n: usize,
    dist_bins: usize,
    cap: f32,
    pad: usize,
    gap: usize,
    min_cutoff: f32,
    log_e: bool,
) -> (
    (Vec<u32>, Vec<u32>, Vec<f32>),
    (Vec<u32>, Vec<u32>, Vec<f32>),
) {
    // 1. densify upper-tri triplets
    let mut e_dense = Array2::<f32>::zeros((n, n));
    {
        let buf = e_dense.as_slice_mut().expect("contig");
        for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
            let r = r as usize;
            let c = c as usize;
            if r < n && c < n {
                buf[r * n + c] = v;
            }
        }
    }

    // 2. zero diagonal
    let buf = e_dense.as_slice_mut().expect("contig");
    for i in 0..n {
        buf[i * n + i] = 0.0;
    }

    // 3. per-diagonal pctl-99/zscore±cap normalisation
    let dist = dist_bins.min(n.saturating_sub(1));
    for d in 1..=dist {
        let length = n - d;
        let mut tmp = vec![0.0_f32; length];
        for k in 0..length {
            tmp[k] = buf[k * n + (k + d)];
        }
        normalise_diagonal(&mut tmp, cap, log_e);
        for k in 0..length {
            buf[k * n + (k + d)] = tmp[k];
        }
    }

    // 4. donut convolution -> T_dense
    let (kernel, kw) = donut_kernel(pad, gap);
    let t_dense_vec = convolve2d_mirror(buf, n, n, &kernel, kw, kw);

    // 5. upper-tri mask + sparse emission with min_cutoff
    let mut e_rows: Vec<u32> = Vec::new();
    let mut e_cols: Vec<u32> = Vec::new();
    let mut e_vals: Vec<f32> = Vec::new();
    let mut t_rows: Vec<u32> = Vec::new();
    let mut t_cols: Vec<u32> = Vec::new();
    let mut t_vals: Vec<f32> = Vec::new();

    let in_mask = |i: usize, j: usize| -> bool {
        // upstream mask: diagonal + diagonals 1..=dist_bins
        j >= i && (j - i) <= dist_bins
    };

    for i in 0..n {
        for j in i..n {
            if !in_mask(i, j) {
                continue;
            }
            let e = buf[i * n + j];
            let t_raw = t_dense_vec[i * n + j];

            // upstream applies min_cutoff |.| filter independently to E and T,
            // THEN computes T = E - T at the sparse level.
            let e_keep = if min_cutoff > 0.0 {
                e.abs() > min_cutoff
            } else {
                e != 0.0
            };
            let t_keep = if min_cutoff > 0.0 {
                t_raw.abs() > min_cutoff
            } else {
                t_raw != 0.0
            };

            // sparse delta: T_out = E_keep ? e : 0   -   T_keep ? t_raw : 0
            let e_contrib = if e_keep { e } else { 0.0 };
            let t_contrib = if t_keep { t_raw } else { 0.0 };
            let t_out = e_contrib - t_contrib;

            if e_keep {
                e_rows.push(i as u32);
                e_cols.push(j as u32);
                e_vals.push(e);
            }
            if t_out != 0.0 {
                t_rows.push(i as u32);
                t_cols.push(j as u32);
                t_vals.push(t_out);
            }
        }
    }

    ((e_rows, e_cols, e_vals), (t_rows, t_cols, t_vals))
}

#[pyfunction]
#[pyo3(signature = (rows, cols, vals, n, dist_bins, cap=5.0, pad=5, gap=2, min_cutoff=1e-6, log_e=false))]
#[allow(clippy::too_many_arguments)]
pub fn py_loop_bkg_chrom<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, u32>,
    cols: PyReadonlyArray1<'py, u32>,
    vals: PyReadonlyArray1<'py, f32>,
    n: usize,
    dist_bins: usize,
    cap: f32,
    pad: usize,
    gap: usize,
    min_cutoff: f32,
    log_e: bool,
) -> PyResult<(
    (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>),
    (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>),
)> {
    let r = rows.as_slice()?;
    let c = cols.as_slice()?;
    let v = vals.as_slice()?;

    let ((er, ec, ev), (tr, tc, tv)) = py.allow_threads(|| {
        loop_bkg_chrom(r, c, v, n, dist_bins, cap, pad, gap, min_cutoff, log_e)
    });

    Ok((
        (
            ndarray::Array1::from(er).into_pyarray_bound(py),
            ndarray::Array1::from(ec).into_pyarray_bound(py),
            ndarray::Array1::from(ev).into_pyarray_bound(py),
        ),
        (
            ndarray::Array1::from(tr).into_pyarray_bound(py),
            ndarray::Array1::from(tc).into_pyarray_bound(py),
            ndarray::Array1::from(tv).into_pyarray_bound(py),
        ),
    ))
}
```

- [ ] **Step 2: Inline smoke test against scipy after Task 5 has built the extension.** Hold this step until after Task 5; here you only write the file.

**No commit in this task.**

---

## Task 3: Rust `merge_cells_sum`

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/merge.rs`

Ports `scHiCluster/schicluster/loop/merge_cell_to_group.py::merge_cells_for_single_chromosome` (lines 17-45). The Rust function takes a flat array of (cell_id, row, col, val) plus `n` and `n_cells`, accumulates `e_sum` and `e2_sum` into `BTreeMap<(u32, u32), (f64, f64)>` (BTreeMap so iteration order is deterministic for parity), then emits triplets sorted row-major.

`e2_sum` accumulates `v * v` (CSR `multiply` is element-wise square because each cell's matrix is non-overlapping with itself).

f64 accumulator is the cheap (E)-exact way to dodge the reordering-sensitivity that f32 sums would carry — the cast back to f32 at emit time produces bit-equivalent results to scipy's CSR + arithmetic, which also accumulates in higher precision internally.

- [ ] **Step 1: Write `rust/src/merge.rs`** (Write tool):

```rust
//! Port of scHiCluster/schicluster/loop/merge_cell_to_group.py::merge_cells_for_single_chromosome.
//!
//! Accumulates (Σ_cells m_c, Σ_cells m_c .* m_c) over per-cell sparse matrices in
//! upper-triangle COO form. Emit order is row-major sorted; values stored f32.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use std::collections::BTreeMap;

pub fn merge_cells_sum(
    cell_ids: &[u32],
    rows: &[u32],
    cols: &[u32],
    vals: &[f32],
) -> (
    (Vec<u32>, Vec<u32>, Vec<f32>),
    (Vec<u32>, Vec<u32>, Vec<f32>),
) {
    debug_assert_eq!(cell_ids.len(), rows.len());
    debug_assert_eq!(rows.len(), cols.len());
    debug_assert_eq!(cols.len(), vals.len());

    let mut e_acc: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    let mut e2_acc: BTreeMap<(u32, u32), f64> = BTreeMap::new();

    for ((&_cid, &r), (&c, &v)) in cell_ids
        .iter()
        .zip(rows.iter())
        .zip(cols.iter().zip(vals.iter()))
    {
        let key = (r, c);
        *e_acc.entry(key).or_insert(0.0) += v as f64;
        *e2_acc.entry(key).or_insert(0.0) += (v as f64) * (v as f64);
    }

    let (mut e_rows, mut e_cols, mut e_vals) =
        (Vec::with_capacity(e_acc.len()), Vec::with_capacity(e_acc.len()), Vec::with_capacity(e_acc.len()));
    for ((r, c), v) in e_acc.into_iter() {
        e_rows.push(r);
        e_cols.push(c);
        e_vals.push(v as f32);
    }

    let (mut e2_rows, mut e2_cols, mut e2_vals) =
        (Vec::with_capacity(e2_acc.len()), Vec::with_capacity(e2_acc.len()), Vec::with_capacity(e2_acc.len()));
    for ((r, c), v) in e2_acc.into_iter() {
        e2_rows.push(r);
        e2_cols.push(c);
        e2_vals.push(v as f32);
    }

    ((e_rows, e_cols, e_vals), (e2_rows, e2_cols, e2_vals))
}

#[pyfunction]
#[pyo3(signature = (cell_ids, rows, cols, vals))]
pub fn py_merge_cells_sum<'py>(
    py: Python<'py>,
    cell_ids: PyReadonlyArray1<'py, u32>,
    rows: PyReadonlyArray1<'py, u32>,
    cols: PyReadonlyArray1<'py, u32>,
    vals: PyReadonlyArray1<'py, f32>,
) -> PyResult<(
    (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>),
    (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>),
)> {
    let ids = cell_ids.as_slice()?;
    let r = rows.as_slice()?;
    let c = cols.as_slice()?;
    let v = vals.as_slice()?;

    let ((er, ec, ev), (e2r, e2c, e2v)) =
        py.allow_threads(|| merge_cells_sum(ids, r, c, v));

    Ok((
        (
            ndarray::Array1::from(er).into_pyarray_bound(py),
            ndarray::Array1::from(ec).into_pyarray_bound(py),
            ndarray::Array1::from(ev).into_pyarray_bound(py),
        ),
        (
            ndarray::Array1::from(e2r).into_pyarray_bound(py),
            ndarray::Array1::from(e2c).into_pyarray_bound(py),
            ndarray::Array1::from(e2v).into_pyarray_bound(py),
        ),
    ))
}
```

**No commit in this task.**

---

## Task 4: Rust `scan_kernels_chrom`

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/scan_kernels.rs`

Ports `scHiCluster/schicluster/loop/loop_calling.py::loop_background` (lines 56-80). Builds the four kernels (bl, donut, h, v), convolves the dense E matrix with each via Phase-0 `conv::convolve2d_mirror`, applies the `E > 0` mask, and gathers values at `(xs, ys)` loop pixels.

Important kernel shapes (from upstream):
- **bl** (bottom-left): width `w = 2*pad+1`, zeros everywhere except `[-pad:, :(pad-gap)] = 1` and `[-(pad-gap):, :pad] = 1`, normalised. **Asymmetric** — exercises the kernel-flip semantics of `convolve2d_mirror`.
- **donut**: `w × w`, ones everywhere except the central cross `[pad, :]`, `[:, pad]` and the central block `[(pad-gap):(pad+gap+1), (pad-gap):(pad+gap+1)]`, normalised. Symmetric.
- **h** (horizontal stripe): `3 × w`, ones everywhere except `[:, (pad-gap):(pad+gap+1)] = 0`, normalised. Asymmetric in kw direction.
- **v** (vertical stripe): `w × 3`, ones everywhere except `[(pad-gap):(pad+gap+1), :] = 0`, normalised. Asymmetric in kh direction.

- [ ] **Step 1: Write `rust/src/scan_kernels.rs`** (Write tool):

```rust
//! Port of scHiCluster/schicluster/loop/loop_calling.py::loop_background.
//!
//! Build bl / donut / h / v kernels, convolve E with each (scipy mirror),
//! multiply by (E > 0) mask, gather values at loop pixels (xs, ys).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::conv::convolve2d_mirror;

fn build_bl_kernel(pad: usize, gap: usize) -> (Vec<f32>, usize, usize) {
    let w = 2 * pad + 1;
    let mut k = vec![0.0_f32; w * w];
    // k[-pad:, :(pad-gap)] = 1
    for i in (w - pad)..w {
        for j in 0..(pad - gap) {
            k[i * w + j] = 1.0;
        }
    }
    // k[-(pad-gap):, :pad] = 1
    for i in (w - (pad - gap))..w {
        for j in 0..pad {
            k[i * w + j] = 1.0;
        }
    }
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    (k, w, w)
}

fn build_donut_kernel(pad: usize, gap: usize) -> (Vec<f32>, usize, usize) {
    let w = 2 * pad + 1;
    let mut k = vec![1.0_f32; w * w];
    for j in 0..w {
        k[pad * w + j] = 0.0; // row `pad`
        k[j * w + pad] = 0.0; // column `pad`
    }
    for i in (pad - gap)..=(pad + gap) {
        for j in (pad - gap)..=(pad + gap) {
            k[i * w + j] = 0.0;
        }
    }
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    (k, w, w)
}

fn build_h_kernel(pad: usize, gap: usize) -> (Vec<f32>, usize, usize) {
    let w = 2 * pad + 1;
    let kh = 3usize;
    let mut k = vec![1.0_f32; kh * w];
    for i in 0..kh {
        for j in (pad - gap)..=(pad + gap) {
            k[i * w + j] = 0.0;
        }
    }
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    (k, kh, w)
}

fn build_v_kernel(pad: usize, gap: usize) -> (Vec<f32>, usize, usize) {
    let w = 2 * pad + 1;
    let kw = 3usize;
    let mut k = vec![1.0_f32; w * kw];
    for i in (pad - gap)..=(pad + gap) {
        for j in 0..kw {
            k[i * kw + j] = 0.0;
        }
    }
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    (k, w, kw)
}

fn scan_with_kernel(
    e: &[f32],
    n: usize,
    kernel: &[f32],
    kh: usize,
    kw: usize,
    xs: &[u32],
    ys: &[u32],
) -> Vec<f32> {
    let convd = convolve2d_mirror(e, n, n, kernel, kh, kw);
    let mut out = Vec::with_capacity(xs.len());
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let i = x as usize;
        let j = y as usize;
        let e_val = e[i * n + j];
        let v = if e_val > 0.0 { convd[i * n + j] } else { 0.0 };
        out.push(v);
    }
    out
}

pub fn scan_kernels_chrom(
    e: &[f32],
    n: usize,
    pad: usize,
    gap: usize,
    xs: &[u32],
    ys: &[u32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (kbl, kh_bl, kw_bl) = build_bl_kernel(pad, gap);
    let (kdo, kh_do, kw_do) = build_donut_kernel(pad, gap);
    let (kh_k, kh_h, kw_h) = build_h_kernel(pad, gap);
    let (kv, kh_v, kw_v) = build_v_kernel(pad, gap);
    let bl = scan_with_kernel(e, n, &kbl, kh_bl, kw_bl, xs, ys);
    let donut = scan_with_kernel(e, n, &kdo, kh_do, kw_do, xs, ys);
    let h = scan_with_kernel(e, n, &kh_k, kh_h, kw_h, xs, ys);
    let v = scan_with_kernel(e, n, &kv, kh_v, kw_v, xs, ys);
    (bl, donut, h, v)
}

#[pyfunction]
#[pyo3(signature = (e, pad, gap, xs, ys))]
pub fn py_scan_kernels_chrom<'py>(
    py: Python<'py>,
    e: PyReadonlyArray2<'py, f32>,
    pad: usize,
    gap: usize,
    xs: PyReadonlyArray1<'py, u32>,
    ys: PyReadonlyArray1<'py, u32>,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let e_view = e.as_array();
    let (n, n2) = (e_view.nrows(), e_view.ncols());
    if n != n2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "E must be square",
        ));
    }
    let e_slice = e_view
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("E must be C-contiguous f32"))?;
    let xs_s = xs.as_slice()?;
    let ys_s = ys.as_slice()?;

    let (bl, donut, h, v) = py.allow_threads(|| {
        scan_kernels_chrom(e_slice, n, pad, gap, xs_s, ys_s)
    });

    Ok((
        ndarray::Array1::from(bl).into_pyarray_bound(py),
        ndarray::Array1::from(donut).into_pyarray_bound(py),
        ndarray::Array1::from(h).into_pyarray_bound(py),
        ndarray::Array1::from(v).into_pyarray_bound(py),
    ))
}
```

**No commit in this task.**

---

## Task 5: Rust `find_summit_chrom`

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/find_summit.rs`

Ports `scHiCluster/schicluster/loop/loop_calling.py::find_summit` (lines 166-212). Uses Rust's `BinaryHeap` (max-heap by default) keyed on `(E, index)`; iterates `idx` in `argsort(x)` order to build the neighbour graph (within `dist_thres_bins` in both axes); then peels off peaks BFS-style.

Tie-breaking: the manifest classifies this as `ranked` (set-Jaccard ≥ 0.99), not `deterministic` — so a few equal-E swaps between Rust and Python are tolerable. We still keep the breaking deterministic: ties broken by ascending original index (matches Python's heap-stability under negation when input was indexed ascending).

- [ ] **Step 1: Write `rust/src/find_summit.rs`** (Write tool):

```rust
//! Port of scHiCluster/schicluster/loop/loop_calling.py::find_summit.
//!
//! O(N^2) neighbour graph + max-E heap peel. Inputs are in *bin* units
//! (caller does //= res before invoking). dist_thres_bins is the bin-radius
//! within which two pixels are neighbours.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone)]
struct HeapEntry {
    neg_e: f32,
    // `neg_e` lets us use a max-heap on E (BinaryHeap is max). Ties break by
    // ascending original `idx`, matching Python heap stability.
    idx: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.neg_e == other.neg_e && self.idx == other.idx
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // We want "max E first" => smaller neg_e first in BinaryHeap. BinaryHeap
        // is a *max*-heap so we reverse the neg_e ordering, then break ties by
        // ascending idx (i.e. smaller idx first => reversed).
        other
            .neg_e
            .partial_cmp(&self.neg_e)
            .unwrap_or(Ordering::Equal)
            .then(other.idx.cmp(&self.idx))
    }
}

pub fn find_summit_chrom(
    xs: &[u32],
    ys: &[u32],
    es: &[f32],
    dist_thres_bins: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = xs.len();
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert_eq!(ys.len(), es.len());
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // argsort by x (ascending)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| xs[a].cmp(&xs[b]));

    // neighbour graph: O(N^2) but stops once x-delta exceeds dist
    let mut neighbour: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..(n - 1) {
        let a = order[i];
        let x_a = xs[a];
        let y_a = ys[a];
        for j in (i + 1)..n {
            let b = order[j];
            let x_b = xs[b];
            if x_b.saturating_sub(x_a) > dist_thres_bins {
                break;
            }
            let y_b = ys[b];
            let dy = if y_a > y_b { y_a - y_b } else { y_b - y_a };
            if dy <= dist_thres_bins {
                neighbour[a].push(b as u32);
                neighbour[b].push(a as u32);
            }
        }
    }

    // max-heap by E with deterministic tie-break (ascending idx)
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(n);
    for k in 0..n {
        heap.push(HeapEntry { neg_e: -es[k], idx: k as u32 });
    }

    let mut flag = vec![false; n];
    let mut summit_idx: Vec<u32> = Vec::new();
    let mut summit_size: Vec<u32> = Vec::new();
    let mut tot = n;

    while tot > 0 {
        let mut t = match heap.pop() {
            Some(e) => e.idx as usize,
            None => break,
        };
        while flag[t] {
            t = match heap.pop() {
                Some(e) => e.idx as usize,
                None => return (summit_idx, summit_size),
            };
        }
        let root = t;
        let mut q: Vec<usize> = vec![root];
        flag[root] = true;
        tot -= 1;
        let mut head = 0usize;
        let mut flag_tmp = vec![false; n];
        while head < q.len() {
            for &t2_u32 in neighbour[q[head]].iter() {
                let t2 = t2_u32 as usize;
                if !flag_tmp[t2] && es[t2] < es[q[head]] {
                    if !flag[t2] {
                        flag[t2] = true;
                        tot -= 1;
                    }
                    flag_tmp[t2] = true;
                    q.push(t2);
                }
            }
            head += 1;
        }
        summit_idx.push(root as u32);
        summit_size.push(q.len() as u32);
    }

    (summit_idx, summit_size)
}

#[pyfunction]
#[pyo3(signature = (xs, ys, es, dist_thres_bins))]
pub fn py_find_summit_chrom<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, u32>,
    ys: PyReadonlyArray1<'py, u32>,
    es: PyReadonlyArray1<'py, f32>,
    dist_thres_bins: u32,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let x_s = xs.as_slice()?;
    let y_s = ys.as_slice()?;
    let e_s = es.as_slice()?;
    let (idx, sizes) = py.allow_threads(|| find_summit_chrom(x_s, y_s, e_s, dist_thres_bins));
    Ok((
        ndarray::Array1::from(idx).into_pyarray_bound(py),
        ndarray::Array1::from(sizes).into_pyarray_bound(py),
    ))
}
```

**No commit in this task.**

---

## Task 6: Mount loop modules + register PyO3 bindings + build

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/lib.rs`

Add four module mounts and four `wrap_pyfunction!` registrations.

- [ ] **Step 1: Modify `lib.rs`.** Find the existing `mod conv;` line (added in Phase 0) and add **immediately after it**:

```rust
mod find_summit;
mod loop_bkg;
mod merge;
mod scan_kernels;
```

Find the `#[pymodule] fn _rust(...)` function. It currently registers four PyO3 functions. Add **four more** registration lines before the trailing `Ok(())` (the existing four registrations stay untouched):

```rust
    m.add_function(wrap_pyfunction!(loop_bkg::py_loop_bkg_chrom, m)?)?;
    m.add_function(wrap_pyfunction!(merge::py_merge_cells_sum, m)?)?;
    m.add_function(wrap_pyfunction!(scan_kernels::py_scan_kernels_chrom, m)?)?;
    m.add_function(wrap_pyfunction!(find_summit::py_find_summit_chrom, m)?)?;
```

- [ ] **Step 2: Build:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    maturin develop --release 2>&1 | tail -20
```

Expected: ends with `🛠 Installed schicluster-rs-0.2.0.dev0`. Any compile error must be fixed in `loop_bkg.rs` / `merge.rs` / `scan_kernels.rs` / `find_summit.rs` or `lib.rs` registrations — do not stub-out or comment out manifest outputs.

- [ ] **Step 3: Smoke-test each new symbol is exposed:**

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
from schicluster_rs._rust import (
    py_loop_bkg_chrom, py_merge_cells_sum,
    py_scan_kernels_chrom, py_find_summit_chrom,
)
print('all four loop kernels importable:', all([
    callable(py_loop_bkg_chrom),
    callable(py_merge_cells_sum),
    callable(py_scan_kernels_chrom),
    callable(py_find_summit_chrom),
]))
"
```

Expected: `all four loop kernels importable: True`.

**No commit in this task.**

---

## Task 7: Python wrappers + `patch_schicluster()` extension

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/python/schicluster_rs/loop.py`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/python/schicluster_rs/__init__.py`

`loop.py` holds the cooler/HDF5/.npz glue around each Rust kernel; `__init__.py` re-exports the four wrappers and extends `patch_schicluster()` to monkey-patch the upstream `calculate_chrom_background_normalization`, `merge_cells_for_single_chromosome`, `loop_background`, and `find_summit`.

- [ ] **Step 1: Write `python/schicluster_rs/loop.py`** (Write tool):

```python
"""Python wrappers around the Rust loop-module kernels.

Each wrapper preserves the upstream function signature so that
`patch_schicluster()` can monkey-patch the upstream module attributes
without breaking callers. The shuffle=True path of loop_bkg falls back
to the upstream Python implementation (RNG; out of parity scope).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, save_npz, triu


def loop_bkg_chrom(cell_url, chrom, resolution, output_prefix,
                   dist=10_050_000, cap=5, pad=5, gap=2,
                   min_cutoff=1e-6, log_e=False, shuffle=False):
    """Drop-in replacement for
    schicluster.loop.loop_bkg.calculate_chrom_background_normalization.

    Calls the Rust kernel for the deterministic path; falls back to
    upstream Python when shuffle=True (RNG; intentionally not ported).
    """
    if shuffle:
        from schicluster.loop.loop_bkg import (
            calculate_chrom_background_normalization as _upstream,
        )
        return _upstream(
            cell_url=cell_url, chrom=chrom, resolution=resolution,
            output_prefix=output_prefix, dist=dist, cap=cap, pad=pad, gap=gap,
            min_cutoff=min_cutoff, log_e=log_e, shuffle=True,
        )

    try:
        from schicluster_rs._rust import py_loop_bkg_chrom as _rust_loop_bkg
    except ImportError:
        from schicluster.loop.loop_bkg import (
            calculate_chrom_background_normalization as _upstream,
        )
        return _upstream(
            cell_url=cell_url, chrom=chrom, resolution=resolution,
            output_prefix=output_prefix, dist=dist, cap=cap, pad=pad, gap=gap,
            min_cutoff=min_cutoff, log_e=log_e, shuffle=False,
        )

    import cooler
    cool = cooler.Cooler(cell_url)
    matrix = triu(cool.matrix(balance=False, sparse=True).fetch(chrom)).astype(np.float32)
    n = matrix.shape[0]
    coo = matrix.tocoo()
    rows = coo.row.astype(np.uint32, copy=False)
    cols = coo.col.astype(np.uint32, copy=False)
    vals = coo.data.astype(np.float32, copy=False)

    dist_bins = int(dist // resolution)
    (er, ec, ev), (tr, tc, tv) = _rust_loop_bkg(
        rows, cols, vals, int(n), int(dist_bins),
        float(cap), int(pad), int(gap), float(min_cutoff), bool(log_e),
    )
    e_sparse = csr_matrix((ev, (er, ec)), shape=(n, n), dtype=np.float32)
    t_sparse = csr_matrix((tv, (tr, tc)), shape=(n, n), dtype=np.float32)
    save_npz(f"{output_prefix}.E.npz", e_sparse)
    save_npz(f"{output_prefix}.T.npz", t_sparse)
    return


def merge_cells_for_single_chromosome(output_dir, output_prefix, merge_type="E"):
    """Drop-in replacement for
    schicluster.loop.merge_cell_to_group.merge_cells_for_single_chromosome.

    Loads every *.<merge_type>.npz CSR matrix in output_dir, sums via Rust,
    writes the two HDF outputs the upstream emits.
    """
    try:
        from schicluster_rs._rust import py_merge_cells_sum as _rust_merge
    except ImportError:
        from schicluster.loop.merge_cell_to_group import (
            merge_cells_for_single_chromosome as _upstream,
        )
        return _upstream(output_dir=output_dir, output_prefix=output_prefix,
                         merge_type=merge_type)

    import pathlib
    from scipy.sparse import load_npz
    from schicluster.cool import write_coo

    cell_paths = sorted(str(p) for p in pathlib.Path(output_dir).glob(f"*.{merge_type}.npz"))
    if not cell_paths:
        raise FileNotFoundError(f"no *.{merge_type}.npz under {output_dir}")

    cell_ids_list, rows_list, cols_list, vals_list = [], [], [], []
    n_dims = None
    for cell_idx, path in enumerate(cell_paths):
        m = load_npz(path).tocoo()
        if n_dims is None:
            n_dims = m.shape[0]
        if m.nnz == 0:
            continue
        cell_ids_list.append(np.full(m.nnz, cell_idx, dtype=np.uint32))
        rows_list.append(m.row.astype(np.uint32, copy=False))
        cols_list.append(m.col.astype(np.uint32, copy=False))
        vals_list.append(m.data.astype(np.float32, copy=False))

    cell_ids = np.concatenate(cell_ids_list) if cell_ids_list else np.empty(0, np.uint32)
    rows = np.concatenate(rows_list) if rows_list else np.empty(0, np.uint32)
    cols = np.concatenate(cols_list) if cols_list else np.empty(0, np.uint32)
    vals = np.concatenate(vals_list) if vals_list else np.empty(0, np.float32)

    (er, ec, ev), (e2r, e2c, e2v) = _rust_merge(cell_ids, rows, cols, vals)
    e_sum = csr_matrix((ev, (er, ec)), shape=(n_dims, n_dims), dtype=np.float32)
    e2_sum = csr_matrix((e2v, (e2r, e2c)), shape=(n_dims, n_dims), dtype=np.float32)
    write_coo(f"{output_prefix}.{merge_type}.hdf", e_sum, chunk_size=None)
    write_coo(f"{output_prefix}.{merge_type}2.hdf", e2_sum, chunk_size=None)
    return


def loop_background(E, pad, gap, loop):
    """Drop-in replacement for
    schicluster.loop.loop_calling.loop_background.

    Returns 4 arrays (loop_bl, loop_donut, loop_h, loop_v).
    """
    try:
        from schicluster_rs._rust import py_scan_kernels_chrom as _rust_scan
    except ImportError:
        from schicluster.loop.loop_calling import loop_background as _upstream
        return _upstream(E=E, pad=pad, gap=gap, loop=loop)

    e32 = np.ascontiguousarray(E, dtype=np.float32)
    xs = np.ascontiguousarray(loop[0], dtype=np.uint32)
    ys = np.ascontiguousarray(loop[1], dtype=np.uint32)
    bl, donut, h, v = _rust_scan(e32, int(pad), int(gap), xs, ys)
    return bl, donut, h, v


def find_summit(loop, res, dist_thres):
    """Drop-in replacement for schicluster.loop.loop_calling.find_summit.

    Returns the subset of `loop` rows that are peaks, with an added 'size' column.
    """
    try:
        from schicluster_rs._rust import py_find_summit_chrom as _rust_summit
    except ImportError:
        from schicluster.loop.loop_calling import find_summit as _upstream
        return _upstream(loop=loop, res=res, dist_thres=dist_thres)

    import pandas as pd
    loop = loop.copy()
    cord = (loop[["x1", "y1"]].values // res).astype(np.uint32)
    xs = np.ascontiguousarray(cord[:, 0])
    ys = np.ascontiguousarray(cord[:, 1])
    es = np.ascontiguousarray(loop["E"].values, dtype=np.float32)
    idx, sizes = _rust_summit(xs, ys, es, int(dist_thres))
    out = loop.iloc[np.asarray(idx, dtype=np.int64)].copy()
    out["size"] = np.asarray(sizes, dtype=np.int64)
    return out


__all__ = [
    "loop_bkg_chrom",
    "merge_cells_for_single_chromosome",
    "loop_background",
    "find_summit",
]
```

- [ ] **Step 2: Extend `python/schicluster_rs/__init__.py`** to expose the loop wrappers and extend `patch_schicluster()`.

(a) Find the existing `try:` block that imports Rust symbols. Add **two new imports** to that block — no other changes there:

Replace this block:
```python
try:
    from schicluster_rs._rust import (
        py_random_walk_cpu_csr as _rwr_csr,
        py_impute_chromosome_inner as _impute_inner,
        py_convolve2d_mirror as _conv2d_mirror,
        set_num_threads as _set_num_threads,
    )
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
```

With:
```python
try:
    from schicluster_rs._rust import (
        py_random_walk_cpu_csr as _rwr_csr,
        py_impute_chromosome_inner as _impute_inner,
        py_convolve2d_mirror as _conv2d_mirror,
        py_loop_bkg_chrom as _loop_bkg_chrom,
        py_merge_cells_sum as _merge_cells_sum,
        py_scan_kernels_chrom as _scan_kernels_chrom,
        py_find_summit_chrom as _find_summit_chrom,
        set_num_threads as _set_num_threads,
    )
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
```

(b) After the existing `def convolve2d_mirror(...)` function, **before** `def patch_schicluster()`, add a single import:

```python
from schicluster_rs.loop import (
    loop_bkg_chrom,
    merge_cells_for_single_chromosome,
    loop_background as _loop_background,
    find_summit as _find_summit,
)
```

(c) Extend the `def patch_schicluster() -> bool:` body to also rebind upstream loop functions. Replace the existing body:

```python
    if not _RUST_AVAILABLE:
        return False
    try:
        from schicluster.impute import impute_chromosome as _mod
        _mod.random_walk_cpu = random_walk_cpu
        _mod.impute_chromosome = impute_chromosome
        return True
    except ImportError:
        return False
```

With:
```python
    if not _RUST_AVAILABLE:
        return False
    try:
        from schicluster.impute import impute_chromosome as _impute_mod
        _impute_mod.random_walk_cpu = random_walk_cpu
        _impute_mod.impute_chromosome = impute_chromosome
        # ---- loop module ----
        from schicluster.loop import loop_bkg as _loop_bkg_mod
        from schicluster.loop import merge_cell_to_group as _merge_mod
        from schicluster.loop import loop_calling as _loop_calling_mod
        _loop_bkg_mod.calculate_chrom_background_normalization = loop_bkg_chrom
        _merge_mod.merge_cells_for_single_chromosome = merge_cells_for_single_chromosome
        _loop_calling_mod.loop_background = _loop_background
        _loop_calling_mod.find_summit = _find_summit
        return True
    except ImportError:
        return False
```

(d) Extend `__all__` to include the four new public names:

Replace:
```python
__all__ = [
    "random_walk_cpu", "impute_chromosome", "patch_schicluster",
    "set_num_threads", "convolve2d_mirror",
]
```

With:
```python
__all__ = [
    "random_walk_cpu", "impute_chromosome", "patch_schicluster",
    "set_num_threads", "convolve2d_mirror",
    "loop_bkg_chrom", "merge_cells_for_single_chromosome",
    "loop_background", "find_summit",
]
```

And add right after the new import block in (b):
```python
# Re-exports for the public API surface declared in __all__.
loop_background = _loop_background
find_summit = _find_summit
```

- [ ] **Step 3: Smoke-test the wrappers import:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
import schicluster_rs as P
print('loop_bkg_chrom:', callable(P.loop_bkg_chrom))
print('merge_cells_for_single_chromosome:', callable(P.merge_cells_for_single_chromosome))
print('loop_background:', callable(P.loop_background))
print('find_summit:', callable(P.find_summit))
print('all in __all__:', all(n in P.__all__ for n in (
    'loop_bkg_chrom','merge_cells_for_single_chromosome','loop_background','find_summit')))
"
```

Expected: all four `True`.

**No commit in this task.**

---

## Task 8: Extend reference + candidate drivers + run the parity gate

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/tests/py_reference_driver.py`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/tests/_run_candidate.py`

Both drivers grow new functions that read the `loop_small.npz` fixture (+ the `loop_small.cool` cooler for `loop_bkg`) and emit the 10 manifest output keys. The reference driver also fixes the housekeeping comment / fresh-scipy recomputation nit raised by the Phase 0 reviewer.

- [ ] **Step 1: Replace `tests/py_reference_driver.py`** (Write tool — full rewrite):

```python
"""Run upstream Python references and dump JSON.

Invoked under $PYTHON_REF_ENV (schicluster env, py 3.6). Emits one JSON
block per manifest output that has a reference for it; later phases
extend this driver.

py3.6 NOTE: do NOT add `from __future__ import annotations` (added in
3.7) or PEP 585 subscripted generics — schicluster env is Python 3.6.

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
    """Return a dict of arrays loaded from an npz."""
    with np.load(str(path)) as z:
        return {k: z[k] for k in z.files}


# ----- conv.convolved (Phase 0; recompute fresh via scipy to address Phase 0 nit) -----

def ref_conv_convolved(conv):
    a = conv["input"]
    k = conv["kernel"]
    return convolve(a, k, mode="mirror").astype(np.float32).tolist()


# ----- loop_bkg.{E,T} -----

def ref_loop_bkg(temp_dir):
    """Call upstream calculate_chrom_background_normalization against the
    fixture cooler; load the resulting .E.npz / .T.npz; return triplets.
    """
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


# ----- merge.{e_sum,e2_sum} -----

def ref_merge(loop_pack, temp_dir):
    """Save the per-cell synthetic CSRs as *.E.npz under temp_dir, then call
    upstream merge_cells_for_single_chromosome and parse the HDF outputs.
    """
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
    # Upstream writes HDF; read back via pandas.HDFStore as a key-named dataframe.
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


# ----- scan_kernels.{bl,donut,h,v} -----

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


# ----- find_summit.{idx,sizes} -----

def ref_find_summit(loop_pack):
    x1 = loop_pack["summit.x1"]
    y1 = loop_pack["summit.y1"]
    es = loop_pack["summit.E"]
    n = x1.size
    df = pd.DataFrame({
        "x1": x1, "y1": y1, "E": es,
        # Upstream find_summit reads loop['E']; we don't need the other cols.
    })
    summit_df = upstream_find_summit(loop=df, res=LOOP_RESOLUTION,
                                     dist_thres=LOOP_SUMMIT_DIST_BINS)
    # find_summit returns rows of the original loop df with an added 'size' column.
    # The manifest output is set-of-original-indices + cluster sizes aligned to those.
    selected = summit_df.index.to_numpy().astype(np.uint32)
    sizes = summit_df["size"].to_numpy().astype(np.uint32)
    return {
        "idx": selected.tolist(),
        "sizes": sizes.tolist(),
    }


def main():
    conv = _load_npz(CONV_FIXTURE)
    loop_pack = _load_npz(LOOP_FIXTURE)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "convolved": ref_conv_convolved(conv),
    }

    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="schicluster_rs_ref_"))
    try:
        e, t = ref_loop_bkg(temp_dir)
        payload["loop_bkg"] = {"E": e, "T": t}

        e_sum, e2_sum = ref_merge(loop_pack, temp_dir)
        payload["merge"] = {"e_sum": e_sum, "e2_sum": e2_sum}

        payload["scan_kernels"] = ref_scan_kernels(loop_pack)
        payload["find_summit"] = ref_find_summit(loop_pack)
    finally:
        shutil.rmtree(str(temp_dir), ignore_errors=True)

    OUT.write_text(json.dumps(payload))
    print("wrote {} (top-level keys: {})".format(OUT, sorted(payload.keys())))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Replace `tests/_run_candidate.py`** (Write tool — full rewrite):

```python
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
from scipy.sparse import csr_matrix, load_npz, save_npz

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


def _load_npz(path: pathlib.Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ----- conv.convolved (Phase 0) -----

def cand_conv_convolved(conv: dict) -> list:
    a = np.ascontiguousarray(conv["input"], dtype=np.float32)
    k = np.ascontiguousarray(conv["kernel"], dtype=np.float32)
    return np.asarray(schicluster_rs.convolve2d_mirror(a, k), dtype=np.float32).tolist()


# ----- loop_bkg.{E,T} -----

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


# ----- merge.{e_sum,e2_sum} -----

def cand_merge(loop_pack: dict, temp_dir: pathlib.Path):
    cell_ids = loop_pack["merge.cell_ids"]
    rows = loop_pack["merge.input.rows"]
    cols = loop_pack["merge.input.cols"]
    vals = loop_pack["merge.input.vals"]
    n = LOOP_N_BINS
    for cid in sorted(np.unique(cell_ids).tolist()):
        mask = cell_ids == cid
        m = csr_matrix(
            (vals[mask], (rows[mask], cols[mask])), shape=(n, n), dtype=np.float32
        )
        save_npz(str(temp_dir / f"cell{cid}.E.npz"), m)
    out_prefix = str(temp_dir / "merge_small")
    schicluster_rs.merge_cells_for_single_chromosome(
        output_dir=str(temp_dir), output_prefix=out_prefix, merge_type="E",
    )
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


# ----- scan_kernels.{bl,donut,h,v} -----

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


# ----- find_summit.{idx,sizes} -----

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
    return {
        "idx": selected.tolist(),
        "sizes": sizes.tolist(),
    }


def main() -> None:
    conv = _load_npz(CONV_FIXTURE)
    loop_pack = _load_npz(LOOP_FIXTURE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "convolved": cand_conv_convolved(conv),
    }
    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="schicluster_rs_cand_"))
    try:
        e, t = cand_loop_bkg(temp_dir)
        payload["loop_bkg"] = {"E": e, "T": t}

        e_sum, e2_sum = cand_merge(loop_pack, temp_dir)
        payload["merge"] = {"e_sum": e_sum, "e2_sum": e2_sum}

        payload["scan_kernels"] = cand_scan_kernels(loop_pack)
        payload["find_summit"] = cand_find_summit(loop_pack)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT} (top-level keys: {sorted(payload.keys())})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update the location pointers in `data/manifest.yaml`** for nested outputs so the harness's `_dig` resolves them correctly.

The parity_harness uses `$.<a>.<b>.<c>` to navigate nested dicts. The current manifest entries for `loop_bkg.E` etc. use `"$.loop_bkg.E"` — that's right; the driver emits `{"loop_bkg": {"E": {...}}, ...}` so `$.loop_bkg.E` resolves correctly. **No manifest edit needed.**

Verify quickly with a one-off:

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
import sys
sys.path.insert(0, str('/large_storage/zhoulab/shengmao/rebuildpy'))
sys.path.insert(0, str('/large_storage/zhoulab/shengmao/rust-scHiCluster/tests'))
from parity_harness import load_outputs, _dig
sample = {'loop_bkg': {'E': {'rows': [1,2], 'cols': [3,4], 'vals': [0.1, 0.2]}}}
print(_dig(sample, '\$.loop_bkg.E') is not None)
"
```

Expected: `True`.

- [ ] **Step 4: But — the harness compares JSON-deserialised values to `compute_parity`, which expects either arrays or simple dicts. Inspect a representative nested case to confirm the metric is computed sensibly.** For loop_bkg.E the value is `{"rows":..., "cols":..., "vals":...}` — a dict, not an array. `engine.parity_metrics.parity_deterministic` likely won't know how to handle that.

Read the `engine.parity_metrics` source quickly to check:

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
import inspect, sys
sys.path.insert(0, '/large_storage/zhoulab/shengmao/rebuildpy')
from engine import parity_metrics as pm
print(inspect.getsource(pm.parity_deterministic))
" | head -40
```

If `parity_deterministic` only handles numpy arrays, we **must** flatten the dict into a single dense array before comparison. The Phase 0 harness already calls `_to_numpy(obj)` which short-circuits on dicts. Fix: extend `parity_harness._to_numpy` to densify CSR-triplet-shaped dicts into a dense `(n, n)` matrix using `LOOP_N_BINS`.

Modify `tests/parity_harness.py` (Edit tool) — replace the `_to_numpy` function:

```python
def _to_numpy(obj: Any) -> Any:
    """Re-hydrate a JSON-dumped reference / candidate object into a numpy form
    that engine.parity_metrics knows how to compare.

    For CSR-triplet dicts {"rows": [...], "cols": [...], "vals": [...]} we
    densify into a square matrix using max(rows ∪ cols) + 1 as the side
    length (small fixtures only). Plain lists become np.ndarray. Everything
    else passes through.
    """
    if isinstance(obj, dict) and set(obj.keys()) >= {"rows", "cols", "vals"}:
        rows = np.asarray(obj["rows"], dtype=np.int64)
        cols = np.asarray(obj["cols"], dtype=np.int64)
        vals = np.asarray(obj["vals"], dtype=np.float64)
        if rows.size == 0:
            return np.zeros((1, 1), dtype=np.float64)
        n = int(max(rows.max(), cols.max())) + 1
        dense = np.zeros((n, n), dtype=np.float64)
        dense[rows, cols] = vals
        return dense
    if isinstance(obj, list):
        try:
            return np.asarray(obj)
        except ValueError:
            return obj
    return obj
```

- [ ] **Step 5: Make the harness emit a sensible metric for the `ranked` and `classification` outputs** (`find_summit.idx` set-Jaccard; `find_summit.sizes` agreement on the matched intersection). The default `engine.parity_metrics` `ranked` / `classification` may not accept the JSON shape — read the source to confirm. If `parity_ranked` expects two 1-D integer arrays, we're fine because `_to_numpy` on a list returns `np.asarray(list)`. If `parity_classification` expects two arrays of equal length, we need to align `sizes` arrays first.

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
import inspect, sys
sys.path.insert(0, '/large_storage/zhoulab/shengmao/rebuildpy')
from engine import parity_metrics as pm
print('--- ranked ---'); print(inspect.getsource(pm.parity_ranked))
print('--- classification ---'); print(inspect.getsource(pm.parity_classification))
"
```

If `parity_classification` requires equal-length inputs, the implementer extends `_to_numpy` (or the harness `evaluate`) for the special-case `find_summit.sizes` output: align reference and candidate by their shared `idx` intersection, slice both `sizes` arrays to that intersection, and pass the two aligned arrays.

A clean way to do this: add a small helper in `parity_harness.py` and call it in `evaluate` when the output `name` ends with `.sizes`. Read the actual `parity_classification` source first; the simplest fix may be acceptable as-is if it tolerates ragged inputs.

- [ ] **Step 6: Run the full parity gate via the orchestrator:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV bash tests/run_parity.sh 2>&1 | tail -30
```

Expected: ends with `11 passed, 6 skipped`. (1 conv from Phase 0 + 10 new loop outputs; the 6 skips are domain × 3 + compartment × 2 + embedding × 1.)

If any of the 10 new outputs fail, diagnose with the rebuildpy fail-suspicion list:
- **`loop_bkg.E` mismatch**: check `percentile_linear` matches `np.percentile` default; check `zscore_ddof0` matches `scipy.stats.zscore(ddof=0)`; verify the diagonal write-back uses `min(zscored)` for non-positives, not `0`.
- **`loop_bkg.T` mismatch**: check the donut kernel is built identically (1 everywhere except inner block, normalised); confirm the upper-tri mask is applied *after* the convolution but *before* the min_cutoff filter, and that the subtraction order is `T = E - T` (not `T - E`).
- **`merge.e_sum / e2_sum` mismatch**: BTreeMap iteration is row-major — confirm that. Check f64 accumulation cast to f32 matches scipy's behaviour (it should, both promote intermediate sums).
- **`scan_kernels.{bl,donut,h,v}` mismatch on the asymmetric kernels**: this stresses the convolve kernel-flip. The Phase-0 `convolve2d_mirror` was validated only on a symmetric kernel; if `scan_kernels.bl` fails but `scan_kernels.donut` passes, the kernel-flip indexing in `convolve2d_mirror` may have a bug that the symmetric Phase 0 fixture didn't catch. Fix in `conv.rs`, not by widening the gate.
- **`find_summit.idx`** failing the Jaccard ≥ 0.99: equal-E tie-break order differs; verify the deterministic ascending-idx tie-break.
- **`find_summit.sizes`** mismatch over the intersection: usually indicates a downstream side effect of the previous bug; fix find_summit first.

**Do not widen any manifest threshold.** Fix the source code. Re-run.

**No commit in this task.**

---

## Task 9: Single Phase-1 commit

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/Cargo.toml`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/pyproject.toml`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/ITERATION_LOG.md`

- [ ] **Step 1: Bump the version `0.2.0-dev0` → `0.2.0` in `rust/Cargo.toml`:**

Change line 3 from `version = "0.2.0-dev0"` to `version = "0.2.0"`.

- [ ] **Step 2: Bump `pyproject.toml`:**

Change `version = "0.2.0.dev0"` to `version = "0.2.0"`.

- [ ] **Step 3: Append the Phase 1 iteration block to `ITERATION_LOG.md`.** After the existing Iteration 0 block (and its `---` divider), append:

```yaml
iteration: 1
title: Phase 1 — loop module ported (loop_bkg / merge / scan_kernels / find_summit)
admissibility: E
action: |
  Per-chrom whole-function Rust ports of:
    - calculate_chrom_background_normalization (loop_bkg.rs)
    - merge_cells_for_single_chromosome      (merge.rs)
    - loop_background                        (scan_kernels.rs)
    - find_summit                            (find_summit.rs)
  All five convolutions (1 in loop_bkg, 4 in scan_kernels) reuse the
  Phase 0 convolve2d_mirror primitive. merge accumulates in f64 BTreeMap
  for deterministic ordering and emits row-major triplets. find_summit
  uses Rust BinaryHeap with deterministic ascending-idx tie-break.
status: accepted
fixture: data/fixtures/loop_small.npz + data/fixtures/loop_small.cool
parity:
  loop_bkg.E:        { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  loop_bkg.T:        { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  merge.e_sum:       { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  merge.e2_sum:      { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  scan_kernels.bl:   { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  scan_kernels.donut:{ class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  scan_kernels.h:    { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  scan_kernels.v:    { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  find_summit.idx:   { class: ranked, threshold: 0.99, pass: true }
  find_summit.sizes: { class: classification, threshold: 1.0, pass: true }
notes: |
  shuffle=True path of loop_bkg falls back to upstream Python (RNG; out of
  parity scope per spec §11). Phase 0 nit on run_parity.sh comment +
  py_reference_driver fresh-scipy recomputation are addressed by the
  Phase 1 driver rewrite.

---
```

- [ ] **Step 4: Rebuild and re-run the gate one last time to confirm clean state:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output maturin develop --release 2>&1 | tail -5
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    python -m pytest -q tests/test_exact_match.py 2>&1 | tail -3
```

Expected: install line says `schicluster-rs-0.2.0`; pytest reports `11 passed, 6 skipped`.

- [ ] **Step 5: Commit everything as one Phase 1 commit.**

Confirm the staged diff first (this should include the new Rust files, the lib.rs registrations, the Python wrappers + __init__.py edits, the synthesize.py extensions, the driver rewrites, the .gitignore additions, the harness `_to_numpy` change, the Cargo.toml / pyproject.toml bumps, and the ITERATION_LOG append):

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
git status --short
git diff --stat
```

Then add and commit:

```bash
git add \
    rust/Cargo.toml \
    rust/src/lib.rs \
    rust/src/loop_bkg.rs \
    rust/src/merge.rs \
    rust/src/scan_kernels.rs \
    rust/src/find_summit.rs \
    python/schicluster_rs/__init__.py \
    python/schicluster_rs/loop.py \
    pyproject.toml \
    data/fixtures/synthesize.py \
    tests/parity_harness.py \
    tests/py_reference_driver.py \
    tests/_run_candidate.py \
    ITERATION_LOG.md \
    .gitignore
git commit -m "$(cat <<'EOF'
feat: Phase 1 loop module ported to Rust (10 manifest outputs green)

Ports the per-chrom numerical hot paths of scHiCluster's loop module:
  - calculate_chrom_background_normalization → rust/src/loop_bkg.rs
  - merge_cells_for_single_chromosome        → rust/src/merge.rs
  - loop_background                          → rust/src/scan_kernels.rs
  - find_summit                              → rust/src/find_summit.rs

All five convolutions reuse Phase 0's convolve2d_mirror primitive. merge
accumulates in f64 BTreeMap for deterministic row-major emission. The
find_summit max-heap uses ascending-idx tie-break.

Python wrappers in python/schicluster_rs/loop.py keep cooler / .npz /
HDF5 I/O in Python; patch_schicluster() now rebinds the upstream loop
module functions so every caller (CLI, snakemake, hicluster.domain) gets
the Rust kernels transparently.

Parity gate (data/manifest.yaml read-only, 11 of 17 outputs now green):
  conv.convolved          deterministic-bounded 1e-6 (Phase 0)
  loop_bkg.E              deterministic-bounded 1e-6
  loop_bkg.T              deterministic-bounded 1e-6
  merge.e_sum             deterministic-bounded 1e-6
  merge.e2_sum            deterministic-bounded 1e-6
  scan_kernels.bl/donut/h/v deterministic-bounded 1e-6
  find_summit.idx         ranked Jaccard ≥ 0.99
  find_summit.sizes       classification = 1.0
Skips (6): insulation + topdom × 2 (Phase 2), compartment × 2 (Phase 3),
embedding × 1 (Phase 4).

Version 0.2.0.dev0 → 0.2.0.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
git log --oneline -3
```

Expected: one new commit at HEAD containing all Phase 1 changes; second line shows the Phase 0 close at `27373f4`.

---

## Self-review

**1. Spec coverage.** Spec §5 rows 2–11 (loop) are covered by Tasks 2–8 (10 outputs). Spec §6 loop block is the surface Tasks 2–5 implement. Spec §7 (E) acceleration baseline is what Task 9 records as Iteration 1. Spec §9 phase 1 ("loop_call.rs + python/schicluster_rs/loop.py + monkey-patch hooks") matches Tasks 5–7. Spec §10 deliverables for Phase 1 (iteration log update, version bump) are Task 9. Spec §11 "shuffle path excluded from gate" is honored by the wrapper falling back to upstream when shuffle=True. **No spec gap.**

**2. Placeholder scan.** No `TBD` / `TODO` / `implement later` / `similar to Task N` text appears. The few `<...>` placeholders in commit messages and iteration-log timing fields are intentional fill-ins for runtime-measured values, the way Phase 0's Task 12 handled them.

**3. Type consistency.** `loop_bkg_chrom`, `merge_cells_sum`, `scan_kernels_chrom`, `find_summit_chrom` are referenced consistently between `rust/src/*.rs`, `lib.rs` registrations, `python/schicluster_rs/loop.py` wrappers, `__init__.py` re-exports, and the two driver scripts. The PyO3 binding names follow the `py_<rust_fn>` convention from Phase 0. Manifest output paths (`$.loop_bkg.E` etc.) align with the JSON structure both drivers emit. The `_to_numpy` densification matches the CSR-triplet schema both drivers use for sparse outputs.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-03-rust-port-phase-1-loop-module.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
