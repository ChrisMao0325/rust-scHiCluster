# scHiCluster Rust port — Phase 2 (domain module + native TopDom)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit discipline:** per user direction, this plan ships as **one commit at the end** (Task 8). Subagent implementers in earlier tasks must NOT run `git commit`. They write code, run smoke checks, leave changes staged or unstaged for the controller to assemble the single Phase 2 commit. The final task is the only commit step.

**Goal:** Turn 3 of the remaining 6 pre-registered parity-manifest outputs green by porting the `single_chrom_calculate_insulation_score` numerical loop and a **fully native TopDom** (replacing the `rpy2`/`R` round-trip) to Rust.

**Architecture:** One Rust file per concern — `rust/src/insulation.rs` for the sliding-window insulation score (f64 sums, cast to f32 on emit), `rust/src/topdom.rs` for the full TopDom port. TopDom mirrors the upstream R structure: diamond mean signal → gap-region detection → change-point / local-extreme detection → R-compatible Wilcoxon rank-sum p-values (normal approximation with continuity + tie correction) → bin→domain BED conversion. A new `python/schicluster_rs/domain/` subpackage holds the wrappers; `patch_schicluster()` rebinds the two upstream functions. The parity harness gains two `topdom.bed.*` dispatch helpers (set-Jaccard on domain intervals, classification on per-bin tag labels) since both gates read the same JSON node.

**Tech Stack:** Rust 1.95 + PyO3 0.22 + ndarray 0.16 + sprs 0.11 (existing); no new crate dependencies — Wilcoxon's normal CDF uses an inline Abramowitz approximation of `erf` (~1.5e-7 accuracy, plenty for `p < 0.05` thresholds).

---

## Phase 2 outputs (already pre-registered in `data/manifest.yaml` — read-only)

| Manifest output | Algorithm class | Threshold | Upstream Python / R |
|---|---|---|---|
| `insulation.score` | deterministic-bounded | 1e-6 | `domain/call_domain.py::single_chrom_calculate_insulation_score` (sliding-window submatrix sums on f64) |
| `topdom.bed.interval_jaccard` | ranked (Jaccard on domain intervals) | 0.95 | `domain/TopDom.R::RunTopDom` via rpy2 |
| `topdom.bed.bin_label_agreement` | classification (per-bin tag labels) | 0.98 | same |

Phase 2 is complete when `pytest -q tests/test_exact_match.py` reports `14 passed, 3 skipped` (1 conv from Phase 0 + 10 loop from Phase 1 + 3 domain). The 3 skips are `compartment.comp`, `compartment.strength` (Phase 3) and `embedding.cell_by_feature` (Phase 4).

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `data/fixtures/synthesize.py` | MODIFY | Adds `domain_small_fixture` — synthetic dense Hi-C matrix with 4 planted blocks at n=80, the shared input for both insulation and TopDom. |
| `data/fixtures/domain_small.npz` | NEW (generated, .gitignored) | Packed: `topdom.matrix` (80×80 f32 symmetric), `topdom.window_size`, `insulation.window_size`. |
| `rust/src/insulation.rs` | NEW | `insulation_score_chrom` — sliding-window submatrix sums; f64 accumulators cast to f32 on emit. |
| `rust/src/topdom.rs` | NEW | Full TopDom port: diamond signal, gap regions, change-point, local-extreme detection, Wilcoxon ranksum p-values (normal-approx + continuity + tie correction), bin→domain BED. |
| `rust/src/lib.rs` | MODIFY | Mounts the two new modules; registers two new PyO3 functions in `#[pymodule] _rust`. |
| `python/schicluster_rs/domain/__init__.py` | NEW | Python wrappers around the Rust kernels; preserves upstream signatures so `patch_schicluster()` can hot-swap. |
| `python/schicluster_rs/__init__.py` | MODIFY | Imports the two new Rust symbols + the domain wrappers; extends `__all__`; extends `patch_schicluster()` to rebind upstream `single_chrom_calculate_insulation_score` and the rpy2 `run_top_dom` helper (so `call_domain_and_insulation` transparently uses the Rust kernels). |
| `tests/py_reference_driver.py` | MODIFY | Adds 3 reference computations (calls upstream Python: `single_chrom_calculate_insulation_score` directly + the rpy2 path for TopDom). |
| `tests/_run_candidate.py` | MODIFY | Adds 3 candidate computations (calls Rust on the same fixture). |
| `tests/parity_harness.py` | MODIFY | Adds two custom dispatches for `topdom.bed.interval_jaccard` (set-Jaccard on domain `(start, end)` tuples) and `topdom.bed.bin_label_agreement` (per-bin tag labels after coordinate alignment). |
| `rust/Cargo.toml` | MODIFY (Task 8 only) | Version `0.2.0` → `0.3.0`. |
| `pyproject.toml` | MODIFY (Task 8 only) | Version `0.2.0` → `0.3.0`. |
| `docs/ITERATION_LOG.md` | MODIFY (Task 8 only) | Append Phase 2 iteration block. |

---

## Task 1: Phase 2 fixture — `domain_small.npz`

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/synthesize.py`

A synthetic Hi-C matrix with 4 planted 20-bin blocks gives TopDom enough structure to actually find domains, and exercises insulation's both branches (`i < w` and `i >= w`).

- [ ] **Step 1: Extend `synthesize.py`.** Add the following near the existing Phase 1 fixture functions (e.g., after `loop_small_packed_fixture`):

```python
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
```

Extend `main()` to also write `data/fixtures/domain_small.npz`:

```python
    # ---- domain_small (Phase 2) ----
    domain_pack = domain_small_fixture()
    np.savez(FIXTURE_DIR / "domain_small.npz", **domain_pack)
    print(f"wrote {FIXTURE_DIR / 'domain_small.npz'} ({len(domain_pack)} keys)")
    print(f"  matrix.shape    = {domain_pack['topdom.matrix'].shape}, "
          f"dtype = {domain_pack['topdom.matrix'].dtype}")
    print(f"  window_size     = {int(domain_pack['topdom.window_size'])}")
```

- [ ] **Step 2: Add the new artefact to `.gitignore`:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
grep -qxF "data/fixtures/domain_small.npz" .gitignore || echo "data/fixtures/domain_small.npz" >> .gitignore
```

- [ ] **Step 3: Regenerate fixtures in `rebuild-rust` and confirm shape:**

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    python data/fixtures/synthesize.py
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
import numpy as np, pathlib
d = np.load(pathlib.Path('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/domain_small.npz'))
print('keys:', sorted(d.files))
m = d['topdom.matrix']
print('matrix:', m.shape, m.dtype, 'symmetric:', np.allclose(m, m.T), 'diag zero:', np.all(np.diag(m) == 0))
print('window_size:', int(d['topdom.window_size']))
"
```

Expected: keys `['insulation.window_size', 'topdom.matrix', 'topdom.window_size']`, matrix `(80, 80) float32`, symmetric True, diag zero True, window_size 5.

- [ ] **Step 4: Verify the fixture is also loadable in `schicluster` (py3.6):**

```bash
env -u VIRTUAL_ENV conda run -n schicluster --no-capture-output python -c "
import numpy as np
d = np.load('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/domain_small.npz')
print('schicluster-env load ok; matrix shape =', d['topdom.matrix'].shape)
"
```

Expected: prints `matrix shape = (80, 80)`. If load fails (np.savez format compat), report DONE_WITH_CONCERNS — numpy .npz is broadly backwards-compatible so this should just work.

**No commit in this task.**

---

## Task 2: Rust `insulation.rs`

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/insulation.rs`

Ports `scHiCluster/schicluster/domain/call_domain.py::single_chrom_calculate_insulation_score`. The upstream loop computes for each row `i` an intra-block sum and an inter-block sum, with two branches (`i < w` and `i >= w`). The Rust port mirrors the branches exactly and accumulates in f64 to dodge f32 reduction-order drift.

`save_count=True` returns `[inter, intra]` per row; the Phase 2 manifest output is the `score[i] = inter / (inter + intra)` form (save_count=False). The Rust kernel exposes both paths via a boolean param.

- [ ] **Step 1: Write `rust/src/insulation.rs`** (Write tool, exact content):

```rust
//! Port of scHiCluster/schicluster/domain/call_domain.py::single_chrom_calculate_insulation_score.
//!
//! For each row `i`:
//!   intra = (mat[a:i, a:i].sum() + mat[i:i+w, i:i+w].sum()) / area
//!   inter = mat[a:i, i:i+w].sum() / area
//! with `a = max(0, i-w)` and area constants matching upstream's two branches:
//!   i < w  : intra denom = i(i+1)/2 + w(w+1)/2  ; inter denom = i * (i + w)
//!   i >= w : intra denom = w(w+1)              ; inter denom = w * w
//!
//! score[0] = 1.0 (initial); score[i] = inter / (inter + intra) for i >= 1.
//! save_count=True returns a (n, 2) array with [inter, intra] per row.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

fn block_sum(m: &Array2<f32>, r0: usize, r1: usize, c0: usize, c1: usize) -> f64 {
    let mut acc = 0.0_f64;
    for r in r0..r1 {
        for c in c0..c1 {
            acc += m[(r, c)] as f64;
        }
    }
    acc
}

pub fn insulation_score_chrom(matrix: &Array2<f32>, w: usize) -> Vec<f32> {
    let n = matrix.nrows();
    let mut score = vec![1.0_f32; n];
    for i in 1..n {
        let (intra, inter) = if i < w {
            let intra_num =
                block_sum(matrix, 0, i, 0, i) + block_sum(matrix, i, (i + w).min(n), i, (i + w).min(n));
            let intra_denom =
                (i * (i + 1)) as f64 / 2.0 + (w * (w + 1)) as f64 / 2.0;
            let inter_num = block_sum(matrix, 0, i, i, (i + w).min(n));
            let inter_denom = (i * (i + w)) as f64;
            (intra_num / intra_denom, inter_num / inter_denom)
        } else {
            let intra_num = block_sum(matrix, i - w, i, i - w, i)
                + block_sum(matrix, i, (i + w).min(n), i, (i + w).min(n));
            let intra_denom = (w * (w + 1)) as f64;
            let inter_num = block_sum(matrix, i - w, i, i, (i + w).min(n));
            let inter_denom = (w * w) as f64;
            (intra_num / intra_denom, inter_num / inter_denom)
        };
        let denom = inter + intra;
        score[i] = if denom > 0.0 { (inter / denom) as f32 } else { 1.0 };
    }
    score
}

pub fn insulation_score_chrom_save_count(matrix: &Array2<f32>, w: usize) -> Vec<[f32; 2]> {
    let n = matrix.nrows();
    let mut out = vec![[1.0_f32, 1.0_f32]; n];
    for i in 1..n {
        let (intra, inter) = if i < w {
            let intra_num =
                block_sum(matrix, 0, i, 0, i) + block_sum(matrix, i, (i + w).min(n), i, (i + w).min(n));
            let intra_denom =
                (i * (i + 1)) as f64 / 2.0 + (w * (w + 1)) as f64 / 2.0;
            let inter_num = block_sum(matrix, 0, i, i, (i + w).min(n));
            let inter_denom = (i * (i + w)) as f64;
            (intra_num / intra_denom, inter_num / inter_denom)
        } else {
            let intra_num = block_sum(matrix, i - w, i, i - w, i)
                + block_sum(matrix, i, (i + w).min(n), i, (i + w).min(n));
            let intra_denom = (w * (w + 1)) as f64;
            let inter_num = block_sum(matrix, i - w, i, i, (i + w).min(n));
            let inter_denom = (w * w) as f64;
            (intra_num / intra_denom, inter_num / inter_denom)
        };
        out[i] = [inter as f32, intra as f32];
    }
    out
}

#[pyfunction]
#[pyo3(signature = (matrix, window_size, save_count=false))]
pub fn py_insulation_score_chrom<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f32>,
    window_size: usize,
    save_count: bool,
) -> PyResult<PyObject> {
    let m = matrix.as_array().to_owned();
    if save_count {
        let rows = py.allow_threads(|| insulation_score_chrom_save_count(&m, window_size));
        let n = rows.len();
        let mut flat = Vec::with_capacity(n * 2);
        for [inter, intra] in rows {
            flat.push(inter);
            flat.push(intra);
        }
        let arr = ndarray::Array2::from_shape_vec((n, 2), flat).expect("shape");
        Ok(arr.into_pyarray_bound(py).into_py(py))
    } else {
        let v = py.allow_threads(|| insulation_score_chrom(&m, window_size));
        let arr = ndarray::Array1::from(v);
        Ok(arr.into_pyarray_bound(py).into_py(py))
    }
}

// silence the unused-import lint that PyArray2/PyArray1 produce when only one branch is exercised
#[allow(dead_code)]
fn _unused_imports_anchor(_a: &PyArray1<f32>, _b: &PyArray2<f32>) {}
```

- [ ] **Step 2: Syntax check via cargo check** (module not yet mod-mounted in `lib.rs`):

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster/rust
cargo check --quiet 2>&1 | tail -5
```

Expected: clean (pre-existing warnings in lib.rs are unrelated).

**No commit in this task.**

---

## Task 3: Rust `topdom.rs` (full TopDom port)

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/topdom.rs`

This is the largest task in Phase 2. The R original (~600 LoC of TopDom.R) collapses into ~500 LoC of Rust. The port preserves every subtle behaviour of the upstream — including the `NaN` for `mean.cf[n-1]` from `Get.Diamond.Matrix` returning `NA` at the last bin, the 1-indexed off-by-ones in `Which.process.region`, and the **continuity + tie correction** in `wilcox.test(exact=FALSE, alternative="less")`. The parity gate uses `ranked` (Jaccard) + `classification` rather than elementwise so small `p-value < 0.05` drifts near borderline bins don't false-fail.

The structure follows the R original exactly so a reviewer can diff side-by-side:

- `erf` + `pnorm` (inline Abramowitz, no new dep)
- `ranks_average` (R's `rank(..., ties.method="average")`)
- `wilcoxon_ranksum_less` (R's `wilcox.test(x, y, alternative="less", exact=FALSE, correct=TRUE)`)
- `diamond_mean` (R's `Get.Diamond.Matrix` + `mean`)
- `which_gap_region2` (R's `Which.Gap.Region2`)
- `which_process_region` (R's `Which.process.region`)
- `data_norm` (R's `Data.Norm`)
- `change_point` (R's `Change.Point` — the Fv/Ev fixed-point algorithm)
- `detect_local_extreme` (R's `Detect.Local.Extreme`)
- `get_diamond_matrix2` + `get_upstream_triangle` + `get_downstream_triangle` (R helpers used by `Get.Pvalue`)
- `get_pvalue` (R's `Get.Pvalue` — per-bin Wilcoxon)
- `convert_bin_to_domain` (R's `Convert.Bin.To.Domain.TMP`)
- `topdom_chrom` (R's `TopDom` end-to-end orchestrator)

- [ ] **Step 1: Write `rust/src/topdom.rs`** (Write tool). Because this file is large the implementer should paste it verbatim from this plan rather than transcribing. Exact content:

```rust
//! Native port of scHiCluster/schicluster/domain/TopDom.R::TopDom.
//!
//! Replaces the rpy2 -> R round-trip with a self-contained Rust implementation.
//! Structure mirrors the R original 1-to-1 (function names + comments preserved
//! so a reviewer can diff side-by-side).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

const PVALUE_CUT: f64 = 0.05;

// ----- Normal CDF (Abramowitz 7.1.26 erf, error ~1.5e-7) ----------------------

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    sign * y
}

fn pnorm(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

// ----- Average ranks with ties (matches R's rank(ties.method="average")) -----

fn ranks_average(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[order[j]] == values[order[i]] {
            j += 1;
        }
        // ranks are 1-indexed; average of (i+1)..=j
        let avg = (i as f64 + 1.0 + j as f64) / 2.0;
        for k in i..j {
            ranks[order[k]] = avg;
        }
        i = j;
    }
    ranks
}

// ----- R's wilcox.test(x, y, alternative="less", exact=FALSE, correct=TRUE) --
//
// W = sum_of_ranks(x in combined) - n_x*(n_x+1)/2
// E[W] = n_x * n_y / 2
// Var[W] = n_x*n_y*(n_x+n_y+1)/12  - n_x*n_y/(12*(n_x+n_y)*(n_x+n_y-1)) * sum(t_i^3 - t_i)
// For "less" with continuity correction:
//   z = (W - E[W] + 0.5) / sqrt(Var[W])
//   p = pnorm(z)

fn wilcoxon_ranksum_less(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 1.0;
    }
    let n_x = x.len() as f64;
    let n_y = y.len() as f64;

    let mut combined: Vec<f64> = Vec::with_capacity(x.len() + y.len());
    combined.extend_from_slice(x);
    combined.extend_from_slice(y);
    let ranks = ranks_average(&combined);
    let w: f64 = ranks[..x.len()].iter().sum::<f64>() - n_x * (n_x + 1.0) / 2.0;

    // Tie counts on the combined sample.
    let mut sorted = combined.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut tie_correction = 0.0_f64;
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == sorted[i] {
            j += 1;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_correction += t * t * t - t;
        }
        i = j;
    }

    let mu = n_x * n_y / 2.0;
    let n_total = n_x + n_y;
    let mut var = n_x * n_y * (n_total + 1.0) / 12.0;
    if n_total > 1.0 {
        var -= n_x * n_y * tie_correction / (12.0 * n_total * (n_total - 1.0));
    }
    if var <= 0.0 {
        return if w < mu { 0.0 } else if w > mu { 1.0 } else { 0.5 };
    }
    let sigma = var.sqrt();
    let z = (w - mu + 0.5) / sigma;
    pnorm(z)
}

// ----- TopDom: diamond signal -------------------------------------------------
//
// R: Get.Diamond.Matrix(mat, i, size) -> mat[max(1, i-size+1):i, (i+1):min(i+size, n_bins)]
// We use 0-based [lo, hi) ranges. For i = n_bins-1 the diamond is empty and we
// return NaN (R returns NA, mean(NA) = NA).

fn diamond_mean(matrix: &[Vec<f64>], n: usize, i: usize, size: usize) -> f64 {
    if i + 1 >= n {
        return f64::NAN;
    }
    let r_lo = if i + 1 >= size { i + 1 - size } else { 0 };
    let r_hi = i + 1; // [r_lo, r_hi)
    let c_lo = i + 1;
    let c_hi = (i + size + 1).min(n); // [c_lo, c_hi)
    let nr = r_hi - r_lo;
    let nc = c_hi - c_lo;
    if nr == 0 || nc == 0 {
        return f64::NAN;
    }
    let mut acc = 0.0_f64;
    for r in r_lo..r_hi {
        for c in c_lo..c_hi {
            acc += matrix[r][c];
        }
    }
    acc / (nr * nc) as f64
}

// ----- TopDom: gap regions ---------------------------------------------------
//
// R: Which.Gap.Region2(mat, w): for each i, gap if sum(mat[i, max(1, i-w):min(i+w, n_bins)]) == 0.

fn which_gap_region2(matrix: &[Vec<f64>], n: usize, w: usize) -> Vec<usize> {
    let mut idx = Vec::new();
    for i in 0..n {
        let c_lo = if i >= w { i - w } else { 0 };
        let c_hi = (i + w + 1).min(n);
        let mut s = 0.0_f64;
        for c in c_lo..c_hi {
            s += matrix[i][c];
        }
        if s == 0.0 {
            idx.push(i);
        }
    }
    idx
}

// ----- TopDom: contiguous non-gap regions ------------------------------------
//
// R: Which.process.region(rmv.idx, n_bins, min.size=3) — returns dataframe of
// [start, end] (1-based, inclusive) for contiguous runs of non-rmv indices, with
// abs(end - start) >= min.size.

#[derive(Clone, Copy, Debug)]
struct Region {
    start: usize, // 0-based, inclusive
    end: usize,   // 0-based, inclusive
}

fn which_process_region(rmv: &[usize], n: usize, min_size: usize) -> Vec<Region> {
    let rmv_set: std::collections::HashSet<usize> = rmv.iter().copied().collect();
    let proc_set: Vec<usize> = (0..n).filter(|i| !rmv_set.contains(i)).collect();
    let mut regions = Vec::new();
    if proc_set.is_empty() {
        return regions;
    }
    let mut i = 0;
    while i < proc_set.len() {
        let start = proc_set[i];
        let mut j = i + 1;
        while j < proc_set.len() && proc_set[j] - proc_set[j - 1] <= 1 {
            j += 1;
        }
        let end = proc_set[j - 1];
        if end >= start && end - start >= min_size {
            regions.push(Region { start, end });
        }
        i = j;
    }
    regions
}

// ----- TopDom: Data.Norm + Change.Point + Detect.Local.Extreme ---------------

fn data_norm(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut rx = vec![0.0; n];
    let mut ry = vec![0.0; n];
    if n == 0 {
        return (rx, ry);
    }
    rx[0] = x[0];
    ry[0] = y[0];
    if n == 1 {
        return (rx, ry);
    }
    let dx: Vec<f64> = (1..n).map(|i| (x[i] - x[i - 1]).abs()).collect();
    let dy: Vec<f64> = (1..n).map(|i| (y[i] - y[i - 1]).abs()).collect();
    let mean_dx: f64 = dx.iter().sum::<f64>() / dx.len() as f64;
    let mean_dy: f64 = dy.iter().sum::<f64>() / dy.len() as f64;
    let sx = if mean_dx > 0.0 { 1.0 / mean_dx } else { 1.0 };
    let sy = if mean_dy > 0.0 { 1.0 / mean_dy } else { 1.0 };
    for i in 1..n {
        rx[i] = rx[i - 1] + (x[i] - x[i - 1]) * sx;
        ry[i] = ry[i - 1] + (y[i] - y[i - 1]) * sy;
    }
    (rx, ry)
}

fn change_point(x: &[f64], y: &[f64]) -> Vec<usize> {
    // R returns 1-based change point indices; we return 0-based.
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut fv = vec![f64::NAN; n];
    let mut cp = vec![0usize];
    fv[0] = 0.0;
    let mut i = 0usize;
    while i < n.saturating_sub(1) {
        let mut j = i + 1;
        fv[j] = ((x[j] - x[i]).powi(2) + (y[j] - y[i]).powi(2)).sqrt();
        while j < n - 1 {
            j += 1;
            let denom = ((x[j] - x[i]).powi(2) + (y[j] - y[i]).powi(2)).sqrt();
            if denom == 0.0 {
                cp.push(j - 1);
                j -= 1;
                break;
            }
            let mut sum_err = 0.0_f64;
            for &k in (i + 1..j).collect::<Vec<usize>>().iter() {
                sum_err += ((y[j] - y[i]) * x[k]
                    - (x[j] - x[i]) * y[k]
                    - (x[i] * y[j])
                    + (x[j] * y[i]))
                    .abs();
            }
            fv[j] = denom - sum_err / denom;
            if fv[j].is_nan() || fv[j - 1].is_nan() {
                cp.push(j - 1);
                j -= 1;
                break;
            }
            if fv[j] < fv[j - 1] {
                cp.push(j - 1);
                j -= 1;
                break;
            }
        }
        i = j;
    }
    cp.push(n - 1);
    cp
}

fn detect_local_extreme(x: &[f64]) -> Vec<f64> {
    // R returns a numeric vector with values in {-1, 0, 1}; we store as f64 too
    // because step 3 marks transient -2 values.
    let n = x.len();
    let mut ret = vec![0.0_f64; n];
    if n == 0 {
        return ret;
    }
    let mut x_local: Vec<f64> = x.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).collect();
    if n <= 3 {
        let mut min_i = 0;
        let mut max_i = 0;
        for i in 1..n {
            if x_local[i] < x_local[min_i] {
                min_i = i;
            }
            if x_local[i] > x_local[max_i] {
                max_i = i;
            }
        }
        ret[min_i] = -1.0;
        ret[max_i] = 1.0;
        return ret;
    }
    let x_idx: Vec<f64> = (1..=n).map(|v| v as f64).collect();
    let (nx, ny) = data_norm(&x_idx, &x_local);
    x_local = ny;
    let cp = change_point(&nx, &x_local);
    if cp.len() <= 2 || cp.len() == n {
        return ret;
    }
    for i in 1..(cp.len() - 1) {
        let c = cp[i];
        let cm = cp[i - 1];
        // local max/min based on direct neighbours
        if c >= 1 && c + 1 < n {
            if x_local[c] >= x_local[c - 1] && x_local[c] >= x_local[c + 1] {
                ret[c] = 1.0;
            } else if x_local[c] < x_local[c - 1] && x_local[c] < x_local[c + 1] {
                ret[c] = -1.0;
            }
        }
        // override with extreme within [cp[i-1], cp[i]] if more extreme exists
        let mut local_min = f64::INFINITY;
        let mut local_min_i = cm;
        let mut local_max = f64::NEG_INFINITY;
        let mut local_max_i = cm;
        for j in cm..=c {
            if x_local[j] < local_min {
                local_min = x_local[j];
                local_min_i = j;
            }
            if x_local[j] > local_max {
                local_max = x_local[j];
                local_max_i = j;
            }
        }
        let min_val = x_local[cm].min(x_local[c]);
        let max_val = x_local[cm].max(x_local[c]);
        if local_min < min_val {
            // R: ret[cp[i-1] - 1 + which.min(...)] = -1 — we already computed 0-based local_min_i
            ret[local_min_i] = -1.0;
        }
        if local_max > max_val {
            ret[local_max_i] = 1.0;
        }
    }
    ret
}

// ----- TopDom: Get.Diamond.Matrix2 + Upstream/Downstream triangles ----------

fn get_diamond_matrix2_flat(matrix: &[Vec<f64>], n: usize, i: usize, size: usize) -> Vec<f64> {
    // R returns a (size, size) matrix with NA fill; we return a flat Vec<f64>
    // skipping NaN (since wilcox.test silently drops NAs in inputs).
    let mut out = Vec::new();
    if i >= n - 1 {
        return out;
    }
    let lower = (i + 1).min(n - 1);
    let upper = (i + size).min(n - 1);
    for k in 0..size {
        if i >= k && i + 1 <= n {
            let row = i - k;
            let cols_n = upper - lower + 1;
            for c in 0..cols_n {
                out.push(matrix[row][lower + c]);
            }
        }
    }
    out
}

fn upper_triangle_flat(matrix: &[Vec<f64>], r0: usize, r1: usize) -> Vec<f64> {
    // R's upper.tri(diag=F): strictly above the main diagonal.
    let mut out = Vec::new();
    for r in r0..r1 {
        for c in (r + 1)..r1 {
            out.push(matrix[r][c]);
        }
    }
    out
}

fn get_upstream_triangle(matrix: &[Vec<f64>], n: usize, i: usize, size: usize) -> Vec<f64> {
    let lower = if i >= size { i - size } else { 0 };
    upper_triangle_flat(matrix, lower, (i + 1).min(n))
}

fn get_downstream_triangle(matrix: &[Vec<f64>], n: usize, i: usize, size: usize) -> Vec<f64> {
    if i + 1 >= n {
        return Vec::new();
    }
    let upper = (i + size + 1).min(n);
    upper_triangle_flat(matrix, i + 1, upper)
}

// ----- TopDom: Get.Pvalue ---------------------------------------------------
//
// R loops i in 1..(n-1) and stores pvalue[i] = wilcox.test(dia, c(ups, downs),
// alternative="less", exact=F)$p.value. pvalue[n] stays at 1.
// NA replacement: pvalue[is.na(pvalue)] = 1.

fn get_pvalue(matrix_scaled: &[Vec<f64>], n: usize, size: usize) -> Vec<f64> {
    let mut pvalue = vec![1.0_f64; n];
    for i in 0..n.saturating_sub(1) {
        let dia = get_diamond_matrix2_flat(matrix_scaled, n, i, size);
        let ups = get_upstream_triangle(matrix_scaled, n, i, size);
        let downs = get_downstream_triangle(matrix_scaled, n, i, size);
        let mut others = ups;
        others.extend(downs);
        let dia_clean: Vec<f64> = dia.into_iter().filter(|v| !v.is_nan()).collect();
        let others_clean: Vec<f64> = others.into_iter().filter(|v| !v.is_nan()).collect();
        let p = wilcoxon_ranksum_less(&dia_clean, &others_clean);
        pvalue[i] = if p.is_nan() { 1.0 } else { p };
    }
    pvalue
}

// ----- TopDom: Convert.Bin.To.Domain.TMP ------------------------------------
//
// Output BED rows: (start_bin, end_bin, tag) where tag is one of
//   "gap" | "domain" | "boundary"
// (chrom is the same for the whole chrom, prepended by the Python wrapper).

#[derive(Clone, Debug)]
pub struct BedRow {
    pub from_id: usize, // 0-based inclusive
    pub to_id: usize,   // 0-based inclusive
    pub tag: u8,        // 0=gap, 1=domain, 2=boundary
}

const TAG_GAP: u8 = 0;
const TAG_DOMAIN: u8 = 1;
const TAG_BOUNDARY: u8 = 2;

fn convert_bin_to_domain(
    n: usize,
    signal_idx: &[usize],
    gap_idx: &[usize],
    pvalues: &[f64],
) -> Vec<BedRow> {
    // gap rows: complement of gap_idx
    let not_gap: Vec<usize> = (0..n)
        .filter(|i| !gap_idx.contains(i))
        .collect();
    let gap_regions = which_process_region(&not_gap, n, 0);
    let mut gap_rows: Vec<BedRow> = gap_regions
        .iter()
        .map(|r| BedRow { from_id: r.start, to_id: r.end, tag: TAG_GAP })
        .collect();

    // domain rows: complement of (signal_idx ∪ gap_idx)
    let mut rmv: std::collections::HashSet<usize> = signal_idx.iter().copied().collect();
    rmv.extend(gap_idx.iter().copied());
    let rmv_v: Vec<usize> = rmv.into_iter().collect();
    let domain_regions = which_process_region(&rmv_v, n, 0);
    let mut domain_rows: Vec<BedRow> = domain_regions
        .iter()
        .map(|r| BedRow { from_id: r.start, to_id: r.end, tag: TAG_DOMAIN })
        .collect();

    // boundary rows: complement of signal_idx (min_size=1)
    let not_signal: Vec<usize> = (0..n)
        .filter(|i| !signal_idx.contains(i))
        .collect();
    let boundary_regions = which_process_region(&not_signal, n, 1);
    let boundary_rows: Vec<BedRow> = boundary_regions
        .iter()
        .filter_map(|r| {
            if r.start + 1 < n {
                Some(BedRow { from_id: r.start + 1, to_id: r.end, tag: TAG_BOUNDARY })
            } else {
                None
            }
        })
        .collect();

    // R sorts by chromStart (from.id); we already produce in order.
    gap_rows.extend(domain_rows.drain(..));
    gap_rows.sort_by_key(|r| r.from_id);

    // R's TMP variant: reclassify a domain row as boundary if every bin in it
    // has p-value < pvalue.cut (extreme constant-signal domains).
    for row in gap_rows.iter_mut() {
        if row.tag == TAG_DOMAIN {
            let n_constr = (row.from_id..=row.to_id)
                .filter(|&i| pvalues.get(i).map(|p| *p < PVALUE_CUT).unwrap_or(false))
                .count();
            let span = row.to_id - row.from_id + 1;
            if n_constr == span {
                row.tag = TAG_BOUNDARY;
            }
        }
    }

    // Merge consecutive boundary rows (R: stack.bdr collapse).
    let mut merged: Vec<BedRow> = Vec::new();
    let mut stack: Option<BedRow> = None;
    for row in gap_rows.into_iter() {
        if row.tag == TAG_BOUNDARY {
            stack = Some(match stack {
                Some(s) => BedRow {
                    from_id: s.from_id.min(row.from_id),
                    to_id: s.to_id.max(row.to_id),
                    tag: TAG_BOUNDARY,
                },
                None => row,
            });
        } else {
            if let Some(s) = stack.take() {
                merged.push(s);
            }
            merged.push(row);
        }
    }
    if let Some(s) = stack {
        merged.push(s);
    }
    // Re-add never-merged boundary_rows that lived between gaps.
    merged.extend(boundary_rows);
    merged.sort_by_key(|r| r.from_id);
    merged
}

// ----- TopDom: Scaling matrix data + per-region step 2 / step 3 -----------

fn topdom_chrom_inner(
    matrix: &[Vec<f64>],
    n: usize,
    window_size: usize,
    stat_filter: bool,
) -> Vec<BedRow> {
    // Step 1: mean.cf per bin
    let mut mean_cf = vec![0.0_f64; n];
    for i in 0..n {
        mean_cf[i] = diamond_mean(matrix, n, i, window_size);
    }

    // Step 2: gap regions + per-region local extreme
    let gap_idx = which_gap_region2(matrix, n, window_size);
    let mut local_ext = vec![-0.5_f64; n];
    let proc_regions = which_process_region(&gap_idx, n, 3);
    for r in proc_regions.iter() {
        let slice: Vec<f64> = mean_cf[r.start..=r.end].to_vec();
        let ext = detect_local_extreme(&slice);
        for (k, v) in ext.iter().enumerate() {
            local_ext[r.start + k] = *v;
        }
    }

    let mut pvalue = vec![1.0_f64; n];
    if stat_filter {
        // Step 3a: scale.matrix.data — zscore each off-diagonal d in 1..(2*window_size)
        let mut scaled: Vec<Vec<f64>> =
            matrix.iter().map(|r| r.clone()).collect();
        for d in 1..=(2 * window_size) {
            if d >= n {
                break;
            }
            let len = n - d;
            let mut diag: Vec<f64> = Vec::with_capacity(len);
            for i in 0..len {
                diag.push(matrix[i][i + d]);
            }
            let mean = diag.iter().sum::<f64>() / len as f64;
            let var = diag.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (len as f64 - 1.0).max(1.0);
            let sd = var.sqrt();
            if sd > 0.0 {
                for i in 0..len {
                    scaled[i][i + d] = (matrix[i][i + d] - mean) / sd;
                    scaled[i + d][i] = scaled[i][i + d];
                }
            } else {
                for i in 0..len {
                    scaled[i][i + d] = 0.0;
                    scaled[i + d][i] = 0.0;
                }
            }
        }

        // Step 3b: per-region p-values
        for r in proc_regions.iter() {
            let sub_n = r.end - r.start + 1;
            let sub: Vec<Vec<f64>> = (r.start..=r.end)
                .map(|i| scaled[i][r.start..=r.end].to_vec())
                .collect();
            let p_local = get_pvalue(&sub, sub_n, window_size);
            for (k, v) in p_local.iter().enumerate() {
                pvalue[r.start + k] = *v;
            }
        }

        // Step 3c: reclassify local minima as -2 if p-value < 0.05, then map -2 -> -1
        for i in 0..n {
            if (local_ext[i] - (-1.0)).abs() < 1e-12 && pvalue[i] < PVALUE_CUT {
                local_ext[i] = -2.0;
            }
        }
        for v in local_ext.iter_mut() {
            if (*v - (-1.0)).abs() < 1e-12 {
                *v = 0.0;
            } else if (*v - (-2.0)).abs() < 1e-12 {
                *v = -1.0;
            }
        }
    }

    // Step 4: bin -> domain
    let signal_idx: Vec<usize> = (0..n).filter(|&i| local_ext[i] == -1.0).collect();
    let gap_idx_final: Vec<usize> = (0..n).filter(|&i| local_ext[i] == -0.5).collect();
    convert_bin_to_domain(n, &signal_idx, &gap_idx_final, &pvalue)
}

#[pyfunction]
#[pyo3(signature = (matrix, window_size, stat_filter=true))]
pub fn py_topdom_chrom<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f32>,
    window_size: usize,
    stat_filter: bool,
) -> PyResult<Bound<'py, PyList>> {
    let arr = matrix.as_array();
    let n = arr.nrows();
    if arr.ncols() != n {
        return Err(pyo3::exceptions::PyValueError::new_err("matrix must be square"));
    }
    let mat: Vec<Vec<f64>> = (0..n)
        .map(|r| (0..n).map(|c| arr[(r, c)] as f64).collect())
        .collect();
    let rows = py.allow_threads(|| topdom_chrom_inner(&mat, n, window_size, stat_filter));

    let list = PyList::empty_bound(py);
    for row in rows {
        let tup = pyo3::types::PyTuple::new_bound(
            py,
            &[
                row.from_id.into_py(py),
                row.to_id.into_py(py),
                (row.tag as u8).into_py(py),
            ],
        );
        list.append(tup)?;
    }
    Ok(list)
}

// Anchor against unused warnings on the only-numpy-mode build.
#[allow(dead_code)]
fn _unused_anchor(_a: &PyArray1<f32>) {}
```

- [ ] **Step 2: Syntax check via cargo check:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster/rust
cargo check --quiet 2>&1 | tail -10
```

Expected: clean (pre-existing warnings unrelated).

**No commit in this task.**

---

## Task 4: Mount domain + topdom in `lib.rs` + register PyO3 + build

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/lib.rs`

- [ ] **Step 1: Add two `mod` declarations.** Find `mod scan_kernels;` (added in Phase 1) and append two siblings after it:

```rust
mod insulation;
mod topdom;
```

- [ ] **Step 2: Register two new PyO3 functions inside `#[pymodule] fn _rust(...)`.** Append before `Ok(())`:

```rust
    m.add_function(wrap_pyfunction!(insulation::py_insulation_score_chrom, m)?)?;
    m.add_function(wrap_pyfunction!(topdom::py_topdom_chrom, m)?)?;
```

- [ ] **Step 3: Build:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    maturin develop --release 2>&1 | tail -10
```

Expected: ends with `🛠 Installed schicluster-rs-0.2.0`. If a compile error surfaces in `topdom.rs` (the largest new file), fix it from the verbatim source above — do not stub out.

- [ ] **Step 4: Smoke-test both new symbols are reachable from Python:**

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
from schicluster_rs._rust import py_insulation_score_chrom, py_topdom_chrom
print('both domain kernels importable:', callable(py_insulation_score_chrom) and callable(py_topdom_chrom))
"
```

Expected: `True`.

- [ ] **Step 5: Re-run the Phase-0+1 gate to confirm no regression:**

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    python -m pytest -q tests/test_exact_match.py 2>&1 | tail -3
```

Expected: `11 passed, 6 skipped` — unchanged from Phase 1 close. The Phase 2 outputs are still skipped because drivers haven't been extended yet (that's Task 6).

**No commit in this task.**

---

## Task 5: Python wrappers + `patch_schicluster()` extension

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/python/schicluster_rs/domain/__init__.py`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/python/schicluster_rs/__init__.py`

Upstream's `domain/call_domain.py::call_domain_and_insulation` (a) builds a sparse matrix, (b) calls `run_top_dom` (an rpy2 closure) for the BED, and (c) calls `single_chrom_calculate_insulation_score` for the score. Phase 2 swaps both inner calls. We do NOT touch `call_domain_and_insulation` itself.

The `topdom` wrapper produces a dataframe with the same columns the upstream `run_top_dom` returns (`chrom`, `chromStart`, `chromEnd`, `name`), so `call_domain_and_insulation` keeps working unchanged.

- [ ] **Step 1: Write `python/schicluster_rs/domain/__init__.py`** (Write tool):

```python
"""Python wrappers around the Rust domain-module kernels.

Two upstream targets:

* ``single_chrom_calculate_insulation_score`` — sliding-window insulation
  score per bin. Direct one-to-one with the Rust kernel.
* The rpy2 closure ``run_top_dom`` inside
  ``schicluster.domain.call_domain.call_domain_and_insulation`` — the
  pure-Rust TopDom returns the same 4-column DataFrame so the upstream
  orchestrator (chrom labelling, blacklist boundary aggregation, etc.)
  keeps working unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TAG_GAP = 0
TAG_DOMAIN = 1
TAG_BOUNDARY = 2
TAG_TO_NAME = {TAG_GAP: "gap", TAG_DOMAIN: "domain", TAG_BOUNDARY: "boundary"}


def insulation_score_chrom(matrix, window_size=10, save_count=False):
    """Drop-in replacement for
    schicluster.domain.call_domain.single_chrom_calculate_insulation_score.

    ``matrix`` can be a scipy.sparse matrix or a numpy array; we densify to
    f32 contiguous before passing to Rust (the fixture sizes are tiny so
    densification cost is irrelevant; production single-chrom matrices fit
    in memory at 25 kb resolution).
    """
    try:
        from schicluster_rs._rust import py_insulation_score_chrom as _rust_ins
    except ImportError:
        from schicluster.domain.call_domain import (
            single_chrom_calculate_insulation_score as _upstream,
        )
        return _upstream(matrix=matrix, window_size=window_size, save_count=save_count)

    if hasattr(matrix, "toarray"):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    dense_f32 = np.ascontiguousarray(dense, dtype=np.float32)
    return _rust_ins(dense_f32, int(window_size), bool(save_count))


def _topdom_chrom_to_df(matrix, bins, window_size, stat_filter=True):
    """Rust TopDom -> pandas.DataFrame with the same 4 columns the rpy2 closure emits.

    `bins` must be a DataFrame with the upstream columns
    ``['chr', 'from.coord', 'to.coord']`` — matches what
    ``call_domain_and_insulation`` constructs before calling ``run_top_dom``.
    Returns columns: ``chrom``, ``chromStart``, ``chromEnd``, ``name``.
    """
    try:
        from schicluster_rs._rust import py_topdom_chrom as _rust_topdom
    except ImportError as e:
        raise ImportError("schicluster_rs Rust extension not available") from e

    if hasattr(matrix, "toarray"):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    dense_f32 = np.ascontiguousarray(dense, dtype=np.float32)
    rows = _rust_topdom(dense_f32, int(window_size), bool(stat_filter))

    if not rows:
        return pd.DataFrame([], columns=["chrom", "chromStart", "chromEnd", "name"])

    # bins is 0-indexed; from_id / to_id are bin indices (inclusive)
    from_coord = bins["from.coord"].to_numpy()
    to_coord = bins["to.coord"].to_numpy()
    chrom_name = str(bins["chr"].iloc[0])

    records = []
    for from_id, to_id, tag in rows:
        records.append({
            "chrom": chrom_name,
            "chromStart": int(from_coord[from_id]),
            "chromEnd": int(to_coord[to_id]),
            "name": TAG_TO_NAME[int(tag)],
        })
    return pd.DataFrame.from_records(records,
                                     columns=["chrom", "chromStart", "chromEnd", "name"])


def run_top_dom(j, p, x, bins, window_size):
    """Drop-in replacement for the rpy2 closure ``RunTopDom`` inside
    upstream ``schicluster/domain/TopDom.R``.

    ``j`` / ``p`` / ``x`` come from a scipy CSC matrix the upstream Python
    builds (matrix.indices+1, matrix.indptr, matrix.data). We rebuild the
    dense symmetric matrix here and hand it to the Rust kernel.
    """
    from scipy.sparse import csc_matrix
    j = np.asarray(j).astype(np.int64) - 1  # back to 0-indexed
    p = np.asarray(p).astype(np.int64)
    x = np.asarray(x).astype(np.float32)
    n = len(bins)
    csc = csc_matrix((x, j, p), shape=(n, n))
    dense = np.asarray(csc.todense(), dtype=np.float32)
    bins_df = pd.DataFrame(bins) if not isinstance(bins, pd.DataFrame) else bins
    return _topdom_chrom_to_df(dense, bins_df, window_size, stat_filter=True)


__all__ = [
    "insulation_score_chrom",
    "run_top_dom",
    "TAG_GAP", "TAG_DOMAIN", "TAG_BOUNDARY",
]
```

- [ ] **Step 2: Extend `python/schicluster_rs/__init__.py`.**

(a) Extend the Rust import try-block to pull the two new symbols (use Edit). Replace:

```
        py_find_summit_chrom as _find_summit_chrom,
        set_num_threads as _set_num_threads,
```

with:

```
        py_find_summit_chrom as _find_summit_chrom,
        py_insulation_score_chrom as _insulation_score_chrom,
        py_topdom_chrom as _topdom_chrom,
        set_num_threads as _set_num_threads,
```

(b) Before the existing `def patch_schicluster()` line, add a domain re-import block. Use Edit and locate the existing block:

```
from schicluster_rs.loop import (
    loop_bkg_chrom,
    merge_cells_for_single_chromosome,
    loop_background as _loop_background,
    find_summit as _find_summit,
)

# Re-exports for the public API surface declared in __all__.
loop_background = _loop_background
find_summit = _find_summit
```

Replace with:

```
from schicluster_rs.loop import (
    loop_bkg_chrom,
    merge_cells_for_single_chromosome,
    loop_background as _loop_background,
    find_summit as _find_summit,
)

# Re-exports for the public API surface declared in __all__.
loop_background = _loop_background
find_summit = _find_summit

from schicluster_rs.domain import (
    insulation_score_chrom,
    run_top_dom as _run_top_dom,
)
```

(c) Extend the body of `patch_schicluster()` to rebind the upstream domain bindings. Locate the existing body inside `def patch_schicluster() -> bool:` and append before `return True` (and before the matching `except ImportError: return False`):

```python
        # ---- domain module ----
        from schicluster.domain import call_domain as _domain_mod
        _domain_mod.single_chrom_calculate_insulation_score = insulation_score_chrom
        # call_domain_and_insulation builds a local `run_top_dom` closure
        # inside its body via `def run_top_dom(matrix, bins): ...`, so we
        # cannot rebind it at module level. Instead, monkey-patch the
        # rpy2 `r.source(...)` call so the closure that's built from it
        # uses our Rust kernel:
        _domain_mod.r = _DomainRStub()
```

And at the top of `__init__.py` (e.g. just before the `def patch_schicluster()`), add the stub:

```python
class _DomainRStub:
    """Stand-in for the rpy2 `r` global used by
    ``schicluster.domain.call_domain.call_domain_and_insulation``.

    Upstream does ``r.source('TopDom.R')`` then references ``r.RunTopDom(j, p,
    x, bins, window_size)`` to compute domains. We swap that to call the
    Rust kernel directly so the upstream orchestrator (boundary aggregation,
    insulation concatenation, .npz / .h5ad / .nc writers) keeps working.
    """
    @staticmethod
    def source(_path):
        return None

    @staticmethod
    def RunTopDom(j, p, x, bins, window_size):
        return _run_top_dom(j, p, x, bins, window_size)
```

(d) Extend `__all__`. Locate the existing list and add `"insulation_score_chrom"`:

```python
__all__ = [
    "random_walk_cpu", "impute_chromosome", "patch_schicluster",
    "set_num_threads", "convolve2d_mirror",
    "loop_bkg_chrom", "merge_cells_for_single_chromosome",
    "loop_background", "find_summit",
    "insulation_score_chrom",
]
```

- [ ] **Step 3: Smoke-test wrapper imports:**

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
import schicluster_rs as P
print('insulation_score_chrom:', callable(P.insulation_score_chrom))
print('domain._run_top_dom:', callable(P._run_top_dom))
print('in __all__:', 'insulation_score_chrom' in P.__all__)
import numpy as np
m = np.random.default_rng(0).standard_normal((30, 30)).astype(np.float32)
m = (m + m.T) / 2; np.fill_diagonal(m, 0)
s = P.insulation_score_chrom(m, window_size=3, save_count=False)
print('insulation score shape:', s.shape, 'finite:', np.all(np.isfinite(s)))
"
```

Expected: callables True, finite True.

**No commit in this task.**

---

## Task 6: Extend reference + candidate drivers + parity harness

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/tests/py_reference_driver.py`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/tests/_run_candidate.py`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/tests/parity_harness.py`

The two TopDom outputs (`topdom.bed.interval_jaccard` and `topdom.bed.bin_label_agreement`) both read from `$.topdom.bed` — the harness `evaluate` dispatches by output.name to compute the appropriate metric.

- [ ] **Step 1: Extend `tests/py_reference_driver.py`** (py3.6-safe; activates the schicluster env's R + rpy2 — confirmed working in Phase 1).

Use Edit. Locate:

```python
from schicluster.loop.loop_calling import find_summit as upstream_find_summit
```

Add immediately after:

```python
from schicluster.domain.call_domain import (
    single_chrom_calculate_insulation_score as upstream_insulation,
)
from rpy2.robjects import r as _rpy2_r, pandas2ri as _rpy2_pandas2ri, numpy2ri as _rpy2_numpy2ri
import schicluster as _schicluster
_rpy2_pandas2ri.activate()
_rpy2_numpy2ri.activate()
_rpy2_r.source(str(__import__('pathlib').Path(_schicluster.__path__[0]) / 'domain/TopDom.R'))
```

Locate the constants block (after `LOOP_CHROM = "chr1"`) and add:

```python
DOMAIN_FIXTURE = REPO_ROOT / "data" / "fixtures" / "domain_small.npz"
DOMAIN_WINDOW_SIZE = 5
DOMAIN_BIN_RESOLUTION = 10_000  # synthetic bin size for the fixture
DOMAIN_CHROM = "chr1"
```

Locate the `ref_find_summit` function. Add the following after it (and before `def main()`):

```python
def ref_insulation_score(domain_pack):
    from scipy.sparse import csc_matrix
    m = domain_pack["topdom.matrix"]
    csc = csc_matrix(m.astype(np.float32))
    w = int(domain_pack["insulation.window_size"])
    score = upstream_insulation(csc, window_size=w, save_count=False)
    return np.asarray(score, dtype=np.float32).tolist()


def ref_topdom_bed(domain_pack):
    m = domain_pack["topdom.matrix"]
    w = int(domain_pack["topdom.window_size"])
    from scipy.sparse import csc_matrix
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
```

Locate the `main()` function and add (after the existing `payload["find_summit"] = ...` assignment but before the `finally:`):

```python
        domain_pack = _load_npz(DOMAIN_FIXTURE)
        payload["insulation"] = {"score": ref_insulation_score(domain_pack)}
        payload["topdom"] = {"bed": ref_topdom_bed(domain_pack)}
```

- [ ] **Step 2: Extend `tests/_run_candidate.py`** (py3.10, can use modern syntax).

Add immediately after the existing constants:

```python
DOMAIN_FIXTURE = REPO_ROOT / "data" / "fixtures" / "domain_small.npz"
DOMAIN_WINDOW_SIZE = 5
DOMAIN_BIN_RESOLUTION = 10_000
DOMAIN_CHROM = "chr1"
```

Add the following two functions after `cand_find_summit`:

```python
def cand_insulation_score(domain_pack: dict) -> list:
    m = np.ascontiguousarray(domain_pack["topdom.matrix"], dtype=np.float32)
    w = int(domain_pack["insulation.window_size"])
    score = schicluster_rs.insulation_score_chrom(m, window_size=w, save_count=False)
    return np.asarray(score, dtype=np.float32).tolist()


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
```

In `main()`, add (after the existing `payload["find_summit"] = ...` assignment):

```python
        domain_pack = _load_npz(DOMAIN_FIXTURE)
        payload["insulation"] = {"score": cand_insulation_score(domain_pack)}
        payload["topdom"] = {"bed": cand_topdom_bed(domain_pack)}
```

- [ ] **Step 3: Extend `tests/parity_harness.py`** to dispatch on the two topdom output names.

Use Edit. Locate the existing block in `evaluate(...)` that special-cases `find_summit.sizes` and add a similar dispatch for the two `topdom.bed.*` cases. Insert before the existing `find_summit.sizes` branch:

```python
    # Special case: topdom.bed.interval_jaccard — set Jaccard over domain (start, end) pairs.
    if output.name == "topdom.bed.interval_jaccard":
        ref_bed = _dig(reference_blob, "$.topdom.bed") or []
        cand_bed = _dig(candidate_blob, "$.topdom.bed") or []
        ref_doms = {(r["chromStart"], r["chromEnd"]) for r in ref_bed if r["name"] == "domain"}
        cand_doms = {(r["chromStart"], r["chromEnd"]) for r in cand_bed if r["name"] == "domain"}
        union = ref_doms | cand_doms
        if not union:
            metric_value = 1.0
        else:
            metric_value = len(ref_doms & cand_doms) / len(union)
        ok = metric_value >= output.threshold
        return {
            "status": "pass" if ok else "fail",
            "metric": metric_value,
            "threshold": output.threshold,
            "message": f"{output.name}: Jaccard={metric_value:.4f} vs threshold={output.threshold!r}",
        }

    # Special case: topdom.bed.bin_label_agreement — per-bin tag agreement after
    # expanding both BEDs onto the chrom's bin grid.
    if output.name == "topdom.bed.bin_label_agreement":
        ref_bed = _dig(reference_blob, "$.topdom.bed") or []
        cand_bed = _dig(candidate_blob, "$.topdom.bed") or []
        if not ref_bed and not cand_bed:
            metric_value = 1.0
        else:
            # Use the candidate's max coord (or ref's) to size the grid.
            def _max_end(bed):
                return max((r["chromEnd"] for r in bed), default=0)
            grid_end = max(_max_end(ref_bed), _max_end(cand_bed))
            # bin resolution is implicit in the data; the fixture uses 10_000.
            # Use 1 bp grid for simplicity since the fixture is tiny.
            ref_tags = ["gap"] * grid_end
            cand_tags = ["gap"] * grid_end
            for r in ref_bed:
                for i in range(r["chromStart"], min(r["chromEnd"], grid_end)):
                    ref_tags[i] = r["name"]
            for r in cand_bed:
                for i in range(r["chromStart"], min(r["chromEnd"], grid_end)):
                    cand_tags[i] = r["name"]
            n_total = max(grid_end, 1)
            n_match = sum(1 for a, b in zip(ref_tags, cand_tags) if a == b)
            metric_value = n_match / n_total
        ok = metric_value >= output.threshold
        return {
            "status": "pass" if ok else "fail",
            "metric": metric_value,
            "threshold": output.threshold,
            "message": f"{output.name}: agreement={metric_value:.4f} vs threshold={output.threshold!r}",
        }
```

- [ ] **Step 4: Run the orchestrator end-to-end:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV bash tests/run_parity.sh 2>&1 | tail -20
```

Expected: `14 passed, 3 skipped`. Diagnose failures with the rebuildpy fail-suspicion list:

- **`insulation.score` mismatch**: f64 sum order should be deterministic per-row; check the upstream uses `.sum()` on sparse blocks (which scipy does in C, so likely deterministic). If the error exceeds `1e-6`, double-check the `i < w` branch boundary conditions.
- **`topdom.bed.interval_jaccard` < 0.95**: usually means a single boundary detection differs. Inspect the diff between ref and cand BEDs (print them) — common causes are NaN in `mean.cf[n-1]` propagating differently, or change-point algorithm picking a different inflection on a flat region. Fix the Rust source, not the threshold.
- **`topdom.bed.bin_label_agreement` < 0.98**: usually downstream of the above; fix interval Jaccard first.

If you find a real bug in the Rust port that requires re-edit, you'll need to re-run `maturin develop --release` before re-running the gate.

**Hard cap: 5 iterations.** If still failing after 5 fix-and-rerun cycles, report DONE_WITH_CONCERNS with the failing metrics + diagnosis. Do NOT widen the manifest.

**No commit in this task.**

---

## Task 7: Final verification

**Files:** none (read-only).

- [ ] **Step 1: Re-run the full gate end-to-end one more time:**

```bash
env -u VIRTUAL_ENV bash tests/run_parity.sh 2>&1 | tail -10
```

Expected: `14 passed, 3 skipped`.

- [ ] **Step 2: Capture the metric values for the iteration log.** For each of the 3 new outputs, run the harness directly and record the metric:

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
import sys, json, pathlib
sys.path.insert(0, '/large_storage/zhoulab/shengmao/rebuildpy')
sys.path.insert(0, '/large_storage/zhoulab/shengmao/rust-scHiCluster/tests')
from parity_harness import load_outputs, load_dumps, evaluate
ref, cand = load_dumps()
for o in load_outputs():
    if o.name.startswith('insulation') or o.name.startswith('topdom'):
        r = evaluate(o, ref, cand)
        print(f'{o.name:38}  status={r[\"status\"]:<10} metric={r[\"metric\"]}')
"
```

Write down the three metric values for the iteration log in Task 8.

**No commit in this task.**

---

## Task 8: Single Phase-2 commit

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/Cargo.toml`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/pyproject.toml`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/docs/ITERATION_LOG.md`

- [ ] **Step 1: Bump versions** — `Cargo.toml` line 3 from `version = "0.2.0"` to `version = "0.3.0"`; `pyproject.toml` similarly.

- [ ] **Step 2: Append the Phase 2 iteration block to `docs/ITERATION_LOG.md`.** After the existing Iteration 1 block and its `---` divider, append:

```yaml
iteration: 2
title: Phase 2 — domain module ported, native TopDom drops rpy2/R
admissibility: E
action: |
  Per-chrom Rust ports of:
    - single_chrom_calculate_insulation_score (insulation.rs) — f64 sums
      cast to f32 on emit; sliding-window submatrix block sums match
      scipy's CSR.sum() semantics.
    - TopDom (topdom.rs) — full native port of TopDom.R, replacing the
      rpy2 round-trip. Includes:
        * diamond mean signal + gap region detection (Which.Gap.Region2)
        * Data.Norm + Change.Point + Detect.Local.Extreme
        * Wilcoxon rank-sum p-values (normal approximation with
          continuity + tie correction, matching R's wilcox.test(exact=F,
          alternative="less"))
        * Convert.Bin.To.Domain.TMP with boundary merging
  Wilcoxon's normal CDF uses an inline Abramowitz erf (~1.5e-7 error,
  more than enough for p < 0.05 thresholds). No new crate dependencies.
status: accepted
fixture: data/fixtures/domain_small.npz
parity:
  insulation.score:                  { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  topdom.bed.interval_jaccard:       { class: ranked,                threshold: 0.95,  pass: true }
  topdom.bed.bin_label_agreement:    { class: classification,        threshold: 0.98,  pass: true }
notes: |
  patch_schicluster() now monkey-patches schicluster.domain.call_domain's
  rpy2 `r` global with a stub whose RunTopDom routes through the Rust
  kernel — that's how the upstream `call_domain_and_insulation` keeps
  working without edits (the rpy2 closure inside it now calls Rust).
  insulation_score_chrom is monkey-patched at module level directly.

---
```

- [ ] **Step 3: Rebuild + final gate check:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    maturin develop --release 2>&1 | tail -5
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    python -m pytest -q tests/test_exact_match.py 2>&1 | tail -3
```

Expected: `schicluster-rs-0.3.0` installed, `14 passed, 3 skipped`.

- [ ] **Step 4: Single commit** — controller assembles. Stage everything new + modified, commit with the message below:

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
git add \
    rust/Cargo.toml \
    rust/src/lib.rs \
    rust/src/insulation.rs \
    rust/src/topdom.rs \
    python/schicluster_rs/__init__.py \
    python/schicluster_rs/domain/__init__.py \
    pyproject.toml \
    data/fixtures/synthesize.py \
    tests/parity_harness.py \
    tests/py_reference_driver.py \
    tests/_run_candidate.py \
    docs/ITERATION_LOG.md \
    docs/superpowers/plans/2026-06-04-rust-port-phase-2-domain-module.md \
    .gitignore
git commit -m "$(cat <<'EOF'
feat: Phase 2 domain module ported to Rust, drops rpy2/R for TopDom (3 manifest outputs green)

Ports the per-chrom numerical hot paths of scHiCluster's domain module
to Rust, replacing the rpy2 -> R round-trip with a fully native TopDom:

  - single_chrom_calculate_insulation_score -> rust/src/insulation.rs
    sliding-window submatrix sums in f64, cast to f32 on emit
  - TopDom.R::TopDom                        -> rust/src/topdom.rs
    diamond mean signal, gap regions, change-point / local-extreme
    detection, R-compatible Wilcoxon rank-sum p-values (normal approx
    with continuity + tie correction), bin -> domain BED conversion

The Wilcoxon p-value path uses an inline Abramowitz approximation of erf
(~1.5e-7 error, well below the p < 0.05 threshold relevant here). No new
crate dependencies.

Python wrappers in python/schicluster_rs/domain/ preserve upstream
signatures. patch_schicluster() monkey-patches both
single_chrom_calculate_insulation_score (module-level rebind) and the
rpy2 `r` global inside call_domain (stub whose RunTopDom delegates to
Rust) so upstream's call_domain_and_insulation orchestrator keeps
working unchanged — including blacklist boundary aggregation and the
.npz / .h5ad / .nc writers.

Parity gate (data/manifest.yaml read-only, 14 of 17 outputs now green):
  conv.convolved            deterministic-bounded 1e-6 (Phase 0)
  loop_bkg.{E,T}            deterministic-bounded 1e-6 (Phase 1)
  merge.{e_sum,e2_sum}      deterministic-bounded 1e-6 (Phase 1)
  scan_kernels.{bl,donut,h,v} deterministic-bounded 1e-6 (Phase 1)
  find_summit.{idx,sizes}   ranked / classification (Phase 1)
  insulation.score          deterministic-bounded 1e-6
  topdom.bed.interval_jaccard         ranked      >= 0.95
  topdom.bed.bin_label_agreement      classification >= 0.98
Skips (3): compartment x 2 (Phase 3), embedding x 1 (Phase 4).

Other notes:
  - Two custom dispatches in tests/parity_harness.py compute Jaccard on
    domain (start, end) tuples and per-bin tag agreement after expanding
    both BEDs onto the chrom's bin grid; both gates read $.topdom.bed.
  - Reference driver runs upstream TopDom.R via rpy2 in the schicluster
    env (R 4.3.3 already wired up there since Phase 0).

Version 0.2.0 -> 0.3.0.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
git log --oneline -3
```

Expected: one new commit at HEAD; previous Phase 1 close at `489fddb`.

---

## Self-review

**1. Spec coverage.** Spec §5 rows 12–14 (`insulation.score`, `topdom.bed.interval_jaccard`, `topdom.bed.bin_label_agreement`) are covered by Tasks 2, 3 (Rust kernels) + 5 (Python wrappers) + 6 (drivers + harness). Spec §6 domain block is the surface Tasks 2/3 implement. Spec §7 (E) acceleration is the iteration-2 default (no rayon yet — serial port, parity-first). Spec §9 phase 2 ("domain.rs + topdom.rs + python/schicluster_rs/domain.py + monkey-patch single_chrom_calculate_insulation_score, run_top_dom") matches Tasks 2–5. Spec §10 deliverables (iteration log, version bump) are Task 8. Spec §11 risks: the TopDom Wilcoxon tie drift is explicitly absorbed by the `ranked` + `classification` gate (not deterministic), as designed.

**2. Placeholder scan.** No `TBD` / `TODO` / `implement later` / `similar to Task N`. The `<...>` placeholders inside the iteration-log timing block are intentional runtime fill-ins (same pattern Phase 1 Task 9 used).

**3. Type consistency.** `insulation_score_chrom`, `topdom_chrom` names match across `rust/src/*.rs`, `lib.rs` registrations, `python/schicluster_rs/domain/__init__.py` wrappers, `__init__.py` re-exports, and both driver scripts. PyO3 binding names follow the `py_<rust_fn>` convention. The `BedRow` Rust struct matches the `(from_id, to_id, tag)` tuple the Python wrapper unpacks. Tag constants (`TAG_GAP=0`, `TAG_DOMAIN=1`, `TAG_BOUNDARY=2`) consistent between Rust and Python.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-04-rust-port-phase-2-domain-module.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
