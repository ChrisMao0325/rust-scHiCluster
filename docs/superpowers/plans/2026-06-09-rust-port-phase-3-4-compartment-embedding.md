# scHiCluster Rust port — Phase 3 + 4 (compartment + embedding)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit discipline:** per user direction, this plan ships as **one commit at the end** (Task 7). Subagent implementers do NOT run `git commit` — they write code, run smoke checks, leave changes staged or unstaged for the controller to assemble the final Phase 3+4 commit.

**Goal:** Turn the **last 3 outputs** in the manifest green, taking the gate to **17 passed, 0 skipped** and closing the loop on the full scHiCluster numerical port.

**Architecture:** Two independent ports bundled into one phase because both are pure-numerical (no R / no statistics-package dependency) and the embedding speedup is honestly modest (I/O-bound). `compartment.rs` ports `single_chrom_compartment` (sparse row-normalize + CpG-weighted matvec) and `compartment_strength` (decay-normalized A/B/AB sums); `embedding.rs` ports `make_chrom_features` (upper-tri index extraction + scalar scaling). SVD stays sklearn per the spec — Phase 4's gate is `embedding.cell_by_feature` (the cell × feature matrix **before** SVD), not the embedding itself.

**Tech Stack:** Rust 1.95 + PyO3 0.22 + ndarray 0.16 + sprs 0.11 (existing); no new crate dependencies.

---

## Phase 3+4 outputs (already pre-registered in `data/manifest.yaml` — read-only)

| Manifest output | Algorithm class | Threshold | Upstream Python |
|---|---|---|---|
| `compartment.comp` | deterministic-bounded | 1e-6 | `compartment/call_compartment.py::single_chrom_compartment` → `comp` |
| `compartment.strength` | deterministic-bounded | 1e-6 | same → `[AA, BB, AB]` |
| `embedding.cell_by_feature` | deterministic-strict | 0.0 (f32 exact) | `embedding/calc_embedding.py::make_chrom_matrix` (before SVD) |

Phase 3+4 is complete when `pytest -q tests/test_exact_match.py` reports **17 passed, 0 skipped**.

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `data/fixtures/synthesize.py` | MODIFY | Adds `compartment_small_fixture` (100×100 symmetric Hi-C + CpG ratio vector with planted zeros) and `embedding_small_fixture` (5 cells × 50×50 matrices). |
| `data/fixtures/compartment_small.npz` | NEW (generated, gitignored) | |
| `data/fixtures/embedding_small.npz` | NEW (generated, gitignored) | |
| `rust/src/compartment.rs` | NEW | `compartment_chrom` — `(comp, strength)` from sparse triplets + dense `cpg_ratio`. f64 accumulators throughout. |
| `rust/src/embedding.rs` | NEW | `make_chrom_features` — given (cells × n × n) f32 array + `dist_bins` + `scale_factor`, emit (cells × n_features) f32 matrix matching `matrix[triu_filter_idx].ravel() * scale_factor` bit-for-bit. |
| `rust/src/lib.rs` | MODIFY | Mounts the two new modules; registers two new PyO3 functions. |
| `python/schicluster_rs/compartment/__init__.py` | NEW | Python wrapper around the Rust compartment kernel; signature-compatible with upstream `single_chrom_compartment`. |
| `python/schicluster_rs/embedding/__init__.py` | NEW | Python wrapper around `make_chrom_features`; signature-compatible with upstream `make_chrom_matrix` (reads cool files in Python, hands dense matrices to Rust). |
| `python/schicluster_rs/__init__.py` | MODIFY | Imports the two new Rust symbols + the two wrappers; extends `__all__`; extends `patch_schicluster()` to rebind `single_chrom_compartment` and `make_chrom_matrix`. |
| `tests/py_reference_driver.py` | MODIFY | Adds 3 reference computations (upstream Python on the fixtures). |
| `tests/_run_candidate.py` | MODIFY | Adds 3 candidate computations (Rust on the same fixtures). |
| `rust/Cargo.toml` | MODIFY (Task 7) | Version `0.3.0` → `0.4.0`. |
| `pyproject.toml` | MODIFY (Task 7) | Version `0.3.0` → `0.4.0`. |
| `docs/ITERATION_LOG.md` | MODIFY (Task 7) | Append Phase 3+4 iteration block. |

The harness needs no new dispatches — all three outputs are 1D arrays handled by the existing `deterministic` path.

---

## Task 1: Phase 3+4 fixtures

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/synthesize.py`

- [ ] **Step 1: Append two new fixture functions** in `synthesize.py` after `domain_small_fixture`:

```python
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
```

- [ ] **Step 2: Extend the `main()` body** to also write `compartment_small.npz` and `embedding_small.npz`. Find the existing Phase-2 `domain_small` write and append:

```python
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
```

- [ ] **Step 3: Append the two new artefacts to `.gitignore`:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
for line in "data/fixtures/compartment_small.npz" "data/fixtures/embedding_small.npz"; do
  grep -qxF "$line" .gitignore || echo "$line" >> .gitignore
done
```

- [ ] **Step 4: Regenerate fixtures + verify** in `rebuild-rust`:

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    python data/fixtures/synthesize.py 2>&1 | tail -20
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
import numpy as np, pathlib
c = np.load(pathlib.Path('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/compartment_small.npz'))
e = np.load(pathlib.Path('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/embedding_small.npz'))
print('compartment keys:', sorted(c.files), 'matrix:', c['compartment.matrix'].shape, 'cpg nnz:', int((c['compartment.cpg_ratio'] > 0).sum()))
print('embedding keys:', sorted(e.files), 'cells:', e['embedding.cells'].shape, 'n_bins:', int(e['embedding.n_bins']), 'scale:', int(e['embedding.scale_factor']))
"
```

Expected:
- compartment: matrix `(100, 100)`, cpg nnz 90 (10 zeroed out of 100).
- embedding: cells `(5, 50, 50)`, n_bins 50, scale 100000.

- [ ] **Step 5: Verify schicluster env (py3.6) loads them too:**

```bash
env -u VIRTUAL_ENV conda run -n schicluster --no-capture-output python -c "
import numpy as np
c = np.load('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/compartment_small.npz')
e = np.load('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/embedding_small.npz')
print('schicluster env ok:', c['compartment.matrix'].shape, e['embedding.cells'].shape)
"
```

Expected: `(100, 100) (5, 50, 50)`.

**No commit in this task.**

---

## Task 2: Rust `compartment.rs`

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/compartment.rs`

Ports `compartment/call_compartment.py::single_chrom_compartment` + `compartment_strength`. The Python pipeline:

```python
# single_chrom_compartment(matrix, cpg_ratio):
matrix = matrix - diags(matrix.diagonal())       # zero diagonal
matrix = matrix + diags(col_sum == 0)            # identity on zero-sum cols
matrix.data /= np.repeat(col_sum, nnz_per_row)   # row-major normalize
comp = matrix.dot(cpg)                           # CpG-weighted matvec

# compartment_strength(matrix, comp, cpg):
bin_filter = cpg > 0
tmp = comp[bin_filter]
a_pos = tmp > np.percentile(tmp, 80)
b_pos = tmp < np.percentile(tmp, 20)
decay[i] = matrix.diagonal(i).mean()
matrix.data /= decay[|col - row|]
E = matrix.tocsr()[bin_filter, bin_filter]
AA = E[a_pos, a_pos].sum(); BB = E[b_pos, b_pos].sum(); AB = E[a_pos, b_pos].sum()
```

The Rust port mirrors this exactly, working from raw upper-triangle triplets (we densify since n is small enough that dense ops are cheap and bit-equivalent). All intermediate arithmetic in f64; cast comp/strength to f64 for emit (manifest output is double in tolist()).

- [ ] **Step 1: Write `rust/src/compartment.rs`** (Write tool, exact content):

```rust
//! Port of scHiCluster/schicluster/compartment/call_compartment.py::
//!   single_chrom_compartment + compartment_strength.
//!
//! Matches the upstream sparse pipeline using a dense f64 working buffer:
//!   1. zero the diagonal
//!   2. add identity on zero-sum columns
//!   3. row-major normalize each datum by its column sum
//!   4. comp = matrix · cpg_ratio
//!   5. strength = decay-normalized A/B partition sums

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn percentile_linear(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return values[0];
    }
    let frac = p / 100.0 * (n as f64 - 1.0);
    let lo = frac.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let w = frac - lo as f64;
    values[lo] * (1.0 - w) + values[hi] * w
}

pub fn compartment_chrom(
    matrix: &Array2<f32>,
    cpg_ratio: &[f32],
    calc_strength: bool,
) -> (Vec<f64>, Option<[f64; 3]>) {
    let n = matrix.nrows();
    assert_eq!(matrix.ncols(), n);
    assert_eq!(cpg_ratio.len(), n);

    // Step 1: copy to f64, zero diagonal.
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|r| (0..n).map(|c| matrix[(r, c)] as f64).collect())
        .collect();
    for i in 0..n {
        m[i][i] = 0.0;
    }

    // Step 2: column sums; identity on zero-sum columns.
    let mut col_sum = vec![0.0_f64; n];
    for r in 0..n {
        for c in 0..n {
            col_sum[c] += m[r][c];
        }
    }
    for j in 0..n {
        if col_sum[j] == 0.0 {
            m[j][j] = 1.0;
            col_sum[j] = 1.0;
        }
    }

    // Step 3: row-major normalize — divide each stored value m[r][c] by col_sum[r].
    // Note upstream uses `np.repeat(col_sum, nnz_per_row)` over CSR data: that's
    // the column sum **of the row**, i.e. each row r's stored values get divided
    // by col_sum[r]. (For a symmetric matrix col_sum[r] == row_sum[r].)
    for r in 0..n {
        if col_sum[r] != 0.0 {
            let inv = 1.0 / col_sum[r];
            for c in 0..n {
                m[r][c] *= inv;
            }
        }
    }

    // Step 4: comp = matrix · cpg_ratio.
    let cpg_f64: Vec<f64> = cpg_ratio.iter().map(|v| *v as f64).collect();
    let mut comp = vec![0.0_f64; n];
    for r in 0..n {
        let mut acc = 0.0_f64;
        for c in 0..n {
            acc += m[r][c] * cpg_f64[c];
        }
        comp[r] = acc;
    }

    let strength = if calc_strength {
        Some(compartment_strength_inner(&m, n, &comp, &cpg_f64))
    } else {
        None
    };

    (comp, strength)
}

fn compartment_strength_inner(
    matrix: &[Vec<f64>],
    n: usize,
    comp: &[f64],
    cpg: &[f64],
) -> [f64; 3] {
    let bin_filter: Vec<bool> = cpg.iter().map(|v| *v > 0.0).collect();
    let mut tmp: Vec<f64> = (0..n).filter(|&i| bin_filter[i]).map(|i| comp[i]).collect();
    if tmp.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut tmp_for_p80 = tmp.clone();
    let mut tmp_for_p20 = tmp.clone();
    let p80 = percentile_linear(&mut tmp_for_p80, 80.0);
    let p20 = percentile_linear(&mut tmp_for_p20, 20.0);
    drop(tmp);

    // a_pos / b_pos in the FILTERED index space (the comp values past bin_filter)
    let filtered_idx: Vec<usize> = (0..n).filter(|&i| bin_filter[i]).collect();
    let a_pos: Vec<bool> = filtered_idx.iter().map(|&i| comp[i] > p80).collect();
    let b_pos: Vec<bool> = filtered_idx.iter().map(|&i| comp[i] < p20).collect();

    // decay[k] = mean of matrix's k-th diagonal (over all of n)
    let mut decay = vec![0.0_f64; n];
    for k in 0..n {
        let mut s = 0.0_f64;
        let mut count = 0usize;
        for i in 0..(n - k) {
            s += matrix[i][i + k];
            count += 1;
        }
        decay[k] = if count > 0 { s / count as f64 } else { 0.0 };
    }

    // Build E = matrix / decay[|col - row|] then index into [bin_filter, bin_filter].
    let m = filtered_idx.len();
    let mut e = vec![0.0_f64; m * m];
    for (i_out, &i_in) in filtered_idx.iter().enumerate() {
        for (j_out, &j_in) in filtered_idx.iter().enumerate() {
            let d = if j_in >= i_in { j_in - i_in } else { i_in - j_in };
            let denom = decay[d];
            let val = if denom != 0.0 { matrix[i_in][j_in] / denom } else { 0.0 };
            e[i_out * m + j_out] = val;
        }
    }

    let mut aa = 0.0_f64;
    let mut bb = 0.0_f64;
    let mut ab = 0.0_f64;
    for i in 0..m {
        for j in 0..m {
            let v = e[i * m + j];
            if a_pos[i] && a_pos[j] {
                aa += v;
            }
            if b_pos[i] && b_pos[j] {
                bb += v;
            }
            if a_pos[i] && b_pos[j] {
                ab += v;
            }
        }
    }
    [aa, bb, ab]
}

#[pyfunction]
#[pyo3(signature = (matrix, cpg_ratio, calc_strength=false))]
pub fn py_compartment_chrom<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f32>,
    cpg_ratio: PyReadonlyArray1<'py, f32>,
    calc_strength: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, PyObject)> {
    let m = matrix.as_array().to_owned();
    let cpg = cpg_ratio.as_slice()?.to_vec();
    let (comp, strength) = py.allow_threads(|| compartment_chrom(&m, &cpg, calc_strength));
    let comp_arr = ndarray::Array1::from(comp).into_pyarray_bound(py);
    let strength_obj: PyObject = match strength {
        Some(s) => {
            let arr = ndarray::Array1::from(s.to_vec()).into_pyarray_bound(py);
            arr.into_py(py)
        }
        None => py.None(),
    };
    Ok((comp_arr, strength_obj))
}
```

- [ ] **Step 2: cargo check:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster/rust
cargo check --quiet 2>&1 | tail -5
```

Expected: clean (pre-existing warnings unrelated).

**No commit in this task.**

---

## Task 3: Rust `embedding.rs`

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/embedding.rs`

Ports `embedding/calc_embedding.py::make_chrom_matrix`'s inner extraction loop. Upstream does `chrom_matrix[i, :] = matrix[idx].ravel()` per cell, then multiplies the whole array by `scale_factor`. Since the matrix is f32, the operations are: f32 read → f32 store → f32 scalar multiply. **No reduction, no cross-element dependency** — the manifest gate is `deterministic-strict atol=0` (exact f32 bit-equality), so the Rust port must produce identical bits.

`make_idx` returns the upper triangle (`k=1`, strictly above diagonal) filtered to `col - row < dist_bins + 1`. The order matters: numpy's `triu_indices(n, k=1)` enumerates row-major `(r, c)` with `r < c`. We replicate that order exactly.

- [ ] **Step 1: Write `rust/src/embedding.rs`** (Write tool, exact content):

```rust
//! Port of scHiCluster/schicluster/embedding/calc_embedding.py::make_chrom_matrix's
//! inner extraction loop. SVD stays sklearn — Phase 4's gate is the cell × feature
//! matrix **before** SVD.
//!
//! Operations are pure f32 reads + a single f32 scalar multiply, no reduction —
//! the parity gate is deterministic-strict (atol=0), exact f32 bit-equality.

use ndarray::ArrayView3;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

/// Build the (n_features,) list of `(row, col)` pairs `make_idx` produces.
/// Mirrors numpy's `triu_indices(n, k=1)` row-major enumeration + col-row filter.
fn upper_tri_indices_dist_filtered(n: usize, dist_bins_plus_1: usize) -> Vec<(usize, usize)> {
    let mut idx = Vec::new();
    for r in 0..n {
        for c in (r + 1)..n {
            if c - r < dist_bins_plus_1 {
                idx.push((r, c));
            }
        }
    }
    idx
}

pub fn make_chrom_features(
    cells: ArrayView3<f32>,
    dist_bins_plus_1: usize,
    scale_factor: f32,
) -> Vec<Vec<f32>> {
    let n_cells = cells.shape()[0];
    let n_bins = cells.shape()[1];
    debug_assert_eq!(cells.shape()[2], n_bins);
    let idx = upper_tri_indices_dist_filtered(n_bins, dist_bins_plus_1);
    let n_features = idx.len();
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n_cells);
    for ci in 0..n_cells {
        let mut row = Vec::with_capacity(n_features);
        for &(r, c) in idx.iter() {
            row.push(cells[(ci, r, c)] * scale_factor);
        }
        out.push(row);
    }
    out
}

#[pyfunction]
#[pyo3(signature = (cells, dist_bins_plus_1, scale_factor))]
pub fn py_make_chrom_features<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray3<'py, f32>,
    dist_bins_plus_1: usize,
    scale_factor: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let view = cells.as_array();
    let n_cells = view.shape()[0];
    let n_bins = view.shape()[1];
    if view.shape()[2] != n_bins {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cells must be (n_cells, n_bins, n_bins) — square per-cell matrices",
        ));
    }
    let rows = py.allow_threads(|| make_chrom_features(view, dist_bins_plus_1, scale_factor));
    let n_features = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut flat = Vec::with_capacity(n_cells * n_features);
    for row in rows {
        flat.extend(row);
    }
    let arr = ndarray::Array2::from_shape_vec((n_cells, n_features), flat).expect("shape");
    Ok(arr.into_pyarray_bound(py))
}
```

- [ ] **Step 2: cargo check:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster/rust
cargo check --quiet 2>&1 | tail -5
```

Expected: clean.

**No commit in this task.**

---

## Task 4: Mount + build

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/rust/src/lib.rs`

- [ ] **Step 1: Add two `mod` declarations** after `mod topdom;` (Phase 2 added that one):

Edit `old_string`:
```rust
mod topdom;
```
Edit `new_string`:
```rust
mod topdom;
mod compartment;
mod embedding;
```

- [ ] **Step 2: Register two new PyO3 functions** inside `#[pymodule] fn _rust(...)`. Locate the existing `m.add_function(wrap_pyfunction!(topdom::py_topdom_chrom, m)?)?;` line and insert two new lines after it (before `Ok(())`):

Edit `old_string`:
```rust
    m.add_function(wrap_pyfunction!(topdom::py_topdom_chrom, m)?)?;
    Ok(())
```
Edit `new_string`:
```rust
    m.add_function(wrap_pyfunction!(topdom::py_topdom_chrom, m)?)?;
    m.add_function(wrap_pyfunction!(compartment::py_compartment_chrom, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::py_make_chrom_features, m)?)?;
    Ok(())
```

- [ ] **Step 3: Build:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    maturin develop --release 2>&1 | tail -10
```

Expected: `🛠 Installed schicluster-rs-0.3.0` (the bump to 0.4.0 is Task 7).

- [ ] **Step 4: Smoke-test imports:**

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
from schicluster_rs._rust import py_compartment_chrom, py_make_chrom_features
print('both Phase 3+4 kernels importable:', callable(py_compartment_chrom) and callable(py_make_chrom_features))
"
```

Expected: `True`.

- [ ] **Step 5: Confirm Phase 0-2 gate still green** (Phase 3+4 outputs still skipped — drivers come in Task 6):

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    python -m pytest -q tests/test_exact_match.py 2>&1 | tail -3
```

Expected: `14 passed, 3 skipped` (unchanged from Phase 2 close).

**No commit in this task.**

---

## Task 5: Python wrappers + `patch_schicluster()` extension

**Files:**
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/python/schicluster_rs/compartment/__init__.py`
- Create: `/large_storage/zhoulab/shengmao/rust-scHiCluster/python/schicluster_rs/embedding/__init__.py`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/python/schicluster_rs/__init__.py`

- [ ] **Step 1: Write `python/schicluster_rs/compartment/__init__.py`** (Write tool):

```python
"""Python wrapper around the Rust compartment kernel.

Drop-in replacement for
schicluster.compartment.call_compartment.single_chrom_compartment.
"""
from __future__ import annotations

import numpy as np


def single_chrom_compartment(matrix, cpg_ratio, calc_strength=False):
    """Match upstream signature: (matrix, cpg_ratio) -> (comp, scores).

    Falls back to upstream when the Rust extension is unavailable.
    """
    try:
        from schicluster_rs._rust import py_compartment_chrom as _rust_compartment
    except ImportError:
        from schicluster.compartment.call_compartment import (
            single_chrom_compartment as _upstream,
        )
        return _upstream(matrix=matrix, cpg_ratio=cpg_ratio, calc_strength=calc_strength)

    if hasattr(matrix, "toarray"):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    dense_f32 = np.ascontiguousarray(dense, dtype=np.float32)

    # cpg_ratio may be a pandas Series (upstream's call site does that); coerce.
    cpg_arr = cpg_ratio.values if hasattr(cpg_ratio, "values") else np.asarray(cpg_ratio)
    cpg_f32 = np.ascontiguousarray(cpg_arr, dtype=np.float32)

    comp, strength = _rust_compartment(dense_f32, cpg_f32, bool(calc_strength))
    comp_arr = np.asarray(comp)
    if calc_strength:
        return comp_arr, np.asarray(strength)
    else:
        return comp_arr, None


__all__ = ["single_chrom_compartment"]
```

- [ ] **Step 2: Write `python/schicluster_rs/embedding/__init__.py`** (Write tool):

```python
"""Python wrapper around the Rust embedding cell-by-feature kernel.

Drop-in replacement for
schicluster.embedding.calc_embedding.make_chrom_matrix.

The upstream function reads .cool files per cell; we preserve that surface
(reading cool stays Python) and only the in-memory extraction goes through
Rust. SVD is intentionally untouched and continues to use sklearn.
"""
from __future__ import annotations

import numpy as np


def make_chrom_matrix(cell_table, chrom, nbins, output_path,
                      scale_factor, dist, resolution):
    """Reads cool files per cell, calls Rust extraction kernel, writes npz.

    Output array shape matches upstream's: (n_cells, n_features) float32.
    """
    try:
        from schicluster_rs._rust import py_make_chrom_features as _rust_features
    except ImportError:
        from schicluster.embedding.calc_embedding import (
            make_chrom_matrix as _upstream,
        )
        return _upstream(
            cell_table=cell_table, chrom=chrom, nbins=nbins,
            output_path=output_path, scale_factor=scale_factor,
            dist=dist, resolution=resolution,
        )

    import cooler
    n_cells = cell_table.size
    cells = np.zeros((n_cells, nbins, nbins), dtype=np.float32)
    for i, (_, cell_url) in enumerate(cell_table.items()):
        cool = cooler.Cooler(cell_url)
        cells[i, :, :] = np.ascontiguousarray(
            cool.matrix(balance=False, sparse=False).fetch(chrom),
            dtype=np.float32,
        )
    dist_bins_plus_1 = int(dist / resolution + 1)
    out = _rust_features(cells, dist_bins_plus_1, float(scale_factor))
    np.savez(output_path, np.asarray(out, dtype=np.float32))
    return


__all__ = ["make_chrom_matrix"]
```

- [ ] **Step 3: Extend `python/schicluster_rs/__init__.py`** with four changes (using the Edit tool):

(a) Add the two new Rust imports. Locate:
```
        py_topdom_chrom as _topdom_chrom,
        set_num_threads as _set_num_threads,
```
Replace with:
```
        py_topdom_chrom as _topdom_chrom,
        py_compartment_chrom as _compartment_chrom,
        py_make_chrom_features as _make_chrom_features,
        set_num_threads as _set_num_threads,
```

(b) Add wrapper re-imports + a re-export. Locate (the domain re-export block added in Phase 2):
```
from schicluster_rs.domain import (
    insulation_score_chrom,
    run_top_dom as _run_top_dom,
)
```
Append:
```
from schicluster_rs.domain import (
    insulation_score_chrom,
    run_top_dom as _run_top_dom,
)

from schicluster_rs.compartment import single_chrom_compartment as _single_chrom_compartment
from schicluster_rs.embedding import make_chrom_matrix as _make_chrom_matrix

single_chrom_compartment = _single_chrom_compartment
make_chrom_matrix = _make_chrom_matrix
```

(c) Extend `patch_schicluster()` body. Locate the Phase-2 domain block ending with `_domain_mod.r = _DomainRStub()` then `return True`:
```
        _domain_mod.r = _DomainRStub()
        return True
```
Replace with:
```
        _domain_mod.r = _DomainRStub()
        # ---- compartment module ----
        from schicluster.compartment import call_compartment as _comp_mod
        _comp_mod.single_chrom_compartment = single_chrom_compartment
        # ---- embedding module ----
        from schicluster.embedding import calc_embedding as _emb_mod
        _emb_mod.make_chrom_matrix = make_chrom_matrix
        return True
```

(d) Extend `__all__`. Locate:
```
__all__ = [
    "random_walk_cpu", "impute_chromosome", "patch_schicluster",
    "set_num_threads", "convolve2d_mirror",
    "loop_bkg_chrom", "merge_cells_for_single_chromosome",
    "loop_background", "find_summit",
    "insulation_score_chrom",
]
```
Replace with:
```
__all__ = [
    "random_walk_cpu", "impute_chromosome", "patch_schicluster",
    "set_num_threads", "convolve2d_mirror",
    "loop_bkg_chrom", "merge_cells_for_single_chromosome",
    "loop_background", "find_summit",
    "insulation_score_chrom",
    "single_chrom_compartment", "make_chrom_matrix",
]
```

- [ ] **Step 4: Smoke-test:**

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output python -c "
import schicluster_rs as P
import numpy as np

print('single_chrom_compartment callable:', callable(P.single_chrom_compartment))
print('make_chrom_matrix callable:', callable(P.make_chrom_matrix))
print('both in __all__:', all(n in P.__all__ for n in ('single_chrom_compartment', 'make_chrom_matrix')))

# Functional smoke on the actual fixture
c = np.load('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/compartment_small.npz')
m = c['compartment.matrix']
cpg = c['compartment.cpg_ratio']
comp, strength = P.single_chrom_compartment(m, cpg, calc_strength=True)
print('compartment comp shape:', comp.shape, 'finite:', bool(np.all(np.isfinite(comp))))
print('strength:', strength)

e = np.load('/large_storage/zhoulab/shengmao/rust-scHiCluster/data/fixtures/embedding_small.npz')
from schicluster_rs._rust import py_make_chrom_features
dist_bins_plus_1 = int(e['embedding.dist']) // int(e['embedding.resolution']) + 1
feat = py_make_chrom_features(e['embedding.cells'], dist_bins_plus_1, float(e['embedding.scale_factor']))
print('embedding cell_by_feature shape:', feat.shape, 'dtype:', feat.dtype)
"
```

Expected: comp shape `(100,)` finite True, strength a 3-element array of floats, feature matrix `(5, M)` float32 (M depending on dist/resolution).

- [ ] **Step 5: Phase 0-2 gate still green:**

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    python -m pytest -q tests/test_exact_match.py 2>&1 | tail -3
```

Expected: `14 passed, 3 skipped`.

**No commit in this task.**

---

## Task 6: Extend drivers + run gate

**Files:**
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/tests/py_reference_driver.py`
- Modify: `/large_storage/zhoulab/shengmao/rust-scHiCluster/tests/_run_candidate.py`

No new harness dispatches needed — all 3 outputs are 1-D arrays handled by the existing `deterministic` path.

- [ ] **Step 1: Extend `tests/py_reference_driver.py`** (py3.6-safe).

Add new imports (after the existing upstream domain imports):
```python
from schicluster.compartment.call_compartment import (
    single_chrom_compartment as upstream_compartment,
)
from schicluster.embedding.calc_embedding import make_idx as upstream_make_idx
```

Add new constants alongside the existing ones:
```python
COMPARTMENT_FIXTURE = REPO_ROOT / "data" / "fixtures" / "compartment_small.npz"
EMBEDDING_FIXTURE = REPO_ROOT / "data" / "fixtures" / "embedding_small.npz"
```

Add three new reference functions after `ref_topdom_bed`:
```python
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
```

In `main()`, after the existing `payload["topdom"] = ...` assignment inside the `try:` block:
```python
        comp_pack = _load_npz(COMPARTMENT_FIXTURE)
        comp_vals, strength_vals = ref_compartment(comp_pack)
        payload["compartment"] = {"comp": comp_vals, "strength": strength_vals}

        emb_pack = _load_npz(EMBEDDING_FIXTURE)
        payload["embedding"] = {"cell_by_feature": ref_embedding_features(emb_pack)}
```

- [ ] **Step 2: Extend `tests/_run_candidate.py`** (py3.10):

Add constants:
```python
COMPARTMENT_FIXTURE = REPO_ROOT / "data" / "fixtures" / "compartment_small.npz"
EMBEDDING_FIXTURE = REPO_ROOT / "data" / "fixtures" / "embedding_small.npz"
```

Add three new candidate functions after `cand_topdom_bed`:
```python
def cand_compartment(comp_pack: dict) -> tuple[list, list]:
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
```

In `main()`, after the existing `payload["topdom"] = ...`:
```python
        comp_pack = _load_npz(COMPARTMENT_FIXTURE)
        comp_vals, strength_vals = cand_compartment(comp_pack)
        payload["compartment"] = {"comp": comp_vals, "strength": strength_vals}

        emb_pack = _load_npz(EMBEDDING_FIXTURE)
        payload["embedding"] = {"cell_by_feature": cand_embedding_features(emb_pack)}
```

- [ ] **Step 3: Run the orchestrator end-to-end:**

```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
env -u VIRTUAL_ENV bash tests/run_parity.sh 2>&1 | tail -10
```

Expected: **17 passed, 0 skipped**. If a Phase 3+4 output fails, diagnose:

- **`compartment.comp` mismatch** (most likely): the row-normalization step in upstream uses `np.repeat(col_sum, getnnz(axis=1))` which divides each stored datum by `col_sum[row_of_datum]`. The Rust port matches this by doing `m[r][c] /= col_sum[r]`. For a symmetric matrix, that's `col_sum == row_sum`. If error > 1e-6, check the f64 accumulators in both `col_sum` and the matvec.

- **`compartment.strength` mismatch**: the `decay[i]` is the **arithmetic mean of the i-th diagonal**, including zeros from the zeroed-diagonal step. Confirm `count` is `n - k`, not just non-zero count.

- **`embedding.cell_by_feature` mismatch** (atol=0!): exact f32 bit-equality required. This should pass on first try — the operation is `cells[i, r, c] * scale_factor`, identical f32 to numpy. If it fails, double-check the `(r, c)` enumeration order matches `numpy.triu_indices(n, k=1)` — that's row-major with `r < c`, in increasing `r` then increasing `c`. Also confirm `dist_bins_plus_1 = dist // resolution + 1` matches upstream's `(yy - xx) < (dist / resolution + 1)`.

If you find a real bug, edit the Rust source, rebuild (`maturin develop --release`), re-run.

**Hard cap: 5 iterations.** Never widen the manifest.

- [ ] **Step 4: Final verification:**

```bash
env -u VIRTUAL_ENV conda run -n rebuild-rust --no-capture-output \
    python -m pytest -v tests/test_exact_match.py 2>&1 | tail -25
```

All 17 PASSED.

**No commit in this task.**

---

## Task 7: Single Phase 3+4 commit (controller-handled)

The controller bumps versions, appends the iteration log, and assembles the single commit.

Version `0.3.0 → 0.4.0`. Iteration log:

```yaml
iteration: 3
title: Phase 3+4 — compartment + embedding ported, gate fully green
admissibility: E
action: |
  Final per-chrom Rust ports:
    - single_chrom_compartment + compartment_strength (compartment.rs)
      f64 accumulators throughout; matches upstream's row-major
      normalize-by-col-sum, decay-normalized A/B partition sums.
    - make_chrom_matrix's extraction kernel (embedding.rs)
      Pure f32 read + scalar multiply, no reduction — meets the
      deterministic-strict gate (atol=0, exact f32 bit-equality).
  SVD intentionally stays sklearn per design spec §6 — embedding's
  Phase-4 manifest output is the cell-by-feature matrix *before* SVD.
status: accepted
fixture: data/fixtures/{compartment_small,embedding_small}.npz
parity:
  compartment.comp:        { class: deterministic-bounded, threshold: 1.0e-6, pass: true, metric: <fill> }
  compartment.strength:    { class: deterministic-bounded, threshold: 1.0e-6, pass: true, metric: <fill> }
  embedding.cell_by_feature: { class: deterministic-strict, threshold: 0.0, pass: true, metric: <fill> }
notes: |
  Final 3-of-17 outputs turn green; gate is now fully green.
  patch_schicluster() additionally rebinds single_chrom_compartment +
  make_chrom_matrix at module level so the upstream multiprocess
  orchestrators transparently use Rust.
```

(The `<fill>` values come from the actual gate run.)

---

## Self-review

**1. Spec coverage.** Spec §5 rows 15–17 (compartment + embedding) → Tasks 2 (Rust compartment), 3 (Rust embedding), 5 (Python wrappers), 6 (drivers). Spec §6 keeps SVD in Python (Task 5 wrapper preserves that explicitly). Spec §7 (E) baseline — no rayon/SIMD, parity-first.

**2. Placeholder scan.** No `TBD`/`TODO` markers. The `<fill>` in the iteration-log YAML is intentional runtime-measured (same pattern Phase 1/2 used).

**3. Type consistency.** `compartment_chrom`, `make_chrom_features` names match across `rust/src/*.rs`, `lib.rs` registrations, `python/schicluster_rs/compartment|embedding/__init__.py` wrappers, `__init__.py` re-exports, and both drivers.
