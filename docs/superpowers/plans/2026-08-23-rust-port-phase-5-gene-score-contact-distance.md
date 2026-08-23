# Phase 5 — gene-score + contact-distance Rust Ports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port scHiCluster's `gene-score` per-gene window-sum loop and `contact-distance` per-cell TSV reader to Rust, clearing four newly pre-registered parity outputs and taking the manifest gate from 17 to 21.

**Architecture:** Two new leaf crates under `rust/src/`, each with a thin Python wrapper that keeps upstream's orchestration (ProcessPoolExecutor, HDF5 writes, pandas matrix build) untouched. `patch_schicluster()` rebinds the upstream per-cell workers at module level. `gene_score.rs` replaces ~78.7k scipy sparse-slice allocations per cell with binary-searched CSR range sums, parallel across genes. `contact_distance.rs` replaces `pd.read_csv` with a streaming gzip line reader, so no DataFrame is ever built.

**Tech Stack:** Rust 1.95 / PyO3 0.22 / numpy 0.22 / rayon 1.10 / flate2 1.0 (new); Python 3.10 candidate env (`rebuild-rust`), Python 3.6 reference env (`schicluster`).

---

## Environment facts (verified 2026-08-23, do not re-derive)

- `$PYTHON_REF_ENV` = conda env `schicluster`, **Python 3.6.13**, numpy 1.19.2, scipy 1.5.2, pandas 1.1.5, cooler 0.9.3. `schicluster.draft.gene_score` and `schicluster.cool.contact_distance` both import cleanly.
  **Python 3.6 means: no `from __future__ import annotations`, no PEP 585 generics, no f-strings in `tests/py_reference_driver.py`. Use `.format()`.**
- `$RUST_TEST_ENV` = conda env `rebuild-rust`, Python 3.10.20, scipy 1.15.2.
- The four upstream semantics this plan depends on were measured **identically in both scipy 1.5.2 and 1.15.2**:
  - `D[-1:4, 0:5]` on a 10x10 CSR → shape `(0, 5)`, sum `0.0` (negative slice start resolves to `n-1`).
  - `D[7:10, 8:11]` on 10 columns → shape `(3, 2)` (overrun clips).
  - `csr_matrix` built from int32 data: `.sum()` returns `numpy.int64`.
  - `np.histogram([-0.5, 0.0, 0.999, 1.0, 2.999, 3.0, 3.001], bins=[0,1,2,3])` → `[2, 1, 2]`; last bin right-closed, out-of-range dropped.
- `cargo add flate2 --dry-run` resolves; default features are the pure-Rust `rust_backend` (miniz_oxide), so no system zlib is needed and the CI wheel matrix stays portable.

## File structure

| Path | Responsibility |
|---|---|
| `rust/src/gene_score.rs` (create) | CSR rectangular-window sums, Python-slice bound resolution, rayon across genes |
| `rust/src/contact_distance.rs` (create) | streaming (multi-member) gzip TSV reader, numpy-compatible histogram binning, per-chrom distinct-bin-pair sets |
| `rust/src/lib.rs` (modify) | `mod` declarations + two `add_function` registrations |
| `rust/Cargo.toml` (modify) | add `flate2 = "1.0"` |
| `python/schicluster_rs/gene_score/__init__.py` (create) | `gene_score_impute` / `gene_score_raw` with upstream signatures |
| `python/schicluster_rs/contact_distance/__init__.py` (create) | `compute_decay` with upstream signature |
| `python/schicluster_rs/__init__.py` (modify) | re-export + `patch_schicluster()` rebinds + `__all__` |
| `python/schicluster_rs/__main__.py` (modify) | `_SUPPORTED` + help text |
| `data/manifest.yaml` (modify) | append 4 pre-registered outputs; **touch nothing existing** |
| `data/fixtures/synthesize.py` (modify) | `gene_score_small` + `contact_distance_small` fixtures |
| `tests/test_gene_score_semantics.py` (create) | fast in-env unit parity vs scipy |
| `tests/test_contact_distance_semantics.py` (create) | fast in-env unit parity vs pandas/numpy |
| `tests/py_reference_driver.py` (modify) | 3 new reference dump blocks |
| `tests/_run_candidate.py` (modify) | 3 new candidate dump blocks |

---

## Task 1: Pre-register the four manifest outputs

The protocol requires the gate be written **before** the kernels exist. Nothing in the existing 17 entries may change.

**Files:**
- Modify: `data/manifest.yaml` (insert before the `acceleration:` block)

- [ ] **Step 1: Append the four output blocks**

Insert immediately after the `embedding.cell_by_feature` block and before the blank line preceding `acceleration:`:

```yaml

  # ---- Phase 5 (gene-score) ----
  - name: gene_score.impute
    type: "1d f64 array, one score per gene"
    location_reference: "$.gene_score.impute"
    location_candidate: "$.gene_score.impute"
    metric: deterministic
    algorithm_class: deterministic-bounded
    threshold: 1.0e-6     # numpy's pairwise .sum() is not reproduced bit-for-bit; see spec 3.4
  - name: gene_score.raw
    type: "1d i64 array, one count per gene"
    location_reference: "$.gene_score.raw"
    location_candidate: "$.gene_score.raw"
    metric: deterministic
    algorithm_class: deterministic-strict
    threshold: 0.0        # int32 counts: addition is exact and order-independent

  # ---- Phase 5 (contact-distance) ----
  - name: contact_distance.decay
    type: "1d i64 histogram counts over log-spaced distance bins"
    location_reference: "$.contact_distance.decay"
    location_candidate: "$.contact_distance.decay"
    metric: deterministic
    algorithm_class: deterministic-strict
    threshold: 0.0
  - name: contact_distance.sparsity
    type: "1d i64 distinct off-diagonal bin-pair count per chrom, chrom-name sorted"
    location_reference: "$.contact_distance.sparsity"
    location_candidate: "$.contact_distance.sparsity"
    metric: deterministic
    algorithm_class: deterministic-strict
    threshold: 0.0
```

- [ ] **Step 2: Verify the file still parses and existing entries are untouched**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust python -c "
import yaml
m = yaml.safe_load(open('data/manifest.yaml'))
names = [o['name'] for o in m['outputs']]
print(len(names), 'outputs')
assert len(names) == 21, names
assert len(set(names)) == 21
for n in ['gene_score.impute','gene_score.raw','contact_distance.decay','contact_distance.sparsity']:
    assert n in names, n
print('OK')
"
git diff --stat data/manifest.yaml
```
Expected: `21 outputs` then `OK`; the diff shows only insertions (`+`), zero deletions.

- [ ] **Step 3: Commit**

```bash
git add data/manifest.yaml
git commit -m "gate: pre-register 4 Phase-5 outputs (gene_score, contact_distance)"
```

---

## Task 2: Fixtures for both new modules

**Files:**
- Modify: `data/fixtures/synthesize.py`

- [ ] **Step 1: Append the gene-score fixture builder**

Add after `embedding_small_fixture`, before `def main()`:

```python
# ---- Phase 5 fixture parameters (gene-score) ----
GENE_N_BINS = 60
GENE_RESOLUTION = 10_000
GENE_CHROM = "chr1"
GENE_CHROM_SIZE = GENE_N_BINS * GENE_RESOLUTION - 1


def _gene_windows():
    """(start_bin, end_bin, gene_id) triples, already floor-divided by resolution.

    Deliberately pins the three upstream edge cases from the design spec:
      * GENE_AT_BIN0 starts at bin 0, so gene_score_impute's (xx-1) slice start
        is -1, which scipy resolves to n-1 -> empty window -> score 0.0.
      * GENE_OVERRUN ends at the last bin, so the (yy+2) column bound overruns
        n_cols and scipy clips it.
      * GENE_EMPTY sits in a bin range the synthetic matrix leaves at zero.
    """
    return [
        (0, 4, "GENE_AT_BIN0"),
        (5, 9, "GENE_NORMAL_A"),
        (12, 12, "GENE_SINGLE_BIN"),
        (20, 31, "GENE_NORMAL_B"),
        (40, 44, "GENE_EMPTY"),
        (GENE_N_BINS - 3, GENE_N_BINS - 1, "GENE_OVERRUN"),
    ]


def gene_score_small_fixture(seed: int = 49):
    """Sparse upper-tri contact matrix + a gene table + a raw contact file.

    Writes gene_score_small.cool (impute-mode input) and
    gene_score_small.contact.tsv.gz (raw-mode input), and returns the arrays
    both drivers need.
    """
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for d in range(1, 25):
        for r in range(GENE_N_BINS - d):
            c = r + d
            # leave bins 38..46 empty so GENE_EMPTY really is empty
            if 38 <= r <= 46 or 38 <= c <= 46:
                continue
            if rng.uniform() < 0.35:
                rows.append(r)
                cols.append(c)
                vals.append(float(rng.uniform(0.1, 5.0)))
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]

    bins = _bins_df(GENE_N_BINS, GENE_RESOLUTION, GENE_CHROM)
    pixels = pd.DataFrame({
        "bin1_id": rows,
        "bin2_id": cols,
        "count": vals.astype(np.float32),
    })
    cool_path = FIXTURE_DIR / "gene_score_small.cool"
    if cool_path.exists():
        cool_path.unlink()
    cooler.create_cooler(cool_uri=str(cool_path), bins=bins, pixels=pixels,
                         ordered=True, dtypes={"count": np.float32})
    print("wrote {} ({} bins, {} nnz)".format(cool_path, GENE_N_BINS, len(pixels)))

    # Raw-mode input: 4-column contact TSV (chrom1=0, pos1=1, chrom2=2, pos2=3).
    # gene_score_raw does (pos - 1) // resolution, so emit 1-based midpoints.
    raw_path = FIXTURE_DIR / "gene_score_small.contact.tsv.gz"
    lines = []
    for r, c in zip(rows, cols):
        n_dup = 1 + int(rng.integers(0, 3))
        for _ in range(n_dup):
            p1 = int(r) * GENE_RESOLUTION + 1
            p2 = int(c) * GENE_RESOLUTION + 1
            lines.append("{}\t{}\t{}\t{}".format(GENE_CHROM, p1, GENE_CHROM, p2))
    with gzip.open(str(raw_path), "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote {} ({} contacts)".format(raw_path, len(lines)))

    starts, ends, ids = zip(*_gene_windows())
    return {
        "gene_score.chrom": np.asarray([GENE_CHROM], dtype="<U16"),
        "gene_score.chrom_size": np.asarray([GENE_CHROM_SIZE], dtype=np.int64),
        "gene_score.resolution": np.asarray(GENE_RESOLUTION, dtype=np.int64),
        "gene_score.gene_start_bin": np.asarray(starts, dtype=np.int64),
        "gene_score.gene_end_bin": np.asarray(ends, dtype=np.int64),
        "gene_score.gene_id": np.asarray(ids, dtype="<U16"),
    }
```

- [ ] **Step 2: Append the contact-distance fixture builder**

Add directly after `gene_score_small_fixture`:

```python
# ---- Phase 5 fixture parameters (contact-distance) ----
CD_RESOLUTION = 10_000
CD_CHROMS = ["chr1", "chr2"]
CD_CHROM_SIZES = [10_000_000, 6_000_000]


def _cd_bin_edges():
    """Exactly upstream's log-spaced edges, from the largest chrom size."""
    nbins = np.floor(np.log2(max(CD_CHROM_SIZES) / 2500) / 0.125)
    return 2500 * np.exp2(0.125 * np.arange(nbins + 1))


def contact_distance_small_fixture(seed: int = 50):
    """7-column contact TSV exercising every filter and histogram edge rule.

    Column layout matches the upstream defaults: chrom1=1, pos1=2, chrom2=5,
    pos2=6, with filler in 0, 3, 4.
    """
    rng = np.random.default_rng(seed)
    edges = _cd_bin_edges()
    rows = []

    def emit(c1, p1, c2, p2):
        rows.append("r{}\t{}\t{}\t+\t-\t{}\t{}".format(len(rows), c1, p1, c2, p2))

    # 1. ordinary cis contacts on both known chroms
    for chrom, size in zip(CD_CHROMS, CD_CHROM_SIZES):
        for _ in range(400):
            p1 = int(rng.integers(0, size - 1))
            span = int(rng.integers(1, min(size - p1, 2_000_000)))
            emit(chrom, p1, chrom, p1 + span)
    # 2. below the first edge (2500 bp) -> dropped by np.histogram
    for _ in range(20):
        p1 = int(rng.integers(0, 1_000_000))
        emit("chr1", p1, "chr1", p1 + int(rng.integers(1, 2000)))
    # 3. exactly the last edge -> kept, lands in the final (right-closed) bin
    emit("chr1", 0, "chr1", int(round(edges[-1])))
    # 4. beyond the last edge -> dropped
    emit("chr1", 0, "chr1", int(round(edges[-1])) + 5000)
    # 5. trans contacts -> dropped by the cis filter
    for _ in range(30):
        emit("chr1", int(rng.integers(0, 5_000_000)), "chr2", int(rng.integers(0, 5_000_000)))
    # 6. unknown chrom -> dropped by the isin filter
    for _ in range(15):
        p1 = int(rng.integers(0, 1_000_000))
        emit("chrUn", p1, "chrUn", p1 + 50_000)
    # 7. duplicate bin pairs -> sparsity counts distinct pairs only
    for _ in range(25):
        emit("chr2", 1_000_000, "chr2", 1_300_000)
    # 8. same-bin contacts -> counted in the histogram, excluded from sparsity
    for _ in range(10):
        emit("chr1", 2_000_000, "chr1", 2_000_500)

    out = FIXTURE_DIR / "contact_distance_small.tsv.gz"
    with gzip.open(str(out), "wt") as fh:
        fh.write("\n".join(rows) + "\n")
    print("wrote {} ({} contacts)".format(out, len(rows)))

    return {
        "contact_distance.chroms": np.asarray(CD_CHROMS, dtype="<U16"),
        "contact_distance.chrom_sizes": np.asarray(CD_CHROM_SIZES, dtype=np.int64),
        "contact_distance.bin_edges": edges.astype(np.float64),
        "contact_distance.resolution": np.asarray(CD_RESOLUTION, dtype=np.int64),
        "contact_distance.cols": np.asarray([1, 2, 5, 6], dtype=np.int64),
    }
```

- [ ] **Step 3: Add the `gzip` import and wire both into `main()`**

At the top of the file, change:
```python
import pathlib
```
to:
```python
import gzip
import pathlib
```

At the end of `main()`, after the embedding block, add:
```python
    # ---- gene_score_small (Phase 5) ----
    gs_pack = gene_score_small_fixture()
    np.savez(FIXTURE_DIR / "gene_score_small.npz", **gs_pack)
    print("wrote {} ({} keys)".format(FIXTURE_DIR / 'gene_score_small.npz', len(gs_pack)))
    print("  genes           = {}".format(gs_pack['gene_score.gene_id'].tolist()))
    # ---- contact_distance_small (Phase 5) ----
    cd_pack = contact_distance_small_fixture()
    np.savez(FIXTURE_DIR / "contact_distance_small.npz", **cd_pack)
    print("wrote {} ({} keys)".format(FIXTURE_DIR / 'contact_distance_small.npz', len(cd_pack)))
    print("  hist bins       = {}".format(cd_pack['contact_distance.bin_edges'].size - 1))
```

- [ ] **Step 4: Generate and eyeball the fixtures**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust python data/fixtures/synthesize.py 2>&1 | tail -12
```
Expected: lines for `gene_score_small.cool`, `gene_score_small.contact.tsv.gz`, `gene_score_small.npz` (6 keys), `contact_distance_small.tsv.gz`, `contact_distance_small.npz` (5 keys), with `hist bins = 95`.

- [ ] **Step 5: Confirm the fixture actually reproduces the bin-0 quirk**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust python -c "
import cooler, numpy as np
from scipy.sparse import triu
c = cooler.Cooler('data/fixtures/gene_score_small.cool')
D = triu(c.matrix(balance=False, sparse=True).fetch('chr1'), k=1).tocsr()
xx, yy = 0, 4
print('bin0 gene score =', D[(xx-1):(yy+1), xx:(yy+2)].sum(), '(must be 0.0)')
print('would-be score  =', D[0:(yy+1), xx:(yy+2)].sum(), '(must be > 0)')
xx, yy = 40, 44
print('empty gene      =', D[(xx-1):(yy+1), xx:(yy+2)].sum(), '(must be 0.0)')
"
```
Expected: `bin0 gene score = 0.0`, `would-be score` strictly positive, `empty gene = 0.0`. If the bin-0 score is not 0.0 the fixture is wrong — stop and fix before continuing.

- [ ] **Step 6: Commit**

```bash
git add data/fixtures/synthesize.py
git commit -m "fixtures: gene_score_small + contact_distance_small, pinning the upstream slice and histogram edge cases"
```

---

## Task 3: Rust gene-score kernel

**Files:**
- Create: `rust/src/gene_score.rs`
- Modify: `rust/src/lib.rs`
- Test: `tests/test_gene_score_semantics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gene_score_semantics.py`:

```python
"""In-env unit parity for the Rust gene-score window-sum kernel.

Runs entirely inside $RUST_TEST_ENV against scipy directly, so it is fast
and needs no cross-env dump. The cross-env manifest gate in
test_exact_match.py is the authoritative check; this file catches
semantic regressions in seconds.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

_rust = pytest.importorskip("schicluster_rs._rust")


def _call(D, row_start, row_end, col_start, col_end):
    D = D.tocsr()
    D.sort_indices()
    return np.asarray(_rust.py_gene_score_chrom(
        np.ascontiguousarray(D.indptr, dtype=np.int64),
        np.ascontiguousarray(D.indices, dtype=np.int64),
        np.ascontiguousarray(D.data, dtype=np.float64),
        int(D.shape[0]), int(D.shape[1]),
        np.ascontiguousarray(row_start, dtype=np.int64),
        np.ascontiguousarray(row_end, dtype=np.int64),
        np.ascontiguousarray(col_start, dtype=np.int64),
        np.ascontiguousarray(col_end, dtype=np.int64),
    ))


def _scipy_ref(D, row_start, row_end, col_start, col_end):
    return np.asarray([
        D[r0:r1, c0:c1].sum()
        for r0, r1, c0, c1 in zip(row_start, row_end, col_start, col_end)
    ], dtype=np.float64)


@pytest.fixture
def dense_csr():
    n = 10
    return csr_matrix(np.arange(n * n).reshape(n, n).astype(np.float64))


def test_negative_row_start_resolves_to_n_minus_1(dense_csr):
    """xx=0 makes the impute window's row start -1; scipy reads that as n-1,
    yielding an empty window. The kernel must reproduce that, not 'fix' it."""
    got = _call(dense_csr, [-1], [4], [0], [5])
    assert got[0] == 0.0
    assert _scipy_ref(dense_csr, [-1], [4], [0], [5])[0] == 0.0


def test_column_overrun_clips(dense_csr):
    got = _call(dense_csr, [7], [10], [8], [11])
    assert got[0] == pytest.approx(_scipy_ref(dense_csr, [7], [10], [8], [11])[0])


def test_empty_and_inverted_windows_are_zero(dense_csr):
    got = _call(dense_csr, [5, 8], [5, 3], [0, 0], [10, 10])
    assert list(got) == [0.0, 0.0]


def test_matches_scipy_on_random_sparse_windows():
    rng = np.random.default_rng(7)
    n = 120
    dense = rng.exponential(1.0, (n, n))
    dense[dense < 1.2] = 0.0
    D = csr_matrix(np.triu(dense, k=1))
    n_g = 300
    r0 = rng.integers(-1, n, n_g)
    r1 = r0 + rng.integers(0, 20, n_g)
    c0 = rng.integers(0, n, n_g)
    c1 = c0 + rng.integers(0, 20, n_g)
    got = _call(D, r0, r1, c0, c1)
    ref = _scipy_ref(D, r0, r1, c0, c1)
    assert np.max(np.abs(got - ref)) < 1e-6


def test_no_genes_returns_empty(dense_csr):
    assert _call(dense_csr, [], [], [], []).size == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust python -m pytest tests/test_gene_score_semantics.py -q 2>&1 | tail -5
```
Expected: FAIL — `AttributeError: module 'schicluster_rs._rust' has no attribute 'py_gene_score_chrom'`.

- [ ] **Step 3: Write the kernel**

Create `rust/src/gene_score.rs`:

```rust
//! Port of scHiCluster/schicluster/draft/gene_score.py's per-gene window-sum
//! loop, shared by both `gene_score_impute` and `gene_score_raw`.
//!
//! Upstream evaluates `D[r0:r1, c0:c1].sum()` once per gene. With ~78.7k genes
//! that is 78.7k scipy submatrix allocations per cell. Here the CSR is visited
//! in place: for each row in the window, the sorted column indices are binary
//! searched for the column range and the matching `data` slice is summed.
//!
//! Parallelism is across genes only. Each gene's reduction stays serial and in
//! CSR row-major order, so this is admissibility class (E) relative to a serial
//! Rust baseline. It is NOT bit-equal to numpy's pairwise `.sum()`, which is why
//! `gene_score.impute` gates as deterministic-bounded at 1e-6 rather than strict.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Resolve one Python slice bound against an axis of length `n`, exactly as
/// CPython (and therefore scipy's sparse `__getitem__`) does: negative values
/// are offset by `n`, then the result is clamped into `[0, n]`.
///
/// This is what makes `gene_score_impute`'s `(xx-1)` start behave as `n-1` when
/// `xx == 0`, producing an empty window and a score of 0.0.
fn resolve_bound(v: i64, n: usize) -> usize {
    let n_i = n as i64;
    let r = if v < 0 { v + n_i } else { v };
    if r < 0 {
        0
    } else if r > n_i {
        n
    } else {
        r as usize
    }
}

/// Sum a CSR matrix over each of `n_genes` rectangular windows.
pub fn window_sums(
    indptr: &[i64],
    indices: &[i64],
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    row_start: &[i64],
    row_end: &[i64],
    col_start: &[i64],
    col_end: &[i64],
) -> Vec<f64> {
    (0..row_start.len())
        .into_par_iter()
        .map(|g| {
            let r0 = resolve_bound(row_start[g], n_rows);
            let r1 = resolve_bound(row_end[g], n_rows);
            let c0 = resolve_bound(col_start[g], n_cols);
            let c1 = resolve_bound(col_end[g], n_cols);
            if r0 >= r1 || c0 >= c1 {
                return 0.0;
            }
            let c0 = c0 as i64;
            let c1 = c1 as i64;
            let mut acc = 0.0f64;
            for r in r0..r1 {
                let lo = indptr[r] as usize;
                let hi = indptr[r + 1] as usize;
                let row_cols = &indices[lo..hi];
                // scipy guarantees sorted indices on a canonical CSR; the
                // Python wrapper calls sort_indices() defensively.
                let a = row_cols.partition_point(|&c| c < c0);
                let b = row_cols.partition_point(|&c| c < c1);
                for k in a..b {
                    acc += data[lo + k];
                }
            }
            acc
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_rows, n_cols, row_start, row_end, col_start, col_end))]
pub fn py_gene_score_chrom<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_rows: usize,
    n_cols: usize,
    row_start: PyReadonlyArray1<'py, i64>,
    row_end: PyReadonlyArray1<'py, i64>,
    col_start: PyReadonlyArray1<'py, i64>,
    col_end: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = data.as_slice()?;
    let row_start = row_start.as_slice()?;
    let row_end = row_end.as_slice()?;
    let col_start = col_start.as_slice()?;
    let col_end = col_end.as_slice()?;

    if indptr.len() != n_rows + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indptr length {} does not match n_rows + 1 = {}",
            indptr.len(),
            n_rows + 1
        )));
    }
    if indices.len() != data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "indices and data must have equal length",
        ));
    }
    let n_genes = row_start.len();
    if row_end.len() != n_genes || col_start.len() != n_genes || col_end.len() != n_genes {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "row_start / row_end / col_start / col_end must have equal length",
        ));
    }

    let out = py.allow_threads(|| {
        window_sums(
            indptr, indices, data, n_rows, n_cols, row_start, row_end, col_start, col_end,
        )
    });
    Ok(Array1::from_vec(out).into_pyarray_bound(py))
}
```

- [ ] **Step 4: Register the module**

In `rust/src/lib.rs`, add to the `mod` list (after `mod embedding;`):
```rust
mod gene_score;
```

and inside the `#[pymodule]` body, after the `embedding::py_make_chrom_features` line:
```rust
    m.add_function(wrap_pyfunction!(gene_score::py_gene_score_chrom, m)?)?;
```

- [ ] **Step 5: Build and run the test**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust maturin develop --release 2>&1 | tail -3
conda run -n rebuild-rust python -m pytest tests/test_gene_score_semantics.py -q 2>&1 | tail -5
```
Expected: build succeeds, then `5 passed`.

- [ ] **Step 6: Commit**

```bash
git add rust/src/gene_score.rs rust/src/lib.rs tests/test_gene_score_semantics.py
git commit -m "feat(rust): gene_score window-sum kernel, replicating scipy slice-bound semantics"
```

---

## Task 4: Python gene-score wrappers

**Files:**
- Create: `python/schicluster_rs/gene_score/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gene_score_semantics.py`:

```python
def test_wrapper_matches_upstream_on_the_fixture():
    """End-to-end within this env: our impute wrapper vs a literal transcription
    of upstream's loop, on the committed fixture."""
    cooler = pytest.importorskip("cooler")
    import pandas as pd
    from scipy.sparse import triu
    from schicluster_rs.gene_score import gene_score_impute

    pack = np.load("data/fixtures/gene_score_small.npz", allow_pickle=False)
    chrom_sizes = pd.Series(pack["gene_score.chrom_size"],
                            index=[str(c) for c in pack["gene_score.chrom"]])
    gene_meta = pd.DataFrame(
        {0: [str(pack["gene_score.chrom"][0])] * pack["gene_score.gene_id"].size,
         1: pack["gene_score.gene_start_bin"],
         2: pack["gene_score.gene_end_bin"]},
        index=[str(g) for g in pack["gene_score.gene_id"]],
    )
    cool_path = "data/fixtures/gene_score_small.cool"

    def upstream(cell_path, chrom_sizes, gene_meta):
        cool = cooler.Cooler(cell_path)
        result = []
        for chrom in chrom_sizes.index:
            D = triu(cool.matrix(balance=False, sparse=True).fetch(chrom), k=1).tocsr()
            gene = gene_meta.loc[gene_meta[0] == chrom, [1, 2]].values
            for xx, yy in gene:
                result.append(D[(xx - 1):(yy + 1), xx:(yy + 2)].sum())
        return result

    ref = np.asarray(upstream(cool_path, chrom_sizes, gene_meta), dtype=np.float64)
    got = np.asarray(gene_score_impute(cool_path, chrom_sizes, gene_meta), dtype=np.float64)
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) < 1e-6
    # the bin-0 gene must be zero in BOTH
    assert ref[0] == 0.0 and got[0] == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust python -m pytest tests/test_gene_score_semantics.py::test_wrapper_matches_upstream_on_the_fixture -q 2>&1 | tail -5
```
Expected: FAIL — `ModuleNotFoundError: No module named 'schicluster_rs.gene_score'`.

- [ ] **Step 3: Write the wrappers**

Create `python/schicluster_rs/gene_score/__init__.py`:

```python
"""Rust-backed replacements for scHiCluster's per-cell gene-score workers.

Drop-in for schicluster.draft.gene_score.gene_score_impute / gene_score_raw.
Cooler reads (impute mode) and the pandas groupby matrix build (raw mode) stay
in Python; only the per-gene window-sum loop crosses into Rust.

Upstream quirk preserved deliberately: gene_score_impute's window is
D[(xx-1):(yy+1), xx:(yy+2)]. When xx == 0 the row start is -1, which scipy
resolves to n-1, so the window is empty and the gene scores 0.0. Porting that
"correctly" would silently change every first-bin gene's score, so the Rust
kernel reproduces the slice semantics instead. See tutorial/gene_score.md.
"""
from __future__ import annotations

import numpy as np


def _window_sums(matrix, row_start, row_end, col_start, col_end):
    """Sum `matrix` over each rectangular window, via the Rust kernel."""
    from schicluster_rs._rust import py_gene_score_chrom

    csr = matrix.tocsr()
    csr.sort_indices()
    return np.asarray(py_gene_score_chrom(
        np.ascontiguousarray(csr.indptr, dtype=np.int64),
        np.ascontiguousarray(csr.indices, dtype=np.int64),
        np.ascontiguousarray(csr.data, dtype=np.float64),
        int(csr.shape[0]), int(csr.shape[1]),
        np.ascontiguousarray(row_start, dtype=np.int64),
        np.ascontiguousarray(row_end, dtype=np.int64),
        np.ascontiguousarray(col_start, dtype=np.int64),
        np.ascontiguousarray(col_end, dtype=np.int64),
    ))


def gene_score_impute(cell_path, chrom_sizes, gene_meta):
    """Per-cell gene scores from an imputed .cool. Signature matches upstream."""
    import cooler

    cool = cooler.Cooler(cell_path)
    result = []
    for chrom in chrom_sizes.index:
        from scipy.sparse import triu
        matrix = triu(cool.matrix(balance=False, sparse=True).fetch(chrom), k=1).tocsr()
        gene = gene_meta.loc[gene_meta[0] == chrom, [1, 2]].values
        if gene.size == 0:
            continue
        xx = gene[:, 0].astype(np.int64)
        yy = gene[:, 1].astype(np.int64)
        result.extend(_window_sums(matrix, xx - 1, yy + 1, xx, yy + 2).tolist())
    return result


def gene_score_raw(cell_path, chrom_sizes, gene_meta, resolution,
                   chrom1, pos1, chrom2, pos2):
    """Per-cell gene scores straight from a contact file. Signature matches upstream.

    The matrix build below is a literal transcription of upstream so that the
    only behavioural difference is the window-sum kernel.
    """
    import pandas as pd
    from scipy.sparse import csr_matrix

    data = pd.read_csv(cell_path, sep='\t', index_col=None, header=None, comment='#')
    data = data.loc[(data[chrom1] == data[chrom2]) & data[chrom1].isin(chrom_sizes.index)]
    result = []
    for chrom in chrom_sizes.index:
        n_bins = (chrom_sizes.loc[chrom] // resolution) + 1
        chrfilter = (data[chrom1] == chrom)
        if chrfilter.sum() == 0:
            matrix = csr_matrix((n_bins, n_bins))
        else:
            D = data.loc[chrfilter].copy()
            D[[pos1, pos2]] = (D[[pos1, pos2]] - 1) // resolution
            D = D.groupby(by=[pos1, pos2])[chrom1].count().reset_index()
            matrix = csr_matrix((D[chrom1].astype(np.int32), (D[pos1], D[pos2])),
                                (n_bins, n_bins))
        gene = gene_meta.loc[gene_meta[0] == chrom, [1, 2]].values
        if gene.size == 0:
            continue
        xx = gene[:, 0].astype(np.int64)
        yy = gene[:, 1].astype(np.int64)
        # Raw mode carries int32 counts; the sums are exact integers, and
        # upstream's .sum() returns numpy.int64, so cast to match its dtype.
        sums = _window_sums(matrix, xx, yy + 1, xx, yy + 1)
        result.extend(np.rint(sums).astype(np.int64).tolist())
    return result


__all__ = ["gene_score_impute", "gene_score_raw"]
```

- [ ] **Step 4: Run the test**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust python -m pytest tests/test_gene_score_semantics.py -q 2>&1 | tail -5
```
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add python/schicluster_rs/gene_score/__init__.py tests/test_gene_score_semantics.py
git commit -m "feat(py): gene_score_impute / gene_score_raw wrappers over the Rust kernel"
```

---

## Task 5: Rust contact-distance kernel

**Files:**
- Modify: `rust/Cargo.toml`
- Create: `rust/src/contact_distance.rs`
- Modify: `rust/src/lib.rs`
- Test: `tests/test_contact_distance_semantics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_contact_distance_semantics.py`:

```python
"""In-env unit parity for the Rust contact-distance reader.

Compares against a literal pandas/numpy transcription of upstream's
compute_decay, on the committed fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_rust = pytest.importorskip("schicluster_rs._rust")

FIXTURE = "data/fixtures/contact_distance_small.tsv.gz"
PACK = np.load("data/fixtures/contact_distance_small.npz", allow_pickle=False)
CHROMS = [str(c) for c in PACK["contact_distance.chroms"]]
EDGES = PACK["contact_distance.bin_edges"]
RESOLUTION = int(PACK["contact_distance.resolution"])
C1, P1, C2, P2 = (int(x) for x in PACK["contact_distance.cols"])


def _upstream_reference():
    chrom_sizes = pd.Series(PACK["contact_distance.chrom_sizes"], index=CHROMS)
    data = pd.read_csv(FIXTURE, sep='\t', header=None, index_col=None)
    data = data.loc[(data[C1] == data[C2]) & data[C1].isin(chrom_sizes.index)]
    hist = np.histogram(np.abs(data[P2] - data[P1]), EDGES)[0]
    data[[P1, P2]] = data[[P1, P2]] // RESOLUTION
    grouped = data.groupby(by=[C1, P1, P2])[C2].count().reset_index()
    sparsity = grouped.loc[grouped[P1] != grouped[P2], C1].value_counts()
    return hist, sparsity


def _rust_call():
    hist, sparsity = _rust.py_contact_decay_cell(
        FIXTURE, CHROMS, np.ascontiguousarray(EDGES, dtype=np.float64),
        RESOLUTION, C1, P1, C2, P2,
    )
    return np.asarray(hist, dtype=np.int64), dict(sparsity)


def test_decay_histogram_matches_numpy():
    ref_hist, _ = _upstream_reference()
    got_hist, _ = _rust_call()
    assert got_hist.tolist() == ref_hist.tolist()


def test_sparsity_matches_pandas_groupby():
    _, ref_sparsity = _upstream_reference()
    _, got_sparsity = _rust_call()
    assert got_sparsity == {str(k): int(v) for k, v in ref_sparsity.items()}


def test_out_of_range_distances_are_dropped():
    """The fixture emits one contact at exactly edges[-1] (kept, final bin) and
    one beyond it (dropped), plus 20 below edges[0] (dropped)."""
    ref_hist, _ = _upstream_reference()
    got_hist, _ = _rust_call()
    assert got_hist[-1] == ref_hist[-1] >= 1


def test_unknown_chrom_and_trans_are_excluded():
    _, got_sparsity = _rust_call()
    assert "chrUn" not in got_sparsity
    assert set(got_sparsity).issubset(set(CHROMS))


def test_missing_file_raises():
    with pytest.raises(Exception):
        _rust.py_contact_decay_cell(
            "data/fixtures/does_not_exist.tsv.gz", CHROMS,
            np.ascontiguousarray(EDGES, dtype=np.float64),
            RESOLUTION, C1, P1, C2, P2,
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust python -m pytest tests/test_contact_distance_semantics.py -q 2>&1 | tail -5
```
Expected: FAIL — `AttributeError: module 'schicluster_rs._rust' has no attribute 'py_contact_decay_cell'`.

- [ ] **Step 3: Add the flate2 dependency**

In `rust/Cargo.toml`, under `[dependencies]`, after the `rayon` line, add:
```toml
flate2 = "1.0"
```

Default features give the pure-Rust `rust_backend` (miniz_oxide) — no system zlib, so the cibuildwheel matrix stays portable.

- [ ] **Step 4: Write the kernel**

Create `rust/src/contact_distance.rs`:

```rust
//! Port of scHiCluster/schicluster/cool/contact_distance.py::compute_decay.
//!
//! Upstream reads the whole gzipped contact TSV into a pandas DataFrame just to
//! use four columns. Here the file is streamed line by line in constant memory
//! and no DataFrame is ever built. This is the one place in the port where Rust
//! owns file I/O; see the design spec 4.2.
//!
//! Both outputs are integer counts, exact under any summation order, so they
//! gate as deterministic-strict.

use flate2::read::MultiGzDecoder;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

/// Locate `v` in `edges` under `np.histogram`'s rules: bins are right-open
/// except the final bin, which is right-closed; values outside
/// `[edges[0], edges[last]]` are dropped entirely.
fn hist_bin(edges: &[f64], v: f64) -> Option<usize> {
    let nb = edges.len() - 1;
    if !(v >= edges[0] && v <= edges[nb]) {
        return None;
    }
    if v == edges[nb] {
        return Some(nb - 1);
    }
    // partition_point counts edges <= v; v >= edges[0] guarantees count >= 1.
    let i = edges.partition_point(|&e| e <= v) - 1;
    Some(i.min(nb - 1))
}

fn open_maybe_gzip(path: &str) -> std::io::Result<Box<dyn Read>> {
    let file = File::open(path)?;
    if path.ends_with(".gz") {
        // MultiGzDecoder, not GzDecoder: concatenated gzip members are common
        // in contact files produced by streaming writers.
        Ok(Box::new(MultiGzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

pub struct Decay {
    pub hist: Vec<u64>,
    pub sparsity: Vec<(String, u64)>,
}

#[allow(clippy::too_many_arguments)]
pub fn compute_decay(
    path: &str,
    chroms: &[String],
    edges: &[f64],
    resolution: i64,
    chrom1: usize,
    pos1: usize,
    chrom2: usize,
    pos2: usize,
) -> Result<Decay, String> {
    let known: HashSet<&str> = chroms.iter().map(|s| s.as_str()).collect();
    let n_bins = edges.len() - 1;
    let mut hist = vec![0u64; n_bins];
    let mut pairs: HashMap<String, HashSet<(i64, i64)>> = HashMap::new();

    let reader = BufReader::new(open_maybe_gzip(path).map_err(|e| format!("{}: {}", path, e))?);
    let max_col = chrom1.max(pos1).max(chrom2).max(pos2);

    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("{}:{}: {}", path, lineno + 1, e))?;
        let line = line.trim_end_matches(['\r', '\n']);
        // pandas read_csv defaults to skip_blank_lines=True.
        if line.trim().is_empty() {
            continue;
        }
        let mut c1 = "";
        let mut p1 = "";
        let mut c2 = "";
        let mut p2 = "";
        let mut seen = 0usize;
        for (i, f) in line.split('\t').enumerate() {
            if i == chrom1 {
                c1 = f;
            }
            if i == pos1 {
                p1 = f;
            }
            if i == chrom2 {
                c2 = f;
            }
            if i == pos2 {
                p2 = f;
            }
            seen = i;
        }
        if seen < max_col {
            return Err(format!(
                "{}:{}: expected at least {} tab-separated fields, found {}",
                path,
                lineno + 1,
                max_col + 1,
                seen + 1
            ));
        }
        // Cis contacts on known chroms only — upstream's two filters, in order.
        if c1 != c2 || !known.contains(c1) {
            continue;
        }
        // Upstream does NOT pass comment='#' here (unlike filter-contact), so a
        // '#' line is a hard failure upstream too. Fail loudly rather than skip.
        let p1: i64 = p1
            .trim()
            .parse()
            .map_err(|_| format!("{}:{}: cannot parse pos1 {:?} as an integer", path, lineno + 1, p1))?;
        let p2: i64 = p2
            .trim()
            .parse()
            .map_err(|_| format!("{}:{}: cannot parse pos2 {:?} as an integer", path, lineno + 1, p2))?;

        // Histogram uses RAW positions; upstream only floor-divides afterwards.
        if let Some(b) = hist_bin(edges, (p2 - p1).abs() as f64) {
            hist[b] += 1;
        }
        let b1 = p1.div_euclid(resolution);
        let b2 = p2.div_euclid(resolution);
        if b1 != b2 {
            // Ordered pair, matching upstream's groupby on (chrom, pos1, pos2).
            pairs.entry(c1.to_string()).or_default().insert((b1, b2));
        }
    }

    // Chroms with no surviving off-diagonal pair are absent from upstream's
    // value_counts() output, so omit rather than emit zeros. Sorted by chrom
    // name so both parity dumps agree on ordering.
    let mut sparsity: Vec<(String, u64)> = pairs
        .into_iter()
        .map(|(k, v)| (k, v.len() as u64))
        .collect();
    sparsity.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(Decay { hist, sparsity })
}

#[pyfunction]
#[pyo3(signature = (path, chroms, bin_edges, resolution, chrom1, pos1, chrom2, pos2))]
#[allow(clippy::too_many_arguments)]
pub fn py_contact_decay_cell<'py>(
    py: Python<'py>,
    path: &str,
    chroms: Vec<String>,
    bin_edges: PyReadonlyArray1<'py, f64>,
    resolution: i64,
    chrom1: usize,
    pos1: usize,
    chrom2: usize,
    pos2: usize,
) -> PyResult<(Vec<u64>, Vec<(String, u64)>)> {
    let edges = bin_edges.as_slice()?;
    if edges.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bin_edges must contain at least 2 edges",
        ));
    }
    if resolution <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "resolution must be positive",
        ));
    }
    let out = py
        .allow_threads(|| {
            compute_decay(path, &chroms, edges, resolution, chrom1, pos1, chrom2, pos2)
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((out.hist, out.sparsity))
}
```

- [ ] **Step 5: Register the module**

In `rust/src/lib.rs`, add after `mod gene_score;`:
```rust
mod contact_distance;
```

and inside `#[pymodule]`, after the `gene_score::py_gene_score_chrom` line:
```rust
    m.add_function(wrap_pyfunction!(contact_distance::py_contact_decay_cell, m)?)?;
```

- [ ] **Step 6: Build and run the test**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust maturin develop --release 2>&1 | tail -3
conda run -n rebuild-rust python -m pytest tests/test_contact_distance_semantics.py -q 2>&1 | tail -8
```
Expected: build succeeds (flate2 + miniz_oxide compile), then `5 passed`.

- [ ] **Step 7: Commit**

```bash
git add rust/Cargo.toml rust/Cargo.lock rust/src/contact_distance.rs rust/src/lib.rs tests/test_contact_distance_semantics.py
git commit -m "feat(rust): streaming contact-distance reader, replacing the pandas DataFrame build"
```

---

## Task 6: Python contact-distance wrapper

**Files:**
- Create: `python/schicluster_rs/contact_distance/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_contact_distance_semantics.py`:

```python
def test_wrapper_returns_upstream_frame_shapes():
    from schicluster_rs.contact_distance import compute_decay

    chrom_sizes = pd.DataFrame(PACK["contact_distance.chrom_sizes"], index=CHROMS)
    sparsity_df, decay_df = compute_decay(
        cell_name="cell_A", contact_path=FIXTURE, bins=EDGES,
        chrom_sizes=chrom_sizes, resolution=RESOLUTION,
        chrom1=C1, pos1=P1, chrom2=C2, pos2=P2,
    )
    ref_hist, ref_sparsity = _upstream_reference()
    assert list(decay_df.columns) == ["cell_A"]
    assert list(sparsity_df.columns) == ["cell_A"]
    assert decay_df["cell_A"].tolist() == ref_hist.tolist()
    for chrom, count in ref_sparsity.items():
        assert int(sparsity_df.loc[str(chrom), "cell_A"]) == int(count)
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust python -m pytest tests/test_contact_distance_semantics.py::test_wrapper_returns_upstream_frame_shapes -q 2>&1 | tail -5
```
Expected: FAIL — `ModuleNotFoundError: No module named 'schicluster_rs.contact_distance'`.

- [ ] **Step 3: Write the wrapper**

Create `python/schicluster_rs/contact_distance/__init__.py`:

```python
"""Rust-backed replacement for scHiCluster's per-cell contact-distance worker.

Drop-in for schicluster.cool.contact_distance.compute_decay. Returns the same
[sparsity_frame, decay_frame] pair the upstream orchestrator concatenates, so
pd.concat / to_hdf downstream are untouched.

The bin edges are computed by numpy in the orchestrator and passed straight
through, so Rust never recomputes exp2 and there is no ULP drift to reason
about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_decay(cell_name, contact_path, bins, chrom_sizes, resolution,
                  chrom1=1, chrom2=5, pos1=2, pos2=6):
    """Distance-decay histogram + per-chrom sparsity for one cell."""
    from schicluster_rs._rust import py_contact_decay_cell

    hist, sparsity = py_contact_decay_cell(
        str(contact_path),
        [str(c) for c in chrom_sizes.index],
        np.ascontiguousarray(bins, dtype=np.float64),
        int(resolution),
        int(chrom1), int(pos1), int(chrom2), int(pos2),
    )
    # Upstream builds this from value_counts(), so chroms with no surviving
    # off-diagonal pair are simply absent. Preserve that.
    sparsity_series = pd.Series(
        {str(k): int(v) for k, v in sparsity},
        dtype='int64',
    )
    return [
        pd.DataFrame(sparsity_series).set_axis([cell_name], axis=1),
        pd.DataFrame(np.asarray(hist, dtype=np.int64), columns=[cell_name]),
    ]


__all__ = ["compute_decay"]
```

- [ ] **Step 4: Run the test**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust python -m pytest tests/test_contact_distance_semantics.py -q 2>&1 | tail -5
```
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add python/schicluster_rs/contact_distance/__init__.py tests/test_contact_distance_semantics.py
git commit -m "feat(py): compute_decay wrapper returning upstream's frame pair"
```

---

## Task 7: Wire into patch_schicluster() and the CLI

**Files:**
- Modify: `python/schicluster_rs/__init__.py`
- Modify: `python/schicluster_rs/__main__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_patch_wiring.py`:

```python
"""patch_schicluster() must rebind the two new upstream workers in place."""
from __future__ import annotations

import pytest

pytest.importorskip("schicluster")
pytest.importorskip("schicluster_rs._rust")


def test_patch_rebinds_gene_score_and_contact_distance():
    import schicluster_rs
    from schicluster_rs.gene_score import gene_score_impute, gene_score_raw
    from schicluster_rs.contact_distance import compute_decay

    assert schicluster_rs.patch_schicluster() is True

    from schicluster.draft import gene_score as gs_mod
    from schicluster.cool import contact_distance as cd_mod

    assert gs_mod.gene_score_impute is gene_score_impute
    assert gs_mod.gene_score_raw is gene_score_raw
    assert cd_mod.compute_decay is compute_decay


def test_cli_lists_the_new_subcommands():
    from schicluster_rs.__main__ import _SUPPORTED

    assert "gene-score" in _SUPPORTED
    assert "contact-distance" in _SUPPORTED
```

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust python -m pytest tests/test_patch_wiring.py -q 2>&1 | tail -5
```
Expected: FAIL (assertion on `gs_mod.gene_score_impute`, or a skip if upstream `schicluster` is not installed in `rebuild-rust` — if it skips, that is acceptable and this test will be exercised by the reference/candidate drivers instead).

- [ ] **Step 2: Re-export from the package root**

In `python/schicluster_rs/__init__.py`, after the existing embedding import line
(`from schicluster_rs.embedding import make_chrom_matrix as _make_chrom_matrix`), add:

```python
from schicluster_rs.gene_score import (
    gene_score_impute as _gene_score_impute,
    gene_score_raw as _gene_score_raw,
)
from schicluster_rs.contact_distance import compute_decay as _compute_decay
```

and after the existing `make_chrom_matrix = _make_chrom_matrix` line, add:

```python
gene_score_impute = _gene_score_impute
gene_score_raw = _gene_score_raw
compute_decay = _compute_decay
```

- [ ] **Step 3: Rebind inside patch_schicluster()**

In `patch_schicluster()`, immediately after the embedding block
(`_emb_mod.make_chrom_matrix = make_chrom_matrix`) and before `return True`, add:

```python
        # ---- gene-score module (Phase 5) ----
        # The orchestrator resolves these names from module globals at
        # exe.submit() time, so rebinding here is enough for the existing
        # ProcessPoolExecutor to pick up the Rust versions.
        from schicluster.draft import gene_score as _gene_score_mod
        _gene_score_mod.gene_score_impute = gene_score_impute
        _gene_score_mod.gene_score_raw = gene_score_raw
        # ---- contact-distance module (Phase 5) ----
        from schicluster.cool import contact_distance as _contact_distance_mod
        _contact_distance_mod.compute_decay = compute_decay
```

- [ ] **Step 4: Extend `__all__`**

Change the `__all__` list's final entry line from:
```python
    "single_chrom_compartment", "make_chrom_matrix",
]
```
to:
```python
    "single_chrom_compartment", "make_chrom_matrix",
    "gene_score_impute", "gene_score_raw", "compute_decay",
]
```

- [ ] **Step 5: Add both subcommands to the CLI**

In `python/schicluster_rs/__main__.py`, in `_SUPPORTED`, after the `"cpg-ratio"` entry add:
```python
    "gene-score",        # per-gene CSR window sums (Phase 5)
    "contact-distance",  # streaming gzip contact reader (Phase 5)
```

In the help text block, after the `cpg-ratio` line add:
```
                gene-score          Per-gene contact scores from imputed or raw contacts.
                contact-distance    Per-cell distance decay + per-chrom sparsity.
```

In the module docstring's subcommand list, after the `cpg-ratio` bullet add:
```
* ``gene-score``     - per-gene rectangular window sums over the per-chrom
  CSR, binary-searched in place instead of allocating one scipy submatrix
  per gene. Both --mode impute and --mode raw are Rust-backed; impute mode
  is where the win is, since raw mode still builds its matrix in pandas.
* ``contact-distance`` - streams the gzipped contact TSV in Rust and
  histograms distances without ever building a DataFrame.
```

- [ ] **Step 6: Run the test**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust maturin develop --release 2>&1 | tail -2
conda run -n rebuild-rust python -m pytest tests/test_patch_wiring.py -q 2>&1 | tail -5
```
Expected: `2 passed` (or `2 skipped` if upstream `schicluster` is absent from `rebuild-rust`).

- [ ] **Step 7: Commit**

```bash
git add python/schicluster_rs/__init__.py python/schicluster_rs/__main__.py tests/test_patch_wiring.py
git commit -m "feat: patch_schicluster() rebinds gene-score + contact-distance; expose both in the CLI"
```

---

## Task 8: Reference driver blocks (Python 3.6)

**Files:**
- Modify: `tests/py_reference_driver.py`

**Python 3.6 only.** No f-strings, no `from __future__ import annotations`, no PEP 585 generics.

- [ ] **Step 1: Add fixture path constants**

After the existing `EMBEDDING_FIXTURE` constant, add:
```python
GENE_SCORE_FIXTURE = REPO_ROOT / "data" / "fixtures" / "gene_score_small.npz"
GENE_SCORE_COOL = REPO_ROOT / "data" / "fixtures" / "gene_score_small.cool"
GENE_SCORE_CONTACTS = REPO_ROOT / "data" / "fixtures" / "gene_score_small.contact.tsv.gz"
CONTACT_DISTANCE_FIXTURE = REPO_ROOT / "data" / "fixtures" / "contact_distance_small.npz"
CONTACT_DISTANCE_TSV = REPO_ROOT / "data" / "fixtures" / "contact_distance_small.tsv.gz"
```

- [ ] **Step 2: Add the upstream imports**

After the existing `from schicluster.embedding.calc_embedding import make_idx as upstream_make_idx` line, add:
```python
from schicluster.draft.gene_score import (
    gene_score_impute as upstream_gene_score_impute,
    gene_score_raw as upstream_gene_score_raw,
)
from schicluster.cool.contact_distance import compute_decay as upstream_compute_decay
```

- [ ] **Step 3: Add the three reference functions**

Add before `def main()`:

```python
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
```

- [ ] **Step 4: Wire into `main()`**

Inside the `try:` block, after the embedding line, add:
```python
        gs_pack = _load_npz(GENE_SCORE_FIXTURE)
        payload["gene_score"] = {
            "impute": ref_gene_score_impute(gs_pack),
            "raw": ref_gene_score_raw(gs_pack),
        }

        cd_pack = _load_npz(CONTACT_DISTANCE_FIXTURE)
        cd_decay, cd_sparsity = ref_contact_distance(cd_pack)
        payload["contact_distance"] = {"decay": cd_decay, "sparsity": cd_sparsity}
```

- [ ] **Step 5: Run the reference driver**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n schicluster --no-capture-output python tests/py_reference_driver.py 2>&1 | tail -4
conda run -n rebuild-rust python -c "
import json
d = json.load(open('data/fixtures/reference_output.json'))
print('gene_score.impute[:3] =', d['gene_score']['impute'][:3])
print('gene_score.raw[:3]    =', d['gene_score']['raw'][:3])
print('decay sum             =', sum(d['contact_distance']['decay']))
print('sparsity              =', d['contact_distance']['sparsity'])
assert d['gene_score']['impute'][0] == 0.0, 'bin-0 gene must score 0 upstream'
print('OK')
"
```
Expected: the driver prints its top-level keys including `gene_score` and `contact_distance`; the assertion passes.

- [ ] **Step 6: Commit**

```bash
git add tests/py_reference_driver.py
git commit -m "tests: reference dumps for gene_score + contact_distance (py3.6 ref env)"
```

---

## Task 9: Candidate driver blocks and the full gate

**Files:**
- Modify: `tests/_run_candidate.py`

- [ ] **Step 1: Add fixture path constants and imports**

Mirror the reference driver. After the existing embedding fixture constant, add:
```python
GENE_SCORE_FIXTURE = REPO_ROOT / "data" / "fixtures" / "gene_score_small.npz"
GENE_SCORE_COOL = REPO_ROOT / "data" / "fixtures" / "gene_score_small.cool"
GENE_SCORE_CONTACTS = REPO_ROOT / "data" / "fixtures" / "gene_score_small.contact.tsv.gz"
CONTACT_DISTANCE_FIXTURE = REPO_ROOT / "data" / "fixtures" / "contact_distance_small.npz"
CONTACT_DISTANCE_TSV = REPO_ROOT / "data" / "fixtures" / "contact_distance_small.tsv.gz"
```

- [ ] **Step 2: Add the candidate functions**

Add before the driver's `main()`:

```python
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
```

- [ ] **Step 3: Wire into `main()`**

After the embedding payload line, add:
```python
    gs_pack = _load_npz(GENE_SCORE_FIXTURE)
    payload["gene_score"] = {
        "impute": cand_gene_score_impute(gs_pack),
        "raw": cand_gene_score_raw(gs_pack),
    }

    cd_pack = _load_npz(CONTACT_DISTANCE_FIXTURE)
    cd_decay, cd_sparsity = cand_contact_distance(cd_pack)
    payload["contact_distance"] = {"decay": cd_decay, "sparsity": cd_sparsity}
```

If `_run_candidate.py` does not already `import pandas as pd`, add it to its imports.

- [ ] **Step 4: Run the full two-env gate**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
REBUILDPY_DIR=/large_storage/zhoulab/shengmao/rebuildpy \
PYTHONPATH=/large_storage/zhoulab/shengmao/rebuildpy \
bash tests/run_parity.sh 2>&1 | tail -25
```
Expected: `21 passed`. If any of the four new outputs fails, do **not** touch `data/manifest.yaml` — fix the port. The most likely culprits, in order:
1. `gene_score.impute` mismatch at index 0 → the negative-slice resolution in `resolve_bound` is wrong.
2. `gene_score.raw` off by a constant → the `(pos - 1) // resolution` in the raw wrapper was dropped or duplicated.
3. `contact_distance.decay` off by one in the last bin → `hist_bin`'s right-closed special case.
4. `contact_distance.sparsity` too large → pairs are being sorted, or same-bin pairs are not excluded.

- [ ] **Step 5: Run the in-env test suite too**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
PYTHONNOUSERSITE=1 conda run -n rebuild-rust python -m pytest tests/ -q 2>&1 | tail -6
```
Expected: all green, no regressions in `test_parity.py` or the 17 pre-existing manifest outputs.

- [ ] **Step 6: Commit**

```bash
git add tests/_run_candidate.py
git commit -m "tests: candidate dumps for gene_score + contact_distance; manifest gate 21/21"
```

---

## Task 10: Record the phase in ITERATION_LOG and bump the version

**Files:**
- Modify: `docs/ITERATION_LOG.md`
- Modify: `pyproject.toml`
- Modify: `rust/Cargo.toml`

- [ ] **Step 1: Append the iteration blocks**

Append to `docs/ITERATION_LOG.md`, filling `<fill>` from the actual gate run:

```yaml
iteration: 4
title: Phase 5 — gene-score ported (per-gene CSR window sums)
admissibility: E
action: |
  Rust port of the per-gene window-sum loop shared by gene_score_impute and
  gene_score_raw (gene_score.rs). Upstream evaluates D[r0:r1, c0:c1].sum()
  once per gene — 78,691 scipy submatrix allocations per cell on the
  reference workload. The kernel instead binary-searches each row's sorted
  column indices for the column range and sums the matching data slice in
  place. Rayon parallelises across genes; each gene's reduction stays serial
  and in CSR row-major order.

  Cooler reads (impute) and the pandas groupby matrix build (raw) stay Python.
status: accepted
fixture: data/fixtures/gene_score_small.{npz,cool,contact.tsv.gz}
parity:
  gene_score.impute: { class: deterministic-bounded, threshold: 1.0e-6, pass: true, metric: <fill> }
  gene_score.raw:    { class: deterministic-strict,  threshold: 0.0,    pass: true, metric: <fill> }
notes: |
  gene_score.impute cannot gate deterministic-strict: scipy's .sum() is
  np.add.reduce, which uses pairwise summation with a 128-element block, and
  the kernel accumulates left-to-right. Observed drift is far below the 1e-6
  gate. raw mode carries int32 counts, so its sums are exact under any order
  and it does gate strict.

  The fixture deliberately pins upstream's D[(xx-1):(yy+1), ...] quirk: when
  xx == 0 the row start is -1, which scipy resolves to n-1, so the window is
  empty and the gene scores 0.0. resolve_bound() reproduces CPython slice
  semantics rather than "fixing" this — porting it as [0:(yy+1)] would
  silently change every first-bin gene's score.

---

iteration: 5
title: Phase 5 — contact-distance ported (streaming gzip reader)
admissibility: E
action: |
  Rust port of compute_decay (contact_distance.rs). Upstream builds a pandas
  DataFrame of every contact just to use four columns; the kernel streams the
  gzipped TSV line by line in constant memory via flate2's MultiGzDecoder,
  histograms |pos2-pos1| over the caller-supplied log-spaced edges, and counts
  distinct off-diagonal bin pairs per chrom in a HashSet.

  First I/O dependency in the crate (flate2 1.0, pure-Rust miniz_oxide
  backend so no system zlib and the wheel matrix stays portable). This is the
  one deliberate move of the I/O seam, justified in the design spec 4.2: the
  cost being removed *is* the read.
status: accepted
fixture: data/fixtures/contact_distance_small.tsv.gz
parity:
  contact_distance.decay:    { class: deterministic-strict, threshold: 0.0, pass: true, metric: <fill> }
  contact_distance.sparsity: { class: deterministic-strict, threshold: 0.0, pass: true, metric: <fill> }
notes: |
  Both outputs are integer counts, exact under any summation order, so both
  gate strict. Bin edges are computed by numpy in Python and passed in, so
  Rust never recomputes exp2 and there is no ULP drift.

  np.histogram's edge rules are replicated exactly: bins are right-open except
  the final bin which is right-closed, and values outside [edges[0],
  edges[-1]] are dropped. For hg38 the top edge is ~231.7 Mb against a 249.0
  Mb chr1, so upstream silently drops the longest cis contacts — replicated,
  not fixed.

---
```

- [ ] **Step 2: Bump the version**

In `pyproject.toml` change `version = "0.4.0"` to `version = "0.5.0"`.
In `rust/Cargo.toml` change `version = "0.4.0"` to `version = "0.5.0"`.

- [ ] **Step 3: Verify the build picks up the new version**

Run:
```bash
cd /large_storage/zhoulab/shengmao/rust-scHiCluster
conda run -n rebuild-rust maturin develop --release 2>&1 | tail -2
conda run -n rebuild-rust python -c "import schicluster_rs, importlib.metadata as m; print(m.version('schicluster-rs'))"
```
Expected: `0.5.0`.

- [ ] **Step 4: Commit**

```bash
git add docs/ITERATION_LOG.md pyproject.toml rust/Cargo.toml rust/Cargo.lock
git commit -m "log: iterations 4-5 (gene-score, contact-distance); bump 0.4.0 -> 0.5.0"
```

---

## Self-review

**1. Spec coverage.** Spec §3 (gene-score) → Tasks 3, 4. §3.3 edge cases → Task 2 Step 5 (fixture proof) and Task 3 Step 1 (unit tests). §3.4 gate classes → Task 1. §3.5 patch → Task 7. §4 (contact-distance) → Tasks 5, 6; §4.2 flate2 seam → Task 5 Step 3; §4.4 semantics → Task 5 Steps 1, 4. §5 manifest amendment → Task 1. §6 fixtures and harness → Tasks 2, 8, 9. §8.7 CLI exposure → Task 7 Step 5. §8.8 version bump → Task 10.

Deliberately deferred to the Phase 6 plan, per the spec's phase split: §7 (acceleration), §8.1–§8.6 (AUDIT, MATH, RECONSTRUCTION_REPORT, notebooks, README/PERFORMANCE/tutorial docs). Note `tutorial/gene_score.md` — referenced by the docstring written in Task 4 — is a Phase 6 deliverable (§8.6); the docstring reference is forward-looking and correct once Phase 6 lands.

**2. Placeholder scan.** The `<fill>` markers in Task 10 are runtime-measured parity metrics, the same convention Phases 1–3 used in `ITERATION_LOG.md`. No TBD/TODO elsewhere; every code step carries complete code.

**3. Type consistency.** `py_gene_score_chrom` takes `(indptr, indices, data, n_rows, n_cols, row_start, row_end, col_start, col_end)` and returns `PyArray1<f64>` — matching its call sites in `tests/test_gene_score_semantics.py` and `_window_sums`. `py_contact_decay_cell` takes `(path, chroms, bin_edges, resolution, chrom1, pos1, chrom2, pos2)` and returns `(Vec<u64>, Vec<(String, u64)>)` — matching both the unit test and the wrapper. `_window_sums` / `_gene_meta_from_pack` / `_chrom_sizes_from_pack` are spelled identically in Tasks 4, 8 and 9. Manifest JSON paths (`$.gene_score.impute`, `$.gene_score.raw`, `$.contact_distance.decay`, `$.contact_distance.sparsity`) match the payload keys built in Tasks 8 and 9.
