# Rust port of scHiCluster gene-score / contact-distance (Phase 5), plus acceleration and close-out (Phase 6)

**Date:** 2026-08-23
**Repo:** `rust-scHiCluster` (extends `schicluster_rs` 0.4.0)
**Protocol:** [rebuildpy](../../../../rebuildpy/README.md) — pre-registered class-aware parity gate.
**Upstream reference:** `/large_storage/zhoulab/shengmao/scHiCluster`, version `1.3.5.dev22+gd566046`.
**Predecessor spec:** [2026-06-02 loop/domain/compartment/embedding design](2026-06-02-rust-port-loop-domain-compartment-embedding-design.md).

## 1. Scope and phase renumbering

The predecessor spec §9 reserved "Phase 5" for the acceleration pass and close-out. This spec renumbers:

| Phase | Content | Status |
|---|---|---|
| 0–4 | scaffold, loop, domain, compartment, embedding | done, gate 17/17 green at `3d4a860` |
| **5** | **`gene-score` + `contact-distance` ports** (this spec §3–§6) | this work |
| **6** | **acceleration pass + rebuildpy close-out** (this spec §7–§8) | this work |

Phase 5 lands before Phase 6 because `RECONSTRUCTION_REPORT.md` and the evolution plot must cover every ported module. References to "Phase 5" inside `docs/MATH.md` and `docs/RECONSTRUCTION_REPORT.md` are updated to "Phase 6" as part of Phase 6.

The constraint from the predecessor spec holds unchanged: **upstream Python at `/large_storage/zhoulab/shengmao/scHiCluster` is never edited.** Integration is by `patch_schicluster()` module-level rebinding.

## 2. Motivation

`gene-score` in `impute` mode is the dominant cost in the user's production workload: 78,691 genes in `ProstateCancer/gene_meta.tsv`, each triggering a `D[a:b, c:d].sum()` scipy sparse slice that materialises a fresh submatrix. Per cell that is ~78.7k allocations. The workload currently runs as a 97-task SLURM array at 8 h wall-clock per task.

`contact-distance` reads every cell's gzipped contact TSV with `pd.read_csv` and builds a DataFrame only to histogram two columns and count distinct bin pairs. There are 112,709 such files (~2.6 MB gzipped each) in `ProstateCancer/rmbkl/`.

## 3. Phase 5 — `gene-score`

### 3.1 What moves to Rust

Upstream `schicluster/draft/gene_score.py` has two per-cell workers and one orchestrator:

| Function | Disposition |
|---|---|
| `gene_score_impute(cell_path, chrom_sizes, gene_meta)` | cooler read stays Python; the per-gene window-sum loop moves to Rust |
| `gene_score_raw(cell_path, chrom_sizes, gene_meta, resolution, ...)` | pandas parse + groupby + `csr_matrix` build stay Python; the per-gene window-sum loop moves to Rust |
| `gene_score(...)` orchestrator | stays Python (`ProcessPoolExecutor`, `to_hdf`) |

Both workers reduce to the same kernel: sum a CSR over a list of rectangular windows.

### 3.2 Rust surface

```rust
// rust/src/gene_score.rs
#[pyfunction]
fn py_gene_score_chrom(
    indptr:  PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data:    PyReadonlyArray1<f64>,
    n_rows:  usize,
    n_cols:  usize,
    row_start: PyReadonlyArray1<i64>,   // per gene, pre-clip, may be negative
    row_end:   PyReadonlyArray1<i64>,
    col_start: PyReadonlyArray1<i64>,
    col_end:   PyReadonlyArray1<i64>,
) -> Vec<f64>;
```

One call per chromosome, all that chrom's genes batched. Wrapped in `py.allow_threads`. Rayon parallelises across genes; each gene's reduction stays serial and in CSR row-major order, so the rewrite is admissibility class **(E)** with respect to a serial Rust baseline.

Per gene: for each row in the clipped row range, binary-search the row's `indices` span for `col_start` and `col_end` and sum that contiguous slice of `data`. scipy guarantees `has_sorted_indices` on a freshly built CSR; the Python wrapper calls `sort_indices()` defensively before handing the buffers over.

Cost per gene goes from "allocate a submatrix of `nnz_window` entries" to `O(rows_in_window · log nnz_per_row + nnz_window)` with no allocation.

### 3.3 Parity semantics that must be replicated, not fixed

Verified against scipy 1.15.2 in `rebuild-rust`:

1. **Negative slice start.** `gene_score_impute` computes `D[(xx-1):(yy+1), xx:(yy+2)]`. When `xx == 0` the start is `-1`, which scipy resolves as `n-1`, yielding an empty `(0, k)` window and a score of `0.0` — not a window anchored at row 0. Measured: on a 10×10 test matrix, `D[-1:4, 0:5].sum()` is `0.0` where `D[0:4, 0:5].sum()` is `340.0`. Every gene whose start bin is 0 scores 0 upstream. The Rust kernel reproduces this: a negative `row_start` is resolved modulo `n_rows` exactly as Python slicing does, then clipped.
2. **Overrun clipping.** `yy+2` may exceed `n_cols`; scipy clips silently. Measured: `D[7:10, 8:11]` on a 10-column matrix yields 2 columns.
3. **Empty windows** (start ≥ end after resolution and clipping) sum to `0.0`.

### 3.4 Reduction-order and gate class

scipy's `.sum()` is `np.add.reduce`, which uses **pairwise** summation with a 128-element block, not left-to-right accumulation. Bit-exact agreement would require reimplementing numpy's pairwise blocking in Rust. That is not worth the complexity for this output:

- **`gene_score.impute`** — f64 sums over windows of at most a few thousand entries. Expected relative drift ~1e-13. Class `deterministic-bounded`, threshold `1.0e-6`.
- **`gene_score.raw`** — the CSR carries `int32` counts (`D[chrom1].astype(np.int32)`), and integer addition is associative and exact, so order is irrelevant. Upstream's `.sum()` returns `np.int64`. Class `deterministic-strict`, threshold `0.0`.

The Rust kernel accumulates in f64 for both modes; raw-mode counts are small integers exactly representable in f64, and the Python wrapper casts to `int64` on emit to match upstream's dtype.

### 3.5 Python layer

New module `python/schicluster_rs/gene_score/__init__.py` exporting `gene_score_impute` and `gene_score_raw` with signatures identical to upstream. Each:

1. Obtains the chrom CSR the same way upstream does (`triu(cool.matrix(...).fetch(chrom), k=1).tocsr()` for impute; the pandas groupby build for raw).
2. Slices `gene_meta` for that chrom into four index arrays.
3. Calls `py_gene_score_chrom` once.
4. Appends results in the same chrom-then-gene order upstream produces, so the returned list aligns positionally with `gene_meta.index`.

`patch_schicluster()` rebinds `schicluster.draft.gene_score.gene_score_impute` and `.gene_score_raw` at module level. The orchestrator resolves those names from module globals at `exe.submit` time, so the existing `ProcessPoolExecutor` picks up the Rust versions with no change to upstream. On Linux the default `fork` start method inherits the patched module; the wrappers are also importable by qualified name for `spawn`.

## 4. Phase 5 — `contact-distance`

### 4.1 What moves to Rust

Upstream `schicluster/cool/contact_distance.py::compute_decay` is one per-cell worker: read the gzipped TSV, keep cis contacts on known chroms, histogram `|pos2 - pos1|` over log-spaced bins, then count distinct off-diagonal bin pairs per chrom. The whole body moves to Rust; the `contact_distance` orchestrator (`ProcessPoolExecutor`, `pd.concat`, `to_hdf`) stays Python.

### 4.2 I/O seam change

This is the one deliberate deviation from the predecessor spec §2, which put file I/O in the "stays Python" column. Rust owns the read here, because the cost being removed *is* the read: `pd.read_csv` building a DataFrame of millions of rows to use four columns. New dependency:

```toml
flate2 = "1.0"
```

`MultiGzDecoder` (not `GzDecoder`) so multi-member gzip streams are handled, wrapped in a `BufReader`, streamed line by line in constant memory. This is the crate's first I/O dependency; Phases 0–4 added none.

### 4.3 Rust surface

```rust
// rust/src/contact_distance.rs
#[pyfunction]
fn py_contact_decay_cell(
    path:       &str,
    chroms:     Vec<String>,          // known chrom names, from chrom_sizes.index
    bin_edges:  PyReadonlyArray1<f64>, // computed by numpy in Python, passed in
    resolution: i64,
    chrom1: usize, pos1: usize, chrom2: usize, pos2: usize,  // 0-based column indices
) -> PyResult<(Vec<u64>, Vec<(String, u64)>)>;   // (decay histogram, per-chrom sparsity)
```

Bin edges are computed by numpy in Python (`2500 * np.exp2(0.125 * np.arange(nbins+1))`, 133 edges for a human genome) and passed in as f64. Rust never recomputes them, so there is no `exp2` ULP drift to reason about.

### 4.4 Semantics to match exactly

Verified against numpy in `rebuild-rust`:

1. **`np.histogram` edge rule.** Bins are right-open except the final bin, which is right-**closed**; values below `edges[0]` or above `edges[-1]` are dropped entirely. Measured on `bins=[0,1,2,3]` with `v=[-0.5, 0.0, 0.999, 1.0, 2.999, 3.0, 3.001]`: counts `[2,1,2]`, 5 of 7 values kept. Rust uses `partition_point` on the monotonic edge array and special-cases equality with the last edge. Note the top edge for hg38 is ~231.7 Mb while chr1 is 249.0 Mb, so the longest-range cis contacts are dropped upstream — replicated, not fixed.
2. **Filter order.** Upstream histograms **raw** positions, and only afterwards floor-divides by `resolution` for the sparsity count. It does not deduplicate before the histogram. Same order in Rust.
3. **Sparsity definition.** `groupby([chrom1, pos1, pos2]).count()` then filter `pos1 != pos2` then `value_counts()` on chrom — i.e. the number of **distinct off-diagonal bin pairs** per chrom. Rust uses one `HashSet<(u32, u32)>` per chrom. Chroms with no surviving pairs are absent from upstream's `value_counts()` output; the wrapper reproduces that by omitting them rather than emitting zeros.
4. **No comment handling.** Unlike `filter-contact`, `compute_decay` calls `read_csv` without `comment='#'`. A `#` line is a hard failure upstream. Rust returns a `PyValueError` naming the file and line rather than silently skipping.

### 4.5 Gate class

Both outputs are integer counts, exact under any summation order: `contact_distance.decay` and `contact_distance.sparsity` are both `deterministic-strict`, threshold `0.0`.

### 4.6 Python layer

New module `python/schicluster_rs/contact_distance/__init__.py` exporting `compute_decay` with upstream's signature, reassembling the Rust tuple into the `[DataFrame, DataFrame]` pair the orchestrator expects — a one-column frame named for the cell in each case. `patch_schicluster()` rebinds `schicluster.cool.contact_distance.compute_decay` at module level.

## 5. Manifest amendment

`data/manifest.yaml` is READ-ONLY with respect to the 17 existing entries: **no existing threshold, class, or name is modified.** Four new entries are *pre-registered* — written and committed before the corresponding kernels exist, per the protocol's pre-registration rule:

| Output | Class | Threshold |
|---|---|---|
| `gene_score.impute` | `deterministic-bounded` | `1.0e-6` |
| `gene_score.raw` | `deterministic-strict` | `0.0` |
| `contact_distance.decay` | `deterministic-strict` | `0.0` |
| `contact_distance.sparsity` | `deterministic-strict` | `0.0` |

Gate goes 17 → 21 outputs.

## 6. Phase 5 fixtures and harness

New fixtures in `data/fixtures/`, generated by extending `data/fixtures/synthesize.py` (seed 42, consistent with Phases 0–4):

- `gene_score_small.npz` — a synthetic CSR per chrom plus a gene table that **deliberately includes a gene starting at bin 0** (to pin the §3.3.1 negative-slice quirk), a gene overrunning the last bin (§3.3.2), and a gene in an empty region (§3.3.3).
- `gene_score_small.cool` — the impute-mode input, so the cooler read path is exercised.
- `contact_distance_small.tsv.gz` — a synthetic contact file with contacts on both sides of the histogram's outer edges, exact-edge hits, trans contacts, unknown-chrom contacts, and duplicate bin pairs.

`tests/py_reference_driver.py` and `tests/_run_candidate.py` each gain one dump block per new output. `tests/test_exact_match.py` picks the new entries up from the manifest with no change.

Fixtures are synthetic and committed-by-generator as in prior phases; the real data on disk (`ProstateCancer/domain_nonepi/*.Q.cool`, `ProstateCancer/rmbkl/*.tsv.gz`, `ProstateCancer/gene_meta.tsv`) is used for **benchmarking only** (§7.2), never as a gate fixture.

## 7. Phase 6 — acceleration pass

### 7.1 Admissibility policy

**(E)-exact rewrites only.** No rewrite that reorders a floating-point reduction is committed, per the playbook's associativity rule. Consequences:

- `MATH.md` records that no (B) rewrite was accepted, with the per-candidate justification. It is not left as a placeholder.
- The predecessor spec §7 listed "banded layout for sliding-window insulation" as (E). That is incorrect for a prefix-sum formulation: `sum(w) = P[b] − P[a]` does not reproduce the float value of summing the window directly. It is logged in `ITERATION_LOG.md` as **rejected for inadmissibility**, not silently skipped.
- `embedding.cell_by_feature` gates at `deterministic-strict 0.0`; it is a gather plus scalar multiply with no reduction, so parallelising it is bit-exact. It stays in scope.

### 7.2 Honest benchmarking

Iteration 0 recorded Rust `convolve2d_mirror` at 0.018 s against scipy's 0.0052 s — a thread-spawn artifact of the 64×64 gate fixture, not a real regression. Phase 6 adds a benchmark harness with realistic input sizes, kept separate from the frozen gate fixtures, drawn from the real data named in §6. Every ITERATION_LOG entry carries both a small-fixture and a realistic-size timing so the evolution plot is not misleading.

### 7.3 Candidate rewrites

Each is attempted and logged as accepted or rejected, with timings and a parity re-run:

| Candidate | Class | Notes |
|---|---|---|
| Hoist mirror-index tables out of `conv.rs`'s inner loop | (E) | index arithmetic only, no float touched |
| Rayon across genes in `gene_score.rs` | (E) | genes independent, in-gene reduction serial |
| `target-cpu=native` via `RUSTFLAGS`, opt-in | (E) | benchmarks only, never the shipped wheel |
| Prefix-sum insulation window | (B) | **rejected for inadmissibility** under §7.1 |

The list is a starting point, not a closed set; further (E) candidates found during the pass are logged the same way.

## 8. Phase 6 — close-out deliverables

Every item is non-skippable and must be produced from an actual run, not written from memory:

1. `docs/AUDIT.md` — generated by `python -m engine.py_function_audit --py-source /large_storage/zhoulab/shengmao/scHiCluster --rust-crate rust/src`, replacing the placeholder.
2. `docs/MATH.md` — per §7.1.
3. `docs/ITERATION_LOG.md` — iterations 4 (gene-score), 5 (contact-distance), 6+ (one per acceleration attempt, accepted and rejected alike).
4. `docs/RECONSTRUCTION_REPORT.md` — all 8 sections filled, with §5's checklist actually executed: `maturin build --release` produces a wheel, that wheel `pip install`s clean into a fresh Python 3.10 env, `pytest -q` green against the release build, notebooks pre-executed, license and version checked.
5. Notebooks, all pre-executed: `examples/compare_Python_vs_Rust.ipynb`, `examples/function_by_function_Python_parity.ipynb`, `examples/evolution.ipynb` (rendering `examples/evolution.png`), `examples/tutorial_loop_domain.ipynb`.
6. Docs updated for the two new subcommands: `README.md` CLI table and example flow, `docs/PERFORMANCE.md` speed/parity tables, and new `tutorial/gene_score.md` and `tutorial/contact_distance.md`.
7. `python/schicluster_rs/__main__.py` — `gene-score` and `contact-distance` added to `_SUPPORTED` and the help text, marked as Rust-backed.
8. Version `0.4.0` → `0.5.0` in `pyproject.toml` and `rust/Cargo.toml`.

## 9. Known limitations, recorded up front

- **`gene-score raw` mode wins far less than `impute` mode.** The pandas parse and groupby that build the matrix stay Python; only the per-gene sum is accelerated. Impute mode is where the 78.7k-slice loop lives.
- **`contact-distance` is bounded by gzip inflate.** `flate2` does not decompress meaningfully faster than zlib. The win is skipping DataFrame construction, so the honest target is ~3–5×, not the ~10× seen on long chromosomes in the impute path.
- **The upstream `xx == 0` empty-window quirk (§3.3.1) is preserved.** Genes starting in bin 0 score 0. This is a faithfulness decision, not an endorsement; it is noted in `tutorial/gene_score.md` so users are not surprised.
- **`gene_score.impute` cannot gate at `deterministic-strict`** because numpy's pairwise summation is not reproduced. See §3.4.
- Contacts longer than the top histogram edge (~231.7 Mb) are dropped by upstream and therefore by the port. See §4.4.1.

## 10. Out of scope

- Renaming or restructuring anything Phases 0–4 already shipped.
- Any edit to upstream `scHiCluster`.
- The remaining `hicluster` subcommands not already exposed by the CLI.
- (B) or (C) class rewrites, per §7.1.
