# Performance, accuracy and architecture

Detail companion to the top-level [README](../README.md). Covers what the
package computes, how fast it is, how close to upstream it stays, the
repository layout and the algorithmic notes for the SpMM hot loop.

## What the impute path computes

`schicluster_rs.impute_chromosome(...)` runs the same end-to-end pipeline
as `schicluster.impute.impute_chromosome.impute_chromosome`:

1. Read raw single-cell contact matrix from a `.cool` file.
2. Drop the diagonal.
3. 2-D Gaussian convolution (mirror padding) — replaces
   `scipy.ndimage.gaussian_filter`.
4. Drop the diagonal again.
5. Row-normalise → P.
6. Random-walk-with-restart fixed point: `Q = (1−rp)·P·Q + rp·P` for up
   to 30 iterations or until `‖Q_t − Q_{t-1}‖_F < tol`.
7. Symmetrise: `E = Q + Qᵀ`.
8. SQRTVC normalise: `E ← D^{-1/2} · E · D^{-1/2}` where `D = diag(Eᵀ𝟙)`.
9. Filter the upper triangle to entries with `j − i ≤ output_dist_bins`.
10. Write the result to an HDF5 file (cooler-compatible).

Steps 2–9 run inside Rust; only the cooler read in step 1 and the HDF5
write in step 10 cross the Python boundary.

## What the loop path computes (Phase 1)

`patch_schicluster()` rebinds the upstream `schicluster.loop` per-chrom
inner functions to Rust kernels: `calculate_chrom_background_normalization`
(diagonal pctl-99 z-score plus donut-minus convolution), the per-chrom
cell-to-group `(Σ, Σ²)` accumulator, the four background convolutions
(`scan_kernel`'s bl / donut / horizontal / vertical), and the graph plus
heap-based `find_summit`. Cooler/HDF5 I/O, the paired t-test and BH-FDR
stay in Python.

## What the gene-score path computes (Phase 5)

`patch_schicluster()` rebinds `schicluster.draft.gene_score`'s two per-cell
workers. For each gene, upstream sums the contact matrix over a rectangular
window around its bins — `D[r0:r1, c0:c1].sum()` — allocating a fresh scipy
submatrix every time. With a human gene set that is ~78,700 allocations per
cell. The Rust kernel binary-searches each row's sorted column indices for the
column range and reduces in place, parallel across genes. Cooler reads (impute
mode) and the pandas `groupby` matrix build (raw mode) stay in Python.

## What the contact-distance path computes (Phase 5)

`compute_decay` streams a cell's gzipped contact TSV line by line in Rust
(`flate2::MultiGzDecoder` + `BufReader`), filters to cis contacts on known
chromosomes, histograms `|pos2 - pos1|` over caller-supplied log-spaced edges,
and counts distinct off-diagonal bin pairs per chromosome in a `HashSet`. No
DataFrame is ever built. This is the one place in the port where Rust owns file
I/O — justified because the cost being removed *is* the read.

## Speed

Real Chang 2024 LC462 mouse cortex Droplet Hi-C, 25 kb resolution
(impute path):

| step                         | scipy upstream | rust      | speedup |
|------------------------------|----------------|-----------|---------|
| chr1   (n = 7820 bins)       | 30.5 s         | **3.2 s** | **9.6×** |
| chr19  (n = 2461 bins)       |  0.4 s         | **0.27 s**| 1.5× |
| 20 chrs end-to-end per cell  | 87 s           | **33 s**  | **2.7×** |

Multi-process parallelism (8 workers × 2 rayon threads = 16 cores total):
8 chr1 in parallel from 29 s → 9.4 s — an additional **3.1×** beyond the
per-cell speedup, by avoiding rayon thread oversubscription.

### Phase 5 / Phase 6 benchmark workloads

Measured by `examples/bench_phase6.py` under the rebuildpy protocol: BLAS and
rayon pinned to 4 threads, one warmup run discarded, then 3 timed runs.

| workload | Python | Rust | speedup |
|---|---|---|---|
| `conv2d_mirror` 1024², 11×11 donut kernel | 0.0692 s | **0.0306 s** | **2.3×** |
| `gene_score`, 20 000 windows over a 4000² f32 CSR | 1.0636 s | **0.0049 s** | **217×** |
| `contact_distance`, one real production cell | 0.2656 s | **0.0610 s** | **4.4×** |

Two caveats worth stating plainly:

* **The conv figure is post-acceleration.** Before Phase 6 it was **0.9×** —
  genuinely *slower* than `scipy.ndimage.convolve`, because the inner loop
  recomputed `mirror_index` (an integer modulo plus two branches) once per
  `(i, j, p, q)`: ~127M times for that input. Hoisting the index tables fixed
  it; see [ACCELERATION_LOG.md](ACCELERATION_LOG.md) iter 1.
* **The gene-score figure is warmup-excluded.** A single one-shot call that
  also pays rayon's thread-pool spin-up is closer to **46×**. Both numbers are
  real and answer different questions; since gene-score processes thousands of
  cells per job, the steady-state figure is the one users experience.

`target-cpu=native` was measured and **rejected**: 0.0943 s against the portable
build's 0.0946 s, indistinguishable from noise. The shipped wheel stays
portable, and there is no opt-in knob to recommend.

## Accuracy

Bit-equivalent to upstream within float-32 ε. On real Chang chr1
(n = 7820, ~1.7 M output non-zeros):

* `max |E_rust − E_scipy|` = `8.94 × 10⁻⁸`
* Pearson correlation = `1.000000`
* nnz match exactly.

### Parity gate

The rebuildpy parity manifest (`data/manifest.yaml`) pre-registers a
threshold per ported function and is read-only once the agent loop
starts. As of 0.5.0 **all 21 outputs pass**:

| Output | Class | Threshold | Status |
|---|---|---|---|
| `conv.convolved` | deterministic-bounded | `1e-6` | green (1.79e-7) |
| `loop_bkg.E` / `loop_bkg.T` | deterministic-bounded | `1e-6` | green (≤ 9.5e-7) |
| `merge.e_sum` / `merge.e2_sum` | deterministic-bounded | `1e-6` | green (≤ 1.2e-7) |
| `scan_kernels.bl/donut/h/v` | deterministic-bounded | `1e-6` | green (≤ 2.4e-7) |
| `find_summit.idx` | ranked (Jaccard) | `0.99` | green (1.0) |
| `find_summit.sizes` | classification | `1.0` | green (1.0) |
| `insulation.score` | deterministic-bounded | `1e-6` | green (5.96e-08) |
| `topdom.bed.interval_jaccard` | ranked | `0.95` | green (1.0) |
| `topdom.bed.bin_label_agreement` | classification | `0.98` | green (1.0) |
| `compartment.comp` | deterministic-bounded | `1e-6` | green (2.08e-17) |
| `compartment.strength` | deterministic-bounded | `1e-6` | green (1.71e-13) |
| `embedding.cell_by_feature` | deterministic-strict | `0` (f32 exact) | green (0.0) |
| `gene_score.impute` | deterministic-bounded | `1e-6` | green (**0.0**, bit-exact) |
| `gene_score.raw` | deterministic-strict | `0` | green (0.0) |
| `contact_distance.decay` | deterministic-strict | `0` | green (0.0) |
| `contact_distance.sparsity` | deterministic-strict | `0` | green (0.0) |

Worst deterministic error across the whole gate: **9.54e-07** (`loop_bkg.T`),
against a 1e-6 threshold.

`tests/test_parity.py` (legacy) additionally runs `random_walk_cpu` over
`(n, rp) ∈ {50, 200, 500} × {0.05, 0.5, 0.9}` and asserts
max-relative-error < `1e-4` against scipy's reference implementation.

## Layout

```
rust-scHiCluster/
├── README.md                 install + tutorial
├── LICENSE                   MIT
├── docs/                     this file + protocol artefacts
│   ├── PERFORMANCE.md        (you are here)
│   ├── DISCOVERY.md          rebuildpy Phase 0.5
│   ├── MATH.md               (B)-rewrite perturbation bounds
│   ├── ITERATION_LOG.md      per-phase port history
│   ├── ACCELERATION_LOG.md   Phase 6 acceleration search
│   ├── AUDIT.md              Python-API coverage
│   ├── RECONSTRUCTION_REPORT.md
│   └── superpowers/          per-phase specs + plans
├── pyproject.toml            maturin build config
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs            module registrations + impute path
│       ├── utils.rs          sparse / banded / dense helpers
│       ├── conv.rs           shared 2-D convolution (mirror mode)
│       ├── loop_bkg.rs       Phase 1 — per-cell loop background
│       ├── merge.rs          Phase 1 — cell-to-group sparse sum
│       ├── scan_kernels.rs   Phase 1 — 4 background kernel scans
│       ├── find_summit.rs    Phase 1 — graph + max-heap peak merge
│       ├── insulation.rs     Phase 2 — sliding-window insulation
│       ├── topdom.rs         Phase 2 — native TopDom (drops rpy2/R)
│       ├── compartment.rs    Phase 3 — CpG-weighted compartment
│       ├── embedding.rs      Phase 4 — cell-by-feature extraction
│       ├── gene_score.rs     Phase 5 — per-gene CSR window sums
│       └── contact_distance.rs  Phase 5 — streaming gzip reader
├── python/schicluster_rs/
│   ├── __init__.py           public API + patch_schicluster()
│   ├── loop/
│   │   ├── __init__.py       loop wrappers (Phase 1)
│   │   └── snakemake_template_loop.txt
│   ├── domain/               insulation + TopDom wrappers (Phase 2)
│   ├── compartment/          compartment wrapper (Phase 3)
│   ├── embedding/            cell-by-feature wrapper (Phase 4)
│   ├── gene_score/           gene-score wrappers (Phase 5)
│   └── contact_distance/     compute_decay wrapper (Phase 5)
├── tests/
│   ├── test_parity.py        legacy random_walk_cpu unit parity
│   ├── test_exact_match.py   rebuildpy manifest gate
│   ├── parity_harness.py     manifest → engine.parity_metrics dispatch
│   ├── py_reference_driver.py (runs upstream Python in schicluster env)
│   ├── _run_candidate.py     (runs Rust in rebuild-rust env)
│   └── run_parity.sh         two-env orchestrator
└── data/
    ├── manifest.yaml         pre-registered parity gate (read-only)
    └── fixtures/             synthetic test fixtures (gitignored)
```

## Algorithm notes (impute path)

The hot loop is the iterative random-walk-with-restart, implemented as a
**Sparse-times-Dense matrix multiplication (SpMM)** with rayon row-wise
parallelism:

* `P` (sparse, ~7 nnz per row after Gaussian smoothing) stays as CSR.
* `Q` (the iterate) is stored dense, since RWR diffuses it to ≥ 30 %
  density after 1–2 iterations anyway.
* Each iteration: `Q' = (1−rp) · (P · Q) + rp · P`. The `P · Q` matmul is
  computed row-wise; each output row is independent, so rayon splits
  row-chunks across cores. Within each row, the inner AXPY (accumulate
  `P[i,k] · Q[k, :]` for sparse k) vectorises cleanly.

Other steps (Gaussian convolution, SQRTVC normalize, triangle filter) are
similarly multi-threaded over rows or chunks.

For users who can tolerate ≪1 % deviation from the strict scipy result, a
`band_factor` parameter is available that runs the RWR with a banded `Q`
(only entries with `|j − i| ≤ band_factor × output_dist_bins`), giving an
additional ~4× speedup. Default is 0 (off, strict).

## Algorithm notes (loop path)

* All five convolutions in the loop pipeline (one donut-minus in
  `loop_bkg`, four in `scan_kernels`) share the same `convolve2d_mirror`
  primitive — kernel-flip semantics (i.e. scipy's `convolve`, not
  `correlate`) plus mirror reflect-without-edge-repeat boundary, parallel
  across output rows with fixed-order per-row reduction so the rewrite is
  (E)-exact vs a serial outer loop.
* `merge_cells_sum` accumulates in `BTreeMap<(u32, u32), f64>` so the
  emit order is row-major deterministic and the cross-cell sum dodges the
  f32 reduction-order drift `rayon` would introduce.
* `find_summit` uses Rust's `BinaryHeap` keyed on `(−E, idx)` with
  ascending-idx tie-break — that matches Python `heapq` stability when
  ties on E are broken by insertion order.

## Algorithm notes (gene-score path)

The subtle part of this kernel is not the search — it is the **reduction
order**, and getting it wrong fails the parity gate while being *more*
numerically accurate.

`scipy.sparse.csr_matrix.sum(axis=None)` is not a flat `data.sum()`. scipy
computes it as `(self @ np.ones(n_cols, dtype=res_dtype)).sum()`:

1. a CSR matvec against a ones vector, accumulating **each row serially in
   stored column order**, then
2. `np.add.reduce` over the dense row-sums vector, which uses **pairwise**
   summation with an 8-way unrolled base case and a 128-element blocksize,

with `res_dtype` being the matrix's own dtype for floats. Imputed cools store
`count` as f32, so the whole reduction happens in f32 and upstream's own answer
carries about `3.8e-6` of rounding on a window summing to ~48.

Three candidate reductions were measured against it:

| Candidate | Max abs error vs upstream | 1e-6 gate |
|---|---|---|
| f64 accumulate (the obvious port) | `3.35e-6` | fails |
| flat f32 pairwise over window values | `3.81e-6` | fails |
| f32 matvec-then-pairwise (scipy's own) | `0.0` | passes, bit-exact |

Because `data/manifest.yaml` is read-only and the protocol forbids widening a
threshold to make a port pass, the kernel reproduces scipy's two-stage order.
Verified on every fixture gene and 2000 randomised f32 windows in
`tests/test_gene_score_semantics.py`.

The kernel also reproduces CPython's slice-bound semantics rather than
"correcting" them: `gene_score_impute`'s window is
`D[(xx-1):(yy+1), xx:(yy+2)]`, and when `xx == 0` the start `-1` resolves to
`n-1`, giving an empty window and a score of `0.0`. See
[../tutorial/gene_score.md](../tutorial/gene_score.md).

## Algorithm notes (contact-distance path)

`np.histogram`'s edge rules are reproduced exactly: bins are right-open except
the final bin, which is right-**closed**, and values outside
`[edges[0], edges[-1]]` are dropped. For hg38 the top edge is ~231.7 Mb against
a 249.0 Mb chr1, so the longest cis contacts are silently discarded upstream —
and here too, faithfully. Bin edges are computed by numpy in Python and passed
through, so Rust never recomputes `exp2` and there is no ULP drift to reason
about. Sparsity counts **distinct ordered off-diagonal bin pairs** per
chromosome, matching upstream's `groupby(...).count()` -> filter ->
`value_counts()`.

Reduction-order discipline is tracked in [MATH.md](MATH.md), the per-phase port
history in [ITERATION_LOG.md](ITERATION_LOG.md), and the acceleration search in
[ACCELERATION_LOG.md](ACCELERATION_LOG.md).
