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

## Accuracy

Bit-equivalent to upstream within float-32 ε. On real Chang chr1
(n = 7820, ~1.7 M output non-zeros):

* `max |E_rust − E_scipy|` = `8.94 × 10⁻⁸`
* Pearson correlation = `1.000000`
* nnz match exactly.

### Parity gate

The rebuildpy parity manifest (`data/manifest.yaml`) pre-registers a
threshold per ported function and is read-only once the agent loop
starts. As of 0.2.0, 11 of 17 outputs pass; the remaining 6 are
deferred to phases 2–4:

| Output | Class | Threshold | Status |
|---|---|---|---|
| `conv.convolved` | deterministic-bounded | `1e-6` | green (1.79e-7) |
| `loop_bkg.E` / `loop_bkg.T` | deterministic-bounded | `1e-6` | green (≤ 9.5e-7) |
| `merge.e_sum` / `merge.e2_sum` | deterministic-bounded | `1e-6` | green (≤ 1.2e-7) |
| `scan_kernels.bl/donut/h/v` | deterministic-bounded | `1e-6` | green (≤ 2.4e-7) |
| `find_summit.idx` | ranked (Jaccard) | `0.99` | green (1.0) |
| `find_summit.sizes` | classification | `1.0` | green (1.0) |
| `insulation.score` | deterministic-bounded | `1e-6` | Phase 2 |
| `topdom.bed.*` | ranked + classification | `0.95` / `0.98` | Phase 2 |
| `compartment.comp` / `.strength` | deterministic-bounded | `1e-6` | Phase 3 |
| `embedding.cell_by_feature` | deterministic-strict | `0` (f32 exact) | Phase 4 |

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
│   ├── ITERATION_LOG.md      acceleration attempts
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
│       └── find_summit.rs    Phase 1 — graph + max-heap peak merge
├── python/schicluster_rs/
│   ├── __init__.py           public API + patch_schicluster()
│   └── loop/
│       ├── __init__.py       loop wrappers (Phase 1)
│       └── snakemake_template_loop.txt
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

Reduction-order discipline is tracked in [MATH.md](MATH.md) and per-attempt
in [ITERATION_LOG.md](ITERATION_LOG.md).
