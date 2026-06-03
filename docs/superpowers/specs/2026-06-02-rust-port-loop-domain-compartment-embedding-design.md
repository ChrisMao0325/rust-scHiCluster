# Rust port of scHiCluster loop / domain / compartment / embedding / merge

**Date:** 2026-06-02
**Repo:** `rust-scHiCluster` (existing — extends `schicluster_rs`)
**Protocol:** [rebuildpy](../../../../../rebuildpy/README.md) — 6-step, two-agent loop, pre-registered class-aware parity gate.
**Upstream reference:** `/large_storage/zhoulab/shengmao/scHiCluster` (editable in `schicluster` env, version `1.3.5.dev22+gd566046`).

## 1. Identity & scope

Extend the existing `schicluster_rs` Rust kernel (currently: `random_walk_cpu`, `impute_chromosome_inner`) with native ports of every remaining numerical hot path in scHiCluster, dropping the rpy2/R dependency by reimplementing TopDom natively:

| Module | Functions ported to Rust (per-chrom/cell whole-function) | Stays in Python |
|---|---|---|
| `loop` | `loop_bkg_chrom`, `merge_cells_sum`, `scan_kernels_chrom`, `find_summit_chrom` | cooler/HDF5 I/O, `paired_t_test`, `multipletests` FDR, `ProcessPoolExecutor` |
| `domain` | `insulation_score_chrom`, `topdom_chrom` (full TopDom: diamond signal, gap regions, change-point/local-extreme, Wilcoxon ranksum p-values, bin→domain conversion) | boundary aggregation, AnnData/xarray writes |
| `compartment` | `compartment_chrom` (incl. strength) | `bedtools nuc`, AnnData/xarray writes |
| `embedding` | `make_idx`, cell×feature upper-tri assembly | **SVD stays sklearn** (`TruncatedSVD/arpack`); not a bottleneck and SVD has no elementwise parity |
| `merge_cell_to_group` | `merge_cells_sum` (sparse accumulation across cells, folded into `loop`) | cooler creation, multi-process orchestration |

**Constraint from user:** *keep all upstream Python code unchanged*. Integration is by `patch_schicluster()` monkey-patch rebinding the upstream per-chrom/per-cell functions to Rust-backed wrappers — identical to the existing `impute_chromosome` integration. No edits to `/large_storage/zhoulab/shengmao/scHiCluster/`.

## 2. Phase 0.5 — Discovery summary

- **Target already ported?** Partial. `rust-scHiCluster` currently ports only `impute_chromosome` (RWR + Gaussian + SQRTVC). Loop/domain/compartment/embedding/merge are **not yet ported**. This work extends the existing repo rather than starting a new `rs-*`.
- **Naming deviation:** the existing PyPI name is `schicluster-rs` / import `schicluster_rs`, not the rebuildpy convention `rs-schicluster` / `rs_schicluster`. Keep the existing name for backwards compatibility; record the alias in `DISCOVERY.md` and append to `engine/discover_rust_deps.py::ALIAS_MAP` in a downstream PR.
- **Dependency mapping for the ported functions:**

| Upstream Python dep | Rust replacement |
|---|---|
| `numpy` | `ndarray 0.16` (already in `Cargo.toml`) |
| `scipy.sparse` (CSR/COO ops) | `sprs 0.11` (already in `Cargo.toml`) |
| `scipy.ndimage.convolve` (mirror) | new internal `conv2d_mirror` (extend existing `gaussian_filter_2d`) |
| `scipy.stats.zscore` / `np.percentile` | derived inline (linear-interpolation percentile, ddof=0 zscore) |
| `scipy.stats` Wilcoxon ranksum, normal approx | derived inline — match R `wilcox.test(exact=F, alternative="less")` with continuity + tie correction |
| `cooler` / `h5py` (I/O) | **stays Python** — at the I/O seam |
| `statsmodels.stats.multitest` (FDR) | **stays Python** — out of scope per parity safety |
| `sklearn.decomposition.TruncatedSVD` | **stays Python** — out of scope |
| `rpy2` + R `TopDom.R` | **fully ported to Rust** (`topdom.rs`) |
| `pandas`, `anndata`, `xarray` (DataFrame/h5ad/nc writes) | **stays Python** — at the I/O seam |

## 3. Environments (rebuildpy step 2)

| Role | Env path | Python | Purpose |
|---|---|---|---|
| `$PYTHON_REF_ENV` | `/home/shengmao/miniconda/envs/schicluster` | 3.6 | Runs `tests/py_reference_driver.py`; has upstream `schicluster` editable. **Also has R + rpy2 + the `Matrix` R package** (R_HOME set by `conda activate`), so it runs the upstream TopDom-via-rpy2 path directly — confirmed: `hicluster domain` already works here. **TopDom reference comes from this env**, no separate R env needed. |
| `$RUST_TEST_ENV` | `/home/shengmao/miniconda/envs/rebuild-rust` | 3.10.20 | Runs `tests/_run_candidate.py`; needs `maturin`, `cooler`, `scipy`, `numpy`, `pandas`, `pytest`, the built `schicluster_rs` extension |

`rebuild-rust` is currently empty of project deps — provisioning is part of phase implementation, not part of the spec. The Rust toolchain (`cargo 1.95.0`, `rustc 1.95.0`) is already on `PATH` via rustup at `/home/shengmao/.cargo/bin/`.

**Cross-env parity flow** (single-process "import both" is impossible because `schicluster_rs` is `abi3-py39` and the schicluster env is py 3.6; so it's dump-and-compare across two envs):

```
                  ┌──────────────────────────────────────────────┐
                  │ data/fixtures/<chrom>.npz, <group>.cool, ... │
                  └────────────┬─────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
      $PYTHON_REF_ENV                       $RUST_TEST_ENV
      schicluster py-3.6                    rebuild-rust py-3.10
      (incl. rpy2 + R + TopDom)             (maturin + schicluster_rs)
      py_reference_driver.py                _run_candidate.py
            │                                     │
            ▼                                     ▼
       ref/<fn>_py.json                     cand/<fn>_rs.json
       (incl. ref/topdom.bed                (incl. cand/topdom.bed
        from rpy2 → TopDom.R)                from Rust topdom.rs)
            └─────────────────┬───────────────────┘
                              ▼
               tests/test_exact_match.py (in $RUST_TEST_ENV)
               → engine.parity_metrics.compute_parity / is_pass
               → assert per-output gates from data/manifest.yaml
```

## 4. Repo layout additions

The existing `rust-scHiCluster` repo gets the rebuildpy artefacts added; no existing files moved.

```
rust-scHiCluster/
├── rust/src/
│   ├── lib.rs                       (existing — extended with new module mounts)
│   ├── conv.rs                      (NEW — general 2-D convolve, mirror)
│   ├── loop_call.rs                 (NEW — loop_bkg / merge / scan / find_summit)
│   ├── domain.rs                    (NEW — insulation score)
│   ├── topdom.rs                    (NEW — full TopDom port)
│   ├── compartment.rs               (NEW)
│   ├── embedding.rs                 (NEW — make_idx + feature assembly)
│   └── utils.rs                     (NEW — shared sparse / banded helpers split out of lib.rs)
├── python/schicluster_rs/
│   ├── __init__.py                  (existing — extended: new wrappers + extended patch_schicluster())
│   ├── loop.py                      (NEW)
│   ├── domain.py                    (NEW)
│   ├── compartment.py               (NEW)
│   └── embedding.py                 (NEW)
├── tests/
│   ├── test_parity.py               (existing — kept; legacy unit parity for random_walk_cpu)
│   ├── test_exact_match.py          (NEW — rebuildpy gate against manifest.yaml)
│   ├── test_smoke.py                (NEW — import + run end-to-end on fixture)
│   ├── py_reference_driver.py       (NEW — runs upstream in $PYTHON_REF_ENV, incl. TopDom via rpy2)
│   ├── _run_candidate.py            (NEW — runs schicluster_rs in $RUST_TEST_ENV)
│   └── run_parity.sh                (NEW — orchestrates the 2 envs via `conda run -n …`)
├── data/
│   ├── manifest.yaml                (NEW — pre-registered gate, READ-ONLY after Phase 3 starts)
│   └── fixtures/                    (NEW — small synthetic + a real small chrom)
├── examples/                        (existing — extended)
│   ├── tutorial_quickstart.ipynb    (existing)
│   ├── benchmark_vs_scipy.ipynb     (existing)
│   ├── compare_Python_vs_Rust.ipynb (NEW — pipeline-level parity per output)
│   ├── function_by_function_Python_parity.ipynb (NEW)
│   ├── evolution.ipynb              (NEW — per-iteration narrative + subplots)
│   ├── py_per_function_dump.py      (NEW)
│   └── evolution.png                (NEW — auto-rendered)
├── docs/superpowers/specs/
│   └── 2026-06-02-...-design.md     (this document)
├── DISCOVERY.md                     (NEW — Phase 0.5 artefact, see §2 above)
├── MATH.md                          (NEW — (B)-rewrite perturbation bounds)
├── AUDIT.md                         (NEW — Python API coverage; auto-gen via engine.py_function_audit)
├── ITERATION_LOG.md                 (NEW — one YAML block per acceleration attempt)
└── RECONSTRUCTION_REPORT.md         (NEW — 8 sections, filled at Phase 4)
```

## 5. Per-output parity manifest (pre-registration)

The single `data/manifest.yaml` lists one `outputs:` block per ported function. Defaults derived from `PARITY_TAXONOMY.md`. The threshold column below is the proposed pre-registration; it is **read-only** once Phase 3 starts.

| Output name (key in JSON dumps) | scHiCluster source | Class | Threshold | Rationale |
|---|---|---|---|---|
| `loop_bkg.E` | `loop/loop_bkg.py::calculate_chrom_background_normalization` | `deterministic-bounded` | `atol=1e-6` | Per-diagonal pctl-99/zscore + donut convolve. Convolve may use rayon → (B). `MATH.md` bound. |
| `loop_bkg.T` | same | `deterministic-bounded` | `atol=1e-6` | Same. |
| `merge.e_sum` / `merge.e2_sum` | `loop/merge_cell_to_group.py::merge_cells_for_single_chromosome` | `deterministic-bounded` | `atol=1e-6` | Sparse accumulation; may parallelise → (B). |
| `scan_kernels.bl` / `.donut` / `.h` / `.v` | `loop/loop_calling.py::scan_kernel` | `deterministic-bounded` | `atol=1e-6` | 4 convolutions, masked. |
| `find_summit.idx` / `.sizes` | `loop/loop_calling.py::find_summit` | `ranked` (set-of-selected-indices Jaccard) + `deterministic` on the matched `sizes` slice | Jaccard ≥ 0.99 on `idx`; `atol=0` on `sizes` over the intersection | Deterministic integer-index output; Jaccard absorbs tie-break ambiguity if heap order shifts for equal-`E` pixels. |
| `insulation.score` | `domain/call_domain.py::single_chrom_calculate_insulation_score` | `deterministic-bounded` | `atol=1e-6` | Sliding-window sums; parallel → (B). |
| `topdom.bed` (chrom, start, end, tag) | `domain/TopDom.R` (via rpy2) | `ranked` + `classification` (composite — both must pass) | `interval_jaccard ≥ 0.95` AND `bin_label_agreement ≥ 0.98` | Wilcoxon p-value ties + change-point tie-break drift; element-wise impossible. |
| `compartment.comp` | `compartment/call_compartment.py::single_chrom_compartment` | `deterministic-bounded` | `atol=1e-6` | Sparse row-norm + weighted avg. |
| `compartment.strength` (AA, BB, AB) | `compartment/call_compartment.py::compartment_strength` | `deterministic-bounded` | `atol=1e-6` | Diagonal decay normalise + masked sums. |
| `embedding.cell_by_feature` | `embedding/calc_embedding.py::make_chrom_matrix` (excl. SVD) | `deterministic-strict` (f32) | `atol=0` (exact f32 bit-equality) | Upstream output is float32; the op is upper-tri fancy-index + one scalar multiply — no reduction, so bit-exact is the right gate. A sub-`f32` ε tolerance would be meaningless. |

For `topdom.bed`, the composite parity uses two metrics defined in `engine/parity_metrics.py`: interval Jaccard on `(start, end)` tuples ignoring tag, plus per-bin tag label agreement after coordinate-aligning both BEDs onto the chrom's bin grid. Both must pass; ALL outputs must pass per `manifest.yaml`.

**SVD, FDR, t-test are out of scope** and have no manifest entry — they remain Python.

## 6. Rust surface (PyO3 contracts)

One coarse PyO3 entrypoint per leaf computation, all returning plain numpy arrays / triplet tuples; Python wrappers re-assemble dataframes / writes.

```rust
// conv.rs
pub fn convolve2d_mirror(a: &[f32], nrows: usize, ncols: usize,
                         kernel: &[f32], kh: usize, kw: usize) -> Vec<f32>;
// scipy.ndimage.convolve semantics: kernel-flipped, 'mirror' reflect-no-repeat.

// loop_call.rs — PyO3 surface
#[pyfunction] fn py_loop_bkg_chrom(rows, cols, vals, n, resolution,
    dist, cap, pad, gap, min_cutoff, log_e) -> ((Er,Ec,Ev), (Tr,Tc,Tv));
#[pyfunction] fn py_merge_cells_sum(list_of_(rows,cols,vals), n) -> ((Sr,Sc,Sv), (S2r,S2c,S2v));
#[pyfunction] fn py_scan_kernels_chrom(E_rows, E_cols, E_vals, n,
    pad, gap, loop_xs, loop_ys) -> (bl, donut, h, v);          // four 1-D f32 arrays
#[pyfunction] fn py_find_summit_chrom(x1, y1, E_values, dist_thres_bins)
    -> (selected_idx[u32], sizes[u32]);

// domain.rs
#[pyfunction] fn py_insulation_score_chrom(rows, cols, vals, n, window_size, save_count)
    -> Array1<f32> | Array2<f32>;

// topdom.rs
#[pyfunction] fn py_topdom_chrom(dense: PyReadonlyArray2<f64>, window_size: usize)
    -> Vec<(from_id: u32, to_id: u32, tag: u8)>;                // tag: 0=gap, 1=domain, 2=boundary

// compartment.rs
#[pyfunction] fn py_compartment_chrom(rows, cols, vals, n, cpg_ratio, calc_strength)
    -> (comp[f64; n], scores[f64; 3] | None);

// embedding.rs
#[pyfunction] fn py_make_chrom_features(per_cell_dense_views, n_bins,
    dist_bins, scale_factor) -> Array2<f32>;                    // cells × idx.size
```

All take `py.allow_threads(|| ...)` around the compute. f64 used wherever the upstream uses f64 (compartment, embedding, topdom); f32 where upstream uses f32 (loop_bkg, scan, find_summit, insulation — match upstream dtype).

## 7. Acceleration plan (Phase 3b)

Per-rewrite admissibility tracked in `ITERATION_LOG.md`; bounds in `MATH.md`.

| Rewrite class | Where it applies | Proof |
|---|---|---|
| Serial fixed-order loop (baseline iter 0) | every leaf | (E) exact |
| `par_chunks_mut` over output rows for convolution, insulation window, find_summit graph build, compartment matvec, scan_kernels | conv.rs, loop_call.rs, domain.rs, compartment.rs | (E) if the per-row work is independent and reduces serially within the row; (B) `n·eps·max\|x\|` if any cross-row sum is reordered |
| Zero-copy `ArrayView` on the numpy buffer | every PyO3 entry | (E) |
| Banded layout for sliding-window insulation | domain.rs | (E) — math identity |
| `LTO + codegen-units=1` (already on) | crate-wide | (E) |
| `target-cpu=native` (opt-in via env) | benchmarks only | (E) |
| TopDom: in-place diagonal scaling, vectorised diamond mean | topdom.rs | (E) where reductions stay ordered |

Anything that reorders a `sum` across threads — most rayon reductions — gets the `n·eps·max|x|` bound (PARITY_TAXONOMY §1 reduction-order rule) and the affected output stays `deterministic-bounded ≤ 1e-6`. No `deterministic-strict` for parallelised paths.

## 8. Verification harness

- `tests/py_reference_driver.py` (run in `$PYTHON_REF_ENV`): for each fixture, calls every upstream `schicluster` reference path directly — loop_bkg, merge, scan, find_summit, **insulation_score**, **TopDom via the existing rpy2 → `TopDom.R` wrapper** (`schicluster.domain.call_domain.run_top_dom` style), compartment, embedding-matrix. Dumps a per-output JSON with keys matching the manifest. The schicluster env has working R + rpy2 + the `Matrix` R package once activated, so TopDom is the canonical reference here — no separate R env needed.
- `tests/_run_candidate.py` (run in `$RUST_TEST_ENV`): imports `schicluster_rs`, runs the Rust wrappers on the same fixtures, dumps per-output JSON with the same keys.
- `tests/test_exact_match.py` (run in `$RUST_TEST_ENV`): loads both dumps, applies `engine/parity_metrics.compute_parity` per manifest output, asserts `is_pass`.
- `tests/run_parity.sh`: orchestrates two `conda run` invocations — `conda run -n schicluster python tests/py_reference_driver.py …` then `conda run -n rebuild-rust pytest -q tests/test_exact_match.py`.

**Fixtures** (`data/fixtures/`):
- `synthetic_<n>.npz` (n ∈ {64, 256, 1024}): seeded synthetic upper-tri sparse matrices for fast unit parity per leaf.
- One small real chrom from `scHiCluster/example/` or `scHiCluster/files/` if present (TBD during phase 1 — falls back to synthetic if no usable real data is shipped).
- For loop calling we additionally synthesise a tiny per-cell `.E.npz`/`.T.npz` set + the cell→group cools, since these are pipeline-mid artefacts not normally shipped.

## 9. Phasing

Each phase: provision new code → equivalence pass clears gates → acceleration pass → log → commit. Single shared manifest is written upfront covering all phases.

| Phase | Modules | Outputs added to manifest | Notes |
|---|---|---|---|
| 0 | rebuildpy artefact scaffold + `data/manifest.yaml` + `conv.rs` shared 2-D convolution | none yet | unblocks loop + scan |
| 1 | `loop_call.rs` + `python/schicluster_rs/loop.py` + monkey-patch hooks for `loop_bkg`, `merge_cells_for_single_chromosome`, `scan_kernel`, `find_summit` | `loop_bkg.E`, `loop_bkg.T`, `merge.e_sum`, `merge.e2_sum`, `scan_kernels.{bl,donut,h,v}`, `find_summit.{idx,sizes}` | |
| 2 | `domain.rs` + `topdom.rs` + `python/schicluster_rs/domain.py` + monkey-patch `single_chrom_calculate_insulation_score`, `run_top_dom` (replace rpy2 call site) | `insulation.score`, `topdom.bed` | biggest equivalence risk; Wilcoxon parity is the long pole |
| 3 | `compartment.rs` + `python/schicluster_rs/compartment.py` + monkey-patch `single_chrom_compartment`, `compartment_strength` | `compartment.comp`, `compartment.strength` | |
| 4 | `embedding.rs` + `python/schicluster_rs/embedding.py` + monkey-patch `make_chrom_matrix` (SVD untouched) | `embedding.cell_by_feature` | small win; for completeness |
| 5 | Acceleration pass across all leaves; ITERATION_LOG; evolution plot; notebooks; RECONSTRUCTION_REPORT | — | version bump 0.1.1 → 0.2.0; rebuild + reinstall in `rebuild-rust` |

## 10. Deliverables (Phase 4, non-skippable)

- `data/manifest.yaml` clearing all per-output gates.
- `DISCOVERY.md` (Phase 0.5 outcome — already drafted in §2).
- `MATH.md` with derived `n·eps·max|x|` bounds for every (B) rewrite.
- `AUDIT.md` auto-generated by `engine.py_function_audit --py-source scHiCluster --rust-crate rust/src`.
- `ITERATION_LOG.md` one YAML block per acceleration attempt (accepted + rejected).
- `RECONSTRUCTION_REPORT.md` (8 sections) including the two-panel `examples/evolution.png`.
- `examples/compare_Python_vs_Rust.ipynb`, `examples/tutorial_loop_domain.ipynb`, `examples/function_by_function_Python_parity.ipynb`, `examples/evolution.ipynb` — all pre-executed.
- `tests/test_exact_match.py` green under `pytest -q` in `$RUST_TEST_ENV` on a `--release` build.
- `maturin build --release` produces a wheel that `pip install`s cleanly in a fresh `python=3.10` env.

## 11. Known risks / limitations

- **TopDom parity ceiling.** R `wilcox.test(exact=F, alternative="less")` uses a normal approximation with continuity correction and a specific tie-correction in the variance; reproducing it bit-for-bit in Rust is feasible but historically a source of `1e-12`-level p-value drift, which can flip the `pvalue < 0.05` filter on borderline bins. The composite `ranked + classification` gate for `topdom.bed` is sized to absorb that without widening below `interval_jaccard ≥ 0.95` / `bin_label_agreement ≥ 0.98`. If drift exceeds those thresholds, the protocol forbids widening — we instead match R's tie-breaking more faithfully (implementation correction, not gate change).
- **Embedding speedup is modest.** `make_chrom_matrix` is largely cooler/HDF5-bound; the Rust win is the upper-tri extraction + scaling. Honest target: `< 2×` on real data. Recorded in `RECONSTRUCTION_REPORT §6 — Known limitations`.
- **Fixtures.** Loop calling depends on `.E.cool` / `.T.cool` group artefacts that are mid-pipeline; we synthesise small versions in `data/fixtures/`. If a public end-to-end fixture surfaces later we add it.
- **Shuffle path** in `loop_bkg` (`shuffle=True`) is RNG-driven; that branch is excluded from the gate (out-of-scope per the protocol, like other stochastic paths). Documented in `RECONSTRUCTION_REPORT §6`.
- **`schicluster` env is Python 3.6.** The reference driver runs there in isolation — no `schicluster_rs` import needed. The single-process "import both" pattern is unavailable; cross-env dump-and-compare is the design.

## 12. Out of scope

- SVD (`TruncatedSVD`), FDR (`multipletests`), paired t-test — remain Python by user decision.
- Cooler / HDF5 / NetCDF / AnnData I/O — remain Python.
- ProcessPoolExecutor orchestration — remains Python.
- Compartment CpG-ratio precomputation (`bedtools nuc` shell-out) — remains Python.
- Renaming `schicluster_rs` to `rs_schicluster` — kept as alias for backwards compatibility.

## 13. Approval gate

This spec is the read-only design artefact for Phase 0. Once approved by the user, `data/manifest.yaml` is written from §5, the rebuildpy artefacts in §4 are created, and Phase 3 (the two-agent loop) begins. After approval, the implementation plan is produced by `superpowers:writing-plans` — not this document.
