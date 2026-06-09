# ITERATION_LOG — Phase 3 / Phase 5 Acceleration attempts

> One YAML block per attempt (accepted, rejected for gate, rejected for no speedup, or rejected for inadmissibility). Format follows `rebuildpy/PROTOCOL.md §3b`.

---

iteration: 0
title: Baseline serial-row-parallel translation of convolve2d_mirror
admissibility: E
action: |
  Direct translation of scipy.ndimage.convolve(mode='mirror') to Rust.
  Parallel across output rows (independent), serial within each row's
  reduction. No SIMD intrinsics, no algebraic rewrites. mirror_index
  shared with the pre-existing gaussian_filter_2d via utils.rs.
status: accepted
fixture: data/fixtures/conv_small.npz
timing:
  python_ref_s: 0.0052
  rust_cand_s: 0.018
parity:
  conv.convolved:
    metric_value: 1.788e-07
    threshold: 1.0e-6
    pass: true
notes: |
  Baseline for the shared 2-D convolution primitive. Loop module
  (Phase 1) will exercise this same function 5x per chrom.

---

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
  for deterministic row-major emission. find_summit uses Rust BinaryHeap
  with deterministic ascending-idx tie-break.
status: accepted
fixture: data/fixtures/loop_small.npz + data/fixtures/loop_small.cool
parity:
  loop_bkg.E:         { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  loop_bkg.T:         { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  merge.e_sum:        { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  merge.e2_sum:       { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  scan_kernels.bl:    { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  scan_kernels.donut: { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  scan_kernels.h:     { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  scan_kernels.v:     { class: deterministic-bounded, threshold: 1.0e-6, pass: true }
  find_summit.idx:    { class: ranked, threshold: 0.99, pass: true }
  find_summit.sizes:  { class: classification, threshold: 1.0, pass: true }
notes: |
  shuffle=True path of loop_bkg falls back to upstream Python (RNG; out of
  parity scope per spec §11). One real bug surfaced during gate iteration:
  percentile_linear in loop_bkg.rs was sorting the input slice in place,
  corrupting downstream zscore. Fixed by sorting a copy. Three iterations
  to clear the gate (loop_bkg parity → harness alignment → multiclass
  classification average='macro'). Final pytest: 11 passed, 6 skipped.

---

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
  insulation.score:                  { class: deterministic-bounded, threshold: 1.0e-6, pass: true, metric: 5.96e-08 }
  topdom.bed.interval_jaccard:       { class: ranked,                threshold: 0.95,  pass: true, metric: 1.0 }
  topdom.bed.bin_label_agreement:    { class: classification,        threshold: 0.98,  pass: true, metric: 1.0 }
notes: |
  patch_schicluster() now monkey-patches schicluster.domain.call_domain's
  rpy2 `r` global with a stub whose RunTopDom routes through the Rust
  kernel — that's how the upstream `call_domain_and_insulation` keeps
  working without edits (the rpy2 closure inside it now calls Rust).
  insulation_score_chrom is monkey-patched at module level directly.

  One bug surfaced during gate iteration: _topdom_chrom_to_df (Python
  layer, not Rust) had an off-by-one in chromEnd — R's Convert.Bin.To.
  Domain.TMP sets to.coord = from.coord of the next row, equivalent to
  to_coord[to_id + 1]. The Phase 2 plan's verbatim wrapper used
  to_coord[to_id], cutting every domain 10 kb short. Fixed at the Python
  layer; Rust topdom kernel was correct on the first build.

---

iteration: 3
title: Phase 3+4 — compartment + embedding ported, gate fully green (17/17)
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
  compartment.comp:          { class: deterministic-bounded, threshold: 1.0e-6, pass: true, metric: 0.0 }
  compartment.strength:      { class: deterministic-bounded, threshold: 1.0e-6, pass: true, metric: 0.0 }
  embedding.cell_by_feature: { class: deterministic-strict,  threshold: 0.0,    pass: true, metric: 0.0 }
notes: |
  Final 3-of-17 outputs turn green; manifest gate is now fully green
  (17 passed, 0 skipped). patch_schicluster() rebinds
  single_chrom_compartment + make_chrom_matrix at module level so
  upstream's multiprocess orchestrators transparently use Rust.

  One harness bug surfaced during gate iteration: rebuildpy's is_pass()
  for deterministic-strict uses strict-`<` against the threshold, which
  makes `threshold: 0.0` (exact f32 bit-equality, the right gate for
  embedding's pure-index op) mathematically unreachable. Fixed locally
  in tests/parity_harness.py with a deterministic-strict + threshold==0
  special case that uses `<= 0.0` — the minimal correct interpretation
  of "exact bit-equality". Manifest values unchanged.

---
