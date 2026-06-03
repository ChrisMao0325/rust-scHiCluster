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
