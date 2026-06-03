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
