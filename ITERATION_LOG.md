# ITERATION_LOG — Phase 3 / Phase 5 Acceleration attempts

> One YAML block per attempt (accepted, rejected for gate, rejected for no speedup, or rejected for inadmissibility). Format follows `rebuildpy/PROTOCOL.md §3b`.

---

iteration: 0
title: Baseline serial translation
admissibility: E
action: |
  Direct translation of each Python function to serial-loop Rust.
  No rayon, no SIMD, no algebraic rewrites.
status: baseline
fixture: data/fixtures/conv_small.npz
timing: TBD-at-Phase-0-close
parity: TBD-at-Phase-0-close
notes: Iteration 0 is the equivalence baseline; Acceleration begins at iteration 1.

---
