# Acceleration Log — schicluster-rs (Phase 6)

> Phase 6's (E)-exact-only acceleration search. **Schema is strict**: the
> `## iter N — ...` headers and the fenced `yaml` blocks below are parsed by
> `engine/plot_evolution.py` to render `examples/evolution.png`. Keep the field
> names verbatim.
>
> This file is deliberately separate from [ITERATION_LOG.md](ITERATION_LOG.md),
> which records the per-phase **port** history (iterations 0-5) in this repo's
> own long-form format. Those entries cover five different modules and three
> different workloads, and iterations 1-3 carry no timings at all, so they
> cannot share an axis with an acceleration search. The evolution figure's
> subject is this file.
>
> All timings are `--release` builds, produced by `examples/bench_phase6.py`
> with BLAS and rayon pinned to 4 threads, one warmup run discarded, then 3
> timed runs (mean + stddev; median + IQR if stddev exceeded 10% of the mean).
> `wall_clock_mean_s` is the summed Rust wall-clock across all three benchmark
> workloads, so a rewrite touching any one of them moves the number.

---

## Baseline — Phase 5 close (commit cf5dda9)

```yaml
iter: 0
status: baseline
action: null
admissibility: null
playbook_section: null
wall_clock_mean_s: 0.1399
wall_clock_stddev_s: 0.0001
wall_clock_runs_s: [0.0740, 0.0741, 0.0740]
warmup_run_s: 0.0739
parity_metric: 0.0
parity_class: deterministic-bounded
parity_threshold: 1.0e-6
parity_passes: true
notes: |
  Starting point for the acceleration search: every kernel as committed at the
  end of Phase 5, all 21 manifest outputs green, the worst per-output parity
  metric across the gate being 0.0.

  Benchmark workloads (see examples/bench_phase6.py):
    conv2d_mirror     1024x1024 input, 11x11 donut kernel
    gene_score        20000 windows over a 4000x4000 f32 CSR
    contact_distance  one real ProstateCancer/rmbkl cell

  Per-workload baseline (python -> rust, speedup):
    conv              0.0692 -> 0.0740 s   0.9x   <- SLOWER THAN SCIPY
    gene_score        1.0636 -> 0.0049 s   217.6x
    contact_distance  0.2656 -> 0.0610 s   4.4x

  Two things this baseline establishes, both of which correct earlier records:

  1. The conv primitive is genuinely slower than scipy.ndimage.convolve, not
     just at the 64x64 gate fixture (iteration 0 of ITERATION_LOG.md, which
     assumed thread-spawn overhead) but at 1024x1024 with an 11x11 kernel. It
     is the one real acceleration target in this crate, and it matters: the
     loop pipeline calls it five times per chromosome.

  2. gene_score's Phase 5 headline of 46.3x was warmup-inclusive — it charged
     rayon's thread-pool spin-up to the first and only timed call. Under the
     rebuildpy protocol (warmup discarded, 3 timed runs) the same workload is
     217.6x. Both numbers are real; they answer different questions. The
     warmup-inclusive figure is what a single one-shot call costs, the
     warmup-excluded figure is what steady-state per-cell work costs. Since
     gene-score runs thousands of cells per job, the steady-state number is the
     one users experience, and PERFORMANCE.md quotes both.
```

---

## iter 1 — hoist mirror-index tables out of the conv inner loop

```yaml
iter: 1
status: ACCEPT
action: hoist_mirror_index_tables
playbook_section: "§3.4"
admissibility: exact
admissibility_evidence: |
  mirror_index is a pure function of (offset, extent) — it never reads the
  input data. Tabulating it per axis and indexing the table produces the
  identical index sequence the inner loop computed before, so every float
  multiply-add happens in the same order with the same operands. Pre-flipping
  the kernel likewise changes only addressing, not the accumulation order. No
  perturbation bound is needed: the output is bit-identical, confirmed by
  conv.convolved holding its metric at exactly 1.7881393432617188e-07, the
  same value iteration 0 of ITERATION_LOG.md recorded.
wall_clock_mean_s: 0.0946
wall_clock_stddev_s: 0.0009
wall_clock_runs_s: [0.0297, 0.0317, 0.0303]
warmup_run_s: 0.0281
speedup_vs_previous: 1.479
speedup_vs_baseline: 1.479
parity_metric: 0.0
parity_class: deterministic-bounded
parity_threshold: 1.0e-6
parity_passes: true
notes: |
  The conv workload went 0.0740 -> 0.0306 s, a 2.42x speedup on that kernel and
  1.48x on the summed benchmark. More importantly it flips the verdict from the
  baseline: conv2d_mirror was 0.9x against scipy.ndimage.convolve (i.e. slower)
  and is now 2.3x faster.

  Why it was slow: the inner loop called mirror_index once per (i, j, p, q).
  For a 1024x1024 input with an 11x11 kernel that is ~127M calls, each doing an
  integer modulo plus two branches. The column mirror depends only on (j, q)
  and the row mirror only on (i, p), so neither belonged in the innermost loop
  at all. Both are now tabulated once per call — (nrows + kh) and (ncols + kw)
  entries — and the inner loop does an array load instead.

  This is the acceleration target the crate actually had. The other two
  workloads are unchanged, as expected: neither touches conv.
```

---
