# Acceleration Log — schicluster-rs (Phase 6)

> Phase 6's (E)-exact-only acceleration search. **Schema is strict**: the
> `## iter N — ...` headers and the fenced `yaml` blocks below are parsed by
> `engine/plot_evolution.py` to render `../rust-scHiCluster-benchmark/examples/evolution.png`. Keep the field
> names verbatim.
>
> This file is deliberately separate from [ITERATION_LOG.md](ITERATION_LOG.md),
> which records the per-phase **port** history (iterations 0-5) in this repo's
> own long-form format. Those entries cover five different modules and three
> different workloads, and iterations 1-3 carry no timings at all, so they
> cannot share an axis with an acceleration search. The evolution figure's
> subject is this file.
>
> All timings are `--release` builds, produced by `../rust-scHiCluster-benchmark/examples/bench_phase6.py`
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

  Benchmark workloads (see ../rust-scHiCluster-benchmark/examples/bench_phase6.py):
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

## iter 2 — enable target-cpu=native (rejected: no measurable gain)

```yaml
iter: 2
status: REJECT_SLOW
action: target_cpu_native
playbook_section: "§3.4"
admissibility: exact
admissibility_evidence: |
  No kernel in this crate enables reassociating fast-math flags, so a wider
  vector unit still executes the same fixed-order reduction. Confirmed
  empirically rather than assumed: rebuilt with RUSTFLAGS="-C
  target-cpu=native", the full 21-output gate stayed green and every metric was
  bit-identical to the portable build (conv.convolved 1.7881393432617188e-07,
  loop_bkg.E 5.960464477539062e-07, scan_kernels.donut 2.384185791015625e-07,
  insulation.score 5.960464477539063e-08). Admissible — but admissibility is
  not the reason it was rejected.
wall_clock_mean_s: 0.0943
wall_clock_stddev_s: 0.0007
wall_clock_runs_s: [0.0293, 0.0299, 0.0306]
warmup_run_s: 0.0284
speedup_vs_previous: 1.003
speedup_vs_baseline: 1.483
parity_metric: 0.0
parity_class: deterministic-bounded
parity_threshold: 1.0e-6
parity_passes: true
notes: |
  Rejected for no speedup, not for inadmissibility. 0.0943 s against the
  portable build's 0.0946 s is a 0.3% difference on a summed stddev of ~0.0008
  — indistinguishable from noise. Per-workload it is the same story: conv
  0.0299 vs 0.0306, contact_distance 0.0590 vs 0.0591, and gene_score actually
  measured slightly worse (0.0054 vs 0.0049).

  That result is unsurprising in hindsight. After iter 1 the conv inner loop is
  bound by memory access through two index tables, not by arithmetic width, and
  the other two kernels are bound by gzip inflate and by sparse-index chasing
  respectively. None of them is a dense FMA loop, which is what wider vectors
  would help.

  Cargo.toml is unchanged and the released wheel stays portable. Since there is
  no measurable gain, PERFORMANCE.md does not advertise this as an opt-in knob
  — recommending a non-portable build for a 0.3% noise-level difference would
  be misleading.
```

---

## iter 3 — prefix-sum sliding window for insulation (rejected: inadmissible)

```yaml
iter: 3
status: REJECT_INADMISSIBLE
action: prefix_sum_insulation_window
playbook_section: "§2.1"
admissibility: bounded
admissibility_evidence: |
  A prefix-sum window computes sum(w) as P[b] - P[a]. In exact arithmetic that
  is an identity, but in floating point it is not: P[b] and P[a] are each the
  result of accumulating over a prefix far longer than the window, so the
  subtraction cancels two large, separately-rounded quantities. The error is
  governed by the prefix length, not the window length, so this is a (B)
  bounded-epsilon rewrite, not (E).
perturbation_bound: |
  Direct window sum:  |dS| <= (w-1)*eps_32*max|x|
  Prefix-sum window:  |dS| <= (n-1)*eps_32*max|x|, plus cancellation
                             amplification |P[b]| / |P[b] - P[a]|
  With the Phase 2 fixture (n = 80, w = 5, eps_32 = 1.19e-7) the prefix form is
  ~20x looser before cancellation is considered; on a real 25 kb chr1
  (n = 7820) it is ~2000x looser. insulation.score currently sits at 5.96e-08
  against a 1.0e-6 gate, i.e. ~17x of headroom — which a 2000x looser bound
  would consume outright.
wall_clock_mean_s: null
wall_clock_stddev_s: null
parity_metric: null
parity_passes: null
notes: |
  Not implemented. Phase 6's admissibility policy is (E)-exact only, so a (B)
  rewrite is out of scope regardless of its speedup.

  Recorded here rather than skipped silently because the predecessor design
  spec (docs/superpowers/specs/2026-06-02-rust-port-loop-domain-compartment-
  embedding-design.md §7) listed "banded layout for sliding-window insulation"
  in its (E) column, and that classification is wrong for a prefix-sum
  formulation. insulation.rs keeps its direct per-window reduction.

  A banded *storage* layout with an unchanged per-window reduction would still
  be (E) and remains available to a future phase; it is the prefix-sum
  arithmetic, not the banding, that crosses into (B).

  This block is intentionally absent from ../rust-scHiCluster-benchmark/examples/evolution.png:
  plot_evolution only plots entries whose status is `baseline` or `ACCEPT`.
```

---
