# Acceleration Log — full-kernel survey (Phase 6b)

> A second (E)-exact-only acceleration pass, prompted by a simple question the
> first pass could not answer: **had the other kernels ever been profiled?**
> They had not. [ACCELERATION_LOG.md](ACCELERATION_LOG.md) covered three
> workloads (`conv`, `gene_score`, `contact_distance`); this pass extends
> `examples/bench_phase6.py` to **eleven**, one per kernel in the crate, and
> acts on what that revealed.
>
> **Separate file, separate figure, on purpose.** `wall_clock_mean_s` here is
> the summed Rust wall-clock over the 11-workload set, which is not comparable
> with the 3-workload total in `ACCELERATION_LOG.md`. Plotting both series on
> one axis would show a spurious jump at the point the benchmark grew, so this
> series carries its own baseline and renders to
> `examples/evolution_survey.png`. Schema is the strict rebuildpy one, parsed by
> `engine/plot_evolution.py`.
>
> All timings are `--release` builds with BLAS and rayon pinned to 4 threads,
> one warmup run discarded, then 3 timed runs.

---

## Baseline — after ACCELERATION_LOG.md iter 1, all 11 workloads

```yaml
iter: 0
status: baseline
action: null
admissibility: null
playbook_section: null
wall_clock_mean_s: 1.6000
wall_clock_stddev_s: 0.0099
wall_clock_runs_s: [0.0279, 0.0308, 0.0290]
warmup_run_s: 0.0287
parity_metric: 0.0
parity_class: deterministic-bounded
parity_threshold: 1.0e-6
parity_passes: true
notes: |
  First measurement of every kernel in the crate against a Python reference.
  Two of the eleven came back slower than Python, and one of those was hidden
  behind a bad reference of my own making.

  | workload         | python   | rust     | speedup |
  |------------------|----------|----------|---------|
  | conv             | 0.0691 s | 0.0292 s | 2.4x    |
  | gaussian/impute  | n/a      | 0.4496 s | see below |
  | gene_score       | 1.0769 s | 0.0049 s | 217.8x  |
  | contact_distance | 0.2397 s | 0.0589 s | 4.1x    |
  | insulation       | 0.3531 s | 0.0334 s | 10.6x   |
  | compartment      | 2.6432 s | 0.1185 s | 22.3x   |
  | topdom           | n/a      | 0.0298 s | R reference, no in-process compare |
  | merge            | 0.0746 s | 0.5708 s | **0.1x  <- 8x SLOWER than scipy** |
  | find_summit      | 0.1536 s | 0.0075 s | 20.6x   |
  | scan_kernels     | 0.4234 s | 0.2895 s | 1.5x    |
  | embedding        | 0.0062 s | 0.0025 s | 2.5x    |

  Two reference bugs were found and fixed before trusting any of this, both of
  which had inverted a verdict:

  * **compartment** first measured 0.1x (Rust 10x slower). The reference was a
    dense numpy matvec that skipped `compartment_strength` entirely, while the
    Rust call was asked for it — and `compartment_strength` does n sparse
    `.diagonal(i)` calls. With a faithful transcription of upstream's algorithm
    the true figure is **22.3x faster**.
  * **merge** first measured 2.8x faster. The reference was a pure-Python dict
    loop, a strawman: upstream actually does `e_sum += matrix` with scipy
    sparse addition at C level. Against the real reference merge is **8x
    slower**, not 2.8x faster.

  The lesson is recorded because it generalises: a benchmark whose reference is
  wrong does not report "no result", it reports a confident wrong result, and
  the sign of the error is unpredictable.

  `gaussian` has no in-process Python column — a fair comparison is the whole
  upstream `impute_chromosome`, which lives in the Python 3.6 reference env.
  That end-to-end measurement is the standing real-data harness at
  `../rust-scHiCluster-benchmark` (5 real prostate-cancer cells x 2
  resolutions x 2 chromosomes), and it is used for iter 1 below.
```

---

## iter 1 — hoist mirror-index tables out of gaussian_filter_2d

```yaml
iter: 1
status: ACCEPT
action: hoist_mirror_index_tables_gaussian
playbook_section: "§3.4"
admissibility: exact
admissibility_evidence: |
  Identical argument to ACCELERATION_LOG.md iter 1, applied to a different
  function: mirror_index is a pure function of (offset, extent) and never reads
  the data, so tabulating it per axis reproduces the same index sequence and
  therefore the same float operand order. Verified on real data, not just the
  gate: re-imputing 100 kb chr1 and chr19 for a real cell gives
  max|new_rust - old_rust| = 0.0 (bit-identical), with the error against
  upstream unchanged at 4.470348358154297e-08 — exactly the value the June
  benchmark recorded. The 21-output gate stays green.
wall_clock_mean_s: 1.5866
wall_clock_stddev_s: 0.0099
wall_clock_runs_s: [0.4319, 0.4408, 0.4360]
warmup_run_s: 0.4405
speedup_vs_previous: 1.008
speedup_vs_baseline: 1.008
parity_metric: 0.0
parity_class: deterministic-bounded
parity_threshold: 1.0e-6
parity_passes: true
notes: |
  This is the correction of a wrong claim. When the first acceleration pass
  fixed conv, the impute path was assumed to have inherited the win because it
  "uses the convolution". It does not: lib.rs has its own separable
  gaussian_filter_2d that never calls convolve2d_mirror, and it carried the
  identical defect — mirror_index recomputed per (i, j, kx), twice (once per
  separable pass). For a 25 kb chr1 (n = 7820, kernel length 3) that is ~367M
  integer modulos per imputed chromosome.

  The synthetic `gaussian` workload barely moves (0.4496 -> 0.4362 s, 1.03x)
  because at 2048x2048 with the full output triangle the RWR fixed point
  dominates and the Gaussian is a small fraction of the total. That workload is
  a poor probe for this change, and the summed-total delta above understates it
  badly.

  The real-data harness tells the true story. Re-timing the Rust side across
  5 real cells (upstream numbers unchanged, so they are reused):

  | res    | chrom | rust before | rust after | improvement | vs upstream        |
  |--------|-------|-------------|------------|-------------|--------------------|
  | 25000  | chr1  | 11.944 s    | 10.792 s   | 1.11x       | 4.79x (was 4.33x)  |
  | 25000  | chr19 |  0.637 s    |  0.461 s   | 1.38x       | 4.59x (was 3.32x)  |
  | 100000 | chr1  |  0.811 s    |  0.618 s   | 1.31x       | 5.60x (was 4.27x)  |
  | 100000 | chr19 |  0.187 s    |  0.053 s   | **3.53x**   | **3.68x (was 1.04x)** |

  The 100 kb chr19 row is the important one. That configuration previously
  showed **1.04x** — the Rust port bought a user essentially nothing on small
  chromosomes at coarse resolution, because the un-hoisted Gaussian dominated
  once the matrix was small enough for RWR not to. It is now 3.68x. The
  improvement is largest exactly where the port used to look pointless.
```

---

## iter 2 — replace merge's BTreeMap with a packed key and a stable radix sort

```yaml
iter: 2
status: ACCEPT
action: merge_flat_radix_sort_and_reduce
playbook_section: "§3.4"
admissibility: exact
admissibility_evidence: |
  Two properties carry the (E) claim, and both are load-bearing:

  1. An LSD radix sort is **stable by construction** — each pass scans forward
     and appends at per-digit running offsets — so equal keys keep their input
     order and every key's f64 additions happen in exactly the order the
     BTreeMap performed them. `sort_unstable_by_key` was measured at ~2x faster
     still and is deliberately NOT used: it would reorder those additions,
     making the rewrite (B).
  2. The key is packed as `row * ncols + col`, whose ascending order is
     row-major order, reproducing BTreeMap's (row, col) tuple ordering exactly.
     Emission order is therefore unchanged.

  Confirmed by the gate: merge.e_sum stays at 0.0 and merge.e2_sum at
  1.1920928955078125e-07, the same values recorded since Phase 1.
wall_clock_mean_s: 1.0527
wall_clock_stddev_s: 0.0135
wall_clock_runs_s: [0.0532, 0.0582, 0.0531]
warmup_run_s: 0.0551
speedup_vs_previous: 1.508
speedup_vs_baseline: 1.520
parity_metric: 0.0
parity_class: deterministic-bounded
parity_threshold: 1.0e-6
parity_passes: true
notes: |
  merge went 0.5708 -> 0.0548 s, a **10.4x** speedup on that kernel and 1.52x
  on the summed benchmark, flipping it from 0.1x to **1.5x faster than scipy**.

  The old implementation used two BTreeMap<(u32, u32), f64>, so every input
  triplet paid two O(log k) tree descents with a heap-allocated node per unique
  key. On 50 cells x 40k nnz that is 4M tree operations and ~2M allocations.
  The BTreeMap was not an accident — Phase 1 chose it specifically to get
  deterministic row-major emission, and that requirement is real. The rewrite
  keeps the guarantee and drops the cost.

  Intermediate result worth recording: a plain `sort_by_key` (stable comparison
  sort) already gave 0.586 -> 0.097 s, and the radix sort took it to 0.052 s.
  The diagnostic that justified going further was measuring
  `sort_unstable_by_key` at 0.048 s, which showed the sort held ~2x of headroom
  — enough to be worth an O(n) stable sort, and confirming the remaining cost
  was the sort rather than the reduce.
```

---

## Kernels examined and left alone

Recorded so the absence of an iteration is a decision rather than an oversight:

| Kernel | Measured | Verdict |
|---|---|---|
| `gene_score.rs` | 217.8x | Nothing to do. |
| `compartment.rs` | 22.3x | Nothing to do (once the reference was fixed). |
| `find_summit.rs` | 20.6x | Nothing to do. Upstream is a pure-Python windowed double loop plus `heapq`. |
| `insulation.rs` | 10.6x | Nothing to do. The prefix-sum rewrite that would speed it further is (B) — rejected in [ACCELERATION_LOG.md](ACCELERATION_LOG.md) iter 3. |
| `contact_distance.rs` | 4.1x | Bounded by gzip inflate; no (E) headroom without changing the decompressor. |
| `embedding.rs` | 2.5x | Pure gather plus scalar multiply; already memory-bound. |
| `conv.rs` | 2.4x | Already accelerated in [ACCELERATION_LOG.md](ACCELERATION_LOG.md) iter 1. |
| `scan_kernels.rs` | 1.5x | Four convolutions, all routed through the already-accelerated `conv.rs`. Modest but positive; no further (E) candidate identified. |
| `topdom.rs` | n/a | Its reference is R via rpy2 in the Python 3.6 env, so there is no in-process comparison. Absolute cost is small (0.030 s at n=2000). Not investigated further. |
| `loop_bkg.rs` | n/a | Routes through `conv.rs`, so it inherited iter 1 of the other log. Not separately profiled. |
