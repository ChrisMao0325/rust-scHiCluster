# Performance and accuracy

Summary of how fast `schicluster-rs` is and how closely it matches upstream
scHiCluster. For headline numbers in context, see the
[README](../README.md#performance).

Methodology, per-kernel tables, implementation internals and the acceleration
history live in the
[developer documentation](../../rust-scHiCluster-benchmark/README.md) — they
are deliberately kept out of the package docs so that using the package does
not require understanding the Rust.

## Speed

Measured against the original Python implementation, warmup excluded, 4
threads.

| Analysis | Speedup |
|---|---|
| Gene scoring (per-gene loop) | ~218× |
| Compartment calling | ~22× |
| Loop summit finding | ~21× |
| Insulation score | ~10.6× |
| Imputation, real cells at 25–100 kb | 4.6–5.6× |
| Contact-distance decay | ~4.1× |
| Convolution / embedding feature extraction | ~2.5× |
| Cell-to-group merge, loop background scans | ~1.5× |

Two figures deserve context:

- **Gene scoring** is ~218× steady-state but ~46× on a single one-shot call,
  which additionally pays rayon's thread-pool startup. Since gene scoring runs
  thousands of cells per job, the steady-state number is the one users see.
- **Imputation** benefits most on long chromosomes at fine resolution. The
  smallest configuration measured (100 kb chr19) is 3.68×.

### Thread configuration matters more than you might expect

Rust parallelism composes badly with process-level fan-out unless balanced. On
8 chr1 imputations in parallel:

| Configuration | Wall clock |
|---|---|
| 8 workers × all-cores each (the default) | 29 s |
| 8 workers × 2 rayon threads (16 cores total) | **9.4 s** |

A **3.1×** difference from configuration alone. Use
`schicluster_rs.set_num_threads(n)` so that `processes × threads ≈ cores`.
Results are unaffected by thread count — only speed is.

## Accuracy

Every release is validated against the original Python implementation (and, for
TopDom, R) through a **pre-registered parity gate**: 21 outputs, each with a
tolerance fixed before the code was written and never loosened afterwards.

**All 21 outputs pass.** Nine are bit-for-bit identical, including gene scores,
contact-distance decay and embedding features. The worst deviation anywhere is
`9.54e-07` — below float32 precision, and far below any biological signal.

Full per-output table and reproduction instructions:
[`RECONSTRUCTION_REPORT.md`](RECONSTRUCTION_REPORT.md).

## Preserved upstream behaviours

Two upstream behaviours are reproduced deliberately rather than "fixed", so
your results stay comparable with anything you have already produced. Both are
documented where a user would meet them, in
[`tutorial/gene_score.md`](../tutorial/gene_score.md) and
[`tutorial/contact_distance.md`](../tutorial/contact_distance.md):

- **Gene scoring in `--mode impute`: genes starting in a chromosome's first bin
  score `0.0`**, because of how upstream slices the contact matrix. `--mode
  raw` is unaffected — the same gene can score `0.0` in one mode and non-zero
  in the other.
- **Contact-distance drops very long-range contacts.** The log-spaced histogram
  tops out near 231.7 Mb for human while chr1 is 249.0 Mb, so upstream
  discards the longest cis contacts.

## What runs in Rust and what does not

Rust handles the per-chromosome and per-cell numerical work. Everything else
stays Python, unchanged:

| Rust | Python |
|---|---|
| Imputation (RWR, Gaussian smoothing, SQRTVC) | cooler / HDF5 / NetCDF / AnnData I/O |
| Loop background, kernel scans, summit finding | Paired *t*-test and BH-FDR |
| Insulation score and TopDom (**no R needed**) | `TruncatedSVD` for embeddings |
| Compartment score and A/B strength | `bedtools nuc` for CpG ratios |
| Cell × feature extraction | `ProcessPoolExecutor` orchestration |
| Per-gene window sums | pandas matrix building in `gene-score --mode raw` |
| Contact-distance histogram and sparsity | |

This is why `import schicluster...` keeps working after
`patch_schicluster()` — only the inner numerical functions are swapped.
