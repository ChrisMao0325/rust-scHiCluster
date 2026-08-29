# schicluster_rs tutorials

Per-module usage guides. Each shows the direct API, the drop-in
monkey-patch route, and the equivalent command line.

| Module | Guide | Speedup |
|---|---|---|
| Imputation | [impute.md](impute.md) | 4–6× on real cells |
| Loop calling | [loop.md](loop.md) | per-chromosome kernels |
| Domains + insulation | [domain.md](domain.md) | ~10×, and **no R required** |
| Compartments | [compartment.md](compartment.md) | ~20× |
| Embedding | [embedding.md](embedding.md) | modest; mostly I/O-bound |
| Gene scores | [gene_score.md](gene_score.md) | ~200× on the per-gene loop |
| Contact distance | [contact_distance.md](contact_distance.md) | ~4× |

## Two integration styles

**Drop-in monkey-patch (recommended for existing pipelines).** Apply
`schicluster_rs.patch_schicluster()` once and your existing
`from schicluster.* import ...` calls transparently use the Rust kernels.
No edits to existing scripts, no changes to your snakemake workflows.
See the top-level [README](../README.md) Tutorial section.

**Direct API (these tutorials).** Import functions from `schicluster_rs`
and call them by name. Useful when you're writing new code, want a hard
dependency on `schicluster_rs` rather than going through upstream
`schicluster`, or want to use a single kernel without booting the
upstream module's orchestration.

Both end up running the same Rust code. The monkey-patch swaps function
identities inside the upstream modules; the direct API gives you the
functions by name from `schicluster_rs`. Mix and match freely — they
share the same underlying Rust extension.

## Do the results match upstream?

Yes. Every function below is validated against the original Python
implementation on each release, against tolerances fixed in advance. All 21
checked outputs pass and nine are bit-for-bit identical. See
[docs/PERFORMANCE.md](../docs/PERFORMANCE.md) for the summary and
[docs/RECONSTRUCTION_REPORT.md](../docs/RECONSTRUCTION_REPORT.md) for
per-output numbers.

## What stays in Python

The Rust port is deliberately scoped to the per-chrom / per-cell
numerical hot paths. Everything below stays in Python:

- cooler / HDF5 / netCDF / AnnData I/O
- the paired `t`-test and `statsmodels.stats.multitest` FDR in loop
  calling
- `sklearn.decomposition.TruncatedSVD` in embedding
- `bedtools nuc` shell-out in `cpg-ratio`
- `ProcessPoolExecutor` orchestration across cells

This means imports from `schicluster.*` still work; we only swap the
hot inner functions.
