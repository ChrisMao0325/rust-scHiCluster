# schicluster_rs tutorials

Per-module usage guides for the Rust-backed direct API.

| Module | Tutorial | Phase | Rust speedup |
|---|---|---|---|
| Impute | [impute.md](impute.md) | 0 | ~10× on long chromosomes |
| Loop calling | [loop.md](loop.md) | 1 | per-chrom inner kernels |
| Domain (insulation + TopDom) | [domain.md](domain.md) | 2 | drops rpy2 / R |
| Compartment | [compartment.md](compartment.md) | 3 | per-chrom inner kernel |
| Embedding (cell-by-feature) | [embedding.md](embedding.md) | 4 | modest (I/O-bound); SVD stays sklearn |
| Gene score | [gene_score.md](gene_score.md) | 5 | ~217× on the per-gene window loop |
| Contact distance | [contact_distance.md](contact_distance.md) | 5 | ~4.4× (gzip-inflate bound) |

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

## Parity guarantees

Every public function below is parity-gated against the upstream Python
reference. Per-output classes and thresholds are pre-registered in
[`data/manifest.yaml`](../data/manifest.yaml) and live snapshots of the
metrics are in [docs/ITERATION_LOG.md](../docs/ITERATION_LOG.md) (per-phase
ports) and [docs/ACCELERATION_LOG.md](../docs/ACCELERATION_LOG.md)
(the acceleration search). Full
gate status, accuracy notes and algorithm descriptions are in
[docs/PERFORMANCE.md](../docs/PERFORMANCE.md).

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
