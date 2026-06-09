# rust-scHiCluster

Rust re-implementation of the inner numerical pipeline of
[scHiCluster](https://github.com/zhoujt1994/scHiCluster) (Zhou *et al.*
2019, PNAS) — single-cell Hi-C contact-matrix imputation and loop
calling. Drop-in monkey-patch: upstream Python code stays unchanged, the
hot per-chrom kernels run in Rust.

Speed, accuracy, parity-gate status, algorithm notes and the
[rebuildpy](https://github.com/omicverse/rebuildpy) protocol artefacts
live in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Install

Requires **Rust ≥ 1.78** and **maturin ≥ 1.4**:

```bash
git clone https://github.com/omicverse/rust-scHiCluster
cd rust-scHiCluster
maturin develop --release   # build + install into the active venv
```

Or from PyPI (Linux x86_64 manylinux2014, CPython 3.10):

```bash
pip install schicluster-rs
```

Other platforms install from sdist and require Rust ≥ 1.78 in the build
environment. Pre-built wheels for Python 3.9–3.13 across linux / macOS /
Windows are added via cibuildwheel.

## Tutorial

**Drop-in monkey-patch (recommended)** — no code changes anywhere:

```python
import schicluster_rs
schicluster_rs.set_num_threads(2)        # 8 workers × 2 = 16 cores
schicluster_rs.patch_schicluster()

# every downstream call to scHiCluster's impute_chromosome,
# calculate_chrom_background_normalization, merge_cells_for_single_chromosome,
# loop_background, and find_summit now uses Rust:
from schicluster.impute.impute_chromosome import impute_chromosome
impute_chromosome(scool_url=..., chrom='chr1', resolution=25_000,
                  output_path=..., rp=0.5, tol=0.01,
                  pad=1, std=1.0, output_dist=10_050_000)
```

**Direct API** — call individual Rust kernels by name. Per-module usage
guides live in [`tutorial/`](tutorial/README.md):

- [tutorial/impute.md](tutorial/impute.md)
- [tutorial/loop.md](tutorial/loop.md)
- [tutorial/domain.md](tutorial/domain.md)
- [tutorial/compartment.md](tutorial/compartment.md)
- [tutorial/embedding.md](tutorial/embedding.md)

### Snakemake (loop calling)

A Rust-backed snakemake template ships at
`python/schicluster_rs/loop/snakemake_template_loop.txt`. It prepends
`patch_schicluster()` so every loop rule that snakemake fans out
transparently uses the Rust kernels. Drop it into your workflow as a
replacement for upstream's `loop/snakemake_template_loop.txt`.

### Command-line tool

A thin `schicluster-rs` CLI wraps a handful of `hicluster` subcommands —
it pre-applies `patch_schicluster()`, then hands argv off to upstream's
argparse. Swap `hicluster` → `schicluster-rs` in your scripts to get the
Rust backend with no other changes.

Currently exposed (all other subcommands: use `hicluster` directly):

| Subcommand | Rust speedup? | Notes |
|---|---|---|
| `schicluster-rs filter-contact ...` | no | Pure I/O passthrough. Included so a single binary covers the full pipeline. |
| `schicluster-rs prepare-impute ...` | indirect | Generates the snakemake workflow. The fanned-out `hic-internal impute-chromosome` rules still need `schicluster_rs` importable in each worker for the per-chrom kernel to route through Rust (e.g. via a sitecustomize that calls `patch_schicluster()`). |
| `schicluster-rs domain ...` | **yes** | Per-cell insulation score + native TopDom (drops the rpy2/R round-trip). `patch_schicluster()` rebinds both the insulation kernel and the TopDom closure inside upstream's `call_domain_and_insulation` orchestrator. |
| `schicluster-rs compartment ...` | **yes** | Per-chrom CpG-weighted compartment score + decay-normalised A/B/AB strength. Rebind happens at module level so the per-cell `ProcessPoolExecutor` inside `multiple_cell_compartment` transparently uses Rust. |
| `schicluster-rs embedding ...` | **yes** (modest) | Cell-by-feature upper-tri extraction with distance filter + scalar scaling, before SVD. SVD stays sklearn — intentional, see [docs/PERFORMANCE.md](docs/PERFORMANCE.md). Mostly I/O-bound. |
| `schicluster-rs cpg-ratio ...` | no | `bedtools nuc` + pandas. Included as the upstream prerequisite step for `compartment`. |

Concrete example covering the full
[scHiCluster tutorial flow](https://zhoujt1994.github.io/scHiCluster/intro.html):

```bash
# 1. Blacklist-filter contact pairs (no Rust speedup, single binary surface)
schicluster-rs filter-contact \
    --contact_table contact_table.tsv \
    --chrom_size_path chrom_sizes.txt \
    --output_dir rmbkl/ \
    --blacklist_1d_path mm10-blacklist.v2.bed.gz \
    --chr1 1 --pos1 2 --chr2 3 --pos2 4

# 2. Generate the per-chunk Snakefiles for imputation
schicluster-rs prepare-impute \
    --cell_table contact_table_rmbkl.tsv \
    --batch_size 1536 --pad 1 --cpu_per_job 96 \
    --chr1 1 --pos1 2 --chr2 3 --pos2 4 \
    --output_dir impute/100K/ \
    --chrom_size_path chrom_sizes.txt \
    --output_dist 500000000 --window_size 500000000 --step_size 500000000 \
    --resolution 100000

# 3. Cell embedding (Rust-backed extraction; SVD stays sklearn)
schicluster-rs embedding \
    --cell_table_path impute/100K/cell_table.tsv \
    --output_dir embedding/ \
    --chrom_size_path chrom_sizes.txt \
    --dim 50 --dist 1000000 --resolution 100000 --scale_factor 100000 \
    --cpu 20 --norm_sig --save_raw

# 4a. CpG ratio (prerequisite for compartment)
schicluster-rs cpg-ratio \
    --fasta_path mm10_with_chrl.fa \
    --hdf_output_path cpg_ratio_100k.hdf \
    --chrom_size_path chrom_sizes.txt \
    --resolution 100000

# 4b. Compartment score (Rust-backed per-chrom)
schicluster-rs compartment \
    --cell_table_path impute/100K/cell_table.tsv \
    --output_prefix dataset/sample.impute \
    --cpg_profile_path cpg_ratio_100k.hdf \
    --cpu 96

# 5. Domain boundaries + insulation (Rust-backed, no R / rpy2 needed)
schicluster-rs domain \
    --cell_table_path impute/25K/cell_table.tsv \
    --output_prefix dataset/sample \
    --resolution 25000 --window_size 10 --cpu 96
```

Run `schicluster-rs --help` for the dispatcher help, or
`schicluster-rs <subcommand> --help` for upstream's per-subcommand
options.

## Citation

If you use this package, please cite the original scHiCluster paper:

> Zhou, J., Ma, J., Chen, Y., Cheng, C., Bao, B., Peng, J., Sejnowski,
> T. J., Dixon, J. R. & Ecker, J. R. (2019). *Robust single-cell Hi-C
> clustering by convolution- and random-walk-based imputation.* PNAS,
> 116(28):14011-14018.

## License

MIT.
