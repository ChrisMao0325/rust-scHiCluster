# rust-scHiCluster

Fast scHiCluster-compatible single-cell Hi-C analysis powered by Rust.

`schicluster-rs` is a drop-in accelerator for
[scHiCluster](https://github.com/zhoujt1994/scHiCluster) (Zhou *et al.* 2019,
PNAS). It runs the same analyses — imputation, loop calling, domains and
insulation, compartments, embeddings, gene scores and contact-distance decay —
and produces the same numbers, but the heavy per-chromosome maths runs in Rust
instead of Python.

**What this means in practice:**

- **Your existing scripts keep working.** One `patch_schicluster()` call swaps
  the fast kernels in behind the scenes. You don't change any other line.
- **Results match upstream.** Every output is checked against the original
  Python (and R) implementation on every release; most are bit-for-bit
  identical. See [Accuracy](#accuracy).
- **R is no longer required.** Domain calling reimplements TopDom natively, so
  you don't need R, `rpy2` or the `Matrix` R package installed.
- **Typical speedups:** 4–6× on imputation, 10× on insulation, 20× on
  compartments, and ~200× on gene scoring. Full numbers in
  [Performance](#performance).

---

## Installation

```bash
pip install schicluster-rs
```

Pre-built wheels cover Linux x86-64 (manylinux2014) for CPython 3.9–3.13. Other
platforms build from source and need Rust ≥ 1.78 available.

You will normally also want upstream scHiCluster, since `schicluster-rs`
accelerates it rather than replacing its workflow entrypoints:

```bash
pip install schicluster
```

<details>
<summary>Installing from a git checkout</summary>

Requires Rust ≥ 1.78 and maturin ≥ 1.4:

```bash
git clone https://github.com/omicverse/rust-scHiCluster
cd rust-scHiCluster
pip install maturin
maturin develop --release      # builds and installs into the active env
```
</details>

**Optional dependencies.** `cooler` is needed for anything that reads `.cool`
files (imputation, gene scores in `impute` mode, embeddings); install it with
`pip install cooler`. `pytables` is needed to write `.hdf` outputs.

---

## Quick start

### Speed up an existing scHiCluster script

Add two lines at the top. Nothing else changes.

```python
import schicluster_rs
schicluster_rs.patch_schicluster()

# ...your existing scHiCluster code, unmodified...
from schicluster.impute.impute_chromosome import impute_chromosome
impute_chromosome(scool_url='cells.scool', chrom='chr1', resolution=25_000,
                  output_path='chr1.hdf', pad=1, std=1.0,
                  rp=0.5, tol=0.01, output_dist=10_050_000)
```

`patch_schicluster()` redirects scHiCluster's per-chromosome functions to the
Rust versions. It returns `True` on success and `False` — never an exception —
if the Rust extension or upstream scHiCluster is unavailable, so a script
degrades to plain Python rather than crashing.

### Speed up an existing command-line pipeline

Replace `hicluster` with `schicluster-rs`. The options are identical.

```bash
# before
hicluster domain --cell_table_path cell_table.tsv --output_prefix sample \
    --resolution 25000 --window_size 10 --cpu 32

# after
schicluster-rs domain --cell_table_path cell_table.tsv --output_prefix sample \
    --resolution 25000 --window_size 10 --cpu 32
```

### Use a single function directly

```python
import schicluster_rs
import pandas as pd

# Per-gene contact scores for one cell, from an imputed .cool file
chrom_sizes = pd.read_csv('chrom.sizes', sep='\t', header=None,
                          index_col=0).squeeze('columns')
gene_meta = pd.read_csv('gene_meta.tsv', sep='\t', header=None, index_col=3)

# gene_score_impute is the *per-cell worker*. In a normal run the
# `gene-score` command converts coordinates to bins before calling it, so
# when you call it directly you have to do that conversion yourself:
gene_meta[1] = gene_meta[1] // 10_000      # start bp -> bin
gene_meta[2] = gene_meta[2] // 10_000      # end   bp -> bin

scores = schicluster_rs.gene_score_impute('cell.cool', chrom_sizes, gene_meta)
```

> **On the command line you do not do this.** `schicluster-rs gene-score` reads
> `gene_meta.tsv` in base pairs and applies `--resolution` (and `--slop`) for
> you. The conversion above is only needed when calling the per-cell function
> directly, bypassing that step.

### Controlling parallelism

```python
schicluster_rs.set_num_threads(4)
```

Results never depend on the thread count — only speed does. When you also fan
out across processes (`--cpu N`, `ProcessPoolExecutor`, snakemake), set this so
that `processes × threads ≈ cores`. The default lets every worker grab all
cores, which oversubscribes badly: 8 workers × 2 threads beat 8 workers ×
all-cores by **3×** in our tests.

---

## Python API

Import `schicluster_rs` and call these directly, or let `patch_schicluster()`
route upstream's calls to them for you.

### Setup

| Function | Purpose |
|---|---|
| `patch_schicluster()` | Redirect upstream scHiCluster's hot functions to Rust. Returns `bool`. |
| `set_num_threads(n)` | Set the Rust thread-pool size. Call before other functions. |

### Imputation

```python
impute_chromosome(scool_url, chrom, resolution, output_path, *,
                  pad=1, std=1.0, rp=0.5, tol=0.01,
                  output_dist=..., window_size=..., step_size=10_000_000,
                  min_cutoff=0, logscale=False, band_factor=0)
```

Random-walk-with-restart imputation for one chromosome of one cell; writes an
HDF5 file. `random_walk_cpu(p, rp=0.5, tol=0.01, n_iter=30)` exposes just the
random-walk step on a sparse matrix.

> `band_factor` is an opt-in speed/accuracy trade-off, off by default. Any
> non-zero value restricts the random walk to a diagonal band and makes results
> *approximate*. Leave it at `0` unless you have specifically validated it.

### Domains and compartments

```python
insulation_score_chrom(matrix, window_size=10, save_count=False)
single_chrom_compartment(matrix, cpg_ratio, calc_strength=False)
```

TopDom domain calling is reached through `patch_schicluster()` — it replaces
upstream's `rpy2` bridge, so **R is not needed**.

### Loop calling

```python
loop_bkg_chrom(...)                      # per-cell background normalisation
merge_cells_for_single_chromosome(...)   # sum cells into a group
loop_background(...)                     # four background kernel scans
find_summit(...)                         # merge loop pixels into summits
```

The statistics on top (paired *t*-test, BH-FDR) remain upstream's Python.

### Gene scores

```python
gene_score_impute(cell_path, chrom_sizes, gene_meta)
gene_score_raw(cell_path, chrom_sizes, gene_meta, resolution,
               chrom1, pos1, chrom2, pos2)
```

`impute` mode reads imputed `.cool` files, `raw` mode reads a contact file
directly.

**These are per-cell workers, not the whole analysis.** The `gene-score`
command wraps them: it reads your gene annotation in **base pairs**, applies
`--resolution` and `--slop` to convert to bins, then fans the workers out
across cells. Call them directly and you take on that conversion yourself —
`gene_meta` positions must already be in **bins**, i.e.
`(start - slop) // resolution` and `(end + slop) // resolution`.

See [Behaviour notes](#behaviour-notes) for one important quirk.

### Embeddings

```python
make_chrom_matrix(cell_table, chrom, nbins, output_path,
                  scale_factor, dist, resolution)
```

Builds the cell × feature matrix. The SVD afterwards stays scikit-learn's.

### Contact-distance decay

```python
compute_decay(cell_name, contact_path, bins, chrom_sizes, resolution,
              chrom1=1, chrom2=5, pos1=2, pos2=6)
```

Returns `[sparsity_frame, decay_frame]`, matching upstream.

Per-module guides with worked examples live in
[`tutorial/`](tutorial/README.md).

---

## CLI usage

`schicluster-rs <subcommand>` accepts exactly the same options as
`hicluster <subcommand>` — it applies the Rust patch, then hands off to
upstream's own argument parser.

| Subcommand | Faster? | Notes |
|---|---|---|
| `impute` (via `prepare-impute`) | **yes** | 4–6× on real cells |
| `domain` | **yes** | Insulation + native TopDom; **no R needed** |
| `compartment` | **yes** | ~20× on the per-chromosome kernel |
| `embedding` | **yes** | Modest; mostly I/O-bound. SVD stays scikit-learn |
| `gene-score` | **yes** | ~200× on the per-gene loop (`--mode impute`) |
| `contact-distance` | **yes** | ~4×; bounded by gzip decompression |
| `prepare-impute` | indirect | Writes the snakemake workflow |
| `filter-contact` | no | Pure I/O; included so one binary covers the pipeline |
| `cpg-ratio` | no | `bedtools nuc`; prerequisite for `compartment` |

For any subcommand not listed, use `hicluster` directly.

<details>
<summary>Complete worked pipeline</summary>

```bash
# 1. Blacklist-filter contact pairs
schicluster-rs filter-contact \
    --contact_table contact_table.tsv \
    --chrom_size_path chrom.sizes \
    --output_dir rmbkl/ \
    --blacklist_1d_path mm10-blacklist.v2.bed.gz \
    --chr1 1 --pos1 2 --chr2 3 --pos2 4

# 2. Generate per-chunk Snakefiles for imputation
schicluster-rs prepare-impute \
    --cell_table contact_table_rmbkl.tsv \
    --batch_size 1536 --pad 1 --cpu_per_job 96 \
    --chr1 1 --pos1 2 --chr2 3 --pos2 4 \
    --output_dir impute/100K/ \
    --chrom_size_path chrom.sizes \
    --output_dist 500000000 --window_size 500000000 \
    --step_size 500000000 --resolution 100000

# 3. Cell embedding
schicluster-rs embedding \
    --cell_table_path impute/100K/cell_table.tsv \
    --output_dir embedding/ --chrom_size_path chrom.sizes \
    --dim 50 --dist 1000000 --resolution 100000 \
    --scale_factor 100000 --cpu 20 --norm_sig --save_raw

# 4a. CpG ratio (prerequisite for compartment)
schicluster-rs cpg-ratio \
    --fasta_path mm10.fa --hdf_output_path cpg_ratio_100k.hdf \
    --chrom_size_path chrom.sizes --resolution 100000

# 4b. Compartment score
schicluster-rs compartment \
    --cell_table_path impute/100K/cell_table.tsv \
    --output_prefix dataset/sample.impute \
    --cpg_profile_path cpg_ratio_100k.hdf --cpu 96

# 5. Domain boundaries + insulation (no R needed)
schicluster-rs domain \
    --cell_table_path impute/25K/cell_table.tsv \
    --output_prefix dataset/sample \
    --resolution 25000 --window_size 10 --cpu 96

# 6. Per-gene contact scores
schicluster-rs gene-score \
    --cell_table_path impute/10K/cell_table.tsv \
    --gene_meta_path gene_meta.tsv --resolution 10000 \
    --output_hdf_path gene_score.hdf --chrom_size_path chrom.sizes \
    --chr1 1 --pos1 2 --chr2 3 --pos2 4 --cpu 64 --mode impute

# 7. Contact-distance decay + per-chrom sparsity
schicluster-rs contact-distance \
    --contact_table contact_table_rmbkl.tsv \
    --chrom_size_path chrom.sizes --output_prefix dataset/sample \
    --resolution 10000 --chr1 1 --pos1 2 --chr2 3 --pos2 4 --cpu 20
```
</details>

### Snakemake

A Rust-backed loop-calling template ships at
`python/schicluster_rs/loop/snakemake_template_loop.txt`. Drop it in as a
replacement for upstream's `loop/snakemake_template_loop.txt`; it applies the
patch so every fanned-out rule uses the fast kernels.

---

## Input formats

All formats are unchanged from upstream scHiCluster — if your files already
work with `hicluster`, they work here.

### Contact pairs — `*.tsv` / `*.tsv.gz`

Tab-separated, **no header**. Column positions are configurable, which is why
every command takes `--chr1 / --pos1 / --chr2 / --pos2`. Defaults are the
4DN pairs layout (`--chr1 1 --pos1 2 --chr2 5 --pos2 6`); output from
`filter-contact` typically uses `--chr1 1 --pos1 2 --chr2 3 --pos2 4`.

```
readname   chr1   1000000   chr1   1050000
```

Gzip is detected by the `.gz` suffix.

### Chromosome sizes — `chrom.sizes`

Two columns, tab-separated, no header. Standard UCSC format.

```
chr1    248956422
chr2    242193529
```

Only chromosomes listed here are analysed; everything else is silently dropped.

### Cell table — `cell_table.tsv`

Two columns, tab-separated, no header: cell ID, then the path to that cell's
file (a contact file or a `.cool`, depending on the step).

```
cell_A    /data/rmbkl/cell_A.contact.rmbkl.tsv.gz
cell_B    /data/rmbkl/cell_B.contact.rmbkl.tsv.gz
```

### Gene metadata — `gene_meta.tsv`

Four columns, tab-separated, no header: chromosome, start, end, gene ID.

Coordinates are in **base pairs** — write them exactly as they come from your
annotation. `schicluster-rs gene-score` divides by `--resolution` (after
applying `--slop`) internally. The only time you convert to bins yourself is
when calling `gene_score_impute` / `gene_score_raw` directly, since those are
the per-cell workers that sit downstream of the conversion.

```
chr1    11121    24894    ENSG00000290825.2
chr1    12010    13670    ENSG00000223972.6
```

### Matrices — `.cool` / `.scool`

Standard [cooler](https://github.com/open2c/cooler) files. Imputed matrices
store `count` as float32.

### CpG profile — `.hdf`

Produced by `schicluster-rs cpg-ratio`; a table with `chrom` and `cpg_ratio`
columns. Required by `compartment`.

---

## Performance

Measured against the original Python implementation. Speedups depend on
chromosome size and resolution.

| Analysis | Speedup |
|---|---|
| Gene scoring (per-gene loop) | **~200×** |
| Compartment calling | **~20×** |
| Loop summit finding | **~20×** |
| Insulation score | **~10×** |
| Imputation (real cells, 25–100 kb) | **4–6×** |
| Contact-distance decay | **~4×** |
| Embedding feature extraction | **~2.5×** |

Combining Rust with sensible process/thread balancing gives a further ~3× on
multi-cell runs — see [Controlling parallelism](#controlling-parallelism).

### Behaviour notes

Two upstream behaviours are **deliberately preserved**, so your results stay
comparable with anything you have already produced:

- **Gene scoring, `--mode impute`: genes starting in the first bin of a
  chromosome score `0.0`.** This comes from how upstream slices the contact
  matrix. It is reproduced faithfully rather than "fixed", because changing it
  would silently alter every first-bin gene. `--mode raw` is unaffected.
- **Contact-distance: very long-range contacts are dropped.** The
  log-spaced distance histogram tops out around 231.7 Mb for human, while chr1
  is 249.0 Mb, so upstream discards the longest cis contacts — and so does
  this package.

---

## Citation

If you use this package, please cite the original scHiCluster paper:

> Zhou, J., Ma, J., Chen, Y., Cheng, C., Bao, B., Peng, J., Sejnowski,
> T. J., Dixon, J. R. & Ecker, J. R. (2019). *Robust single-cell Hi-C
> clustering by convolution- and random-walk-based imputation.* PNAS,
> 116(28):14011-14018.

## License

MIT.
