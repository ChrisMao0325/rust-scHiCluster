# `gene-score` — per-gene contact scores

Sums the contact matrix over a rectangular window around each gene's bins, one
score per gene per cell. Two modes: `impute` reads imputed `.cool` files,
`raw` builds the matrix from a contact file on the fly.

This is the step that dominates a gene-score job: with a human gene set,
upstream does roughly 78,700 separate matrix operations per cell. The
accelerated version does the same arithmetic without that overhead, giving
around **200×** on this loop.

## Direct API

`gene_score_impute` and `gene_score_raw` are the **per-cell workers**. In a
normal run the `gene-score` command reads your annotation in base pairs,
converts it to bins, and fans these out across cells — so calling them directly
means doing the conversion yourself:

```python
import schicluster_rs
import pandas as pd

chrom_sizes = pd.read_csv('chrom.sizes', sep='\t', header=None,
                          index_col=0).squeeze('columns')
gene_meta = pd.read_csv('gene_meta.tsv', sep='\t', header=None, index_col=3)
gene_meta = gene_meta[gene_meta[0].isin(chrom_sizes.index)]
# Bins, not base pairs. With --slop N the command would instead compute
# (start - N) // resolution and (end + N) // resolution.
gene_meta[1] = gene_meta[1] // 10_000
gene_meta[2] = gene_meta[2] // 10_000

scores = schicluster_rs.gene_score_impute('cell.cool', chrom_sizes, gene_meta)
```

`gene_score_raw` takes the contact file plus the column layout instead:

```python
scores = schicluster_rs.gene_score_raw(
    'cell.contact.tsv.gz', chrom_sizes, gene_meta,
    resolution=10_000, chrom1=1, pos1=2, chrom2=5, pos2=6)
```

## Drop-in monkey-patch

```python
import schicluster_rs
schicluster_rs.patch_schicluster()

from schicluster.draft.gene_score import gene_score
gene_score(cell_table_path=..., gene_meta_path=..., resolution=10_000,
           output_hdf_path=..., chrom_size_path=..., cpu=64, mode='impute')
```

`patch_schicluster()` rebinds `gene_score_impute` and `gene_score_raw` at module
level, and upstream's orchestrator resolves those names at `exe.submit()` time,
so its existing `ProcessPoolExecutor` picks up the Rust versions unchanged.

## Command line

```bash
schicluster-rs gene-score \
    --cell_table_path impute/10K/cell_table.tsv \
    --gene_meta_path gene_meta.tsv \
    --resolution 10000 \
    --output_hdf_path gene_score.hdf \
    --chrom_size_path chrom.sizes \
    --chr1 1 --pos1 2 --chr2 3 --pos2 4 \
    --cpu 64 --mode impute
```

`--cell_table_path` is a two-column headerless TSV of `cell_uid` and the path to
that cell's `.cool`. `--gene_meta_path` is a headerless TSV of `chromosome`,
`start`, `end`, `gene_id`.

**Coordinates here are in base pairs, not bins.** Pass your annotation exactly
as it comes; the command applies `--slop` and then divides by `--resolution`
before any scoring happens. Pre-dividing them yourself would bin them twice and
silently produce near-zero scores. (This is the opposite of the direct API
above, which is called *after* that conversion — see
[Direct API](#direct-api).)

`--slop N` widens every gene by `N` bp on each side before binning.

## Behaviour notes

**Genes starting at bin 0 score `0.0` in `--mode impute`.** Upstream's window is
`D[(xx-1):(yy+1), xx:(yy+2)]`. When `xx == 0` the row start is `-1`, which scipy
resolves to `n-1`, so the window is empty and the sum is zero. This port
reproduces that deliberately — "fixing" it would silently change every
first-bin gene's score and break comparability with results you already have.
`--mode raw` uses `[xx:(yy+1), xx:(yy+1)]`, with no `-1`, and does not share the
quirk: the same gene can score `0.0` in impute mode and non-zero in raw mode.

**Scores are bit-identical to upstream, not merely close.** Every gene score
matches the original Python implementation exactly, so swapping in this package
cannot shift a downstream result. (Achieving that took some care, because
imputed matrices are stored in single precision and summation order matters at
that precision — see the
[developer notes](../../rust-scHiCluster-benchmark/docs/IMPLEMENTATION.md#gene_scorers--per-gene-window-sums)
if you are curious.)

**`--mode raw` is far less accelerated than `--mode impute`.** In raw mode the
contact file still has to be parsed and binned in Python; only the per-gene
summation is accelerated. Impute mode is where the expensive loop lives, and
where the speedup is.

**Speed.** Around **200×** on the per-gene loop in steady state. A single
one-shot call is closer to 46×, because it also pays thread-pool startup —
which matters little for real jobs processing thousands of cells. See
[`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md).
