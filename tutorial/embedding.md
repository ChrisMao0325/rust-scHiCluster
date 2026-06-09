# Embedding (Phase 4)

Cell-by-feature upper-triangle extraction with distance filter and
scalar scaling — the cell × feature matrix that becomes input to SVD
for single-cell Hi-C embedding. Ported from
`schicluster.embedding.calc_embedding.make_chrom_matrix`.

**Out of scope (stays sklearn):** `TruncatedSVD`. The parity gate
targets the pre-SVD matrix; SVD itself has no element-wise parity
(rotation / sign ambiguity), and it's not the bottleneck — see
[docs/PERFORMANCE.md](../docs/PERFORMANCE.md).

## Public API

```python
from schicluster_rs import make_chrom_matrix
```

Drop-in for `schicluster.embedding.calc_embedding.make_chrom_matrix`.
Reads cool files per cell in Python (preserving upstream's I/O), then
hands the dense per-cell matrices to a Rust extraction kernel and
writes a single `.npz`.

```python
import pandas as pd
from schicluster_rs import make_chrom_matrix

cell_table = pd.read_csv('impute/100K/cell_table.tsv',
                         sep='\t', index_col=0, header=None).squeeze(axis=1)
make_chrom_matrix(
    cell_table=cell_table,
    chrom='chr1',
    nbins=2475,
    output_path='embedding/raw/chr1.npz',
    scale_factor=100_000,
    dist=1_000_000,
    resolution=100_000,
)
```

| Arg | Notes |
|---|---|
| `cell_table` | pandas Series (cell_id → cool URL) |
| `chrom` | Chromosome name |
| `nbins` | Number of bins for this chrom |
| `output_path` | `.npz` output written via `np.savez` |
| `scale_factor` | Per-feature multiplier (default `100_000` in the upstream CLI) |
| `dist` | Distance cutoff in bp; only `(c − r) < dist/resolution + 1` pixels are kept |
| `resolution` | Bin size in bp |

The cell-by-feature matrix shape is `(n_cells, n_features)` where
`n_features` = count of `(r, c)` pairs with `c > r` and
`c - r < dist // resolution + 1`. Dtype is `float32`.

## Why the speedup is modest

The kernel is just `cells[i, r, c] * scale_factor` per pixel — a pure
f32 read + scalar multiply, no reduction. The bulk of wall-clock is
the cooler reads (Python). Honest expectation: ~2× on realistic input.

The reason it's still worth porting is the **deterministic-strict**
parity gate (`atol = 0`, exact f32 bit-equality), which acts as a
correctness contract — any future acceleration (rayon over cells,
SIMD over features) is constrained to preserve bit-identity.

## Parity

| Output | Class | Threshold | Measured |
|---|---|---|---|
| `embedding.cell_by_feature` | deterministic-strict | `0.0` (exact f32) | `0.0` |

See [docs/PERFORMANCE.md](../docs/PERFORMANCE.md).

## Patch-schicluster integration

`patch_schicluster()` rebinds
`schicluster.embedding.calc_embedding.make_chrom_matrix` at module level,
so the upstream `ProcessPoolExecutor`-fanned-out per-chrom workers inside
`embedding()` transparently use Rust:

```python
import schicluster_rs
schicluster_rs.patch_schicluster()

from schicluster.embedding import embedding
embedding(
    cell_table_path='impute/100K/cell_table.tsv',
    output_dir='embedding/',
    chrom_size_path='chrom_sizes.txt',
    dim=50,
    dist=1_000_000,
    resolution=100_000,
    scale_factor=100_000,
    cpu=20,
    norm_sig=True,
    save_raw=True,
)
```

The final per-chrom `TruncatedSVD` and the concat-then-SVD step at the
end stay in sklearn — those run after `make_chrom_matrix` has emitted
its `.npz` files.

## CLI

```bash
schicluster-rs embedding \
    --cell_table_path impute/100K/cell_table.tsv \
    --output_dir embedding/ \
    --chrom_size_path chrom_sizes.txt \
    --dim 50 --dist 1000000 --resolution 100000 --scale_factor 100000 \
    --cpu 20 --norm_sig --save_raw
```

`schicluster-rs embedding` pre-applies `patch_schicluster()` then
delegates to upstream's argparse, so you get Rust-backed extraction
with no other changes to your script.
