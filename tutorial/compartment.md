# Compartment (Phase 3)

Per-chrom CpG-weighted compartment score and decay-normalised A/B/AB
strength, ported from
`schicluster.compartment.call_compartment.single_chrom_compartment` /
`compartment_strength`.

## Public API

```python
from schicluster_rs import single_chrom_compartment
```

## `single_chrom_compartment(matrix, cpg_ratio, calc_strength=False)`

```python
from schicluster_rs import single_chrom_compartment
import numpy as np

# matrix: symmetric Hi-C contact matrix (sparse or dense)
# cpg_ratio: length-n array of per-bin CpG ratios (pandas Series or numpy)
comp, scores = single_chrom_compartment(
    matrix,
    cpg_ratio,
    calc_strength=True,
)
# comp: shape (n_bins,), the CpG-weighted compartment score
# scores: shape (3,) = [AA, BB, AB]   (None when calc_strength=False)
```

| Arg | Default | Notes |
|---|---|---|
| `matrix` | — | Symmetric; sparse (`.toarray()` called) or dense |
| `cpg_ratio` | — | Per-bin CpG ratio; accepts pandas Series (`.values`) or numpy |
| `calc_strength` | `False` | If `True`, also compute the `[AA, BB, AB]` triple |

## What the kernel does

Matches the upstream sparse pipeline using a dense f64 working buffer:

1. Zero the matrix diagonal.
2. Add identity on any zero-sum columns (numerical safety).
3. Row-major normalise each value by its column sum.
4. `comp = matrix · cpg_ratio` (CpG-weighted matvec).
5. If `calc_strength`: compute per-distance decay (mean of each
   superdiagonal), divide the matrix by `decay[|c − r|]`, then sum the
   `(a, a)`, `(b, b)`, `(a, b)` quadrants where `a`/`b` are the top-20 %
   / bottom-20 % of `comp` restricted to bins with `cpg_ratio > 0`.

f64 throughout; comp returned as f64 array.

## Parity

| Output | Class | Threshold | Measured |
|---|---|---|---|
| `compartment.comp` | deterministic-bounded | `1e-6` | `2.08e-17` |
| `compartment.strength` | deterministic-bounded | `1e-6` | `1.71e-13` |

See [docs/PERFORMANCE.md](../docs/PERFORMANCE.md).

## Patch-schicluster integration

`patch_schicluster()` rebinds
`schicluster.compartment.call_compartment.single_chrom_compartment` →
`schicluster_rs.single_chrom_compartment` at module level, so the
upstream per-cell `ProcessPoolExecutor` inside `multiple_cell_compartment`
transparently uses Rust:

```python
import schicluster_rs
schicluster_rs.patch_schicluster()

from schicluster.compartment import multiple_cell_compartment
multiple_cell_compartment(
    cell_table_path='cell_table.tsv',
    output_prefix='dataset/sample.impute',
    cpg_profile_path='cpg_ratio_100k.hdf',
    cpu=96,
)
```

## Prerequisite: CpG ratio

Upstream's `compartment` subcommand requires a per-bin CpG ratio HDF
produced by `bedtools nuc`. The ratio table is computed by
`schicluster.compartment.call_compartment.get_cpg_profile`, exposed as
`hicluster cpg-ratio` (and `schicluster-rs cpg-ratio` — pure I/O, no
Rust speedup, included so a single binary covers the workflow).

```bash
schicluster-rs cpg-ratio \
    --fasta_path mm10.fa \
    --hdf_output_path cpg_ratio_100k.hdf \
    --chrom_size_path chrom_sizes.txt \
    --resolution 100000
```
