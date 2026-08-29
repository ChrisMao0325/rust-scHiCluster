# `contact-distance` — distance decay and per-chromosome sparsity

For each cell, computes two things from its contact file: a **log-spaced
histogram of cis contact distances** (`|pos2 - pos1|`), and a **per-chromosome
sparsity** count — the number of distinct off-diagonal bin pairs.

Upstream reads the whole gzipped contact file into a pandas DataFrame in order
to use four of its columns. This package streams the file instead, in constant
memory, which is where the speedup comes from.

## Direct API

```python
import numpy as np
import pandas as pd
import schicluster_rs

chrom_sizes = pd.read_csv('chrom.sizes', sep='\t', header=None, index_col=0)

# Upstream's log-spaced edges — computed with numpy and passed straight through,
# so Rust never recomputes exp2 and there is no ULP drift.
nbins = np.floor(np.log2(chrom_sizes[1].values.max() / 2500) / 0.125)
bins = 2500 * np.exp2(0.125 * np.arange(nbins + 1))

sparsity, decay = schicluster_rs.compute_decay(
    cell_name='cell_A',
    contact_path='cell_A.contact.rmbkl.tsv.gz',
    bins=bins,
    chrom_sizes=chrom_sizes,
    resolution=10_000,
    chrom1=1, pos1=2, chrom2=5, pos2=6,
)
```

Returns the same `[sparsity_frame, decay_frame]` pair upstream's orchestrator
concatenates, so `pd.concat` and `to_hdf` downstream are untouched.

## Drop-in monkey-patch

```python
import schicluster_rs
schicluster_rs.patch_schicluster()

from schicluster.cool.contact_distance import contact_distance
contact_distance(contact_table=..., chrom_size_path=..., resolution=10_000,
                 output_prefix='dataset/sample', chrom1=1, chrom2=5,
                 pos1=2, pos2=6, cpu=20)
```

## Command line

```bash
schicluster-rs contact-distance \
    --contact_table contact_table_rmbkl.tsv \
    --chrom_size_path chrom.sizes \
    --output_prefix dataset/sample \
    --resolution 10000 \
    --chr1 1 --pos1 2 --chr2 3 --pos2 4 \
    --cpu 20
```

Writes `<output_prefix>_chromsparsity.hdf5` and `<output_prefix>_decay.hdf5`.

## Behaviour notes

**The longest cis contacts are dropped.** The top histogram edge is
`2500 · 2^(0.125·n)` for the largest chromosome, which for hg38 works out to
about **231.7 Mb** — while chr1 is 249.0 Mb. `np.histogram` discards values
outside `[edges[0], edges[-1]]`, so upstream loses contacts longer than that
top edge, and so does this port. Contacts shorter than 2500 bp are dropped for
the same reason. The final bin is right-**closed**; every other bin is
right-open.

**Sparsity counts distinct bin pairs, not contacts.** Upstream groups by
`(chrom, pos1 // resolution, pos2 // resolution)`, drops same-bin pairs, then
counts per chromosome. Twenty-five reads supporting one bin pair contribute 1,
not 25. Pairs are **ordered**: `(5, 3)` and `(3, 5)` are distinct. Chromosomes
with no surviving off-diagonal pair are absent from the output rather than
present with a zero.

**Both outputs are exact.** They are integer counts, so addition is
order-independent and the parity gate is `deterministic-strict` at `0.0`.

**The speedup is modest, around 4×.** Decompressing the gzipped contact file
is the floor and cannot be made faster; the gain comes from not building a
DataFrame for four columns. Don't expect the larger speedups seen elsewhere in
the package.

**Parse errors are fatal, by design.** Unlike `filter-contact`, upstream's
`compute_decay` calls `read_csv` **without** `comment='#'`, so a `#` line is a
hard failure there too. Rust raises a `ValueError` naming the file and line
number rather than silently skipping it.
