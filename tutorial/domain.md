# Domain (Phase 2)

Per-chrom insulation score + a **native Rust TopDom** that drops the
rpy2 / R round-trip. Ported from `schicluster.domain.call_domain` and
`schicluster.domain.TopDom.R`.

## Public API

```python
from schicluster_rs import insulation_score_chrom
from schicluster_rs.domain import _topdom_chrom_to_df  # or use run_top_dom (CSC triplets in)
```

## `insulation_score_chrom(matrix, window_size=10, save_count=False)`

Sliding-window submatrix-sum insulation score per bin. Drop-in for
`schicluster.domain.call_domain.single_chrom_calculate_insulation_score`.

```python
from schicluster_rs import insulation_score_chrom

# matrix: scipy sparse or dense numpy; symmetric, diagonal-included is fine
score = insulation_score_chrom(matrix, window_size=10, save_count=False)
# shape: (n_bins,); score[0] = 1.0 sentinel
```

| Arg | Default | Notes |
|---|---|---|
| `matrix` | — | scipy sparse (`.toarray()` is called) or dense `np.ndarray` |
| `window_size` | `10` | Bins on each side of a row |
| `save_count` | `False` | If `True`, returns `(n_bins, 2)` with `[inter, intra]` per row |

f64 accumulation, cast to f32 on emit. Matches scipy's `.sum()`
semantics.

## TopDom

`schicluster_rs` ships a native TopDom port that mirrors the upstream R
`TopDom.R` one-to-one — diamond signal, gap regions, change-point /
local-extreme detection, R-compatible Wilcoxon rank-sum p-values (normal
approximation with continuity + tie correction), bin → domain BED
conversion.

**Easiest usage** is via `patch_schicluster()` — the upstream
`schicluster.domain.call_domain.call_domain_and_insulation` orchestrator
keeps working unchanged (it just calls Rust internally):

```python
import schicluster_rs
schicluster_rs.patch_schicluster()

from schicluster.domain.call_domain import call_domain_and_insulation
call_domain_and_insulation(
    cell_url='cell.cool',
    output_prefix='cell.domain',
    resolution=25_000,
    window_size=10,
)
```

**Lower-level direct API** for the per-chrom TopDom:

```python
import pandas as pd
from schicluster_rs.domain import _topdom_chrom_to_df

# matrix: square float32 symmetric (or sparse with .toarray())
# bins: DataFrame with columns ['chr', 'from.coord', 'to.coord']
n = matrix.shape[0]
bins = pd.DataFrame({
    'chr': ['chr1'] * n,
    'from.coord': [i * 25_000 for i in range(n)],
    'to.coord': [(i + 1) * 25_000 for i in range(n)],
})
bed_df = _topdom_chrom_to_df(matrix, bins, window_size=10, stat_filter=True)
# columns: ['chrom', 'chromStart', 'chromEnd', 'name']
# name in {'gap', 'domain', 'boundary'}
```

| Arg | Default | Notes |
|---|---|---|
| `matrix` | — | Square symmetric; sparse or dense |
| `bins` | — | DataFrame with `chr`, `from.coord`, `to.coord` columns (upstream's convention) |
| `window_size` | — | TopDom window in bins |
| `stat_filter` | `True` | Run the Wilcoxon p-value statistical filter (step 3 in TopDom.R) |

## What the patch does

`patch_schicluster()` does two things for domain:

1. Module-level rebind of
   `schicluster.domain.call_domain.single_chrom_calculate_insulation_score`
   → `schicluster_rs.insulation_score_chrom`.
2. Replaces the rpy2 `r` module-level global inside
   `schicluster.domain.call_domain` with a stub whose `RunTopDom`
   delegates to Rust — this is how the local closure
   `def run_top_dom(...)` inside `call_domain_and_insulation` routes
   through Rust without us editing upstream.

You can still call into the rpy2 / R path if you want by **not** calling
`patch_schicluster()` and letting upstream see the real rpy2 `r`.

## Parity

| Output | Class | Threshold | Measured |
|---|---|---|---|
| `insulation.score` | deterministic-bounded | `1e-6` | `5.96e-08` |
| `topdom.bed.interval_jaccard` | ranked | `≥ 0.95` | `1.0` |
| `topdom.bed.bin_label_agreement` | classification | `≥ 0.98` | `1.0` |

Domain intervals match the rpy2 → TopDom.R reference exactly on the
test fixture. See [docs/PERFORMANCE.md](../docs/PERFORMANCE.md).

## Notes

- Wilcoxon rank-sum uses normal approximation with continuity (`+ 0.5`
  for `alternative="less"`) and the standard tie correction, exactly
  matching R's `wilcox.test(exact=FALSE)`.
- The normal CDF uses an inline Abramowitz 7.1.26 erf approximation
  (max error ~ `1.5e-7`, plenty for the `p < 0.05` cutoff). No new
  crate dependencies.
- `_topdom_chrom_to_df` is the recommended Python entry point; the raw
  Rust kernel `py_topdom_chrom` returns `(from_bin_id, to_bin_id, tag)`
  tuples and a small Python layer maps tag IDs to BED row names.
