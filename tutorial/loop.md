# Loop calling (Phase 1)

Four per-chrom kernels ported from `schicluster.loop`:

- background normalisation (`loop_bkg.calculate_chrom_background_normalization`)
- cell-to-group sparse accumulator (`merge_cell_to_group.merge_cells_for_single_chromosome`)
- four-kernel background scan (`loop_calling.loop_background`)
- graph + max-heap peak merging (`loop_calling.find_summit`)

The paired t-test and `statsmodels.stats.multitest` BH-FDR stay in
Python. Cooler / HDF5 / .npz I/O stay in Python.

## Public API

```python
import schicluster_rs

# Per-cell background normalisation; writes <prefix>.E.npz + .T.npz
schicluster_rs.loop_bkg_chrom(
    cell_url='cell.cool',
    chrom='chr1',
    resolution=10_000,
    output_prefix='cell.chr1',
    dist=10_050_000,
    cap=5,
    pad=5,
    gap=2,
    min_cutoff=1e-6,
    # log_e=False, shuffle=False
)
```

| Arg | Default | Notes |
|---|---|---|
| `cell_url` | — | Cooler URL for the per-cell matrix |
| `chrom` | — | Chromosome name |
| `resolution` | — | Bin size in bp |
| `output_prefix` | — | Outputs `<prefix>.E.npz` + `<prefix>.T.npz` |
| `dist` | `10_050_000` | Distance cutoff in bp (≈ window in bins × resolution) |
| `cap` | `5` | z-score clip threshold |
| `pad` | `5` | Donut kernel half-width |
| `gap` | `2` | Donut inner gap |
| `min_cutoff` | `1e-6` | Drop sparse entries below `\|v\| ≤ cutoff` |
| `log_e` | `False` | log-scale path |
| `shuffle` | `False` | RNG path; falls back to upstream Python (out of parity scope) |

```python
# Cell-to-group sparse accumulation: sums + sum-of-squares
# Reads *.E.npz under output_dir, writes <prefix>.E.hdf + <prefix>.E2.hdf
schicluster_rs.merge_cells_for_single_chromosome(
    output_dir='per_cell/',
    output_prefix='group.chr1',
    merge_type='E',     # or 'T'
)
```

```python
# Loop-pixel background scan with four shaped kernels
# E is the group / per-cell imputed contact matrix (dense numpy)
# loop = (xs, ys), pixel coordinates
bl, donut, h, v = schicluster_rs.loop_background(
    E,
    pad=5,
    gap=2,
    loop=(xs, ys),
)
```

`bl`, `donut`, `h`, `v` are 1-D arrays the same length as `xs`/`ys`
giving the four background-kernel convolutions evaluated at each loop
pixel.

```python
# Graph + max-heap peak merging on a loop dataframe
# loop_df must have x1, y1, E columns
summit_df = schicluster_rs.find_summit(
    loop_df,
    res=10_000,
    dist_thres=2,   # in bins, not bp
)
```

Returns a sub-DataFrame of `loop_df` containing peak rows, with an
added `size` column = cluster size.

## Snakemake template

`python/schicluster_rs/loop/snakemake_template_loop.txt` prepends a
`patch_schicluster()` call so every loop rule snakemake fans out
routes through the Rust kernels. Use as a drop-in replacement for
upstream's `schicluster/loop/snakemake_template_loop.txt`.

## Parity

Five outputs, all green:

| Output | Class | Threshold | Measured |
|---|---|---|---|
| `loop_bkg.E` / `loop_bkg.T` | deterministic-bounded | `1e-6` | ≤ 9.5e-7 |
| `merge.e_sum` / `merge.e2_sum` | deterministic-bounded | `1e-6` | ≤ 1.2e-7 |
| `scan_kernels.{bl,donut,h,v}` | deterministic-bounded | `1e-6` | ≤ 2.4e-7 |
| `find_summit.idx` | ranked (Jaccard) | `≥ 0.99` | `1.0` |
| `find_summit.sizes` | classification | `1.0` | `1.0` |

See [docs/PERFORMANCE.md](../docs/PERFORMANCE.md).

## Notes

- `loop_bkg_chrom`'s `shuffle=True` branch is RNG-driven and
  intentionally out of the parity gate; that path falls back to upstream
  Python.
- The merge step accumulates in f64 (cast back to f32 on emit) and
  iterates a deterministic BTreeMap for row-major output order — no
  cross-cell reduction-order drift.
