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

**Direct — impute path**:

```python
from schicluster_rs import random_walk_cpu, impute_chromosome

# Just the iterative RWR step (CSR → CSR):
Q = random_walk_cpu(P, rp=0.5, tol=0.01)

# Full inner pipeline (writes HDF5 like upstream):
impute_chromosome(scool_url='cell.cool', chrom='chr1',
                  resolution=25_000, output_path='chr1.hdf',
                  rp=0.5, tol=0.01, pad=1, std=1.0,
                  output_dist=10_050_000)
```

**Direct — loop path** (Phase 1, available since 0.2.0):

```python
import schicluster_rs

# Per-cell background normalisation (writes <prefix>.E.npz + .T.npz):
schicluster_rs.loop_bkg_chrom(
    cell_url='cell.cool', chrom='chr1', resolution=10_000,
    output_prefix='cell.chr1',
    dist=10_050_000, cap=5, pad=5, gap=2, min_cutoff=1e-6,
)

# Cell-to-group sparse accumulation (reads *.E.npz in output_dir,
# writes <prefix>.E.hdf + <prefix>.E2.hdf):
schicluster_rs.merge_cells_for_single_chromosome(
    output_dir='per_cell/', output_prefix='group.chr1', merge_type='E',
)

# Loop background convolutions at given pixels (returns 4 arrays):
bl, donut, h, v = schicluster_rs.loop_background(E, pad=5, gap=2, loop=(xs, ys))

# Graph + heap peak merging:
summit_df = schicluster_rs.find_summit(loop_df, res=10_000, dist_thres=2)
```

### Multi-process tuning

`schicluster`'s default workflow is `ProcessPoolExecutor(max_workers=N)`.
Each worker forks the rayon thread pool — without explicit sizing, every
worker spawns `num_cpus` threads, leading to `N × num_cpus` contending
threads on a single node.

Set the per-worker rayon thread count via `set_num_threads(n)` in the
worker initialiser. Recommended sizing: `n = num_cpus // num_workers`.
Example: 16-core node with 8 workers → `set_num_threads(2)`.

```python
from concurrent.futures import ProcessPoolExecutor
import schicluster_rs

def worker_init():
    schicluster_rs.set_num_threads(2)
    schicluster_rs.patch_schicluster()

with ProcessPoolExecutor(max_workers=8, initializer=worker_init) as ex:
    list(ex.map(impute_one_cell, cells))
```

### Snakemake (loop calling)

A Rust-backed snakemake template ships at
`python/schicluster_rs/loop/snakemake_template_loop.txt`. It prepends
`patch_schicluster()` so every loop rule that snakemake fans out
transparently uses the Rust kernels. Drop it into your workflow as a
replacement for upstream's `loop/snakemake_template_loop.txt`.

## Citation

If you use this package, please cite the original scHiCluster paper:

> Zhou, J., Ma, J., Chen, Y., Cheng, C., Bao, B., Peng, J., Sejnowski,
> T. J., Dixon, J. R. & Ecker, J. R. (2019). *Robust single-cell Hi-C
> clustering by convolution- and random-walk-based imputation.* PNAS,
> 116(28):14011-14018.

## License

MIT.
