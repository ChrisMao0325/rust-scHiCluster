# rust-scHiCluster

Rust re-implementation of the inner numerical pipeline of
[scHiCluster](https://github.com/zhoujt1994/scHiCluster) (Zhou *et al.*
2019, PNAS) — single-cell Hi-C contact-matrix imputation. **~10× faster
on long chromosomes** at single-cell impute, **bit-equivalent** within
float-32 epsilon.

This is **a separate package** that re-implements the algorithm in Rust;
it is not a fork of the upstream Python tree. The original Python
implementation continues to be available as `pip install schicluster`
(and is used here only as the parity baseline in the test suite).

## What it computes

`schicluster_rs.impute_chromosome(...)` runs the same end-to-end
pipeline as `schicluster.impute.impute_chromosome.impute_chromosome`:

1. Read raw single-cell contact matrix from a `.cool` file.
2. Drop the diagonal.
3. 2-D Gaussian convolution (mirror padding) — replaces
   `scipy.ndimage.gaussian_filter`.
4. Drop the diagonal again.
5. Row-normalise → P.
6. Random-walk-with-restart fixed point: `Q = (1−rp)·P·Q + rp·P` for
   up to 30 iterations or until ‖Q_t − Q_{t-1}‖_F < tol.
7. Symmetrise: `E = Q + Qᵀ`.
8. SQRTVC normalise: `E ← D^{-1/2} · E · D^{-1/2}` where `D = diag(Eᵀ𝟙)`.
9. Filter the upper triangle to entries with `j − i ≤ output_dist_bins`.
10. Write the result to an HDF5 file (cooler-compatible).

Steps 2–9 run inside Rust; only the cooler read in step 1 and the HDF5
write in step 10 cross the Python boundary.

## Speed

Real Chang 2024 LC462 mouse cortex Droplet Hi-C, 25 kb resolution:

| step                         | scipy upstream | rust      | speedup |
|------------------------------|----------------|-----------|---------|
| chr1   (n = 7820 bins)       | 30.5 s         | **3.2 s** | **9.6×** |
| chr19  (n = 2461 bins)       |  0.4 s         | **0.27 s**| 1.5× |
| 20 chrs end-to-end per cell  | 87 s           | **33 s**  | **2.7×** |

Multi-process parallelism (8 workers × 2 rayon threads = 16 cores total):
8 chr1 in parallel from 29 s → 9.4 s = an additional **3.1×** beyond the
per-cell speedup, by avoiding rayon thread oversubscription.

## Accuracy

Bit-equivalent to upstream within float-32 ε. On real Chang chr1
(n = 7820, ~1.7 M output non-zeros):

* `max |E_rust − E_scipy|` = `8.94 × 10⁻⁸`
* Pearson correlation = `1.000000`
* nnz match exactly.

`tests/test_parity.py` runs `random_walk_cpu` over `(n, rp)` ∈
{50, 200, 500} × {0.05, 0.5, 0.9} and asserts max-relative-error < 1e-4
against `scipy`'s reference implementation. **All 11 tests pass.**

## Install

Requires **Rust ≥ 1.78** and **maturin ≥ 1.4**:

```bash
git clone https://github.com/omicverse/rust-scHiCluster
cd rust-scHiCluster
maturin develop --release   # build + install into the active venv
```

PyPI release coming soon (`pip install schicluster-rs`).

## Use

**Drop-in monkey-patch (recommended)** — no code changes anywhere:

```python
import schicluster_rs
schicluster_rs.set_num_threads(2)        # 8 workers × 2 = 16 cores
schicluster_rs.patch_schicluster()

# every downstream call to scHiCluster's impute_chromosome now uses Rust:
from schicluster.impute.impute_chromosome import impute_chromosome
impute_chromosome(scool_url=..., chrom='chr1', resolution=25_000,
                  output_path=..., rp=0.5, tol=0.01,
                  pad=1, std=1.0, output_dist=10_050_000)
```

**Direct**:

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

## Layout

```
rust-scHiCluster/
├── pyproject.toml          maturin build config
├── README.md               this file
├── LICENSE                 MIT
├── rust/
│   ├── Cargo.toml
│   └── src/lib.rs          all algorithms (~500 LoC)
├── python/
│   └── schicluster_rs/__init__.py     thin Python wrapper + monkey-patch
└── tests/
    └── test_parity.py      parity vs scipy on random sparse matrices
```

## Algorithm notes

The hot loop is the iterative random-walk-with-restart, which is
implemented as a **Sparse-times-Dense matrix multiplication (SpMM)**
with rayon row-wise parallelism:

* `P` (sparse, ~7 nnz per row after Gaussian smoothing) stays as CSR.
* `Q` (the iterate) is stored dense, since RWR diffuses it to ≥ 30 %
  density after 1–2 iterations anyway.
* Each iteration: `Q' = (1−rp) · (P · Q) + rp · P`. The `P · Q` matmul
  is computed row-wise; each output row is independent, so rayon
  splits row-chunks across cores. Within each row, the inner AXPY
  (accumulate `P[i,k] · Q[k, :]` for sparse k) vectorises cleanly.

Other steps (Gaussian convolution, SQRTVC normalize, triangle filter)
are similarly multi-threaded over rows or chunks.

For users who can tolerate ≪1% deviation from the strict scipy result,
a `band_factor` parameter is available that runs the RWR with a banded
`Q` (only entries with `|j − i| ≤ band_factor × output_dist_bins`),
giving an additional ~4× speedup. Default is 0 (off, strict).

## Citation

If you use this package, please cite the original scHiCluster paper:

> Zhou, J., Ma, J., Chen, Y., Cheng, C., Bao, B., Peng, J., Sejnowski,
> T. J., Dixon, J. R. & Ecker, J. R. (2019). *Robust single-cell Hi-C
> clustering by convolution- and random-walk-based imputation.* PNAS,
> 116(28):14011-14018.

## License

MIT.
