# Impute (Phase 0)

Single-cell Hi-C contact-matrix imputation via convolution + random walk
with restart, ported from
`schicluster.impute.impute_chromosome.impute_chromosome`.

## Public API

```python
from schicluster_rs import random_walk_cpu, impute_chromosome
```

## `random_walk_cpu(p, rp=0.5, tol=0.01, n_iter=30)`

The iterative random-walk-with-restart fixed point — the inner loop of
imputation. Drop-in for
`schicluster.impute.impute_chromosome.random_walk_cpu`.

```python
from schicluster_rs import random_walk_cpu

# p: scipy CSR (or dense) row-stochastic n × n matrix
Q = random_walk_cpu(p, rp=0.5, tol=0.01)
```

| Arg | Default | Notes |
|---|---|---|
| `p` | — | Row-stochastic n × n; scipy `csr_matrix` or dense ndarray |
| `rp` | `0.5` | Restart probability. `1.0` returns `p` unchanged |
| `tol` | `0.01` | Frobenius-norm convergence tolerance |
| `n_iter` | `30` | Max iterations |

Returns: `scipy.sparse.csr_matrix` of the converged `Q`.

## `impute_chromosome(scool_url, chrom, resolution, output_path, ...)`

Full per-chrom inner pipeline: cooler read → Gaussian convolution →
row-normalize → RWR → symmetrize → SQRTVC normalize → triangle filter →
HDF5 write. Drop-in for
`schicluster.impute.impute_chromosome.impute_chromosome`.

```python
from schicluster_rs import impute_chromosome

impute_chromosome(
    scool_url='cell.cool',
    chrom='chr1',
    resolution=25_000,
    output_path='chr1.hdf',
    rp=0.5,
    tol=0.01,
    pad=1,
    std=1.0,
    output_dist=10_050_000,
)
```

| Arg | Default | Notes |
|---|---|---|
| `scool_url` | — | Cooler URL (single .cool or scool::/cells/<id>) |
| `chrom` | — | Chromosome name |
| `resolution` | — | Bin size in bp |
| `output_path` | — | HDF5 output path |
| `logscale` | `False` | `np.log2(v + 1)` the raw counts first |
| `pad` | `1` | Gaussian kernel half-width |
| `std` | `1.0` | Gaussian std-dev |
| `rp` | `0.5` | RWR restart probability |
| `tol` | `0.01` | RWR Frobenius tolerance |
| `output_dist` | `5e11` | Max bp distance to write |
| `min_cutoff` | `0` | Drop output values with `|v| ≤ cutoff` |
| `band_factor` | `0` | `> 0` enables banded-Q approximation (≪ 1 % deviation, ~4× faster) |

Reads cooler in Python, runs steps 2–9 in Rust, writes HDF5 in Python.

## Multi-process tuning

```python
from concurrent.futures import ProcessPoolExecutor
import schicluster_rs

def worker_init():
    schicluster_rs.set_num_threads(2)   # n // num_workers
    # schicluster_rs.patch_schicluster()  # only if you also use the monkey-patch path

with ProcessPoolExecutor(max_workers=8, initializer=worker_init) as ex:
    list(ex.map(impute_one_cell, cells))
```

Without `set_num_threads`, each worker forks the rayon pool with
`num_cpus` threads → `N × num_cpus` contending threads on one node.
Sizing `n = num_cpus // num_workers` keeps the rayon footprint to one
physical core per worker.

## Parity

`deterministic-bounded`, threshold `1e-6`, measured at `1.79e-7` on the
test fixture (Phase 0). See [docs/PERFORMANCE.md](../docs/PERFORMANCE.md)
for real-data benchmarks (≈ 9.6× speedup on chr1 at 25 kb resolution).
