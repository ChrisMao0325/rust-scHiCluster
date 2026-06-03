# DISCOVERY — rebuildpy Phase 0.5

**Target:** scHiCluster (Python upstream at `/large_storage/zhoulab/shengmao/scHiCluster`, editable in `schicluster` env, version `1.3.5.dev22+gd566046`).

**Existing port status:** **Partial.** `rust-scHiCluster` (this repo) already ports `random_walk_cpu` and `impute_chromosome` (RWR + Gaussian + SQRTVC). Loop / domain / compartment / embedding / merge are NOT yet ported. **This work extends the existing repo; no duplicate `rs-` repo is created.**

**Naming deviation:** the existing PyPI name is `schicluster-rs` / import `schicluster_rs`, not the rebuildpy convention `rs-schicluster` / `rs_schicluster`. Kept for backwards compatibility. To be added to `engine/discover_rust_deps.py::ALIAS_MAP` in a downstream PR upstream of rebuildpy.

**Dependency mapping (for the functions ported in this phase set):**

| Upstream Python dep | Rust replacement |
|---|---|
| `numpy` | `ndarray 0.16` (already in `Cargo.toml`) |
| `scipy.sparse` (CSR/COO) | `sprs 0.11` (already in `Cargo.toml`) |
| `scipy.ndimage.convolve` (mirror) | NEW `rust/src/conv.rs::convolve2d_mirror` |
| `scipy.stats.zscore`, `np.percentile` | inline (Phase 1+) |
| R `wilcox.test(exact=F)` | inline normal-approx with continuity + tie correction (Phase 2) |
| `cooler` / `h5py` (I/O) | **stays Python** |
| `statsmodels.stats.multitest` (FDR) | **stays Python** |
| `sklearn.decomposition.TruncatedSVD` | **stays Python** |
| `rpy2` + R `TopDom.R` | NEW `rust/src/topdom.rs` (Phase 2) |

**Decision:** proceed to Phase 0 scaffold + shared 2-D convolution; subsequent phases per spec §9.
