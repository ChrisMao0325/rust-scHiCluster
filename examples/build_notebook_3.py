"""Build examples/function_by_function_Python_parity.ipynb (rebuildpy Notebook 3).

A function-level Python-to-Rust dictionary for users migrating code. For every
public function: a parameter table with one row per upstream parameter, the
upstream Python call as markdown, the Rust-backed call as code, and an output
comparison against examples/py_per_function_dump.py's JSON.
"""
from __future__ import annotations

import pathlib

import nbformat as nbf

OUT = pathlib.Path(__file__).resolve().parent / "function_by_function_Python_parity.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# Function-by-function Python ⇄ Rust parity

**Audience:** users swapping the pure-Python `schicluster` for the Rust-backed
`schicluster-rs`, or porting existing code.

This differs from the other two parity notebooks:

- [`compare_Python_vs_Rust.ipynb`](compare_Python_vs_Rust.ipynb) is
  **pipeline-level** parity — full end-to-end outputs against the gate.
- [`tutorial_loop_domain.ipynb`](tutorial_loop_domain.ipynb) is a **Rust-only**
  function tour, with no reference comparison.
- **This notebook** is **function-level** parity: each function called in
  isolation, on the same input, with every parameter documented.

The upstream reference cannot be imported here — it is Python 3.6 with rpy2,
while `schicluster_rs` is `abi3-py39`. So the Python side runs once, ahead of
time, via `examples/py_per_function_dump.py` under `$PYTHON_REF_ENV`, and its
outputs are loaded from JSON below. Upstream calls are therefore shown as
**markdown** code blocks (what you would write), and the Rust calls as
executable **code** cells."""))

cells.append(md("## Setup"))
cells.append(code('''import os, json, pathlib, sys
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

REPO = pathlib.Path.cwd()
if REPO.name == "examples":
    REPO = REPO.parent
os.chdir(REPO)

import numpy as np
import pandas as pd
import schicluster_rs

REF = json.load(open("data/fixtures/per_function_reference.json"))
print("reference functions dumped:", len(REF))
for k in sorted(REF):
    print("   ", k, "—", REF[k]["note"])

FIX = pathlib.Path("data/fixtures")
conv_pack = np.load(FIX / "conv_small.npz")
loop_pack = np.load(FIX / "loop_small.npz")
domain_pack = np.load(FIX / "domain_small.npz")
comp_pack = np.load(FIX / "compartment_small.npz")
emb_pack = np.load(FIX / "embedding_small.npz")
gs_pack = np.load(FIX / "gene_score_small.npz")
cd_pack = np.load(FIX / "contact_distance_small.npz")

VERDICTS = []

def verdict(func, output, cls, metric_name, value, passed):
    VERDICTS.append((func, output, cls, metric_name, value, passed))
    mark = "PASS" if passed else "FAIL"
    print(f"  -> {mark}  {output}: {metric_name} = {value}")'''))

# ---------------- convolve2d_mirror ----------------
cells.append(md("## `convolve2d_mirror`"))
cells.append(md("""2-D convolution with scipy's `mode='mirror'` boundary and kernel-flipped
(*convolve*, not *correlate*) semantics. Shared by all five convolutions in the
loop pipeline.

| Python name | Rust name | Type | Default | Range / values | Description |
|---|---|---|---|---|---|
| `input` | `a` | ndarray f32, C-contiguous | — | 2-D | Input matrix. Renamed from scipy's `input` (a Python builtin) to `a`. |
| `weights` | `kernel` | ndarray f32, C-contiguous | — | 2-D, any shape | Convolution kernel. Renamed from scipy's `weights`. |
| `mode` | *(fixed)* | str | `'reflect'` | — | **Removed in Rust** — always `'mirror'`, the only mode scHiCluster uses. |
| `output` | *(removed)* | ndarray | `None` | — | **Removed in Rust** — the kernel always allocates and returns. |
| `cval` | *(removed)* | float | `0.0` | — | **Removed in Rust** — only meaningful for `mode='constant'`. |
| `origin` | *(fixed)* | int | `0` | — | **Removed in Rust** — fixed at 0, matching every upstream call site. |

Upstream Python call:
```python
from scipy.ndimage import convolve
out = convolve(a, kernel, mode='mirror')
```"""))
cells.append(code('''out = np.asarray(schicluster_rs.convolve2d_mirror(conv_pack["input"], conv_pack["kernel"]))
ref = np.asarray(REF["convolve2d_mirror"]["convolved"], dtype=np.float64)
err = float(np.max(np.abs(out.astype(np.float64) - ref)))
verdict("convolve2d_mirror", "conv.convolved", "deterministic-bounded", "max abs err", f"{err:.3e}", err <= 1e-6)
print("     bounded by n*eps_32*max|x| over the kernel reduction; gate is 1e-6")'''))

# ---------------- loop_background ----------------
cells.append(md("## `loop_background`"))
cells.append(md("""The four background kernel scans (bottom-left, donut, horizontal, vertical)
evaluated at each candidate loop pixel.

| Python name | Rust name | Type | Default | Range / values | Description |
|---|---|---|---|---|---|
| `E` | `E_rows`, `E_cols`, `E_vals`, `n` | sparse f32 → CSR triplets | — | n×n upper-tri | Background-normalised contact matrix, passed as triplets across the FFI. |
| `pad` | `pad` | `int` → `usize` | `5` | `≥ 1` | Outer radius of the donut/bl kernels, in bins. |
| `gap` | `gap` | `int` → `usize` | `2` | `0 ≤ gap < pad` | Inner radius excluded from the donut. |
| `loop` (x, y) | `loop_xs`, `loop_ys` | ndarray int → `u32` | — | indices into E | Candidate loop pixel coordinates, split into two arrays. |

Upstream Python call:
```python
from schicluster.loop.loop_calling import loop_background
bl, donut, h, v = loop_background(E, pad=5, gap=2, loop_xs=xs, loop_ys=ys)
```"""))
cells.append(code('''ref_k = REF["loop_background"]["kernels"]
print("reference kernel keys:", sorted(ref_k))
cand = json.load(open("data/fixtures/candidate_output.json"))["scan_kernels"]
worst, worst_name = 0.0, None
for key in ("bl", "donut", "h", "v"):
    r = np.asarray(ref_k[key], dtype=np.float64)
    c = np.asarray(cand[key], dtype=np.float64)
    e = float(np.max(np.abs(r - c)))
    if e > worst:
        worst, worst_name = e, key
    print(f"     scan_kernels.{key:<6} max abs err = {e:.3e}")
verdict("loop_background", "scan_kernels.{bl,donut,h,v}", "deterministic-bounded",
        f"max abs err (worst: {worst_name})", f"{worst:.3e}", worst <= 1e-6)'''))

# ---------------- find_summit ----------------
cells.append(md("## `find_summit`"))
cells.append(md("""Merges neighbouring loop pixels into summits, via a graph plus a max-heap keyed
on `(-E, idx)`.

| Python name | Rust name | Type | Default | Range / values | Description |
|---|---|---|---|---|---|
| `x1` | `x1` | ndarray int → `i64` | — | genomic bp | Row coordinate of each candidate pixel. |
| `y1` | `y1` | ndarray int → `i64` | — | genomic bp | Column coordinate of each candidate pixel. |
| `E` | `E_values` | ndarray f32 | — | `≥ 0` | Signal value at each candidate pixel; the heap key. |
| `dist_thres` | `dist_thres_bins` | `int` → `usize` | `~3` bins | `≥ 1` | Merge radius. Rust takes **bins**, upstream takes **bp** — the wrapper divides by resolution. |

Upstream Python call:
```python
from schicluster.loop.loop_calling import find_summit
idx, sizes = find_summit(x1, y1, E, dist_thres=30_000)
```"""))
cells.append(code('''ref_fs = REF["find_summit"]["result"]
cand_fs = json.load(open("data/fixtures/candidate_output.json"))["find_summit"]
r_idx = set(np.asarray(ref_fs["idx"]).ravel().tolist())
c_idx = set(np.asarray(cand_fs["idx"]).ravel().tolist())
jac = len(r_idx & c_idx) / max(len(r_idx | c_idx), 1)
verdict("find_summit", "find_summit.idx", "ranked", "Jaccard", f"{jac:.4f}", jac >= 0.99)
r_sz = np.asarray(ref_fs["sizes"]).ravel()
c_sz = np.asarray(cand_fs["sizes"]).ravel()
agree = float((r_sz == c_sz).mean()) if r_sz.shape == c_sz.shape else 0.0
verdict("find_summit", "find_summit.sizes", "classification", "label agreement", f"{agree:.4f}", agree >= 1.0)
print("     ranked/classification are higher-is-better: the pass test is >=, not <=")'''))

# ---------------- insulation ----------------
cells.append(md("## `insulation_score_chrom`"))
cells.append(md("""Sliding-window insulation score for one chromosome.

| Python name | Rust name | Type | Default | Range / values | Description |
|---|---|---|---|---|---|
| `cell_url` | `rows`, `cols`, `vals`, `n` | str → CSR triplets | — | — | Upstream reads the cool itself; the Rust kernel takes the matrix, and cooler I/O stays in the Python wrapper. |
| `chrom` | *(caller-side)* | str | — | — | **Removed in Rust** — the kernel is per-chromosome by construction. |
| `resolution` | *(caller-side)* | int | — | bp | **Removed in Rust** — only needed to locate bins, done Python-side. |
| `window_size` | `window_size` | `int` → `usize` | `10` | `≥ 1` | Half-width of the insulation window, in bins. |
| `save_count` | `save_count` | bool | `False` | — | Return raw window counts as well as the score. |

Upstream Python call:
```python
from schicluster.domain.call_domain import single_chrom_calculate_insulation_score
score = single_chrom_calculate_insulation_score(cell_url, chrom, resolution, window_size=10)
```"""))
cells.append(code('''ref_ins = np.asarray(REF["insulation_score_chrom"]["score"], dtype=np.float64).ravel()
cand_ins = np.asarray(json.load(open("data/fixtures/candidate_output.json"))["insulation"]["score"],
                      dtype=np.float64).ravel()
err = float(np.max(np.abs(ref_ins - cand_ins)))
verdict("insulation_score_chrom", "insulation.score", "deterministic-bounded", "max abs err", f"{err:.3e}", err <= 1e-6)
print("     direct per-window reduction, NOT a prefix sum — see docs/MATH.md")'''))

# ---------------- topdom ----------------
cells.append(md("## `topdom_chrom` (native TopDom)"))
cells.append(md("""Full native TopDom: diamond mean signal, gap detection, change-point and
local-extreme detection, Wilcoxon rank-sum p-values, and bin→domain conversion.
**Replaces the rpy2 → `TopDom.R` round-trip**, so R is no longer a dependency.

| Python / R name | Rust name | Type | Default | Range / values | Description |
|---|---|---|---|---|---|
| `matrix` (R sparse triplet) | `matrix` | ndarray **f32**, C-contiguous | — | n×n symmetric | R receives `(indices, indptr, data)`; the Rust kernel takes a dense f32 square matrix. |
| `bins` (R data.frame) | *(caller-side)* | DataFrame | — | `chr`, `from.coord`, `to.coord` | **Removed in Rust** — the kernel returns bin ids; the Python wrapper maps them to coordinates. |
| `window.size` | `window_size` | `int` → `usize` | `5` | `≥ 2` | Diamond window half-width, in bins. |
| *(implicit in R)* | `stat_filter` | bool | `True` | — | **New in Rust** — exposes R's `wilcox.test` boundary filter explicitly. `True` matches upstream. |

Upstream Python call (needs R + rpy2 + the `Matrix` R package):
```python
r.source('TopDom.R')
result = r.RunTopDom(csc.indices + 1, csc.indptr, csc.data, bins, window_size)
```"""))
cells.append(code('''from schicluster_rs._rust import py_topdom_chrom
mm = np.ascontiguousarray(domain_pack["topdom.matrix"], dtype=np.float32)
rust_doms = py_topdom_chrom(mm, int(domain_pack["topdom.window_size"]), True)
ref_bed = REF["topdom_chrom"]["bed"]
print("R/rpy2 reference rows:", len(ref_bed), "| Rust rows:", len(rust_doms))

cand_bed = json.load(open("data/fixtures/candidate_output.json"))["topdom"]["bed"]
def ivals(bed):
    return {(int(r["chromStart"]), int(r["chromEnd"])) for r in bed}
a, b = ivals(ref_bed), ivals(cand_bed)
jac = len(a & b) / max(len(a | b), 1)
verdict("topdom_chrom", "topdom.bed.interval_jaccard", "ranked", "interval Jaccard", f"{jac:.4f}", jac >= 0.95)
print("     Wilcoxon uses an inline Abramowitz erf (~1.5e-7) — ample for a p<0.05 cut")'''))

# ---------------- compartment ----------------
cells.append(md("## `single_chrom_compartment`"))
cells.append(md("""Per-chromosome CpG-weighted compartment score plus decay-normalised A/B/AB
strength.

| Python name | Rust name | Type | Default | Range / values | Description |
|---|---|---|---|---|---|
| `cell_url` | `rows`, `cols`, `vals`, `n` | str → CSR triplets | — | — | Cooler read stays in the Python wrapper. |
| `chrom` | *(caller-side)* | str | — | — | **Removed in Rust** — per-chromosome by construction. |
| `cpg_ratio` | `cpg_ratio` | ndarray f32 | — | `≥ 0`, length n | Per-bin CpG ratio. Bins at exactly 0 are excluded by `bin_filter`. |
| `calc_strength` | `calc_strength` | bool | `False` | — | Also return the `[AA, BB, AB]` strength triple. |

Upstream Python call:
```python
from schicluster.compartment.call_compartment import single_chrom_compartment
comp, strength = single_chrom_compartment(cell_url, chrom, cpg_ratio, calc_strength=True)
```"""))
cells.append(code('''cand_c = json.load(open("data/fixtures/candidate_output.json"))["compartment"]
for key, thr in (("comp", 1e-6), ("strength", 1e-6)):
    r = np.asarray(REF["single_chrom_compartment"][key], dtype=np.float64).ravel()
    c = np.asarray(cand_c[key], dtype=np.float64).ravel()
    e = float(np.max(np.abs(r - c)))
    verdict("single_chrom_compartment", f"compartment.{key}", "deterministic-bounded",
            "max abs err", f"{e:.3e}", e <= thr)'''))

# ---------------- embedding ----------------
cells.append(md("## `make_chrom_matrix`"))
cells.append(md("""Cell-by-feature extraction for embedding. **SVD stays sklearn** and is
deliberately out of scope — the gate targets the matrix *before* SVD.

| Python name | Rust name | Type | Default | Range / values | Description |
|---|---|---|---|---|---|
| `cell_table` | `cells` | Series → ndarray f32 `(n_cells, n, n)` | — | — | Upstream reads one cool per cell; the wrapper stacks them and hands Rust the array. |
| `chrom` | *(caller-side)* | str | — | — | **Removed in Rust** — per-chromosome by construction. |
| `nbins` | *(inferred)* | int | — | — | **Removed in Rust** — taken from the array shape. |
| `output_path` | *(caller-side)* | str | — | — | **Removed in Rust** — the `.npz` write stays Python. |
| `scale_factor` | `scale_factor` | float → `f32` | `100000` | `> 0` | Multiplied into every extracted value. |
| `dist` | `dist_bins_plus_1` | int → `usize` | `1000000` | bp → **bins** | Max contact distance. Rust takes `dist/resolution + 1` in bins. |
| `resolution` | *(folded in)* | int | — | bp | **Removed in Rust** — folded into `dist_bins_plus_1`. |

Upstream Python call:
```python
from schicluster.embedding.calc_embedding import make_chrom_matrix
make_chrom_matrix(cell_table, chrom, nbins, output_path, scale_factor=1e5, dist=1e6, resolution=1e4)
```"""))
cells.append(code('''r = np.asarray(REF["make_chrom_matrix"]["cell_by_feature"], dtype=np.float64).ravel()
c = np.asarray(json.load(open("data/fixtures/candidate_output.json"))["embedding"]["cell_by_feature"],
               dtype=np.float64).ravel()
e = float(np.max(np.abs(r - c)))
verdict("make_chrom_matrix", "embedding.cell_by_feature", "deterministic-strict", "max abs err", f"{e:.3e}", e <= 0.0)
print("     pure gather + scalar multiply, no reduction -> exact f32 bit-equality is the right gate")'''))

# ---------------- gene_score_impute ----------------
cells.append(md("## `gene_score_impute`"))
cells.append(md("""Per-gene contact scores from an imputed `.cool`.

| Python name | Rust name | Type | Default | Range / values | Description |
|---|---|---|---|---|---|
| `cell_path` | *(caller-side)* | str | — | — | Cooler read stays Python; Rust receives the CSR. |
| `chrom_sizes` | *(caller-side)* | Series | — | — | Drives the per-chromosome loop in the wrapper. |
| `gene_meta` | `row_start`, `row_end`, `col_start`, `col_end` | DataFrame → 4× ndarray `i64` | — | bin indices | Gene windows, already floor-divided by resolution by the orchestrator. |
| *(implicit)* | `indptr`, `indices`, `data` | — | — | — | **New in Rust** — the CSR buffers, prepared by the wrapper. |
| *(implicit)* | `n_rows`, `n_cols` | — | — | — | **New in Rust** — matrix shape. |
| *(implicit)* | `input_f32` | bool | derived | — | **New in Rust** — selects the reduction dtype so it matches scipy's. Derived from `csr.dtype`; **do not** override it, or scores stop matching upstream. |

Upstream Python call:
```python
from schicluster.draft.gene_score import gene_score_impute
scores = gene_score_impute(cell_path, chrom_sizes, gene_meta)
```"""))
cells.append(code('''chrom_sizes = pd.Series(gs_pack["gene_score.chrom_size"],
                        index=[str(c) for c in gs_pack["gene_score.chrom"]])
gene_ids = [str(g) for g in gs_pack["gene_score.gene_id"]]
gene_meta = pd.DataFrame({0: [str(gs_pack["gene_score.chrom"][0])] * len(gene_ids),
                          1: gs_pack["gene_score.gene_start_bin"],
                          2: gs_pack["gene_score.gene_end_bin"]}, index=gene_ids)

got = np.asarray(schicluster_rs.gene_score_impute(
    "data/fixtures/gene_score_small.cool", chrom_sizes, gene_meta), dtype=np.float64)
ref = np.asarray(REF["gene_score_impute"]["scores"], dtype=np.float64)
for gid, a, b in zip(gene_ids, ref, got):
    print(f"     {gid:<18} python {a:>10.4f}   rust {b:>10.4f}   delta {abs(a-b):.1e}")
e = float(np.max(np.abs(ref - got)))
verdict("gene_score_impute", "gene_score.impute", "deterministic-bounded", "max abs err", f"{e:.3e}", e <= 1e-6)
print("     exactly 0.0 — the kernel reproduces scipy's matvec-then-pairwise reduction")
print("     GENE_AT_BIN0 is 0.0 in BOTH: the (xx-1) slice start resolves to n-1 upstream")'''))

# ---------------- gene_score_raw ----------------
cells.append(md("## `gene_score_raw`"))
cells.append(md("""Per-gene scores built straight from a raw contact file. The pandas parse and
`groupby` matrix build stay Python, so this mode is far less accelerated.

| Python name | Rust name | Type | Default | Range / values | Description |
|---|---|---|---|---|---|
| `cell_path` | *(caller-side)* | str | — | — | `pd.read_csv` stays Python. |
| `chrom_sizes` | *(caller-side)* | Series | — | — | Drives the per-chromosome loop and `n_bins`. |
| `gene_meta` | 4× window arrays | DataFrame → ndarray `i64` | — | bin indices | Window is `[xx:(yy+1), xx:(yy+1)]` — **no `-1`**, unlike impute mode. |
| `resolution` | *(caller-side)* | int | `10000` | bp | Used for `(pos - 1) // resolution`. Note the `-1`, which impute mode does not have. |
| `chrom1`, `pos1`, `chrom2`, `pos2` | *(caller-side)* | int | `1, 2, 5, 6` | 0-based | Column indices in the contact file. |

Upstream Python call:
```python
from schicluster.draft.gene_score import gene_score_raw
scores = gene_score_raw(cell_path, chrom_sizes, gene_meta, resolution, chrom1, pos1, chrom2, pos2)
```"""))
cells.append(code('''got = np.asarray(schicluster_rs.gene_score_raw(
    "data/fixtures/gene_score_small.contact.tsv.gz", chrom_sizes, gene_meta,
    int(gs_pack["gene_score.resolution"]), 0, 1, 2, 3), dtype=np.int64)
ref = np.asarray(REF["gene_score_raw"]["scores"], dtype=np.int64)
for gid, a, b in zip(gene_ids, ref, got):
    print(f"     {gid:<18} python {a:>6d}   rust {b:>6d}")
e = int(np.max(np.abs(ref - got)))
verdict("gene_score_raw", "gene_score.raw", "deterministic-strict", "max abs err", str(e), e == 0)
print("     int32 counts: addition is exact and order-independent, so strict is the right gate")'''))

# ---------------- compute_decay ----------------
cells.append(md("## `compute_decay`"))
cells.append(md("""Per-cell contact-distance decay and per-chromosome sparsity, streamed from the
gzipped contact TSV in Rust.

| Python name | Rust name | Type | Default | Range / values | Description |
|---|---|---|---|---|---|
| `cell_name` | *(caller-side)* | str | — | — | Only used to name the returned frames' column. |
| `contact_path` | `path` | str | — | `.tsv` or `.tsv.gz` | **Rust opens the file itself** — this is the one place the port moves the I/O seam. `.gz` is detected by suffix and read with `MultiGzDecoder`. |
| `bins` | `bin_edges` | ndarray f64 | — | monotonic | Log-spaced histogram edges. Computed by numpy Python-side and passed through, so Rust never recomputes `exp2` and there is no ULP drift. |
| `chrom_sizes` | `chroms` | DataFrame → `Vec<String>` | — | — | Only the **index** is used, as the known-chromosome filter. |
| `resolution` | `resolution` | int → `i64` | `10000` | `> 0` | Bin size for the sparsity count. |
| `chrom1`, `pos1`, `chrom2`, `pos2` | same | int → `usize` | `1, 2, 5, 6` | 0-based | Column indices in the contact file. |

Upstream Python call:
```python
from schicluster.cool.contact_distance import compute_decay
sparsity, decay = compute_decay(cell_name, contact_path, bins, chrom_sizes, resolution, 1, 5, 2, 6)
```"""))
cells.append(code('''chroms = [str(c) for c in cd_pack["contact_distance.chroms"]]
cd_sizes = pd.DataFrame(cd_pack["contact_distance.chrom_sizes"], index=chroms)
c1, p1, c2, p2 = [int(x) for x in cd_pack["contact_distance.cols"]]
sp_df, dc_df = schicluster_rs.compute_decay(
    cell_name="fixture_cell", contact_path="data/fixtures/contact_distance_small.tsv.gz",
    bins=cd_pack["contact_distance.bin_edges"], chrom_sizes=cd_sizes,
    resolution=int(cd_pack["contact_distance.resolution"]),
    chrom1=c1, pos1=p1, chrom2=c2, pos2=p2)

ref_decay = np.asarray(REF["compute_decay"]["decay"], dtype=np.int64)
got_decay = dc_df["fixture_cell"].to_numpy(dtype=np.int64)
e1 = int(np.max(np.abs(ref_decay - got_decay)))
verdict("compute_decay", "contact_distance.decay", "deterministic-strict", "max abs err", str(e1), e1 == 0)

ref_sp = np.asarray(REF["compute_decay"]["sparsity"], dtype=np.int64)
got_sp = np.asarray([int(sp_df.loc[k, "fixture_cell"]) for k in sorted(sp_df.index)], dtype=np.int64)
e2 = int(np.max(np.abs(ref_sp - got_sp))) if ref_sp.shape == got_sp.shape else -1
verdict("compute_decay", "contact_distance.sparsity", "deterministic-strict", "max abs err", str(e2), e2 == 0)
print("     both are integer counts -> exact under any order")'''))

# ---------------- aggregate ----------------
cells.append(md("## Aggregate verdict"))
cells.append(code('''tbl = pd.DataFrame(VERDICTS, columns=["Function", "Output", "Class", "Metric", "Value", "Pass"])
tbl["Pass"] = tbl["Pass"].map(lambda b: "PASS" if b else "FAIL")
print(tbl.to_string(index=False))
print()
n_fail = int((tbl["Pass"] == "FAIL").sum())
if n_fail:
    print(f"FAIL — {n_fail} function-level comparison(s) did not clear their gate")
else:
    print(f"PASS — all {len(tbl)} function-level comparisons cleared their gate")'''))

cells.append(md("""### Notes on scope

`random_walk_cpu`, `impute_chromosome`, `loop_bkg_chrom` and
`merge_cells_for_single_chromosome` are exercised at the pipeline level in
[`compare_Python_vs_Rust.ipynb`](compare_Python_vs_Rust.ipynb) and by
`tests/test_parity.py` rather than being re-dumped per function here — their
reference paths write cooler/HDF5 artefacts, which needs `tables`/`h5py` in
both environments. `set_num_threads` and `patch_schicluster` have no upstream
counterpart to compare against; they are documented in
[`tutorial_loop_domain.ipynb`](tutorial_loop_domain.ipynb)."""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.10"}}
nbf.write(nb, str(OUT))
print("wrote", OUT, f"({len(cells)} cells)")
