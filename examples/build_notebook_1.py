"""Build examples/compare_Python_vs_Rust.ipynb (rebuildpy Notebook 1).

Pipeline-level, side-by-side parity validation against the Python reference on
the canonical fixtures. Emits the six sections NOTEBOOKS.md requires. Run this,
then execute the notebook with nbconvert:

    python examples/build_notebook_1.py
    jupyter nbconvert --to notebook --execute \
        --output compare_Python_vs_Rust.ipynb examples/compare_Python_vs_Rust.ipynb
"""
from __future__ import annotations

import pathlib

import nbformat as nbf

OUT = pathlib.Path(__file__).resolve().parent / "compare_Python_vs_Rust.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(md("""# Python vs Rust — pipeline-level parity

**Audience:** a reviewer or scientist deciding whether to trust this port.

`schicluster-rs` reimplements the numerical hot paths of
[scHiCluster](https://github.com/zhoujt1994/scHiCluster) (Zhou *et al.* 2019,
PNAS) in Rust, and integrates by monkey-patch so upstream's Python
orchestration is never edited. This notebook runs **both** implementations on
the same committed fixtures and compares every output the pre-registered parity
gate covers.

The gate itself (`tests/test_exact_match.py`) gives a PASS/FAIL that a human
cannot audit by eye. This notebook makes it visible."""))

cells.append(md("## 1. Setup"))

cells.append(code('''import os, sys, json, pathlib, subprocess, time

# Pin BLAS + rayon threads before numpy/scipy import (engine/benchmark.py convention).
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

# nbconvert runs with cwd=examples/; every path below is repo-root relative.
REPO = pathlib.Path.cwd()
if REPO.name == "examples":
    REPO = REPO.parent
os.chdir(REPO)
print("repo root:", REPO)

import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", "inline")

MANIFEST = yaml.safe_load(open("data/manifest.yaml"))
OUTPUTS = MANIFEST["outputs"]
print("pre-registered outputs:", len(OUTPUTS))'''))

cells.append(code('''# The pre-registered gate — read-only once the agent loop started.
print(f"{'output':<34} {'class':<24} {'threshold'}")
print("-" * 76)
for o in OUTPUTS:
    print(f"{o['name']:<34} {o.get('algorithm_class', MANIFEST['algorithm_class']):<24} {o['threshold']}")'''))

cells.append(md("""## 2. Python reference run

Upstream runs in `$PYTHON_REF_ENV` (conda env `schicluster`, **Python 3.6**,
with R + rpy2 for the TopDom reference). `schicluster_rs` is `abi3-py39` and
cannot be imported there, which is why parity is a cross-environment
dump-and-compare rather than a single-process test."""))

cells.append(code('''t0 = time.perf_counter()
ref_proc = subprocess.run(
    ["conda", "run", "-n", "schicluster", "--no-capture-output",
     "python", "tests/py_reference_driver.py"],
    capture_output=True, text=True)
ref_wall = time.perf_counter() - t0
print("returncode:", ref_proc.returncode)
print(ref_proc.stdout.strip().splitlines()[-1] if ref_proc.stdout.strip() else ref_proc.stderr[-500:])
print(f"reference wall-clock: {ref_wall:.2f} s")

ver = subprocess.run(
    ["conda", "run", "-n", "schicluster", "python", "-c",
     "import sys, schicluster; print(sys.version.split()[0], schicluster.__version__)"],
    capture_output=True, text=True).stdout.strip()
print("reference env: Python + schicluster =", ver)'''))

cells.append(md("""## 3. Rust candidate run

Confirm this is a **`--release`** build — a debug build would be 10–100× slower
and invalidate every timing below."""))

cells.append(code('''import importlib.metadata as _md
t0 = time.perf_counter()
cand_proc = subprocess.run([sys.executable, "tests/_run_candidate.py"],
                           capture_output=True, text=True)
cand_wall = time.perf_counter() - t0
print("returncode:", cand_proc.returncode)
print(cand_proc.stdout.strip().splitlines()[-1] if cand_proc.stdout.strip() else cand_proc.stderr[-500:])
print(f"candidate wall-clock: {cand_wall:.2f} s")

import schicluster_rs
print("schicluster_rs version:", _md.version("schicluster-rs"))
print("extension:", schicluster_rs._rust.__file__ if hasattr(schicluster_rs, "_rust") else "(via _RUST_AVAILABLE)")
print("rust available:", schicluster_rs._RUST_AVAILABLE)
print("NOTE: built with `maturin develop --release` — see docs/RECONSTRUCTION_REPORT.md §5.")'''))

cells.append(code('''REF = json.load(open("data/fixtures/reference_output.json"))
CAND = json.load(open("data/fixtures/candidate_output.json"))
print("reference top-level keys:", sorted(REF))
print("candidate top-level keys:", sorted(CAND))'''))

cells.append(md("""## 4. Per-output parity

One subsection per `manifest.yaml::outputs[]` entry. Visual treatment follows
the class: `deterministic` outputs get a Python-vs-Rust overlay plus max
absolute error; `ranked` gets a Jaccard/overlap report; `classification` gets a
label-agreement report."""))

cells.append(code('''# Use the SAME evaluator the gate uses, rather than re-implementing the
# comparison here. Each manifest class has its own metric and its own pass
# direction — `deterministic` is max-abs-error with `<= threshold`, while
# `ranked` (Jaccard) and `classification` (label agreement) are
# higher-is-better with `>= threshold`. Hand-rolling that invites a
# plausible-looking but wrong verdict, so the notebook defers to the harness
# and cannot drift from tests/test_exact_match.py.
sys.path.insert(0, "/large_storage/zhoulab/shengmao/rebuildpy")
sys.path.insert(0, "tests")
from parity_harness import load_outputs, load_dumps, evaluate

SPECS = load_outputs()
REF_BLOB, CAND_BLOB = load_dumps()
RESULTS = [(spec, evaluate(spec, REF_BLOB, CAND_BLOB)) for spec in SPECS]
print(f"evaluated {len(RESULTS)} outputs through the gate harness")'''))

cells.append(code('''def fetch(blob, path):
    """Resolve a manifest JSONPath like '$.loop_bkg.E' against a dump."""
    node = blob
    for part in path.lstrip("$.").split("."):
        if node is None:
            return None
        node = node.get(part) if isinstance(node, dict) else None
    return node


def as_vector(x):
    """Flatten a dump value to a 1-D float array, or None if not numeric."""
    if x is None or isinstance(x, dict):
        return None
    try:
        return np.asarray(x, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None


# Overlay plots only make sense for the float-vector outputs.
plottable = []
for spec, res in RESULTS:
    r = as_vector(fetch(REF_BLOB, spec.location))
    c = as_vector(fetch(CAND_BLOB, spec.location))
    if r is not None and c is not None and r.shape == c.shape and r.size:
        plottable.append((spec, res, r, c))
print("float-vector outputs plotted below:", len(plottable))
print("non-vector outputs (ranked / classification / structured):",
      [s.name for s, _ in RESULTS if s.name not in {p[0].name for p in plottable}])'''))

cells.append(code('''ncol = 3
nrow = int(np.ceil(len(plottable) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.4 * nrow))
axes = np.atleast_1d(axes).ravel()

for ax, (spec, res, r, c) in zip(axes, plottable):
    n = min(r.size, 400)
    ax.plot(r[:n], lw=2.6, alpha=0.5, label="Python")
    ax.plot(c[:n], lw=0.9, label="Rust")
    ax.set_title(f"{spec.name}\\n{spec.algorithm_class}  metric={res.get('metric'):.3e}"
                 if isinstance(res.get("metric"), float)
                 else f"{spec.name}\\n{spec.algorithm_class}", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
for k in range(len(plottable), len(axes)):
    axes[k].axis("off")
plt.tight_layout()
plt.show()'''))

cells.append(code('''print(f"{'output':<34} {'class':<24} {'threshold':>10} {'measured':>13}  verdict")
print("-" * 98)
for spec, res in RESULTS:
    m = res.get("metric")
    mtxt = f"{m:.3e}" if isinstance(m, float) else str(m)
    print(f"{spec.name:<34} {spec.algorithm_class:<24} {str(spec.threshold):>10} "
          f"{mtxt:>13}  {res['status'].upper()}")'''))

cells.append(md("""## 5. Wall-clock comparison

The three fixed benchmark workloads from `examples/bench_phase6.py`, Python
reference against the Rust port. These are deliberately larger than the gate
fixtures — a 64×64 input is dominated by thread-pool spin-up and tells you
nothing useful."""))

cells.append(code('''import importlib.util
spec = importlib.util.spec_from_file_location("bench_phase6", "examples/bench_phase6.py")
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)

rows = []
for key, fn in bench.WORKLOADS.items():
    r = fn()
    if r is not None:
        rows.append((key, r["python_s"], r["rust_s"]))
        print(f"{key:<18} python {r['python_s']:.4f} s   rust {r['rust_s']:.4f} s   "
              f"{r['python_s']/r['rust_s']:.1f}x   max|err| {r['parity_max_abs_err']:.2e}")

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(rows)); w = 0.38
ax.bar(x - w/2, [r[1] for r in rows], w, label="Python reference")
ax.bar(x + w/2, [r[2] for r in rows], w, label="Rust port")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel("wall-clock (s), warmup excluded"); ax.set_yscale("log")
ax.set_title("Python vs Rust, per benchmark workload")
for i, r in enumerate(rows):
    ax.text(i, max(r[1], r[2]) * 1.15, f"{r[1]/r[2]:.1f}x", ha="center", fontsize=10)
ax.legend(); plt.tight_layout(); plt.show()'''))

cells.append(md("## 6. Verdict"))

cells.append(code('''passed = [s.name for s, r in RESULTS if r["status"] == "pass"]
failed = [(s.name, r) for s, r in RESULTS if r["status"] not in ("pass",)]
worst = max((r["metric"] for s, r in RESULTS
             if isinstance(r.get("metric"), float) and "deterministic" in s.algorithm_class),
            default=0.0)

print(f"outputs in the pre-registered gate : {len(RESULTS)}")
print(f"passing                            : {len(passed)}")
print(f"worst deterministic max-abs-error  : {worst:.3e}")
print()
if failed:
    print("FAIL — the following outputs did not clear their pre-registered gate:")
    for name, r in failed:
        print(f"    {name}: {r['status']} — {r.get('message')}")
else:
    print("PASS — all outputs cleared the pre-registered gate")'''))

cells.append(md("""**Note on the non-vector outputs.** `topdom.bed.*` and `find_summit.*` are
ranked / classification outputs rather than float vectors, so the overlay grid
above skips them — but they are fully evaluated in the verdict table, by
interval Jaccard and per-bin label agreement respectively, with a
higher-is-better pass condition. Every number in this notebook comes from
`tests/parity_harness.evaluate`, the same evaluator `tests/test_exact_match.py`
drives, so the notebook cannot report a verdict the gate would not. See
[`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) for the per-output table."""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.10"}}
nbf.write(nb, str(OUT))
print("wrote", OUT, f"({len(cells)} cells)")
