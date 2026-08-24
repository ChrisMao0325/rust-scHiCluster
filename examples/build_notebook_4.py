"""Build examples/evolution.ipynb (rebuildpy Notebook 4).

A per-iteration narrative + visualisation of every iteration performed, so the
history is auditable by an outside reviewer. NOTEBOOKS.md's structure rule is
non-negotiable: one `## Iteration N — title` header per iteration, a 3-6
sentence narrative before each code cell, and a subplot per iteration.

Nine iterations are covered: 0-5 are the per-phase ports from
docs/ITERATION_LOG.md, 6-8 are the Phase 6 acceleration search renumbered from
docs/ACCELERATION_LOG.md (whose own `iter:` field is 0-3 because
engine/plot_evolution.py needs its baseline at 0).

Iterations 1-3 have NO recorded timings. They are plotted as gaps, never
invented.
"""
from __future__ import annotations

import pathlib

import nbformat as nbf

OUT = pathlib.Path(__file__).resolve().parent / "evolution.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# Evolution — how this port was actually built

**Audience:** an auditor of the engineering process asking "did the agent really
iterate, or did it skip the loop?"

Nine iterations, each with what changed, why, its admissibility class, and its
measured effect. Iterations 0–5 are the per-phase **ports**, recorded in
[`docs/ITERATION_LOG.md`](../docs/ITERATION_LOG.md). Iterations 6–8 are the
Phase 6 **acceleration search**, recorded in
[`docs/ACCELERATION_LOG.md`](../docs/ACCELERATION_LOG.md) in rebuildpy's strict
canonical schema.

**On missing data.** Iterations 1, 2 and 3 recorded no timings at all, and
iterations 0, 4 and 5 timed three different workloads, so they do not share an
axis. Rather than invent numbers to make a smooth curve, the plots below show
gaps where no measurement exists. A visible gap is honest; a fabricated point is
not. That is also why the aggregate figure at the end is built from the
acceleration log alone — it is the only sequence measured on a fixed
workload."""))

cells.append(code('''import os, pathlib, re, sys
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
REPO = pathlib.Path.cwd()
if REPO.name == "examples":
    REPO = REPO.parent
os.chdir(REPO)

import numpy as np
import matplotlib.pyplot as plt
import yaml
get_ipython().run_line_magic("matplotlib", "inline")

# (iteration, title, wall_clock_s or None, parity_metric or None, status)
HISTORY = [
    (0, "Baseline conv translation",        0.018, 1.788e-07, "port"),
    (1, "Phase 1 - loop module",            None,  9.537e-07, "port"),
    (2, "Phase 2 - domain + native TopDom", None,  5.960e-08, "port"),
    (3, "Phase 3+4 - compartment + embedding", None, 0.0,     "port"),
    (4, "Phase 5 - gene-score",             0.040, 0.0,       "port"),
    (5, "Phase 5 - contact-distance",       0.091, 0.0,       "port"),
    (6, "Accel baseline (Phase 5 close)",   0.1399, 0.0,      "baseline"),
    (7, "Accel - hoist mirror-index tables", 0.0946, 0.0,     "ACCEPT"),
    (8, "Accel - target-cpu=native",        0.0943, 0.0,      "REJECT_SLOW"),
]
THRESHOLD = 1e-6

def panel(upto):
    """Two-panel subplot of the history so far, highlighting iteration `upto`."""
    sub = [h for h in HISTORY if h[0] <= upto]
    its = [h[0] for h in sub]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))

    t = [h[2] for h in sub]
    known = [(i, v) for i, v in zip(its, t) if v is not None]
    if known:
        ax[0].plot([k[0] for k in known], [k[1] for k in known], "o-", color="tab:blue")
    missing = [i for i, v in zip(its, t) if v is None]
    for m in missing:
        ax[0].axvline(m, color="0.85", ls=":", zorder=0)
    if missing:
        ax[0].scatter(missing, [np.nan] * len(missing))
        ax[0].text(0.02, 0.06, "dotted = no timing recorded", transform=ax[0].transAxes,
                   fontsize=8, color="0.4")
    if upto in [h[0] for h in known]:
        v = dict(known)[upto]
        ax[0].scatter([upto], [v], s=170, facecolors="none", edgecolors="red", zorder=5)
    ax[0].set_yscale("log"); ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("wall-clock (s)"); ax[0].set_title("Wall-clock (lower is better)")
    ax[0].set_xlim(-0.5, len(HISTORY) - 0.5); ax[0].grid(alpha=0.3)

    p = [h[3] for h in sub]
    pk = [(i, v) for i, v in zip(its, p) if v is not None]
    ax[1].plot([k[0] for k in pk], [max(k[1], 1e-18) for k in pk], "o-", color="tab:green")
    ax[1].axhline(THRESHOLD, color="red", ls="--", label="threshold = 1e-6")
    if upto in dict(pk):
        ax[1].scatter([upto], [max(dict(pk)[upto], 1e-18)], s=170,
                      facecolors="none", edgecolors="red", zorder=5)
    ax[1].set_yscale("log"); ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("worst parity metric"); ax[1].set_title("Parity (must stay below the line)")
    ax[1].set_xlim(-0.5, len(HISTORY) - 0.5); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    plt.suptitle(f"through iteration {upto}: {dict((h[0], h[1]) for h in HISTORY)[upto]}",
                 fontsize=10)
    plt.tight_layout(); plt.show()

print("history entries:", len(HISTORY))'''))

NARRATIVES = [
    (0, "Baseline translation — `convolve2d_mirror`", """
Phase 0 scaffolded the rebuildpy artefacts, pre-registered `data/manifest.yaml`,
and ported one primitive: `scipy.ndimage.convolve(mode='mirror')` as
`rust/src/conv.rs`. It is a direct translation — parallel across output rows,
each row's reduction serial and fixed-order — so admissibility is **(E) exact**,
with no perturbation bound needed. The `mirror_index` helper was shared with the
pre-existing `gaussian_filter_2d` via `utils.rs`.

The expected effect was a speedup; the measured effect was the opposite. On the
64×64 gate fixture Rust took 0.018 s against scipy's 0.0052 s. That was written
off at the time as thread-spawn overhead on a tiny input — a conclusion
iteration 6 later showed to be wrong.
[ITER_LOG ↩](../docs/ITERATION_LOG.md)"""),
    (1, "Phase 1 — loop module", """
Whole-function ports of `calculate_chrom_background_normalization`,
`merge_cells_for_single_chromosome`, `loop_background` and `find_summit`. All
five convolutions in the loop pipeline reuse the Phase 0 primitive. `merge`
accumulates in an f64 `BTreeMap` so cross-cell emission is deterministic
row-major, and `find_summit` uses a `BinaryHeap` with an ascending-index
tie-break to match Python `heapq`'s insertion-order stability. All (E).

Ten manifest outputs turned green. One real bug surfaced: `percentile_linear`
was sorting its input slice **in place**, corrupting the downstream z-score;
fixed by sorting a copy. No wall-clock was recorded for this phase, which is why
the left panel shows a gap.
[ITER_LOG ↩](../docs/ITERATION_LOG.md)"""),
    (2, "Phase 2 — domain module, native TopDom", """
The biggest equivalence risk in the whole port: a full native TopDom
(`topdom.rs`, 581 lines) replacing the rpy2 → `TopDom.R` round-trip, plus
`insulation.rs`. It reproduces R's diamond mean signal, gap detection,
change-point and local-extreme detection, `wilcox.test(exact=F,
alternative="less")` with continuity and tie correction, and
`Convert.Bin.To.Domain.TMP`. The normal CDF uses an inline Abramowitz erf
(~1.5e-7), ample for a `p < 0.05` cut. All (E), no new crate dependencies.

This removed R from the dependency chain entirely. One off-by-one appeared in
the *Python* wrapper, not the Rust: `chromEnd` used `to_coord[to_id]` where R
uses the next row's `from.coord`, cutting every domain 10 kb short. The Rust
kernel was correct on the first build.
[ITER_LOG ↩](../docs/ITERATION_LOG.md)"""),
    (3, "Phase 3+4 — compartment + embedding", """
`single_chrom_compartment` with f64 accumulators throughout, and
`make_chrom_matrix`'s extraction kernel. Embedding is a pure f32 gather plus one
scalar multiply with **no reduction**, so it meets a `deterministic-strict` gate
at `atol=0` — exact bit-equality is the only meaningful tolerance for an
index-and-scale operation. SVD stays sklearn by design: it is not a bottleneck
and has no element-wise parity.

The final three outputs turned green, taking the gate to 17/17. A harness bug
surfaced: rebuildpy's `is_pass()` used a strict `<` against the threshold, which
makes `threshold: 0.0` mathematically unreachable; patched locally in
`tests/parity_harness.py` to use `<=` for that case. Manifest values untouched.
[ITER_LOG ↩](../docs/ITERATION_LOG.md)"""),
    (4, "Phase 5 — gene-score", """
The per-gene window-sum loop, which upstream evaluates as `D[r0:r1,
c0:c1].sum()` once per gene — 78,691 scipy submatrix allocations per cell on a
real human gene set. The kernel binary-searches each row's sorted column indices
instead, parallel across genes, each gene's reduction fixed-order: **(E)**.

This is where the port nearly went wrong. `csr.sum(axis=None)` is *not* a flat
`data.sum()`; scipy computes it as `(self @ ones(n_cols)).sum()` — a serial
per-row matvec followed by pairwise summation over the row sums, in the
matrix's own dtype. Imputed cools are f32, so upstream's own answer carries
~3.8e-6 of rounding. A straight f64 accumulation is *more accurate* and
therefore lands 3.35e-6 away, failing the pre-registered 1e-6 gate; a flat f32
pairwise sum fails too, at 3.81e-6. Reproducing scipy's two-stage order makes it
bit-exact. The gate was never widened.
[ITER_LOG ↩](../docs/ITERATION_LOG.md)"""),
    (5, "Phase 5 — contact-distance", """
`compute_decay` ported as a streaming reader: `flate2::MultiGzDecoder` plus a
`BufReader`, parsing four columns per line in constant memory, with no DataFrame
ever built. This is the one deliberate move of the I/O seam in the whole port,
justified because the cost being removed *is* the read. First I/O dependency in
the crate; the pure-Rust miniz_oxide backend keeps the wheel matrix portable.

`np.histogram`'s edge rules are reproduced exactly — right-open bins except a
right-closed final bin, out-of-range values dropped. For hg38 the top edge is
~231.7 Mb against a 249.0 Mb chr1, so upstream silently drops the longest cis
contacts, and so does this port. Both outputs are integer counts, so both gate
`deterministic-strict`. Gate reached 21/21.
[ITER_LOG ↩](../docs/ITERATION_LOG.md)"""),
    (6, "Acceleration baseline — measure before optimising", """
Phase 6 opened by building a fixed benchmark (`examples/bench_phase6.py`) at
realistic sizes, timed under rebuildpy's protocol: BLAS and rayon pinned to 4
threads, one warmup run discarded, then three timed runs.

Two earlier records turned out to be wrong. First, `conv2d_mirror` is genuinely
**slower than scipy** — 0.0740 s against 0.0692 s at 1024×1024 with an 11×11
kernel — so iteration 0's "thread-spawn overhead on a tiny input" explanation
does not hold, and this is the crate's one real acceleration target. Second,
gene-score's Phase 5 headline of 46.3× was warmup-inclusive; excluding
thread-pool spin-up, the same workload is 217.6×. No code changed here, only
measurement.
[ACCEL_LOG ↩](../docs/ACCELERATION_LOG.md)"""),
    (7, "Acceleration — hoist the mirror-index tables (accepted)", """
The conv inner loop called `mirror_index` once per `(i, j, p, q)` — a modulo and
two branches, ~127M times for the benchmark input. But the column mirror depends
only on `(j, q)` and the row mirror only on `(i, p)`; neither belonged in the
innermost loop. Both are now tabulated once per call, and the kernel is
pre-flipped so the inner loop reads it forwards.

Pure integer tabulation of a data-independent function: the float operand
sequence is untouched, so this is **(E) exact**, and the proof is empirical —
`conv.convolved` held its metric at exactly `1.7881393432617188e-07`, the same
value iteration 0 recorded. conv went 0.0740 → 0.0306 s (2.42×), flipping it
from 0.9× to **2.3× faster than scipy**. Total benchmark wall-clock 0.1399 →
0.0946 s.
[ACCEL_LOG ↩](../docs/ACCELERATION_LOG.md)"""),
    (8, "Acceleration — `target-cpu=native` (rejected, no gain)", """
Rebuilt with `RUSTFLAGS="-C target-cpu=native"`. Admissibility was verified
rather than assumed: the full 21-output gate stayed green with every metric
bit-identical to the portable build, confirming that no kernel enables
reassociating fast-math flags and that wider vector units still execute the same
fixed-order reduction. So it is **(E)**.

It was rejected anyway, for no speedup: 0.0943 s against 0.0946 s, a 0.3%
difference on a summed stddev of ~0.0008 — indistinguishable from noise. In
hindsight that is unsurprising. After iteration 7 the conv loop is bound by
memory access through two index tables, and the other kernels by gzip inflate
and sparse-index chasing; none is a dense FMA loop. `Cargo.toml` is unchanged
and the shipped wheel stays portable, and `PERFORMANCE.md` does not advertise a
knob that buys nothing.

A fourth candidate — a prefix-sum sliding window for insulation — was rejected
**as inadmissible** rather than measured: `P[b] - P[a]` makes the error scale
with the chromosome length rather than the window length, making it a (B)
rewrite, out of scope under this phase's (E)-only policy. It has no iteration
of its own here because nothing was built.
[ACCEL_LOG ↩](../docs/ACCELERATION_LOG.md)"""),
]

for n, title, body in NARRATIVES:
    cells.append(md(f"## Iteration {n} — {title}\n{body}"))
    cells.append(code(f"panel({n})"))

cells.append(md("""## Aggregate evolution figure

Re-rendered from [`docs/ACCELERATION_LOG.md`](../docs/ACCELERATION_LOG.md) by
`engine.plot_evolution`. Only the acceleration search appears: it is the one
sequence measured on a fixed workload, and `plot_evolution` plots only entries
whose status is `baseline` or `ACCEPT`, so the two rejected candidates are
deliberately absent from the curve while remaining in the log."""))

cells.append(code('''import subprocess
r = subprocess.run([sys.executable, "-m", "engine.plot_evolution",
                    "--port-dir", ".", "--log", "docs/ACCELERATION_LOG.md",
                    "--output", "examples/evolution.png", "--threshold", "1e-6"],
                   capture_output=True, text=True,
                   env={**os.environ, "PYTHONPATH": "/large_storage/zhoulab/shengmao/rebuildpy"})
print(r.stdout.strip() or r.stderr.strip()[-400:])

from IPython.display import Image, display
display(Image(filename="examples/evolution.png"))'''))

cells.append(code('''# Full history table, including the entries the aggregate figure cannot show.
print(f"{'iter':>4}  {'status':<12} {'wall-clock':>12}  {'parity':>12}  title")
print("-" * 88)
for n, title, t, p, status in HISTORY:
    ts = f"{t:.4f} s" if t is not None else "not recorded"
    ps = f"{p:.3e}" if p is not None else "—"
    print(f"{n:>4}  {status:<12} {ts:>12}  {ps:>12}  {title}")
print()
print("Rejected candidates retained in docs/ACCELERATION_LOG.md but absent from the figure:")
print("   iter 2 (log) target_cpu_native            REJECT_SLOW          — admissible, no gain")
print("   iter 3 (log) prefix_sum_insulation_window REJECT_INADMISSIBLE  — (B), never built")'''))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.10"}}
nbf.write(nb, str(OUT))
print("wrote", OUT, f"({len(cells)} cells)")
