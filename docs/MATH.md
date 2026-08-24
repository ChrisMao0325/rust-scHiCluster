# MATH — perturbation bounds for (B) bounded ε-approximation rewrites

> One section per (B) rewrite committed to the port. Bounds are derived, not handwaved. (E) and (C) rewrites do NOT appear here.

## Conventions

- `eps_32 = 2^-23 ≈ 1.19e-7` (IEEE-754 binary32 machine epsilon, since loop / scan / find_summit / insulation are f32 in upstream).
- `eps_64 = 2^-52 ≈ 2.22e-16` (binary64, used for compartment, embedding-feature where upstream is f64, and topdom signal arithmetic).
- `n` denotes the reduction length.
- For an out-of-order parallel reduction `Σ xᵢ` with operands bounded by `M = max|xᵢ|`, the worst-case error vs the canonical left-to-right serial reduction is bounded by `n · eps · M` (Higham 2002, §4).

## (B) rewrites

**None accepted.** Phase 6's acceleration policy is (E)-exact only: no rewrite
that reorders a floating-point reduction was committed, so there is no
perturbation bound to derive. This section records the reasoning per candidate
so the absence is auditable rather than merely empty.

| Candidate | Where | Verdict | Why |
|---|---|---|---|
| Row-parallel convolution | `conv.rs` | **(E), accepted in Phase 0** | Output rows are independent and each row's `kh x kw` reduction runs serially in a fixed order. Nothing is reordered, so no bound applies. |
| Hoisted mirror-index tables | `conv.rs` | **(E), accepted in Phase 6 iter 1** | Pure integer tabulation of a data-independent function, plus a pre-flipped kernel. The float operand sequence is unchanged and the output is bit-identical — `conv.convolved` held its metric at exactly `1.7881393432617188e-07`. |
| Rayon across genes | `gene_score.rs` | **(E), accepted in Phase 5** | Genes are independent outputs. Each gene's reduction reproduces scipy's own two-stage order (serial per-row matvec, then pairwise over the row sums) in the source dtype, and is bit-exact against it. |
| Hoisted mirror-index tables in `gaussian_filter_2d` | `lib.rs` | **(E), Phase 6b iter 1** | Same argument as the `conv.rs` hoist, applied to the impute path's own separable Gaussian, which never routed through the shared primitive. Verified bit-identical on real data: re-imputing a real cell gives `max|new - old| = 0.0`. |
| Packed key + stable radix sort | `merge.rs` | **(E), Phase 6b iter 2** | An LSD radix sort is stable by construction, so each key's f64 additions keep their input order; the packed `row*ncols+col` key preserves row-major emission. `sort_unstable_by_key` is ~2x faster and deliberately unused — it would reorder those additions, making it (B). |
| `target-cpu=native` | crate-wide | **(E), rejected for no gain** | Admissible — verified bit-identical across all 21 outputs — but 0.0943 s vs 0.0946 s is noise. Rejected on speed, not on admissibility. Not shipped. |
| Prefix-sum insulation window | `insulation.rs` | **(B), rejected** | `P[b] - P[a]` is a floating-point identity only in exact arithmetic. Error scales with the prefix length `n` rather than the window length `w`, and cancels two separately-rounded large values. Bound derived in [ACCELERATION_LOG.md](ACCELERATION_LOG.md) iter 3. Out of scope under an (E)-only policy. |
| Cross-cell rayon reduction | `merge.rs` | **(B), not attempted** | Parallelising the cross-cell accumulation would reorder an f32 sum, bounded by `n_cells · eps_32 · max\|x\|`. Still out of scope. Note the determinism requirement that motivated Phase 1's `BTreeMap` was **kept** while the cost was removed — see the radix-sort row above; the two are independent. |

### Why `gene_score` needed exact reduction-order matching, not a bound

The one place this port came close to needing a (B) bound, it turned out a
bound would not have helped — the gate is **absolute**, not relative.

`scipy.sparse.csr_matrix.sum(axis=None)` is not a flat `data.sum()`. It is
computed as `(self @ np.ones(n_cols, dtype=res_dtype)).sum()`: a CSR matvec
that accumulates each row **serially in stored column order**, followed by
`np.add.reduce` over the dense row-sums vector, which uses pairwise summation
(8-way unrolled base case, 128-element blocksize). `res_dtype` is the matrix's
own dtype for floats, so an f32 cool reduces entirely in f32.

Imputed cools store `count` as f32, so upstream's own answer carries about
`3.8e-6` of f32 rounding on a window summing to ~48. Measured against it:

| Candidate reduction | Max abs error vs upstream | 1e-6 gate |
|---|---|---|
| f64 accumulate (the obvious port) | `3.35e-6` | fails |
| flat f32 pairwise over window values | `3.81e-6` | fails |
| f32 matvec-then-pairwise (scipy's own) | `0.0` | passes, bit-exact |

A more *accurate* implementation is further from the reference, because the
reference is the thing being matched. Since `data/manifest.yaml` is read-only
and the protocol forbids widening a threshold to make a port pass, the third
was implemented. Verified on every fixture gene and 2000 randomised f32 windows
in `tests/test_gene_score_semantics.py`.

### If a (B) rewrite is ever accepted

Add one section here per rewrite, with the closed-form bound instantiated on a
named fixture (`n`, `max|x|`, resulting bound), the affected manifest outputs,
and a cross-link to its `ACCELERATION_LOG.md` iteration. Per
`PARITY_TAXONOMY.md §1`, any output touched by a reordered reduction must be
classed no tighter than `deterministic-bounded`, and
`embedding.cell_by_feature` (threshold `0.0`) must never be touched by one.
