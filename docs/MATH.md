# MATH — perturbation bounds for (B) bounded ε-approximation rewrites

> One section per (B) rewrite committed to the port. Bounds are derived, not handwaved. (E) and (C) rewrites do NOT appear here.

## Conventions

- `eps_32 = 2^-23 ≈ 1.19e-7` (IEEE-754 binary32 machine epsilon, since loop / scan / find_summit / insulation are f32 in upstream).
- `eps_64 = 2^-52 ≈ 2.22e-16` (binary64, used for compartment, embedding-feature where upstream is f64, and topdom signal arithmetic).
- `n` denotes the reduction length.
- For an out-of-order parallel reduction `Σ xᵢ` with operands bounded by `M = max|xᵢ|`, the worst-case error vs the canonical left-to-right serial reduction is bounded by `n · eps · M` (Higham 2002, §4).

## (B) rewrites

_None yet — populated by the Acceleration Agent in Phase 5._
