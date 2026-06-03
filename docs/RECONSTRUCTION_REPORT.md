# RECONSTRUCTION_REPORT — schicluster_rs (loop / domain / compartment / embedding / merge ports)

> Filled at Phase 4 close. All 8 sections are non-skippable.

## 1. Identity
- Package: schicluster-rs
- Upstream Python version: scHiCluster `1.3.5.dev22+gd566046`
- Algorithm class: per-output (see `data/manifest.yaml`)
- Threshold: per-output (manifest)
- Final parity value: _TBD_
- Audit class: _TBD_
- Speedup vs Python: _TBD_

## 2. Python API coverage audit
_See `AUDIT.md` (auto-generated)._

## 3. Parity evidence
_TBD — per-output metric values, per-fixture wall-clock + parity, reproducible reference command._

## 4. Acceleration evidence
_TBD — embed `examples/evolution.png`; list accepted vs rejected rewrites with admissibility proofs._

## 5. Code quality audit
- [ ] `maturin build --release` produces a wheel
- [ ] `pip install` clean in a fresh py3.10 env
- [ ] `pytest -q` green under `--release` build
- [ ] Four mandatory notebooks pre-executed
- [ ] License compatible with upstream
- [ ] Version pinned

## 6. Known limitations
_TBD — shuffle path in loop_bkg excluded from gate; embedding speedup modest (I/O-bound); etc._

## 7. Integration
- Crate location: `rust/`
- Wheel location: built via `maturin develop --release` into `rebuild-rust`
- Public API: `python/schicluster_rs/__init__.py`
- Tutorial: `examples/tutorial_loop_domain.ipynb` (TBD)

## 8. Sign-off
- Author: _TBD_
- Date: _TBD_
- Active time: _TBD_
- Final audit class: _TBD_
