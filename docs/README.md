# Documentation index

## If you are using the package

Start here — none of it assumes any knowledge of Rust:

| | |
|---|---|
| [README](../README.md) | Installation, quick start, Python API, CLI, input formats |
| [tutorial/](../tutorial/README.md) | Per-module guides with worked examples |
| [examples/](../../rust-scHiCluster-benchmark/examples/) | Runnable notebooks (in the developer directory) |
| [PERFORMANCE.md](PERFORMANCE.md) | Speed and accuracy summary |

## If you are evaluating whether to trust the port

| | |
|---|---|
| [RECONSTRUCTION_REPORT.md](RECONSTRUCTION_REPORT.md) | Full sign-off: identity, parity evidence for all 21 outputs, acceleration evidence, known limitations |
| [compare_Python_vs_Rust.ipynb](../../rust-scHiCluster-benchmark/examples/compare_Python_vs_Rust.ipynb) | Pipeline-level parity, visualised |
| [function_by_function_Python_parity.ipynb](../../rust-scHiCluster-benchmark/examples/function_by_function_Python_parity.ipynb) | Function-level parity with parameter tables |

## If you are developing the port

Implementation internals and benchmark methodology live **outside this
repository**, in the
[developer documentation](../../rust-scHiCluster-benchmark/README.md), so that
package users never have to read them:

| | |
|---|---|
| [IMPLEMENTATION.md](../../rust-scHiCluster-benchmark/docs/IMPLEMENTATION.md) | How each kernel works and why |
| [BENCHMARKS.md](../../rust-scHiCluster-benchmark/docs/BENCHMARKS.md) | All timings, with methodology |
| [ACCELERATION.md](../../rust-scHiCluster-benchmark/docs/ACCELERATION.md) | Optimisation history and lessons |

### Protocol artefacts (kept here)

These are formal deliverables of the [rebuildpy](https://github.com/omicverse/rebuildpy)
port protocol and are cross-linked from the reconstruction report, so they stay
with the package:

| | |
|---|---|
| [MATH.md](MATH.md) | Perturbation bounds; why no bounded-error rewrite was accepted |
| [AUDIT.md](AUDIT.md) | Python API coverage (read its scope note first) |
| [ITERATION_LOG.md](ITERATION_LOG.md) | Per-phase port history |
| [ACCELERATION_LOG.md](ACCELERATION_LOG.md) | Acceleration search, pass A |
| [ACCELERATION_LOG_SURVEY.md](ACCELERATION_LOG_SURVEY.md) | Acceleration search, pass B (all 11 kernels) |
| [DISCOVERY.md](DISCOVERY.md) | Pre-port survey |
| [superpowers/](superpowers/) | Design specs and implementation plans |
