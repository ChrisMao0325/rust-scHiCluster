#!/usr/bin/env bash
# Two-env parity orchestrator.
#
# Step 1 (under $PYTHON_REF_ENV): generate fixture + reference dump.
# Step 2 (under $RUST_TEST_ENV) : maturin develop --release; generate candidate dump.
# Step 3 (under $RUST_TEST_ENV) : pytest the manifest gate.
#
# Override env names via PYTHON_REF_ENV / RUST_TEST_ENV.

set -euo pipefail

PYTHON_REF_ENV="${PYTHON_REF_ENV:-schicluster}"
RUST_TEST_ENV="${RUST_TEST_ENV:-rebuild-rust}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"

echo "[1/3] Generate fixture + reference dump (env: $PYTHON_REF_ENV)"
conda run -n "$RUST_TEST_ENV" --no-capture-output \
    python data/fixtures/synthesize.py
conda run -n "$PYTHON_REF_ENV" --no-capture-output \
    python tests/py_reference_driver.py

echo "[2/3] Build Rust + generate candidate dump (env: $RUST_TEST_ENV)"
conda run -n "$RUST_TEST_ENV" --no-capture-output \
    maturin develop --release
conda run -n "$RUST_TEST_ENV" --no-capture-output \
    python tests/_run_candidate.py

echo "[3/3] Run parity gate (env: $RUST_TEST_ENV)"
# Use PYTHONNOUSERSITE=1 and an explicit python -m pytest invocation so that
# ~/.local site-packages (e.g. a conflicting scikit-learn build) are ignored.
PYTHONNOUSERSITE=1 conda run -n "$RUST_TEST_ENV" --no-capture-output \
    python -m pytest -q tests/test_exact_match.py
