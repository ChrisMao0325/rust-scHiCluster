"""rebuildpy parity gate, manifest-driven.

One pytest case per manifest output. Outputs without a reference / candidate
dump are marked skip — later phases land the dumps and the corresponding
cases turn green.

Run order:
  1. (in $PYTHON_REF_ENV) python tests/py_reference_driver.py
       -> data/fixtures/reference_output.json
  2. (in $RUST_TEST_ENV)  python tests/_run_candidate.py
       -> data/fixtures/candidate_output.json
  3. (in $RUST_TEST_ENV)  pytest -q tests/test_exact_match.py

The wrapper tests/run_parity.sh chains the three.

The first import below short-circuits the whole module to "skipped" on
machines without rebuildpy (CI runners that don't check it out, fresh
contributor envs); the legacy `test_parity.py` still runs in those cases.
"""
from __future__ import annotations

import pytest

pytest.importorskip(
    "engine.parity_metrics",
    reason="rebuildpy not on PYTHONPATH — set REBUILDPY_DIR or `git clone "
    "https://github.com/omicverse/rebuildpy` next to this repo to run the manifest gate.",
)

from .parity_harness import OutputSpec, evaluate, load_dumps, load_outputs  # noqa: E402


_OUTPUTS = load_outputs()
_REF_BLOB, _CAND_BLOB = load_dumps()


@pytest.mark.parametrize("output", _OUTPUTS, ids=[o.name for o in _OUTPUTS])
def test_parity(output: OutputSpec) -> None:
    result = evaluate(output, _REF_BLOB, _CAND_BLOB)
    if result["status"].startswith("skip-"):
        pytest.skip(result["message"])
    assert result["status"] == "pass", result["message"]
