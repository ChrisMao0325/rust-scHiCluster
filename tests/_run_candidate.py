"""Run schicluster_rs Rust candidate and dump JSON.

Invoked under $RUST_TEST_ENV (rebuild-rust env, py 3.10) after
`maturin develop --release`. Output keys mirror py_reference_driver.py.

Usage (from repo root, after the Rust build):
    python tests/_run_candidate.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import schicluster_rs


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "data" / "fixtures" / "conv_small.npz"
OUT = REPO_ROOT / "data" / "fixtures" / "candidate_output.json"


def _load_fixture() -> dict:
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


def cand_conv_convolved(fixture: dict) -> list:
    """Candidate for manifest output 'conv.convolved'."""
    a = np.ascontiguousarray(fixture["input"], dtype=np.float32)
    k = np.ascontiguousarray(fixture["kernel"], dtype=np.float32)
    out = schicluster_rs.convolve2d_mirror(a, k)
    return np.asarray(out, dtype=np.float32).tolist()


def main() -> None:
    fixture = _load_fixture()
    payload = {
        "convolved": cand_conv_convolved(fixture),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT} (keys: {list(payload.keys())})")


if __name__ == "__main__":
    main()
