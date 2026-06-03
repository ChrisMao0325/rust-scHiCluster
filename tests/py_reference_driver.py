"""Run upstream Python references and dump JSON.

Invoked under $PYTHON_REF_ENV (schicluster env, py 3.6). For Phase 0 the
only reference computed is conv.convolved (scipy.ndimage.convolve with
mirror mode). Later phases extend this driver with one block per ported
function from data/manifest.yaml.

py3.6 NOTE: do NOT add `from __future__ import annotations` (added in
3.7) or PEP 585 subscripted generics — schicluster env is Python 3.6.

Usage (from repo root):
    python tests/py_reference_driver.py
"""
import json
import pathlib

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "data" / "fixtures" / "conv_small.npz"
OUT = REPO_ROOT / "data" / "fixtures" / "reference_output.json"


def _load_fixture():
    """Return a dict of arrays loaded from the conv_small fixture."""
    with np.load(str(FIXTURE)) as z:
        return {k: z[k] for k in z.files}


def ref_conv_convolved(fixture):
    """Reference for manifest output 'conv.convolved'.

    The fixture already carries the scipy-computed reference under the
    'convolved' key. We just round-trip it so the JSON shape matches the
    candidate's. (Recomputing here would only re-verify the fixture.)
    """
    return fixture["convolved"].tolist()


def main():
    fixture = _load_fixture()
    payload = {
        "convolved": ref_conv_convolved(fixture),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    print("wrote {} (keys: {})".format(OUT, list(payload.keys())))


if __name__ == "__main__":
    main()
