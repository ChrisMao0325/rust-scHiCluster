"""Wire the rebuildpy engine into pytest sys.path if available.

The manifest-driven `test_exact_match.py` needs `engine.parity_metrics` from
the rebuildpy kit (default `/large_storage/zhoulab/shengmao/rebuildpy`, override
with `REBUILDPY_DIR`). When rebuildpy is missing — e.g. CI runners that haven't
checked it out, or fresh contributor machines — we silently skip the path
injection. `test_exact_match.py` then guards its import with `pytest.importorskip`
so the manifest gate collects-and-skips cleanly while the rest of the suite
(notably the legacy `test_parity.py`) still runs.
"""
from __future__ import annotations

import os
import pathlib
import sys


_DEFAULT_REBUILDPY = "/large_storage/zhoulab/shengmao/rebuildpy"
_REBUILDPY = pathlib.Path(os.environ.get("REBUILDPY_DIR", _DEFAULT_REBUILDPY))
if _REBUILDPY.exists() and str(_REBUILDPY) not in sys.path:
    sys.path.insert(0, str(_REBUILDPY))
