"""Wire the rebuildpy engine into pytest sys.path.

Override the rebuildpy location with REBUILDPY_DIR.
"""
from __future__ import annotations

import os
import pathlib
import sys


_DEFAULT_REBUILDPY = "/large_storage/zhoulab/shengmao/rebuildpy"
_REBUILDPY = pathlib.Path(os.environ.get("REBUILDPY_DIR", _DEFAULT_REBUILDPY))
if not _REBUILDPY.exists():
    raise SystemExit(
        f"REBUILDPY_DIR={_REBUILDPY} does not exist. Set REBUILDPY_DIR to your "
        f"rebuildpy checkout (the dir containing engine/parity_metrics.py)."
    )
if str(_REBUILDPY) not in sys.path:
    sys.path.insert(0, str(_REBUILDPY))

# Sanity check — fail loudly here rather than from inside test bodies.
from engine import parity_metrics  # noqa: E402,F401
