"""Manifest-driven parity dispatch.

Loads data/manifest.yaml, walks outputs[], looks up the reference and
candidate JSON dumps, applies the class-aware metric from
engine.parity_metrics, and asserts is_pass at the manifest threshold.

If a reference dump is missing for an output, the harness marks it
'skipped' rather than failing. As later phases add reference / candidate
dumps for their outputs, those tests turn green.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

from engine.parity_metrics import compute_parity, is_pass


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.yaml"
REF_DUMP = REPO_ROOT / "data" / "fixtures" / "reference_output.json"
CAND_DUMP = REPO_ROOT / "data" / "fixtures" / "candidate_output.json"


@dataclass
class OutputSpec:
    name: str
    metric: str
    algorithm_class: str
    threshold: float
    location: str  # e.g. "$.convolved"


def load_manifest() -> dict:
    with MANIFEST_PATH.open() as f:
        return yaml.safe_load(f)


def load_outputs(manifest: dict | None = None) -> list[OutputSpec]:
    m = manifest or load_manifest()
    out = []
    for block in m["outputs"]:
        out.append(
            OutputSpec(
                name=block["name"],
                metric=block["metric"],
                algorithm_class=block["algorithm_class"],
                threshold=float(block["threshold"]),
                # location_reference and location_candidate are assumed equal
                # (the drivers dump the same key in both files).
                location=block["location_reference"],
            )
        )
    return out


def _dig(blob: dict, location: str) -> Any:
    """Resolve a JSONPath-like '$.a.b.c' against a nested dict."""
    if not location.startswith("$."):
        raise ValueError(f"location must start with '$.', got {location!r}")
    cur: Any = blob
    for key in location[2:].split("."):
        if key not in cur:
            return None
        cur = cur[key]
    return cur


def _to_numpy(obj: Any) -> Any:
    """Re-hydrate a JSON-dumped reference / candidate object into a numpy form
    that engine.parity_metrics knows how to compare. Pass-through for non-array.
    """
    if isinstance(obj, list):
        try:
            return np.asarray(obj)
        except ValueError:
            return obj
    return obj


def evaluate(output: OutputSpec, reference_blob: dict, candidate_blob: dict) -> dict:
    """Run one output through the class-aware parity check.

    Returns a dict {status, metric, threshold, message} where status is one of
    'pass', 'fail', 'skip-missing-reference', 'skip-missing-candidate'.
    """
    ref = _dig(reference_blob, output.location)
    cand = _dig(candidate_blob, output.location)
    if ref is None:
        return {"status": "skip-missing-reference", "metric": None,
                "threshold": output.threshold, "message": f"{output.name}: no reference dump yet"}
    if cand is None:
        return {"status": "skip-missing-candidate", "metric": None,
                "threshold": output.threshold, "message": f"{output.name}: no candidate dump yet"}

    metric_value = compute_parity(
        reference=_to_numpy(ref),
        candidate=_to_numpy(cand),
        algorithm_class=output.algorithm_class,
    )
    ok = is_pass(metric_value, output.algorithm_class, output.threshold)
    return {
        "status": "pass" if ok else "fail",
        "metric": metric_value,
        "threshold": output.threshold,
        "message": f"{output.name}: metric={metric_value!r} vs threshold={output.threshold!r}",
    }


def load_dumps() -> tuple[dict, dict]:
    """Load the reference and candidate JSON dumps. Either may be missing
    (returns {}) — individual outputs will then be marked skip-missing-*.
    """
    ref = json.loads(REF_DUMP.read_text()) if REF_DUMP.exists() else {}
    cand = json.loads(CAND_DUMP.read_text()) if CAND_DUMP.exists() else {}
    return ref, cand
