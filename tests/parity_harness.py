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
    that engine.parity_metrics knows how to compare.

    For CSR-triplet dicts {"rows": [...], "cols": [...], "vals": [...]} we
    densify into a square matrix whose side length is max(rows union cols) + 1
    (small fixtures only). Plain lists become np.ndarray. Everything else
    passes through.
    """
    if isinstance(obj, dict) and set(obj.keys()) >= {"rows", "cols", "vals"}:
        rows = np.asarray(obj["rows"], dtype=np.int64)
        cols = np.asarray(obj["cols"], dtype=np.int64)
        vals = np.asarray(obj["vals"], dtype=np.float64)
        if rows.size == 0:
            return np.zeros((1, 1), dtype=np.float64)
        n = int(max(rows.max(), cols.max())) + 1
        dense = np.zeros((n, n), dtype=np.float64)
        dense[rows, cols] = vals
        return dense
    if isinstance(obj, list):
        try:
            return np.asarray(obj)
        except ValueError:
            return obj
    return obj


def evaluate(output: OutputSpec, reference_blob: dict, candidate_blob: dict) -> dict:
    """Run one output through the class-aware parity check."""
    ref = _dig(reference_blob, output.location)
    cand = _dig(candidate_blob, output.location)
    if ref is None:
        return {"status": "skip-missing-reference", "metric": None,
                "threshold": output.threshold, "message": f"{output.name}: no reference dump yet"}
    if cand is None:
        return {"status": "skip-missing-candidate", "metric": None,
                "threshold": output.threshold, "message": f"{output.name}: no candidate dump yet"}

    ref_arr = _to_numpy(ref)
    cand_arr = _to_numpy(cand)

    # Pad densified CSR matrices up to a common shape.
    if (isinstance(ref_arr, np.ndarray) and isinstance(cand_arr, np.ndarray)
            and ref_arr.shape != cand_arr.shape and ref_arr.ndim == 2 and cand_arr.ndim == 2):
        n = max(ref_arr.shape[0], cand_arr.shape[0])
        m = max(ref_arr.shape[1], cand_arr.shape[1])
        def _pad(a: np.ndarray) -> np.ndarray:
            out = np.zeros((n, m), dtype=a.dtype)
            out[:a.shape[0], :a.shape[1]] = a
            return out
        ref_arr = _pad(ref_arr)
        cand_arr = _pad(cand_arr)

    # Special case: topdom.bed.interval_jaccard — set Jaccard over domain (start, end) pairs.
    if output.name == "topdom.bed.interval_jaccard":
        ref_bed = _dig(reference_blob, "$.topdom.bed") or []
        cand_bed = _dig(candidate_blob, "$.topdom.bed") or []
        ref_doms = {(r["chromStart"], r["chromEnd"]) for r in ref_bed if r["name"] == "domain"}
        cand_doms = {(r["chromStart"], r["chromEnd"]) for r in cand_bed if r["name"] == "domain"}
        union = ref_doms | cand_doms
        if not union:
            metric_value = 1.0
        else:
            metric_value = len(ref_doms & cand_doms) / len(union)
        ok = metric_value >= output.threshold
        return {
            "status": "pass" if ok else "fail",
            "metric": metric_value,
            "threshold": output.threshold,
            "message": f"{output.name}: Jaccard={metric_value:.4f} vs threshold={output.threshold!r}",
        }

    # Special case: topdom.bed.bin_label_agreement — per-bin tag agreement after
    # expanding both BEDs onto the bin grid (1 bp granularity since the fixture is tiny).
    if output.name == "topdom.bed.bin_label_agreement":
        ref_bed = _dig(reference_blob, "$.topdom.bed") or []
        cand_bed = _dig(candidate_blob, "$.topdom.bed") or []
        if not ref_bed and not cand_bed:
            metric_value = 1.0
        else:
            def _max_end(bed):
                return max((r["chromEnd"] for r in bed), default=0)
            grid_end = max(_max_end(ref_bed), _max_end(cand_bed))
            ref_tags = ["gap"] * grid_end
            cand_tags = ["gap"] * grid_end
            for r in ref_bed:
                for i in range(r["chromStart"], min(r["chromEnd"], grid_end)):
                    ref_tags[i] = r["name"]
            for r in cand_bed:
                for i in range(r["chromStart"], min(r["chromEnd"], grid_end)):
                    cand_tags[i] = r["name"]
            n_total = max(grid_end, 1)
            n_match = sum(1 for a, b in zip(ref_tags, cand_tags) if a == b)
            metric_value = n_match / n_total
        ok = metric_value >= output.threshold
        return {
            "status": "pass" if ok else "fail",
            "metric": metric_value,
            "threshold": output.threshold,
            "message": f"{output.name}: agreement={metric_value:.4f} vs threshold={output.threshold!r}",
        }

    # Special case: find_summit.sizes -- align via the idx intersection.
    if output.name == "find_summit.sizes":
        ref_idx = np.asarray(_dig(reference_blob, "$.find_summit.idx"))
        cand_idx = np.asarray(_dig(candidate_blob, "$.find_summit.idx"))
        ref_sizes = np.asarray(_dig(reference_blob, "$.find_summit.sizes"))
        cand_sizes = np.asarray(_dig(candidate_blob, "$.find_summit.sizes"))
        common = np.intersect1d(ref_idx, cand_idx)
        if common.size == 0:
            return {"status": "fail", "metric": None,
                    "threshold": output.threshold,
                    "message": f"{output.name}: empty idx intersection"}
        ref_map = dict(zip(ref_idx.tolist(), ref_sizes.tolist()))
        cand_map = dict(zip(cand_idx.tolist(), cand_sizes.tolist()))
        ref_arr = np.asarray([ref_map[k] for k in common])
        cand_arr = np.asarray([cand_map[k] for k in common])

    # For classification with integer labels that have >2 unique values, sklearn
    # needs average='macro' rather than the default 'binary'.
    extra_kwargs: dict = {}
    if output.algorithm_class == "classification":
        ref_labels = set(np.asarray(ref_arr).ravel().tolist())
        cand_labels = set(np.asarray(cand_arr).ravel().tolist())
        if len(ref_labels | cand_labels) > 2:
            extra_kwargs["average"] = "macro"

    metric_value = compute_parity(
        reference=ref_arr,
        candidate=cand_arr,
        algorithm_class=output.algorithm_class,
        **extra_kwargs,
    )
    # Special case: deterministic-strict with threshold=0.0 means exact
    # bit-equality (max abs error must be == 0.0).  is_pass uses strict
    # less-than (<), so 0.0 < 0.0 = False — handle with <= here.
    if output.algorithm_class == "deterministic-strict" and output.threshold == 0.0:
        ok = float(metric_value) <= 0.0
    else:
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
