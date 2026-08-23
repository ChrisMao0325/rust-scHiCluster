"""patch_schicluster() must rebind the two new upstream workers in place.

Neither environment can host both packages at once: upstream `schicluster` is a
Python 3.6 editable install that needs rpy2, while `schicluster_rs` is
abi3-py39. So the rebinding is verified against stub modules installed into
sys.modules — that exercises patch_schicluster()'s own logic, which is the part
this repo owns. Functional verification against the real upstream happens in
tests/py_reference_driver.py + tests/_run_candidate.py.
"""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("schicluster_rs._rust")


# Every module attribute patch_schicluster() rebinds, as (module path, attrs).
_UPSTREAM_SURFACE = {
    "schicluster": [],
    "schicluster.impute": [],
    "schicluster.impute.impute_chromosome": ["random_walk_cpu", "impute_chromosome"],
    "schicluster.loop": [],
    "schicluster.loop.loop_bkg": ["calculate_chrom_background_normalization"],
    "schicluster.loop.merge_cell_to_group": ["merge_cells_for_single_chromosome"],
    "schicluster.loop.loop_calling": ["loop_background", "find_summit"],
    "schicluster.domain": [],
    "schicluster.domain.call_domain": ["single_chrom_calculate_insulation_score", "r"],
    "schicluster.compartment": [],
    "schicluster.compartment.call_compartment": ["single_chrom_compartment"],
    "schicluster.embedding": [],
    "schicluster.embedding.calc_embedding": ["make_chrom_matrix"],
    "schicluster.draft": [],
    "schicluster.draft.gene_score": ["gene_score_impute", "gene_score_raw"],
    "schicluster.cool": [],
    "schicluster.cool.contact_distance": ["compute_decay"],
}


@pytest.fixture
def stub_upstream(monkeypatch):
    """Install a minimal fake `schicluster` package tree into sys.modules."""
    if "schicluster" in sys.modules:
        pytest.skip("real upstream schicluster is installed; stubbing would mask it")
    created = {}
    for path, attrs in _UPSTREAM_SURFACE.items():
        mod = types.ModuleType(path)
        for attr in attrs:
            setattr(mod, attr, object())
        created[path] = mod
        monkeypatch.setitem(sys.modules, path, mod)
    # wire submodules as attributes of their parents, as a real package would
    for path, mod in created.items():
        if "." in path:
            parent, _, leaf = path.rpartition(".")
            setattr(created[parent], leaf, mod)
    return created


def test_patch_rebinds_gene_score_and_contact_distance(stub_upstream):
    import schicluster_rs
    from schicluster_rs.gene_score import gene_score_impute, gene_score_raw
    from schicluster_rs.contact_distance import compute_decay

    assert schicluster_rs.patch_schicluster() is True

    gs_mod = stub_upstream["schicluster.draft.gene_score"]
    cd_mod = stub_upstream["schicluster.cool.contact_distance"]

    assert gs_mod.gene_score_impute is gene_score_impute
    assert gs_mod.gene_score_raw is gene_score_raw
    assert cd_mod.compute_decay is compute_decay


def test_patch_still_rebinds_the_phase_0_to_4_surface(stub_upstream):
    """Adding Phase 5 must not regress any earlier rebinding."""
    import schicluster_rs

    assert schicluster_rs.patch_schicluster() is True

    impute = stub_upstream["schicluster.impute.impute_chromosome"]
    assert impute.random_walk_cpu is schicluster_rs.random_walk_cpu
    assert impute.impute_chromosome is schicluster_rs.impute_chromosome
    loop_calling = stub_upstream["schicluster.loop.loop_calling"]
    assert callable(loop_calling.loop_background)
    assert callable(loop_calling.find_summit)
    emb = stub_upstream["schicluster.embedding.calc_embedding"]
    assert emb.make_chrom_matrix is schicluster_rs.make_chrom_matrix


def test_cli_lists_the_new_subcommands():
    from schicluster_rs.__main__ import _SUPPORTED

    assert "gene-score" in _SUPPORTED
    assert "contact-distance" in _SUPPORTED
