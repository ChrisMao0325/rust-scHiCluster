"""schicluster-rs CLI — Rust-backed thin wrapper around upstream hicluster.

Pre-applies ``patch_schicluster()`` so every monkey-patched function
(impute_chromosome, the four loop kernels, insulation + TopDom,
compartment, embedding cell-by-feature) runs on the Rust side, then
hands argv off to the upstream argparse from ``schicluster.__main__``.

Swap ``hicluster`` -> ``schicluster-rs`` in your scripts to get the Rust
backend with no other changes.

Subcommands currently exposed:

* ``filter-contact``  - blacklist filtering of single-cell contact pairs.
  Pure pandas / pybedtools I/O; no per-call Rust speedup, but a single
  binary covers the full pipeline.
* ``prepare-impute``  (alias ``imputation``) - generates the snakemake
  workflow that fans out impute_chromosome calls per chunk. The
  generated Snakefiles still call ``hic-internal impute-chromosome`` via
  ``shell:`` rules, so each worker should also have schicluster-rs
  installed and either source a sitecustomize that calls
  ``patch_schicluster()`` or run rules under ``schicluster-rs
  <subcmd>`` to get the per-chrom kernel routed through Rust.
* ``domain``         - per-cell insulation score + native TopDom
  (drops the rpy2 / R round-trip). patch_schicluster() rebinds both
  the insulation kernel and the TopDom closure inside upstream's
  call_domain_and_insulation orchestrator.
* ``compartment``    - per-chrom CpG-weighted compartment score +
  decay-normalised A/B/AB strength. Rebind happens at module level so
  the per-cell ProcessPoolExecutor inside multiple_cell_compartment
  transparently uses Rust.
* ``embedding``      - cell-by-feature upper-tri extraction with
  distance filter + scalar scaling, before SVD. SVD itself stays
  sklearn (kept intentionally — the Phase-4 parity gate targets the
  pre-SVD matrix; see docs/PERFORMANCE.md).
* ``cpg-ratio``      - bedtools-nuc + pandas; no Rust speedup, just
  the upstream prerequisite step for ``compartment``, included so a
  single binary covers the full compartment workflow.

For any other upstream subcommand, fall back to ``hicluster <subcmd>``
directly; this CLI is intentionally narrow.
"""
from __future__ import annotations

import sys
import textwrap

# Subcommands we explicitly support. Anything else is rejected with a
# helpful message rather than silently passed through, so users do not
# assume Rust acceleration where it is not (yet) wired up.
_SUPPORTED = {
    "filter-contact",
    "prepare-impute",
    "imputation",   # upstream alias for prepare-impute
    "domain",       # insulation + native TopDom (Phase 2)
    "compartment",  # CpG-weighted comp + decay-normalised strength (Phase 3)
    "embedding",    # cell-by-feature extraction; SVD stays sklearn (Phase 4)
    "cpg-ratio",    # bedtools-nuc + pandas; no Rust speedup, included for completeness
}
_HELP_FLAGS = {"-h", "--help", "-v", "--version"}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in _HELP_FLAGS:
        sys.stderr.write(textwrap.dedent(
            """
            schicluster-rs - Rust-backed wrapper around hicluster.

            Usage:
                schicluster-rs <subcommand> [options...]

            Supported subcommands:
                filter-contact      Blacklist-filter single-cell contacts.
                prepare-impute      Write snakemake jobs for imputation.
                domain              Per-cell insulation + native TopDom.
                compartment         CpG-weighted compartment score + A/B strength.
                embedding           Cell-by-feature extraction (pre-SVD).
                cpg-ratio           bedtools-nuc CpG ratio for compartment.

            For per-subcommand options, run:
                schicluster-rs <subcommand> --help

            For any other hicluster subcommand, use hicluster directly.
            """
        ).lstrip())
        return 0 if (len(sys.argv) >= 2 and sys.argv[1] in _HELP_FLAGS) else 1

    subcmd = sys.argv[1]
    if subcmd not in _SUPPORTED:
        sys.stderr.write(
            f"schicluster-rs: subcommand {subcmd!r} is not wired up.\n"
            f"  Run `hicluster {subcmd} ...` directly, or open a PR to extend\n"
            f"  schicluster-rs/python/schicluster_rs/__main__.py::_SUPPORTED.\n"
        )
        return 2

    # Apply the monkey-patch BEFORE upstream's argparse imports any
    # schicluster.* module so the patched functions are picked up by
    # later `from schicluster... import ...` calls inside the upstream
    # subcommand body.
    try:
        import schicluster_rs
    except ImportError as e:  # pragma: no cover - guarded at install time
        sys.stderr.write(f"schicluster-rs: failed to import schicluster_rs: {e}\n")
        return 3
    schicluster_rs.patch_schicluster()

    try:
        from schicluster.__main__ import main as _upstream_main
    except ImportError as e:
        sys.stderr.write(
            "schicluster-rs: upstream schicluster is not installed in this env.\n"
            f"  Install it with `pip install schicluster` first ({e}).\n"
        )
        return 4

    _upstream_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
