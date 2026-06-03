"""schicluster-rs CLI — Rust-backed thin wrapper around upstream hicluster.

Pre-applies ``patch_schicluster()`` so every monkey-patched function
(impute_chromosome, the four loop kernels) runs on the Rust side, then
hands argv off to the upstream argparse from ``schicluster.__main__``.

Swap ``hicluster`` -> ``schicluster-rs`` in your scripts to get the Rust
backend with no other changes. Example, replacing the calls from
``ProstateCancer/example_notebook/imputation.md``::

    schicluster-rs filter-contact \\
        --output_dir rmbkl/ \\
        --blacklist_1d_path mm10-blacklist.v2.bed.gz \\
        --chr1 1 --pos1 2 --chr2 3 --pos2 4 \\
        --contact_table contact_table.tsv \\
        --chrom_size_path chrom_sizes.txt

    schicluster-rs prepare-impute \\
        --cell_table contact_table_rmbkl.tsv \\
        --batch_size 1536 --pad 1 --cpu_per_job 96 \\
        --chr1 1 --pos1 2 --chr2 3 --pos2 4 \\
        --output_dir impute/100K/ \\
        --chrom_size_path chrom_sizes.txt \\
        --output_dist 500000000 --window_size 500000000 --step_size 500000000 \\
        --resolution 100000

Subcommands currently exposed:

* ``filter-contact``  - blacklist filtering of single-cell contact pairs.
  Pure pandas / pybedtools I/O; no per-call Rust speedup, but the patch
  is still applied so any downstream import inside the same process
  benefits.
* ``prepare-impute``  (alias ``imputation``) - generates the snakemake
  workflow that fans out impute_chromosome calls per chunk. The
  generated Snakefiles still call ``hic-internal impute-chromosome`` via
  ``shell:`` rules, so each worker should also have schicluster-rs
  installed and either source a sitecustomize that calls
  ``patch_schicluster()`` or run rules under ``schicluster-rs
  <subcmd>`` to get the per-chrom kernel routed through Rust.

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
    "imputation",  # upstream alias for prepare-impute
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
