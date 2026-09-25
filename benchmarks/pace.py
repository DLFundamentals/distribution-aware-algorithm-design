#!/usr/bin/env python3
"""Single entrypoint for every PACE competition experiment.

    python -m benchmarks.pace --list
    python -m benchmarks.pace --competition pace2025_ds_heuristic --test-source private
    python -m benchmarks.pace --competition pace2025_hs --build-only
    python -m benchmarks.pace --competition pace2025_hs --help

Dominating Set was the first competition wired up and grew its own module before
the others existed. The two implementations still differ in what they emit --
Dominating Set reports solution sizes against a domination lower bound and writes
`pace_private_results.csv`, while the later competitions report normalized quality
and write `pace_results.csv` -- and the paper's tables read those artifacts as
they are. So this unifies the command surface, not the output formats: every
competition is selected the same way and keeps the artifact layout its published
numbers came from.

Flags after `--competition` are passed through untouched to the competition's own
parser, so `--competition <name> --help` shows exactly what that competition
accepts.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from types import ModuleType

from benchmarks import pace2025_dominating_set, pace_competitions


@dataclass(frozen=True)
class PaceEntry:
    competition: str
    problem: str
    track: str
    output_root: str
    description: str
    module: ModuleType
    forward_args: tuple[str, ...]

    def run(self, argv: list[str]) -> int:
        """Resolve `main` at call time, not at import time.

        Binding the function object when the registry is built would capture the
        original before any test or caller could patch the module attribute --
        which silently ran the real benchmark, downloads included.
        """
        return self.module.main(argv)


def _ds_entry(competition: str, track: str, description: str) -> PaceEntry:
    return PaceEntry(
        competition=competition,
        problem="mds",
        track=track,
        output_root=str(pace2025_dominating_set.DEFAULT_OUTPUT_ROOT),
        description=description,
        module=pace2025_dominating_set,
        forward_args=("--track", track),
    )


def _competition_entry(name: str) -> PaceEntry:
    config = pace_competitions.COMPETITIONS[name]
    return PaceEntry(
        competition=name,
        problem=config.problem,
        track=config.track,
        output_root=str(pace_competitions.DEFAULT_OUTPUT_ROOT),
        description=config.description,
        module=pace_competitions,
        forward_args=("--competition", name),
    )


def _build_registry() -> dict[str, PaceEntry]:
    entries = [
        _ds_entry(
            "pace2025_ds_heuristic",
            "heuristic",
            "PACE 2025 Dominating Set heuristic-track instances. The track the paper reports.",
        ),
        _ds_entry(
            "pace2025_ds_exact",
            "exact",
            "PACE 2025 Dominating Set exact-track instances.",
        ),
    ]
    entries.extend(_competition_entry(name) for name in sorted(pace_competitions.COMPETITIONS))
    return {entry.competition: entry for entry in entries}


REGISTRY: dict[str, PaceEntry] = _build_registry()

# The Dominating Set track is fixed by the competition id, so accepting --track
# too would let the two disagree silently.
_TRACK_FLAGS = ("--track",)


def _format_listing() -> str:
    width = max(len(name) for name in REGISTRY)
    lines = [f"{'competition'.ljust(width)}  {'problem':<12} {'track':<10} artifacts"]
    lines.append(f"{'-' * width}  {'-' * 12} {'-' * 10} ---------")
    for name, entry in REGISTRY.items():
        lines.append(f"{name.ljust(width)}  {entry.problem:<12} {entry.track:<10} {entry.output_root}/")
    lines.append("")
    for name, entry in REGISTRY.items():
        lines.append(f"{name}: {entry.description}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.pace",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--competition", choices=sorted(REGISTRY))
    parser.add_argument("--list", action="store_true", help="Print the competitions and exit.")
    parser.add_argument("-h", "--help", action="store_true", dest="show_help")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)

    if args.list:
        print(_format_listing())
        return 0

    if args.competition is None:
        if args.show_help:
            parser.print_help()
            print("\nCompetitions:\n")
            print(_format_listing())
            return 0
        parser.print_usage()
        print("\nerror: --competition is required. Available competitions:\n", file=sys.stderr)
        print(_format_listing(), file=sys.stderr)
        return 2

    entry = REGISTRY[args.competition]

    conflicting = [flag for flag in _TRACK_FLAGS if flag in passthrough]
    if conflicting and entry.module is pace2025_dominating_set:
        print(
            f"error: {conflicting[0]} is implied by --competition {entry.competition}; "
            f"select the other track with --competition pace2025_ds_"
            f"{'exact' if entry.track == 'heuristic' else 'heuristic'}",
            file=sys.stderr,
        )
        return 2

    forwarded = [*entry.forward_args, *passthrough]
    if args.show_help:
        forwarded.append("--help")

    # The delegated parsers take their prog from sys.argv[0], which would
    # otherwise print usage as "pace.py" instead of the command being run.
    original_prog = sys.argv[0]
    sys.argv[0] = "python -m benchmarks.pace"
    try:
        return entry.run(forwarded)
    finally:
        sys.argv[0] = original_prog


if __name__ == "__main__":
    raise SystemExit(main())
