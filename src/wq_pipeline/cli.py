"""Single-binary CLI for the wq_pipeline package.

Subcommands map to the existing module-level main()s:

    wq discover     pull all available data fields -> fields.csv
    wq run          submit the template grid (Step 3 grid runner)
    wq survivors    filter alphas DB by Sharpe / fitness / IS checks
    wq correlate    fetch PnL series, prune correlated alphas, build basket
    wq expressions  bulk-grid raw expressions from a file (no template/{f})

Each subcommand takes its own --help. Top-level help lists the subcommands.
"""
from __future__ import annotations

import sys

from . import correlation, expressions, fields, runner, survivors

HANDLERS = {
    "discover": (fields.main, "pull all available data fields -> fields.csv"),
    "run": (runner.main, "submit the template grid (Step 3 grid runner)"),
    "survivors": (
        survivors.main,
        "filter alphas DB by Sharpe / fitness / IS checks -> survivors.csv",
    ),
    "correlate": (
        correlation.main,
        "fetch PnL, prune correlated alphas, write uncorrelated basket.csv",
    ),
    "expressions": (
        expressions.main,
        "bulk-grid raw expressions from a file (skip template/{f} substitution)",
    ),
}


def _print_top_help() -> None:
    width = max(len(k) for k in HANDLERS)
    print("usage: wq <subcommand> [args]\n")
    print("subcommands:")
    for name, (_, doc) in HANDLERS.items():
        print(f"  {name:<{width}}  {doc}")
    print("\nrun `wq <subcommand> --help` for the args of any subcommand.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _print_top_help()
        return 0
    cmd, rest = args[0], args[1:]
    if cmd not in HANDLERS:
        print(f"unknown subcommand: {cmd!r}\n", file=sys.stderr)
        _print_top_help()
        return 2
    handler, _ = HANDLERS[cmd]
    return handler(rest)


if __name__ == "__main__":
    sys.exit(main())
