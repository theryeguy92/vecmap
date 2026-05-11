#!/usr/bin/env python3
"""
regmap — GPU-accelerated compliance mapping for DOE contractors.

Usage:
  python -m cli.main analyze --regs <path> --procs <path> [options]
  python -m cli.main report  --regs <path> --procs <path> --type gaps|coverage|rankings|hierarchy
  python -m cli.main benchmark [--kernel cosine|graph|full]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _add_reg_proc_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--regs",
        nargs="+",
        metavar="PATH",
        required=True,
        help="regulation embeddings: *_sections.npy file(s) or directory",
    )
    parser.add_argument(
        "--procs",
        nargs="+",
        metavar="PATH",
        required=True,
        help="procedure embeddings: *_sections.npy file(s) or directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        metavar="T",
        help="similarity threshold for compliance edge (default: 0.75)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regmap",
        description="GPU-accelerated compliance mapping for DOE contractors",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ #
    # analyze                                                              #
    # ------------------------------------------------------------------ #
    analyze = sub.add_parser(
        "analyze",
        help="Run full compliance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Load embeddings, compute GPU cosine similarity, run graph algorithms,\n"
            "and print the compliance matrix with gap and ranking analysis."
        ),
    )
    _add_reg_proc_args(analyze)
    analyze.add_argument(
        "--graph",
        metavar="FILE",
        default=None,
        help="export knowledge graph to a Graphviz .dot file",
    )
    analyze.add_argument(
        "--verbose",
        action="store_true",
        help="show extra algorithm outputs (Dijkstra paths, etc.)",
    )

    # ------------------------------------------------------------------ #
    # report                                                               #
    # ------------------------------------------------------------------ #
    report = sub.add_parser(
        "report",
        help="Generate a named compliance report",
    )
    _add_reg_proc_args(report)
    report.add_argument(
        "--type",
        required=True,
        choices=["gaps", "coverage", "rankings", "hierarchy"],
        help="report type",
    )
    report.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="write report text to FILE (default: stdout only)",
    )

    # ------------------------------------------------------------------ #
    # benchmark                                                            #
    # ------------------------------------------------------------------ #
    bench = sub.add_parser(
        "benchmark",
        help="Benchmark CUDA kernels",
    )
    bench.add_argument(
        "--kernel",
        default="full",
        choices=["cosine", "graph", "full"],
        help="which kernels to benchmark (default: full)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    # Ensure project modules are importable regardless of cwd
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        from cli.commands.analyze import run_analyze

        run_analyze(args, PROJECT_ROOT)

    elif args.command == "report":
        from cli.commands.report import run_report

        run_report(args, PROJECT_ROOT)

    elif args.command == "benchmark":
        from cli.commands.benchmark import run_benchmark

        run_benchmark(args, PROJECT_ROOT)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
