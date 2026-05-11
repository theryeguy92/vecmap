"""Tests for CLI module — argument parsing and command structure."""

import pytest


class TestCLIArgs:
    def test_parse_analyze_basic(self):
        """Parsing 'analyze' with required --regs and --procs."""
        from cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "analyze",
                "--regs",
                "/tmp/emb_a",
                "--procs",
                "/tmp/emb_b",
            ]
        )
        assert args.command == "analyze"
        assert args.regs == ["/tmp/emb_a"]
        assert args.procs == ["/tmp/emb_b"]
        assert args.threshold == 0.75  # default

    def test_parse_analyze_with_threshold(self):
        """Custom threshold is parsed correctly."""
        from cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "analyze",
                "--regs",
                "/tmp/a",
                "--procs",
                "/tmp/b",
                "--threshold",
                "0.85",
            ]
        )
        assert args.threshold == 0.85

    def test_parse_analyze_with_graph(self):
        """--graph flag specifies output filename for knowledge graph."""
        from cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "analyze",
                "--regs",
                "/tmp/a",
                "--procs",
                "/tmp/b",
                "--graph",
                "graph.dot",
            ]
        )
        assert args.graph == "graph.dot"

    def test_parse_benchmark_command(self):
        """Benchmark command with --kernel flag."""
        from cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "benchmark",
                "--kernel",
                "cosine",
            ]
        )
        assert args.command == "benchmark"
        assert args.kernel == "cosine"

    def test_parse_report_command(self):
        """Report command requires --regs, --procs, and --type."""
        from cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "report",
                "--regs",
                "/tmp/a",
                "--procs",
                "/tmp/b",
                "--type",
                "gaps",
            ]
        )
        assert args.command == "report"
        assert args.type == "gaps"

    def test_parse_benchmark_default_kernel(self):
        """Benchmark defaults to 'full' kernel."""
        from cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["benchmark"])
        assert args.command == "benchmark"
        assert args.kernel == "full"

    def test_only_valid_subcommands_accepted(self):
        """Invalid subcommand raises SystemExit."""
        from cli.main import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["nonexistent"])
