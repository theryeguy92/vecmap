"""Tests for output modules (matrix, gap_report, rankings, knowledge_graph)."""

import io
import sys
import numpy as np
import pytest
from dataclasses import dataclass


@dataclass
class Section:
    stem: str = "TEST"
    doc_id: str = "TEST-001"
    doc_type: str = "Order"
    subject: str = "Test"
    section_id: str = "1.0"
    heading: str = "Test Section"
    text_preview: str = "Preview text"
    obligations: int = 1
    citations: int = 0
    global_idx: int = 0


@dataclass
class FakeEmbeddingSet:
    """Minimal EmbeddingSet-like for output tests."""
    sections: list
    embeddings: np.ndarray

    @property
    def n(self) -> int:
        return len(self.sections)

    def label(self, idx: int, max_heading: int = 25) -> str:
        s = self.sections[idx]
        return f"{s.doc_id[:18]} | {s.heading[:max_heading]}"


def _make_es(n: int) -> FakeEmbeddingSet:
    """Create a fake EmbeddingSet with n sections."""
    sections = [
        Section(stem=f"DOC-{i}", doc_id=f"DOE-O-{i}", global_idx=i)
        for i in range(n)
    ]
    return FakeEmbeddingSet(sections=sections, embeddings=np.zeros((n, 768), dtype=np.float32))


def _make_sim_matrix(M: int, N: int, seed: int = 42) -> np.ndarray:
    """Synthetic similarity matrix with controlled values."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.4, 0.99, size=(M, N)).astype(np.float32)


class TestOutputMatrix:
    def test_returns_summary_dict(self):
        """print_matrix returns a summary dict with expected keys."""
        from output.matrix import print_matrix

        es_a = _make_es(5)
        es_b = _make_es(3)
        sim = _make_sim_matrix(5, 3)

        saved = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = print_matrix(sim, es_a, es_b, threshold=0.75, colour=False)
        finally:
            sys.stdout = saved

        assert "covered" in result
        assert "partial" in result
        assert "gaps" in result
        assert "coverage" in result
        assert result["covered"] + result["partial"] + result["gaps"] == 5

    def test_dimension_mismatch_asserts(self):
        """Matrix dimensions must match EmbeddingSet sizes."""
        from output.matrix import print_matrix

        es_a = _make_es(5)
        es_b = _make_es(3)
        wrong_sim = np.ones((2, 2), dtype=np.float32)

        with pytest.raises(AssertionError):
            print_matrix(wrong_sim, es_a, es_b)

    def test_coverage_percentage_is_plausible(self):
        """Coverage percentage is between 0 and 100."""
        from output.matrix import print_matrix

        es_a = _make_es(5)
        es_b = _make_es(3)
        sim = _make_sim_matrix(5, 3)

        saved = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = print_matrix(sim, es_a, es_b, colour=False)
        finally:
            sys.stdout = saved

        assert 0.0 <= result["coverage"] <= 100.0

    def test_no_colour_mode_suppresses_ansi(self):
        """colour=False produces no ANSI escape sequences."""
        from output.matrix import print_matrix

        es_a = _make_es(5)
        es_b = _make_es(3)
        sim = _make_sim_matrix(5, 3)

        saved = sys.stdout
        sys.stdout = out = io.StringIO()
        try:
            print_matrix(sim, es_a, es_b, colour=False)
        finally:
            sys.stdout = saved

        output = out.getvalue()
        assert "\033[" not in output, "ANSI codes should be absent in no-colour mode"


class TestOutputGaps:
    def test_returns_list_of_gap_dicts(self):
        """print_gaps returns a list of gap dictionaries."""
        from output.gap_report import print_gaps

        es_a = _make_es(5)
        es_b = _make_es(3)
        sim = _make_sim_matrix(5, 3)

        saved = sys.stdout
        sys.stdout = io.StringIO()
        try:
            gaps = print_gaps(sim, es_a, es_b, colour=False)
        finally:
            sys.stdout = saved

        assert isinstance(gaps, list)
        for g in gaps:
            assert "reg_idx" in g
            assert "best_sim" in g
            assert "reg_label" in g
            assert "proc_label" in g

    def test_high_threshold_produces_more_gaps(self):
        """Higher threshold → more sections classified as gaps."""
        from output.gap_report import print_gaps

        es_a = _make_es(5)
        es_b = _make_es(3)
        sim = _make_sim_matrix(5, 3)

        saved = sys.stdout
        sys.stdout = io.StringIO()
        try:
            gaps_lo = print_gaps(sim, es_a, es_b, threshold=0.5, colour=False)
        finally:
            sys.stdout = saved

        saved = sys.stdout
        sys.stdout = io.StringIO()
        try:
            gaps_hi = print_gaps(sim, es_a, es_b, threshold=0.95, colour=False)
        finally:
            sys.stdout = saved

        assert len(gaps_lo) <= len(gaps_hi), (
            "Higher threshold should produce same or more gaps"
        )
