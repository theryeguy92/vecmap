"""Tests for graph/builder.py — EmbeddingSet loading."""

import numpy as np
from graph.builder import load_embeddings, EmbeddingSet


class TestEmbeddingSetLoad:
    def test_loads_embeddings_from_directory(self, sample_dir):
        """Loading a directory with *_sections.npy files returns an EmbeddingSet."""
        es = load_embeddings([sample_dir])
        assert isinstance(es, EmbeddingSet)
        assert es.n > 0, "Expected at least one section"
        assert es.embeddings.shape == (es.n, 768)
        assert es.embeddings.dtype == np.float32

    def test_sections_have_global_indices(self, sample_dir):
        """Every section gets a unique global_idx across all documents."""
        es = load_embeddings([sample_dir])
        indices = [s.global_idx for s in es.sections]
        assert indices == list(
            range(es.n)
        ), f"global_idx should be sequential 0..{es.n - 1}"

    def test_sections_have_required_fields(self, sample_dir):
        """Every section has the expected dataclass fields populated."""
        es = load_embeddings([sample_dir])
        for s in es.sections:
            assert isinstance(s.stem, str) and s.stem, "stem must be non-empty"
            assert isinstance(s.doc_id, str), "doc_id must be a string"
            assert isinstance(s.section_id, str), "section_id must be a string"
            assert isinstance(s.heading, str), "heading must be a string"

    def test_label_returns_formatted_string(self, small_embeddings):
        """label() produces a string containing doc info."""
        es = small_embeddings
        label = es.label(0)
        assert isinstance(label, str)
        assert len(label) > 0
        assert "|" in label, "label should contain separator"

    def test_label_truncates_long_headings(self, small_embeddings):
        """label() truncates headings to max_heading characters."""
        es = small_embeddings
        short = es.label(0, max_heading=5)
        heading_part = short.split("|")[-1].strip()
        assert len(heading_part) <= 5

    def test_load_from_single_npy(self, sample_dir):
        """Loading a single .npy file directly works."""
        npy = next(sample_dir.glob("*_sections.npy"))
        es = load_embeddings([npy])
        assert es.n > 0
        assert es.embeddings.shape[1] == 768

    def test_missing_meta_falls_back_to_synthetic(self, tmp_path):
        """When _meta.json is missing, synthetic sections are generated."""
        import numpy as np

        # Create a minimal .npy with no _meta.json
        embeddings = np.random.rand(3, 768).astype(np.float32)
        npy_path = tmp_path / "test_sections.npy"
        np.save(str(npy_path), embeddings)

        es = load_embeddings([npy_path])
        assert es.n == 3
        assert es.sections[0].section_id == "0"
        assert es.sections[0].heading == "SEC-0"

    def test_empty_directory_raises(self, tmp_path):
        """Loading a directory with no *_sections.npy raises ValueError."""
        with np.testing.assert_raises(ValueError):
            load_embeddings([tmp_path])

    def test_nonexistent_file_raises(self):
        """Loading a non-existent .npy raises FileNotFoundError."""
        from pathlib import Path

        with np.testing.assert_raises(FileNotFoundError):
            load_embeddings([Path("/nonexistent/file.npy")])
