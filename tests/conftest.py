"""Test fixtures for regmap tests."""

from pathlib import Path
import pytest

SAMPLE_DOCS = Path(__file__).resolve().parent / "sample_docs" / "parsed"


@pytest.fixture
def sample_dir():
    """Path to parsed sample data with .npy embeddings and _meta.json files."""
    if not SAMPLE_DOCS.is_dir():
        pytest.skip(f"Sample docs not available in CI: {SAMPLE_DOCS}")
    return SAMPLE_DOCS


@pytest.fixture
def small_embeddings(sample_dir):
    """Load a small EmbeddingSet (first 3 docs) for fast unit tests."""
    from graph.builder import load_embeddings

    npy_files = sorted(sample_dir.glob("*_sections.npy"))[:3]
    return load_embeddings([npy_files[0].parent])


@pytest.fixture
def two_corpora(sample_dir):
    """Return two EmbeddingSets (doe_orders vs themselves) for similarity tests."""
    from graph.builder import load_embeddings

    npy_files = sorted(sample_dir.glob("*_sections.npy"))
    split = max(3, len(npy_files) // 2)
    a = load_embeddings([npy_files[0].parent])
    # For a second corpus, use a subset of the same dir
    # (different slice of files)
    b = load_embeddings([p for p in npy_files[split : split + 3]])
    return a, b
