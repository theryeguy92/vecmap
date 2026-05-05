#!/usr/bin/env python3
"""
Generate examples/sample_output.txt and examples/sample_knowledge_graph.dot
using real embeddings with anonymized document labels.

Run from the project root:
  python3 examples/generate_examples.py
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "build"))

import numpy as np

from graph.builder import load_embeddings
from output.matrix import print_matrix
from output.gap_report import print_gaps
from output.rankings import print_rankings
from output.knowledge_graph import export_dot
import regmap_cuda


OUT_DIR = PROJECT_ROOT / "examples"

_STEM_TO_LABEL: dict[str, str] = {
    "0200.2a":          "corpus_A",
    "0452.1a":          "corpus_B",
    "0313.1a":          "doc_C",
}


def _anonymize(es, fallback: str) -> None:
    for s in es.sections:
        matched = False
        for key, label in _STEM_TO_LABEL.items():
            if key in s.stem.lower():
                s.doc_id = label
                matched = True
                break
        if not matched:
            s.doc_id = fallback


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _genericize(text: str) -> str:
    return text


def main() -> None:
    parsed = PROJECT_ROOT / "tests" / "sample_docs" / "parsed"

    corpus_a = load_embeddings([
        parsed / "0200.2a_sections.npy",
        parsed / "0452.1A_sections.npy",
    ])
    corpus_b = load_embeddings([
        parsed / "0313.1a-draft_sections.npy",
    ])

    _anonymize(corpus_a, "corpus_A")
    _anonymize(corpus_b, "doc_C")

    sim = regmap_cuda.cosine_similarity(corpus_a.embeddings, corpus_b.embeddings)
    threshold = 0.80

    buf = io.StringIO()
    sys.stdout = buf

    print("regmap analyze")
    print(f"  Source corpus : {corpus_a.n} sections  "
          f"({', '.join(sorted({s.doc_id for s in corpus_a.sections}))})")
    print(f"  Target corpus : {corpus_b.n} sections  "
          f"({', '.join(sorted({s.doc_id for s in corpus_b.sections}))})")
    print(f"  Similarity    : {corpus_a.n} × {corpus_b.n} × 768 dims  "
          f"range=[{sim.min():.4f}, {sim.max():.4f}]")
    print()

    print_matrix(sim, corpus_a, corpus_b, threshold=threshold)
    print()
    print_gaps(sim, corpus_a, corpus_b, threshold=threshold)
    print()

    graph = regmap_cuda.threshold_filter(sim, threshold)
    rp = graph["row_ptr"].astype(np.int32)
    ci = graph["col_indices"].astype(np.int32)
    n  = graph["num_nodes"]
    if graph["num_edges"] > 0:
        ranks = regmap_cuda.pagerank(rp, ci, n)
        print_rankings(ranks, corpus_a, corpus_b, top_n=10)

    sys.stdout = sys.__stdout__

    clean = _genericize(_strip_ansi(buf.getvalue()))
    sample_txt = OUT_DIR / "sample_output.txt"
    sample_txt.write_text(clean, encoding="utf-8")
    print(f"Written : {sample_txt}")

    dot_path, n_edges = export_dot(
        sim, corpus_a, corpus_b,
        threshold=threshold,
        output_path=OUT_DIR / "sample_knowledge_graph.dot",
    )
    print(f"Written : {dot_path}  ({n_edges} edges)")
    print(f"Render  : dot -Tsvg {dot_path.name} -o sample_graph.svg")


if __name__ == "__main__":
    main()
