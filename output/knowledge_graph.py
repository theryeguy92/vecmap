"""
output/knowledge_graph.py — export the similarity graph as a Graphviz DOT file.

Nodes: source sections (boxes, blue) and target sections (ellipses, green).
Edges: similarity above threshold, labelled with score.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from graph.builder import EmbeddingSet


def _dot_escape(s: str) -> str:
    return re.sub(r'["\\\n]', lambda m: "\\" + m.group(), s)


def export_dot(
    sim_matrix: np.ndarray,
    regs: EmbeddingSet,
    procs: EmbeddingSet,
    threshold: float = 0.75,
    output_path: str | Path = "regmap_graph.dot",
    max_edges_per_reg: int = 3,
) -> tuple[Path, int]:
    """
    Write a Graphviz DOT file for the similarity knowledge graph.

    Returns (output_path, total_edges).
    """
    output_path = Path(output_path)
    M, N = sim_matrix.shape

    lines = [
        "digraph regmap {",
        '  graph [rankdir=LR fontname="Helvetica" label="regmap Knowledge Graph"'
        " labelloc=t fontsize=14];",
        '  node [fontname="Helvetica" fontsize=10];',
        '  edge [fontname="Helvetica" fontsize=8];',
        "",
        "  // Source corpus nodes",
        "  subgraph cluster_source {",
        '    label="Source Corpus"; style=filled; color=lightblue;',
    ]
    for i in range(M):
        s = regs.sections[i]
        doc = _dot_escape((s.doc_id or s.stem)[:20])
        head = _dot_escape(s.heading[:22] if s.heading else s.section_id)
        lines.append(
            f'    "SRC-{i}" [label="{doc}\\n{head}" shape=box '
            f"style=filled fillcolor=lightblue];"
        )
    lines += [
        "  }",
        "",
        "  // Target corpus nodes",
        "  subgraph cluster_target {",
        '    label="Target Corpus"; style=filled; color=lightgreen;',
    ]
    for j in range(N):
        s = procs.sections[j]
        doc = _dot_escape((s.doc_id or s.stem)[:20])
        head = _dot_escape(s.heading[:22] if s.heading else s.section_id)
        lines.append(
            f'    "TGT-{j}" [label="{doc}\\n{head}" shape=ellipse '
            f"style=filled fillcolor=lightgreen];"
        )
    lines += ["  }", "", "  // Similarity edges"]

    total_edges = 0
    for i in range(M):
        row = sim_matrix[i]
        idxs = np.where(row >= threshold)[0]
        idxs = idxs[np.argsort(row[idxs])[::-1]][:max_edges_per_reg]
        for j in idxs:
            sim = float(row[j])
            if sim >= 0.9:
                colour = "#006400"
            elif sim >= 0.8:
                colour = "#228B22"
            else:
                colour = "#FF8C00"
            lines.append(
                f'  "SRC-{i}" -> "TGT-{j}" '
                f'[label="{sim:.3f}" color="{colour}" '
                f'penwidth="{1 + (sim - threshold) * 4:.1f}"];'
            )
            total_edges += 1

    lines += ["}", ""]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path, total_edges
