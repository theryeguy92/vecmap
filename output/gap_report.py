"""
output/gap_report.py — list source sections with no match in the target corpus.
"""

from __future__ import annotations

import numpy as np

from graph.builder import EmbeddingSet

_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def print_gaps(
    sim_matrix: np.ndarray,
    regs: EmbeddingSet,
    procs: EmbeddingSet,
    threshold: float = 0.75,
    colour: bool = True,
) -> list[dict]:
    """Print a detailed gap report.  Returns list of gap dicts."""
    if not colour:
        global _RED, _YELLOW, _CYAN, _BOLD, _RESET
        _RED = _YELLOW = _CYAN = _BOLD = _RESET = ""

    M, N = sim_matrix.shape

    gaps = []
    for i in range(M):
        row = sim_matrix[i]
        best_j = int(row.argmax())
        best_sim = float(row[best_j])
        if best_sim < threshold:
            gaps.append(
                {
                    "reg_idx": i,
                    "best_proc": best_j,
                    "best_sim": best_sim,
                    "reg_label": regs.label(i),
                    "proc_label": procs.label(best_j),
                }
            )

    line_w = 72
    print(f"\n{_BOLD}{'='*line_w}{_RESET}")
    print(f"{_BOLD}GAP ANALYSIS  (threshold: {threshold:.2f}){_RESET}")
    print(f"{len(gaps)} of {M} source sections are not covered")
    print("=" * line_w)

    if not gaps:
        print(f"{_BOLD}All source sections are covered.{_RESET}")
        print("=" * line_w)
        return gaps

    for g in gaps:
        sim = g["best_sim"]
        sev_colour = _RED if sim < 0.5 else _YELLOW
        sev_label = "NO MATCH" if sim < 0.5 else "PARTIAL "

        print(f"\n{sev_colour}[{sev_label}]{_RESET}  {_BOLD}{g['reg_label']}{_RESET}")
        if sim >= 0.01:
            print(
                f"           Closest match: {_CYAN}{g['proc_label']}{_RESET}  "
                f"(sim={sim:.4f})"
            )
        else:
            print("           No target section found at all")

        sec = regs.sections[g["reg_idx"]]
        if sec.text_preview:
            preview = sec.text_preview[:100].replace("\n", " ")
            print(f"           Preview: {preview}…")

    print(f"\n{'='*line_w}")
    critical = sum(1 for g in gaps if g["best_sim"] < 0.5)
    partial = len(gaps) - critical
    print(
        f"  {_RED}Critical gaps (sim < 0.5): {critical}{_RESET}   "
        f"{_YELLOW}Partial gaps:  {partial}{_RESET}"
    )
    print("=" * line_w)

    return gaps
