"""
output/matrix.py — terminal similarity matrix table.

For each source section, shows the best-matching target section and
whether the pair is COVERED (sim ≥ threshold), PARTIAL (0.5 ≤ sim < threshold),
or GAP (no match above 0.5).
"""

from __future__ import annotations

import numpy as np

from graph.builder import EmbeddingSet

# ANSI colour codes — degrade gracefully if terminal does not support them
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _status(sim: float, threshold: float) -> tuple[str, str]:
    """Return (label, colour) for a similarity value."""
    if sim >= threshold:
        return "COVERED", _GREEN
    if sim >= 0.5:
        return "PARTIAL", _YELLOW
    return "GAP", _RED


def print_matrix(
    sim_matrix: np.ndarray,
    regs: EmbeddingSet,
    procs: EmbeddingSet,
    threshold: float = 0.75,
    top_n: int = 3,
    colour: bool = True,
) -> dict:
    """
    Print similarity matrix to stdout.

    Returns a summary dict for programmatic use.
    """
    if not colour:
        global _GREEN, _YELLOW, _RED, _BOLD, _RESET
        _GREEN = _YELLOW = _RED = _BOLD = _RESET = ""

    M, N = sim_matrix.shape
    assert M == regs.n and N == procs.n, "matrix dimensions do not match EmbeddingSets"

    W_REG = 38
    W_PROC = 38
    W_SIM = 7
    W_STAT = 8
    line_w = W_REG + W_PROC + W_SIM + W_STAT + 9

    print(f"\n{_BOLD}{'='*line_w}{_RESET}")
    print(f"{_BOLD}SIMILARITY MATRIX  (threshold: {threshold:.2f}){_RESET}")
    print(f"Source corpus: {M} sections   Target corpus: {N} sections")
    print("=" * line_w)
    print(
        f"{'SOURCE SECTION':<{W_REG}}  {'BEST TARGET MATCH':<{W_PROC}}  "
        f"{'SIM':>{W_SIM}}  {'STATUS'}"
    )
    print("-" * line_w)

    n_covered = n_partial = n_gap = 0
    rows = []

    for i in range(M):
        row = sim_matrix[i]
        best_j = int(row.argmax())
        best_sim = float(row[best_j])

        reg_label = regs.label(i)[:W_REG]
        proc_label = procs.label(best_j)[:W_PROC] if best_sim >= 0.01 else "—"
        label, colour_code = _status(best_sim, threshold)

        sim_str = f"{best_sim:.4f}" if best_sim >= 0.01 else "   —  "
        stat_str = f"{colour_code}{label}{_RESET}"

        print(
            f"{reg_label:<{W_REG}}  {proc_label:<{W_PROC}}  "
            f"{sim_str:>{W_SIM}}  {stat_str}"
        )

        if label == "COVERED":
            n_covered += 1
        elif label == "PARTIAL":
            n_partial += 1
        else:
            n_gap += 1

        rows.append(
            {"reg_idx": i, "proc_idx": best_j, "sim": best_sim, "status": label}
        )

    print("=" * line_w)
    coverage_pct = 100.0 * n_covered / M if M else 0.0
    print(
        f"{'SUMMARY':<{W_REG}}  "
        f"{_GREEN}COVERED:{_RESET} {n_covered}  "
        f"{_YELLOW}PARTIAL:{_RESET} {n_partial}  "
        f"{_RED}GAP:{_RESET} {n_gap}  "
        f"Coverage: {_BOLD}{coverage_pct:.1f}%{_RESET}"
    )
    print("=" * line_w)

    return {
        "rows": rows,
        "covered": n_covered,
        "partial": n_partial,
        "gaps": n_gap,
        "coverage": coverage_pct,
    }
