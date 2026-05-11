"""
output/rankings.py — PageRank-based critical document ranking.
"""

from __future__ import annotations

import numpy as np

from graph.builder import EmbeddingSet

_GOLD = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def print_rankings(
    ranks: np.ndarray,
    regs: EmbeddingSet,
    procs: EmbeddingSet,
    top_n: int = 15,
    colour: bool = True,
) -> None:
    """Print PageRank-sorted list of the most critical source and target sections."""
    if not colour:
        global _GOLD, _CYAN, _BOLD, _DIM, _RESET
        _GOLD = _CYAN = _BOLD = _DIM = _RESET = ""

    n_regs = regs.n
    n_procs = procs.n
    n_total = n_regs + n_procs
    assert (
        len(ranks) == n_total
    ), f"rank array length {len(ranks)} != n_regs+n_procs {n_total}"

    reg_ranks = [(i, float(ranks[i])) for i in range(n_regs)]
    proc_ranks = [(i - n_regs, float(ranks[i])) for i in range(n_regs, n_total)]

    reg_ranks.sort(key=lambda x: x[1], reverse=True)
    proc_ranks.sort(key=lambda x: x[1], reverse=True)

    line_w = 72
    print(f"\n{_BOLD}{'='*line_w}{_RESET}")
    print(f"{_BOLD}CRITICAL DOCUMENTS  (PageRank, top {top_n}){_RESET}")
    print("=" * line_w)
    print(f"  {'#':>3}  {'Score':>8}  {'Document':<22}  {'Section'}")
    print("-" * line_w)

    for rank_pos, (idx, score) in enumerate(reg_ranks[:top_n], 1):
        medal = ""
        if rank_pos == 1:
            medal = f"{_GOLD}★{_RESET} "
        elif rank_pos <= 3:
            medal = f"{_GOLD}·{_RESET} "
        else:
            medal = "  "
        label = regs.label(idx, max_heading=28)
        doc_part, _, sec_part = label.partition(" | ")
        print(f"  {medal}{rank_pos:>3}  {score:>8.5f}  {doc_part:<22}  {sec_part}")

    if len(reg_ranks) > top_n:
        print(f"  {_DIM}... {len(reg_ranks) - top_n} more source sections{_RESET}")

    print(f"\n{_BOLD}MOST-CONNECTED TARGETS  (top {min(top_n, 10)}){_RESET}")
    print("-" * line_w)
    print(f"  {'#':>3}  {'Score':>8}  {'Document':<22}  {'Section'}")
    print("-" * line_w)
    for rank_pos, (idx, score) in enumerate(proc_ranks[: min(top_n, 10)], 1):
        label = procs.label(idx, max_heading=28)
        doc_part, _, sec_part = label.partition(" | ")
        print(
            f"  {_CYAN}  {rank_pos:>3}{_RESET}  {score:>8.5f}  {doc_part:<22}  {sec_part}"
        )

    print("=" * line_w)
