"""
cli/commands/analyze.py — full pipeline: load embeddings → GPU → output.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def _load_bridge(project_root: Path):
    """Import regmap_cuda from the build directory."""
    build_dir = project_root / "build"
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    try:
        import regmap_cuda

        return regmap_cuda
    except ImportError as e:
        print(f"ERROR: Could not import regmap_cuda from {build_dir}")
        print(f"       Run: cd {project_root}/build && make regmap_cuda")
        print(f"       ({e})")
        sys.exit(1)


def run_analyze(args, project_root: Path) -> None:
    from graph.builder import load_embeddings
    from output.matrix import print_matrix
    from output.gap_report import print_gaps
    from output.rankings import print_rankings
    from output.knowledge_graph import export_dot

    cuda = _load_bridge(project_root)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 1. Load embeddings                                                   #
    # ------------------------------------------------------------------ #
    print(f"\nLoading source embeddings from: {args.regs}")
    regs = load_embeddings([Path(p) for p in args.regs])
    print(f"  {regs.n} source sections from {len(args.regs)} file(s)")

    print(f"Loading target embeddings from:  {args.procs}")
    procs = load_embeddings([Path(p) for p in args.procs])
    print(f"  {procs.n} target sections from {len(args.procs)} file(s)")

    if regs.n == 0 or procs.n == 0:
        print("ERROR: no sections loaded — check paths")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. GPU cosine similarity                                             #
    # ------------------------------------------------------------------ #
    print(f"\nRunning cosine_warp on GPU  ({regs.n} × {procs.n} × 768)…")
    t_sim_start = time.perf_counter()
    sim_matrix = cuda.cosine_similarity(regs.embeddings, procs.embeddings)
    t_sim = (time.perf_counter() - t_sim_start) * 1000
    print(
        f"  Done in {t_sim:.2f} ms  "
        f"sim range=[{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]"
    )

    # ------------------------------------------------------------------ #
    # 3. Threshold filter → CSR graph                                      #
    # ------------------------------------------------------------------ #
    threshold = args.threshold
    graph = cuda.threshold_filter(sim_matrix, threshold)
    n_nodes = graph["num_nodes"]
    n_edges = graph["num_edges"]
    print(
        f"  threshold_filter(t={threshold}): {n_edges} edges  "
        f"({n_edges / max(regs.n * procs.n, 1) * 100:.1f}% of matrix)"
    )

    rp = graph["row_ptr"].astype(np.int32)
    ci = graph["col_indices"].astype(np.int32)
    nr = graph["num_regs"]
    np_ = graph["num_procs"]
    vals_dist = np.maximum(0.0, 1.0 - graph["values"]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 4. Similarity matrix                                                  #
    # ------------------------------------------------------------------ #
    _summary = print_matrix(sim_matrix, regs, procs, threshold=threshold)

    # ------------------------------------------------------------------ #
    # 5. Gap analysis via BFS                                               #
    # ------------------------------------------------------------------ #
    bfs_result = cuda.bfs(rp, ci, n_nodes, nr, np_, source=0)
    _gap_procs = int(bfs_result["num_gaps"])

    _gap_rows = print_gaps(sim_matrix, regs, procs, threshold=threshold)

    # ------------------------------------------------------------------ #
    # 6. PageRank critical document ranking                                 #
    # ------------------------------------------------------------------ #
    if n_edges > 0:
        ranks = cuda.pagerank(rp, ci, n_nodes)
        print_rankings(ranks, regs, procs, top_n=min(10, regs.n))
    else:
        print("\n(PageRank skipped — no edges above threshold)")

    # ------------------------------------------------------------------ #
    # 7. Optional: Dijkstra strongest similarity paths                      #
    # ------------------------------------------------------------------ #
    if n_edges > 0 and args.verbose:
        dijk = cuda.dijkstra(rp, ci, vals_dist, n_nodes, source=0)
        finite = dijk["distances"][dijk["distances"] < 1e8]
        print(
            f"\nDijkstra from REG-0: {len(finite)} reachable nodes  "
            f"closest={finite.min():.4f}  farthest={finite.max():.4f}"
        )

    # ------------------------------------------------------------------ #
    # 8. Optional: knowledge graph export                                   #
    # ------------------------------------------------------------------ #
    if args.graph:
        dot_path = Path(args.graph)
        dot_path, n_dot_edges = export_dot(
            sim_matrix,
            regs,
            procs,
            threshold=threshold,
            output_path=dot_path,
        )
        print(f"\nKnowledge graph written: {dot_path}  ({n_dot_edges} edges)")
        print(f"  Render with: dot -Tsvg {dot_path} -o regmap_graph.svg")

    # ------------------------------------------------------------------ #
    # 9. Timing summary                                                     #
    # ------------------------------------------------------------------ #
    t_total = (time.perf_counter() - t0) * 1000
    print(f"\nTotal wall time: {t_total:.1f} ms  " f"(GPU similarity: {t_sim:.2f} ms)")
