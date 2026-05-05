"""
cli/commands/benchmark.py — benchmark CUDA kernels with synthetic data.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def _perf(fn, *args, warmup: int = 1, runs: int = 5) -> float:
    """Return median wall-clock time in ms over `runs` calls."""
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def run_benchmark(args, project_root: Path) -> None:
    from cli.commands.analyze import _load_bridge
    cuda = _load_bridge(project_root)

    kernel = getattr(args, "kernel", "full").lower()

    print(f"\n{'='*60}")
    print(f"regmap benchmark — {kernel.upper()} kernels")
    print(f"Device: RTX 5080 (sm_120)")
    print("=" * 60)

    if kernel in ("cosine", "full"):
        _bench_cosine(cuda)

    if kernel in ("graph", "full"):
        _bench_graph(cuda)

    print("=" * 60)


def _bench_cosine(cuda) -> None:
    configs = [
        (100,  100, 768, "small  "),
        (1000, 500, 768, "medium "),
        (10000,500, 768, "large  "),
    ]
    print(f"\n{'Kernel':<18}  {'Config':<30}  {'Time':>8}  {'Pairs/s':>12}")
    print("-" * 60)
    for M, N, D, tag in configs:
        A = np.random.randn(M, D).astype(np.float32)
        B = np.random.randn(N, D).astype(np.float32)
        try:
            ms = _perf(cuda.cosine_similarity, A, B)
            pairs_per_s = M * N / (ms / 1000)
            print(f"{'cosine_warp':<18}  {M}×{N}×{D} {tag}  {ms:>8.2f} ms  "
                  f"{pairs_per_s:>10.0f}/s")
        except Exception as e:
            print(f"{'cosine_warp':<18}  {M}×{N}×{D}  ERROR: {e}")


def _bench_graph(cuda) -> None:
    """Benchmark graph kernels on a synthetic similarity graph."""
    for n_regs, n_procs in [(500, 500), (1000, 500)]:
        n = n_regs + n_procs
        label = f"{n_regs}r×{n_procs}p"

        # Synthetic similarity matrix
        np.random.seed(42)
        sim = np.random.rand(n_regs, n_procs).astype(np.float32)

        try:
            ms_tf = _perf(cuda.threshold_filter, sim, 0.75)
        except Exception as e:
            print(f"threshold_filter {label}  ERROR: {e}")
            continue

        graph = cuda.threshold_filter(sim, 0.75)
        rp  = graph["row_ptr"].astype(np.int32)
        ci  = graph["col_indices"].astype(np.int32)
        wts = (1.0 - graph["values"]).astype(np.float32)

        print(f"\n  Graph: {label}  nodes={n}  edges={graph['num_edges']}")
        print(f"  {'Kernel':<20}  {'Time':>8}")
        print(f"  {'-'*28}")
        print(f"  {'threshold_filter':<20}  {ms_tf:>8.2f} ms")

        def _try(name, fn, *a):
            try:
                ms = _perf(fn, *a)
                print(f"  {name:<20}  {ms:>8.2f} ms")
            except Exception as e:
                print(f"  {name:<20}  ERROR: {e}")

        _try("bfs",              cuda.bfs,  rp, ci, n, n_regs, n_procs, 0)
        _try("topological_sort", cuda.topological_sort, rp, ci, n)
        _try("pagerank",         cuda.pagerank, rp, ci, n)
        _try("kruskal_mst",      cuda.kruskal_mst, rp, ci, wts, n)
        _try("dijkstra",         cuda.dijkstra, rp, ci, wts, n, 0)

        # Floyd-Warshall on a smaller matrix
        small = min(n, 500)
        dist_mat = np.where(sim[:small, :min(n_procs, small)] > 0.8,
                            (1.0 - sim[:small, :min(n_procs, small)]).astype(np.float32),
                            np.float32(1e9))
        sq = min(dist_mat.shape)
        dist_sq = dist_mat[:sq, :sq].copy()
        np.fill_diagonal(dist_sq, 0.0)
        if sq <= 500:
            _try(f"floyd_warshall({sq})",
                 cuda.floyd_warshall, dist_sq)
