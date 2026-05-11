"""
cli/commands/report.py — generate named reports from existing analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def run_report(args, project_root: Path) -> None:
    from graph.builder import load_embeddings
    from cli.commands.analyze import _load_bridge

    if not args.regs or not args.procs:
        print("ERROR: --regs and --procs are required for report generation")
        sys.exit(1)

    cuda = _load_bridge(project_root)

    regs = load_embeddings([Path(p) for p in args.regs])
    procs = load_embeddings([Path(p) for p in args.procs])
    sim = cuda.cosine_similarity(regs.embeddings, procs.embeddings)

    threshold = getattr(args, "threshold", 0.75)
    out_path = Path(args.output) if getattr(args, "output", None) else None

    report_type = args.type.lower()

    if report_type == "gaps":
        from output.gap_report import print_gaps

        gaps = print_gaps(sim, regs, procs, threshold=threshold)
        if out_path:
            lines = [f"GAP REPORT  threshold={threshold}\n"]
            for g in gaps:
                lines.append(
                    f"[GAP] {g['reg_label']}  "
                    f"closest={g['proc_label']}  sim={g['best_sim']:.4f}\n"
                )
            out_path.write_text("".join(lines), encoding="utf-8")
            print(f"\nReport written to {out_path}")

    elif report_type == "coverage":
        from output.matrix import print_matrix

        summary = print_matrix(sim, regs, procs, threshold=threshold)
        if out_path:
            lines = [
                f"COVERAGE REPORT  threshold={threshold}\n",
                f"covered={summary['covered']}  partial={summary['partial']}  "
                f"gaps={summary['gaps']}  coverage={summary['coverage']:.1f}%\n\n",
            ]
            for r in summary["rows"]:
                reg_lbl = regs.label(r["reg_idx"])
                proc_lbl = procs.label(r["proc_idx"])
                lines.append(
                    f"{r['status']:<8}  {reg_lbl}  ->  {proc_lbl}  "
                    f"sim={r['sim']:.4f}\n"
                )
            out_path.write_text("".join(lines), encoding="utf-8")
            print(f"\nReport written to {out_path}")

    elif report_type == "rankings":
        graph = cuda.threshold_filter(sim, threshold)
        rp = graph["row_ptr"].astype(np.int32)
        ci = graph["col_indices"].astype(np.int32)
        n = graph["num_nodes"]
        if graph["num_edges"] > 0:
            ranks = cuda.pagerank(rp, ci, n)
            from output.rankings import print_rankings

            print_rankings(ranks, regs, procs, top_n=20)
        else:
            print("No edges above threshold — cannot compute rankings")

    elif report_type == "hierarchy":
        graph = cuda.threshold_filter(sim, threshold)
        rp = graph["row_ptr"].astype(np.int32)
        ci = graph["col_indices"].astype(np.int32)
        n = graph["num_nodes"]
        if graph["num_edges"] > 0:
            topo = cuda.topological_sort(rp, ci, n)
            order = topo["order"]
            print(f"\nHIERARCHY (topological order)  has_cycle={topo['has_cycle']}")
            for pos, node_idx in enumerate(order):
                if node_idx < regs.n:
                    label = regs.label(node_idx)
                    kind = "REG "
                else:
                    label = procs.label(node_idx - regs.n)
                    kind = "PROC"
                print(f"  {pos:3d}  [{kind}]  {label}")
        else:
            print("No edges above threshold — cannot compute hierarchy")

    else:
        print(f"Unknown report type: {report_type!r}")
        print("Valid types: gaps | coverage | rankings | hierarchy")
        sys.exit(1)
