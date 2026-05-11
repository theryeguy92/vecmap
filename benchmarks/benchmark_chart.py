#!/usr/bin/env python3
"""
Generate benchmarks/benchmark_chart.png — cosine kernel comparison bar chart.

RTX 5080 (Blackwell sm_120), 10,000 × 500 × 768 dims.
Run from the project root: python3 benchmarks/benchmark_chart.py
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_PATH = Path(__file__).resolve().parent / "benchmark_chart.png"

kernels = [
    "cosine_naive\n(baseline)",
    "cosine_tiled\n(shared memory)",
    "cosine_warp\n(warp shuffle)",
]
times = [9.51, 6.88, 3.99]
speedups = [1.00, 1.38, 2.38]
colours = ["#c0392b", "#e67e22", "#27ae60"]

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

x = np.arange(len(kernels))
bar_w = 0.52
bars = ax.bar(
    x, times, width=bar_w, color=colours, zorder=3, linewidth=0, edgecolor="none"
)

# Grid lines — horizontal only, subtle
ax.yaxis.grid(True, color="#2a2d38", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[:].set_visible(False)
ax.tick_params(colors="#9aa0b2", labelsize=11)

# Axis labels
ax.set_xticks(x)
ax.set_xticklabels(kernels, color="#d0d4e0", fontsize=11, linespacing=1.5)
ax.set_ylabel("Kernel time (ms)", color="#9aa0b2", fontsize=11, labelpad=10)
ax.set_ylim(0, 11.8)
ax.yaxis.set_tick_params(color="#9aa0b2")

# Bar annotations: time + speedup badge
for i, (bar, t, s) in enumerate(zip(bars, times, speedups)):
    bx = bar.get_x() + bar.get_width() / 2
    by = bar.get_height()

    # Time label above bar
    ax.text(
        bx,
        by + 0.22,
        f"{t:.2f} ms",
        ha="center",
        va="bottom",
        color="#e8ecf5",
        fontsize=12,
        fontweight="bold",
    )

    # Speedup badge inside bar (bottom-aligned)
    badge = f"{s:.2f}×" if s > 1.0 else "baseline"
    badge_colour = "#ffffff" if s == 1.0 else "#ffffff"
    ax.text(
        bx,
        0.35,
        badge,
        ha="center",
        va="bottom",
        color=badge_colour,
        fontsize=10,
        fontweight="bold",
        alpha=0.85,
    )

# Title and subtitle
ax.set_title(
    "CUDA Cosine Similarity Kernels",
    color="#e8ecf5",
    fontsize=15,
    fontweight="bold",
    pad=14,
)
fig.text(
    0.5,
    0.91,
    "RTX 5080 (Blackwell sm_120)  ·  10,000 × 500 × 768 dimensions",
    ha="center",
    color="#9aa0b2",
    fontsize=10,
)

# Legend patches
patches = [
    mpatches.Patch(color=c, label=f"{k.splitlines()[0]}  {t:.2f} ms  ({s:.2f}×)")
    for k, t, s, c in zip(kernels, times, speedups, colours)
]
ax.legend(
    handles=patches,
    loc="upper right",
    facecolor="#1a1d26",
    edgecolor="#2a2d38",
    labelcolor="#c8ccd8",
    fontsize=9.5,
    framealpha=0.9,
    handlelength=1.2,
    handleheight=1.1,
    borderpad=0.8,
)

# Throughput callout bottom-right
fig.text(
    0.96,
    0.04,
    "553M pairs/sec  ·  full pipeline ~12 ms",
    ha="right",
    color="#27ae60",
    fontsize=9.5,
    fontstyle="italic",
)

plt.tight_layout(rect=[0, 0.0, 1, 0.90])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT_PATH}")
