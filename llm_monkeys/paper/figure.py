"""Figure 1 of the paper: pass@k against k for the six SLMs, on both datasets.

    python -m paper.figure results/paper_tables.json --out figure_pass_at_k

Draws the "coverage" part of paper.tables' output and writes a PDF and a PNG.
The dashed line is Gemini-3.8-Flash's one-shot accuracy as the paper quotes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TITLES = {"med_qa": "MedQA", "medbullets": "Medbullets"}
# The values the paper's text quotes: the author's run on MedQA, ours on
# Medbullets, where the author has none.
REFERENCE = {"med_qa": 87.0, "medbullets": 75.8}
# The paper's model order, in the first six categorical colours of the palette.
COLORS = {"Gemma 4 26B": "#2a78d6", "GPT-oss 120B": "#eb6834", "GPT-oss 20B": "#1baf7a",
          "Qwen 3.6 27B": "#eda100", "Qwen 3.5 9B": "#e87ba4", "Qwen 3.5 4B": "#008300"}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e4e3de", "#8f8e88"
ARIAL = Path("/System/Library/Fonts/Supplemental/Arial.ttf")


def draw(coverage: dict[str, dict[str, list[float]]], out: Path) -> list[Path]:
    if ARIAL.is_file():
        font_manager.fontManager.addfont(str(ARIAL))
        plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "text.color": INK})
    fig, axes = plt.subplots(1, len(TITLES), figsize=(7.0, 3.2), sharey=True)
    for ax, dataset in zip(axes, TITLES):
        for label, color in COLORS.items():
            curve = coverage.get(dataset, {}).get(label)
            if curve:
                ax.plot(range(1, len(curve) + 1), [v * 100 for v in curve], color=color, lw=1.8)
        ax.axhline(REFERENCE[dataset], color=INK2, lw=1.0, ls=(0, (4, 3)))
        ax.text(95, REFERENCE[dataset] - 1.0, "Gemini-3.8-Flash, one shot", fontsize=7, color=INK2,
                va="top", ha="right")
        ticks = [1, 2, 5, 10, 20, 50, 100]
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(1, 100)
        ax.set_ylim(40, 100)
        ax.set_title(TITLES[dataset], loc="left")
        ax.set_xlabel("number of samples k")
    axes[0].set_ylabel("pass@k, %")
    handles = [Line2D([], [], color=c, lw=1.8, label=label) for label, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(COLORS), fontsize=7.5,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.5)
    paths = [out.with_suffix(".pdf"), out.with_suffix(".png")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tables", type=Path, help="the JSON paper.tables wrote")
    parser.add_argument("--out", type=Path, default=Path("figure_pass_at_k"),
                        help="output path without extension")
    args = parser.parse_args(argv)
    for path in draw(json.loads(args.tables.read_text())["coverage"], args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
