"""Figure 1: what BFCL's Qwen parser does with four kinds of model output.

Rows are output shapes seen in the saved outputs (arguments shortened). Columns: what the model writes, what
the parser does (upstream `_extract_tool_calls` at 6ea5797), and what BFCL records. Blue lines under an
outcome say what changes with the patch's opt-in mode for that kind (patch/README.md), off by default.
No data numbers in this figure.

    .venv/bin/python -I writeup/figures/fig_1_parser.py   ->  writeup/figures/generated/fig_1_parser.{svg,png}
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

PATCH = "#0072b2"
ROWS = [
    ("Well formed", ["<tool_call>", '{"name": "ls", "arguments": {"a": true}}', "</tool_call>"],
     ("parsed", S.GOOD), ["checked against the", "expected call"], []),
    ("Closing tag missing", ["<tool_call>", '{"name": "ls", "arguments": {"a": true}}', ""],
     ("no match:\ndropped", S.BAD), ["scored as if the model", "called nothing"],
     ["tolerant mode: parsed and", "checked like row 1"]),
    ('"name" key missing', ["<tool_call>", '{"cd", "arguments": {"folder": "x"}}', "</tool_call>"],
     ("invalid JSON:\ndropped", S.BAD), ["scored as if the model", "called nothing"],
     ["key repair: rebuilt as a call,", "then checked like row 1"]),
    ("Schema copied", ["<tool_call>", '{"name": "ls", "description": "...",', ' "parameters": {...}}', "</tool_call>"],
     ("valid JSON:\nkept", "#b07800"), ["multi-turn: the next prompt", "raises, the whole task is lost"],
     ["safe history: the call is skipped,", "the task continues"]),
]


def box(ax, x, y, w, h, fc, ec, lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))


def arrow(ax, x1, y1, x2, y2, color=S.MUTED, ls="-", rad=0.0, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


def main():
    S.setup()
    W, H = S.FIG_W, 4.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    cx, cw = 0.05, 3.25          # code boxes
    px, pw = 3.75, 1.25          # parser chips
    ox = 5.45                    # outcome text
    rh, gap, top = 0.9, 0.16, H - 0.5

    for x, t in ((cx, "What the model writes"), (px - 0.05, "BFCL's parser"), (ox, "What BFCL records")):
        ax.text(x, H - 0.22, t, fontsize=11, fontweight="bold", va="center")

    ys = []
    ax.text(ox, H - 0.43, "blue: with the patch's opt-in mode", fontsize=8, color=PATCH, style="italic",
            va="center")
    for i, (label, code, (chip, col), outcome, patched) in enumerate(ROWS):
        y0 = top - (i + 1) * rh - i * gap
        yc = y0 + rh / 2
        ys.append(yc)
        box(ax, cx, y0, cw, rh, "#f6f5f0", S.RULE)
        ax.text(cx + 0.1, y0 + rh - 0.13, f"{i + 1}  {label}", fontsize=9.5, fontweight="bold", va="center")
        lines = [l for l in code]
        step = 0.155
        for j, line in enumerate(lines):
            ax.text(cx + 0.1, y0 + rh - 0.32 - j * step, line, family="DejaVu Sans Mono", fontsize=7.6,
                    va="center", color=S.INK)
        if label == "Closing tag missing":
            ax.text(cx + 0.1, y0 + rh - 0.32 - 2 * step, "(output ends here)", fontsize=7.6, style="italic",
                    va="center", color=S.BAD)
        box(ax, px, yc - 0.27, pw, 0.54, "white", col, lw=1.6)
        ax.text(px + pw / 2, yc, chip, ha="center", va="center", fontsize=9, color=col, fontweight="bold",
                linespacing=1.15)
        arrow(ax, cx + cw + 0.04, yc, px - 0.04, yc)
        arrow(ax, px + pw + 0.04, yc, ox - 0.08, yc)
        oy = yc + (0.22 if patched else 0.09)  # outcome text, moved up to make room for the patched line
        ax.text(ox, oy, outcome[0], fontsize=9.5, va="center")
        ax.text(ox, oy - 0.2, outcome[1], fontsize=9.5, va="center")
        for j, line in enumerate(patched):
            ax.text(ox, yc - 0.2 - j * 0.16, line, fontsize=8.5, color=PATCH, style="italic", va="center")

    S.save(fig, "fig_1_parser")
    plt.close(fig)


if __name__ == "__main__":
    main()
