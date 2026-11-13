"""Figure 1: what BFCL's Qwen parser does with four kinds of model output.

Rows are output shapes seen in the saved outputs (arguments shortened). Columns: what the model writes, what
the parser does (upstream `_extract_tool_calls` at 6ea5797), what BFCL records by default, and, in the blue
column, what changes with the patch's opt-in mode for that kind (patch/README.md), off by default.
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
PATCH_BAND = "#eaf3f9"  # light blue behind the opt-in column, so it reads apart from the default outcome
KEPT = "#b07800"
TINT = {S.GOOD: "#e9f6f1", S.BAD: "#fbeceb", KEPT: "#fbf3e0"}  # chip fills, a light tint of the edge colour
# label, code lines, (parser's reason, verdict, colour), default outcome (headline, detail lines),
# opt-in mode (name, what it changes) or None
ROWS = [
    ("Well formed", ["<tool_call>", '{"name": "ls", "arguments": {"a": true}}', "</tool_call>"],
     ("match, valid JSON", "parsed", S.GOOD), ("Checked", ["against the expected call"]), None),
    ("Closing tag missing", ["<tool_call>", '{"name": "ls", "arguments": {"a": true}}', ""],
     ("no match", "dropped", S.BAD), ("Scored as no call", ["as if the model", "called nothing"]),
     ("tolerant mode", ["parsed and checked", "like row 1"])),
    ('"name" key missing', ["<tool_call>", '{"cd", "arguments": {"folder": "x"}}', "</tool_call>"],
     ("invalid JSON", "dropped", S.BAD), ("Scored as no call", ["as if the model", "called nothing"]),
     ("key repair", ["rebuilt as a call, then", "checked like row 1"])),
    ("Schema copied", ["<tool_call>", '{"name": "ls", "description": "...",', ' "parameters": {...}}', "</tool_call>"],
     ("valid JSON", "kept", KEPT), ("Whole task lost", ["multi-turn: the next", "prompt raises"]),
     ("safe history", ["the call is skipped,", "the task continues"])),
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

    cx, cw = 0.05, 2.72          # code boxes
    px, pw = 3.02, 1.12          # parser chips
    ox = 4.42                    # default outcome text
    bx, bw = 6.2, W - 6.2 - 0.03  # opt-in band
    mx = bx + 0.13               # opt-in text
    rh, gap, top = 0.9, 0.16, H - 0.62
    bottom = top - len(ROWS) * rh - (len(ROWS) - 1) * gap

    # the opt-in column sits on one blue band from its heading to the last row
    ax.add_patch(FancyBboxPatch((bx, bottom - 0.05), bw, H - 0.04 - (bottom - 0.05),
                                boxstyle="round,pad=0,rounding_size=0.08", fc=PATCH_BAND, ec="none", zorder=1))

    heads = ((cx, "What the model writes", "four kinds of output", S.INK, S.MUTED),
             (px - 0.02, "BFCL's parser", "regex, then json.loads", S.INK, S.MUTED),
             (ox, "What BFCL records", "by default", S.INK, S.MUTED),
             (mx, "With the patch", "opt-in modes, off by default", PATCH, PATCH))
    for x, t, sub, c1, c2 in heads:
        ax.text(x, H - 0.22, t, fontsize=10.5, fontweight="bold", va="center", color=c1)
        ax.text(x, H - 0.43, sub, fontsize=8, va="center", color=c2)

    for i, (label, code, (why, verdict, col), (out, detail), mode) in enumerate(ROWS):
        y0 = top - (i + 1) * rh - i * gap
        yc = y0 + rh / 2
        if i:  # a thin rule between rows, across the outcome columns
            ax.plot([px, W - 0.03], [y0 + rh + gap / 2] * 2, color=S.RULE, lw=0.6, zorder=1.5)
        box(ax, cx, y0, cw, rh, "#f6f5f0", S.RULE)
        ax.text(cx + 0.1, y0 + rh - 0.13, f"{i + 1}  {label}", fontsize=9.5, fontweight="bold", va="center")
        step = 0.155
        for j, line in enumerate(code):
            ax.text(cx + 0.1, y0 + rh - 0.32 - j * step, line, family="DejaVu Sans Mono", fontsize=7.6,
                    va="center", color=S.INK)
        if label == "Closing tag missing":
            ax.text(cx + 0.1, y0 + rh - 0.32 - 2 * step, "(output ends here)", fontsize=7.6, style="italic",
                    va="center", color=S.BAD)
        # parser chip: the reason in small type, the verdict in bold
        box(ax, px, yc - 0.27, pw, 0.54, TINT[col], col, lw=1.6)
        ax.text(px + pw / 2, yc + 0.11, why, ha="center", va="center", fontsize=8, color=col)
        ax.text(px + pw / 2, yc - 0.09, verdict, ha="center", va="center", fontsize=10.5, color=col,
                fontweight="bold")
        arrow(ax, cx + cw + 0.05, yc, px - 0.04, yc)
        arrow(ax, px + pw + 0.05, yc, ox - 0.08, yc)

        def cell(x, headline, lines, colour):
            n = len(lines)
            ytop = yc + 0.09 * n
            ax.text(x, ytop, headline, fontsize=9.5, fontweight="bold", va="center", color=colour)
            for j, line in enumerate(lines):
                ax.text(x, ytop - 0.2 - j * 0.17, line, fontsize=8.8, va="center", color=S.INK)

        cell(ox, out, detail, S.INK)
        if mode:
            cell(mx, mode[0], mode[1], PATCH)
        else:
            ax.text(mx, yc, "no change", fontsize=8.8, va="center", color=S.MUTED)

    S.save(fig, "fig_1_parser")
    plt.close(fig)


if __name__ == "__main__":
    main()
