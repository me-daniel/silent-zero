"""Shared names, colours, order and fonts for all figures (one script per figure, like fig_*.py here).

Figures are drawn at FIG_W inches and shown at about the text width of the write-up (~46rem), so fonts are
set close to their on-screen size. Colours follow the Okabe-Ito palette (colour-blind safe), and
amber always means "closing tag missing". Row labels say what each build is, so a figure reads without the text.
"""
import json
import os
import sys

import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "generated")
sys.path.insert(0, os.path.dirname(HERE))
import build  # noqa: E402  (score files and helpers, same source as the text)

FIG_W = 8.0

# row labels: what each build is, readable without the text
LABELS = {
    "Q8_0": "Q8_0 (8-bit)",
    "Q4_K_M": "Q4_K_M (4-bit)",
    "IQ3_XXS": "P (3-bit, public imatrix)",
    "IQ3_XXS_R": "R (3-bit, public imatrix)",
    "IQ3_XXS_W": "W (3-bit, public calibration text)",
    "IQ3_XXS_V": "V (3-bit, public calibration text)",
    "IQ3_XXS_X": "X (3-bit, my text)",
    "IQ3_XXS_G": "G (3-bit, my text)",
    "IQ3_XXS_T": "T (3-bit, my text + tool calls)",
    "IQ2_M": "IQ2_M (2-bit)",
}

INK = "#1f1f1c"
MUTED = "#6b6a64"
RULE = "#d9d6cc"
GOOD = "#009e73"
BAD = "#d55e00"
WARN = "#e69f00"
SCORE = "#4d5d6c"      # a BFCL score
TAG = "#e69f00"        # "closing tag missing", the same colour in every figure

# kinds of malformed step (audit/classify.py classes, grouped), Okabe-Ito
KINDS = [
    ("closing tag missing", ["unclosed"], TAG),
    ("key names missing", ["keys_dropped"], "#009e73"),
    ("no tags at all", ["untagged_call"], "#56b4e9"),
    ("tool schema copied", ["schema_echo"], "#cc79a7"),
    ("other", ["bad_json", "no_name", "bad_name", "bad_args", "bad_newlines", "not_object"], "#b9b7af"),
]


one_decimal = build.one_decimal  # same rounding as the text


def value(ax, x, y, text):
    """A value label right of a bar, on a white background so grid lines don't cross it."""
    ax.text(x, y, text, va="center", fontsize=9, color=MUTED, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


def rows(ax, builds):
    """Horizontal rows top to bottom with the build labels, no y axis line."""
    ys = list(range(len(builds)))[::-1]
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[b] for b in builds])
    ax.set_ylim(-0.6, len(builds) - 0.4)
    ax.tick_params(axis="y", length=0, labelsize=9.5, labelcolor=INK)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    return ys


def setup():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",  # bundled with matplotlib, so the figures render the same anywhere
        "font.size": 10.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",  # text as outlines: the figure looks the same without the font
        "svg.hashsalt": "silent-zero",  # stable ids, so unchanged figures give unchanged files
        "savefig.facecolor": "white",
    })


def data(audit="data/audit", verify="data/verify"):
    audit, verify = os.path.join(ROOT, audit), os.path.join(ROOT, verify)
    A = json.load(open(os.path.join(audit, "audit.json")))
    PR = json.load(open(os.path.join(audit, "paired.json")))
    return A, PR, verify


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("svg", "png"):
        kw = {"dpi": 200} if ext == "png" else {"metadata": {"Date": None}}
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", **kw)
    print(f"wrote writeup/figures/generated/{name}.svg and .png")
