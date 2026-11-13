"""Shared names, colours, order and fonts for all figures (one script per figure, like fig_*.py here).

Figures are drawn at FIG_W inches and shown at about the text width of the write-up (~46rem), so fonts are
set close to their on-screen size. Colours follow the Okabe-Ito palette (colour-blind safe), and
amber always means "closing tag missing". Rows are grouped into public builds and mine, with the same names as in the text.
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

# row labels: the public builds by file name and bits, mine by their name in the text (writeup/build.NAMES),
# set in monospace like in the text
NAMES = build.NAMES
BITS = {"Q8_0": 8, "Q4_K_M": 4, "IQ3_XXS": 3, "IQ2_M": 2}
GROUP_PUBLIC = "Public builds"
GROUP_MINE = "My builds (IQ3_XXS, 3-bit)"
MINE_BAND = "#f3f1ea"  # light background behind my builds, so the two groups read apart at a glance


def label(b):
    return f"{b} ({BITS[b]}-bit)" if b in BITS else NAMES[b]


INK = "#1f1f1c"
MUTED = "#6b6a64"
RULE = "#d9d6cc"
GOOD = "#009e73"
BAD = "#c0392b"  # red: a broken call (distinct from the amber of a missing tag)
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


def value(ax, x, y, text, b=None):
    """A value label right of a bar, on the row's background so grid lines don't cross it."""
    bg = MINE_BAND if b in build.MINE else "white"
    ax.text(x, y, text, va="center", fontsize=9, color=MUTED, zorder=5,
            bbox=dict(facecolor=bg, edgecolor="none", pad=1.0))


def rows(ax, builds):
    """Horizontal rows top to bottom in two groups, public builds and mine, each under a bold heading.

    Returns {build: y}. My builds sit on a light band, and their names are in monospace like in the text.
    """
    groups = [(GROUP_PUBLIC, [b for b in builds if b in build.PUBLIC]),
              (GROUP_MINE, [b for b in builds if b in build.MINE])]
    assert sum(len(g) for _, g in groups) == len(builds), builds
    ticks, labels, ys, y = [], [], {}, 0.0
    for title, bs in groups:
        if not bs:
            continue
        y -= 0.15 if ticks else 0
        ticks.append(y)
        labels.append(title)
        y -= 0.85
        top = y + 0.5
        for b in bs:
            ys[b] = y
            ticks.append(y)
            labels.append(label(b))
            y -= 1
        if title == GROUP_MINE:
            ax.axhspan(y + 0.5, top + 0.85, color=MINE_BAND, zorder=0, lw=0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, labelsize=9.5, labelcolor=INK)
    for t, text in zip(ax.get_yticklabels(), labels):
        if text in (GROUP_PUBLIC, GROUP_MINE):
            t.set_fontweight("bold")
            t.set_color(MUTED)
            t.set_fontsize(9)
        elif text in NAMES.values() and text not in BITS:
            t.set_fontfamily("DejaVu Sans Mono")
            t.set_fontsize(9)
    ax.set_ylim(y + 0.5, 0.5)
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
