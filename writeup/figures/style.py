"""Shared names, colours, order and fonts for all figures (one script per figure, like fig_*.py here).

Figures are drawn at FIG_W inches and shown at about the text width of the write-up (~46rem), so fonts are
set close to their on-screen size. Colours follow the Okabe-Ito palette (colour-blind safe). The public
imatrix (P, R, and the other public builds) is orange, my own imatrices are blue, so the two groups read the
same way in every figure.
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

# label shown in the figures, in display order
NAMES = {
    "Q8_0": "Q8_0", "Q4_K_M": "Q4_K_M", "IQ3_XXS": "P", "IQ3_XXS_R": "R", "IQ2_M": "IQ2_M",
    "IQ3_XXS_W": "W", "IQ3_XXS_V": "V", "IQ3_XXS_X": "X", "IQ3_XXS_G": "G", "IQ3_XXS_T": "T",
}
ORDER = list(NAMES)

INK = "#1f1f1c"
MUTED = "#6b6a64"
RULE = "#d9d6cc"
PUBLIC = "#d55e00"     # public imatrix
PUBLIC_LIGHT = "#f2b48a"
OWN = "#0072b2"        # own imatrices
OWN_LIGHT = "#9cc7e4"
REF = "#555555"        # 8-bit reference
GOOD = "#009e73"
BAD = "#d55e00"
WARN = "#e69f00"

COLOURS = {"Q8_0": REF, "Q4_K_M": PUBLIC, "IQ2_M": PUBLIC, "IQ3_XXS": PUBLIC, "IQ3_XXS_R": PUBLIC}
COLOURS.update({b: OWN for b in ("IQ3_XXS_W", "IQ3_XXS_V", "IQ3_XXS_X", "IQ3_XXS_G", "IQ3_XXS_T")})

# malformed-call classes in a fixed order and colour (Okabe-Ito, avoiding the two build-group colours)
CLASSES = [
    ("unclosed", "unclosed tag", "#e69f00"),
    ("keys_dropped", "key names dropped", "#009e73"),
    ("untagged_call", "no tags at all", "#56b4e9"),
    ("schema_echo", "schema copied", "#cc79a7"),
    ("bad_json", "other", "#999999"),
    ("no_name", "no usable name", "#f0e442"),
    ("bad_name", "no usable name", "#f0e442"),
    ("bad_args", "other", "#999999"),
    ("bad_newlines", "other", "#999999"),
    ("not_object", "other", "#999999"),
]


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
