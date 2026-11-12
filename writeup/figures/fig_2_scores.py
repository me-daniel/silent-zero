"""Figure 2: official and tolerant single-turn scores per build.

One row per build. The open circle is BFCL's official score, the filled circle the score of the same saved
outputs re-scored in tolerant mode (patch/verify.py). Orange: public imatrix, blue: own imatrices, grey: 8-bit.
Scores come from the same files as the write-up (writeup/build.py).

    .venv/bin/python -I writeup/figures/fig_2_scores.py   ->  writeup/figures/generated/fig_2_scores.{svg,png}
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

BUILDS = ["Q8_0", "IQ3_XXS", "IQ3_XXS_R", "IQ3_XXS_W", "IQ3_XXS_V", "IQ3_XXS_X", "IQ3_XXS_G", "IQ3_XXS_T"]
CATS = [("simple_python", "simple_python (400 items)"), ("live_simple", "live_simple (258 items)")]


def main():
    S.setup()
    A, _, verify = S.data()
    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, 3.6), sharey=True)
    ys = list(range(len(BUILDS)))[::-1]
    for ax, (cat, title) in zip(axes, CATS):
        assert A["IQ3_XXS"][cat]["items"] == int(title.split("(")[1].split()[0]), title
        lo, hi = 100, 0
        for y, b in zip(ys, BUILDS):
            off_acc = A[b][cat]["official"]["accuracy"]
            tol_acc = S.build.score_head(verify, "tolerant", b, cat)["accuracy"]
            off, tol = 100 * off_acc, 100 * tol_acc
            lo, hi = min(lo, off), max(hi, tol)
            c = S.COLOURS[b]
            ax.plot([off, tol], [y, y], color=c, lw=2.2, alpha=0.45, solid_capstyle="round", zorder=1)
            ax.scatter([off], [y], s=46, facecolor="white", edgecolor=c, lw=1.6, zorder=3)
            ax.scatter([tol], [y], s=46, color=c, zorder=4)
            if tol - off >= 5:  # same rounding as writeup/build.py
                ax.text((off + tol) / 2, y + 0.22, f"+{100 * (tol_acc - off_acc):.1f}", ha="center", va="bottom", fontsize=8.5,
                        color=c)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlim(lo - 3, hi + 3)
        ax.grid(axis="x", color=S.RULE, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("accuracy (%)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([S.NAMES[b] for b in BUILDS])
    for t, b in zip(axes[0].get_yticklabels(), BUILDS):
        t.set_color(S.COLOURS[b])
        t.set_fontweight("bold")
    axes[0].set_ylim(-0.6, len(BUILDS) - 0.3)

    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=S.MUTED, mew=1.4, ms=7, label="official"),
         plt.Line2D([], [], marker="o", ls="", color=S.MUTED, ms=7, label="tolerant"),
         plt.Line2D([], [], marker="s", ls="", color=S.REF, ms=7, label="8-bit"),
         plt.Line2D([], [], marker="s", ls="", color=S.PUBLIC, ms=7, label="public imatrix"),
         plt.Line2D([], [], marker="s", ls="", color=S.OWN, ms=7, label="own imatrix")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, "fig_2_scores")
    plt.close(fig)


if __name__ == "__main__":
    main()
