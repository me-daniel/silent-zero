"""Figure 3: simple_python score per build, and how much of it returns when the missing closing tag is tolerated.

One question: how much of each build's score loss is the missing tag? The grey bar is BFCL's official score,
the amber extension what the same saved outputs gain when re-scored with the tolerant parser
(patch/verify.py --mode tolerant), the dotted line the score of Q8_0. Same data as the text (writeup/build.py).

    .venv/bin/python -I writeup/figures/fig_3_scores.py  ->  writeup/figures/generated/fig_3_scores.{svg,png}
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

BUILDS = ["Q8_0", "IQ3_XXS"] + S.build.MINE


def main():
    S.setup()
    A, _, verify = S.data()
    fig, ax = plt.subplots(figsize=(S.FIG_W, 4.0))
    ys = S.rows(ax, BUILDS)
    for b in BUILDS:
        y = ys[b]
        off_acc = A[b]["simple_python"]["official"]["accuracy"]
        tol_acc = S.build.score_head(verify, "tolerant", b, "simple_python")["accuracy"]
        assert tol_acc >= off_acc, b
        ax.barh(y, 100 * off_acc, color=S.SCORE, height=0.6)
        ax.barh(y, 100 * (tol_acc - off_acc), left=100 * off_acc, color=S.TAG, height=0.6)
        gain = 100 * (tol_acc - off_acc)
        label = S.one_decimal(100 * off_acc) + (f"  +{S.one_decimal(gain)}" if gain >= 0.05 else "")
        S.value(ax, 100 * tol_acc + 1, y, label, b)
    q8 = 100 * A["Q8_0"]["simple_python"]["official"]["accuracy"]  # reference: the 8-bit score
    ax.axvline(q8, color=S.INK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(q8 + 0.8, ys["Q8_0"] + 0.85, "Q8_0 score", fontsize=8.5, color=S.MUTED, va="center")
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel(f"simple_python accuracy (%), {A['Q8_0']['simple_python']['items']} items")
    h = [plt.Rectangle((0, 0), 1, 1, color=S.SCORE), plt.Rectangle((0, 0), 1, 1, color=S.TAG)]
    ax.legend(h, ["BFCL score", "added when the missing closing tag is tolerated"], loc="upper center",
              bbox_to_anchor=(0.4, -0.15), ncol=2, frameon=False, fontsize=9, handlelength=1.0)
    fig.tight_layout()
    S.save(fig, "fig_3_scores")
    plt.close(fig)


if __name__ == "__main__":
    main()
