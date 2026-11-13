"""Figure 2: malformed model steps in multi_turn_base, by kind, per build.

One question: which kinds of broken call does each build write in multi-turn? Classes from audit/classify.py,
grouped (style.KINDS). Same data as the text (data/audit/audit.json).

    .venv/bin/python -I writeup/figures/fig_2_malformed.py  ->  writeup/figures/generated/fig_2_malformed.{svg,png}
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

BUILDS = ["Q8_0", "Q4_K_M", "IQ3_XXS", "IQ2_M", "IQ3_XXS_R", "IQ3_XXS_G", "IQ3_XXS_T"]


def main():
    S.setup()
    A, _, _ = S.data()
    fig, ax = plt.subplots(figsize=(S.FIG_W, 3.2))
    ys = S.rows(ax, BUILDS)
    known = {c for _, cs, _ in S.KINDS for c in cs}
    xmax = 0
    for b in BUILDS:
        y = ys[b]
        cl = A[b]["multi_turn_base"]["classes"]
        assert not (set(cl) - known - {"ok", "text"}), (b, set(cl) - known)
        x = 0
        for _, cs, col in S.KINDS:
            v = sum(cl.get(c, 0) for c in cs)
            if v:
                ax.barh(y, v, left=x, color=col, height=0.6, edgecolor="white", lw=0.6)
                x += v
        S.value(ax, x + 1.5, y, str(x), b)
        xmax = max(xmax, x)
    ax.set_xlim(0, xmax * 1.1)
    ax.set_xlabel(f"malformed steps in multi_turn_base ({A['Q8_0']['multi_turn_base']['tasks']} tasks)")
    h = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in S.KINDS]
    ax.legend(h, [k for k, _, _ in S.KINDS], loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=5,
              frameon=False, fontsize=9, handlelength=1.0, columnspacing=1.2)
    fig.tight_layout()
    S.save(fig, "fig_2_malformed")
    plt.close(fig)


if __name__ == "__main__":
    main()
