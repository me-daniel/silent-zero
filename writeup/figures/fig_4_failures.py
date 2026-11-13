"""Figure 4: why the multi-turn tasks fail that the 8-bit build passes.

One question: are the new multi-turn failures parsing failures? For each build, the tasks Q8_0 passes and the
build fails, split into tasks with a malformed call or crash (audit/classify.py) and, for the rest, BFCL's
error type: the model stops without a call, or its calls run but the outcome is wrong (final state or
results). Same data as the text (data/audit/audit.json, paired.json).

    .venv/bin/python -I writeup/figures/fig_4_failures.py  ->  writeup/figures/generated/fig_4_failures.{svg,png}
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

BUILDS = ["Q4_K_M", "IQ3_XXS", "IQ2_M", "IQ3_XXS_R", "IQ3_XXS_G", "IQ3_XXS_T"]
CAUSES = [  # (label, BFCL error types, colour)
    ("contains a malformed call or a crash", None, S.BAD),
    ("model stops without a call", {"empty_turn_model_response"}, "#8a8984"),
    ("calls run, wrong outcome", {"instance_state_mismatch", "execution_response_mismatch"}, "#cfccc3"),
]


def main():
    S.setup()
    A, PR, _ = S.data()
    fig, ax = plt.subplots(figsize=(S.FIG_W, 3.0))
    ys = S.rows(ax, BUILDS)
    q8 = set(A["Q8_0"]["multi_turn_base"]["passed_ids"])
    for b in BUILDS:
        y = ys[b]
        m = A[b]["multi_turn_base"]
        new = q8 & set(m["failed_ids"])
        broken = new & set(m["malformed_or_crash_ids"])
        assert (len(new), len(broken)) == (PR[b]["new_failures"], PR[b]["new_failures_with_broken_call"]), b
        assert 2 * len(broken) < len(new), (b, "caption: a minority of new failures has a broken call")
        types = [m["failed_error_types"][i].split(":")[-1] for i in new - broken]
        counts = [len(broken)] + [sum(t in ts for t in types) for _, ts, _ in CAUSES[1:]]
        assert sum(counts) == len(new), (b, set(types))
        x = 0
        for v, (_, _, col) in zip(counts, CAUSES):
            if v:
                ax.barh(y, v, left=x, color=col, height=0.6, edgecolor="white", lw=0.6)
                x += v
        S.value(ax, x + 0.6, y, f"{len(broken)} of {len(new)} with a malformed call", b)
    ax.set_xlim(0, 64)
    ax.set_xlabel("tasks that Q8_0 passes and this build fails")
    h = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in CAUSES]
    ax.legend(h, [k for k, _, _ in CAUSES], loc="upper center", bbox_to_anchor=(0.4, -0.24), ncol=3,
              frameon=False, fontsize=9, handlelength=1.0)
    fig.tight_layout()
    S.save(fig, "fig_4_failures")
    plt.close(fig)


if __name__ == "__main__":
    main()
