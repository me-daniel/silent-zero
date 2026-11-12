"""Figure 3: multi-turn (multi_turn_base) per build.

Left: malformed model steps by kind (audit/classify.py classes, grouped), with the number of tasks that
crashed. Right: the tasks that Q8_0 passes and the build fails, split by whether they contain a malformed call
or crash, and otherwise by BFCL's error type. Same data as the text (data/audit/audit.json).

    .venv/bin/python -I writeup/figures/fig_3_multiturn.py  ->  writeup/figures/generated/fig_3_multiturn.{svg,png}
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

BUILDS = ["Q8_0", "Q4_K_M", "IQ3_XXS", "IQ3_XXS_R", "IQ3_XXS_G", "IQ3_XXS_T", "IQ2_M"]
CAUSES = [  # (key, label, colour): greys are failures a parser cannot fix
    ("broken", "malformed call or crash", "#1f1f1c"),
    ("empty_turn_model_response", "turn ends without a call", "#8a8984"),
    ("instance_state_mismatch", "wrong final state", "#b9b7af"),
    ("execution_response_mismatch", "wrong results", "#dedcd4"),
]


def main():
    S.setup()
    A, PR, _ = S.data()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.FIG_W, 3.9), sharey=True,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    ys = list(range(len(BUILDS)))[::-1]

    # left: malformed steps by kind
    groups = []
    for key, label, col in S.CLASSES:
        if label not in [g[0] for g in groups]:
            groups.append((label, col, []))
        next(g for g in groups if g[0] == label)[2].append(key)
    used = set()
    xmax = 0
    for y, b in zip(ys, BUILDS):
        m = A[b]["multi_turn_base"]
        x = 0
        for label, col, keys in groups:
            v = sum(m["classes"].get(k, 0) for k in keys)
            if v:
                ax1.barh(y, v, left=x, color=col, height=0.62, edgecolor="white", lw=0.6)
                used.add(label)
                x += v
        crashes = len(m["crashed_tasks"])
        note = f"{x}" + (f", {crashes} tasks crashed" if crashes else "")
        ax1.text(x + 2, y, note, va="center", fontsize=8.5, color=S.MUTED)
        xmax = max(xmax, x)
    ax1.set_xlim(0, xmax * 1.45)
    ax1.set_title("Malformed steps by kind", loc="left", fontsize=10.5)
    ax1.set_xlabel("steps")
    h1 = [plt.Rectangle((0, 0), 1, 1, color=col) for label, col, _ in groups if label in used]
    ax1.legend(h1, [label for label, _, _ in groups if label in used], loc="upper center",
               bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False, fontsize=8.5, handlelength=1.0)

    # right: tasks Q8_0 passes and the build fails
    q8 = set(A["Q8_0"]["multi_turn_base"]["passed_ids"])
    for y, b in zip(ys, BUILDS):
        if b == "Q8_0":
            ax2.text(0.5, y, "reference", va="center", fontsize=8.5, color=S.MUTED, style="italic")
            continue
        m = A[b]["multi_turn_base"]
        new = q8 & set(m["failed_ids"])
        broken = new & set(m["malformed_or_crash_ids"])
        assert (len(new), len(broken)) == (PR[b]["new_failures"], PR[b]["new_failures_with_broken_call"]), b
        assert 2 * len(broken) < len(new), (b, "caption: broken calls are a minority of new failures")
        counts = {"broken": len(broken)}
        for i in new - broken:
            k = m["failed_error_types"][i].split(":")[-1]
            assert k in dict((c[0], 1) for c in CAUSES), (b, k)
            counts[k] = counts.get(k, 0) + 1
        x = 0
        for key, _, col in CAUSES:
            v = counts.get(key, 0)
            if v:
                ax2.barh(y, v, left=x, color=col, height=0.62, edgecolor="white", lw=0.6)
                x += v
        ax2.text(x + 0.8, y, f"{len(broken)} of {len(new)}", va="center", fontsize=8.5, color=S.MUTED)
    ax2.set_xlim(0, 52)
    ax2.set_title("Tasks Q8_0 passes and the build fails", loc="left", fontsize=10.5)
    ax2.set_xlabel("tasks")
    h2 = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in CAUSES]
    ax2.legend(h2, [label for _, label, _ in CAUSES], loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2,
               frameon=False, fontsize=8.5, handlelength=1.0)

    for ax in (ax1, ax2):
        ax.grid(axis="x", color=S.RULE, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax1.set_ylim(-0.55, len(BUILDS) - 0.3)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([S.NAMES[b] for b in BUILDS])
    for t, b in zip(ax1.get_yticklabels(), BUILDS):
        t.set_color(S.COLOURS[b])
        t.set_fontweight("bold")
    fig.tight_layout(w_pad=2.5)
    S.save(fig, "fig_3_multiturn")
    plt.close(fig)


if __name__ == "__main__":
    main()
