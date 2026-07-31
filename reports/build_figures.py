"""The two summary figures the report and deck need but the notebooks never produced.

Everything else in reports/figures/ was written by the notebook that measured it. These
two synthesise across rounds, so they belong to the report rather than to any one
notebook, and they are built from the same ledgers via reports/facts.py.

    python reports/build_figures.py
"""

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import facts  # noqa: E402


def journey():
    """Public Macro F1 across the milestone submissions, with the noise band."""
    labels = [m[0] for m in facts.MILESTONES]
    scores = [m[1] for m in facts.MILESTONES]
    x = np.arange(len(scores))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    # The band is drawn around the FINAL score: any point inside it is a tie with the
    # best, which is the judgement every round in this project was held to.
    ax.axhspan(scores[-1] - facts.NOISE_FLOOR, scores[-1] + facts.NOISE_FLOOR,
               color="tab:green", alpha=0.10,
               label=f"+/- {facts.NOISE_FLOOR} noise floor around the best")
    ax.plot(x, scores, "o-", lw=2, ms=8, color="tab:blue")

    for i, (xi, s) in enumerate(zip(x, scores)):
        gain = s - scores[i - 1] if i else 0.0
        note = f"{s:.5f}" + (f"\n{gain:+.4f}" if i else "")
        ax.annotate(note, (xi, s), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(": ", ":\n") for l in labels], fontsize=8,
                       rotation=20, ha="right")
    ax.set_ylabel("Kaggle public Macro F1")
    # Not "representation, then calibration": the first calibration step is the largest
    # single jump on this curve, so the honest headline is that the decision rule and the
    # input did the work while the model did none.
    ax.set_title("Where the score came from: the decision rule and the input, "
                 "never the model", fontsize=11)
    ax.set_ylim(min(scores) - 0.012, max(scores) + 0.016)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = facts.FIGURES / "kaggle_journey.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return out


def local_vs_kaggle():
    """Which local protocol tracked the leaderboard, in level and in ranking."""
    rows = [
        # (label, standard CV, grouped CV, Kaggle at that model's shipped threshold)
        ("Supplied TF-IDF\nLightGBM", 0.7391, 0.6910, facts.SUPPLIED_LGBM_KAGGLE),
        ("Raw text\nall blocks", 0.8811, 0.7864, 0.77349),
        ("Raw text\nablation-chosen", facts.BEST_CV_STANDARD, facts.BEST_CV_GROUPED,
         0.77942),
    ]
    labels = [r[0] for r in rows]
    std = np.array([r[1] for r in rows])
    grp = np.array([r[2] for r in rows])
    kag = np.array([r[3] for r in rows])
    x = np.arange(len(rows))
    w = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    ax.bar(x - w, std, w, label="standard 5-fold CV", color="tab:gray")
    ax.bar(x, grp, w, label="grouped CV (held-out domain)", color="tab:blue")
    ax.bar(x + w, kag, w, label="Kaggle public", color="tab:green")
    for xi, (s, g, k) in zip(x, zip(std, grp, kag)):
        for off, v in [(-w, s), (0, g), (w, k)]:
            ax.text(xi + off, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Macro F1"); ax.set_ylim(0.6, 0.95)
    # Not "grouped CV is unbiased": it under-predicts the supplied-feature model by
    # 0.045. The defensible claim is that its error is several times smaller.
    ax.set_title("Grouped CV lands far closer to Kaggle than standard CV",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # The same information as an error: how far each local protocol sat from Kaggle.
    ax = axes[1]
    ax.bar(x - w / 2, std - kag, w, label="standard CV minus Kaggle", color="tab:gray")
    ax.bar(x + w / 2, grp - kag, w, label="grouped CV minus Kaggle", color="tab:blue")
    ax.axhline(0, color="k", lw=0.9)
    for xi, (a, b) in zip(x, zip(std - kag, grp - kag)):
        ax.text(xi - w / 2, a + 0.003, f"{a:+.3f}", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, b + 0.003, f"{b:+.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("optimism (local minus Kaggle)")
    ax.set_title("Optimism of each protocol", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = facts.FIGURES / "local_vs_kaggle.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return out


def methodology_flow():
    """The nine phases in order, colour-coded by what each one actually returned.

    The colouring is the argument: four foundation phases, two that returned nothing, and
    three that moved the score. Drawn from facts.PHASES so it cannot fall out of step with
    the report's section order.
    """
    outcome = {                     # phase number -> (colour, what it returned)
        1: ("#4C72B0", "foundation"), 2: ("#4C72B0", "foundation"),
        3: ("#4C72B0", "foundation"), 4: ("#4C72B0", "foundation"),
        5: ("#8C8C8C", "null on the leaderboard"),
        6: ("#8C8C8C", "null on the leaderboard"),
        7: ("#DD8452", "partial, and a confound found"),
        8: ("#2CA02C", "+0.044"), 9: ("#2CA02C", "+0.022"),
    }

    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top, bottom = 0.945, 0.045
    n = len(facts.PHASES)
    slot = (top - bottom) / n
    box_h = slot * 0.74

    for i, (num, title, nbs, decision) in enumerate(facts.PHASES):
        y = top - (i + 1) * slot + (slot - box_h) / 2
        colour, returned = outcome[num]

        ax.add_patch(plt.Rectangle((0.015, y), 0.40, box_h, facecolor=colour,
                                   edgecolor="none", alpha=0.90, zorder=2))
        # Titles vary from 22 to 50 characters, so wrap and lay the lines out evenly
        # rather than pinning them to fixed offsets and letting the long ones run out
        # of the box.
        lines = textwrap.wrap(f"{num}. {title}", width=38)
        rows = [(t, 10.5, "bold", 1.0) for t in lines]
        rows.append((f"notebooks {nbs}", 8.5, "normal", 0.92))
        step = box_h / (len(rows) + 1)
        for j, (text, size, weight, alpha) in enumerate(rows):
            ax.text(0.035, y + box_h - (j + 1) * step, text, color="white",
                    fontsize=size, fontweight=weight, alpha=alpha, va="center", zorder=3)

        ax.annotate("", xy=(0.435, y + box_h / 2), xytext=(0.415, y + box_h / 2),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
        ax.text(0.445, y + box_h / 2 + 0.018, decision, fontsize=9.8, va="center")
        ax.text(0.445, y + box_h / 2 - 0.021, returned, fontsize=8.5, va="center",
                color=colour, fontweight="bold")

        if i < n - 1:                       # the spine linking one phase to the next
            ax.annotate("", xy=(0.215, y - (slot - box_h)), xytext=(0.215, y),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4))

    ax.text(0.015, 0.975, "How the project actually ran, in the order it ran",
            fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = facts.FIGURES / "methodology_flow.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    facts.check()
    for fn in (journey, local_vs_kaggle, methodology_flow):
        p = fn()
        assert p.exists() and p.stat().st_size > 0, p
        print(f"wrote {p.name}  ({p.stat().st_size / 1000:.0f} kB)")
