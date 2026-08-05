"""Stage 8 - are we grading every model at a cutoff we never actually use?

`cross_val_score(..., scoring="f1_macro")` calls `estimator.predict()`, which cuts at
probability 0.5. Every trial in this repo's ledger was scored that way.

We do not submit that way. Submissions threshold by predicted *share* - the top k rows
by score become machine, where k comes from the frozen per-group shares. That change
alone took the leaderboard from 0.65738 to 0.73583 on an unchanged model, the single
largest jump in the project's history.

So the selection protocol and the submission procedure disagree about what a good model
is, and they have disagreed by 0.08 at least once. This stage measures the size of that
disagreement per length band. It matters most for band 0, whose 0.6266 is what makes
the average look bad: if that number is mostly a calibration artifact, the short-document
"bottleneck" is not a bottleneck at all and the ranking there is fine.

Costs five fits, reusing the same cached fold scores stage 4 needs.
"""

import numpy as np

from experiments.common import *
from experiments.stage4_transductive import base_fold_scores


def run(env):
    y_dev, folds5 = env["y_dev"], env["folds5"]

    log_stage("stage 8: 0.5 cutoff vs share thresholding, per band")
    base = base_fold_scores(env, "g5")

    rows = []
    for i, (tr, te) in enumerate(folds5):
        s, yt = base[i], y_dev[te]
        true_share = float(yt.mean())
        at_true = ensemble.macro_f1_at_share(s, yt, true_share)
        at_53 = ensemble.macro_f1_at_share(s, yt, 0.53)
        auc = combiners_auc(yt, s)
        rows.append((i, BASE5[i], at_true, at_53, auc, true_share))

    print(f"\n{'band':<6}{'@0.5 cut':>10}{'@true share':>13}{'@0.53':>9}"
          f"{'AUC':>8}{'true share':>12}")
    for i, base_f1, at_true, at_53, auc, share in rows:
        print(f"{i:<6}{base_f1:>10.4f}{at_true:>13.4f}{at_53:>9.4f}"
              f"{auc:>8.4f}{share:>12.3f}", flush=True)

    m05 = float(np.mean([r[1] for r in rows]))
    mts = float(np.mean([r[2] for r in rows]))
    print(f"\nmean  {m05:>10.4f}{mts:>13.4f}")
    print(f"share thresholding is worth {mts - m05:+.4f} on the grouped protocol")

    b0 = rows[0]
    print(f"\nband 0: {b0[1]:.4f} at the 0.5 cutoff, {b0[2]:.4f} at its true share.")
    if b0[2] - b0[1] > 0.05:
        print("Band 0's weakness is mostly CALIBRATION, not ranking - its AUC is fine "
              "and the 0.5 cutoff is simply in the wrong place for short documents. "
              "Chasing short-document features would have been chasing an artifact.")
    else:
        print("Band 0 is genuinely hard to RANK, not just to threshold - the "
              "short-document problem is real and worth fixing at the feature level.")
    return rows


def combiners_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


if __name__ == "__main__":
    run(ENV)
