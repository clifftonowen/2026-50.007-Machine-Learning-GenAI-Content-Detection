"""Stage 9 - are short documents hard, or merely unseen?

Stage 8 found the model is close to blind on the shortest band: AUC 0.6676 there
against 0.89 to 0.98 everywhere else. Before doing feature surgery on that, there is a
confound to rule out, and it is a serious one.

The grouped protocol holds out an entire length band. So when band 0 is the test fold,
the model was trained on bands 1-4 and *has never seen a document under 388 characters*.
Its failure there could mean short text is genuinely hard, or only that nothing in
training resembled it.

That distinction decides what to do next, and the two cases point opposite ways:

- Real difficulty: short documents carry less signal, the style features are noise at
  that length, and feature work is warranted. The 0.6266 band score is a fair estimate
  of what happens on the leaderboard's short rows.
- Mere unfamiliarity: the protocol is manufacturing a problem that never occurs at test
  time, because the actual training set contains 3,198 short documents and the actual
  test set contains short documents too. The 0.6266 is then an artifact, and the
  headroom it appears to offer is not real.

Standard CV settles it. Its folds are random, so every fold's training half contains
short documents, and the out-of-fold score for the short rows answers the question
directly. Five fits.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.common import *


def oof_standard(env):
    """Out-of-fold dev scores under standard 5-fold CV, cached."""
    path = paths.DATA_PROCESSED / "night_oof_standard_lgbm.npy"
    if path.exists():
        return np.load(path)
    folds = list(evaluation.make_cv().split(env["X_dev"], env["y_dev"]))
    oof = evaluation.oof_from_folds(build_lgbm({}), env["X_dev"], env["y_dev"], folds)
    assert np.isfinite(oof).all()
    np.save(path, oof.astype(np.float32))
    return oof


# AUC per band when the band WAS held out wholesale, from stage 8.
GROUPED_AUC = {0: 0.6676, 1: 0.8919, 2: 0.9556, 3: 0.9780, 4: 0.9664}


def run(env, n_bands=5):
    y_dev = env["y_dev"]
    log_stage("stage 9: is band 0 hard, or just unseen?")

    oof = oof_standard(env)
    bands = clustering.length_groups(env["dev_texts"], n_groups=n_bands)

    print(f"\n{'band':<6}{'AUC, band held out':>21}{'AUC, standard CV':>19}"
          f"{'recovered':>11}")
    rows = []
    for b in np.unique(bands):
        m = bands == b
        std_auc = float(roc_auc_score(y_dev[m], oof[m]))
        grp_auc = GROUPED_AUC[int(b)]
        rows.append((int(b), grp_auc, std_auc))
        print(f"{b:<6}{grp_auc:>21.4f}{std_auc:>19.4f}{std_auc - grp_auc:>11.4f}",
              flush=True)

    b0_grp, b0_std = rows[0][1], rows[0][2]
    share = float(y_dev.mean())
    print(f"\noverall out-of-fold macro F1 at share {share:.3f}: "
          f"{ensemble.macro_f1_at_share(oof, y_dev, share):.4f}")

    print()
    if b0_std > 0.90:
        print(f"Band 0 recovers to {b0_std:.4f} once training contains short documents. "
              "The short-document 'bottleneck' is an artifact of holding out a whole "
              "length range - which never happens at test time, since both train and "
              "test span every length. Do NOT spend the night on short-text features; "
              "the 0.034 of apparent headroom is not there to collect.")
    elif b0_std > b0_grp + 0.10:
        print(f"Band 0 improves a lot ({b0_grp:.4f} to {b0_std:.4f}) but stays below "
              "the other bands. Part artifact, part real: short text is harder, and "
              "the grouped protocol exaggerates by how much.")
    else:
        print(f"Band 0 stays weak ({b0_std:.4f}) even when short documents are in "
              "training. Short text is genuinely low-signal and feature work is "
              "the right response.")
    return rows


if __name__ == "__main__":
    run(ENV)
