"""Stage 11 - is the pseudo-label gain real, or the length-band artifact again?

Stage 4's sweep looked encouraging and climbing:

    keep 0.3   -0.0460
    keep 0.5   +0.0045
    keep 0.7   +0.0149    (clears the 0.0084 noise floor, 4 of 5 folds)

Two things about it do not add up. Pseudo-label accuracy *fell* as we kept more rows
(0.690 to 0.647 on band 0) while the score *rose*, which is backwards for a method whose
entire premise is that confident labels are good labels. And almost the whole gain came
from one fold: band 4 went 0.8364 to 0.9187, while the other four moved by 0.005 or less.

Band 4 is the longest documents. Under the grouped protocol its model trained on bands
0-3 and had never seen a long document - so handing it pseudo-labelled long documents
does not adapt it to a domain, it just supplies the length range the protocol took away.
That is the same artifact that made band 0 look catastrophic in Finding 2, running in
the opposite direction and flattering us this time instead of scaring us.

If that is what happened, the gain vanishes under random folds, where every training
half already contains every length and there is no missing band to restore. If it
survives, self-training is genuinely adapting to the target corpora and is worth a
submission slot.

Reuses stage 9's cached out-of-fold scores as the first round, so this costs five fits.
"""

import numpy as np
import scipy.sparse as sp

from experiments.common import *
from experiments.stage4_transductive import pseudo_label_scores


def run(env, keep_fracs=(0.7,)):
    X_dev, y_dev = env["X_dev"], env["y_dev"]
    log_stage("stage 11: pseudo-labelling under random folds")

    oof_path = paths.DATA_PROCESSED / "night_oof_standard_lgbm.npy"
    assert oof_path.exists(), "run stage 9 first - it caches the standard-CV OOF"
    oof = np.load(oof_path)
    folds = list(evaluation.make_cv().split(X_dev, y_dev))

    # Baseline on these same folds: the cached OOF scored fold by fold, so the
    # comparison is paired on identical splits.
    base = np.array([evaluation.macro_f1(
        y_dev[te], (oof[te] >= 0.5).astype(int)) for _, te in folds])
    print(f"baseline on random folds: {base.mean():.4f}  "
          f"{np.round(base, 4).tolist()}", flush=True)

    out = {}
    for frac in keep_fracs:
        def fold_scores(_f, frac=frac):
            scores = []
            for i, (tr, te) in enumerate(folds):
                share = float(y_dev[te].mean())
                keep, lab = pseudo_label_scores(oof[te], share, frac)
                Xa = sp.vstack([X_dev[tr], X_dev[te][keep]], format="csr")
                ya = np.r_[y_dev[tr], lab]
                m = build_lgbm({}).fit(Xa, ya)
                scores.append(evaluation.macro_f1(y_dev[te], m.predict(X_dev[te])))
                print(f"  frac {frac}: fold {i} {scores[-1]:.4f} vs {base[i]:.4f} "
                      f"(label accuracy {np.mean(lab == y_dev[te][keep]):.3f})",
                      flush=True)
            return scores

        rec = manual_trial(f"pseudostd{frac:g}", {"keep_frac": frac, "folds": "standard"},
                           fold_scores, protocol="shift")
        d = np.array(rec["scores"]) - base
        print(f"\nkeep {frac:g} on random folds: {rec['mean']:.4f}  "
              f"paired {d.mean():+.4f} +/- {np.std(d, ddof=1) / np.sqrt(len(d)):.4f}  "
              f"{int((d > 0).sum())}/5 folds", flush=True)
        out[frac] = (rec, float(d.mean()))

    grouped_gain = 0.0149
    best = max(v[1] for v in out.values())
    print(f"\ngrouped protocol said {grouped_gain:+.4f}; random folds say {best:+.4f}")
    if best < grouped_gain / 2:
        print("The gain was mostly the missing-length-band artifact. Self-training is "
              "restoring what the protocol removed, not adapting to the test corpora. "
              "Do not spend a submission slot on it.")
    else:
        print("The gain survives on random folds, so it is genuine target-domain "
              "adaptation rather than a protocol artifact. Worth a submission.")
    return out


if __name__ == "__main__":
    run(ENV)
