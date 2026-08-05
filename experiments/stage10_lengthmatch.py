"""Stage 10 - match the training length distribution to the test one.

The one distribution shift we can actually see and measure is length. Test documents run
noticeably longer than training documents (median 1,723 characters against 1,146), and
the five equal-width training bands map onto the test set very unevenly:

    band 0 (shortest)  20% of train ->  11% of test
    band 1             20%           ->  16%
    band 2             20%           ->  11%
    band 3             20%           ->  28%
    band 4 (longest)   20%           ->  34%

Classic covariate-shift correction says weight each training row by
p_test(x) / p_train(x). Stage 1 tried that on the full feature space and it was a bad
idea there, because the discriminator turned out to be entangled with the label. On
length alone the ratio is five numbers we can read straight off the two histograms, it
carries no label information, and the shift it corrects is real rather than inferred.

Evaluated on standard CV. The grouped protocol cannot judge this: each of its folds
holds out an entire length band, so a length reweighting of the training half is
measuring something other than what we want to know.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.common import *


def length_weights(env, n_bands=5):
    """p_test / p_train per length band, as a per-training-row weight."""
    train_len = np.array([len(t) for t in env["train_texts"]], dtype=float)
    test_len = np.array([len(t) for t in env["test_texts"]], dtype=float)

    edges = np.quantile(train_len, np.linspace(0, 1, n_bands + 1)[1:-1])
    tb = np.digitize(train_len, edges)
    eb = np.digitize(test_len, edges)

    w = np.ones(len(train_len))
    print(f"{'band':<6}{'train':>9}{'test':>9}{'weight':>9}")
    for b in range(n_bands):
        p_tr = float((tb == b).mean())
        p_te = float((eb == b).mean())
        ratio = p_te / p_tr if p_tr > 0 else 1.0
        w[tb == b] = ratio
        print(f"{b:<6}{p_tr:>9.3f}{p_te:>9.3f}{ratio:>9.3f}", flush=True)
    return w / w.mean(), tb, eb


def test_matched_score(scores, y, bands, test_band_share, share):
    """Macro F1 computed per band, then averaged with the TEST set's band weights.

    Straight macro F1 over dev weights every band by how common it is in dev, which is
    a fifth each by construction. The leaderboard weights them by how common they are in
    the test set. This reports the second number.
    """
    per_band, weights = [], []
    for b in sorted(np.unique(bands)):
        m = bands == b
        if len(np.unique(y[m])) < 2:
            continue
        per_band.append(ensemble.macro_f1_at_share(scores[m], y[m], share))
        weights.append(test_band_share[b])
    per_band, weights = np.array(per_band), np.array(weights)
    return float((per_band * weights).sum() / weights.sum()), per_band


def run(env, n_bands=5):
    y_dev, X_dev, dev_idx = env["y_dev"], env["X_dev"], env["dev_idx"]
    log_stage("stage 10: length-matched importance weighting")

    w_all, tb_all, eb = length_weights(env, n_bands)
    w = w_all[dev_idx]
    bands = tb_all[dev_idx]
    test_share = {b: float((eb == b).mean()) for b in range(n_bands)}
    share = float(y_dev.mean())

    folds = list(evaluation.make_cv().split(X_dev, y_dev))

    def oof_with(weights, tag):
        # "lgbm" deliberately reuses stage 9's cache file - it is the same unweighted
        # standard-CV out-of-fold run, and refitting it here would cost five fits for
        # an identical array.
        path = (paths.DATA_PROCESSED / "night_oof_standard_lgbm.npy" if tag == "lgbm"
                else paths.DATA_PROCESSED / f"night_oof_std_{tag}.npy")
        if path.exists():
            return np.load(path)
        out = np.full(len(y_dev), np.nan)
        for i, (tr, te) in enumerate(folds):
            kw = {} if weights is None else {"sample_weight": weights[tr]}
            m = build_lgbm({}).fit(X_dev[tr], y_dev[tr], **kw)
            out[te] = ensemble.member_score(m, X_dev[te])
            print(f"  {tag} fold {i} done", flush=True)
        np.save(path, out.astype(np.float32))
        return out

    plain = oof_with(None, "lgbm")           # shared with stage 9's cache name below
    weighted = oof_with(w, "lgbmlenw")

    rows = []
    for tag, s in (("unweighted", plain), ("length-matched", weighted)):
        flat = ensemble.macro_f1_at_share(s, y_dev, share)
        matched, per_band = test_matched_score(s, y_dev, bands, test_share, share)
        auc = float(roc_auc_score(y_dev, s))
        rows.append((tag, flat, matched, auc, per_band))
        print(f"\n{tag}:")
        print(f"  macro F1, dev-weighted   {flat:.4f}")
        print(f"  macro F1, test-weighted  {matched:.4f}")
        print(f"  AUC                      {auc:.4f}")
        print(f"  per band                 {np.round(per_band, 4).tolist()}", flush=True)

    d_matched = rows[1][2] - rows[0][2]
    d_auc = rows[1][3] - rows[0][3]
    print(f"\nlength-matched minus unweighted: test-weighted F1 {d_matched:+.4f}, "
          f"AUC {d_auc:+.4f}")
    if d_auc > 0.002:
        print("Reweighting training toward the test length profile improves ranking. "
              "Worth a submission - it costs one argument at fit time.")
    else:
        print("No ranking gain. The model was already handling the length mix; the "
              "shift is real but the model was not suffering from it.")
    return rows


if __name__ == "__main__":
    run(ENV)
