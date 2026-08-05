"""Stage 6 - turn whatever passed into checked submission CSVs.

Every candidate is written through the same recipe as the 0.80143 benchmark: score the
test set, threshold each id group at its frozen share, patch the ten test rows whose
text appears verbatim in train, and verify the realised shares before the file counts
as written. The shares are not re-tuned here - they were fixed at the leaderboard
vertex, and re-fitting them on the same leaderboard is how a 0.0084 noise floor turns
into a phantom gain.

A candidate that changes fewer than ~100 rows against the benchmark cannot be resolved
against that floor, so the writer flags it as not worth a submission slot.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from experiments.common import *


def submit_lgbm_defaults(env):
    """The current best model, rewritten here as the reproducibility check."""
    log_stage("submission: lightgbm defaults (benchmark reproduction)")
    m = build_lgbm({}).fit(env["X_full"], env["y"])
    scores = ensemble.member_score(m, env["X_test"])
    np.save(paths.DATA_PROCESSED / "night_scores_lgbm_defaults.npy", scores)
    return write_night_submission(scores, "night_lgbm_defaults.csv")


def submit_seedbag(env, seeds=(42, 43, 44, 45, 46)):
    """Rank-average several seeds, fitting one model at a time.

    `SeedBag` keeps every fitted model alive so it can score later, which is fine on a
    CV fold and fatal on the full 20,000 x 40,385 training matrix with under a gigabyte
    of memory spare - it was killed twice that way. Scoring each model as soon as it is
    fitted and keeping only the running rank total costs one model's memory instead of
    five, and gives an identical answer.
    """
    log_stage("submission: lightgbm seed-bag")
    import gc

    total = np.zeros(env["X_test"].shape[0])
    for i, s in enumerate(seeds):
        m = LGBMClassifier(**{**FIXED, "random_state": s}).fit(env["X_full"], env["y"])
        total += ensemble.to_rank(ensemble.member_score(m, env["X_test"]))
        del m
        gc.collect()
        print(f"  seed {s} done ({i + 1}/{len(seeds)})", flush=True)

    scores = total / len(seeds)
    np.save(paths.DATA_PROCESSED / "night_scores_seedbag.npy", scores)
    return write_night_submission(scores, "night_lgbm_seedbag.csv")


def submit_vocab_refit(env):
    """Blocks H and I rebuilt with vectorizers fitted on train text plus test text."""
    log_stage("submission: transductive vocabulary refit")
    from experiments.stage4_transductive import refit_ngrams, style_matrix

    train_texts = np.asarray(env["train_texts"])
    test_texts = np.asarray(env["test_texts"])
    S_full, S_test = style_matrix(env)

    Ntr, Nte = refit_ngrams(list(train_texts) + list(test_texts),
                            [train_texts, test_texts])
    Xtr = sp.hstack([S_full, Ntr], format="csr")
    Xte = sp.hstack([S_test, Nte], format="csr")
    m = build_lgbm({}).fit(Xtr, env["y"])
    scores = ensemble.member_score(m, Xte)
    np.save(paths.DATA_PROCESSED / "night_scores_vocabrefit.npy", scores)
    return write_night_submission(scores, "night_vocabrefit.csv")


def submit_pseudo(env, keep_frac, base_scores=None):
    """Self-training: pseudo-label the confident test rows, refit including them."""
    log_stage(f"submission: pseudo-label self-training (keep {keep_frac:g})")
    from experiments.stage4_transductive import pseudo_label_scores

    X_full, X_test, y = env["X_full"], env["X_test"], env["y"]
    if base_scores is None:
        path = paths.DATA_PROCESSED / "night_scores_lgbm_defaults.npy"
        base_scores = (np.load(path) if path.exists()
                       else ensemble.member_score(
                           build_lgbm({}).fit(X_full, y), X_test))

    # Pseudo-label each id group at its own frozen share, since the two groups have
    # very different machine rates and one global share would mislabel both.
    groups = text.id_group(env["test_ids"])
    keep_idx, keep_lab = [], []
    for g, share in BEST_SHARES.items():
        m = np.flatnonzero(groups == g)
        idx, lab = pseudo_label_scores(base_scores[m], share, keep_frac)
        keep_idx.append(m[idx])
        keep_lab.append(lab)
    keep_idx, keep_lab = np.concatenate(keep_idx), np.concatenate(keep_lab)
    print(f"  kept {len(keep_idx)} of {len(base_scores)} test rows as pseudo-labels "
          f"({keep_lab.mean():.3f} machine)", flush=True)

    Xa = sp.vstack([X_full, X_test[keep_idx]], format="csr")
    ya = np.r_[y, keep_lab]
    m = build_lgbm({}).fit(Xa, ya)
    scores = ensemble.member_score(m, X_test)
    np.save(paths.DATA_PROCESSED / f"night_scores_pseudo{keep_frac:g}.npy", scores)
    return write_night_submission(scores, f"night_pseudo{keep_frac:g}.csv")


def submit_lengthmatched(env):
    """Train with each row weighted by how common its length band is in the test set."""
    log_stage("submission: length-matched importance weighting")
    from experiments.stage10_lengthmatch import length_weights

    w, _, _ = length_weights(env)
    m = build_lgbm({}).fit(env["X_full"], env["y"], sample_weight=w)
    scores = ensemble.member_score(m, env["X_test"])
    np.save(paths.DATA_PROCESSED / "night_scores_lengthmatched.npy", scores)
    return write_night_submission(scores, "night_lengthmatched.csv")


def submit_ensemble(env, lane, config, member_names):
    """Blend cached member test scores with a chosen combiner and write the CSV."""
    log_stage(f"submission: ensemble {lane}/{config}")
    from src import combiners

    oof, test = {}, {}
    for n in member_names:
        oof[n] = np.load(paths.DATA_PROCESSED / f"night_member_{n}_oof_g3.npy")
        test[n] = np.load(paths.DATA_PROCESSED / f"night_member_{n}_test.npy")

    names = sorted(oof)
    R, _ = ensemble.rank_matrix(oof, names)
    Q, _ = ensemble.rank_matrix(test, names)
    predict = combiners.fit_full(lane, config, R, env["y_dev"])
    scores = predict(Q)
    tag = f"{lane}_{config.get('kind')}"
    np.save(paths.DATA_PROCESSED / f"night_scores_ens_{tag}.npy", scores)
    return write_night_submission(scores, f"night_ens_{tag}.csv")


if __name__ == "__main__":
    submit_lgbm_defaults(ENV)
