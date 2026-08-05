"""Stage 1 - cheap, high-expected-value shots.

Three ideas that cost minutes each and attack different parts of the problem:

1. Covariate-shift correction. A logistic discriminator trained to tell train rows
   from test rows (no labels involved, so no leakage) identifies which features are
   domain markers rather than machine-vs-human evidence. Two uses: drop the worst
   offenders, or importance-weight training rows by the density ratio so the fit
   leans toward the part of feature space the test set actually occupies.
2. NBSVM on the n-gram blocks - a linear model with a different inductive bias from
   the boosted trees, and the strongest classical text baseline never tried here.
3. LightGBM seed-bag - rank-averaging five seeds. Only the ranking feeds the share
   threshold, so this removes seed noise with essentially no risk.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from experiments.common import *
from src.nbsvm import NBSVM


def fit_discriminator(env):
    """Train-vs-test logistic discriminator over the CHOSEN feature space."""
    X_full, X_test = env["X_full"], env["X_test"]
    Xd = sp.vstack([X_full, X_test], format="csr")
    yd = np.r_[np.zeros(X_full.shape[0]), np.ones(X_test.shape[0])]
    # 200 lbfgs iterations is ample here: this model is only used for a coefficient
    # ranking and a density ratio, neither of which needs a tightly converged fit,
    # and the default 3000 costs half an hour on 40k columns for no gain.
    disc = LogisticRegression(C=1.0, max_iter=200, tol=1e-3,
                              class_weight="balanced").fit(Xd, yd)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(yd, disc.decision_function(Xd))
    print(f"discriminator in-sample AUC {auc:.4f} "
          f"(1.0 = train and test are trivially separable)", flush=True)
    return disc


def run(env):
    y_dev, folds5, folds3 = env["y_dev"], env["folds5"], env["folds3"]
    X_dev, dev_idx, built = env["X_dev"], env["dev_idx"], env["built"]
    results = {}

    # ---- 1. NBSVM (seconds per fold, so it runs first for early signal) -----
    log_stage("stage 1a: NBSVM on the n-gram blocks")
    NG = ["H_char_ngrams", "I_word_ngrams"]
    X_ng = tf.stack(built, NG)[0][dev_idx]
    assert X_ng.min() >= 0, "NBSVM needs non-negative features"
    print(f"n-gram sub-matrix {X_ng.shape}", flush=True)

    variants = [
        ("nbsvm_lr_a1_c1", dict(alpha=1.0, clf="lr", C=1.0)),
        ("nbsvm_lr_a1_c4", dict(alpha=1.0, clf="lr", C=4.0)),
        ("nbsvm_lr_a025_c1", dict(alpha=0.25, clf="lr", C=1.0)),
        ("nbsvm_svc_a1_c01", dict(alpha=1.0, clf="svc", C=0.1)),
    ]
    for name, cfg in variants:
        clf = (LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=3000)
               if cfg["clf"] == "lr" else
               LinearSVC(C=cfg["C"], class_weight="balanced", max_iter=5000))
        rec = night_trial(name, cfg, NBSVM(estimator=clf, alpha=cfg["alpha"]), X=X_ng)
        report(rec, name)
        results[name] = {"rec": rec, "X": "ngrams", "cfg": cfg}

    # ---- 2. covariate shift ------------------------------------------------
    log_stage("stage 1b: covariate-shift correction")
    disc = fit_discriminator(env)
    coef = np.abs(disc.coef_.ravel())
    order = np.argsort(-coef)

    for k in (500, 2000, 5000):
        keep = np.setdiff1d(np.arange(X_dev.shape[1]), order[:k])
        rec = night_trial(f"dropmarkers{k}", {"drop_top": k},
                          build_lgbm({}), X=X_dev[:, keep])
        report(rec, f"drop top-{k} domain markers")
        results[f"dropmarkers{k}"] = {"rec": rec, "keep": keep}

    # Density-ratio weights p(test|x)/(1-p(test|x)), clipped so a handful of extreme
    # rows cannot dominate the fit, then normalised to mean 1 so the effective
    # sample size stays comparable to the unweighted baseline.
    p = disc.predict_proba(X_dev)[:, 1]
    for clip in (5.0, 20.0):
        w = np.clip(p / np.maximum(1 - p, 1e-6), 1 / clip, clip)
        w = w / w.mean()

        def fold_scores(folds, w=w):
            out = []
            for tr, te in folds:
                m = build_lgbm({}).fit(X_dev[tr], y_dev[tr], sample_weight=w[tr])
                out.append(evaluation.macro_f1(y_dev[te], m.predict(X_dev[te])))
            return out

        rec = manual_trial(f"densityweight{clip:g}", {"clip": clip}, fold_scores)
        report(rec, f"density-ratio weights (clip {clip:g})")
        results[f"densityweight{clip:g}"] = {"rec": rec, "weights": w}

    # ---- 3. LightGBM seed bag ---------------------------------------------
    log_stage("stage 1c: LightGBM seed-bag")
    bag = SeedBag(build=lambda seed: LGBMClassifier(**{**FIXED, "random_state": seed}),
                  seeds=(42, 43, 44, 45, 46))
    rec = night_trial("lgbm_seedbag5", {"seeds": 5}, bag)
    report(rec, "lgbm seed-bag (5 seeds, rank-averaged)")
    results["lgbm_seedbag5"] = {"rec": rec}

    # ---- summary -----------------------------------------------------------
    log_stage("stage 1 summary (5-band selection protocol)")
    for name, r in sorted(results.items(),
                          key=lambda kv: -kv[1]["rec"]["paired_mean"]):
        rec = r["rec"]
        flag = "PASS" if passes_g5(rec) else "    "
        print(f"  {flag} {name:28s} paired {rec['paired_mean']:+.4f} "
              f"+/- {rec['paired_se']:.4f}  {rec['folds_better']}/5 folds", flush=True)
    return results


if __name__ == "__main__":
    run(ENV)
