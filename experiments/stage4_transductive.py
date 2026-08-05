"""Stage 4 - transductive adaptation, the largest lever on the domain gap.

The test text is available and unlabeled, and nothing in the brief forbids using it.
Two independent ways to exploit that:

A. Vocabulary refit. Blocks H and I fit their vectorizers on train text only, so every
   character or word n-gram that occurs in the test corpora but not the training
   corpora is invisible to the model - and the corpora are disjoint by construction.
   Refitting the vectorizers on train+test text uses no labels at all and lets the
   representation see the vocabulary it will be asked about.

B. Self-training. Score the unlabeled rows, keep the confident ones as pseudo-labels,
   refit including them. This adapts the decision boundary itself rather than the
   feature space. Gated by a simulation first: hold out one length band, treat it as
   unlabeled, and check the band's macro F1 actually improves before doing it for real.

Both are evaluated against the same LightGBM-defaults baseline as everything else.
"""

import time

import numpy as np
import scipy.sparse as sp

from experiments.common import *


def refit_ngrams(fit_texts, apply_sets):
    """Rebuild blocks H and I with vectorizers fitted on `fit_texts`.

    Parameters
    ----------
    fit_texts : list of str
        Corpus the vectorizers learn their vocabulary from.
    apply_sets : list of array-like of str
        Corpora to transform with the fitted vectorizers.

    Returns
    -------
    list of CSR, one per entry in apply_sets (H and I horizontally stacked).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True,
                           min_df=3, max_features=20000, sublinear_tf=True,
                           dtype=np.float32).fit(fit_texts)
    word = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words=None,
                           min_df=5, max_features=20000, sublinear_tf=True,
                           dtype=np.float32).fit(fit_texts)
    return [sp.hstack([char.transform(t), word.transform(t)], format="csr").astype(np.float32)
            for t in apply_sets]


def style_matrix(env):
    """Blocks A-F only - the part of the representation that never needs refitting."""
    STYLE = ["A_function_words", "B_punctuation", "C_casing", "D_structure",
             "E_length", "F_diversity"]
    Xtr, Xte, _ = tf.stack(env["built"], STYLE)
    return Xtr, Xte


def run_vocab_refit(env, protocol="g5"):
    """A: refit the n-gram vectorizers on fold-train text plus the real test text."""
    log_stage(f"stage 4a: vocabulary refit on train+test text ({protocol})")
    y_dev, dev_texts = env["y_dev"], env["dev_texts"]
    test_texts = np.asarray(env["test_texts"])
    S_full, S_test = style_matrix(env)
    S_dev = S_full[env["dev_idx"]]

    def fold_scores(folds):
        out = []
        for i, (tr, te) in enumerate(folds):
            t = time.time()
            # Fit on fold-train text + the unlabeled test corpus. The held-out band's
            # own text is excluded, so the fold stays honest; at submission time the
            # analogue is all-train + test, which is exactly what gets built later.
            fit_corpus = list(dev_texts[tr]) + list(test_texts)
            Ntr, Nte = refit_ngrams(fit_corpus, [dev_texts[tr], dev_texts[te]])
            Xtr = sp.hstack([S_dev[tr], Ntr], format="csr")
            Xte = sp.hstack([S_dev[te], Nte], format="csr")
            m = build_lgbm({}).fit(Xtr, y_dev[tr])
            out.append(evaluation.macro_f1(y_dev[te], m.predict(Xte)))
            print(f"  fold {i}: {out[-1]:.4f}  ({time.time() - t:.0f}s)", flush=True)
        return out

    rec = manual_trial("vocabrefit", {"fit_on": "foldtrain+test"}, fold_scores,
                       protocol=protocol)
    return report(rec, f"vocab refit ({protocol})")


def pseudo_label_scores(scores, share, keep_frac):
    """Pseudo-labels for the most confident `keep_frac` of rows.

    Labels come from the share threshold (the same rule the submission uses), and
    confidence is distance from that threshold in score space, so the kept rows are
    the ones furthest from the decision boundary on either side.

    Returns
    -------
    idx : ndarray, rows to keep
    labels : ndarray of int, their pseudo-labels
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = ensemble.threshold_at_share(scores, share)
    k = int(round(share * len(scores)))
    cut = np.sort(scores)[::-1][max(k - 1, 0)]
    conf = np.abs(scores - cut)
    keep = np.argsort(-conf)[:int(round(keep_frac * len(scores)))]
    return keep, labels[keep]


def base_fold_scores(env, protocol="g5"):
    """First-round scores for each held-out band, cached across confidence cutoffs.

    Every cutoff starts from the same first-round model, so fitting it once per fold
    instead of once per (fold, cutoff) removes two thirds of this stage's compute.
    """
    path = paths.DATA_PROCESSED / f"night_pseudo_base_{protocol}.npz"
    folds = env["folds5"] if protocol == "g5" else env["folds3"]
    if path.exists():
        z = np.load(path)
        return [z[f"f{i}"] for i in range(len(folds))]

    X_dev, y_dev = env["X_dev"], env["y_dev"]
    out = {}
    for i, (tr, te) in enumerate(folds):
        t = time.time()
        m0 = build_lgbm({}).fit(X_dev[tr], y_dev[tr])
        out[f"f{i}"] = ensemble.member_score(m0, X_dev[te]).astype(np.float32)
        print(f"  base fold {i}: {time.time() - t:.0f}s", flush=True)
    np.savez_compressed(path, **out)
    return [out[f"f{i}"] for i in range(len(folds))]


def run_pseudo_simulation(env, keep_fracs=(0.3, 0.5, 0.7), protocol="g5"):
    """B (gate): treat each held-out band as unlabeled and self-train on it."""
    log_stage(f"stage 4b: pseudo-label simulation ({protocol})")
    X_dev, y_dev = env["X_dev"], env["y_dev"]
    base = base_fold_scores(env, protocol)
    out = {}

    for frac in keep_fracs:
        def fold_scores(folds, frac=frac):
            scores = []
            for i, (tr, te) in enumerate(folds):
                t = time.time()
                # The real run knows its target shares from the leaderboard, so the
                # simulation is given the band's true share as the same kind of prior.
                share = float(y_dev[te].mean())
                keep, lab = pseudo_label_scores(base[i], share, frac)
                Xa = sp.vstack([X_dev[tr], X_dev[te][keep]], format="csr")
                ya = np.r_[y_dev[tr], lab]
                m1 = build_lgbm({}).fit(Xa, ya)
                scores.append(evaluation.macro_f1(y_dev[te], m1.predict(X_dev[te])))
                print(f"  frac {frac}: fold {i} {scores[-1]:.4f} "
                      f"(pseudo-label accuracy {np.mean(lab == y_dev[te][keep]):.3f}, "
                      f"{time.time() - t:.0f}s)", flush=True)
            return scores

        rec = manual_trial(f"pseudo{frac:g}", {"keep_frac": frac}, fold_scores,
                           protocol=protocol)
        out[frac] = report(rec, f"pseudo-label keep {frac:g}")
    return out


def run(env):
    results = {}
    # Run the vocabulary refit on the 3-band protocol, not the 5-band one. Refitting a
    # char 2-5 vectoriser over ~20k documents is by far the most expensive thing in the
    # night, and three folds still gives the confirmation-grade answer for 40% less.
    results["vocabrefit_g3"] = run_vocab_refit(env, "g3")
    results.update({f"pseudo{k:g}": v
                    for k, v in run_pseudo_simulation(env).items()})

    log_stage("stage 4 summary (5-band selection protocol)")
    for name, rec in sorted(results.items(), key=lambda kv: -kv[1]["paired_mean"]):
        flag = "PASS" if passes_g5(rec) else "    "
        print(f"  {flag} {name:24s} mean {rec['mean']:.4f}  paired "
              f"{rec['paired_mean']:+.4f} +/- {rec['paired_se']:.4f}  "
              f"{rec['folds_better']}/5 folds", flush=True)
    return results


if __name__ == "__main__":
    run(ENV)
