"""Domain clustering, leave-one-cluster-out CV, and per-cluster share.

**The problem this exists to solve.** Every round of this project has selected models
and features on 5-fold stratified CV over `train_features.csv`. The COLING paper shows
the test set is drawn from entirely different source corpora than train, so that
protocol scores a candidate on data from the same domains it was fitted on, and
rewards whatever memorises those domains best. That is a plausible explanation for the
project's recurring pattern of local gains that do not transfer: round 3's ensemble
gained 0.0017 CV and zero on Kaggle, and round 4's best combiner gained 0.0075
out-of-fold AUC and 0.0017 on Kaggle.

**The fix.** Cluster the training rows into pseudo-domains, then hold out a whole
cluster at a time. A feature that only works because it recognises the domain scores
well under stratified CV and badly under this, and the gap between the two is a direct
estimate of how much of that feature's apparent value will not survive the shift.
`14_ablation.ipynb` selects on the grouped number.

The clusters are a *proxy* for the true source corpora, which are not labelled in the
data. `cluster_report` exists to show how good a proxy they are before anything is
built on top of them, and `stability` gates the whole protocol: clusters that do not
survive a change of seed cannot support a claim about transfer.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def reduce_dims(X_train, X_test=None, *, n_components=100, random_state=RANDOM_STATE):
    """TruncatedSVD then standardisation, fitted on train only.

    SVD rather than PCA because the style matrix is sparse and TruncatedSVD does not
    densify it. Standardising afterwards matters more than usual here: the dense style
    blocks mix quantities on wildly different scales (a rate in [0, 1] beside a raw
    character count in the thousands), and k-means minimises Euclidean distance, so
    without it the largest-magnitude column silently becomes the only one that counts.

    Parameters
    ----------
    X_train : sparse or dense, shape (n_train, n_features)
    X_test : same, optional
    n_components : int, default 100
    random_state : int, default 42

    Returns
    -------
    Z_train : ndarray, shape (n_train, n_components)
    Z_test : ndarray or None
    model : dict
        The fitted `svd` and `scaler`, for reuse.
    """
    n_components = min(n_components, min(X_train.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    Z_train = svd.fit_transform(X_train)
    scaler = StandardScaler().fit(Z_train)
    Z_train = scaler.transform(Z_train)
    Z_test = scaler.transform(svd.transform(X_test)) if X_test is not None else None
    return Z_train, Z_test, {"svd": svd, "scaler": scaler,
                             "explained": float(svd.explained_variance_ratio_.sum())}


def sweep_k(Z, ks=range(2, 13), *, random_state=RANDOM_STATE, sample=5000):
    """Inertia and silhouette across candidate cluster counts.

    Silhouette is computed on a subsample because it is O(n^2) in the number of rows
    and 20,000 rows would dominate the runtime of the whole notebook.

    Parameters
    ----------
    Z : ndarray, shape (n_samples, n_components)
    ks : iterable of int
    random_state : int
    sample : int, default 5000
        Rows used for the silhouette estimate.

    Returns
    -------
    pd.DataFrame
        Columns: k, inertia, silhouette.
    """
    rng = np.random.default_rng(random_state)
    idx = (rng.choice(len(Z), size=sample, replace=False)
           if len(Z) > sample else np.arange(len(Z)))
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Z)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": float(silhouette_score(Z[idx], km.labels_[idx]))})
    return pd.DataFrame(rows)


def fit_clusters(Z_train, Z_test=None, *, k=5, random_state=RANDOM_STATE):
    """Fit k-means on train and assign test rows to the nearest centroid.

    Test rows are *assigned*, never fitted on, so the cluster definition depends only
    on training data and the same clusters can be used to reason about both sets.

    Returns
    -------
    train_labels : ndarray of int, shape (n_train,)
    test_labels : ndarray of int or None
    km : fitted KMeans
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Z_train)
    test_labels = km.predict(Z_test) if Z_test is not None else None
    return km.labels_, test_labels, km


def stability(Z, *, k=5, seeds=(42, 43, 44)):
    """Adjusted Rand index between k-means fits that differ only by seed.

    The gate on everything downstream. If refits under different seeds do not agree,
    the partition is an artifact of initialisation rather than structure in the data,
    and a leave-one-cluster-out protocol built on it would measure noise. Treat ARI
    below roughly 0.8 as a failure and either change k or abandon the grouped protocol.

    Returns
    -------
    pd.DataFrame
        One row per seed pair, columns: seed_a, seed_b, ari.
    """
    fits = {s: KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(Z)
            for s in seeds}
    rows = [{"seed_a": a, "seed_b": b,
             "ari": float(adjusted_rand_score(fits[a], fits[b]))}
            for i, a in enumerate(seeds) for b in list(seeds)[i + 1:]]
    return pd.DataFrame(rows)


def cluster_report(train_labels, y, *, oof_scores=None, share=None):
    """Per-cluster size, class balance, and out-of-fold error rate.

    The error column is the first per-row error analysis in this project. Every
    previous diagnostic looked at how models disagree with *each other* on unlabelled
    test rows; this asks which kinds of training document the models actually get
    wrong, which is the question that tells you what to fix.

    Parameters
    ----------
    train_labels : ndarray of int, shape (n_train,)
    y : ndarray of int, shape (n_train,)
    oof_scores : ndarray, optional
        Out-of-fold machine-class scores for the same rows, e.g. from
        `tuning.load_oof("lightgbm")`. Thresholded at `share` to get labels.
    share : float, optional
        Predicted machine share used to threshold `oof_scores`. Defaults to the true
        share of `y`, matching the project's matched-share convention.

    Returns
    -------
    pd.DataFrame
        Columns: cluster, n, frac_of_train, machine_share, and when scores are given,
        error_rate and macro_f1.
    """
    from . import ensemble, evaluation  # local import keeps the module dependency-light

    train_labels = np.asarray(train_labels)
    y = np.asarray(y)
    share = float(y.mean()) if share is None else float(share)

    pred = (ensemble.threshold_at_share(oof_scores, share)
            if oof_scores is not None else None)

    rows = []
    for c in np.unique(train_labels):
        m = train_labels == c
        row = {"cluster": int(c), "n": int(m.sum()),
               "frac_of_train": float(m.mean()),
               "machine_share": float(y[m].mean())}
        if pred is not None:
            row["error_rate"] = float((pred[m] != y[m]).mean())
            # Guard: macro F1 is undefined on a single-class slice.
            row["macro_f1"] = (float(evaluation.macro_f1(y[m], pred[m]))
                               if len(np.unique(y[m])) > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def coverage(train_labels, test_labels):
    """How train mass and test mass distribute across the same clusters.

    The concrete shape of the distribution shift. A cluster holding a large share of
    test and almost none of train is a domain the models have effectively never seen,
    and is where the leaderboard score is being lost.

    Returns
    -------
    pd.DataFrame
        Columns: cluster, n_train, n_test, frac_train, frac_test, ratio.
        `ratio` is frac_test / frac_train; large means test-heavy and under-trained.
    """
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    clusters = np.union1d(np.unique(train_labels), np.unique(test_labels))
    rows = []
    for c in clusters:
        n_tr, n_te = int((train_labels == c).sum()), int((test_labels == c).sum())
        f_tr, f_te = n_tr / len(train_labels), n_te / len(test_labels)
        rows.append({"cluster": int(c), "n_train": n_tr, "n_test": n_te,
                     "frac_train": f_tr, "frac_test": f_te,
                     "ratio": f_te / f_tr if f_tr > 0 else np.inf})
    return pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)


def cluster_cv(labels, y, *, min_per_class=50):
    """Leave-one-cluster-out folds, as a list of (train_idx, test_idx).

    A drop-in replacement for `evaluation.make_cv()`'s output in any
    `cross_val_score`-style call, but the folds are domains rather than random
    stratified slices, so the score answers "does this generalise to a domain it has
    never seen" instead of "does this fit this domain".

    Clusters too small or too single-class to be a valid evaluation fold are skipped
    rather than silently producing a degenerate macro F1, and the skips are returned
    so the notebook can report them instead of hiding them.

    Parameters
    ----------
    labels : ndarray of int, shape (n_samples,)
        Cluster assignment per row.
    y : ndarray of int, shape (n_samples,)
    min_per_class : int, default 50
        A held-out cluster needs at least this many rows of each class.

    Returns
    -------
    folds : list of (ndarray, ndarray)
    skipped : pd.DataFrame
        Columns: cluster, n, n_pos, n_neg, reason. Empty if nothing was skipped.
    """
    labels, y = np.asarray(labels), np.asarray(y)
    folds, skipped = [], []

    for c in np.unique(labels):
        test_idx = np.flatnonzero(labels == c)
        train_idx = np.flatnonzero(labels != c)
        n_pos, n_neg = int((y[test_idx] == 1).sum()), int((y[test_idx] == 0).sum())

        reason = None
        if min(n_pos, n_neg) < min_per_class:
            reason = f"held-out cluster has {n_pos} pos / {n_neg} neg"
        elif len(np.unique(y[train_idx])) < 2:
            reason = "training remainder is single-class"

        if reason:
            skipped.append({"cluster": int(c), "n": len(test_idx),
                            "n_pos": n_pos, "n_neg": n_neg, "reason": reason})
        else:
            folds.append((train_idx, test_idx))

    assert folds, ("no cluster is usable as a held-out fold - lower k, or raise "
                   "min_per_class only if you can justify it")
    cols = ["cluster", "n", "n_pos", "n_neg", "reason"]
    return folds, pd.DataFrame(skipped, columns=cols)


def per_cluster_share(test_labels, global_share, cluster_shares=None):
    """Predicted machine share for each test row, varying by cluster.

    The project applies one predicted share to all 6,999 test rows. The paper's test
    corpora have true machine shares from 46.5% (CUDRT) to 82.8% (MixSet), and share
    is the strongest single lever on this task, so a single global cut is leaving
    something on the table if the clusters really do track the corpora.

    **This is speculative and is gated in 13_clustering section 7.** It should only be
    used if the clusters are stable, interpretable, and line up with the id groups or
    the paper's proportions. Otherwise it is fitting noise on the one axis where a
    mistake costs the most.

    Parameters
    ----------
    test_labels : ndarray of int, shape (n_test,)
    global_share : float
        Fallback for any cluster not named in `cluster_shares`.
    cluster_shares : dict, optional
        cluster -> target share. Missing clusters fall back to `global_share`.

    Returns
    -------
    ndarray of float, shape (n_test,)
        Per-row target share, for use with a per-cluster threshold.
    """
    test_labels = np.asarray(test_labels)
    cluster_shares = cluster_shares or {}
    out = np.full(len(test_labels), float(global_share))
    for c, s in cluster_shares.items():
        out[test_labels == c] = float(s)
    return out


def threshold_per_group(scores, groups, target_shares):
    """Threshold within each group so every group hits its own predicted share.

    The mechanism behind both the id-group split in notebook 11 and the per-cluster
    share above. Each group is cut independently with the exact top-k rule, so a
    group's realised share is within one row of its target regardless of how the
    groups differ in score distribution.

    Parameters
    ----------
    scores : ndarray, shape (n_samples,)
    groups : ndarray, shape (n_samples,)
        Any hashable group key per row.
    target_shares : dict
        group key -> target share. Every group present must have an entry.

    Returns
    -------
    ndarray of int, shape (n_samples,)
        0/1 labels.
    """
    from . import ensemble

    scores, groups = np.asarray(scores, dtype=np.float64), np.asarray(groups)
    missing = set(np.unique(groups)) - set(target_shares)
    assert not missing, f"no target share given for groups {sorted(missing)}"

    preds = np.zeros(len(scores), dtype=int)
    for g, share in target_shares.items():
        m = groups == g
        if m.any():
            preds[m] = ensemble.threshold_at_share(scores[m], share)
    return preds
