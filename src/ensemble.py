"""Rank-based blending for the round-4 ensemble.

Everything here follows from one fact about how this project submits: notebooks
08/09 threshold by *share* (`write_at_share` takes an empirical quantile), so only
the induced **ranking** of the test rows matters. Two consequences shape this
module, and both are corrections to what notebook 07 did:

1. **Combine on ranks, not probabilities.** Rescaling a single member monotonically
   is a no-op (which is why `decision_function` already works for LinearSVC in
   09 section 2), but averaging raw probabilities is *not* rank-invariant - it
   weights each member by its probability spread, and LightGBM's probabilities are
   far sharper than a penalized logistic's. 07 averaged probabilities, so its member
   weighting was an artifact of calibration rather than a choice. Converting every
   member to ranks first removes that entirely, and means LinearSVC no longer needs
   a `CalibratedClassifierCV` wrapper just to expose `predict_proba`.

2. **Select on ROC AUC, not macro-F1 at 0.5.** AUC *is* ranking quality and is
   invariant to the share, so it measures exactly what the submission procedure
   uses. F1 at a fixed 0.5 cutoff is confounded by the calibration effect that
   notebook 09 spent three rounds isolating - and that is the metric 07 selected
   its member subset on.

`macro_f1_at_share` is the local mirror of `write_at_share`, for reporting F1
alongside AUC without reintroducing the 0.5-cutoff confound.
"""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, roc_auc_score


def member_score(estimator, X):
    """Any monotone score for the machine class - probability or margin.

    Uses `predict_proba` when available and falls back to `decision_function`
    otherwise. The fallback is exact rather than approximate: the blend ranks
    every member, so a bare `LinearSVC` needs no `CalibratedClassifierCV` wrapper
    to participate (notebook 09 section 2 relies on the same equivalence).

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
    X : array-like or sparse matrix

    Returns
    -------
    ndarray, shape (n_samples,)
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def to_rank(scores):
    """Map any member score to (0, 1] by its rank - the common scale for blending.

    Accepts probabilities or raw `decision_function` margins interchangeably; the
    output depends only on the ordering, which is the point.

    Parameters
    ----------
    scores : array-like, shape (n_samples,)

    Returns
    -------
    ndarray, shape (n_samples,)
        Average ranks divided by n, so ties are handled and the scale is uniform.
    """
    scores = np.asarray(scores, dtype=np.float64)
    return rankdata(scores) / len(scores)


def rank_matrix(score_map, members=None):
    """Stack several members' scores into an (n_samples, n_members) rank matrix.

    Parameters
    ----------
    score_map : dict
        member name -> score array.
    members : list, optional
        Column order. Defaults to `sorted(score_map)`; pass explicitly to keep the
        order aligned with a weight vector.

    Returns
    -------
    R : ndarray, shape (n_samples, n_members)
    members : list
        The column order actually used.
    """
    members = list(members) if members is not None else sorted(score_map)
    lengths = {len(score_map[m]) for m in members}
    assert len(lengths) == 1, f"members have different lengths: {lengths}"
    R = np.column_stack([to_rank(score_map[m]) for m in members])
    return R, members


def blend(R, weights):
    """Weighted sum of member ranks.

    Parameters
    ----------
    R : ndarray, shape (n_samples, n_members)
    weights : array-like, shape (n_members,)

    Returns
    -------
    ndarray, shape (n_samples,)
        The blended score. Only its ordering is used downstream.
    """
    return R @ np.asarray(weights, dtype=np.float64)


def macro_f1_at_share(scores, y_true, share):
    """Macro F1 when the top `share` of rows by score are labelled machine.

    The local mirror of `write_at_share`: threshold by quantile rather than by a
    fixed probability cutoff, so the predicted class balance is held fixed and the
    metric reflects ranking quality rather than calibration. Use the evaluation
    set's OWN true share - the holdout inherits train's 0.6252, which is not the
    test set's ~0.53.

    Parameters
    ----------
    scores : array-like, shape (n_samples,)
    y_true : array-like, shape (n_samples,)
    share : float
        Fraction of rows to predict as class 1.

    Returns
    -------
    float
    """
    scores = np.asarray(scores, dtype=np.float64)
    thr = np.quantile(scores, 1 - share)
    return f1_score(y_true, (scores >= thr).astype(int), average="macro")


def _compositions(total, parts):
    """Yield every way to write `total` as an ordered sum of `parts` non-negatives."""
    if parts == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in _compositions(total - i, parts - 1):
            yield (i,) + rest


def search_weights(R, y_true, step=0.1, metric=roc_auc_score):
    """Exhaustive sweep of the weight simplex, scored by `metric`.

    With ranks as inputs the blend is scale-free, so weights are constrained to sum
    to 1 and are directly readable as member importances. A member the sweep gives
    weight 0 is one the ensemble does not want - which is how ComplementNB gets to
    argue for itself rather than being pre-rejected the way 07 rejected it.

    Cost is the number of compositions: 3,003 for 6 members at step 0.1, a few
    seconds. Halving the step to 0.05 raises that to ~53,000 - do that only to
    refine around an already-chosen region.

    Parameters
    ----------
    R : ndarray, shape (n_samples, n_members)
    y_true : array-like, shape (n_samples,)
    step : float, default 0.1
        Weight granularity; 1/step must be a whole number.
    metric : callable, default roc_auc_score
        `(y_true, scores) -> float`, higher is better.

    Returns
    -------
    pd.DataFrame
        One row per weight vector: w0..wk plus "score", sorted best first.
    """
    n_steps = int(round(1 / step))
    assert abs(n_steps * step - 1.0) < 1e-9, f"1/step must be a whole number: {step}"
    n_members = R.shape[1]

    rows = []
    for comp in _compositions(n_steps, n_members):
        w = np.array(comp, dtype=np.float64) / n_steps
        rows.append({**{f"w{i}": w[i] for i in range(n_members)},
                     "score": float(metric(y_true, R @ w))})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def fit_stacker(R, y_true):
    """Non-negative least-squares meta-learner over member ranks.

    The learned-weight alternative to `search_weights`, for when the member count
    makes an exhaustive sweep coarse. Non-negativity matters: members here are
    correlated 0.9+, and an unconstrained fit will happily assign large opposing
    weights that fit OOF noise and do not transport.

    Uses `LinearRegression(positive=True)` rather than a logistic meta-learner -
    sklearn's `LogisticRegression` has no non-negativity constraint, and since only
    the blend's ordering is used, the squared-error fit's monotone output is
    equivalent for this purpose.

    Parameters
    ----------
    R : ndarray, shape (n_samples, n_members)
    y_true : array-like, shape (n_samples,)

    Returns
    -------
    weights : ndarray, shape (n_members,)
        Non-negative and normalized to sum to 1, so they are comparable with
        `search_weights` output.
    """
    fit = LinearRegression(positive=True, fit_intercept=True).fit(R, y_true)
    w = np.clip(fit.coef_, 0, None)
    total = w.sum()
    assert total > 0, "stacker assigned zero weight to every member"
    return w / total


def diversity_table(score_map, y_true, members=None):
    """Pairwise Spearman rank correlation and matched-share label disagreement.

    Mirrors notebook 07 section 3, but on the scale the combiner actually uses.
    07 reported Pearson correlation of probabilities and disagreement at a 0.5
    cutoff - both of which confound genuine ranking diversity with the members'
    differing calibration. Disagreement here is measured with every member
    thresholded to the SAME share (the evaluation set's true class balance), so it
    counts only rows the members genuinely order differently.

    Parameters
    ----------
    score_map : dict
        member name -> score array.
    y_true : array-like, shape (n_samples,)
        Used for its class balance, which sets the shared threshold.
    members : list, optional

    Returns
    -------
    corr : pd.DataFrame
        Spearman rank correlation between members.
    disagree : pd.DataFrame
        Fraction of rows given different labels at matched share.
    """
    R, members = rank_matrix(score_map, members)
    share = float(np.mean(y_true))

    rho = np.asarray(spearmanr(R).statistic)
    if rho.ndim == 0:
        # scipy returns a bare scalar for exactly 2 columns, not a 2x2 matrix.
        rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    corr = pd.DataFrame(rho, index=members, columns=members)

    labels = {m: (R[:, i] >= np.quantile(R[:, i], 1 - share)).astype(int)
              for i, m in enumerate(members)}
    disagree = pd.DataFrame(
        [[float((labels[a] != labels[b]).mean()) for b in members] for a in members],
        index=members, columns=members,
    )
    return corr, disagree


def summarize_members(score_map, y_true, members=None):
    """Per-member AUC and matched-share macro F1 - the single-member baseline table.

    Parameters
    ----------
    score_map : dict
    y_true : array-like
    members : list, optional

    Returns
    -------
    pd.DataFrame
        Columns: member, auc, f1_at_share. Sorted by AUC descending.
    """
    members = list(members) if members is not None else sorted(score_map)
    share = float(np.mean(y_true))
    rows = [{"member": m,
             "auc": float(roc_auc_score(y_true, score_map[m])),
             "f1_at_share": macro_f1_at_share(score_map[m], y_true, share)}
            for m in members]
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def seed_average(score_arrays):
    """Rank-average several fits of the same model that differ only by seed.

    Near-free variance reduction: the fits are highly correlated, so this trims
    seed noise from the ranking without changing what the model is. Averaging on
    ranks rather than probabilities keeps it consistent with the rest of the
    module.

    Parameters
    ----------
    score_arrays : sequence of array-like
        One score vector per seed, all the same length.

    Returns
    -------
    ndarray, shape (n_samples,)
    """
    return np.mean([to_rank(s) for s in score_arrays], axis=0)
