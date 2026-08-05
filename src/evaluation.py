"""Shared evaluation protocol — the locked Macro-F1 CV.

Fixed once so every notebook scores models the same way: 5-fold StratifiedKFold,
shuffle=True, random_state=42, scoring on Macro F1. Import these instead of
redefining the protocol per notebook.
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_STATE = 42
N_SPLITS = 5
SCORING = "f1_macro"


def make_cv():
    """Return the project-standard StratifiedKFold splitter.

    Returns
    -------
    StratifiedKFold
        5 splits, shuffled, seeded with random_state=42.
    """
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def macro_f1(y_true, y_pred) -> float:
    """Macro F1 — the primary metric (unweighted mean of class-0 and class-1 F1).

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)

    Returns
    -------
    float
    """
    return f1_score(y_true, y_pred, average="macro")


def evaluate_model_cv(estimator, X, y, *, cv=None, scoring=SCORING):
    """Cross-validate an sklearn estimator under the locked protocol.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    cv : splitter, optional
        Defaults to `make_cv()`.
    scoring : str, default "f1_macro"

    Returns
    -------
    dict
        {"scores": ndarray, "mean": float, "std": float}
    """
    if cv is None:
        cv = make_cv()
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    return {"scores": scores, "mean": float(scores.mean()), "std": float(scores.std())}


class FoldSplitter:
    """Wrap an explicit fold list in the sklearn splitter interface.

    `clustering.cluster_cv` returns folds as a plain list of (train_idx, test_idx),
    which `cross_val_score` accepts but `combiners.cv_evaluate` does not (it calls
    `cv.split(M, y)`). This adapter lets the grouped-fold protocols drive the
    combiner machinery without changing either side.

    Parameters
    ----------
    folds : list of (ndarray, ndarray)
    """

    def __init__(self, folds):
        self.folds = list(folds)

    def split(self, X=None, y=None, groups=None):
        yield from self.folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)


def oof_from_folds(estimator, X, y, folds, *, scorer=None):
    """Out-of-fold scores under an explicit fold list, aligned to the full array.

    `cross_val_predict` requires folds that partition the data; grouped folds from
    `clustering.cluster_cv` may skip a band, so this fills only the held-out rows
    and leaves the rest NaN for the caller to check.

    Parameters
    ----------
    estimator : sklearn-compatible estimator (cloned per fold)
    X : array or sparse matrix, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    folds : list of (train_idx, test_idx)
    scorer : callable, optional
        `(fitted_estimator, X_test) -> scores`. Defaults to
        `ensemble.member_score` (probability or margin - only ordering is used).

    Returns
    -------
    ndarray of float, shape (n_samples,)
        NaN where no fold held the row out.
    """
    from sklearn.base import clone

    from . import ensemble

    if scorer is None:
        scorer = ensemble.member_score
    y = np.asarray(y)
    out = np.full(len(y), np.nan)
    for tr, te in folds:
        model = clone(estimator).fit(X[tr], y[tr])
        out[te] = scorer(model, X[te])
    return out
