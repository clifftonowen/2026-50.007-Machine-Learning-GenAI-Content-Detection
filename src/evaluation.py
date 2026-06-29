"""Shared evaluation protocol — the locked Macro-F1 CV.

Fixed once so every notebook scores models the same way: 5-fold StratifiedKFold,
shuffle=True, random_state=42, scoring on Macro F1. Import these instead of
redefining the protocol per notebook.
"""

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
