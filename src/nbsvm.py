"""NBSVM: Naive-Bayes log-count-ratio feature scaling + a linear classifier.

Wang & Manning (2012, "Baselines and Bigrams") showed that scaling each column of a
bag-of-ngrams matrix by its class log-count ratio, then fitting a linear model, is a
consistently strong text-classification baseline. The scaling injects the NB evidence
per feature while the discriminative fit handles correlated features - a different
inductive bias from both the plain linear models and the boosted trees this project
has tried, which is exactly what an ensemble member should bring.

Only valid on non-negative features (the ratio is a ratio of summed counts), so use it
on the TF-IDF n-gram blocks (H, I), not the signed style moments.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression


class NBSVM(BaseEstimator, ClassifierMixin):
    """Log-count-ratio column scaling wrapped around any linear estimator.

    Parameters
    ----------
    estimator : sklearn linear classifier, optional
        Fit on the scaled matrix. Defaults to LogisticRegression(C=1.0,
        class_weight="balanced", max_iter=2000). Must expose `decision_function`
        or `predict_proba`.
    alpha : float, default 1.0
        Laplace smoothing added to the per-class feature sums before the ratio.
    """

    def __init__(self, estimator=None, alpha=1.0):
        self.estimator = estimator
        self.alpha = alpha

    def _ratio(self, X, y):
        """Column log-count ratio r = log(p/|p|_1) - log(q/|q|_1)."""
        X = sp.csr_matrix(X)
        assert X.min() >= 0, "NBSVM needs non-negative features"
        p = self.alpha + np.asarray(X[y == 1].sum(axis=0)).ravel()
        q = self.alpha + np.asarray(X[y == 0].sum(axis=0)).ravel()
        return np.log((p / p.sum()) / (q / q.sum()))

    def _scale(self, X):
        return sp.csr_matrix(sp.csr_matrix(X).multiply(self.r_))

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.r_ = self._ratio(X, y)
        base = self.estimator if self.estimator is not None else LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000)
        self.model_ = clone(base).fit(self._scale(X), y)
        return self

    def decision_function(self, X):
        Z = self._scale(X)
        if hasattr(self.model_, "decision_function"):
            return self.model_.decision_function(Z)
        return self.model_.predict_proba(Z)[:, 1]

    def predict(self, X):
        return self.model_.predict(self._scale(X))
