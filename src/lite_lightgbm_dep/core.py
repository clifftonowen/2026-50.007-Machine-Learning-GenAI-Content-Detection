"""Shared numerical definitions for :mod:`src.lite_lightgbm`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

Matrix = np.ndarray | sp.spmatrix
ClassWeight = str | dict[int, float] | None
EPSILON = 1e-15


@dataclass(frozen=True, slots=True)
class LiteLightGBMConfig:
    """Immutable snapshot of the estimator's supported hyperparameters."""

    n_estimators: int = 100
    learning_rate: float = 0.1
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    min_child_weight: float = 1e-3
    min_split_gain: float = 0.0
    max_bin: int = 255
    min_data_in_bin: int = 3
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    colsample_bytree: float = 1.0
    subsample: float = 1.0
    subsample_freq: int = 0
    class_weight: ClassWeight = None
    random_state: int | None = None

def sigmoid(raw_scores: np.ndarray) -> np.ndarray:
    """Compute a numerically stable element-wise logistic sigmoid."""
    # Evaluate positive and negative inputs separately so that neither
    # exponential receives a positive argument large enough to overflow.
    scores = np.asarray(raw_scores)
    result = np.empty_like(scores, dtype=np.float64)

    positive = scores >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))

    negative = ~positive
    exp_scores = np.exp(scores[negative])
    result[negative] = exp_scores / (1.0 + exp_scores)
    return result


def soft_threshold(values: np.ndarray | float, reg_alpha: float):
    """Apply the L1 soft-threshold operator used by leaf values and gains."""
    return np.sign(values) * np.maximum(np.abs(values) - reg_alpha, 0.0)


def binary_gradients_hessians(
    raw_scores: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted binary-log-loss gradients and Hessians."""
    probabilities = sigmoid(raw_scores)
    gradients = sample_weight * (probabilities - labels)
    hessians = sample_weight * probabilities * (1.0 - probabilities)
    return gradients, hessians

