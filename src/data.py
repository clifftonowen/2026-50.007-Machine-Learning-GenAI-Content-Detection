"""Feature-matrix loading for Tasks 1-3.

Loads the course-supplied pre-computed TF-IDF matrices directly. Do NOT
re-tokenize / re-fit a vectorizer on raw text for Tasks 1-3 — the features are
the model input. Skeleton: confirm the exact column names on first load and
fill in the schema-dependent bits.
"""

import numpy as np
import pandas as pd

from . import paths

RANDOM_STATE = 42  # fixed seed everywhere


def _feature_columns(df: pd.DataFrame) -> list:
    """Return the TF-IDF feature column names (everything except id/label).

    Parameters
    ----------
    df : pd.DataFrame
        A loaded *_features.csv frame.

    Returns
    -------
    list
        Ordered feature-column names. Confirm naming in 01_eda (feat_0.., 0.., …).
    """
    return [c for c in df.columns if c not in {"id", "label"}]


def load_train_features():
    """Load training features and labels from data/raw/train_features.csv.

    Returns
    -------
    X : ndarray, shape (n_train, 5000)
    y : ndarray, shape (n_train,)
        Labels: 1 = machine-generated, 0 = human-authored.
    ids : ndarray, shape (n_train,)
    """
    # id is not numeric (mix of UUID and digit-only strings); force str so pandas
    # never coerces it and silently misaligns downstream submissions
    df = pd.read_csv(paths.TRAIN_FEATURES_CSV, dtype={"id": str})
    cols = _feature_columns(df)
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    ids = df["id"].to_numpy()
    return X, y, ids


def load_test_features():
    """Load holdout features from data/raw/test_features.csv.

    Returns
    -------
    X : ndarray, shape (n_test, 5000)
    ids : ndarray, shape (n_test,)
    """
    df = pd.read_csv(paths.TEST_FEATURES_CSV, dtype={"id": str})
    cols = _feature_columns(df)
    X = df[cols].to_numpy(dtype=np.float64)
    ids = df["id"].to_numpy()
    return X, ids


def write_submission(ids, labels, filename: str):
    """Write a Kaggle submission CSV with the required `id,label` header.

    Parameters
    ----------
    ids : array-like
        Row ids, aligned with `labels`.
    labels : array-like
        Predicted labels (1 = machine, 0 = human).
    filename : str
        File name written under submissions/ (e.g. "LogReg_Prediction.csv").
    """
    paths.SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "label": np.asarray(labels, dtype=np.int64)})
    out.to_csv(paths.SUBMISSIONS / filename, index=False)
