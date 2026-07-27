"""Feature-matrix loading for Tasks 1-3.

Loads the course-supplied pre-computed TF-IDF matrices directly. Do NOT
re-tokenize / re-fit a vectorizer on raw text for Tasks 1-3 — the features are
the model input. Skeleton: confirm the exact column names on first load and
fill in the schema-dependent bits.

Two loading paths share one column ordering:

- **dense** (default) — float64 ndarray, what notebooks 01-09 were run against.
  Left as the default so those notebooks keep reproducing byte-identically;
  they are the Task 4 record.
- **sparse** (`sparse=True`) — float32 CSR. 01_eda measured the matrix at 98.6%
  sparse, so the dense train matrix is 800 MB of mostly zeros. CSR cuts that
  ~50x and is what makes a time-boxed hyperparameter search worth running.
  LightGBM, XGBoost, LogisticRegression and LinearSVC all accept CSR directly,
  and row indexing (`X[dev_idx]`) works unchanged.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import paths

RANDOM_STATE = 42  # fixed seed everywhere

# Rows per chunk when streaming a *_features.csv into CSR. Small enough that peak
# memory stays ~50 MB instead of the ~800 MB a single dense read would need.
_CHUNK_ROWS = 2000


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


def _build_csr(csv_path, meta_cols):
    """Stream a *_features.csv into CSR without ever materializing it densely.

    Parameters
    ----------
    csv_path : Path
        A data/raw/*_features.csv file.
    meta_cols : list
        Non-feature columns to carry alongside, e.g. ["id", "label"].

    Returns
    -------
    X : scipy.sparse.csr_matrix, dtype float32
    meta : pd.DataFrame
        The `meta_cols`, row-aligned with X.
    """
    blocks, meta, cols = [], [], None
    # id is not numeric (mix of UUID and digit-only strings); force str so pandas
    # never coerces it and silently misaligns downstream submissions
    for chunk in pd.read_csv(csv_path, dtype={"id": str}, chunksize=_CHUNK_ROWS):
        if cols is None:
            # Same helper the dense path uses, so column ORDER is identical by
            # construction - a mismatch here would silently invalidate every
            # persisted model without raising anything.
            cols = _feature_columns(chunk)
        blocks.append(sp.csr_matrix(chunk[cols].to_numpy(dtype=np.float32)))
        meta.append(chunk[meta_cols])
    return sp.vstack(blocks, format="csr"), pd.concat(meta, ignore_index=True)


def _load_csr_cached(csv_path, stem, meta_cols):
    """Load a feature CSV as CSR, caching the result under data/processed/.

    Parsing the 210 MB train CSV takes long enough that four teammates each doing
    it once per run is real budget. The cache is rebuilt whenever the source CSV
    is newer than it, so a re-downloaded data/raw/ can never go unnoticed.

    Parameters
    ----------
    csv_path : Path
    stem : str
        Cache filename stem, e.g. "train_features".
    meta_cols : list

    Returns
    -------
    X : scipy.sparse.csr_matrix
    meta : pd.DataFrame
    """
    paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    matrix_path = paths.DATA_PROCESSED / f"{stem}_csr.npz"
    meta_path = paths.DATA_PROCESSED / f"{stem}_meta.npz"

    fresh = (
        matrix_path.exists()
        and meta_path.exists()
        and matrix_path.stat().st_mtime >= csv_path.stat().st_mtime
    )
    if fresh:
        cached = np.load(meta_path, allow_pickle=True)
        meta = pd.DataFrame({c: cached[c] for c in meta_cols})
        return sp.load_npz(matrix_path), meta

    X, meta = _build_csr(csv_path, meta_cols)
    sp.save_npz(matrix_path, X)
    np.savez(meta_path, **{c: meta[c].to_numpy() for c in meta_cols})
    return X, meta


def load_train_features(*, sparse: bool = False):
    """Load training features and labels from data/raw/train_features.csv.

    Parameters
    ----------
    sparse : bool, default False
        If True, return a float32 CSR matrix (cached under data/processed/)
        instead of the dense float64 ndarray. The default is dense so notebooks
        01-09 reproduce exactly as they were run.

    Returns
    -------
    X : ndarray or csr_matrix, shape (n_train, 5000)
    y : ndarray, shape (n_train,)
        Labels: 1 = machine-generated, 0 = human-authored.
    ids : ndarray, shape (n_train,)
    """
    if sparse:
        X, meta = _load_csr_cached(paths.TRAIN_FEATURES_CSV, "train_features",
                                   ["id", "label"])
        return X, meta["label"].to_numpy(dtype=np.int64), meta["id"].to_numpy()

    # id is not numeric (mix of UUID and digit-only strings); force str so pandas
    # never coerces it and silently misaligns downstream submissions
    df = pd.read_csv(paths.TRAIN_FEATURES_CSV, dtype={"id": str})
    cols = _feature_columns(df)
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    ids = df["id"].to_numpy()
    return X, y, ids


def load_test_features(*, sparse: bool = False):
    """Load holdout features from data/raw/test_features.csv.

    Parameters
    ----------
    sparse : bool, default False
        See `load_train_features`.

    Returns
    -------
    X : ndarray or csr_matrix, shape (n_test, 5000)
    ids : ndarray, shape (n_test,)
    """
    if sparse:
        X, meta = _load_csr_cached(paths.TEST_FEATURES_CSV, "test_features", ["id"])
        return X, meta["id"].to_numpy()

    df = pd.read_csv(paths.TEST_FEATURES_CSV, dtype={"id": str})
    cols = _feature_columns(df)
    X = df[cols].to_numpy(dtype=np.float64)
    ids = df["id"].to_numpy()
    return X, ids


def check_sparse_path(n_rows: int = 100):
    """Assert the sparse loader agrees with the dense one. Run once per machine.

    The sparse path is the one change in round 4 that could silently corrupt
    every downstream result - a shifted column order or a dropped row would train
    happily and score badly with nothing raising. This compares the two paths
    directly on the first `n_rows` rows (read densely with `nrows=`, so it stays
    cheap) and checks the sparsity against 01_eda's measured 98.6%.

    Parameters
    ----------
    n_rows : int, default 100
        Rows to compare element-wise.

    Returns
    -------
    dict
        {"shape", "density", "max_abs_diff"} - printed by the caller.
    """
    X, y, ids = load_train_features(sparse=True)
    assert X.shape == (20000, 5000), X.shape
    assert len(y) == len(ids) == 20000, (len(y), len(ids))

    density = X.nnz / (X.shape[0] * X.shape[1])
    assert 0.010 < density < 0.020, f"density {density:.4f} - expected ~0.014 (01_eda)"

    head = pd.read_csv(paths.TRAIN_FEATURES_CSV, dtype={"id": str}, nrows=n_rows)
    dense_head = head[_feature_columns(head)].to_numpy(dtype=np.float64)
    max_abs_diff = float(np.abs(X[:n_rows].toarray() - dense_head).max())
    assert max_abs_diff < 1e-6, f"sparse/dense disagree by {max_abs_diff}"
    assert list(ids[:n_rows]) == list(head["id"]), "id order differs between paths"

    return {"shape": X.shape, "density": round(density, 5),
            "max_abs_diff": max_abs_diff}


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
