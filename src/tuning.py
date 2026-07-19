"""Incremental, resumable, team-mergeable hyperparameter trial tracking.

One JSON file per (model, params) combination, written as soon as it's scored. Reruns
skip trials that already have a result file (resumable - a killed sweep only loses the
trial it was on, not everything before it), and teammates' parallel trials merge
automatically by dropping their files into the same directory (mergeable - no manual
CSV-concatenation step for anyone to get wrong). Extracted from notebooks/05_tuning.ipynb
since both search stages need it.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import evaluation, paths


def trial_id(params: dict) -> str:
    """Deterministic short hash of a sorted param dict, used as the trial filename.

    Parameters
    ----------
    params : dict
        Hyperparameter values for one trial. Sorted before hashing so key order never
        changes the id.

    Returns
    -------
    str
        8-character hex digest.
    """
    key = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def trial_path(model: str, owner: str, params: dict) -> Path:
    """Path for one trial's result file: tuning_trials/<model>_<owner>_<trial_id>.json.

    Parameters
    ----------
    model : str
        Short model name, e.g. "lightgbm".
    owner : str
        Teammate initials/name, kept in the filename so provenance survives file
        sharing/renaming.
    params : dict
        The trial's hyperparameters (hashed into the filename via trial_id).

    Returns
    -------
    Path
    """
    paths.TUNING_TRIALS.mkdir(parents=True, exist_ok=True)
    return paths.TUNING_TRIALS / f"{model}_{owner}_{trial_id(params)}.json"


def run_trial(model: str, owner: str, estimator, params: dict, X, y, cv) -> dict:
    """Score one hyperparameter trial under the locked CV protocol, caching the result.

    If a result file for this exact (model, params) already exists, loads and returns it
    instead of recomputing - this is what makes an interrupted sweep resumable, and what
    lets teammates skip trials someone else already ran (e.g. after a git pull).

    Parameters
    ----------
    model : str
        Short model name used in the trial filename, e.g. "lightgbm".
    owner : str
        Teammate initials/name.
    estimator : sklearn-compatible estimator
        Already constructed with `params` applied - this function does not call
        `set_params` itself, so the caller must build a fresh estimator per trial.
    params : dict
        The hyperparameters this trial represents; stored alongside the score for the
        merge step, and hashed into the cache filename. Values are coerced to native
        Python types (not numpy scalars) so the saved JSON round-trips cleanly.
    X, y : array-like
        Training data (already restricted to the dev split by the caller).
    cv : cross-validation splitter
        Passed straight through to evaluation.evaluate_model_cv.

    Returns
    -------
    dict
        {"model", "owner", "params", "mean", "std", "scores"}
    """
    clean_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
    path = trial_path(model, owner, clean_params)
    if path.exists():
        return json.loads(path.read_text())

    result = evaluation.evaluate_model_cv(estimator, X, y, cv=cv)
    record = {
        "model": model,
        "owner": owner,
        "params": clean_params,
        "mean": result["mean"],
        "std": result["std"],
        "scores": result["scores"].tolist(),
    }
    path.write_text(json.dumps(record, indent=2))
    return record


def load_trials(model: str) -> pd.DataFrame:
    """Merge every teammate's trial files for one model into a single sorted table.

    Globs tuning_trials/<model>_*.json - this is the merge step. As long as everyone's
    trial files are present in the directory (e.g. after a git pull), no manual
    concatenation is needed.

    Parameters
    ----------
    model : str
        Short model name, e.g. "lightgbm".

    Returns
    -------
    pd.DataFrame
        One row per trial, columns: owner, params, mean, std, scores. Sorted by mean
        descending. Empty DataFrame (with these columns) if no trials exist yet.
    """
    records = [json.loads(f.read_text()) for f in sorted(paths.TUNING_TRIALS.glob(f"{model}_*.json"))]
    cols = ["owner", "params", "mean", "std", "scores"]
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records)[cols]
    return df.sort_values("mean", ascending=False).reset_index(drop=True)
