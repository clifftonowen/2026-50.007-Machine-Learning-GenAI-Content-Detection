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
import time
from pathlib import Path

import numpy as np
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


def run_search(model, owner, build_estimator, sampler, X, y, cv, budget_seconds):
    """Run trials from `sampler` until a wall-clock budget is spent, then stop.

    Round 4 gives each teammate a fixed compute budget rather than a fixed trial
    count, because trial cost varies several-fold across machines and model
    configs - a count that fits one laptop overruns another. Since `run_trial`
    writes each result the moment it finishes, stopping partway through loses
    nothing and the team merge is unaffected: a short run simply contributes
    fewer trials.

    Before starting each trial the projected finish time is checked against the
    budget using the mean duration of the trials run so far, so the budget is a
    real cap rather than one that a single slow trial can blow past.

    Cached trials (already on disk, e.g. after a `git pull`) return instantly and
    are not counted when estimating trial duration.

    Parameters
    ----------
    model : str
        Short model name used in the trial filenames, e.g. "lightgbm_v2_stage1".
        Use a name no existing search already uses - `load_trials` globs on it,
        so reusing a key silently merges two different search spaces.
    owner : str
        Teammate identifier, e.g. "jovyan_lr_lo".
    build_estimator : callable
        `params -> estimator`. Called fresh per trial; `run_trial` does not call
        `set_params` itself.
    sampler : iterable of dict
        Parameter combinations, e.g. a `ParameterSampler` or `ParameterGrid`.
    X, y : array-like
        Training data, already restricted to the dev split by the caller.
    cv : cross-validation splitter
    budget_seconds : float
        Wall-clock cap for the whole loop.

    Returns
    -------
    pd.DataFrame
        One row per trial run in THIS call (not the merged history - use
        `load_trials` for that), sorted by mean descending.
    """
    start = time.monotonic()
    records, durations = [], []

    for i, params in enumerate(sampler, start=1):
        elapsed = time.monotonic() - start
        projected = np.mean(durations) if durations else 0.0
        if elapsed + projected > budget_seconds:
            print(f"\nBudget reached after {elapsed / 60:.1f} min "
                  f"({len(records)} trials this run) - stopping before trial {i}.")
            break

        t0 = time.monotonic()
        record = run_trial(model, owner, build_estimator(params), params, X, y, cv)
        took = time.monotonic() - t0
        records.append(record)
        # A cached hit returns in milliseconds and would drag the estimate down.
        if took > 1.0:
            durations.append(took)

        elapsed = time.monotonic() - start
        rate = len(records) / (elapsed / 3600) if elapsed > 0 else 0.0
        print(f"[{i}] mean={record['mean']:.4f} std={record['std']:.4f}  "
              f"({took:.0f}s, {elapsed / 60:.1f}/{budget_seconds / 60:.0f} min, "
              f"{rate:.0f} trials/hr)")
    else:
        print(f"\nSampler exhausted after {(time.monotonic() - start) / 60:.1f} min "
              f"({len(records)} trials) - budget was not the limit.")

    cols = ["owner", "params", "mean", "std", "scores"]
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records)[cols]
    return df.sort_values("mean", ascending=False).reset_index(drop=True)


def oof_path(name: str, owner: str) -> Path:
    """Path for one ensemble member's out-of-fold scores.

    Parameters
    ----------
    name : str
        Member name, e.g. "lightgbm_v2". Must not contain "__".
    owner : str
        Teammate identifier, kept in the filename so provenance survives.

    Returns
    -------
    Path
    """
    assert "__" not in name, f"'__' is the name/owner separator: {name}"
    paths.OOF.mkdir(parents=True, exist_ok=True)
    return paths.OOF / f"{name}__{owner}.npy"


def save_oof(name: str, owner: str, scores) -> Path:
    """Persist one member's out-of-fold scores for the team to merge via git.

    Stored as float32: 16,000 values is 64 KB, small enough to track in git and
    merge exactly like the trial JSONs. That is what lets three people generate
    different ensemble members in parallel and have notebook 10 pick them all up
    after a `git pull`, with nobody re-running anyone else's fits.

    Scores may be probabilities OR raw decision-function margins - the ensemble
    combines members on ranks, so only the ordering matters.

    Parameters
    ----------
    name : str
        Member name, e.g. "logreg_elasticnet".
    owner : str
    scores : array-like, shape (n_dev,)

    Returns
    -------
    Path
        Where it was written.
    """
    scores = np.asarray(scores, dtype=np.float32)
    assert scores.ndim == 1, scores.shape
    assert np.isfinite(scores).all(), "non-finite OOF scores"
    path = oof_path(name, owner)
    np.save(path, scores)
    return path


def load_oof(name: str):
    """Load one member's out-of-fold scores, whoever produced them.

    Parameters
    ----------
    name : str
        Member name, e.g. "xgboost".

    Returns
    -------
    ndarray, shape (n_dev,)

    Raises
    ------
    FileNotFoundError
        If no teammate has produced this member yet.
    ValueError
        If two teammates both produced it - that is duplicated compute and an
        ambiguous choice, so it should be resolved by deleting one rather than
        silently picking.
    """
    matches = sorted(paths.OOF.glob(f"{name}__*.npy"))
    if not matches:
        raise FileNotFoundError(
            f"no OOF file for '{name}' - has its owner run their section and pushed?")
    if len(matches) > 1:
        raise ValueError(
            f"'{name}' produced by multiple owners: {[m.name for m in matches]}. "
            "Delete all but one.")
    return np.load(matches[0])


def available_oof() -> pd.DataFrame:
    """List every merged OOF member currently on disk.

    Returns
    -------
    pd.DataFrame
        Columns: member, owner, n. Empty (with those columns) if none exist yet.
    """
    rows = []
    for f in sorted(paths.OOF.glob("*__*.npy")):
        member, owner = f.stem.split("__", 1)
        rows.append({"member": member, "owner": owner, "n": len(np.load(f))})
    return pd.DataFrame(rows, columns=["member", "owner", "n"])
