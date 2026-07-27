"""The four combiner families explored in 10_stacked_ensemble.ipynb, plus the
harness that scores them honestly and the ledger that merges four people's results.

`ensemble.py` supplies the primitives (ranking, blending, the simplex sweep, the NNLS
stacker). This module supplies everything built on top of them for round 4, where four
teammates each sweep a different *family* of combiner against one shared out-of-fold
matrix:

    lane A  weights      which weight vector      sweep / nnls / hill-climb / equal
    lane B  voting       which decision rule      hard / weighted / k-of-n / soft
    lane C  meta         which learned combiner   logistic / tree / gbm / forward-select
    lane D  aggregation  which operator           mean / median / trimmed / power / ...

Two design decisions are worth stating up front, because both are corrections to how
notebook 10 originally scored its combiners.

**Combiners are scored out of fold.** Notebook 10 fitted its weights on the same 16,000
OOF rows it then reported them on, and judged the resulting bias "small". With four
people sweeping dozens of configurations and a merged leaderboard that takes the maximum
over all of them, it is not small: whichever lane has the most free parameters wins on
optimism alone, and the winner would be a search-space artifact rather than a better
combiner. `cv_evaluate` refits every combiner inside the locked folds and reports the
held-out score, so a 3,003-point sweep and a parameter-free median are compared on equal
terms. Each result also carries its in-sample score, so the size of the bias is measured
rather than assumed.

**Hard votes are tie-broken on rank.** Submissions are thresholded by share: the top
`share * n` rows are labelled machine. A majority vote over k members takes only k+1
distinct values, so the cut lands *inside* a block of tied rows, and which of those rows
end up labelled machine is then decided by their position in the file rather than by any
model. With six members that block is typically a fifth of the test set. Every discrete
combiner here therefore adds a rank-based tiebreak, scaled to be provably too small to
reorder distinct vote levels (`_break_ties`), so the vote decides the levels and the
members' own ranking decides within them. `tied_fraction_at_share` measures how much of a
submission a combiner would otherwise leave to row order.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from . import ensemble, paths

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _break_ties(discrete, M):
    """Make a discrete-valued combiner score continuous without reordering it.

    Adds a fraction of the row's mean rank, scaled to strictly less than the
    smallest gap between distinct values of `discrete`. Rows on the same vote level
    are therefore ordered by how strongly the members ranked them, while the vote
    levels themselves stay exactly as the vote rule left them.

    Without this, the share cut falls inside a tied block and the rows inside it are
    labelled by file position - see the module docstring and `tied_fraction_at_share`.

    Parameters
    ----------
    discrete : ndarray, shape (n_samples,)
        Vote counts or any other coarse score.
    M : ndarray, shape (n_samples, n_members)
        The member matrix the tiebreak is drawn from.

    Returns
    -------
    ndarray, shape (n_samples,)
    """
    discrete = np.asarray(discrete, dtype=np.float64)
    uniq = np.unique(discrete)
    gap = float(np.min(np.diff(uniq))) if len(uniq) > 1 else 1.0
    return discrete + 0.49 * gap * ensemble.to_rank(M.mean(axis=1))


def tied_fraction_at_share(scores, share):
    """Fraction of rows whose label is decided by file position, not by the scores.

    The share cut labels the top `round(share * n)` rows. If the score at that cut is
    shared by a block of rows, the block straddles the boundary and `threshold_at_share`
    splits it by row order. This returns how big that block is, as a fraction of all
    rows - 0 for a combiner with distinct scores, and roughly 1/(k+1) for an
    untie-broken vote over k members.

    Report it next to any discrete combiner: a submission where a fifth of the labels
    come from row order is not measuring the ensemble.

    Parameters
    ----------
    scores : array-like, shape (n_samples,)
    share : float

    Returns
    -------
    float
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    k = int(np.clip(round(share * n), 0, n))
    if k == 0 or k == n:
        return 0.0
    cut = np.sort(scores)[::-1][k - 1]
    return float(np.mean(scores == cut))


def _labels_at_share(M, share):
    """Threshold every column of `M` independently to the same predicted share.

    Members disagree about which rows are machine-generated, but at a common share
    they agree about *how many*. That is the matched-share convention the whole
    project uses, applied per member so a vote counts genuine ordering differences
    rather than differences in calibration.

    Parameters
    ----------
    M : ndarray, shape (n_samples, n_members)
    share : float

    Returns
    -------
    ndarray, shape (n_samples, n_members)
        Integer 0/1 labels.
    """
    return np.column_stack([ensemble.threshold_at_share(M[:, j], share)
                            for j in range(M.shape[1])])


def _as_predict(fn, **info):
    """Attach a JSON-serializable `info` dict (learned weights, chosen subset) to a
    fitted combiner's predict callable, so `cv_evaluate` can report it."""
    fn.info = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in info.items()}
    return fn


# ---------------------------------------------------------------------------
# Lane A - weighted rank blending
# ---------------------------------------------------------------------------

def hill_climb_weights(M, y, *, step=0.05, max_rounds=50, metric=roc_auc_score):
    """Greedy coordinate ascent on the weight simplex.

    The middle ground between `ensemble.search_weights` (exhaustive, but its cost is
    combinatorial in the member count so the grid has to stay coarse) and
    `ensemble.fit_stacker` (a closed-form least-squares fit, but it optimises squared
    error rather than the ranking metric anyone cares about). Hill climbing optimises
    AUC directly at a fine step size, at the price of finding a local optimum.

    Starts from equal weights, then repeatedly tries moving `step` of weight from one
    member to another and keeps any move that improves the metric.

    Parameters
    ----------
    M : ndarray, shape (n_samples, n_members)
    y : array-like, shape (n_samples,)
    step : float, default 0.05
        Weight moved per accepted step.
    max_rounds : int, default 50
        Cap on full passes; the loop also stops as soon as a pass accepts nothing.
    metric : callable, default roc_auc_score

    Returns
    -------
    ndarray, shape (n_members,)
        Non-negative weights summing to 1.
    """
    k = M.shape[1]
    w = np.full(k, 1.0 / k)
    best = float(metric(y, M @ w))

    for _ in range(max_rounds):
        improved = False
        for src in range(k):
            if w[src] < step:
                continue
            for dst in range(k):
                if src == dst:
                    continue
                trial = w.copy()
                trial[src] -= step
                trial[dst] += step
                score = float(metric(y, M @ trial))
                if score > best:
                    w, best, improved = trial, score, True
        if not improved:
            break
    return w


def _build_weights(config):
    """Lane A: every variant reduces to a weight vector, then a linear rank blend."""
    kind = config["kind"]

    def build(M, y):
        if kind == "equal":
            w = np.full(M.shape[1], 1.0 / M.shape[1])
        elif kind == "sweep":
            sweep = ensemble.search_weights(M, y, step=config.get("step", 0.1))
            w = sweep.iloc[0][[f"w{i}" for i in range(M.shape[1])]].to_numpy(float)
        elif kind == "nnls":
            w = ensemble.fit_stacker(M, y)
        elif kind == "hill_climb":
            w = hill_climb_weights(M, y, step=config.get("step", 0.05))
        else:
            raise KeyError(kind)
        return _as_predict(lambda Q, w=w: ensemble.blend(Q, w), weights=w)

    return build


# ---------------------------------------------------------------------------
# Lane B - voting mechanisms
# ---------------------------------------------------------------------------

def _build_vote(config):
    """Lane B: turn members into votes, then count them.

    `share` sets where each member's own machine/human line falls before voting.
    Voting at the evaluation set's true share is the neutral choice; sweeping it is
    a real knob, because a lower share makes every member's "machine" vote scarcer
    and therefore makes the vote count more conservative.
    """
    kind = config["kind"]
    share = config.get("share")

    def build(M, y):
        s = float(np.mean(y)) if share is None else float(share)
        info = {"share": s}

        if kind == "soft_vote":
            # The one rule that does not threshold: a plain mean of the member
            # columns. On raw probabilities this is exactly notebook 07's combiner,
            # kept here so the leaderboard shows what 07 would have scored.
            return _as_predict(lambda Q: Q.mean(axis=1), **info)

        if kind == "weighted_vote":
            # Weight each member's vote by its own AUC above chance, learned on the
            # training rows only. A member that cannot rank gets almost no say.
            edge = np.array([max(roc_auc_score(y, M[:, j]) - 0.5, 0.0)
                             for j in range(M.shape[1])])
            assert edge.sum() > 0, "every member scored at or below chance"
            w = edge / edge.sum()
            info["weights"] = w

            def predict(Q, w=w, s=s):
                return _break_ties(_labels_at_share(Q, s) @ w, Q)

            return _as_predict(predict, **info)

        if kind in ("hard_vote", "k_of_n"):
            k = config.get("k")
            if kind == "hard_vote":
                k = None  # majority is reported, not enforced, by the tiebreak

            def predict(Q, s=s, k=k):
                votes = _labels_at_share(Q, s).sum(axis=1).astype(np.float64)
                if k is not None:
                    # A k-of-n rule collapses to a binary indicator, which is even
                    # coarser than the vote count, so the tiebreak matters more here.
                    votes = (votes >= k).astype(np.float64)
                return _break_ties(votes, Q)

            info["k"] = k
            return _as_predict(predict, **info)

        raise KeyError(kind)

    return build


# ---------------------------------------------------------------------------
# Lane C - meta-learner stacking
# ---------------------------------------------------------------------------

def forward_select(M, y, *, max_members=None, metric=roc_auc_score):
    """Greedy forward selection of a member subset, combined by mean rank.

    Answers "which members belong in the blend" rather than "what weight should each
    get". Notebook 07 rejected ComplementNB by hand for being weak; this lets the
    data make that call, and reports it as an explicit subset rather than as a small
    weight buried in a vector.

    Parameters
    ----------
    M : ndarray, shape (n_samples, n_members)
    y : array-like, shape (n_samples,)
    max_members : int, optional
        Defaults to all of them.
    metric : callable, default roc_auc_score

    Returns
    -------
    list of int
        Column indices, in the order they were added.
    """
    k = M.shape[1]
    max_members = k if max_members is None else min(max_members, k)
    chosen, best = [], -np.inf

    while len(chosen) < max_members:
        gains = []
        for j in range(k):
            if j in chosen:
                continue
            score = float(metric(y, M[:, chosen + [j]].mean(axis=1)))
            gains.append((score, j))
        if not gains:
            break
        score, j = max(gains)
        if score <= best:
            break  # nothing left to add helps
        chosen, best = chosen + [j], score

    return chosen


def _build_meta(config):
    """Lane C: fit an actual classifier on the member ranks.

    The risk this lane carries, and which `cv_evaluate` exists to expose: a tree or
    a boosted model has enough capacity to memorise 16,000 rows of a 6-column matrix,
    so its in-sample score will look excellent while its held-out score does not move.
    Keep depths shallow and read the `auc_gap` column.
    """
    kind = config["kind"]

    def build(M, y):
        if kind == "forward_select":
            chosen = forward_select(M, y, max_members=config.get("max_members"))
            assert chosen, "forward selection chose no members"
            return _as_predict(lambda Q, c=chosen: Q[:, c].mean(axis=1), members=chosen)

        if kind == "logistic":
            model = LogisticRegression(C=config.get("C", 1.0), max_iter=2000,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)
        elif kind == "tree":
            model = DecisionTreeClassifier(max_depth=config.get("max_depth", 3),
                                           min_samples_leaf=config.get("min_samples_leaf", 50),
                                           class_weight="balanced",
                                           random_state=RANDOM_STATE)
        elif kind == "gbm":
            model = GradientBoostingClassifier(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 2),
                learning_rate=config.get("learning_rate", 0.1),
                random_state=RANDOM_STATE)
        else:
            raise KeyError(kind)

        model.fit(M, y)
        coefs = model.coef_[0] if hasattr(model, "coef_") else None
        importances = getattr(model, "feature_importances_", None)
        return _as_predict(lambda Q, m=model: m.predict_proba(Q)[:, 1],
                           **({"coef": coefs} if coefs is not None else {}),
                           **({"importances": importances} if importances is not None else {}))

    return build


# ---------------------------------------------------------------------------
# Lane D - rank aggregation operators
# ---------------------------------------------------------------------------

def aggregate(M, how, **kw):
    """Combine member columns with an order statistic instead of a weighted sum.

    Every operator here is unweighted, which is the point: lane A already asks which
    weights are best, so this lane isolates the effect of the aggregation function
    itself. The useful ones are the non-linear operators, because a weighted sum can
    never imitate them:

    - `median` / `trimmed` ignore outlying members, so one badly-ranked row from a
      weak member cannot drag a row that the others agree on.
    - `power` with p < 1 behaves like a soft minimum (a row must satisfy *every*
      member to score highly) and with p > 1 like a soft maximum (one enthusiastic
      member is enough). p = 1 is the plain mean.
    - `min` / `max` are those two limits taken literally, and are mostly here as the
      endpoints that show how far the family reaches.
    - `borda_topk` gives points only above a rank cutoff, so members vote on their
      confident rows and abstain elsewhere.

    Plain Borda count is deliberately absent: on normalized ranks it is the mean
    multiplied by the member count, so it would appear on the leaderboard as a
    distinct operator while being provably identical to `mean`.

    Parameters
    ----------
    M : ndarray, shape (n_samples, n_members)
        Member ranks in (0, 1].
    how : str
        One of mean, median, trimmed, power, geometric, min, max, borda_topk.
    **kw
        `trim` for trimmed (fraction removed from each end, default 0.2),
        `p` for power (default 1.0), `cut` for borda_topk (default 0.5).

    Returns
    -------
    ndarray, shape (n_samples,)
    """
    if how == "mean":
        return M.mean(axis=1)
    if how == "median":
        return np.median(M, axis=1)
    if how == "min":
        return M.min(axis=1)
    if how == "max":
        return M.max(axis=1)
    if how == "geometric":
        # Ranks are strictly positive, so the log is always defined.
        return np.exp(np.log(M).mean(axis=1))
    if how == "power":
        p = float(kw.get("p", 1.0))
        assert p != 0, "use how='geometric' for the p -> 0 limit"
        # Keep the outer 1/p root: for p < 0 it flips the ordering back the right way.
        return np.power(np.power(M, p).mean(axis=1), 1.0 / p)
    if how == "trimmed":
        trim = float(kw.get("trim", 0.2))
        cut = int(np.floor(trim * M.shape[1]))
        if cut == 0:
            return M.mean(axis=1)
        ordered = np.sort(M, axis=1)
        assert 2 * cut < M.shape[1], f"trim={trim} removes every member"
        return ordered[:, cut:M.shape[1] - cut].mean(axis=1)
    if how == "borda_topk":
        cut = float(kw.get("cut", 0.5))
        return np.maximum(M - cut, 0.0).sum(axis=1)
    raise KeyError(how)


def caruana_select(M, y, *, n_iters=25, metric=roc_auc_score):
    """Caruana-style greedy ensemble selection with replacement.

    Repeatedly adds whichever member most improves the running mean, allowing the
    same member to be picked again. Selecting with replacement is what makes this a
    weighting method and not just a subset method: a member chosen 8 times out of 25
    has an effective weight of 0.32, arrived at greedily rather than by searching a
    grid, and the counts are integers so the result reads as "this many votes each".

    Parameters
    ----------
    M : ndarray, shape (n_samples, n_members)
    y : array-like, shape (n_samples,)
    n_iters : int, default 25
        Number of picks; also the denominator of the resulting weights.
    metric : callable, default roc_auc_score

    Returns
    -------
    ndarray, shape (n_members,)
        Selection counts divided by the number of picks made.
    """
    k = M.shape[1]
    counts = np.zeros(k)
    running = np.zeros(len(y))

    for i in range(n_iters):
        scores = [float(metric(y, (running + M[:, j]) / (i + 1))) for j in range(k)]
        j = int(np.argmax(scores))
        counts[j] += 1
        running = running + M[:, j]

    return counts / counts.sum()


def _build_aggregate(config):
    """Lane D: an operator applied to the member columns, or greedy selection."""
    kind = config["kind"]
    kw = {k: v for k, v in config.items() if k != "kind"}

    def build(M, y):
        if kind == "caruana":
            w = caruana_select(M, y, n_iters=config.get("n_iters", 25))
            return _as_predict(lambda Q, w=w: ensemble.blend(Q, w), weights=w)
        return _as_predict(lambda Q, kind=kind, kw=kw: aggregate(Q, kind, **kw))

    return build


# ---------------------------------------------------------------------------
# The evaluation harness
# ---------------------------------------------------------------------------

LANES = {"weights": _build_weights, "vote": _build_vote,
         "meta": _build_meta, "aggregate": _build_aggregate}


def make_combiner(lane, config):
    """Look up a lane's factory and bind it to one configuration.

    Parameters
    ----------
    lane : str
        One of "weights", "vote", "meta", "aggregate".
    config : dict
        Must contain "kind"; remaining keys are the variant's parameters. Must be
        JSON-serializable, since it is hashed into the result filename.

    Returns
    -------
    callable
        `build(M, y) -> predict`, where `predict(Q) -> scores`. The returned
        `predict` carries an `info` dict of whatever the fit learned.
    """
    assert lane in LANES, f"unknown lane {lane!r}, expected one of {sorted(LANES)}"
    return LANES[lane](config)


def cv_evaluate(build, M, y, cv, *, share=None):
    """Score a combiner out of fold, and measure how optimistic its in-sample score is.

    The combiner is refitted inside each fold of the locked CV split and scored on the
    rows it did not see, exactly as a model would be. This is the whole point of the
    module: a 3,003-point weight sweep and a parameter-free median both get one number
    that means the same thing, so a merged leaderboard across four lanes compares
    methods rather than search-space sizes.

    `auc_insample` fits on everything and scores on the same rows - the number notebook
    10 originally reported. `auc_gap` is the difference, and is the direct measurement
    of the selection bias rather than an assurance that it is small.

    Note the one thing this does not undo: `M` is itself built from out-of-fold member
    predictions generated under these same folds, so the members are honest, but a
    member's column was produced by a model that saw other folds' rows. That is
    inherent to stacking on OOF scores and applies equally to every combiner here.

    Parameters
    ----------
    build : callable
        `build(M_train, y_train) -> predict`, from `make_combiner`.
    M : ndarray, shape (n_samples, n_members)
    y : array-like, shape (n_samples,)
    cv : cross-validation splitter
        The locked protocol, `evaluation.make_cv()`.
    share : float, optional
        Predicted class balance for the secondary F1. Defaults to `y`'s own share.

    Returns
    -------
    dict
        auc_mean, auc_std, f1_mean, fold_aucs, auc_insample, auc_gap, info, seconds.
    """
    y = np.asarray(y)
    share = float(np.mean(y)) if share is None else float(share)
    t0 = time.monotonic()

    fold_aucs, fold_f1s = [], []
    for train_idx, test_idx in cv.split(M, y):
        predict = build(M[train_idx], y[train_idx])
        scores = predict(M[test_idx])
        fold_aucs.append(float(roc_auc_score(y[test_idx], scores)))
        fold_f1s.append(ensemble.macro_f1_at_share(scores, y[test_idx], share))

    full = build(M, y)
    insample = float(roc_auc_score(y, full(M)))

    return {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "f1_mean": float(np.mean(fold_f1s)),
        "fold_aucs": [float(a) for a in fold_aucs],
        "auc_insample": insample,
        "auc_gap": insample - float(np.mean(fold_aucs)),
        "info": getattr(full, "info", {}),
        "seconds": round(time.monotonic() - t0, 2),
    }


def fit_full(lane, config, M, y):
    """Refit one combiner on all rows and return its predict callable.

    Used after the leaderboard has picked a winner, to score the holdout and the test
    set. Selection happens on `cv_evaluate`; only the chosen combiner is ever refitted
    this way, so the fit-on-everything step cannot leak into the comparison.

    Parameters
    ----------
    lane : str
    config : dict
    M : ndarray, shape (n_samples, n_members)
    y : array-like, shape (n_samples,)

    Returns
    -------
    callable
        `predict(Q) -> scores`, carrying `.info`.
    """
    return make_combiner(lane, config)(M, y)


# ---------------------------------------------------------------------------
# The team ledger - one JSON per combiner config, merged by git
# ---------------------------------------------------------------------------

def result_id(lane: str, config: dict) -> str:
    """Deterministic short hash of a lane plus config, used as the result filename."""
    key = json.dumps({"lane": lane, "config": config}, sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def result_path(lane: str, owner: str, config: dict) -> Path:
    """ensemble_trials/<lane>_<owner>_<result_id>.json.

    Same shape as `tuning.trial_path`, and for the same reason: the filename is the
    cache key, so a rerun skips work already done and four teammates merge by dropping
    files into one directory. Note the hash covers only lane and config, so two people
    running the same configuration produce two files rather than one - duplicated
    compute, but never a conflicting result.
    """
    paths.ENSEMBLE_TRIALS.mkdir(parents=True, exist_ok=True)
    return paths.ENSEMBLE_TRIALS / f"{lane}_{owner}_{result_id(lane, config)}.json"


def save_result(lane: str, owner: str, config: dict, result: dict) -> Path:
    """Write one combiner's evaluation to the shared ledger.

    Parameters
    ----------
    lane, owner : str
    config : dict
        The combiner configuration, JSON-serializable.
    result : dict
        As returned by `cv_evaluate`.

    Returns
    -------
    Path
    """
    path = result_path(lane, owner, config)
    record = {"lane": lane, "owner": owner, "config": config, **result}
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def load_results(lane=None) -> pd.DataFrame:
    """Merge every teammate's combiner results into one leaderboard.

    Parameters
    ----------
    lane : str, optional
        Restrict to one lane. Defaults to all of them, which is the merge step
        sections 11 onward rely on.

    Returns
    -------
    pd.DataFrame
        One row per combiner, sorted by held-out AUC descending. Columns: lane, owner,
        kind, config, auc_mean, auc_std, f1_mean, auc_insample, auc_gap, seconds, info.
        Empty (with those columns) if nothing has been run yet.
    """
    cols = ["lane", "owner", "kind", "config", "auc_mean", "auc_std", "f1_mean",
            "auc_insample", "auc_gap", "seconds", "info"]
    pattern = f"{lane}_*.json" if lane else "*.json"
    records = []
    for f in sorted(paths.ENSEMBLE_TRIALS.glob(pattern)):
        rec = json.loads(f.read_text())
        rec["kind"] = rec.get("config", {}).get("kind")
        records.append(rec)
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records)
    for c in cols:
        if c not in df:
            df[c] = None
    return df[cols].sort_values("auc_mean", ascending=False).reset_index(drop=True)


def run_lane(lane, owner, configs, matrices, y, cv, *, share=None, default="rank"):
    """Evaluate every configuration in one lane, caching each result as it lands.

    Resumable in the same way `tuning.run_search` is: a configuration whose result file
    already exists is skipped, so an interrupted sweep restarts where it stopped and a
    `git pull` of a teammate's results removes work rather than duplicating it.

    Parameters
    ----------
    lane : str
    owner : str
    configs : list of dict
        Each needs "kind"; an optional "matrix" key names which matrix to run against,
        defaulting to `default`. That is what lets one voting config run on raw
        probabilities (reproducing notebook 07's combiner) while the rest run on ranks.
    matrices : dict
        name -> ndarray, e.g. {"rank": R, "raw": S}.
    y : array-like, shape (n_samples,)
    cv : cross-validation splitter
    share : float, optional
        Passed to `cv_evaluate` for the secondary F1.
    default : str, default "rank"
        Matrix used by configs that do not name one.

    Returns
    -------
    pd.DataFrame
        This lane's merged results across all owners, best first.
    """
    for i, config in enumerate(configs, start=1):
        path = result_path(lane, owner, config)
        opts = {k: v for k, v in config.items() if k != "kind"}
        label = f"{config.get('kind')} {opts if opts else ''}"
        if path.exists():
            print(f"[{i}/{len(configs)}] cached   {label}")
            continue

        which = config.get("matrix", default)
        assert which in matrices, f"config asks for matrix {which!r}, have {sorted(matrices)}"
        result = cv_evaluate(make_combiner(lane, config), matrices[which], y, cv,
                             share=share)
        save_result(lane, owner, config, result)
        print(f"[{i}/{len(configs)}] auc {result['auc_mean']:.4f} "
              f"+/-{result['auc_std']:.4f}  gap {result['auc_gap']:+.4f}  "
              f"({result['seconds']:.0f}s)  {label}")

    return load_results(lane)
