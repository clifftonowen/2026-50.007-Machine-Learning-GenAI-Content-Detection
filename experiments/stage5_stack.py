"""Stage 5 - ensemble the survivors, on the winning representation.

The combiner machinery in src/combiners.py was built in round 4 and only ever pointed
at the supplied-TF-IDF members, where every member scored 0.65-0.74. It has never seen
the raw-text representation. That is the gap this stage closes.

Selection follows round 4's one durable finding: `auc_gap` - how much a combiner's
in-sample AUC flatters its out-of-fold AUC - predicted which combiner transferred to
the leaderboard, and held-out AUC alone did not. So the shortlist is ranked by low gap
among combiners that at least match the best single member, and the winner still has
to clear the 3-band paired bar before it earns a submission.

Member OOF comes from the 3-band grouped folds, not standard CV, for the same reason:
a combiner tuned on optimistic member scores learns the wrong weights.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from experiments.common import *
from src import combiners
from src.nbsvm import NBSVM

PROTOCOL = "g3"


def member_specs(env):
    """The candidate pool. Each entry: name -> (make_estimator, X_dev, X_full, X_test).

    Deliberately diverse rather than a list of near-identical GBDTs: two boosted-tree
    members, a log-count-ratio linear model, a plain linear model and a randomized
    forest. Correlated members add compute and no ranking information.
    """
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import ExtraTreesClassifier

    X_dev, X_full, X_test = env["X_dev"], env["X_full"], env["X_test"]
    NG = ["H_char_ngrams", "I_word_ngrams"]
    ng_full, ng_test, _ = tf.stack(env["built"], NG)
    ng_dev = ng_full[env["dev_idx"]]

    # XGBoost is deliberately absent. On this 40,385-column sparse matrix its hist
    # tree method asked for 12.4 GB and died on a 15 GB machine with ~1 GB free. It is
    # also the member with the least to offer: its errors correlate with LightGBM's, so
    # the blend loses little by dropping it. n_jobs is capped rather than -1 for the
    # same reason - 16 worker copies of the intermediate state is what runs us out.
    return {
        "lgbm": (lambda: LGBMClassifier(**FIXED), X_dev, X_full, X_test),
        "nbsvm": (lambda: NBSVM(estimator=LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000)),
            ng_dev, ng_full, ng_test),
        "linsvc": (lambda: LinearSVC(C=0.14, class_weight="balanced", max_iter=5000),
                   X_dev, X_full, X_test),
        "extratrees": (lambda: ExtraTreesClassifier(
            n_estimators=150, n_jobs=4, random_state=42, class_weight="balanced"),
            X_dev, X_full, X_test),
    }


CONFIGS = {
    "aggregate": [{"kind": "mean"}, {"kind": "median"}, {"kind": "trimmed", "trim": 0.2},
                  {"kind": "power", "p": 0.5}, {"kind": "power", "p": 2.0},
                  {"kind": "caruana", "n_iters": 25}],
    "weights": [{"kind": "equal"}, {"kind": "nnls"},
                {"kind": "hill_climb", "step": 0.02}, {"kind": "hill_climb", "step": 0.05}],
    "vote": [{"kind": "soft_vote"}, {"kind": "weighted_vote"}, {"kind": "hard_vote"}],
    "meta": [{"kind": "logistic", "C": 1.0}, {"kind": "logistic", "C": 10.0},
             {"kind": "forward_select"}],
}


def run(env, members=None):
    y_dev, folds3 = env["y_dev"], env["folds3"]
    specs = member_specs(env)
    if members is not None:
        specs = {k: v for k, v in specs.items() if k in members}

    log_stage(f"stage 5a: member OOF + test scores ({PROTOCOL} folds)")
    oof_map, test_map = {}, {}
    for name, (make_est, Xd, Xf, Xt) in specs.items():
        # One member running out of memory should cost that member, not the stage. The
        # first attempt lost a finished LightGBM fit to an XGBoost allocation failure.
        try:
            oof, test = member_artifacts(name, make_est, Xd, Xf, Xt, protocol=PROTOCOL)
        except Exception as exc:
            print(f"  {name:12s} SKIPPED: {type(exc).__name__}: "
                  f"{str(exc)[:120]}", flush=True)
            continue
        oof_map[name], test_map[name] = oof, test
        auc = combiners.roc_auc_score(y_dev, oof)
        print(f"  {name:12s} OOF AUC {auc:.4f}", flush=True)

    assert len(oof_map) >= 2, f"need at least two members to blend, got {sorted(oof_map)}"

    names = sorted(oof_map)
    R, _ = ensemble.rank_matrix(oof_map, names)
    print("\nmember rank correlations:")
    print(pd.DataFrame(np.corrcoef(R.T), index=names, columns=names).round(3).to_string())

    log_stage("stage 5b: combiner lanes")
    # The ledger filename hashes only (lane, config), and `load_results` merges every
    # owner's files for a lane. Round 4's 48 results are in that directory, computed on
    # different members under standard CV, so they are not comparable to these and must
    # not share a leaderboard. The owner tag carries the member set for that reason,
    # and the board is filtered to it.
    ens_owner = f"{OWNER}_{'-'.join(names)}"
    frames = []
    for lane, configs in CONFIGS.items():
        df = combiners.run_lane(lane, ens_owner, configs, {"rank": R}, y_dev,
                                evaluation.FoldSplitter(folds3), share=0.53)
        frames.append(df[df["owner"] == ens_owner])

    board = pd.concat(frames, ignore_index=True)
    best_single = max(combiners.roc_auc_score(y_dev, oof_map[n]) for n in names)
    print(f"\nbest single member OOF AUC {best_single:.4f}")

    ok = board[board["auc_mean"] >= best_single].sort_values("auc_gap")
    print("\ncombiners matching or beating the best single member, lowest optimism first:")
    cols = [c for c in ["lane", "config", "auc_mean", "auc_gap", "f1_mean"]
            if c in board.columns]
    print((ok if len(ok) else board.sort_values("auc_mean", ascending=False))
          .head(12)[cols].round(4).to_string(index=False))

    return {"board": board, "shortlist": ok, "oof": oof_map, "test": test_map,
            "members": names, "R": R}


if __name__ == "__main__":
    run(ENV)
