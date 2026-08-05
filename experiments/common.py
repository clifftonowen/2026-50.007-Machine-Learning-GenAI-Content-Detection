"""Shared setup for the overnight experiment scripts.

Recreates notebook 17's protocol exactly - same CHOSEN blocks, same locked dev split,
same 5-band selection folds and 3-band confirmation folds, same LightGBM-defaults
baseline - so every candidate scored here is comparable to the existing trial ledger.
Import * is deliberate: these scripts are sequenced one-night runners, not library code.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from src import clustering, data, ensemble, evaluation, paths, text, tuning
from src import text_features as tf

OWNER = "brian_night"
STAGE1 = "lgbm_text_stage1"          # existing ledger key holding the baseline trial

CHOSEN = ["A_function_words", "B_punctuation", "C_casing", "D_structure",
          "E_length", "F_diversity", "H_char_ngrams", "I_word_ngrams"]

FIXED = dict(class_weight="balanced", subsample_freq=1, random_state=42,
             n_jobs=-1, verbose=-1)

BEST_SHARES = {"uuid": 0.6198, "numeric": 0.4756}
BENCH_FILE, BENCH_F1 = "chosen_pergroup62_48.csv", 0.80143
NOISE_FLOOR = 0.0084


def build_lgbm(params):
    return LGBMClassifier(**params, **FIXED)


def _load():
    train_ids, train_texts, y = text.load_train_text()
    test_ids, test_texts = text.load_test_text()

    dev_idx = np.load(paths.DATA_PROCESSED / "dev_idx.npy")
    y_dev = y[dev_idx]
    dev_texts = np.asarray(train_texts)[dev_idx]

    built = tf.load_blocks(CHOSEN)
    X_full, X_test, names = tf.stack(built, CHOSEN)
    X_dev = X_full[dev_idx]
    assert X_dev.shape == (16000, 40385), X_dev.shape

    folds5, _ = clustering.cluster_cv(
        clustering.length_groups(dev_texts, n_groups=5), y_dev)
    folds3, _ = clustering.cluster_cv(
        clustering.length_groups(dev_texts, n_groups=3), y_dev)
    assert len(folds5) == 5 and len(folds3) == 3

    return dict(train_ids=train_ids, train_texts=train_texts, y=y,
                test_ids=test_ids, test_texts=test_texts,
                dev_idx=dev_idx, y_dev=y_dev, dev_texts=dev_texts,
                built=built, X_full=X_full, X_test=X_test, names=names,
                X_dev=X_dev, folds5=folds5, folds3=folds3,
                cv_standard=evaluation.make_cv())


ENV = _load()
globals().update(ENV)

# Baseline: LightGBM defaults, the model behind the 0.80143 submission. Both trials
# are cached in the ledger, so these return instantly after the first-ever run.
_base5 = tuning.run_trial(STAGE1, "shared", build_lgbm({}), {}, X_dev, y_dev, folds5)
_base3 = tuning.run_trial("night_g3", "shared", build_lgbm({}),
                          {"night_model": "lgbm_defaults"}, X_dev, y_dev, folds3)
BASE5 = np.array(_base5["scores"])
BASE3 = np.array(_base3["scores"])


def night_trial(name, params, estimator, protocol="g5", X=None):
    """Score one candidate under a protocol, cached in the shared trial ledger.

    Parameters
    ----------
    name : str
        Candidate key; the ledger model key becomes "night_<protocol>_<name>".
    params : dict
        Hashed into the cache filename; include everything that defines the config.
    estimator : sklearn-compatible estimator
    protocol : "g5" | "g3" | "std"
    X : matrix, optional
        Feature matrix over the dev rows; defaults to the CHOSEN-blocks X_dev.
        Row order must match dev_idx (the fold indices assume it).

    Returns
    -------
    dict with the run_trial record plus paired stats against the matching baseline.
    """
    cv = {"g5": folds5, "g3": folds3, "std": cv_standard}[protocol]
    rec = tuning.run_trial(f"night_{protocol}_{name}", OWNER, estimator, params,
                           X_dev if X is None else X, y_dev, cv)
    return with_paired(rec, protocol)


def with_paired(rec, protocol):
    """Attach paired-vs-baseline stats to a trial record.

    A no-op unless the protocol has a matching baseline: pairing scores fold-by-fold
    only means something when both sides were scored on the same folds, so the
    standard-CV and shift-band protocols get no paired columns rather than misleading
    ones computed against the length-band baseline.
    """
    base = {"g5": BASE5, "g3": BASE3, "std": None, "shift": None}[protocol]
    if base is not None:
        d = np.array(rec["scores"]) - base
        rec = dict(rec, paired_mean=float(d.mean()),
                   paired_se=float(np.std(d, ddof=1) / np.sqrt(len(d))),
                   folds_better=int((d > 0).sum()), n_folds=len(d))
    return rec


def manual_trial(name, params, fold_scores_fn, protocol="g5"):
    """Like night_trial, for candidates cross_val_score can't drive (sample weights,
    seed bags, transductive refits). `fold_scores_fn(folds) -> list of macro F1`.
    Cached under the same ledger schema so load_trials merges it.
    """
    model_key = f"night_{protocol}_{name}"
    path = tuning.trial_path(model_key, OWNER, params)
    if path.exists():
        return with_paired(json.loads(path.read_text()), protocol)
    folds = {"g5": folds5, "g3": folds3, "shift": folds3}[protocol]
    scores = [float(s) for s in fold_scores_fn(folds)]
    rec = {"model": model_key, "owner": OWNER, "params": params,
           "mean": float(np.mean(scores)), "std": float(np.std(scores)),
           "scores": scores}
    path.write_text(json.dumps(rec, indent=2))
    return with_paired(rec, protocol)


def report(rec, label=""):
    tag = label or rec["model"]
    extra = ""
    if "paired_mean" in rec:
        extra = (f"  paired {rec['paired_mean']:+.4f} +/- {rec['paired_se']:.4f}"
                 f"  folds {rec['folds_better']}/{rec['n_folds']}")
    print(f"{tag:45s} mean {rec['mean']:.4f}{extra}", flush=True)
    return rec


def passes_g5(rec):
    return rec["paired_mean"] > 2 * rec["paired_se"] and rec["folds_better"] == rec["n_folds"]


def confirms_g3(rec):
    return rec["paired_mean"] > 0 and rec["folds_better"] >= 2


class SeedBag(BaseEstimator, ClassifierMixin):
    """Rank-averaged bag of one estimator refit under several seeds.

    Only the ranking feeds the share threshold, so averaging per-seed rank
    positions removes seed noise without touching anything else.
    """

    def __init__(self, build=None, seeds=(42, 43, 44, 45, 46)):
        self.build = build
        self.seeds = seeds

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = [self.build(seed).fit(X, y) for seed in self.seeds]
        return self

    def decision_function(self, X):
        ranks = [ensemble.to_rank(ensemble.member_score(m, X)) for m in self.models_]
        return np.mean(ranks, axis=0)

    def predict(self, X):
        proba = np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)
        return (proba >= 0.5).astype(int)


def log_stage(msg):
    print(f"\n=== {msg} [{time.strftime('%H:%M:%S')}] ===", flush=True)


def shift_scores():
    """Per-row 'how test-like is this training row', from a train-vs-test discriminator.

    Cached to disk; uses no labels, so it is honest to compute once over everything.
    """
    path = paths.DATA_PROCESSED / "night_shift_scores.npy"
    if path.exists():
        return np.load(path)

    import scipy.sparse as sp
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    Xd = sp.vstack([X_full, X_test], format="csr")
    yd = np.r_[np.zeros(X_full.shape[0]), np.ones(X_test.shape[0])]
    disc = LogisticRegression(C=1.0, max_iter=200, tol=1e-3,
                              class_weight="balanced").fit(Xd, yd)
    print(f"  shift discriminator in-sample AUC "
          f"{roc_auc_score(yd, disc.decision_function(Xd)):.4f}", flush=True)
    s = disc.decision_function(X_full).astype(np.float32)
    np.save(path, s)
    return s


def shift_folds(n_bands=3):
    """Leave-one-band-out folds over 'test-likeness' rather than document length.

    The established protocol bands by length, which measures generalisation across
    length and only indirectly across domain - but the actual gap on this task is that
    train and test come from disjoint corpora. Banding by a train-vs-test
    discriminator's score puts the rows that least resemble the test set in one fold
    and the rows that most resemble it in another, so holding out the test-like band
    simulates the deployment condition directly.

    Reported alongside the length protocol, never instead of it: the length bands are
    what the existing ledger and the 0.0084 noise floor are calibrated against.
    """
    s = shift_scores()[dev_idx]
    qs = np.linspace(0, 1, n_bands + 1)[1:-1]
    labels = np.digitize(s, np.quantile(s, qs)).astype(int)
    folds, _ = clustering.cluster_cv(labels, y_dev)
    return folds, labels


def member_artifacts(name, make_est, X_dev_m, X_full_m, X_test_m, protocol="g3"):
    """Out-of-fold dev scores plus full-train test scores for one ensemble member.

    Cached to disk, so a member fitted here is never refitted on a later run. OOF
    scores come from the grouped folds rather than standard CV, because the combiner
    has to be selected on the protocol that predicts transfer, not the one that
    flatters it.

    Returns
    -------
    oof : ndarray, shape (n_dev,)
    test : ndarray, shape (n_test,)
    """
    oof_path = paths.DATA_PROCESSED / f"night_member_{name}_oof_{protocol}.npy"
    test_path = paths.DATA_PROCESSED / f"night_member_{name}_test.npy"
    if oof_path.exists() and test_path.exists():
        return np.load(oof_path), np.load(test_path)

    folds = {"g5": folds5, "g3": folds3}[protocol]
    t = time.time()
    oof = evaluation.oof_from_folds(make_est(), X_dev_m, y_dev, folds)
    assert np.isfinite(oof).all(), f"{name}: grouped folds left rows unscored"
    test = ensemble.member_score(make_est().fit(X_full_m, y), X_test_m)
    np.save(oof_path, oof.astype(np.float32))
    np.save(test_path, np.asarray(test, dtype=np.float32))
    print(f"  member {name}: OOF + test scores in {time.time() - t:.0f}s", flush=True)
    return oof, test


def write_night_submission(scores, fname, *, note=""):
    """Apply the frozen per-group shares and write a checked submission CSV."""
    groups = text.id_group(test_ids)
    masks = text.group_masks(test_ids)
    preds = clustering.threshold_per_group(scores, groups, BEST_SHARES)

    dups = text.find_train_test_duplicates(train_ids, train_texts, y,
                                           test_ids, test_texts)
    if len(dups):
        pos = pd.Index(np.asarray(test_ids, dtype=object))
        for _, row in dups.iterrows():
            preds[pos.get_loc(row["test_id"])] = int(row["label"])

    data.write_submission(test_ids, preds, fname)
    sub = pd.read_csv(paths.SUBMISSIONS / fname, dtype={"id": str})
    assert list(sub.columns) == ["id", "label"]
    lab = sub["label"].to_numpy()
    for g, m in masks.items():
        assert abs(lab[m].mean() - BEST_SHARES[g]) < 2 / m.sum(), f"{g} share off target"

    bench_df = pd.read_csv(paths.SUBMISSIONS / BENCH_FILE, dtype={"id": str})
    # The row-by-row comparison below is only meaningful if both files list the same
    # ids in the same order; a silent reorder would report a huge spurious difference.
    assert list(bench_df["id"]) == list(sub["id"]), "id order differs from the benchmark"
    bench = bench_df["label"].to_numpy()
    n_diff = int((lab != bench).sum())
    print(f"  wrote {fname}: share {lab.mean():.4f}, {len(dups)} duplicates patched, "
          f"{n_diff}/{len(lab)} rows differ from {BENCH_FILE} ({BENCH_F1}){note}",
          flush=True)
    if n_diff < 100:
        print("    UNDER-POWERED: too few rows differ to resolve against the "
              "0.0084 noise floor - not worth a submission slot.", flush=True)
    return n_diff
