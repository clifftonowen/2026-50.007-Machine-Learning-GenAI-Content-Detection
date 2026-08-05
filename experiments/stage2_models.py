"""Stage 2 - model families never run on the winning representation.

Only LightGBM and XGBoost were ever evaluated on the 40,385-feature raw-text
representation; the other ten families in notebook 04 were benchmarked on the old
supplied TF-IDF, where everything scored 0.65-0.74 and the ranking there says little
about the ranking here. This stage fixes that.

Several candidates need dense input, so the per-fold SVD embedding is computed once
and cached rather than recomputed inside each pipeline - the projection is by far the
expensive part and it is identical for every model that consumes it. Fitting the SVD
on fold-train only keeps the held-out band genuinely held out.
"""

import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.common import *

SVD_DIM = 256
CATBOOST_CAP_SECONDS = 90 * 60


def svd_folds(env, protocol="g5", n_components=SVD_DIM):
    """Per-fold SVD embeddings, fitted on fold-train only, cached to disk.

    Returns list of (train_idx, test_idx, Z_train, Z_test).
    """
    path = paths.DATA_PROCESSED / f"night_svd{n_components}_{protocol}.npz"
    folds = env["folds5"] if protocol == "g5" else env["folds3"]
    X_dev = env["X_dev"]

    if path.exists():
        z = np.load(path)
        return [(folds[i][0], folds[i][1], z[f"tr{i}"], z[f"te{i}"])
                for i in range(len(folds))]

    out, store = [], {}
    for i, (tr, te) in enumerate(folds):
        t = time.time()
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        Ztr = svd.fit_transform(X_dev[tr])
        Zte = svd.transform(X_dev[te])
        sc = StandardScaler().fit(Ztr)
        Ztr, Zte = sc.transform(Ztr).astype(np.float32), sc.transform(Zte).astype(np.float32)
        store[f"tr{i}"], store[f"te{i}"] = Ztr, Zte
        out.append((tr, te, Ztr, Zte))
        print(f"  fold {i}: SVD-{n_components} {time.time() - t:.0f}s, "
              f"explained {svd.explained_variance_ratio_.sum():.3f}", flush=True)
    np.savez_compressed(path, **store)
    return out


def dense_trial(env, name, params, make_est, protocol="g5"):
    """Evaluate an estimator on the cached per-fold SVD embeddings."""
    emb = svd_folds(env, protocol)
    y_dev = env["y_dev"]

    def fold_scores(_folds):
        out = []
        for tr, te, Ztr, Zte in emb:
            m = make_est().fit(Ztr, y_dev[tr])
            out.append(evaluation.macro_f1(y_dev[te], m.predict(Zte)))
        return out

    return manual_trial(name, params, fold_scores, protocol=protocol)


def run(env):
    X_dev, y_dev = env["X_dev"], env["y_dev"]
    results = {}

    # ---- dense models on the shared SVD embedding --------------------------
    log_stage(f"stage 2a: SVD-{SVD_DIM} embedding (fit per fold on fold-train)")
    svd_folds(env, "g5")

    log_stage("stage 2b: dense families on the SVD embedding")
    dense = [
        ("histgb_svd", {"dim": SVD_DIM, "lr": 0.1},
         lambda: HistGradientBoostingClassifier(random_state=42, class_weight="balanced")),
        ("svcrbf_svd_c4", {"dim": SVD_DIM, "C": 4.0, "gamma": "scale"},
         lambda: SVC(C=4.0, gamma="scale", class_weight="balanced", cache_size=1000)),
        ("svcrbf_svd_c16", {"dim": SVD_DIM, "C": 16.0, "gamma": "scale"},
         lambda: SVC(C=16.0, gamma="scale", class_weight="balanced", cache_size=1000)),
        ("extratrees_svd", {"dim": SVD_DIM, "n_estimators": 500},
         lambda: ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42,
                                      class_weight="balanced")),
    ]
    for name, params, make_est in dense:
        t = time.time()
        rec = dense_trial(env, name, params, make_est)
        report(rec, f"{name} ({time.time() - t:.0f}s)")
        results[name] = {"rec": rec, "space": "svd"}

    # ---- sparse-native families -------------------------------------------
    log_stage("stage 2c: sparse-native families on the full representation")
    rec = night_trial("extratrees_sparse", {"n_estimators": 300},
                      ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42,
                                           class_weight="balanced"))
    report(rec, "extratrees (raw sparse)")
    results["extratrees_sparse"] = {"rec": rec, "space": "sparse"}

    # Tensor-sketch approximates a degree-2 polynomial kernel on the full 40k
    # columns - feature interactions the SVD basis cannot represent.
    rec = night_trial("polysketch_lr", {"degree": 2, "n_components": 2000},
                      make_pipeline(
                          PolynomialCountSketch(degree=2, n_components=2000,
                                                random_state=42),
                          StandardScaler(),
                          LogisticRegression(C=1.0, class_weight="balanced",
                                             max_iter=3000)))
    report(rec, "polynomial count sketch + logreg")
    results["polysketch_lr"] = {"rec": rec, "space": "sparse"}

    # ---- CatBoost, time-capped --------------------------------------------
    log_stage("stage 2d: CatBoost")
    try:
        from catboost import CatBoostClassifier

        def make_cat(iterations):
            return CatBoostClassifier(iterations=iterations, learning_rate=0.1,
                                      depth=6, random_seed=42, verbose=0,
                                      thread_count=-1, auto_class_weights="Balanced")

        tr, te = env["folds5"][0]
        t = time.time()
        make_cat(50).fit(X_dev[tr], y_dev[tr])
        per_iter = (time.time() - t) / 50
        want = 600
        projected = per_iter * want * len(env["folds5"])
        print(f"  {per_iter:.2f}s/iteration -> {want} iterations x 5 folds "
              f"= {projected / 60:.0f} min projected", flush=True)

        if projected > CATBOOST_CAP_SECONDS:
            want = max(100, int(CATBOOST_CAP_SECONDS / (per_iter * len(env["folds5"]))))
            print(f"  over the {CATBOOST_CAP_SECONDS / 60:.0f} min cap - "
                  f"reducing to {want} iterations", flush=True)
        rec = night_trial(f"catboost_{want}", {"iterations": want, "depth": 6},
                          make_cat(want))
        report(rec, f"catboost ({want} iterations)")
        results[f"catboost_{want}"] = {"rec": rec, "space": "sparse"}
    except Exception as exc:
        print(f"  CatBoost skipped: {type(exc).__name__}: {exc}", flush=True)

    # ---- summary -----------------------------------------------------------
    log_stage("stage 2 summary (5-band selection protocol)")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["rec"]["paired_mean"]):
        rec = r["rec"]
        flag = "PASS" if passes_g5(rec) else "    "
        print(f"  {flag} {name:28s} mean {rec['mean']:.4f}  paired "
              f"{rec['paired_mean']:+.4f} +/- {rec['paired_se']:.4f}  "
              f"{rec['folds_better']}/5 folds", flush=True)
    return results


if __name__ == "__main__":
    run(ENV)
