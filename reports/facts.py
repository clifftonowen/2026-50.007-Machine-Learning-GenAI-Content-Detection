"""Every number the report and the deck quote, read from the result ledgers.

Both deliverables import this rather than hard-coding figures, so the docx and the pptx
cannot disagree with each other or drift from `data/processed/` as the remaining results
land. `check()` asserts that the handful of values the prose names in sentences still
match the ledgers; a report that contradicts its own evidence is the failure mode worth
engineering against here.

Run directly to print everything the documents will say:

    python reports/facts.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import paths  # noqa: E402

FIGURES = paths.FIGURES
PROCESSED = paths.DATA_PROCESSED

# Public-leaderboard sampling noise on ~3,570 rows at F1 ~0.8. Every "is this a real
# gain" judgement in both documents is made against this number.
NOISE_FLOOR = 0.0084

TEAM_NAME = "[TODO: registered Kaggle team name]"
MEMBERS = ["Cliffton", "Brian", "Jovyan", "Koko", "Vincent"]

# The shipped submission.
BEST_FILE = "chosen_pergroup62_48.csv"
BEST_KAGGLE = 0.80143
BEST_SHARES = {"uuid": 0.6198, "numeric": 0.4756}
BEST_CV_STANDARD = 0.8764
BEST_CV_GROUPED = 0.8089
BEST_N_FEATURES = 40385

# Reference points the narrative is measured against.
SUPPLIED_LGBM_KAGGLE = 0.73583      # lightgbm_share50.csv, supplied features
SUPPLIED_LGBM_CV = 0.7391           # baseline_results.csv
ROUND4_BEST_KAGGLE = 0.73754        # ensemble_weights_hill_climb_share50.csv

# The first two leaderboard results, both from the supplied-feature era. Neither is in a
# ledger: they predate round5_results.csv, and they are quoted from the Kaggle submission
# history, so they are named here rather than derived.
FIRST_KAGGLE_DEFAULT_THR = 0.65738  # tuned LightGBM at its default 0.5 probability cut
FIRST_KAGGLE_TUNED_THR = 0.66502    # same model, threshold tuned on local holdout
TUNED_LGBM_CV = 0.7443              # holdout_metrics.csv, after the two-stage search
TUNED_LGBM_HOLDOUT = 0.7437         # holdout_metrics.csv, untouched 4,000 rows

CHOSEN_BLOCKS = [
    ("A function words", 318, "rates of 318 English stop words"),
    ("B punctuation", 37, "per-1000-character symbol rates, plus markdown counts"),
    ("C casing", 5, "uppercase, ALL-CAPS, Titlecase, internal-capital rates"),
    ("D structure", 10, "newlines, line lengths, bullets, headers, hard-wrap"),
    ("E length", 10, "log counts, word and sentence length moments, burstiness"),
    ("F diversity", 5, "TTR, hapax rate, Yule's K, MATTR over a 100-word window"),
    ("H char n-grams", 20000, "char_wb 2-5 TF-IDF, sublinear"),
    ("I word n-grams", 20000, "word 1-2 TF-IDF, stop words kept, no lemmatisation"),
]

# Round 5 ablation, notebook 14. loss_grouped is Macro F1 given up by dropping a family
# under leave-one-cluster-out CV.
ABLATION = [
    ("F_diversity", 0.0510, 5), ("H_char_ngrams", 0.0299, 20000),
    ("E_length", 0.0177, 10), ("C_casing", 0.0129, 5),
    ("I_word_ngrams", 0.0123, 20000), ("D_structure", 0.0012, 10),
    ("B_punctuation", 0.0011, 37), ("A_function_words", 0.0002, 318),
    ("G_readability", -0.0001, 5), ("tfidf_supplied", -0.0074, 5000),
]

# Notebook 15: does the grouped score depend on which grouping is used? The random
# control is the load-bearing row.
GROUPINGS = [
    ("k-means, seed 42", 0.7894, 2), ("k-means, seed 43", 0.8051, 2),
    ("length terciles", 0.8089, 3), ("random 3-way (control)", 0.8742, 3),
]

# Round 6 feature expansion, all measured under the 3-band protocol.
EXPANSION_NULLS = [
    ("extended diversity (J, 14 cols)", -0.0022),
    ("variability (K, 12 cols)", 0.0049),
    ("J and K together", 0.0011),
    ("char n-grams at 5x capacity", 0.0023),
]

MILESTONES = [
    # The first submission belongs on this curve. Starting the journey at the calibrated
    # 0.73583 hides the largest single gain in the project, which was the move off the
    # default 0.5 cut, and makes the later attribution look like the opposite of the truth.
    ("Tuned LightGBM at the default 0.5 cut", FIRST_KAGGLE_DEFAULT_THR),
    ("Same model, global share 0.4996", SUPPLIED_LGBM_KAGGLE),
    ("Six-member rank-blended ensemble", ROUND4_BEST_KAGGLE),
    ("Raw-text features, all blocks", 0.77349),
    ("Ablation-chosen feature set", 0.77942),
    ("uuid share corrected", 0.79706),
    ("uuid at its vertex", 0.80049),
    ("Both groups at their vertices", BEST_KAGGLE),
]


# Cumulative explained variance at Task 2's four fixed component counts, measured in
# notebook 01. The headline is that even 2,000 components keep only 77% of the variance,
# which is the quantitative reason a TF-IDF matrix this sparse resists compression.
PCA_VARIANCE = [(2000, 0.770), (1000, 0.564), (500, 0.390), (100, 0.158)]
PCA_ELBOW = (638, 0.4455)           # knee of the cumulative curve, and what it retains

# Marks and bands, read off the course brief slides rather than inferred. Recorded here
# because the report is written to these bands: the top band for the report asks for
# "interesting insights" and the top band for the presentation asks for motivation,
# design decisions and limitations, which is why both documents are organised around
# decisions rather than around results.
RUBRIC = {
    "task1": 5, "task2": 5, "task3": 15, "report": 15, "presentation": 10,
}
BRIEF_NOTES = [
    "The supplied preprocessed features are mandatory for Tasks 1 and 2.",
    "Preprocessing the text ourselves is allowed for Task 3, must be shared in full, "
    "and earns no extra marks on its own.",
    "Task 3 bonus is +3/+2/+1 for the top three private-leaderboard teams, capped so the "
    "task total cannot exceed 15.",
    "Task 2's top band requires a detailed analysis of the reduced components, not only "
    "the four Macro F1 scores.",
]

# The brief and the repo disagree on the Task 1 filename. Neither file has been generated
# yet, so this is live rather than historical.
TASK1_FILENAME_CONFLICT = ("LogReg_predictions.csv", "LogReg_Prediction.csv")

# The nine phases, in the order they happened. Each is one report section and at least one
# slide. `decision` is the thing that phase settled; the report expands it into a
# considered-alternatives table plus the tradeoff we accepted.
PHASES = [
    (1, "Understand the data before modelling it", "01",
     "measure class balance, sparsity, length and intrinsic dimensionality first"),
    (2, "Dimensionality reduction", "01, 03",
     "PCA at the four required component counts, then no PCA downstream"),
    (3, "Lock the validation protocol", "01, src/",
     "stratified 5-fold cross-validation on Macro F1, split saved to disk"),
    (4, "Baselines before any tuning", "04",
     "rank twelve model families at their defaults"),
    (5, "Tuning, and a first leaderboard result that went badly", "05a-f, 06a-d",
     "two-stage search per model, then submit the winner"),
    (6, "Ensembling, and an honest null", "07, 10",
     "six members, four combiner families, forty-eight configurations"),
    (7, "Threshold calibration", "08, 09, 11",
     "predicted share rather than probability threshold, and a confound to unpick"),
    (8, "The pivot to raw-text features", "12, 13, 14, 15",
     "function words, punctuation, casing, layout, diversity and n-grams"),
    (9, "Calibration revisited, per group", "16, 17",
     "threshold the two test populations separately"),
]


def scored_submissions():
    """Every submission with a recorded public score, best first."""
    df = pd.read_csv(PROCESSED / "round5_results.csv")
    return (df.dropna(subset=["kaggle_f1"])
            .sort_values("kaggle_f1", ascending=False)
            .reset_index(drop=True))


def baselines():
    """The 12-model baseline table from notebook 04."""
    df = pd.read_csv(PROCESSED / "baseline_results.csv", index_col=0)
    return df[["rank", "mean", "std"]].sort_values("rank")


def holdout():
    """CV against untouched holdout, the supplied-feature era."""
    df = pd.read_csv(PROCESSED / "holdout_metrics.csv")
    return df[["model", "cv_mean_f1", "holdout_f1"]]


def representations():
    """Best model per representation, notebook 12."""
    df = pd.read_csv(PROCESSED / "representation_results.csv")
    t = df.pivot_table(index="representation", columns="clf", values="mean")
    return t.max(axis=1).sort_values(ascending=False).rename("best_macro_f1")


def share_surface():
    """The per-group share points, uuid and numeric shares against public score."""
    df = scored_submissions()
    keep = ["round5_features_share50.csv", "chosen_uuid56.csv", "chosen_uuid62.csv",
            "chosen_pergroup62_48.csv", "chosen_pergroup62_55.csv"]
    return (df[df["file"].isin(keep)][["file", "uuid_share", "numeric_share",
                                       "kaggle_f1"]]
            .sort_values("kaggle_f1", ascending=False).reset_index(drop=True))


def figure(name):
    """Resolve a figure path, failing loudly rather than embedding nothing."""
    p = FIGURES / name
    assert p.exists() and p.stat().st_size > 0, f"missing or empty figure: {p}"
    return p


def check():
    """Assert the values the prose names still match the ledgers."""
    sub = scored_submissions()
    got = dict(zip(sub["file"], sub["kaggle_f1"]))

    problems = []
    if abs(got.get(BEST_FILE, -1) - BEST_KAGGLE) > 1e-9:
        problems.append(f"{BEST_FILE}: ledger {got.get(BEST_FILE)} != {BEST_KAGGLE}")
    for fname, expected in [("chosen_uuid62.csv", 0.80049),
                            ("chosen_uuid56.csv", 0.79706),
                            ("round5_features_share50.csv", 0.77942),
                            ("chosen_pergroup62_55.csv", 0.79123)]:
        if abs(got.get(fname, -1) - expected) > 1e-9:
            problems.append(f"{fname}: ledger {got.get(fname)} != {expected}")

    row = sub[sub["file"] == BEST_FILE].iloc[0]
    for key, want in BEST_SHARES.items():
        if abs(row[f"{key}_share"] - want) > 1e-4:
            problems.append(f"{BEST_FILE} {key} share {row[f'{key}_share']} != {want}")

    if abs(baselines().loc["LightGBM", "mean"] - SUPPLIED_LGBM_CV) > 5e-5:
        problems.append("baseline LightGBM CV does not match SUPPLIED_LGBM_CV")

    reps = representations()
    if abs(reps["tfidf_supplied"] - 0.7303) > 5e-5:
        problems.append("supplied-TF-IDF control moved in representation_results.csv")
    if abs(reps["text_all"] - 0.8811) > 5e-5:
        problems.append("text_all moved in representation_results.csv")

    total = sum(n for _, n, _ in CHOSEN_BLOCKS)
    if total != BEST_N_FEATURES:
        problems.append(f"block widths sum to {total}, not {BEST_N_FEATURES}")

    hold = holdout().set_index("model")
    if abs(hold.loc["lightgbm", "cv_mean_f1"] - TUNED_LGBM_CV) > 5e-5:
        problems.append("tuned LightGBM CV moved in holdout_metrics.csv")
    if abs(hold.loc["lightgbm", "holdout_f1"] - TUNED_LGBM_HOLDOUT) > 5e-5:
        problems.append("tuned LightGBM holdout moved in holdout_metrics.csv")

    # The narrative claims tuning gained about half a point of CV over the default. That
    # only holds if the baseline table and the post-search number are both what we think.
    if not 0.003 < TUNED_LGBM_CV - SUPPLIED_LGBM_CV < 0.008:
        problems.append("the tuning gain is no longer the ~0.005 the report describes")

    if len(PHASES) != 9 or [p[0] for p in PHASES] != list(range(1, 10)):
        problems.append("PHASES must be nine phases numbered 1 to 9")
    if sum(RUBRIC.values()) != 50:
        problems.append(f"rubric marks sum to {sum(RUBRIC.values())}, not 50")

    assert not problems, "facts drifted from the ledgers:\n  " + "\n  ".join(problems)
    return True


if __name__ == "__main__":
    check()
    print(f"best: {BEST_FILE} = {BEST_KAGGLE}  "
          f"(uuid {BEST_SHARES['uuid']}, numeric {BEST_SHARES['numeric']})")
    print(f"local: standard {BEST_CV_STANDARD}, grouped {BEST_CV_GROUPED}\n")
    print(scored_submissions()[["file", "uuid_share", "numeric_share",
                                "kaggle_f1"]].to_string(index=False))
    print("\nrepresentations:")
    print(representations().round(4).to_string())
    print("\nall facts agree with the ledgers")
