"""Project-root-anchored paths.

Single source of truth for filesystem locations so notebooks never hardcode
relative strings. Import these constants instead of building paths by hand.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TUNING_TRIALS = DATA_PROCESSED / "tuning_trials"
OOF = DATA_PROCESSED / "oof"  # per-member out-of-fold scores, tracked in git
ENSEMBLE_TRIALS = DATA_PROCESSED / "ensemble_trials"  # combiner results, tracked in git
NOTEBOOKS = PROJECT_ROOT / "notebooks"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = PROJECT_ROOT / "models"
SUBMISSIONS = PROJECT_ROOT / "submissions"

# Raw data files (confirm exact schema on first load).
TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"
TRAIN_FEATURES_CSV = DATA_RAW / "train_features.csv"
TEST_FEATURES_CSV = DATA_RAW / "test_features.csv"
SAMPLE_SUBMISSION_CSV = DATA_RAW / "sample_submission.csv"
