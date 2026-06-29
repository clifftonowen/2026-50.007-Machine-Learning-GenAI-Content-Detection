# GenAI Content Detection (50.007 ML)

A binary text classifier that labels a piece of text as `1` = **machine-generated** or
`0` = **human-authored**, predicted from a course-supplied **5000-dimension TF-IDF feature
vector**. Scored on **Macro F1**.

This is a team coursework project. Work happens in numbered Jupyter notebooks that run in
order, with shared helpers in `src/`.

---

## Quickstart

```bash
# 1. Clone and enter the repo
git clone https://github.com/clifftonowen/2026-50.007-Machine-Learning-GenAI-Content-Detection
cd 2026-50.007-Machine-Learning-GenAI-Content-Detection

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the data (see "Get the data" below) into data/raw/

# 5. Make your own branch before doing any work (see "Working as a team")
git checkout -b task1-logreg

# 6. Launch Jupyter and open notebooks/ in order
jupyter notebook
```

---

## Working as a team — always branch first

> **Do not commit to `main` directly.** Create a short-lived branch for your task, push it,
> and open a pull request to merge back into `main`. This keeps everyone's work isolated and
> reviewable.

```bash
git checkout main
git pull                       # start from the latest main
git checkout -b task3-svm      # create + switch to your branch

# ... do your work, then ...
git add notebooks/04_models.ipynb
git commit -m "feat: explored SVM baselines in 04_models"
git push -u origin task3-svm   # push your branch, then open a PR on GitHub
```

**Branch naming** — mirror the task / notebook you're working on:

| Branch | For |
|---|---|
| `task1-logreg` | Task 1 — logistic regression from scratch |
| `task2-pca-knn` | Task 2 — PCA + KNN |
| `task3-<model>` | Task 3 — e.g. `task3-svm`, `task3-randomforest` |
| `NN-short-topic` | anything else, e.g. `01-eda-tweaks` |

**Commit messages** use Conventional-Commit prefixes: `feat:` for notebook/code work,
`docs:` for README/docs (e.g. `feat: ran 03_pca_knn notebook`).


---

## Get the data

The raw data is **not in this repo**. It is downloaded from the course Kaggle page and is
**deliberately gitignored** (the files are large and regenerable; see the FAQ below). After
cloning, download these five files and drop them into `data/raw/`:

| File | Contents | Role |
|---|---|---|
| `train.csv` | id, text, label | EDA + error analysis only (not model input) |
| `test.csv` | id, text | human-readable holdout |
| `train_features.csv` | id, label, 5000 TF-IDF features | **training input for Tasks 1-3** |
| `test_features.csv` | id, 5000 TF-IDF features | holdout features |
| `sample_submission.csv` | id, label | exact submission-format reference |

The feature columns are named `0001`..`5000` (5000 columns); the loaders in `src/data.py`
handle them automatically. Labels: **`1` = machine-generated, `0` = human-authored**.

---

## Workflow

Run the notebooks in ascending order. Each one ends with a
short carry-forward note saying what it hands to the next.

| Notebook | Task |
|---|---|
| `notebooks/01_eda.ipynb` | class balance, sparsity, intrinsic dimensionality; builds & saves the locked train/holdout split |
| `notebooks/02_logreg_from_scratch.ipynb` | **Task 1** — LogReg from scratch → `submissions/LogReg_Prediction.csv` |
| `notebooks/03_pca_knn.ipynb` | **Task 2** — PCA + KNN(n=2) at 2000/1000/500/100 components |
| `notebooks/04_models.ipynb` | **Task 3** — classical-ML model exploration (no deep learning / no LLMs) |
| `notebooks/05_tuning.ipynb` | **Task 3** — two-stage hyperparameter search |
| `notebooks/06_holdout.ipynb` | **Task 3** — final holdout eval + submission CSV |

Run `01_eda.ipynb` first on a fresh clone: it builds and saves the shared train/holdout
split that every later notebook loads, so everyone trains on the exact same data.

At the end, assemble the single labelled `notebooks/SUBMISSION.ipynb` (Tasks 1-3) and the
Task 4 PDF report in `reports/`.

---

## Layout

```
data/
  raw/         course data — immutable, gitignored (you add this)
  processed/   cached splits, result tables, OOF preds — gitignored
notebooks/     the experiment pipeline (run in order) + final SUBMISSION.ipynb
src/           reusable, importable helpers (paths, data loading, evaluation, plotting)
reports/
  figures/     saved plots — gitignored
models/        pickled trained models — gitignored
submissions/   prediction CSVs incl. LogReg_Prediction.csv — gitignored
```

`data/`, `models/`, `submissions/`, and `reports/figures/` are gitignored and kept in git via
`.gitkeep` sentinels, so only code and small text artifacts are version-controlled.

---

## FAQ

**Why isn't the data committed?** `train_features.csv` alone is 210 MB (over GitHub's
100 MB per-file limit), so a push including it would be rejected. The data is also fully
reproducible from the Kaggle page, so we keep it out of git and share it out-of-band (Kaggle
download or a shared drive). Reproducibility instead comes from the immutable `data/raw/`,
the fixed `random_state=42`, and the cached split-index files in `data/processed/`.

**My imports fail with `ModuleNotFoundError: src`.** Notebooks add the project root to
`sys.path` in their first setup cell — run that cell first, and launch Jupyter from the repo
root so the relative paths resolve.
