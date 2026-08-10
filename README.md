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
| `notebooks/02_logreg_from_scratch.ipynb` | **Task 1** — LogReg from scratch → `submissions/LogReg_predictions.csv` |
| `notebooks/03_pca_knn.ipynb` | **Task 2** — PCA + KNN(n=2) at 2000/1000/500/100 components |
| `notebooks/04_models.ipynb` | **Task 3** — twelve classical-ML baselines (no deep learning / no LLMs) |
| `notebooks/05_tuning.ipynb` | **Task 3** — two-stage hyperparameter search; `05b`–`05f` are the per-model searches |
| `notebooks/06_holdout.ipynb` | **Task 3** — holdout eval of the tuned models; `06b`–`06d` per model |
| `notebooks/07_linear_ensemble.ipynb`, `10_stacked_ensemble.ipynb` | **Task 3** — ensembling, four combiner families |
| `notebooks/08_calibration.ipynb`, `09_share_matched_comparison.ipynb`, `11_group_share.ipynb`, `16_share_surface.ipynb` | **Task 3** — the decision rule: threshold, matched-share comparison, per-group share |
| `notebooks/12_text_features.ipynb` | **Task 3** — the raw-text feature blocks that replaced the supplied TF-IDF |
| `notebooks/13_clustering.ipynb`, `14_ablation.ipynb`, `15_expansion.ipynb` | **Task 3** — pseudo-domains, grouped-CV ablation, expansion attempts |
| `notebooks/17_lightgbm_tuning.ipynb` | **Task 3** — the final tuning run, on the raw-text representation |

Run `01_eda.ipynb` first on a fresh clone: it builds and saves the shared train/holdout
split that every later notebook loads, so everyone trains on the exact same data.

### Start here

**`notebooks/SUBMISSION.ipynb` is the graded notebook.** It carries Tasks 1, 2 and 3 in one
place, clearly labelled, with outputs saved, and it is the notebook to read first. The
numbered notebooks above are the working record of how the project got there, which is what
the Task 4 report describes. The report itself is in `reports/`.

---

## Team hyperparameter search (`05_tuning.ipynb`)

Task 3's tuning notebook is built so multiple teammates can search different parts of the
same model's hyperparameter space in parallel, without stepping on each other or losing
work if a run gets interrupted partway through.

**How it works:** every trial (one hyperparameter combination, scored under the locked
5-fold CV) is written to its own small JSON file the moment it finishes, under
`data/processed/tuning_trials/`. Unlike the rest of `data/`, these files **are tracked in
git** - small and human-readable, and safe to merge since each teammate's files have
distinct names. Rerunning the search skips any trial whose file already exists, so an
interrupted run just picks up where it left off instead of starting over, and merging
teammates' results is just "have their files present locally" - no manual concatenation.

**Before running your share of the search:**

1. `git pull` so you have everyone's latest trial files.
2. In `05_tuning.ipynb`'s setup section, set `OWNER` to something that identifies you and
   your slice of the search space, e.g. `"alice_lr_lo"` for Alice covering the low
   `learning_rate` band (see the notebook's section 2 for how ranges are typically split).
   This tags every file you produce so it never collides with a teammate's.
3. Run stage 1 (the coarse search) for your assigned range.
4. Commit and push your new trial files:
   ```bash
   git add data/processed/tuning_trials/
   git commit -m "feat: lightgbm stage-1 trials, learning_rate 0.01-0.05"
   git push
   ```

**Before running stage 2 (the refined search):** `git pull` first. Stage 2 centers its
search on the *merged* best trial across everyone's stage-1 results, not just your own, so
it needs everyone's files present to pick the right center.

**Only one final model gets pickled.** The notebook's last section refits and saves a
single `.pkl` for the overall winning configuration - this isn't something every teammate
should do individually. Pickled sklearn/LightGBM/XGBoost objects don't reliably survive
being unpickled on a different machine's package versions, so only the winner (chosen
after everyone's results are merged) gets persisted that way.

---

## Layout

```
data/
  raw/               course data — immutable, gitignored (you add this)
  processed/         cached splits, result tables, OOF preds — gitignored
    tuning_trials/   hyperparameter search results - tracked in git (see below)
notebooks/     the experiment pipeline (run in order) + final SUBMISSION.ipynb
src/           reusable, importable helpers (paths, data loading, evaluation, plotting, tuning)
reports/
  figures/     saved plots — gitignored
models/        pickled trained models — gitignored
submissions/   prediction CSVs incl. LogReg_predictions.csv — gitignored
```

`data/`, `models/`, `submissions/`, and `reports/figures/` are gitignored and kept in git via
`.gitkeep` sentinels, so only code and small text artifacts are version-controlled. The one
exception is `data/processed/tuning_trials/*.json`, which is deliberately tracked so
teammates can merge hyperparameter search results via git - see "Team hyperparameter search"
above.

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
