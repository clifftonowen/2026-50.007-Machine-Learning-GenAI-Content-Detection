# GenAI Content Detection ( 50.007 ML)

Binary text classifier labelling a piece of text as `1` = machine-generated or
`0` = human-authored, from a supplied **5000-dimension TF-IDF feature vector**.
Scored on **Macro F1**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Get the data

Raw data is obtained out-of-band from the course Kaggle page and is **never committed**
(`data/raw/` is gitignored and immutable). Place these files in `data/raw/`:

| File | Role |
|---|---|
| `train.csv` | id, text, label — EDA + error analysis only |
| `test.csv` | id, text — human-readable holdout |
| `train_features.csv` | id, label, 5000 TF-IDF features — **training input for Tasks 1-3** |
| `test_features.csv` | id, 5000 TF-IDF features — holdout features |
| `sample_submission.csv` | exact submission format reference |

> Confirm the feature-column header names on first load (`feat_0..feat_4999`, `0..4999`, …)
> and record the confirmed schema.

## Workflow

Notebooks are the experiment runner — run them in ascending order:

| Notebook | Task |
|---|---|
| `notebooks/01_eda.ipynb` | class balance, sparsity, intrinsic dimensionality; builds & saves the locked split |
| `notebooks/02_logreg_from_scratch.ipynb` | **Task 1** — LogReg from scratch → `submissions/LogReg_Prediction.csv` |
| `notebooks/03_pca_knn.ipynb` | **Task 2** — PCA + KNN(n=2) at 2000/1000/500/100 components |
| `notebooks/04_models.ipynb` | **Task 3** — classical-ML model exploration (no DL / no LLMs) |
| `notebooks/05_tuning.ipynb` | **Task 3** — two-stage hyperparameter search |
| `notebooks/06_holdout.ipynb` | **Task 3** — final holdout eval + submission CSV |

At the end, assemble the single labelled `notebooks/SUBMISSION.ipynb` (Tasks 1-3) and the
Task 4 PDF report in `reports/`.

## Layout

```
data/{raw,processed}/   notebooks/   src/   reports/{figures,}   models/   submissions/
```

`data/`, `models/`, `submissions/`, `reports/figures/` are gitignored (kept via `.gitkeep`).
