# Task.md — GenAI Content Detection (50.007 Machine Learning, Kaggle Competition)

## Project Summary

Binary text classification task: **given a piece of text, classify it as human-authored (0) or machine-generated (1)**.

- Team project: 4–5 members per team.
- Dataset: sampled from the COLING 2025 GenAI Content Detection Workshop dataset (Wang et al., 2025), pre-processed for this course (stop words removed, lemmatized, top 5000 TF-IDF features computed).
- Split sizes: 20K train, 2K dev, 5K test.
- **Warning from brief:** expect a high training F1 but lower test F1 — read the paper's data sampling section. **Do not over-engineer the solution.**

Paper reference: https://arxiv.org/pdf/2501.11012

---

## Files Provided

| File | Description |
|---|---|
| `train.csv` | Training set, no features |
| `train_features.csv` | Training set with top 5000 TF-IDF features — **use this for Tasks 1–3** |
| `test.csv` | Test set, no features |
| `test_features.csv` | Test set with top 5000 TF-IDF features — **use this for Tasks 1–3** |
| `sample_submission.csv` | Sample submission in the correct format |

**Class label:** `1` = machine-generated, `0` = human-authored.

No further feature engineering is required (already done), though extra feature engineering is allowed if described in the report/presentation.

---

## Tasks

### Task 1: Logistic Regression from scratch (5 marks)
- Implement Logistic Regression **from scratch** — no `sklearn` or any pre-built logistic regression package (0 marks if used).
- Required functions:
  - `sigmoid(z)` — maps a real number to (0, 1)
  - `loss(y, y_hat)` — Log Loss between actual and predicted labels
  - `gradients(X, y, y_hat)` — returns partial derivatives of loss w.r.t. weights (`dw`) and bias (`db`)
  - `train(X, y, bs, epochs, lr)` — training loop
  - `predict(X)` — prediction on validation/test sets
- **Deliverables:**
  - 1a. Code implementation
  - 1b. Test set predictions → submit to Kaggle as **`LogReg_Prediction.csv`**, and include in the final submission

### Task 2: Dimension reduction (PCA) (5 marks)
- Apply PCA to reduce the 5000 TF-IDF features (sklearn allowed for this task).
- Use **KNN** (`n_neighbors=2`, sklearn allowed) as the classifier.
- **Deliverables:**
  - 2a. Code implementation of PCA on train and test sets
  - 2b. Report Macro F1 on the test set for **2000, 1000, 500, and 100** components (submit predictions to Kaggle to get each score, report results in the final report)

### Task 3: Other ML models — race to the top (15 marks)
- Try any other ML models (course-taught or not) to improve performance.
- **No deep learning / LLMs allowed** for this task.
- Bonus marks by private leaderboard placement (released after submission, week 13):
  - 3rd place: +1 mark
  - 2nd place: +2 marks
  - 1st place: +3 marks
- **Deliverables:**
  - 3a. Code for all models tried, with comments on models used and key hyperparameters
  - 3b. Test set predictions submitted to Kaggle **under the registered team name**

### Task 4: Report — documenting the journey (25 marks)
Final report (PDF) must answer:
1. Introduce the best-performing model and how it works.
2. How was the best model achieved / tuned? What parameters were used and tried (the roadmap)?
3. Difficulties faced while tuning — how were they overcome, or reflections if not overcome.
4. Anything self-learned beyond the course content?

---

## Evaluation Metric

Macro F1 score between predicted and actual class labels (human vs. machine).

Per-class F1:

$$F_1 = \frac{TP}{TP + \frac{1}{2}(FP + FN)}$$

Final score = macro average of **Human F1** and **Machine F1**.

## Submission File Format (Kaggle)

CSV with header, one row per test-set id:

```
id,label
2,0
5,1
6,0
...
```
(`1` = machine-generated, `0` = human-authored)

---

## Final Deliverables

1. **Jupyter Notebook** with code for Tasks 1–3, clearly segmented and labeled by task.
2. **Final prediction CSV** submitted to Kaggle for Task 3.
3. **Project report** (PDF) covering Task 4 questions.
4. **Presentation** of the best solution.

## Deadlines

| Item | Deadline |
|---|---|
| Jupyter Notebook + final prediction output + report | **10 Aug 2026, 23:59** |
| In-class project presentation | Week 13 (tentatively **11 Aug 2026**) |

---

## Links

- Kaggle competition / grading rubrics: https://www.kaggle.com/competitions/50-007-machine-learning-may-2026/overview/grading-rubrics
- COLING 2025 paper: https://arxiv.org/pdf/2501.11012
