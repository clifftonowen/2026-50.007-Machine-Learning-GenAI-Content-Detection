# Round 4 — work division

**Members:** Jovyan, Cliffton, Brian Wong, Koko · **Budget:** 3h00 compute each · **Deadline:** 10 Aug 2026

> **Sessions 1–2 (the LightGBM retune) are STOPPED. Sessions 3–4 are superseded by
> [`ENSEMBLE_PLAN.md`](ENSEMBLE_PLAN.md).** Job 1 got partway - 126 stage-1 trials from
> Cliffton's `learning_rate` band, zero from the other two, zero at stage 2, and finishing
> it would cost three people another 2h30 each for an expected gain of about +0.005, inside
> the 0.0084 noise floor. Job 2 was never started and is worth more. The four-way ensembling
> lab in `ENSEMBLE_PLAN.md` replaces it.
>
> The rest of this file is still current and still required reading: the leaderboard facts,
> the sparse-load setup below, and the open private-LB pick question at the bottom. Cliffton's
> 126 trials are worth keeping (`05f` section 4's knob-influence analysis still runs on them)
> but are **still untracked in git** - commit `data/processed/tuning_trials/` before anything
> else.

Task 3 currently sits at **0.73583** public Macro F1 (`lightgbm_share50.csv`, share 0.4996).
This round attacks the two things notebook 09 left open. Read `notebooks/09_share_matched_comparison.ipynb`
sections 7 and 9 first if you have not — everything below depends on what they established.

---

## Why this round exists

Notebook 09 settled the class-balance question and, in doing so, **reversed the project's
working assumption**. It had looked like local CV was anti-correlated with the leaderboard,
so the linear models were being preferred despite LightGBM's better CV. That was wrong.
Holding predicted share fixed at 0.4996, LightGBM beat ElasticNet by **+0.0189** — the same
direction and roughly the same size as its local CV lead. Local validation was never broken;
it was being read through a calibration confound.

Two consequences:

1. **Share tuning is closed.** The peak is flat across share ≈ 0.45–0.55 and every move
   inside that band is smaller than the ~0.0084 public-LB noise floor. Do not spend
   submissions there.
2. **Model quality is the only axis left, and CV gains now transfer at ~1:1.** That makes
   the two jobs below worth real compute for the first time.

### Job 1 — LightGBM was never properly searched

`05_tuning.ipynb` ran 8 random draws over a 7-dimensional space, then a 3×3 grid on
`learning_rate` × `num_leaves` with the other five knobs frozen. 26 configurations total.
And the winner sits at `min_child_samples=7` and `num_leaves=149`, against search ranges
of 5–60 and 15–150 — **the search stopped on its own boundary in two dimensions**, which is
the classic signature of an optimum lying outside the box that was searched.

### Job 2 — the ensemble question was answered with the wrong members

Notebook 07 tested ElasticNet + calibrated LinearSVC: probability correlation **0.9745**,
disagreeing on **11.4%** of rows. Two members that are nearly the same model. Its one truly
diverse member (ComplementNB, 22–28% disagreement) was rejected for being too weak at 0.6580.
So diversity and member quality were never available at the same time.

Notebook 09 supplied the missing pair. At matched share, **LightGBM disagrees with ElasticNet
on 956 test rows (13.7%) while scoring 0.0189 higher.** That is the first candidate pair that
is both diverse and both-strong — and it has never been tested.

07 also measured it wrongly, in two ways that `src/ensemble.py` now fixes. Both follow from
submissions being made by `write_at_share` (a quantile cut), so **only the ranking matters**:

- 07 averaged raw **probabilities**, which is not rank-invariant — it weights members by
  their probability spread, and LightGBM's probabilities are much sharper than a penalized
  logistic's. We blend on **ranks** instead.
- 07 selected on **macro-F1 at a 0.5 cutoff**, the metric confounded by exactly the
  calibration effect 09 spent three rounds isolating. We select on **OOF ROC AUC**, which
  *is* ranking quality and is invariant to the share.

**Honest expectation for the whole round: +0.005 to +0.015.** Both jobs must clear a local
bar before costing a Kaggle slot.

---

## Before you start (once per machine)

```bash
git pull
pip install -r requirements.txt     # scipy is now an explicit dependency
```

You need `data/raw/` populated from Kaggle and `01_eda.ipynb` already run, so that
`data/processed/dev_idx.npy` and `holdout_idx.npy` exist. Nothing below re-splits — and on
a machine where 01 has never run, `05f` section 1 fails on the *second* line with
`FileNotFoundError: dev_idx.npy`, not on the sparse load.

Those two `.npy` files are gitignored, so **every machine has to run `01_eda.ipynb` once**,
even one that has pulled everything. Same for the tuned-parameter JSONs notebook 10 reads
(`best_xgboost_params.json` and friends) — but those are cheap to regenerate, because the
`tuning_trials/*.json` they are derived from *are* tracked: re-run only the final cell of
`05c`/`05d`/`05e` and each one rewrites its params file from the merged trials without
re-searching anything.

> **If a `src/` fix does not seem to take effect, restart the kernel.** Python caches
> imported modules, so editing `src/data.py` does nothing to a kernel that already imported
> it — and Jupyter renders the traceback from the *current* file while running the *old*
> bytecode, which makes the error look like it is pointing at a line that clearly cannot
> raise it. Both new notebooks now enable `%autoreload 2` in their setup cell, so this only
> bites on the very first run after a pull.

The first sparse load builds a cache and asserts it matches the dense path — it takes a
couple of minutes and then never runs again:

```python
from src import data
print(data.check_sparse_path())
```

**Why sparse:** `01_eda` measured the feature matrix at 98.6% sparse, so the dense float64
train matrix is 800 MB of mostly zeros. CSR float32 cuts that ~50× and speeds up every fit.
That is what makes a 3-hour search worth more than 26 trials. `load_train_features()` still
defaults to dense so notebooks 01–09 keep reproducing exactly as they were run — only the
new notebooks pass `sparse=True`.

---

## Session 1 — LightGBM stage-1 search · 2h00 · everyone in parallel · **STOPPED**

> Kept as the record of what was planned and how far it got. See the banner at the top.

Notebook: **`05f_lightgbm_retune.ipynb`**, sections 0–3.

The stage-1 space is split three ways by `learning_rate` band, **equal in log space**
(log₁₀ from −2.000 to −0.523, divided in thirds). README.md already nominates
`learning_rate` as the axis to split on, because it interacts most with every other knob.
Every other dimension is identical for all three of us, so the union covers the full box
exactly once and nobody duplicates anyone else's trials.

| member | `OWNER` | `learning_rate` band |
|---|---|---|
| Jovyan | `jovyan_lr_lo` | 0.010 – 0.031 |
| Cliffton | `cliffton_lr_mid` | 0.031 – 0.097 |
| Brian Wong | `brian_lr_hi` | 0.097 – 0.300 |

**In the notebook, change one line:** `ME = "jovyan"` / `"cliffton"` / `"brian"`. Your owner
string, band and stage-2 seed all derive from it.

**The search is time-boxed, not trial-count-boxed.** `tuning.run_search` stops when the wall
clock is spent, because trial cost varies several-fold across machines — a trial count that
fits one laptop overruns another. Every trial is written the moment it finishes, so
stopping partway loses nothing; a slower machine simply contributes fewer trials to the
merge. The notebook prints trials/hour as it goes, so report your actual coverage.

Widened vs. `05`: `num_leaves` to 400 and `min_child_samples` down to 2 (the two boundaries
the old search hit), plus four knobs it never touched — `min_child_weight`, `subsample`,
`max_bin`, and a lower `colsample_bytree` floor.

```bash
git add data/processed/tuning_trials/
git commit -m "feat: lightgbm_v2 stage-1 trials, learning_rate <your band>"
git push
```

**Then wait for all three pushes.** Stage 2 centres on the *merged* winner across all bands;
running it early would centre it on your own band's local best.

---

## Session 2 — stage-2 refinement · 0h30 · after everyone has pushed · **STOPPED**

`git pull`, then run `05f` sections 4–6.

Stage 2 draws random points from a ±30% box around the merged winner, varying **every** knob
at once. (05's stage 2 grid-searched two knobs and froze five, which cannot find
interactions — and `learning_rate` × `n_estimators` and `num_leaves` × `min_child_samples`
are exactly the interacting pairs.) All three of us draw from the same box with a different
seed, which is an equal three-way split of the same space and merges identically.

Seeds are set automatically from `ME`: Jovyan 0, Cliffton 1, Brian 2.

Section 4 also prints which knobs actually correlated with CV score across every merged
trial — that plot is the evidence for the Task 4 report that widening the box was justified,
rather than an assertion. Section 6 flags any knob whose winner is *still* pinned at a
boundary, which would be the signal to widen again in a round 5.

```bash
git add data/processed/tuning_trials/
git commit -m "feat: lightgbm_v2 stage-2 trials, seed <yours>"
git push
```

---

## Session 3 — ensemble OOF generation · ~0h30 · **superseded by `ENSEMBLE_PLAN.md`**

> Still the right idea, but the member split changed (Koko joined, and the LightGBM member
> is `lightgbm` rather than `lightgbm_v2` now that the retune is stopped). Follow
> [`ENSEMBLE_PLAN.md`](ENSEMBLE_PLAN.md) phase A, not the table below.

Notebook: **`10_stacked_ensemble.ipynb`**, sections 0–3. Change `ME` and nothing else.

Each of us generates different ensemble members. Out-of-fold scores go to
`data/processed/oof/*.npy` and are **tracked in git** — 16,000 float32 values is 64 KB per
member, so `git push` is what hands your work to everyone else. Nobody re-runs anyone
else's fits.

| member | generates | why |
|---|---|---|
| Cliffton | `lightgbm_v2` | 5 seeds × 5 folds — the expensive one; also owns the final refit |
| Jovyan | `xgboost`, `extratrees` | two moderate fits |
| Brian Wong | `logreg_elasticnet`, `linearsvc`, `complementnb` | `saga` is the slow one; the other two are near-free |

Notes on the member set:

- **`xgboost` is already tuned** (`05e`, CV 0.7366) and has **never been holdout-checked or
  submitted**. It is free evidence.
- **`linearsvc` needs no `CalibratedClassifierCV`.** 07 wrapped it in Platt scaling — 5 inner
  fits per outer fold, the expensive part of that notebook — purely to expose
  `predict_proba` for a probability average. Raw margins rank identically, so the wrapper is
  gone.
- **`complementnb` is included even though it is weak.** 07 rejected it because an
  equal-weight vote was always going to reject it. The weight search can give it 0 — the
  point is that it decides, rather than us.
- **`extratrees` is new** and counts toward Task 3's models-explored requirement.

```bash
git add data/processed/oof/
git commit -m "feat: OOF scores for <your members>"
git push
```

---

## Session 4 — ensemble and submissions · Cliffton · **superseded by `ENSEMBLE_PLAN.md`**

> The guard and the "anything inside ±0.0084 is a tie" discipline carry over unchanged. What
> changed: combiners are now scored **out of fold** rather than in-sample, four families are
> explored instead of one, and the batch drops to two files because `lightgbm_share50.csv` is
> already the matched-share control.

`git pull`, then run `10_stacked_ensemble.ipynb` sections 4–8. The weight search runs on
cached OOF arrays, so it is CPU-seconds; the full refits on all 20,000 rows are the only
real cost.

**The guard:** the blend ships only if it beats the best single member on OOF AUC by more
than the fold-to-fold spread. If it does not, that is the finding and the single model ships
— same discipline 07 applied to itself.

### The submission batch, and why it is shaped this way

Benchmark: `lightgbm_share50.csv` at share 0.4996 → **0.73583**. Every file is share-matched
to *exactly* that, reusing 09's control design, so any difference is purely model quality.

| # | file | question it answers |
|---|---|---|
| 1 | `lightgbm_v2_share50.csv` | did re-searching LightGBM help? |
| 2 | `ensemble_share50.csv` | did stacking help, *on top of* the retune? |
| 3 | `xgboost_share50.csv` | how does the tuned XGBoost do — its first score ever? |

Uploading the blend without #1 would confound retuning and ensembling in a single number.
That is precisely the mistake 09 diagnosed, and it is why #1 exists.

Hold the fourth slot. Only once the model question resolves should the share be re-placed
(0.50 vs 0.53) — and **share tuning itself stays closed**.

**Reading the results:** anything inside ±0.0084 is a tie, not a result. Say so plainly.
The precedent is 09 §9, where a 0.0032 move was correctly called noise instead of being
written up as a regression.

---

## Still open — a decision, not an oversight

**The two final private-LB picks.** Both current picks are LightGBM near the same share,
which is no hedge at all. Per 09 §9, the right choice depends on what fraction of test rows
the public leaderboard scores:

- **~51%** → the split cannot follow the `id`-format line (that would give 28.6% or 71.4%),
  so it is random, the halves are exchangeable, and both picks belong near the peak.
- **28.6% or 71.4%** → the split *does* follow the UUID-vs-numeric `id` boundary, the two
  halves may differ in true class balance, and pick 2 should hedge on *share* with a
  train-matched 0.6252 variant instead.

Someone should confirm that number from the competition page before we lock the picks.

## Also outstanding

**`03_pca_knn.ipynb` is still a scaffold** with `pass` in the loop body — no PCA run, no KNN,
no submissions. That is **5 marks unstarted** (Task 2 needs four Kaggle uploads to read back
its required Macro F1 values at 2000/1000/500/100 components), against Task 3's remaining
upside of at most +3 bonus marks. It is not part of round 4, but it should be scheduled
before any round 5.

---

## What changed in the shared code

| file | change |
|---|---|
| `src/data.py` | `sparse=True` on both loaders → cached float32 CSR; `check_sparse_path()` guard. Dense stays the default. |
| `src/tuning.py` | `run_search()` (time-boxed driver), `save_oof()` / `load_oof()` / `available_oof()`. `run_trial` and `load_trials` untouched. |
| `src/ensemble.py` | **new** — rank conversion, weighted blending, simplex weight sweep, NNLS stacker, diversity table, seed averaging. Later gained `threshold_at_share`; see `ENSEMBLE_PLAN.md`. |
| `src/combiners.py` | **new**, added with `ENSEMBLE_PLAN.md` - the four combiner families and the out-of-fold evaluation harness. |
| `.gitignore` | tracks `data/processed/oof/*.npy` and `data/processed/ensemble_trials/*.json`, same pattern as `tuning_trials/`. |
| `requirements.txt` | `scipy` pinned explicitly (it was already a sklearn dependency). |

`05_tuning.ipynb` and notebooks 01–09 are **untouched** — they are the record of the journey
for the Task 4 report. `05f` writes its trials under the new keys `lightgbm_v2_stage1` /
`lightgbm_v2_stage2` so the old `lightgbm_stage1` history cannot be contaminated; section 4
asserts it still holds exactly its original 8 trials.
