# Ensembling lab - work division

**Members:** Cliffton, Brian, Jovyan, Koko · **Budget:** ~1h00 each · **Deadline:** 10 Aug 2026

Supersedes sessions 1–2 of [`ROUND4.md`](ROUND4.md). Everything else in that file - the
sparse-load setup, the leaderboard facts, the private-LB pick question - still stands.

Notebook: **`notebooks/10_stacked_ensemble.ipynb`**. Change `ME` and nothing else.

---

## Why we stopped the LightGBM retune

`05f_lightgbm_retune.ipynb` was designed as a 2h30 job for three people. It got partway:

| | trials on disk |
|---|---|
| `lightgbm_v2_stage1`, Cliffton's `learning_rate` band | 126 |
| `lightgbm_v2_stage1`, Jovyan's band | 0 |
| `lightgbm_v2_stage1`, Brian's band | 0 |
| `lightgbm_v2_stage2` | 0 |

Finishing it costs three people another 2h30 each. `ROUND4.md` set the honest expectation
for the retune at around **+0.005**, against a public-leaderboard noise floor of **0.0084** -
so even a full success would land inside the noise on a single submission. The retune is not
wrong, it is just the worse use of what is left of the budget.

**The ensemble question is worth more because it has never actually been tested.** Notebook
07 concluded ensembling did not help, but it tested ElasticNet + calibrated LinearSVC:
probability correlation **0.9745**, disagreeing on **11.4%** of rows. Two members that are
nearly the same model. Its one genuinely diverse member, ComplementNB, was rejected for being
too weak at 0.6580 - so diversity and member quality were never available at the same time.

Notebook 09 supplied the missing pair. At matched share 0.4996, **LightGBM disagrees with
ElasticNet on 956 test rows (13.7%) while scoring +0.0189 higher.** First candidate pair that
is diverse *and* both-strong, and nobody has combined them.

**Honest expectation for this round: +0.003 to +0.010.** That straddles the noise floor. A
null result here is a real finding and gets written up as one - see "If nothing ships" below.

Cliffton's 126 stage-1 trials are on disk but **still untracked** - commit them before
anything else, or that compute is lost the next time the working tree is cleaned:

```bash
git add data/processed/tuning_trials/
git commit -m "feat: lightgbm_v2 stage-1 trials, learning_rate 0.031-0.097"
```

They are evidence for the Task 4 report (the knob-influence analysis in `05f` section 4 still
runs on them), and if a round 5 ever happens they are a head start rather than wasted work.

---

## The shape of the work

Everyone needs the same out-of-fold matrix before any combiner work can start, so this
splits into two phases with a sync point between them.

### Phase A - generate OOF scores · ~30 min · split by compute cost

Each of us fits our assigned base models with `cross_val_predict` on the locked dev folds and
pushes the resulting `.npy` files. 16,000 float32 values is 64 KB, so these are tracked in
git and `git push` is what hands your work to everyone else.

| member | generates | note |
|---|---|---|
| Cliffton | `lightgbm` | 3 seeds × 5 folds, the expensive one |
| Koko | `xgboost` | also recovers `best_xgboost_params.json` |
| Jovyan | `extratrees`, `complementnb` | one moderate fit, one instant |
| Brian | `logreg_elasticnet`, `linearsvc` | `saga` is the slow one here |

Notebook sections **0–3**. A member whose `.npy` already exists is **skipped**, so this is
resumable and a `git pull` removes work rather than duplicating it.

```bash
git add data/processed/oof/ data/processed/best_xgboost_params.json
git commit -m "feat: OOF scores for <your members>"
git push
```

**Sync point 1.** Wait for all four pushes, then `git pull`.

### Phase B - combiner lanes · ~30 min · split by method

Everyone now works on the same merged matrix, and each of us sweeps a different *family* of
combiner. The lanes are orthogonal by construction - A varies the weights, B the decision
rule, C the learned combiner, D the aggregation operator - so nobody recomputes anyone's work
and the merged leaderboard covers four independent axes at once.

| member | lane key | section | what varies |
|---|---|---|---|
| Cliffton | `weights` | 7 | simplex sweep, NNLS, hill-climb, equal |
| Brian | `vote` | 8 | hard majority, weighted, k-of-n, soft vote |
| Jovyan | `meta` | 9 | logistic, tree, boosted stumps, forward selection |
| Koko | `aggregate` | 10 | mean, median, trimmed, power, top-k Borda, Caruana |

Run sections **0–2, 4–6**, then **only your own lane's section**. The other three no-op.

```bash
git add data/processed/ensemble_trials/
git commit -m "feat: <your lane> combiner results"
git push
```

**Sync point 2.** Cliffton then runs sections 11–13.

---

## Per-lane briefs

Each lane ships a `CONFIGS_*` list. Extend it with your own ideas - that is the point of
splitting this way, and Task 3 is marked partly on how many approaches are documented. Every
result is written to its own JSON keyed by a hash of the config, so adding configs never
conflicts with anyone else's work.

### Lane A - weighted rank blending (Cliffton)

All three variants produce a weight vector and take a linear combination of member ranks.
They fail differently: `sweep` is exhaustive but its cost is combinatorial so the grid stays
coarse; `nnls` is closed-form but minimizes squared error rather than AUC; `hill_climb`
optimizes AUC directly but only finds a local optimum. `equal` is the control - it is what
07's combiner amounts to once members share a scale, so the gap to it is the value of
weighting at all.

**Watch:** `sweep` reports a maximum over 3,003 weight vectors, so it is the config most
exposed to selection bias. Read its `auc_gap`.

### Lane B - voting mechanisms (Brian)

Voting discards magnitude and keeps only each member's verdict, so a badly calibrated but
usually-right member still gets a full vote. `soft_vote` on the `raw` matrix is **literally
notebook 07's combiner**, included so the leaderboard shows what 07 scores rather than us
asserting it was wrong.

**The trap, and it is a real one.** A majority vote over 6 members takes only 7 distinct
values. Submissions are made by labelling the top `share × n` rows, so the cut lands *inside*
a block of tied rows and every row in that block gets labelled by its position in the file
rather than by any model. On this data that is about **a fifth of the test set**. Section 6
measures it. Every discrete combiner in the lane therefore carries a rank-based tiebreak; if
you add your own vote rule, build it with `combiners._break_ties(votes, Q)` rather than
returning raw counts, and check `tied_fraction_at_share` before believing any result.

Sweeping `share` here is a genuine knob and *not* the same as the submission share: it moves
where each member's own machine/human line falls before voting, which changes how
conservative the vote is.

### Lane C - meta-learner stacking (Jovyan)

The other three lanes impose a shape on the combination - a weighted sum, a vote count, an
order statistic. This one imposes none and lets a classifier learn the mapping from six ranks
to a label, so it is the only lane that can find non-linear structure such as "trust LightGBM
except when both linear models strongly disagree".

**The trap.** A deep tree memorizes 16,000 rows of a 6-column matrix. Verified on synthetic
data: a depth-20 tree hits in-sample AUC **1.0000** against held-out **0.8561**. Keep depths
shallow and `min_samples_leaf` large, and report `auc_gap` next to every score. If your best
config has a gap several times lane D's, that is a warning, not a win.

`forward_select` is the odd one out - it answers *which members belong* rather than how to
weight them, and reports an explicit subset. 07 dropped ComplementNB by hand; this lets the
data make that call in a form the report can quote.

### Lane D - rank aggregation operators (Koko)

Every operator is unweighted, which is the point: lane A already asks about weights, so this
lane isolates the aggregation *function*. The interesting ones are non-linear, because no
weighted sum can imitate them. `median` and `trimmed` ignore outlying members, so one bad row
from a weak member cannot drag down a row the others agree on - with ComplementNB in the pool
this is the most likely operator to beat the mean. `power` interpolates between a soft
minimum (`p < 1`, a row must satisfy every member) and a soft maximum (`p > 1`, one
enthusiastic member suffices). `caruana` is greedy selection *with replacement*, so a member
picked 8 times out of 25 has an effective weight of 0.32 arrived at greedily.

**Plain Borda count is deliberately absent.** On normalized ranks it is the mean times the
member count - verified to produce byte-identical labels - so offering it would put a fake
distinct operator on the leaderboard. `borda_topk`, which only awards points above a rank
cutoff, is the version that genuinely differs.

---

## How results are judged

**Every combiner is scored out of fold.** Notebook 10 originally fitted its weights on the
same 16,000 rows it reported them on and called the bias small. With four people sweeping
dozens of configurations and a leaderboard that takes the maximum over all of them, it is not:
whichever lane has the most free parameters wins on optimism alone. `combiners.cv_evaluate`
refits each combiner inside the locked folds and reports held-out AUC, so a 3,003-point sweep
and a parameter-free median are compared on equal terms. `auc_insample` and `auc_gap` are
reported alongside, so the bias is measured rather than assumed away.

**Selection is on AUC, not F1 at a cutoff.** AUC *is* ranking quality and is invariant to the
share, which is exactly what the submission procedure uses. F1 at 0.5 is confounded by the
calibration effect notebook 09 spent three rounds isolating - and it is what 07 selected on.

### The bar, written down before any results exist

> The winning combiner ships **only if** it beats the best single member on held-out AUC by
> more than that member's fold-to-fold standard deviation.

Two independent confirmations follow, because a maximum over enough noise is not zero:

1. **The holdout.** 4,000 rows nothing has touched. Read it as a check on the *sign* of the
   gain, not as a second precise estimate.
2. **The runner-up from a different lane.** If the top two configs are structurally different
   and agree, that is much stronger than one config winning by a hair. If the top six are
   variants of the same idea, treat the margin as the search noise it probably is.

---

## Submissions

5 uploads/day. **Only two slots are justified**: the winning combiner, and the runner-up from
a structurally different lane as a hedge.

`lightgbm_share50.csv` at share 0.4996 scored **0.73583** and is already on the board. This
notebook uses the same tuned LightGBM, so that file *is* the matched-share control - the
ensemble is compared against it directly, with no confound and without spending an upload to
re-establish it. Every file is written at exactly share 0.4996, reusing 09's control design,
so any difference is purely combiner quality.

One caveat to record rather than hide: the benchmark was a single-seed fit while the
`lightgbm` member here is rank-averaged over 3 seeds. That difference runs the same direction
for every file, so it does not affect comparisons between them, but the ensemble's margin
over 0.73583 does include whatever seed-averaging is worth.

**Reading the results:** anything inside ±0.0084 is a tie, not a result. The precedent is
09 §9, where a 0.0032 move was correctly called noise instead of written up as a regression.

**Still closed:** share tuning. 09 established the 0.45–0.55 band is flat to within noise.
No new base models either - this round tests whether the members we have can be combined, not
whether a wider pool would help.

---

## If nothing ships

That is a result, and it is a better one than 07's because it was reached properly. 07
concluded ensembling did not help from **one** combiner on **two near-identical** members,
selected on a **confounded metric** and scored **in-sample**. A null from this round would
rest on four combiner families, six members with measured diversity, an unbiased metric and a
bar fixed in advance.

Three things go in the report either way:

1. **The optimism measurement.** `reports/figures/ensemble_combiner_leaderboard.png` plots
   in-sample against held-out AUC per lane. Report which lane had the largest gap and what it
   would have won under the original in-sample selection.
2. **The tie trap.** A vote over 6 members leaves ~20% of a submission decided by row order.
   It also forced a real fix: `ensemble.threshold_at_share` previously took a quantile cut and
   missed the target share by up to **19 rows on 6,999** when scores tied. It now selects the
   top-k exactly.
3. **Diversity versus member quality.** 07 concluded "diversity is necessary but not
   sufficient" from a single rejected member. With ComplementNB, ExtraTrees and two boosting
   models in the pool, check whether `forward_select` and the weight sweep agree about which
   members to drop.

---

## Also outstanding

**`03_pca_knn.ipynb` is still a scaffold** with `pass` in the loop body. That is **5 marks
unstarted**, and Task 2 needs four Kaggle uploads to read back its required Macro F1 at
2000/1000/500/100 components. Against Task 3's remaining upside of at most +3 bonus marks,
it is the better place to spend the next block of anyone's time.

---

## What changed in the shared code

| file | change |
|---|---|
| `src/combiners.py` | **new** - the four combiner families, the out-of-fold `cv_evaluate` harness, and the `ensemble_trials/` ledger. |
| `src/ensemble.py` | added `threshold_at_share`, now an exact top-k selection instead of a quantile cut (it was off by up to 19 rows on tied scores). `macro_f1_at_share` calls it, so the local metric and the submission writer cannot drift apart. Everything else untouched. |
| `src/paths.py` | added `ENSEMBLE_TRIALS`. |
| `.gitignore` | tracks `data/processed/ensemble_trials/*.json`, same pattern as `tuning_trials/` and `oof/`. |

Notebooks 01–09 are untouched. `05f` is left as-is with Cliffton's 126 trials intact.
