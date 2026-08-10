# Round 5 - raw text, domain clustering, and per-group share

> **Historical planning document.** This is the round-5 work plan as it stood at the time, kept as the record of what we intended and what we predicted. Its "current best" figures and open questions are superseded: the project finished at **0.81249** public Macro F1, and Tasks 1 and 2 were completed. See `README.md` and `notebooks/SUBMISSION.ipynb` for the final state.

**Members:** Cliffton, Brian, Jovyan, Koko · **Deadline:** 10 Aug 2026

Current best at the time of writing: **0.73754** (`ensemble_weights_hill_climb_share50.csv`). Noise floor 0.0084. The project finished at
**0.81249**.

Supersedes [`ENSEMBLE_PLAN.md`](ENSEMBLE_PLAN.md). Read that file's "If nothing ships"
section first if you have not, because round 5 exists precisely because that is what
happened.

---

## Why this round exists

Rounds 3 and 4 both optimised the **model** while holding the **feature representation**
fixed, and both ran out of road:

| round | what was tried | local gain | Kaggle gain |
|---|---|---|---|
| 3 | ensemble of two linear models | +0.0017 CV | ~0 |
| 4 | 4 combiner families, 48 configs | +0.0075 OOF AUC | +0.0017 |

Two rounds of local gains that mostly did not transfer is a pattern, not bad luck. The
COLING paper (`2501.11012v2.pdf`) explains it.

### The test set is different source corpora, not a random subsample

Training data is HC3 + M4GT + MAGE. The test split is CUDRT + IELTS + NLPeer + PeerSum +
MixSet. **Zero overlap.** The paper's Table 1 arithmetic gives train 62.6% machine and
test 53.1%, independently confirming the 0.6252 this project measured and the ~0.53 it
inferred. Table 8 shows the test set is five sub-populations of very different difficulty
and class balance:

| source | share of test | machine share | top-team accuracy |
|---|---|---|---|
| CUDRT | 31% | 46.5% | 67-76% |
| IELTS | 33% | 53.9% | 65-90% |
| NLPeer + PeerSum | 31% | 50.2% / 57.9% | 92-97% |
| MixSet | 5% | 82.8% | **48-67%** |

### Three consequences

1. **The supplied features are the wrong basis for a domain shift.** They are top-5000
   TF-IDF over lemmas with stop words removed, so they are a pure *content* signal.
   Content vocabulary is exactly what changes between corpora. Function words,
   punctuation, orthography and layout transfer across topics, and the course
   preprocessing removed precisely those.
2. **Standard 5-fold CV is the wrong selection criterion.** It trains and tests on the
   same domains, so it rewards whatever memorises them. That is the most likely mechanism
   behind the two rows in the table above.
3. **One global predicted share is a compromise across five populations.** Share moves the
   score roughly four times more than model choice does, and the sub-populations range
   from 46.5% to 82.8% machine.

### What the repo already told us

- **Raw text is fully available and essentially untouched.** `train.csv` has genuine
  natural text with casing, punctuation, newlines and markdown. The only prior analysis is
  one `describe()` of length in `01_eda` section 3.
- **The two test id groups are measurably different documents.** Numeric-id rows have
  median 1,862 characters against the UUID rows' 1,189, markdown `**` at 2.52 per document
  against 0.44, and "reviewer" in 15.8% against 0.2%. They are the peer reviews. **You do
  not need clustering to separate them; the id format already does.**
- **Ten texts appear in both train and test, all label 0.** Recorded as a data-quality
  observation from duplicate detection. Ten rows out of 6,999 is well inside the noise
  floor either way.

**Honest expectation for the round: 0 to +0.02.** Wider than round 4 in both directions,
because this changes the representation rather than the estimator.

---

## The idea that ties it together

Notebook 13 clusters the training rows into pseudo-domains. Notebook 14 then scores every
feature family **twice**: once under the standard protocol and once holding out a whole
cluster.

| column | meaning |
|---|---|
| `cv_standard` | locked 5-fold stratified CV, same domains in train and test |
| `cv_grouped` | leave-one-cluster-out, a document type never trained on |
| `transfer_gap` | `cv_standard - cv_grouped`, how much is domain memorisation |

**The shipped feature set is chosen on `cv_grouped`.** A family with a large
`transfer_gap` is dropped even if it tops the standard table. Rule fixed now, before any
result exists.

This is round 4's lesson one level up. There, held-out AUC ranked `meta/gbm` first and
`weights/hill_climb` sixth on a 0.0004 separation, well inside the bar. The optimism gap
separated them by 40x, and on Kaggle the low-gap combiner gained eight times more. The
mean did not predict transfer; the gap did.

---

## Run order and ownership

**Run 11 -> 12 -> 13 -> 14.** Notebook 11 is independent and cheap; 14 needs both 12 and 13.

| notebook | what it does | owner |
|---|---|---|
| `11_group_share.ipynb` | per-group share, duplicate leak, leaderboard probe | Cliffton |
| `12_text_features.ipynb` | build 9 blocks, re-run the baseline suite | all four, split by cost |
| `13_clustering.ipynb` | K-means, error analysis, the grouped-CV protocol | Koko |
| `14_ablation.ipynb` | leave-one-family-out under both protocols | Cliffton |

### Notebook 12 split (change `ME` and nothing else)

| owner | representations |
|---|---|
| Cliffton | `tfidf_supplied` (the control), `block_A` |
| Brian | `block_B` through `block_G` (dense, fast) |
| Jovyan | `block_H`, `block_I`, `text_all` (sparse, slow) |
| Koko | `style_all`, `supplied_plus_style`, `supplied_plus_all` |

Cliffton builds and caches all nine blocks first (section 3), then everyone pulls. Results
merge through `data/processed/tuning_trials/` exactly as the round-4 searches did.

```bash
git add data/processed/tuning_trials/
git commit -m "feat: representation baselines for <yours>"
git push
```

---

## The nine feature blocks

Built by `src/text_features.py`. **No new dependencies:** sklearn 1.8 already has
stop-word lists, `char_wb` n-grams, KMeans and TruncatedSVD.

| block | contents |
|---|---|
| A function words | rates of ~318 English stop words, the signal the course removed |
| B punctuation | per-1000-character rates of ~30 marks, plus `**`, `##`, `---` |
| C casing | uppercase, ALL-CAPS, Titlecase, internal-capital rates |
| D structure | newlines, line lengths, bullets, headers, hard-wrap indicator |
| E length | log counts, word and sentence length moments, **burstiness** |
| F diversity | TTR, hapax rate, Yule's K, MATTR over a 100-word window |
| G readability | Flesch and Flesch-Kincaid, hand-rolled syllable counter |
| H char n-grams | `char_wb` 2-5, TF-IDF, 20k features |
| I word n-grams | word 1-2 **with stop words kept**, TF-IDF, 20k features |

Two worth flagging. **Burstiness** (block E) is the relative variability of sentence
length: human writing mixes long and short sentences, decoded text tends to a steadier
rhythm, and it is one of the better-established single markers in the literature.
**Block I is the direct control** against the supplied features: same family, same corpus,
but keeping stop words and skipping lemmatization. If it wins, the loss is localised to
the course preprocessing, which is a clean finding either way.

### Leakage discipline

Blocks B-G are per-document functions and cannot leak. Block A uses a **pinned**
vocabulary and is not fitted. H and I are the only fitted transforms and see `train_texts`
alone. Notebook 12 asserts this by rebuilding block I from train and comparing to the
cache.

---

## Gates and guards

Round 5 has more ways to fool itself than previous rounds, so the checks are load-bearing.

**Notebook 12 section 2 - known signals.** The builder must reproduce differences measured
independently before it existed (newlines 7.70 human against 4.47 machine, `**` 0.049
against 0.496). Asserts, so a broken builder stops the notebook.

**Notebook 12 section 6 - the control.** `tfidf_supplied` re-run must land near
`baseline_results.csv` (LightGBM 0.73914, LinearSVC 0.72278). Landing far *above* means
something leaked and no other row is trustworthy. This is why the control is re-run in the
same session rather than compared against the stored table.

**Notebook 13 section 4 - the stability gate.** K-means under three seeds must give
pairwise ARI above ~0.8. Below that, the clusters are an initialisation artifact and the
grouped protocol measures noise. **Do not lower the bar after seeing the number.** If it
fails, notebook 14 refuses to run and round 5's conclusion is limited to notebooks 11 and
12.

**Notebook 13 section 7 - the per-cluster share gate.** Requires stable *and* pure against
the id groups *and* varied in class balance. Share is the lever where a mistake costs the
most, and per-cluster share cannot be validated locally at all. Notebook 11's id-group
split is the defensible version of the same idea and needs no clustering.

---

## Submissions

5 per day. Round 5 justifies **three**, spread across the round rather than spent at once.

| # | file | question |
|---|---|---|
| 1 | `probe_uuid_share56.csv` | does the public leaderboard score the UUID rows? |
| 2 | `pergroup_share56_48.csv` | does splitting the share by id group help? |
| 3 | `round5_features_*.csv` | does the new representation help? |

**The probe is the interesting one.** It is identical to the current best except that only
the UUID rows are re-thresholded. If the public score is unchanged, the public leaderboard
contains no UUID rows; if it moves, it does. That settles a question open since round 3
which gates the final private-leaderboard picks, and it costs one slot.

**Do not submit from notebook 12.** Its table is standard-CV only, which is the criterion
this round exists to distrust. Wait for notebook 14's grouped numbers.

Anything within +/-0.0084 is a tie. Round 4's precedent holds: report a null as a null.

---

## What gets written up regardless of the score

1. **The transfer gap.** If the full model's `cv_standard - cv_grouped` is large, every CV
   number in every previous notebook has been flattering by roughly that much. That
   reframes three rounds of results and is the strongest methodological finding available.
2. **Any family that helps in-domain but not across domains.** Notebook 14 section 5 lists
   them automatically. Each is a concrete instance of the trap.
3. **Permutation importances in words.** Notebook 14 section 6 gives named features on
   dense blocks. "Machine text has lower sentence-length variance" is worth more to a
   reader than a feature index, and the supplied features could never produce a sentence
   like that because their vocabulary is anonymised.
4. **The error analysis.** Notebook 13 section 6 is the first time this project can say
   which *documents* are hard rather than which models disagree.

For calibration in the report: the shared task's own fine-tuned RoBERTa baseline scored
**73.42** macro-F1 on the full English test set, and this project sits at 73.75 with
classical models. Not strictly comparable (the course uses a 5,000-row subsample) but
worth stating.

---

## What changed in the shared code

| file | change |
|---|---|
| `src/text.py` | **new** - raw-text loading, id-group split, train/test duplicate detection |
| `src/text_features.py` | **new** - the nine blocks, cached to `data/processed/textfeat_*.npz` |
| `src/clustering.py` | **new** - SVD + KMeans, stability, `cluster_cv`, `threshold_per_group` |
| `src/data.py` | docstring amended: it used to forbid re-vectorizing raw text, which round 5 deliberately reverses for Task 3 only |

Notebooks 01-10 are untouched.

---

## Not in scope

**Tasks 1 and 2 were handled separately.** At the time of writing neither had been
submitted. Both were completed afterwards and are in `notebooks/SUBMISSION.ipynb`.

**Share tuning stays closed** (notebook 09 section 9). Notebook 11 redistributes share
between the id groups at a nearly unchanged global value; it does not reopen the global
share question.
