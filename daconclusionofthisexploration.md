# What we found looking for a better model

One night's work, 5 August 2026, finished around 3:30am. Everything below is measured.
Nothing below is confirmed on the leaderboard yet.

## The short version

We went looking for a non-deep-learning model that beats **0.80143**, the current best
on the Kaggle leaderboard. No model beat it. The models were never the problem.

What the night actually produced was a correction to the story this project has been
telling itself. We have been treating this as a domain-shift problem — train and test
come from different corpora, so the model gets lost on unfamiliar material. Three
separate measurements say that is not what is happening:

- A classifier built purely to tell training rows from test rows manages only 0.72 AUC,
  and that is its flattering in-sample score. The two sets sit almost on top of each
  other.
- Deleting the features that classifier relies on made things worse, not better.
- Grouping the validation folds by test-likeness costs 0.03. Grouping them by document
  length costs 0.07. Length hurts more than domain does.

We also found, and then talked ourselves out of, an apparently huge lead: one length
band scoring 0.63 while every other band sat between 0.79 and 0.91. It turned out to be
mostly an artifact of how we validate. That story is Finding 2, and it is worth reading
because the same trap is easy to fall into again.

Then five separate corrections all came back at zero or worse — drop the domain-marker
features, reweight rows by test-likeness, reweight by length, fix the threshold, swap
the model family. When that many well-motivated fixes fail in the same direction, the
diagnosis is wrong, not the fixes.

**The diagnosis: this is concept shift, not covariate shift.** In-domain validation,
corrected for everything we can actually measure, predicts 0.8892. The leaderboard gives
0.80143. That 0.088 gap is not the features moving — it is the features *meaning*
something different in IELTS essays and peer reviews than they did in HC3 and MAGE.
Every tool we used tonight assumes the opposite. It is also the reason a year of feature
engineering in this repo bought less than one thresholding trick did.

Two small real gains did survive. The ledger has been scoring models at a threshold we
never submit with, worth about +0.01 to fix. And the blending code, pointed at the
winning features for the first time, beat the best single model by +0.0068 AUC — the
only one of nine interventions to clear its own bar.

One submission file is worth a slot: `night_ens_weights_nnls.csv`.

## Where we started

The current best model is LightGBM with default settings, running on the 40,385
hand-built text features (blocks A-F, H, I), with the predictions cut per id group at
fixed shares: 0.6198 for the uuid rows, 0.4756 for the numeric rows.

Two numbers matter for judging anything new:

| Check | Score |
|---|---|
| 5-band length CV (what we pick on) | 0.8077 |
| 3-band length CV (what we confirm on) | 0.8089 |
| Kaggle leaderboard | 0.80143 |

We re-ran both CV numbers from scratch at the start of the night and got exactly
0.8077 and 0.8089. The old results reproduce, so the ground under us is solid.

## Finding 1: the train/test gap is not a feature-space gap

We trained a classifier whose only job was to guess whether a row came from the
training set or the test set. It never sees the human/machine label, so there is no
cheating involved.

If train and test lived in clearly different parts of feature space, this classifier
would score close to 1.0 and we would know exactly what to fix. It scored **0.7204,
and that is its in-sample score** — the score it gets on the very rows it trained on,
which is the most flattering number it can possibly produce. The honest number is
lower.

So the two sets sit almost on top of each other in this feature space.

This matters because it rules things out. Our CV score drops about 0.09 when we group
the folds instead of shuffling them, and the obvious story was "test documents look
different, so the model is lost." That story is now hard to defend. The documents
look similar. What has changed is what those features *mean* — the same writing
pattern points to "machine" in the training corpora and to something else in the test
corpora.

That is a much harder problem than a feature-space shift, and it explains why a year
of feature engineering in this repo bought less than the share-threshold trick did.

It also tells us something practical: reweighting the training rows to look more like
the test rows cannot buy much, because they already look alike. We ran that experiment
anyway (results below) rather than trust the reasoning alone.

## Finding 2: the band that looked like a goldmine and wasn't

This one is worth reading as a story rather than a result, because we spent an hour
convinced we had found the answer, and the way it fell apart is instructive.

The 0.8077 headline score is an average over five length bands. The bands do not fail
evenly:

| Band | Length (characters) | Machine share | Score |
|---|---|---|---|
| 0 | 27 – 387 | 0.567 | **0.6266** |
| 1 | 388 – 916 | 0.603 | 0.7915 |
| 2 | 917 – 1,382 | 0.616 | 0.8766 |
| 3 | 1,383 – 2,188 | 0.684 | 0.9073 |
| 4 | 2,189 – 17,436 | 0.656 | 0.8364 |

Four bands out of five are between 0.79 and 0.91. One band is at 0.63, and it is
dragging the average down by itself.

The class balance column rules out the boring explanation. Band 0 is the most balanced
band of the five at 0.567 machine, and a balanced band should make macro F1 *easier* to
score well on, not harder. Whatever is wrong with band 0, it is not skew.

Do the arithmetic. Lift band 0 from 0.63 to 0.80 and change nothing else, and the mean
goes from 0.808 to about 0.842. That single band holds more headroom than any model
swap on the table.

**Except that arithmetic is wrong, and it took ten seconds of checking to find out.**
The five bands are equal fifths of our *training* data. They are not equal fifths of the
test set, and we had never looked:

| Band | Share of dev | Share of test |
|---|---|---|
| 0 (shortest) | 20% | **11%** |
| 1 | 20% | 16% |
| 2 | 20% | 11% |
| 3 | 20% | 28% |
| 4 (longest) | 20% | **34%** |

Test documents are considerably longer than training documents — median 1,723 characters
against 1,146. Our worst band covers barely a tenth of what we are actually graded on,
and our two best bands cover well over half.

Reweighting the band scores to the test set's actual composition gives 0.830 rather than
0.808. So the headline number is pessimistic about test *composition* — and yet the real
leaderboard score is 0.80143, closer to the pessimistic figure than the reweighted one.
Something on the test set is harder than our dev bands predict, and equal weighting has
been accidentally cancelling it out.

Then the second shoe dropped. When band 0 is the held-out fold, the model trained on
bands 1 to 4 — meaning it has **never seen a document under 388 characters**. That is
not a hard problem, it is an unfamiliar one, and the two call for opposite responses. So
we re-ran the same model under random folds, where every training half contains short
documents, and measured each band's AUC again:

| Band | AUC, band held out | AUC, random folds | Recovered |
|---|---|---|---|
| 0 (shortest) | 0.6676 | **0.8804** | **+0.2128** |
| 1 | 0.8919 | 0.9303 | +0.0384 |
| 2 | 0.9556 | 0.9637 | +0.0081 |
| 3 | 0.9780 | 0.9809 | +0.0029 |
| 4 | 0.9664 | 0.9874 | +0.0210 |

Band 0 recovers by 0.21. Every other band moves by 0.04 or less. Short documents are
only catastrophic when the model has never been shown one — and both the real training
set and the real test set contain plenty, so that situation never arises when it counts.

Short text is still somewhat harder than long text once you account for this: 0.8804
against 0.93 to 0.99 elsewhere. But that is an ordinary weakness, not a bottleneck. Take
the artifact out and scale for the fact that band 0 is a tenth of the test set, and this
whole lead is worth maybe 0.01, not 0.034.

Dead end — but an expensive-looking one that cost 20 minutes to close, and it left two
real things behind, in the next two sections.

Why would short documents be hard? A good guess is that our style features stop meaning
anything when there is not much text to measure. MATTR reads a 100-token window. Yule's
K and the burstiness numbers need enough sentences to have a spread. On a 200-character
document those features are mostly noise — and the model has learned to trust them,
because on long documents they work well. Notebook 14 found the diversity block to be
the single most valuable one overall, which fits: excellent on long text, misleading on
short.

Before doing anything about it we checked a cheaper explanation, and the check turned up
a problem of its own.

Every trial in this repo's ledger was scored by `predict()`, which cuts at probability
0.5. We do not submit that way — submissions take the top k rows by score. That
difference once moved the leaderboard from 0.65738 to 0.73583 on an unchanged model, so
it is not a detail. If band 0's score was really about the cutoff sitting in the wrong
place for short documents, there would be nothing to fix in the features.

It is not the cutoff. Band 0 gets *worse* under share thresholding, 0.6266 down to
0.6122, and its AUC gives the game away:

| Band | AUC |
|---|---|
| 0 (shortest) | **0.6676** |
| 1 | 0.8919 |
| 2 | 0.9556 |
| 3 | 0.9780 |
| 4 | 0.9664 |

AUC does not care where you put the threshold; it measures ranking alone. At 0.6676 the
model is barely above coin-flipping on short documents while it is near-perfect on
everything else. That is a ranking failure, and no amount of threshold tuning touches it.

The check paid for itself anyway. Scoring at the share instead of at 0.5 is worth
+0.0104 across the grouped protocol (0.8077 becomes 0.8181), which means every
comparison in the ledger has been made with a mildly wrong ruler. Small, systematic, and
worth fixing going forward.

## Finding 3: length hurts more than domain does

We built a second set of folds. Instead of grouping documents by length, we sorted the
training rows by how test-like the discriminator thinks they are and grouped on that.
Holding out the most test-like group is as close as we can get to a dress rehearsal for
the real leaderboard.

The same model, same features, three ways of grouping:

| Grouping | Score |
|---|---|
| Length, 3 bands (the protocol of record) | 0.8089 |
| Length, 5 bands | 0.8077 |
| Test-likeness, 3 bands | **0.8489** |
| Random / standard CV | 0.8764 |

Grouping by test-likeness costs 0.03 against random. Grouping by length costs 0.07 —
more than twice as much.

Put that next to Finding 1 — the discriminator can barely tell train from test in the
first place — and a different story emerges from the one this project has been telling
itself. The thing our model struggles with is not that test documents come from other
corpora. It is that documents of unfamiliar length are hard, and the length bands make
the model face exactly that.

There is a wrinkle, and it is a big one. We now know *why* the length protocol is harsh
— it forces the model to extrapolate to a length range it never trained on, which
Finding 2 showed costs band 0 about 0.21 of AUC all by itself. And we know that
situation never arises at test time, because both the training set and the test set
contain documents of every length.

By that reasoning the length protocol should be far too pessimistic. Instead it predicts
the leaderboard almost exactly: 0.8089 against a real 0.80143, and 0.7864 against 0.77942
the round before. The friendlier shift protocol, which ought to be the fair one, would
have predicted 0.849 and been badly wrong.

So we have a protocol that is right twice for a reason we have just disproved. Two
possibilities: the length penalty is coincidentally cancelling some other test-set
difficulty of similar size, or two data points is simply not enough to conclude anything.
We are keeping the length protocol either way — the whole ledger and the 0.0084 noise
floor are calibrated against it — but this deserves a line in the report rather than
quiet acceptance.

## Finding 4: NBSVM is worse here, clearly

NBSVM is the standard strong non-neural baseline for text classification (Wang and
Manning, 2012). It scales each n-gram column by how much that n-gram favours one class,
then fits a linear model on top. It had never been tried on this project. It is the
first thing a reviewer would ask about.

It loses, and not narrowly:

| Setup | 5-band CV | vs baseline |
|---|---|---|
| NBSVM + logistic, C=4 | 0.7647 | -0.0430 |
| NBSVM + logistic, C=1, alpha=0.25 | 0.7562 | -0.0515 |
| NBSVM + LinearSVC, C=0.1 | 0.7514 | -0.0563 |
| NBSVM + logistic, C=1 | 0.7495 | -0.0582 |
| LightGBM defaults (baseline) | 0.8077 | — |

Every version is 4 to 6 points behind. This is not a tuning problem; a linear model on
n-grams simply cannot represent whatever LightGBM is finding in the style features.

Worth keeping anyway, for one reason. NBSVM gets its answers a completely different
way from a boosted tree, and blends work best when the members disagree. It is a bad
model and a promising ensemble member, and those are not contradictory.

The whole NBSVM sweep took 37 seconds. Cheap answers to obvious questions are worth
buying.

**A footnote that turned out to matter.** Building the ensemble, we ran a plain
LinearSVC on the same 40,385 features and it scored 0.5853 AUC — barely better than
guessing — after burning 28 minutes and printing a convergence failure on every one of
its four fits.

That is not a modelling result, it is a scaling bug, and the convergence warnings say so
directly. The style blocks are raw rates and counts whose standard deviations span a
factor of roughly 460,000. A linear model with a single regularisation constant cannot
cope with that: the largest-scale columns swallow everything, and the optimisation is
ill-conditioned enough that liblinear gives up at 5,000 iterations. LightGBM never
noticed, because trees split on order rather than magnitude.

Two things follow. The same LinearSVC scored 0.7228 on the old supplied TF-IDF, where
every column shares a scale — so the earlier model comparison in notebook 04 was, for
the linear models, partly a measurement of how uniform the features happened to be. And
NBSVM's respectable 0.8168 next to this 0.5853 is mostly explained by the log-count-ratio
step doing the rescaling that nobody wrote by hand. Any future linear member on this
representation needs a scaler in front of it.

## Finding 5: removing domain markers backfires

We took the 500 features the train/test classifier relies on most and deleted them, on
the theory that they carry domain identity rather than evidence. The score dropped to
0.7903, which is 0.017 worse than leaving them alone.

That is Finding 1 showing up again from another angle. The features that separate the
two corpora are the same features that separate machine text from human text. There is
no clean set of "domain-only" columns to throw away.

We stopped this line of work there rather than finishing the 2,000 and 5,000 versions.
Deleting more of the same features was not going to reverse the sign, and by then
Findings 2 and 3 had pointed somewhere better. The density-ratio weighting test went
with it, for a reason worth recording: the discriminator turned out to be entangled
with the label. Its most test-like third of the training data is 74.3% machine, against
54.1% for the least test-like third — so weighting rows by test-likeness would quietly
have been weighting them by *class*, and any gain would have been a class-balance
effect wearing a domain-adaptation costume.

## Finding 6: length is the one gap that is actually there

Everything above says the corpora are not far apart in feature space. But their document
*lengths* clearly are, and that is measurable rather than inferred. So the correction is
worth making on the variable where the shift is real.

Weight each training row by how over- or under-represented its length band is in the
test set: 0.56 for the shortest band, 1.70 for the longest. This is the same
covariate-shift correction that failed in Finding 1, applied to a variable that carries
no label information and where the gap demonstrably exists.

It does nothing. Test-weighted macro F1 moves -0.0008 and AUC -0.0011 — both well inside
noise, and both the wrong sign. The length shift is real, and the model was simply never
suffering from it.

## Finding 7: what is left is the kind of shift we cannot fix

Stack up what the night ruled out and a single answer remains.

Under random folds, with the length bands reweighted to match the test set's actual
composition, the model scores **0.8892**. That is our best honest estimate of test
performance if the only differences between train and test were the ones we can see.

The real leaderboard score is **0.80143**. The gap is 0.088, and nothing we measured
accounts for it:

- Not the feature distribution. The train/test discriminator manages 0.72 AUC in-sample,
  and correcting for it — by dropping marker features or by reweighting rows — made
  things worse both times.
- Not document length. Correcting for the one gap we could measure exactly changed
  nothing.
- Not short documents. They recover to 0.88 AUC once training contains them, and they
  are a tenth of the test set.

What is left is the awkward one. The features look the same in both sets; what differs
is what they *mean*. A pattern that marks machine writing in HC3 or MAGE apparently
marks something else in IELTS essays or peer reviews. That is concept shift rather than
covariate shift, and it is a genuinely harder problem: no amount of unlabeled test data
fixes it, because the unlabeled data cannot tell you the label relationship has changed.

This explains something that has puzzled this project since round 5 — why a year of
feature engineering bought less than a thresholding trick did. The features were never
the bottleneck.

The prediction this makes is testable, and it held. If the problem is meaning rather
than distribution, then showing the model the test set's *vocabulary* should not help
either — the words would still mean what they meant in training. We rebuilt the
character and word n-gram vectorisers on train plus test text, so that every n-gram
appearing only in the test corpora became visible to the model. It scored 0.8058 against
the baseline's 0.8089: worse, on all three folds. Six corrections, six failures, one
consistent explanation.

It also names the only remaining lever, and explains why that one failed too.

Self-training manufactures labels on the target domain, which is the single thing that
can move a concept shift. We simulated it: hold out a band, treat it as unlabelled,
label the most confident 30% from the model's own predictions, retrain including them.
The per-fold numbers show exactly what the method does and does not do.

| Held-out band | Pseudo-label accuracy | Before | After |
|---|---|---|---|
| 0 (shortest) | 0.690 | 0.6266 | **0.4602** |
| 1 | 0.928 | 0.7915 | 0.7989 |
| 2 | 0.989 | 0.8766 | 0.8800 |
| 3 | 0.998 | 0.9073 | 0.9094 |
| 4 (longest) | 0.952 | 0.8364 | **0.7601** |

Where the invented labels are nearly perfect, self-training gives back a rounding error.
Where they are 69% right, it poisons the training set and costs 0.17. This is the method
doing precisely what the textbook says it does — amplifying the model's existing
mistakes — and it is why the simulation gate existed rather than being skipped.

Note the trap in the middle rows: three folds improved. Had we run only those, this
would look like a modest win worth submitting.

Keeping half the test rows instead of a third does better — mean 0.8122 against the
baseline's 0.8077, the first positive number of the night. It still does not pass. The
gain is +0.0045, under the project's 0.0084 noise floor, and it wins on 3 folds out of
5 where the bar is all 5. Look at where it comes from and the reason to distrust it is
obvious: band 4 jumps +0.055 while band 0 drops -0.042, and the average of one large
gain and one large loss is not a small reliable gain. It is two unrelated effects
cancelling.

The direction is interesting though. More pseudo-labels helped, not fewer, even though
the extra ones are less confident. Pushing to keep 0.7 gave +0.0149 — over the noise
floor, better on 4 folds of 5 — and almost all of it came from one place: band 4 went
0.8364 to 0.9187 while the other four moved by 0.005 or less.

Band 4 is the longest documents, and under this protocol its model trained on bands 0-3
and had never seen a long one. So handing it pseudo-labelled long documents does not
adapt it to anything. It hands back the length range the protocol confiscated. That is
Finding 2's artifact again, running the other way — and this time it produces a *good*
number, which is the far more dangerous direction to be wrong in.

We tested it rather than argue about it. Re-run the same self-training under random
folds, where every training half already contains every length and there is no missing
band to restore:

| Protocol | Pseudo-label gain |
|---|---|
| Length bands (one length range withheld) | **+0.0149** |
| Random folds (all lengths present) | **-0.0001** |

The entire gain was the artifact. Label accuracy under random folds is 0.96 to 0.98 —
far better than the grouped run ever managed — and it still buys exactly nothing. Self-
training is not adapting to the test corpora here. It was patching a hole the validation
scheme dug.

This is the result that most deserves to be in the report. A method that looked
submittable on the project's own protocol, cleared its noise floor, won on four folds of
five, and was worth precisely zero.

## Finding 8: the ensemble works, and it is the only thing that did

The blending code in `src/combiners.py` was built in round 4 and has only ever been run
against the old supplied-TF-IDF members. Tonight it finally saw the winning
representation. Four members went in — LightGBM, NBSVM, ExtraTrees, and the broken
LinearSVC — and the rank correlations show why a blend had something to work with:

|  | lgbm | nbsvm | extratrees |
|---|---|---|---|
| **lgbm** | 1.000 | 0.586 | 0.616 |
| **nbsvm** | 0.586 | 1.000 | 0.791 |
| **extratrees** | 0.616 | 0.791 | 1.000 |

LightGBM agrees with the other two only about 0.6 of the time by rank. That is real
disagreement, and disagreement is the raw material a blend converts into accuracy.

Six combiners beat the best single member's 0.8834 AUC. Non-negative least squares wins
on both axes at once — the highest out-of-fold AUC **and** the lowest optimism gap:

| Combiner | AUC | Optimism gap |
|---|---|---|
| **weights / nnls** | **0.8902** | **0.0053** |
| aggregate / caruana | 0.8886 | 0.0070 |
| weights / hill climb 0.02 | 0.8887 | 0.0071 |
| weights / hill climb 0.05 | 0.8879 | 0.0083 |
| meta / logistic C=1 | 0.8877 | 0.0086 |
| best single member (LightGBM) | 0.8834 | — |

That matters because round 4's hard-won lesson was that the optimism gap, not the
held-out score, predicted which blends survived the leaderboard. Usually those two
criteria disagree and you have to choose. Here they point at the same combiner, so there
is nothing to argue about.

The gain is +0.0068 AUC over LightGBM alone. Modest, and AUC is not macro F1 — but it is
ranking quality, which is exactly what the share-threshold submission consumes. It is
also the only intervention out of eight tonight that cleared its own bar.

The weights it chose are worth looking at, because they are not the weights anyone would
have written by hand:

| Member | Own AUC | NNLS weight |
|---|---|---|
| lgbm | 0.8834 | 0.648 |
| nbsvm | 0.8168 | 0.294 |
| linsvc | 0.5853 | 0.058 |
| extratrees | 0.7959 | **0.000** |

ExtraTrees is the second-strongest member by AUC and gets dropped completely, while the
broken LinearSVC keeps a small slice. That looks wrong until you check the correlation
table: ExtraTrees agrees with NBSVM 0.79 of the time, so once NBSVM is in the blend it
has almost nothing left to add. LinearSVC agrees with nothing, so even as mostly-noise
it occasionally breaks a tie the others get wrong.

This is the whole argument for weighted blending in one table. A blend rewards members
for being *different*, not for being *good*, and those come apart. It is also a mild
worry — 5.8% on a member that scores 0.5853 could be NNLS fitting noise — but the weight
is small and the configuration was selected on out-of-fold scores with the lowest
optimism gap in the field, which is the check designed to catch exactly that.

The unweighted lanes show what happens without this discrimination: plain mean scored
0.8639 and soft vote 0.8504, both below the best single member, because they swallowed
the broken member whole.

## The full ledger

Nine interventions. One worked.

| Intervention | Effect | Verdict |
|---|---|---|
| Ensemble, NNLS over 4 members | **+0.0068 AUC** | works, submitted |
| Score at share instead of 0.5 cutoff | **+0.0104** | real, a scoring fix not a model |
| Self-training, keep 0.7 | +0.0149 → **-0.0001** | artifact, discarded |
| Self-training, keep 0.5 | +0.0045 | under the noise floor |
| Length-matched reweighting | -0.0008 | no effect |
| Vocabulary refit on train+test | -0.0031 | worse, on all 3 folds |
| Drop 500 domain-marker features | -0.0174 | worse |
| NBSVM, best of 4 variants | -0.0430 | much worse |
| Self-training, keep 0.3 | -0.0460 | much worse |
| Seed bag, 5 seeds | 0 rows changed | no effect at all |

Two things were cut on the evidence rather than for time: the remaining model families
(CatBoost, RBF-SVM, HistGradientBoosting, a polynomial sketch — about ninety minutes),
because nothing here suggests model choice is the constraint; and the short-document
feature work, once Finding 2 collapsed.

## About the 0.85 target

0.85 is a long way from 0.80143, and it is worth being straight about that.

The published results on this dataset were won with fine-tuned transformers, which
Task 3 rules out. Our grouped CV tracks the leaderboard closely, so reaching 0.85 on
Kaggle means reaching roughly 0.85 locally — a jump of 0.04. Nothing in this repo's
history has moved the number that far except the share-threshold trick, and that one is
already spent.

For an hour tonight I thought band 0 was the answer — 0.034 of the 0.04 we needed,
sitting in one place, with a clear cause. Then we looked at how long the test documents
actually are, and most of that headroom evaporated.

Finding 7 is the real answer, and it is not the one anyone wants. The 0.088 gap between
what in-domain validation predicts and what the leaderboard gives us is concept shift:
the features mean different things in the test corpora. Every tool we brought tonight
assumes the features mean the same thing and only their distribution moved. That is why
five separate corrections all landed at zero or worse. They were solving a problem we do
not have.

So: no, we do not have a credible path to 0.85 tonight, and I would rather say that
plainly than dress up a number that will not survive contact with the leaderboard. What
we do have is a set of small, independent gains worth roughly 0.01 apiece — the
share-versus-cutoff scoring fix, an ensemble that has never been tried on the winning
features, and possibly the vocabulary refit. If they land and do not overlap, 0.82 is
reachable. 0.85 needs a different idea, and the honest candidates all involve either
labelled target data or the transformer models Task 3 forbids.

The one thing tonight genuinely changed is that we now know *which* problem to write the
report about. That is worth more to the Task 4 marks than another 0.005 would have been.

## What to submit

Three files were written, and only one is worth a slot.

| File | Rows differing from the 0.80143 benchmark | Verdict |
|---|---|---|
| `night_ens_weights_nnls.csv` | **326** | **Submit this** |
| `night_lgbm_seedbag.csv` | 0 | Skip - identical file |
| `night_lgbm_defaults.csv` | 0 | Skip - it is the pipeline test |

The seed bag deserves a sentence, because "average five seeds, it cannot lose" is
standard advice and here it did precisely nothing. Not a small gain — *nothing*. Five
LightGBM seeds rank-averaged move zero rows across the threshold, which means the
model's ranking of these 6,999 documents is completely stable across seeds. There was no
seed noise to average away. Cheap to learn, and it retires an idea that would otherwise
sit on the to-do list looking sensible forever.

That leaves the NNLS blend as the night's single submittable candidate. It changes 326
rows, comfortably past the ~100 needed to resolve anything against the 0.0084 noise
floor, so whatever the leaderboard says about it will be a real answer rather than a
coin flip.

Keep `chosen_pergroup62_48.csv` as the second final-selection slot regardless. It is the
known 0.80143 and nothing tonight has earned the right to replace it unseen.

## The pipeline reproduces exactly

Worth stating plainly because it underwrites everything above. We rebuilt the current
best model from scratch through tonight's code — same features, same LightGBM defaults,
same per-group share cut, same duplicate patch — and wrote a fresh submission file. It
differs from the existing 0.80143 file in **0 of 6,999 rows**.

Not "close". Identical. So when a number in this document says an idea lost 0.017, that
is the idea losing 0.017, not a plumbing difference somewhere in a new script.

The writer flagged the file as under-powered and not worth a submission slot, which is
correct — submitting a byte-identical copy of a file already on the leaderboard tells us
nothing. It is a test, not a candidate.

## Housekeeping

New code from tonight:

- `src/nbsvm.py` — the NBSVM classifier, written to plug into the existing trial runner.
- `src/evaluation.py` — two additions: `FoldSplitter` lets the blending code accept
  grouped folds, and `oof_from_folds` produces out-of-fold scores when the folds do not
  cleanly partition the data.
- `experiments/` — one script per stage, plus `run_night.py` to chain them. Every trial
  writes its own JSON file the moment it finishes, so killing a run costs one trial and
  restarting picks up where it stopped.
- `catboost` added to `requirements.txt`.

Three practical notes, all the same lesson in different clothes: on 40,000 columns,
library defaults are expensive.

The first train/test classifier was set to 3,000 solver iterations and ran fifteen
minutes without finishing. Two hundred gives the same answer for our purposes — we need
a feature ranking and a rough probability, not a converged fit.

XGBoost killed the first ensemble run outright. Its histogram tree method asked for
12.4 GB on this matrix, on a 15 GB machine with about 1 GB free. It is out of the member
pool now, and no great loss: its errors correlate with LightGBM's, so the blend gives up
little. Worker threads are capped rather than set to -1 for the same reason — sixteen
copies of the intermediate state is what exhausts memory, not the model itself.

That crash cost a finished LightGBM fit, because a failure in one member took down the
whole stage. Each member is wrapped individually now. Worth doing before the run rather
than after.
