# LiteLightGBM: implementation contract

## Status

`src/lite_lightgbm.py` is the stable public module for the complete documented
LightGBM-like model. Its model routines implement the stated scope, validation
rules, and deterministic training flow.

The completed extraction-only refactor specified in
[`lite_lightgbm_refac.md`](lite_lightgbm_refac.md) keeps the estimator in the
public module and places shared core, binning, and tree internals in the
`lite_lightgbm_dep` implementation package. It preserves imports, results, validation, serialization
behavior, and runtime semantics.

The goal is a deterministic, sparse, histogram-based gradient-boosted tree classifier
that behaves similarly to the subset of LightGBM used by this project. The goal is not
bitwise equality with LightGBM or compatibility with its complete API.

## Supported scope

The first complete version supports:

- binary labels `0` and `1`;
- numeric NumPy arrays and SciPy sparse matrices;
- binary log loss with first- and second-order derivatives;
- histogram-based numeric splits;
- leaf-wise best-first tree growth;
- L1 and L2 leaf regularization;
- row and feature subsampling;
- `None`, `"balanced"`, or explicit binary class weights;
- deterministic serial training from a fixed random seed;
- raw-score, probability, and binary-label prediction.

## How the model works at a high level

The model is an additive collection of small decision trees. It begins with one constant
prediction for every document and then builds trees sequentially. Each new tree is a
correction to the mistakes made by the trees that came before it.

The complete training flow is:

1. **Bin the feature values.** Continuous values are converted into a limited number of
   ordered bins. The tree searches boundaries between bins rather than every distinct
   floating-point value. Bin-boundary candidate scanning is NumPy-vectorized, while the
   documented deterministic selection, feasibility checks, and lower-boundary tie-break
   remain unchanged. Binning is performed once and reused by every tree.
2. **Make an initial prediction.** The initial raw score is the log-odds of the weighted
   positive-class frequency. Before any trees are fitted, every row receives this score.
3. **Measure the current errors.** Convert each row's raw score to a probability and
   calculate its gradient and Hessian under binary log loss. The gradient gives the
   direction and size of the required correction; the Hessian describes the local
   curvature and scales how confidently that correction should be made.
4. **Build one histogram tree.** For every current leaf and candidate feature, accumulate
   gradient sums, Hessian sums, and row counts per bin. Prefix sums then make it possible
   to score every split boundary without repeatedly scanning all rows.
5. **Grow the tree leaf-wise.** Instead of expanding every leaf at the same depth, split
   whichever current leaf offers the largest reduction in loss. Continue until the leaf,
   depth, or minimum-child constraints stop further useful growth.
6. **Assign leaf corrections.** Each terminal leaf receives one regularized value derived
   from the gradients and Hessians of the rows in that leaf. L1 and L2 penalties shrink
   unreliable corrections toward zero.
7. **Update the ensemble.** Add the new tree's output, multiplied by the learning rate, to
   every row's raw score. Recalculate gradients and Hessians, then fit the next tree.

Prediction follows the same representation but does not perform optimization. Bin a new
row with the learned boundaries, route it through every tree, add each selected leaf
value using the learning rate captured during fitting, and apply the sigmoid function.
`predict` uses the same tie behavior as `LGBMClassifier`: class `1` requires a probability
strictly greater than `0.5`, so an exact `[0.5, 0.5]` tie becomes class `0`.

In this project, the estimator's main job is to produce useful raw-score rankings. The
existing global-share and per-group thresholding logic can choose a different final
decision boundary after prediction. That post-processing is deliberately separate from
tree training.

### Why the two defining tree choices matter

- **Histograms** make a very wide feature matrix manageable. Split search depends on the
  number of bins rather than the number of distinct TF-IDF and text-feature values.
- **Leaf-wise growth** spends the limited leaf budget on the part of the tree with the
  largest available improvement. This can fit useful interactions with fewer leaves,
  although it also makes `num_leaves`, `max_depth`, and minimum-child constraints
  important safeguards against overfitting.
- **Second-order boosting** uses both gradients and Hessians. The model is therefore not
  merely fitting hard misclassification labels; it is making regularized Newton-like
  corrections to its current probability estimates.

## What is not included

This is a deliberately restricted model, not a reimplementation of the entire LightGBM
library. The omissions below keep the work achievable, auditable, and relevant to the
project's binary sparse-text problem.

| Not included | Consequence and rationale |
|---|---|
| Exact LightGBM bin construction and split tie behavior | The first binner uses deterministic frequency-weighted quantile boundaries rather than reproducing LightGBM's internal `GreedyFindBin` rules. Split ties also use this project's explicit stable ordering. Trees and raw scores may differ even with the same visible hyperparameters. Exact bin parity is a later fidelity task. |
| Exact `min_child_samples` behavior | This implementation uses stored integer row counts. LightGBM estimates this constraint from Hessian statistics, so weighted training can accept or reject some different splits. Exact counts are simpler and easier to audit. |
| LightGBM random-number streams | NumPy's local generator does not reproduce LightGBM's feature and bagging RNGs. Seeded fits are deterministic within this implementation, but sampled rows, sampled features, and resulting trees need not match LightGBM. |
| Exclusive Feature Bundling (EFB) | Mutually exclusive sparse features are not packed into shared feature bundles. This does not remove their predictive information, but training can use more memory and time on the 40,385-feature representation. Add EFB only if profiling justifies the complexity. |
| Gradient-based One-Side Sampling (GOSS) | Every selected training row contributes to histogram construction. The implementation may be slower, but this matches the project's current ordinary `gbdt` reference more closely than adding GOSS would. |
| DART and other boosting modes | Trees are always added through standard gradient boosting; there is no tree dropout or random-forest boosting mode. |
| Native categorical splits | All columns must already be numeric. Categories are not reordered or partitioned using LightGBM's categorical algorithm. The current project features are numeric, so this is not a project limitation. |
| Missing-value learning | Version one rejects NaN and infinite input instead of learning a default missing-value direction per split. An implicit sparse zero remains an ordinary numeric zero, not a missing value. |
| Multiclass, regression, and ranking objectives | Only binary labels and binary log loss are supported. There are no softmax, regression-loss, LambdaRank, or NDCG-specific paths. |
| Monotonic constraints, interaction constraints, and linear leaves | Every leaf outputs one unconstrained constant. Advanced structural and domain constraints are outside the project requirements. |
| Parallel CPU, GPU, and distributed training | Histogram construction and tree growth are deterministic and serial at the Python orchestration level. NumPy and SciPy may use compiled numerical kernels, but there is no custom threading, accelerator, or multi-machine implementation. |
| Full histogram-cache manager initially | The correctness oracle constructs both child histograms directly. Histogram subtraction is required before project-scale CV; bounded caching remains a later measured optimization. Training will still be slower than LightGBM. |
| Early stopping and training callbacks | The initial API fits exactly `n_estimators` trees. Cross-validation must select that value externally. Early stopping can be added after the core boosting path is stable. |
| Probability calibration and project-specific thresholds | `predict_proba` returns the sigmoid of the ensemble score, but class weighting can distort probability calibration. Share matching and per-group thresholds remain separate project logic. |
| Complete sklearn and LightGBM APIs | The estimator supplies only the cloning, fitting, and prediction protocol needed by this repository. It does not reproduce every metadata method, callback, alias, metric, plotting helper, or model-dump format. |
| LightGBM model-file compatibility | Models from this implementation cannot be loaded by LightGBM, and LightGBM models cannot be loaded here. Persistence will use this project's own Python representation. |
| Bitwise or tree-for-tree equality | Different binning, floating-point accumulation order, and unsupported engine details prevent exact identity. Success is defined by verified invariants, deterministic local behavior, prediction agreement, and paired Macro-F1 performance. |

These exclusions should be stated in the report when the model is described as
"LightGBM-like." They distinguish algorithmic similarity from a claim of library-level
compatibility.

## Public API

The user-facing class is `LiteLightGBM`. It intentionally follows the parameter
names already used in this repository's `LGBMClassifier` experiments. It must expose:

- `fit(X, y, sample_weight=None)`;
- `predict_raw(X)` for additive logits;
- `decision_function(X)` as a scikit-learn-compatible alias for `predict_raw(X)`;
- `predict_proba(X)` with columns `[P(y=0), P(y=1)]`;
- `predict(X)` using class `1` only when its probability is strictly above `0.5`;
- `get_params()` and `set_params()` for the repository's sklearn-based CV utilities.

The module must not import from or inherit from scikit-learn. It implements `get_params`,
`set_params`, fitted-state checks, and prediction validation locally. A small
`__sklearn_tags__` method may return standard-library objects containing
`estimator_type="classifier"` and `input_tags.pairwise=False`; this is optional metadata
for the repository's external CV utilities, not a scikit-learn dependency or part of
training. The estimator must pass a `cross_val_score` smoke test as soon as minimal `fit`
and `predict` implementations exist.

### Module organization and import contract

The only supported user-facing import location is `src.lite_lightgbm`:

```python
from src.lite_lightgbm import LiteLightGBM
```

The internal layout is:

| File | Responsibility |
|---|---|
| `lite_lightgbm.py` | Public façade, estimator validation, boosting loop, and prediction API |
| `lite_lightgbm_dep/core.py` | Shared configuration, type aliases, constants, and numerical helpers |
| `lite_lightgbm_dep/binning.py` | Bin containers, mapper fitting, and dense/sparse bin transformation |
| `lite_lightgbm_dep/tree.py` | Histograms, split search, tree growth, partitioning, and tree traversal |

All names documented in `lite_lightgbm_docs.md` remain importable from
`src.lite_lightgbm` after extraction. The private modules must not be imported directly
by project code or notebooks. Their names and contents are implementation details that
later optimizations may change.

Dependencies must be acyclic: core has no sibling dependency, binning depends on core,
tree depends on core and binning, and the public façade depends on all three. Private
modules must never import the façade. See `lite_lightgbm_refac.md` for the exact symbol
map, migration stages, persistence requirements, and verification matrix.

The project's predicted-share and per-group thresholding remain downstream operations.
They do not belong inside this estimator.

After fitting, the estimator should own these learned attributes:

- `classes_`: exactly `np.array([0, 1])`;
- `n_features_in_`: number of input columns;
- `mapper_`: fitted `BinMapper`;
- `trees_`: ordered list of `DecisionTree` objects;
- `init_score_`: scalar initial raw logit;
- `learning_rate_`: the learning rate captured by `fit`, so later parameter mutation does
  not change an already-fitted model's predictions;
- `feature_importances_`: split counts per original feature.

### Input and parameter validation

`fit` accepts only a non-empty, two-dimensional numeric matrix and a one-dimensional
target of matching length containing both labels `0` and `1`. It rejects non-finite
feature values, non-binary labels, and single-class targets. Sparse inputs must validate
their stored data without densifying.

Optional `sample_weight` must be one-dimensional, finite, non-negative, length-matched,
and have a positive sum. Explicit class-weight dictionaries must contain exactly keys
`0` and `1` with finite, non-negative values, and their effective combined weights must
also have a positive sum.

Validate configuration before allocating binned data:

- `n_estimators`, `learning_rate`, `num_leaves`, `max_bin`, and `min_data_in_bin` are
  positive, with `num_leaves >= 2` and `max_bin >= 2`;
- child constraints, regularization values, and `subsample_freq` are non-negative;
- `0 < colsample_bytree <= 1` and `0 < subsample <= 1`;
- `max_depth <= 0` means unlimited depth; otherwise it is a positive integer;
- `class_weight` is `None`, `"balanced"`, or a valid explicit dictionary.

Prediction rejects calls before fitting, non-finite values, non-matrix input, and a
feature count different from `n_features_in_`.

## Numerical definition

For raw score `F`, probability `p`, label `y`, and effective row weight `w`:

```text
p = sigmoid(F)
gradient = w * (p - y)
hessian = w * p * (1 - p)
```

The initial raw score is the log-odds of the weighted positive-class frequency:

```text
positive_rate = sum(w * y) / sum(w)
init_score = log(positive_rate / (1 - positive_rate))
```

Define one module-level `EPSILON = 1e-15`. Clip `positive_rate` to
`[EPSILON, 1 - EPSILON]` before taking the logarithm. Do not clip ordinary probabilities
before gradient calculation, because doing so changes the binary-log-loss derivatives.

For gradient sum `G`, Hessian sum `H`, L1 penalty `alpha`, and L2 penalty `lambda`:

```text
soft_threshold(G, alpha) = sign(G) * max(abs(G) - alpha, 0)
leaf_value = -soft_threshold(G, alpha) / (H + lambda)
leaf_score = soft_threshold(G, alpha)^2 / (H + lambda)
```

These formulae apply when `H + lambda > EPSILON`. If the denominator is at most
`EPSILON`, return a zero leaf value and zero leaf score. This prevents a numerically
saturated leaf with no usable curvature from creating an infinite correction or gain.

The unpenalized improvement of a candidate split is:

```text
gain = left_leaf_score + right_leaf_score - parent_leaf_score
```

A split is valid only when all of these hold:

- each child meets `min_child_samples`;
- each child Hessian sum meets `min_child_weight`;
- the resulting gain is strictly greater than `min_split_gain`;
- the current leaf has not reached `max_depth` when `max_depth > 0`;
- the tree has not reached `num_leaves`.

After fitting tree `t`, update every training row, including rows omitted by bagging,
with:

```text
raw_score += learning_rate * tree_output
```

Store the fit-time learning rate in `learning_rate_`. Tree node values remain unscaled;
both training-score updates and `predict_raw` multiply them by `learning_rate_` exactly
once. Calling `set_params` after fitting must therefore not alter existing predictions.

## Sparse bin representation

The 40,385-feature project matrix must never be densified. `BinnedDataset` therefore
keeps both CSR and CSC views of the same logical quantized matrix:

- CSR supports selecting or routing all feature entries belonging to a group of rows;
- CSC supports inspecting one feature across the rows in a leaf;
- `mapper.default_bins[j]` is the bin for an implicit sparse zero in feature `j`;
- sparse storage contains only values whose bin differs from the default bin;
- stored sparse values are encoded as `bin_id + 1` so SciPy's implicit zero cannot be
  confused with actual bin zero; and
- encoded sparse-bin values use the smallest safe unsigned dtype: `uint8` through 255,
  `uint16` through 65,535, `uint32` through 4,294,967,295, otherwise `uint64`.
  Mapper metadata remain signed `int64`; SciPy structural indices are unchanged and
  may be signed `int32` or `int64`.

Decode stored values only after widening them to signed `int64`, then subtract one. This
ensures a malformed stored zero is rejected instead of underflowing in an unsigned dtype.

The private `_encoded_bin_dtype(n_bins)` helper receives the already validated and
normalized signed `int64` bin-count array and only selects a dtype; metadata validation
does not happen inside this pure selector.

**OPT2 — compact bin storage.** This is the unsigned encoded-bin representation above.
It reduces bin-data memory and bandwidth without changing the logical bins, mapper,
tree structures, or predictions.

`transform_bins` uses the selected encoded dtype for a dense temporary bin matrix. For
one-based stored values, selected bin IDs are widened to signed `int64` before adding one
and checking the dtype bound, then narrowed to the selected unsigned dtype. CSC and CSR
canonicalization completes before the final checks that both sparse data arrays retain
that dtype.

The first binner is deterministic and frequency-weighted. "Weight" here means the
integer occurrence count of each distinct feature value, including implicit sparse
zeros; training, class, and sample weights never influence bin construction.

Canonicalize sparse input before inspecting it: convert a copy to CSC, sum duplicate
coordinates, eliminate explicitly stored zeros, and sort indices. This makes a sparse
matrix with duplicate entries behave like its logically equivalent dense matrix.

For each feature:

1. collect ascending distinct finite values `v[i]` and their integer occurrence counts
   `c[i]`, adding the multiplicity of implicit zeros to the count for numeric zero;
2. choose the largest candidate bin count
   `B <= min(max_bin, n_distinct, max(1, n_samples // min_data_in_bin))`;
3. for desired cumulative ranks `k * n_samples / B`, choose boundaries between adjacent
   distinct values in ascending order. Choose the feasible boundary whose cumulative
   count is closest to the desired rank; require every completed bin to contain at least
   `min_data_in_bin` rows and enough rows to remain for the unfinished bins. Break equal
   distances toward the lower-valued boundary;
4. if all `B - 1` feasible boundaries cannot be selected, decrement `B` and restart.
   `B = 1` is always valid;
5. store the largest training value in every non-final bin as its cut point. Map any
   value `x` with `np.searchsorted(cut_points, x, side="left")`;
6. map numeric zero through that same expression and record its result as the default
   bin.

The candidate-boundary scan in this procedure is NumPy-vectorized; it preserves the
same exact frequency counts, feasible ranges, desired-rank distances, and stable
lower-boundary tie-break as the reference algorithm. Dense, CSR, CSC, and equivalent
non-canonical sparse inputs therefore retain identical deterministic mappers.

**OPT1 — vectorized bin-boundary selection.** This is the optimization in the preceding
candidate-boundary scan: it replaces the per-candidate Python search with NumPy
cumulative-count, slice, mask, and stable-`argmin` operations while retaining the
sequential outer boundary loop and identical mapper output.

Constant features consequently have one bin and no cut points. Unseen prediction values
below, between, or above training values are handled by the same `searchsorted` rule.

This will be LightGBM-like, not initially identical to LightGBM's `GreedyFindBin`.
Exact bin-boundary parity is a later fidelity task and must not block the first working
booster.

Do not treat an implicit sparse zero as missing. Version one rejects non-finite input;
missing-value routing can be designed separately if the project ever needs it.

## Histogram contract

`BinMapper.bin_offsets` gives each feature a segment in a flattened histogram. For
feature `j`, its bins occupy:

```text
bin_offsets[j] : bin_offsets[j + 1]
```

`build_histogram` returns three arrays over that flattened layout: gradient sums,
Hessian sums, and row counts. It must include implicit default-bin rows. A practical
construction is:

1. aggregate explicitly stored non-default entries;
2. calculate each feature's represented totals;
3. add `leaf_total - represented_total` to its default bin.

Never iterate over individual samples in Python. Use sparse indexing plus NumPy
aggregation (`bincount`, indexed addition, or equivalent vectorized operations). A
short loop over features is acceptable initially and should only be replaced after
profiling proves it is material.

For each feature, prefix sums across ascending bins provide the left-child statistics;
right-child statistics are parent totals minus the left values. Evaluate boundaries
between bins, not after the final bin.

Tie-breaking must be deterministic. Use this order unless parity experiments prove a
different order is necessary:

1. larger gain;
2. lower feature index;
3. lower threshold-bin index.

`default_left` is derived from whether that feature's implicit-zero bin is at or below
the selected threshold. It is not an independently optimized missing-value direction in
version one.

## Leaf-wise tree growth

`fit_tree` maintains a max-priority queue of splittable leaves. Python's `heapq` is a
min-heap, so queue entries should begin with negative gain and include stable integer
tie-break fields.

For each tree:

1. create the root with the selected bagged rows;
2. build its histogram and find its best valid split;
3. push the root candidate onto the queue;
4. pop the globally best candidate;
5. partition that leaf's rows and turn it into an internal node;
6. calculate each child's optimal leaf value;
7. find and enqueue the best split for each eligible child;
8. stop when the queue is empty or the tree reaches `num_leaves`.

Persist row-index arrays only while building a tree. They do not belong in the trained
`DecisionTree` because prediction traverses nodes from input values.

Implement both child histograms directly in the first correct version. Histogram
subtraction is a performance milestone: build the smaller child's histogram and derive
the larger child as `parent - smaller`. Add it only after direct construction has a
correctness test that the subtraction path can be compared against.

## Weighting and sampling

Effective training weights are the product of optional `sample_weight` and class weight.
For `class_weight="balanced"`, class `c` receives:

```text
n_samples / (2 * count_of_class_c)
```

Use a local `np.random.Generator` seeded from `random_state`; never mutate NumPy's global
random state.

- Sample a tree's features once when `colsample_bytree < 1`. Select
  `max(1, floor(colsample_bytree * n_features))` without replacement, then sort them.
- Enable row bagging only when `subsample < 1` and `subsample_freq > 0`.
- On zero-based boosting iteration `t`, draw a new row set when
  `t % subsample_freq == 0`; otherwise reuse the preceding set. Select
  `max(1, floor(subsample * n_samples))` rows without replacement, then sort them.

Sampling must not alter the order of features in model metadata or the prediction path.
These rules guarantee local reproducibility but intentionally do not reproduce
LightGBM's RNG stream or its exact sampled subsets.

## Implementation milestones

### 1. Scalar mathematics

Implement and test `sigmoid`, `soft_threshold`, and
`binary_gradients_hessians` first.

Required checks:

- sigmoid stays finite for logits near `-1000` and `1000`;
- analytic gradients and Hessians agree with finite differences;
- Hessians are non-negative for every finite tested logit and positive for positive
  weights over a moderate unsaturated range such as `[-30, 30]`;
- a saturated, zero-curvature leaf produces finite zero value and gain;
- zero L1 reproduces the unthresholded formula.

### 2. Dense binning and one split

Implement the binner on a tiny dense matrix, histogram construction, prefix statistics,
gain calculation, split constraints, and deterministic tie-breaking. Compare the chosen
split and leaf values with a hand-calculated fixture.

### 3. One complete tree

Implement `partition_rows`, leaf-wise queue growth, and tree prediction. Test one-stump,
no-valid-split, repeated-value, constant-feature, depth-limited, and leaf-limited cases.

### 4. Boosting loop

Implement class weights, initial score, sequential gradients/Hessians, shrinkage, raw
prediction, probability prediction, and label prediction. Assert that binary log loss
decreases on a learnable toy dataset.

### 5. Sparse parity

Implement sparse binning and histograms. The same logical matrix represented densely,
as CSR, and as CSC must produce the same bins, tree structure, raw scores, and labels.
Include non-canonical sparse fixtures containing unsorted indices, duplicate coordinates,
and explicitly stored zeros.

### 6. Minimum viable optimization

Profile deterministic subsets after sparse parity is correct. The direct-child,
per-feature implementation is only a correctness oracle; do not start project-scale CV
with it. Add and test these optimizations in order:

1. **OPT2 — compact bin dtypes** (complete: encoded sparse-bin values use the smallest safe unsigned dtype);
2. feature pre-filtering;
3. vectorized flattened histogram aggregation;
4. histogram subtraction, checked against direct construction;
5. bounded histogram caching if profiling still justifies it.

The extraction-only module refactor in `lite_lightgbm_refac.md` is complete.
It preserved the `src.lite_lightgbm` façade and exact behavior; OPT3 behavior
remains outside that extraction.

The first four are prerequisites for the 5,000-feature benchmark. Feature bundling
remains optional and should be added only if the 40,385-feature profile shows that its
complexity is necessary. GOSS is not part of this sequence because the project's current
reference estimator uses ordinary `gbdt` boosting.

### 7. Project-scale correctness

Fit the supplied 5,000-feature representation before using the 40,385-feature chosen
representation. Run the repository's locked folds and record paired Macro-F1 and
per-sample output differences against LightGBM. Establish a runtime and peak-memory
budget from a single fold before launching the complete locked CV.

## Differential testing against LightGBM

LightGBM 4.7.0 is installed in the project environment. It may be used as a development
oracle but must not be imported by the completed implementation.

Begin with a restricted reference configuration:

```text
objective = binary
boosting_type = gbdt
device = cpu
num_threads = 1
deterministic = true
force_col_wise = true
enable_bundle = false
feature_pre_filter = false
boost_from_average = true
use_missing = false
zero_as_missing = false
feature_fraction = 1.0
bagging_fraction = 1.0
bagging_freq = 0
```

Set `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, child constraints,
regularization, `max_bin`, and `min_data_in_bin` identically on both estimators. Keep
LightGBM's `sigmoid=1`, `max_delta_step=0`, `path_smooth=0`, and
`use_quantized_grad=false`. Because the project has fewer rows than LightGBM's default
bin-construction sample limit, the complete training fold is available for binning.

Because the initial binner is intentionally approximate, compare invariants and behavior
before demanding identical trees:

- identical initial weighted logit;
- matching gradients and Hessians for supplied raw scores;
- matching leaf-value formula for fixed sufficient statistics;
- raw-score correlation;
- prediction agreement;
- paired fold-level Macro-F1 differences;
- deterministic repetition within this implementation.

Later, fixed-bin synthetic datasets can test exact split and tree parity without bin
construction being a confounder.

Use two separate project comparisons:

1. the restricted deterministic reference above, which isolates algorithmic fidelity;
2. the repository's actual tuned `LGBMClassifier` configuration, which retains its
   normal engine defaults and measures replacement quality.

For each locked fold, save raw scores from both models and report Pearson correlation,
Spearman correlation, probability mean absolute error, label agreement, and predicted
positive-share difference. The initial similarity targets are Spearman correlation of at
least `0.95` and label agreement of at least `0.90` against the restricted reference.
Treat these as gates alongside Macro-F1, not as claims of tree equality.

## Acceptance gates

The implementation is ready for project use only when:

- every stub in `lite_lightgbm.py` has a focused test;
- dense and sparse forms are behaviorally identical;
- no training path densifies the project matrix;
- repeated seeded fits produce identical trees and predictions;
- all learned arrays are finite;
- model persistence round-trips predictions exactly;
- an external scikit-learn clone, classifier-tag check, and `cross_val_score` smoke test
  pass without the estimator module importing scikit-learn;
- restricted-reference raw-score Spearman correlation is at least 0.95 and label
  agreement is at least 0.90 on the supplied-feature locked out-of-fold predictions;
- the supplied-feature locked CV score is within 0.01 Macro F1 of the restricted
  LightGBM reference;
- the chosen representation is evaluated with paired grouped-fold differences;
- peak memory and full-fit runtime are measured and reported;
- the final estimator imports when both `lightgbm` and `sklearn` imports are blocked.

Do not loosen a correctness gate to improve the leaderboard result. If fidelity and
predictive quality diverge, report both measurements and keep the simpler, reproducible
implementation as the scientific result.

## Primary references

- [LightGBM features](https://lightgbm.readthedocs.io/en/stable/Features.html)
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
- [Ke et al., LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
