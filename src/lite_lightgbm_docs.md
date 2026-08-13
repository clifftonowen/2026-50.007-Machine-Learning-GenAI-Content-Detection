# LiteLightGBM module documentation

## Overview

`src.lite_lightgbm` is the stable public module for a small, auditable, LightGBM-like
classifier for this project's binary sparse-text problem. It retains the central ideas
behind LightGBM:

- second-order gradient boosting for binary log loss;
- frequency-weighted numeric feature bins;
- gradient, Hessian, and row-count histograms;
- leaf-wise, best-first tree growth;
- L1 and L2 regularized leaf corrections;
- row and feature subsampling; and
- dense NumPy and sparse SciPy input.

The module depends only on NumPy, SciPy, and the Python standard library. It does not
import LightGBM or scikit-learn. Its small parameter and prediction interface is
duck-typed so external evaluation utilities can clone and score the estimator without
becoming implementation dependencies.

The detailed algorithm, implementation milestones, validation rules, and acceptance
gates live in [`lite_lightgbm.md`](lite_lightgbm.md). This file is the API and usage
reference.

## Current status

The module is complete for its documented scope. Its data containers, estimator
constructor, parameter helpers, metadata hook, numerical routines, binning, training,
and prediction paths are implemented against the documented invariants.
Encoded sparse bin values use the smallest safe unsigned dtype (`uint8`, `uint16`,
`uint32`, or `uint64`); mapper metadata remain `int64`, while SciPy structural
indices are unchanged and may be signed `int32` or `int64`.

The extraction-only internal refactor is complete. It changed file ownership,
not estimator behavior or public usage: the implementation definitions now
reside in the `lite_lightgbm_dep` implementation package described below, while
`src.lite_lightgbm` remains the stable façade.

Optimization 3 is complete. After binning, fitting records count-splittable
original feature IDs in `active_features_`. Each tree then allocates histograms
only for its sampled active features, while stored split IDs and feature
importances continue to use the original input-column numbering.

Optimization 4 is complete. Estimator fitting and prediction validate each
internally produced binned dataset once per operation, then reuse its normalized
storage across all trees. Tree growth and traversal use private trusted kernels;
the public development helpers remain fully validating checked entry points.

Optimization 5 is implemented and verified on the supplied 5,000-feature
representation. Histogram construction now processes bounded CSR row blocks,
maps stored bins into the tree-local flattened histogram layout, and aggregates
gradients, Hessians, and exact counts with `np.bincount`. Implicit sparse
defaults are added from each leaf's totals afterward. The former per-feature
CSC routine remains a private reference oracle for parity tests. Its specified
40,385-feature root-memory benchmark remains pending; measured evidence and the
two reported floating-point near-tie split changes are recorded in
`lite_lightgbm_OPT5.md`.

Optimization 6 is algorithmically complete and has direct-oracle parity on a
fixed non-tie fixture: the direct and subtraction paths preserve tree topology
and match predictions within a tight numerical tolerance. During leaf-wise
growth, `fit_tree` retains a histogram only for each queued live leaf, directly
builds the smaller child (ties go left), and derives the other child by
subtracting aligned histogram statistics. Empty bins have exactly zero
statistics; subtraction validates raw residuals against a scale-aware
`128 * eps` bound before that normalization, so material inconsistencies still
fail. Builder-created histograms carry optional finite absolute accumulation
scales to account for default-bin error from large or cancellation-heavy leaf
totals; derived histograms conservatively propagate both operand scales, while
legacy hand-built four-field histograms retain the per-bin fallback. Construction histograms are released
before fitting returns and are not part of the trained model. A private
direct-both-child tree path remains available for parity checks without changing
the public API.

The following OPT6 profile measurements—not OPT7 cache benchmark results—were
collected on the supplied 20,000 by 5,000, 31-leaf representation. Three
subtraction runs took 60.58, 58.93, and 58.60 seconds (median 58.93), while direct runs took
61.17, 60.85, and 59.75 seconds (median 60.85, 1.03x). Histogram builds fell
from 59 to 30. Each local-layout histogram used about 9.03 MB, with an
observed peak of 17 live histograms (about 153 MB). The unbounded mapping has a
worst case of about 271 MB at 31 leaves, with an estimated 2.3 GB at 255 leaves;
OPT7 now bounds retained queued-leaf histogram buffers with a private per-tree
LRU cache; its configured limit applies to retained cache buffers, not every
transient child-construction allocation.

Synthetic near-tie cases differed in topology in 116 of 300 cases and had six
label flips, while the dedicated 128-leaf and 1,200-leaf tests showed no
accumulated histogram-statistic drift.

An independently reproduced first-tree diagnostic on the supplied 20,000 x 5,000
CSR representation (`num_leaves=149`, `colsample_bytree=0.5198042692950159`,
`min_child_samples=7`, `reg_alpha=0.34104824737458306`,
`reg_lambda=0.13010318597055903`, `learning_rate=0.034494`, balanced weights,
`random_state=42`) completed all 297 nodes and 149 leaves with 138 subtraction
operations. Exactly one zero-count gradient residual measured `117.75 * eps`
at the unit scale floor and was rescued by the `128 * eps` guard. This is
measured first-tree evidence, not a full-ensemble runtime estimate.

Optimization 7 is complete. A private tree-local `OrderedDict` LRU cache now
bounds retained queued-leaf histogram buffers by their NumPy array bytes. A
cache hit uses the OPT6 smaller-child direct build plus sibling subtraction; a
miss, eviction, oversized entry, or zero cache budget builds both children
directly. Cache residency never affects split eligibility, and the cache is
discarded at the end of each tree. Focused zero, one-entry, and generous-budget
checks on a fixed non-tie tree preserved topology and matched predictions within
a tight numerical tolerance. The 256 MiB default is
currently a provisional cap, not a target-scale benchmark selection. Repeated
fits with one fixed cache budget are deterministic, but changing the private
budget can alter tie-prone trees because direct and subtraction histogram
paths use different floating-point accumulation orders.

Optimization 8 implementation and correctness are complete. Its trusted
`_find_best_split_validated(...)` kernel batches the sole threshold of all
two-bin features and evaluates wider segments with NumPy vectors; a private
`_find_best_split_scalar_validated(...)` helper retains the pre-OPT8 threshold
loop as a correctness oracle for focused parity checks. Direct-kernel
low-bin/multi-bin benchmark evidence is recorded in
`lite_lightgbm_OPT8.md`; full-tree and target-scale timing remain unmeasured.

A persistent unittest regression suite using the standard library, NumPy, and
SciPy is available at `tests/test_lite_lightgbm_refactor.py`; run it with
`uv run python -m unittest discover -s tests -p "test_lite_lightgbm*.py"` from
the project root. Runtime import checks require the project virtual environment
and its dependencies. The focused split-search regression is
`tests/test_lite_lightgbm_split_search.py`; no full-tree split-search benchmark
is claimed here.

## Import and module layout

Continue importing the estimator and documented development helpers from the public
façade:

```python
from src.lite_lightgbm import LiteLightGBM, fit_bin_mapper, fit_tree
```

The implementation is organized into these files:

| File | Role |
|---|---|
| `lite_lightgbm.py` | Stable façade and `LiteLightGBM` estimator |
| `lite_lightgbm_dep/core.py` | Configuration, shared types, constants, and numerical helpers |
| `lite_lightgbm_dep/binning.py` | Bin mapper and encoded sparse transformation |
| `lite_lightgbm_dep/tree.py` | Histograms, splits, tree construction, and traversal |
| `lite_lightgbm_dep/lightgbm_import.py` | Dependency-free conversion of supported official model dumps |

The dependency modules are implementation details. Application code and notebooks
must not import them directly. All classes and functions documented below remain re-exported
from `src.lite_lightgbm`, so model construction, fitting, prediction, and development
helper usage do not change.

The completed extraction procedure and compatibility gates are in
[`lite_lightgbm_refac.md`](lite_lightgbm_refac.md).

## Supported scope

- Labels must contain both integer classes `0` and `1`.
- Features must be finite and numeric.
- Input may be a two-dimensional NumPy array or SciPy sparse matrix.
- Sparse input must remain sparse throughout training and prediction.
- The only objective is binary log loss.
- `predict_proba` returns columns in `[P(y=0), P(y=1)]` order.
- `predict` assigns class `1` only when `P(y=1) > 0.5`; an exact tie becomes class `0`.

The initial version does not include native categorical splits, learned missing-value
directions, multiclass objectives, regression, ranking, GOSS, EFB, DART, early stopping,
or parallel training. It can import the JSON-compatible result of official
`Booster.dump_model()` for the restricted numerical binary scope described below; it does
not parse LightGBM's native text model format.

## Storing and importing official LightGBM weights

A fitted LightGBM model is a tree ensemble, not a single weight vector. The portable
artifact must retain every split feature, numerical threshold, child relationship, leaf
value, and objective field. Store the dictionary returned by `Booster.dump_model()` as
JSON during the official-training/export step:

```python
import json
from pathlib import Path

model_dump = official_model.booster_.dump_model()
Path("models/lightgbm_dump.json").write_text(
    json.dumps(model_dump), encoding="utf-8"
)
```

Runtime prediction then needs neither LightGBM nor scikit-learn:

```python
from src.lite_lightgbm import LiteLightGBM

model = LiteLightGBM.from_lightgbm_json("models/lightgbm_dump.json")
raw_scores = model.predict_raw(X_test)
probabilities = model.predict_proba(X_test)
labels = model.predict(X_test)
```

`from_lightgbm_dump(model_dump)` accepts the in-memory dictionary directly. Both entry
points convert the official numerical thresholds into one global `BinMapper` and the
official trees into local `DecisionTree` objects. Dumped LightGBM leaf values already
contain shrinkage, and the first tree contains the model's initial binary bias, so an
imported predictor intentionally stores `init_score_ = 0` and `learning_rate_ = 1`.

Import is limited to binary models with one numerical `<=` tree per boosting iteration,
`sigmoid=1`, ordinary sparse-zero semantics, and non-averaged constant leaves. It rejects
categorical splits, linear trees, multiclass models, `zero_as_missing`, averaged outputs,
and other incompatible metadata. LiteLightGBM prediction still rejects NaN and infinite
input. Feature columns must be supplied in exactly the same order used by official
training.

JSON is preferred over pickle for this bridge: it is portable, contains data rather than
executable Python objects, and can be loaded without the official LightGBM package.

## Basic construction

Construction and parameter inspection are available independently of fitting:

```python
from src.lite_lightgbm import LiteLightGBM

model = LiteLightGBM(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
)

assert model.get_params()["n_estimators"] == 200
model.set_params(reg_lambda=0.5)
```

The fitting and prediction flow is:

```python
model.fit(X_train, y_train)
raw_scores = model.predict_raw(X_test)
probabilities = model.predict_proba(X_test)
labels = model.predict(X_test)
```

## `LiteLightGBM`

### Constructor parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `n_estimators` | `100` | Number of sequential correction trees. |
| `learning_rate` | `0.1` | Positive shrinkage applied once to each tree. |
| `num_leaves` | `31` | Maximum terminal leaves in one tree; at least 2. |
| `max_depth` | `-1` | Maximum root-to-leaf depth; values at or below 0 mean unlimited. |
| `min_child_samples` | `20` | Minimum exact row count in both children. |
| `min_child_weight` | `1e-3` | Minimum summed Hessian in both children. |
| `min_split_gain` | `0.0` | Gain that a candidate must strictly exceed. |
| `max_bin` | `255` | Maximum bins learned for one feature. |
| `min_data_in_bin` | `3` | Target minimum row count represented by one bin. |
| `reg_alpha` | `0.0` | L1 regularization on leaf corrections. |
| `reg_lambda` | `0.0` | L2 regularization on leaf corrections. |
| `colsample_bytree` | `1.0` | Fraction of features sampled once per tree. |
| `subsample` | `1.0` | Fraction of rows selected when bagging is enabled. |
| `subsample_freq` | `0` | Number of iterations reusing a row sample; 0 disables bagging. |
| `class_weight` | `None` | `None`, `"balanced"`, or weights for keys `0` and `1`. |
| `random_state` | `None` | Seed for a local `numpy.random.Generator`. |

For `class_weight="balanced"`, class `c` receives
`n_samples / (2 * count_of_class_c)`. Class and sample weights are multiplied together.

### Learned attributes

These attributes exist after a successful `fit` or official-model import:

| Attribute | Meaning |
|---|---|
| `classes_` | Exactly `np.array([0, 1])`. |
| `n_features_in_` | Number of fitted input columns. |
| `mapper_` | Learned `BinMapper`. |
| `trees_` | Ordered list of fitted `DecisionTree` objects. |
| `init_score_` | Weighted positive-rate log-odds. |
| `learning_rate_` | Fit-time learning rate used by prediction. |
| `feature_importances_` | Split counts per original feature. |
| `active_features_` | Sorted `int64` original feature IDs that can satisfy `min_child_samples` on the full training set. |

Capturing `learning_rate_` ensures that calling `set_params` after fitting does not
silently change the predictions of an already-fitted model.

### Methods

#### `from_lightgbm_dump(model_dump)`

Constructs a fitted predictor from the in-memory mapping returned by official
`Booster.dump_model()`. No LightGBM import occurs in this module.

#### `from_lightgbm_json(path)`

Loads a JSON serialization of `Booster.dump_model()` and returns the same converted
predictor. See the compatibility restrictions above.

#### `fit(X, y, sample_weight=None)`

Fits the bin mapper and `n_estimators` sequential leaf-wise trees.

- `X`: finite dense or sparse numeric matrix of shape `(n_samples, n_features)`.
- `y`: one-dimensional binary target containing both labels.
- `sample_weight`: optional finite, non-negative vector with positive total weight.
- Returns the fitted estimator.

Fitting rejects invalid dimensions, non-finite values, invalid labels or weights, and
invalid hyperparameters before allocating the binned dataset.

#### `predict_raw(X)`

Returns an array of additive logits with shape `(n_samples,)`:

```text
raw_scores = init_score_
for each tree:
    raw_scores += learning_rate_ * tree_output
```

Shrinkage is applied before each addition, matching training and avoiding
overflow in an unscaled intermediate tree sum.

Prediction requires a fitted estimator, finite numeric input, and exactly
`n_features_in_` columns.

#### `decision_function(X)`

An exact alias for `predict_raw(X)`. It introduces no scikit-learn dependency.

#### `predict_proba(X)`

Applies the stable sigmoid to raw scores and returns an `(n_samples, 2)` array with
columns `[P(y=0), P(y=1)]`. Class or sample weighting can make these probabilities poorly
calibrated even when their ranking is useful.

#### `predict(X)`

Returns integer labels. Class `1` requires a probability strictly greater than `0.5`.

#### `score(X, y, sample_weight=None)`

Returns mean classification accuracy, optionally weighted by finite,
non-negative sample weights with a positive total. This dependency-free method
supports scikit-learn's default classifier scoring without importing sklearn.

#### `get_params(deep=True)`

Returns every constructor parameter. `deep` is accepted for compatibility with external
cloning tools but has no effect because this estimator contains no nested estimators.

#### `set_params(**params)`

Updates known constructor parameters in place and returns the estimator. Unknown names
raise `ValueError`. Values are validated by `fit`, not by `set_params`.

#### `__sklearn_tags__()`

Returns locally constructed standard-library metadata identifying a non-pairwise
classifier. This optional hook lets the project's external CV tooling inspect the
estimator without the module importing or inheriting from scikit-learn.

## Data containers

### `LiteLightGBMConfig`

An immutable snapshot of all supported constructor parameters. `fit` uses it to pass one
consistent configuration through binning, histogram, split, and tree routines.

### `BinMapper`

Stores the mapping from original feature values to discrete bins:

- `cut_points[j]`: ascending upper bounds for every non-final bin of feature `j`;
- `default_bins[j]`: bin assigned to an implicit sparse zero;
- `n_bins[j]`: number of bins for feature `j`; and
- `bin_offsets`: exclusive prefix sum used by flattened histograms.

A value is mapped with:

```python
np.searchsorted(cut_points[j], value, side="left")
```

A constant feature has one bin and an empty cut-point array.

### `BinnedDataset`

Contains CSR and CSC views of the same quantized matrix:

- `csr` supports row-oriented histogram aggregation;
- `csc` supports per-feature routing and prediction; and
- `mapper` describes the encoding.

Only values whose bin differs from the feature's default bin are stored. A stored bin is
encoded as `bin_id + 1`, leaving SciPy's implicit zero unambiguous. The `shape` property
returns `(n_samples, n_features)`.
Checked tree entry points require both sparse views to encode identical
coordinates and bin values, not merely to be independently well formed.

The encoded values use the smallest safe unsigned dtype for the fitted feature-bin
layout. Consumers widen these values to signed `int64` before decoding them, so an
invalid stored zero is rejected rather than underflowing.

The private `_encoded_bin_dtype(n_bins)` helper receives an already validated and
normalized signed `int64` `n_bins` array and is a pure dtype selector. Validation and
normalization remain in `transform_bins`.

### `Histogram`

Contains three flattened arrays using `BinMapper.bin_offsets`:

- `gradient_sums`;
- `hessian_sums`; and
- exact `counts`;
- optional `gradient_scale` and `hessian_scale` absolute accumulation metadata
  used only to validate histogram-subtraction residuals.

Each selected feature has its own histogram segment, and every row contributes once to
that segment, including through the feature's implicit default bin.

During tree fitting, `HistogramLayout` supplies a compact tree-local equivalent of
this layout. It contains the sorted original feature IDs selected for the tree, their
bin counts and default bins, local prefix offsets, and an original-feature-to-local-slot
lookup. The layout is immutable and shared by every histogram in that tree. Empty
layouts have the single offset `[0]` and produce root-only trees.
If `active_features_` is empty, fitting intentionally completes with these
root-only trees and a constant predictor; inspect that learned attribute when
you need to distinguish this valid no-split outcome from a feature-rich fit.

### `SplitInfo`

Describes the best valid split for one leaf:

- gain, original feature index, and threshold bin;
- whether an implicit zero follows the left branch;
- exact left and right counts; and
- left and right gradient and Hessian sums.

`default_left` is sparse-zero routing metadata, not a learned missing-value direction.

### `TreeNode`

An array-backed tree node containing its depth, unscaled leaf value, split feature and
threshold, sparse-zero direction, child indices, derivative sums, sample count, and
accepted split gain. `is_leaf` is true when both child indices are absent.

### `DecisionTree`

Contains nodes in stable creation order, with the root at index zero, plus the sorted
original feature indices sampled for that tree. Leaf values are stored before learning-
rate shrinkage. Temporary row assignments used during construction are not persisted.

## Numerical helper functions

### `sigmoid(raw_scores)`

Computes a stable element-wise logistic sigmoid. Positive and negative inputs use
separate algebraic forms so logits near `-1000` and `1000` remain finite.

### `soft_threshold(values, reg_alpha)`

Applies the L1 operator:

```text
sign(values) * max(abs(values) - reg_alpha, 0)
```

### `binary_gradients_hessians(raw_scores, labels, sample_weight)`

For `p = sigmoid(raw_scores)`:

```text
gradient = sample_weight * (p - labels)
hessian = sample_weight * p * (1 - p)
```

Probabilities are not clipped before derivative calculation. Numerically saturated
finite logits may therefore have a zero Hessian.

## Binning, histogram, and tree functions

### `fit_bin_mapper(X, config)`

Learns deterministic, frequency-weighted cut points without densifying sparse input.
Occurrence counts determine quantiles; targets, class weights, and sample weights never
affect binning. Bin-boundary candidate scanning is NumPy-vectorized, but the exact
feasibility rules, desired-rank distances, stable lower-boundary tie-break, and sparse
zero semantics are preserved. Sparse matrices are canonicalized so equivalent dense,
CSR, and CSC representations learn the same deterministic mapper.

Direct calls require `config.max_bin` and `config.min_data_in_bin` to be finite,
real, integer-valued scalar numbers (at least `1`); booleans, fractions, complex or
non-numeric values, non-finite values, and non-scalar arrays are rejected.
Integer dtypes are converted directly to Python integers, preserving values above
the exact `float64` range; accepted floating inputs are checked for finite,
integral, exactly representable values before conversion.

### `transform_bins(X, mapper)`

Maps a matrix through a fitted `BinMapper` and returns encoded CSR and CSC views. Input
must have the same feature count used to fit the mapper. Dense input uses the selected
encoded dtype for its temporary bin matrix; one-based values are widened to signed
`int64` before adding one and checking bounds, then narrowed to that dtype. Sparse CSC
and CSR canonicalization runs to completion before both data dtypes are verified.

### `build_histogram(data, row_indices, feature_indices, gradients, hessians)`

Aggregates per-bin gradient sums, Hessian sums, and exact row counts for one leaf.
Implicit sparse rows are added to each selected feature's default bin. Construction must
not densify input or loop over individual samples in Python.
Direct calls require finite, row-aligned gradient and Hessian arrays.

### `find_best_split(...)`

Uses prefix sums to evaluate boundaries between adjacent bins. A valid split must meet
minimum child count, minimum child Hessian, gain, depth, and leaf-count constraints.
Exact gain ties prefer the lower feature index and then the lower threshold-bin index.
For a feature with `n_bins[j]` bins, threshold bins are boundaries in the inclusive
range `0 .. n_bins[j] - 2`; the terminal bin (`n_bins[j] - 1`) is not a split boundary.
Features with one bin therefore have no valid split threshold.

### `partition_rows(data, row_indices, split)`

Returns disjoint left and right row-index arrays. Bins at or below the threshold route
left; larger bins route right. Input ordering is preserved within both results. The
threshold validation follows the `find_best_split` range above and rejects terminal-bin
thresholds, including every threshold for a one-bin feature.

### `fit_tree(...)`

Fits one Newton-correction tree using a max-priority queue. It repeatedly splits the leaf
with the highest valid gain until the queue is empty or `num_leaves` is reached. Returned
leaf corrections are not yet multiplied by the learning rate.
Directly supplied `BinnedDataset` objects must contain canonical CSR and CSC storage with
valid sparse structure, identical logical values, and mapper-compatible encoded bins;
`transform_bins` produces this form.

### `predict_tree_raw(tree, data)`

Traverses one tree and returns the unscaled terminal-node correction for every row.
Direct calls validate the complete mapper, sparse storage, and tree graph. The
estimator's multi-tree prediction path performs the same storage validation once
for the batch and still validates every stored tree graph before traversal.
Fitted internal nodes use the same split-boundary validation as `partition_rows`:
threshold bins must be `0 .. n_bins[feature] - 2`, so terminal-bin and one-bin
thresholds are rejected.

## Numerical safeguards

The module defines `EPSILON = 1e-15`.

- Initial positive rates are clipped to `[EPSILON, 1 - EPSILON]` before taking log-odds.
- Ordinary probabilities are not clipped before gradient calculation.
- If a leaf has `H + reg_lambda <= EPSILON`, its value and score are zero.
- All learned arrays and predictions must remain finite.

## Sparse guarantees

- Project training matrices must never be densified.
- Sparse inputs are canonicalized by summing duplicates, eliminating explicit zeros,
  and sorting indices.
- An implicit sparse zero is an ordinary numeric zero, never a missing value.
- Dense, CSR, and CSC versions of the same logical matrix must produce identical bins,
  trees, raw scores, probabilities, and labels.

## Persistence

The estimator uses ordinary Python, NumPy, SciPy, and dataclass state, so project-level
pickle or joblib persistence can be used externally. A persisted model must reproduce
raw scores, probabilities, and labels exactly after a round trip. It is not compatible
with LightGBM model files.

The internal refactor retains aliases for moved classes in `src.lite_lightgbm` so older
project pickles can resolve their original names. New pickles may record implementation
module paths for nested dataclasses, so the façade and all files in `lite_lightgbm_dep/`
must be distributed together. User code should never compensate by importing dependency
modules directly.

## Further reading

- [`lite_lightgbm.md`](lite_lightgbm.md): implementation contract and acceptance gates.
- [`lite_lightgbm_refac.md`](lite_lightgbm_refac.md): extraction-only module refactor plan.
- [LightGBM features](https://lightgbm.readthedocs.io/en/stable/Features.html)
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
- [LightGBM paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
