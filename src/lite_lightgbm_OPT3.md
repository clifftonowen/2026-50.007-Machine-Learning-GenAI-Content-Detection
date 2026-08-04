# LiteLightGBM optimization 3: feature pre-filtering and local histogram layouts

## Objective

Avoid considering features that cannot produce a valid split, and allocate each
tree's histograms only for features actually eligible in that tree. The project
matrix has roughly 40,000 columns, so eliminating constant or globally
unsplittable columns and unused histogram segments can substantially reduce
work and memory.

This change must preserve original feature indices in trained nodes and
`feature_importances_`. It must not use scikit-learn.

## Safe initial filter

The first implementation may remove a feature only when it is provably unable
to satisfy a count-based split for any subset of the training rows.

A feature is inactive when either:

1. `mapper.n_bins[feature] <= 1`; or
2. no boundary in its full-training bin counts has both
   `left_count >= min_child_samples` and
   `right_count >= min_child_samples`.

The second rule is safe because a leaf or bagged subset cannot contain more
rows on either side of a boundary than the full training set. It must use exact
bin counts, including implicit default-bin rows.

Do not initially filter using `min_child_weight`, gradients, Hessians, gain,
class weights, or sparsity alone. Hessians and gains change during boosting,
and a sparse feature can still be predictive through its default bin.

## Computing active features

After `transform_bins`, calculate full-training bin counts once. For each CSC
feature:

1. decode explicitly stored bins and count them with `np.bincount`;
2. add `n_samples - explicit_entry_count` to the feature's default bin;
3. calculate prefix counts before the final bin;
4. retain the feature if any prefix satisfies both child-count constraints.

The one-time prefilter may use a short loop over features. It must not allocate
a dense `n_samples x n_features` matrix or build gradient/Hessian histograms.
Return sorted original feature indices, preferably as `int64`.

Store the result as `active_features_` after a successful fit. This learned
attribute is diagnostic metadata; nodes and feature importances continue using
the original column numbering.

## Preserve feature-sampling behavior

For compatibility with the current seeded model, feature sampling must continue
to draw from all original feature indices using the existing RNG call and count:

```text
sampled = current colsample_bytree sample from range(n_features)
tree_features = intersection(sampled, active_features)
```

When `colsample_bytree == 1`, use `active_features` directly. Intersect sorted
arrays without changing order. If the intersection is empty, fit a root-only
tree; do not draw a replacement feature. Sampling only from the active pool
would change RNG semantics and the effective candidate set, so that behavior is
outside this optimization.

## Tree-local histogram layout

The current histogram arrays span `mapper.bin_offsets[-1]`, including every
feature. Introduce a small internal immutable layout built once per tree, for
example:

```python
@dataclass(frozen=True, slots=True)
class HistogramLayout:
    feature_indices: np.ndarray       # sorted original indices
    n_bins: np.ndarray                # mapper.n_bins[feature_indices]
    default_bins: np.ndarray          # mapper.default_bins[feature_indices]
    bin_offsets: np.ndarray           # local prefix sum, starting at zero
    feature_to_local: np.ndarray      # original index -> local slot, or -1
```

`feature_to_local` is useful for optimization 5. Use a signed integer dtype so
`-1` can represent an excluded feature. If memory is a concern, `int32` is
sufficient while `n_features < 2**31`; validate before narrowing.

Update `Histogram` to carry or be unambiguously associated with its layout.
`build_histogram` writes feature segments using local offsets.
`find_best_split` iterates local feature slots but writes the corresponding
original feature index into `SplitInfo`. `partition_rows`, prediction, and
`feature_importances_` therefore need no feature remapping.

Create one layout per tree after feature sampling and pass the same object to
every histogram and split search in that tree. Do not rebuild the
`feature_to_local` lookup for every leaf.

## Empty and constant cases

- An empty active-feature set produces a layout with one offset `[0]` and zero
  histogram-length arrays.
- A histogram with no eligible features produces no split.
- A completely constant training matrix still fits `n_estimators` root-only
  trees and returns finite predictions.
- `feature_importances_` retains length `n_features_in_` and remains zero for
  filtered features.

## Verification

Tests must include:

- constant columns mixed with one informative column;
- a non-constant feature whose every boundary violates `min_child_samples`;
- implicit-default sparse counts;
- all features filtered;
- `min_child_samples=0`, where every non-constant feature remains eligible;
- `colsample_bytree=1` and a fractional value;
- a seed whose sampled set contains both active and inactive features;
- original feature indices preserved in nodes and importances;
- dense/CSR/CSC active-feature parity.

For equivalence testing, compare against the pre-optimization implementation
with the same seed. Because filtered features could never produce a valid
split and sampling is preserved before intersection, tree structure and raw
predictions should be exactly equal. Also compare local histogram segments
with the corresponding slices from the old global histogram.

## Benchmark

Report, for both the 5,000-feature and chosen 40,385-feature matrices:

- original and active feature counts;
- original total bins and per-tree local total bins;
- bytes per histogram before and after;
- one 31-leaf tree time with the same sampled features;
- identical split sequence and raw predictions on a fixed seed.

## Acceptance criteria

- Only provably count-unsplittable features are removed.
- Existing seeded feature-sampling semantics are preserved.
- Histograms allocate segments only for the tree's active sampled features.
- Every stored split and importance uses the original feature index.
- Old and new seeded models are behaviorally identical on parity fixtures.
- No training or prediction path densifies the feature matrix.

