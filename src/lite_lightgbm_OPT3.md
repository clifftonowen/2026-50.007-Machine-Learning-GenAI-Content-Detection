# LiteLightGBM optimization 3: feature pre-filtering and local histogram layouts

## Status

Implementation: complete in the current worktree. `LiteLightGBM.fit` performs the
count-only active-feature prefilter after bin transformation, samples from the
original feature range before intersecting with that set, and `fit_tree` builds
one immutable local histogram layout shared by all leaves in a tree.

Verification: the bounded in-memory checks below passed on 2026-08-05. This
worktree has no `test/` or `tests/` directory, so no repository OPT3 test suite is
claimed or reported. The supplied 5,000-column matrix was benchmarked in full.
The chosen 40,385-column representation was reconstructed from its cached
A--F/H/I blocks and benchmarked on a deterministic 512-row slice; a full
20,000-row 40,385-column 31-leaf run is deliberately not claimed because it is
outside the bounded evidence run.

## Evidence (2026-08-05)

All checks were read-only and in-memory. Each was run as a PowerShell
here-string piped to the repository interpreter:

```text
@' ... '@ | .\.venv\Scripts\python.exe -
```

The scripts imported only NumPy, SciPy, and this package. They used
`LiteLightGBMConfig(max_bin=8, min_data_in_bin=1, min_child_weight=0,
min_split_gain=0)` for the small fixtures and fixed `random_state` values.

### Mandatory OPT3 fixtures

| Case | Configuration / result |
| --- | --- |
| Constant columns plus one informative column | 12 x 4 dense, `min_child_samples=2`: `active_features_ == [1]`; the only split used original feature `1`; `feature_importances_ == [0, 1, 0, 0]`. |
| Non-constant but no legal count boundary | 10 x 2 (`5/5` values), `min_child_samples=6`: `active_features_ == []`; three fitted trees were root-only (`[1, 1, 1]` nodes) and predictions were finite. |
| Implicit-default sparse counts | 10 x 1 CSR with two stored ones and eight implicit zeros, `min_child_samples=2`: `default_bins == [0]`, `n_bins == [2]`, and `active_features_ == [0]`; the tree split on feature `0`. |
| All features filtered | 10 x 3 all-zero dense, `min_child_samples=2`, three estimators: `active_features_ == []`, every tree root-only, finite probabilities, and all importances zero. |
| `min_child_samples=0` | 10 x 3 with one constant and two non-constant columns: `n_bins == [1, 2, 3]` and `active_features_ == [1, 2]`. |
| Seeded fractional feature sampling | With active `[1, 3]`, `colsample_bytree=0.5`, `random_state=0` drew `[2, 3]` from the four original columns and used the intersection `[3]`; `colsample_bytree=1.0` used `[1, 3]` directly. |
| Dense/CSR/CSC parity | The same 12 x 4 logical matrix, labels, and seed produced `active_features_ == [1, 3]`, the same split sequence (`[1]`, `[1]` for two trees), and equal raw predictions for dense, CSR, and CSC inputs (absolute tolerance `1e-12`). |

An independent audit also ran an **unfiltered-control behavioral proxy**. For
each of dense, CSR, and CSC input, and for both `colsample_bytree=1.0` and a
fixed fractional value, the optimized fit was paired with an in-memory control
that retained every original feature and the global histogram layout. The six
pairs used identical labels, configuration, and fixed seeds. Node fields and
tree structure, raw predictions, and `feature_importances_` were exactly equal
for every pair. This is a behavioral proxy for the pre-OPT3 path: the repository
does not retain a separate runnable pre-optimization module, so the result must
not be read as a claim of an old-module benchmark or old/new source comparison.

For layout correctness, a direct root histogram on the 12 x 4 fixture was
built once with all four features and once with the local layout for `[1, 3]`.
The corresponding count, gradient, and Hessian segments were equal exactly
(6 global bins versus 4 local bins); all five layout arrays were read-only.

### Real sparse-matrix benchmark

The common benchmark configuration was
`num_leaves=31, max_depth=-1, min_child_weight=1e-3, min_split_gain=0,
max_bin=255, min_data_in_bin=3, reg_alpha=reg_lambda=0,
colsample_bytree=1.0, random_state=123`, with fixed NumPy generator seed
`20260805` for the tree gradients and unit Hessians. Histogram bytes are the
three flattened arrays (`float64`, `float64`, `int64`), i.e. 24 bytes per bin.

| Representation | Rows x original features; stored nnz | Active; global -> local bins | Histogram bytes global -> local; local-layout bytes | Mapper / transform / filter (s); 31-leaf tree (s) |
| --- | --- | --- | --- | --- |
| Supplied CSR (full) | 20,000 x 5,000; 1,356,986 nnz (density `0.01356986`) | 4,999; `376,209 -> 376,204` | `9,029,016 -> 9,028,896`; `199,976` | `9.3879 / 0.2404 / 0.1228`; `76.0002` (61 nodes, 31 leaves, 30 splits) |
| Chosen cached blocks A--F/H/I (deterministic first 512 rows) | Reconstructed full shape 20,000 x 40,385 with 34,789,036 nnz (density `0.04307173`); benchmark slice 512 x 40,385 with 913,964 nnz (density `0.04420171`) | 23,397; `331,000 -> 309,264` | `7,944,000 -> 7,422,336`; `1,071,792` | `9.8312 / 0.8398 / 0.6174`; `110.6532` (61 nodes, 31 leaves, 30 splits) |

The supplied 5,000-column matrix filters only one feature and saves five bins
(120 histogram bytes), so its active-feature/layout gain is negligible; the
40,385-column result is a deterministic 512-row slice, so its filtering may be
overstated relative to the full matrix.

The 5,000-column tree used `min_child_samples=20`. The 40,385-column slice
used `min_child_samples=5` so that its bounded 512-row run reached the full
31-leaf budget; timings are therefore evidence of allocation and layout scale,
not a like-for-like model-quality comparison. Split-sequence / raw-output
SHA-256 prefixes make the fixed-seed reruns checkable: `fb7b0aa78a650b4d` /
`98ede7a0b61e6d6c` (5,000-column run) and `fb723bfb8c1cc2e8` /
`b17b733ac15f87f7` (40,385-column slice). No pre-OPT3 implementation was
available as a runnable repository module in this check, so these hashes do
not claim old/new equivalence.

The exact 40,385-column shape comes from the project’s chosen feature set:
`A_function_words`, `B_punctuation`, `C_casing`, `D_structure`, `E_length`,
`F_diversity`, `H_char_ngrams`, and `I_word_ngrams` (the cached block widths sum
to 40,385). There is no single persisted merged file; the benchmark command
loaded those eight cached CSR blocks and horizontally stacked them in memory.

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
