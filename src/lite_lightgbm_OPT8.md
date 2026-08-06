# LiteLightGBM optimization 8: vectorized split evaluation

## Status

Implementation and correctness are complete. The trusted
`_find_best_split_validated(...)` kernel evaluates thresholds with NumPy vectors
and batches the sole threshold of all two-bin features, while retaining one
bounded loop over wider feature segments and no Python loop over candidate
thresholds. A private `_find_best_split_scalar_validated(...)` helper retains
the pre-OPT8 threshold loop as a correctness oracle for focused parity checks;
normal training calls only the vectorized kernel. Direct-kernel microbenchmarks
now cover the low-bin and representative multi-bin shapes; no full-tree or
target-dataset timing is claimed here.

## Objective

Replace the Python loop over every candidate threshold in split search with
NumPy array operations. Keep one bounded Python loop over selected wider
features so each multi-bin histogram segment is handled independently, and
evaluate all two-bin features in one batched operation because each has only
the threshold between bins 0 and 1.

The optimization must preserve:

- the public `find_best_split(...)` interface and its validation behavior;
- exact child-count and child-Hessian eligibility rules;
- the strict `gain > min_split_gain` rule;
- deterministic tie-breaking by gain, original feature ID, then threshold bin;
- original feature IDs and threshold bins stored in `SplitInfo` and tree nodes;
- the existing L1/L2 score formula and `EPSILON` denominator guard; and
- sparse training without scikit-learn or LightGBM imports.

## Measured motivation

On 2026-08-05, one unprofiled fit on the supplied 20,000 x 5,000 CSR matrix
used the tuned-model settings relevant to split cost:

```text
n_estimators=1
num_leaves=31
min_child_samples=7
colsample_bytree=0.5198042692950159
reg_alpha=0.34104824737458306
reg_lambda=0.13010318597055903
random_state=42
```

The fit took 53.8720 seconds, including about 9.6 seconds of one-time mapper
construction. It produced 61 nodes and sampled 2,599 features.

A separate instrumented run was slower because profiling every Python call adds
overhead, but it identified the bottleneck clearly:

| Routine | Profiled calls | Profiled cumulative time |
| --- | ---: | ---: |
| `find_best_split` | 59 | 49.858 s |
| `soft_threshold` | 12,737,124 | 26.497 s |
| `build_histogram` | 59 | 10.902 s |

The absolute profiled times must not be used as wall-clock estimates. The call
counts show the real problem: split search performs two scalar soft-threshold
operations for nearly every valid boundary. Optimizations 4-7 reduce validation
and histogram work but do not remove this threshold loop.

## Dependencies and implementation order

Implement optimization 4 first. Its checked public wrapper should validate and
normalize direct caller input once, then pass a trusted context to a private
split-search kernel such as `_find_best_split_validated(...)`. Optimization 8
changes that private kernel only.

Optimizations 5-7 are not logical prerequisites for vectorized split evaluation.
They may be implemented before optimization 8 to preserve numeric file order,
but do not couple their histogram or cache changes to this work. Compare scalar
and vectorized split search using identical histograms.

Do not add a public estimator parameter selecting scalar or vectorized search.
A private reference helper or a test-only copy of the scalar kernel is enough.

## Current scalar behavior to preserve

For one feature with `B` bins, the current kernel:

1. calculates gradient, Hessian, and count prefix sums over all `B` bins;
2. visits thresholds `0` through `B - 2` in ascending order;
3. rejects a threshold unless both child counts meet `min_child_samples`;
4. rejects it unless both child Hessians meet `min_child_weight`;
5. assigns score zero when `child_hessian + reg_lambda <= EPSILON`;
6. otherwise applies `soft_threshold(child_gradient, reg_alpha)` and divides
   its square by `child_hessian + reg_lambda`;
7. calculates `gain = left_score + right_score - parent_score`;
8. rejects non-finite gain and gain not strictly above `min_split_gain`; and
9. keeps the largest gain, resolving exact ties toward the lower original
   feature ID and then the lower threshold index.

`default_left` is not searched. It remains
`layout.default_bins[local_slot] <= threshold` for a local layout, or the
equivalent mapper default for the global layout.

## Vectorization boundary

Vectorize thresholds within one feature. Do not flatten every feature and
threshold into a single project-wide array in the first implementation. The
sole threshold of every two-bin feature is a special batched case: gather bin-0
statistics across those features, score them together, and leave wider
segments on the per-feature threshold-vector path.

The per-feature design is intentional:

- `max_bin` normally bounds temporary threshold arrays at 254 elements;
- existing histogram segments and prefix-sum semantics remain obvious;
- no segment-ID or cross-feature boundary mask is required;
- lower-threshold tie-breaking follows directly from first-maximum selection;
- memory does not scale with every selected bin in the tree; and
- a bounded feature loop is much cheaper than millions of threshold iterations.

Do not use `np.vectorize`; it still executes Python once per element.

The two-bin batch keeps direct-helper feature order for statistics but resolves
exact gain ties by original feature ID, matching the scalar global tie-break.
Constant one-bin features are skipped, and mixed bin-count layouts split into
the batch plus the established wider-segment loop.

## Detailed algorithm

### 1. Keep the checked wrapper

`find_best_split(...)` remains the directly callable validating wrapper. After
optimization 4, it owns malformed histogram, mapper, feature-index, parent-stat,
layout, and configuration checks.

The private kernel may assume:

- histogram arrays are one-dimensional, finite, and layout-compatible;
- count arrays are non-negative `int64` values;
- selected features and local layout metadata are valid;
- configuration values are normalized and finite; and
- parent statistics are normalized scalars.

Do not repeat these structural checks inside the per-feature loop.

### 2. Preserve parent scoring

Calculate `parent_score` once using the existing scalar formula. Do not change
the `EPSILON`, L1, L2, or zero-denominator behavior in this optimization.

### 3. Build per-feature prefix vectors

For feature histogram slice `[start:end]`, continue using:

```python
gradient_prefix = np.cumsum(gradients[start:end], dtype=np.float64)
hessian_prefix = np.cumsum(hessians[start:end], dtype=np.float64)
count_prefix = np.cumsum(counts[start:end], dtype=np.int64)
```

Candidate left statistics are the prefixes excluding the final bin:

```python
left_gradient = gradient_prefix[:-1]
left_hessian = hessian_prefix[:-1]
left_count = count_prefix[:-1]
```

Right statistics remain parent totals minus those arrays. Do not independently
sum right-bin suffixes; subtraction is the established numerical behavior.

### 4. Construct one eligibility mask

Build a Boolean mask in the same logical order as the scalar guards:

```text
left_count >= min_child_samples
right_count >= min_child_samples
left_hessian >= min_child_weight
right_hessian >= min_child_weight
```

If no threshold survives, continue to the next feature without allocating score
arrays. Use `np.flatnonzero(valid)` to retain ascending threshold order.

### 5. Score only eligible thresholds

Gather the eligible child statistics and calculate vector scores. Initialize
each side's score to zero. Apply soft thresholding and division only where:

```text
child_hessian + reg_lambda > EPSILON
```

The existing `soft_threshold` already accepts NumPy arrays. Reuse it; do not add
a second L1 implementation with subtly different sign or zero behavior.

Conceptually:

```python
left_score = np.zeros(candidate_count, dtype=np.float64)
left_denominator = eligible_left_hessian + reg_lambda
positive = left_denominator > EPSILON
thresholded = soft_threshold(eligible_left_gradient[positive], reg_alpha)
left_score[positive] = thresholded * thresholded / left_denominator[positive]
```

Apply the same operation to the right side. Then calculate gains with exactly
the established expression:

```python
gains = left_score + right_score - parent_score
```

Remove non-finite gains and gains that do not strictly exceed
`min_split_gain`. Do not suppress warnings by globally changing NumPy error
settings; avoid invalid divisions through the denominator mask.

### 6. Select the feature's winning threshold

Candidate threshold indices are ascending. `np.argmax` returns the first
maximum, so applying it to the surviving gains preserves the lower-threshold
tie-break within one feature.

Do not use an unstable sort. Do not use a tolerance when comparing gains. A
tolerance would change which genuinely distinct split wins.

Convert only the selected array entries to Python `int` or `float` values when
constructing `SplitInfo`. The stored counts and sufficient statistics must come
from the same prefix entries used to score the winner.

### 7. Preserve the global tie-break

Compare the per-feature winner with the global winner using the current exact
logic:

```text
higher gain wins
exact equal gain -> lower original feature ID wins
same feature -> lower threshold wins
```

Do not assume a direct helper caller supplied features in sorted order when no
local layout is present. Tree-local layouts are sorted, but the explicit
feature-ID comparison keeps the public helper behavior independent of input
order.

## Numerical behavior

This optimization changes scalar NumPy operations into element-wise array
operations but does not change histogram accumulation order. The expected goal
is exact split parity.

Require:

- identical candidate count and Hessian eligibility masks;
- identical strict handling of `min_split_gain` and `EPSILON` boundaries;
- identical chosen feature, threshold, direction, and counts;
- exact or bitwise-equal sufficient statistics when they come directly from
  unchanged prefix arrays; and
- deterministic repeated results.

Compare gains exactly first. If the platform produces a small element-wise
floating difference, document it and require a tight scale-aware tolerance for
gain values while still requiring identical selected splits on all non-tie
fixtures. Do not round gains or introduce a comparison tolerance to force tree
parity.

Near-tie fixtures deserve explicit tests. If vectorized arithmetic changes a
winner only at a genuine floating near-tie, retain a narrow scalar fallback for
the tied candidate set rather than falling back for every threshold. Add such a
fallback only after a reproducing test proves it necessary.

## Reference implementation

Retain the scalar threshold loop as a correctness oracle for parity checks.
Prefer one of these forms:

1. a private `_find_best_split_scalar_validated(...)` helper used only by
   tests; or
2. an exact copy in the focused test module.

Do not expose the oracle through `LiteLightGBM` and do not run both algorithms
during normal training. Once randomized and project-data parity is established,
the oracle may live only in tests.

## Verification plan

### Focused split fixtures

Compare scalar and vectorized results for:

- zero selected features and one-bin constant features;
- two bins, the smallest splittable segment;
- first, middle, and final legal threshold winners;
- thresholds rejected by left or right `min_child_samples`;
- thresholds rejected by left or right `min_child_weight`;
- `min_child_samples=0` and `min_child_weight=0`;
- `reg_alpha=0` and positive L1 regularization;
- `reg_lambda=0` and positive L2 regularization;
- denominators below, equal to, and above `EPSILON`;
- mixed positive, negative, and zero gradients;
- gain equal to `min_split_gain`, which must be rejected;
- non-finite calculated gain, which must be rejected;
- exact threshold ties within one feature;
- exact gain ties across features supplied in both sorted and unsorted order;
- global mapper layout and compact `HistogramLayout`; and
- default bins before, at, and after the selected threshold.

### Randomized histogram parity

Generate valid randomized histogram segments with fixed seeds. Derive parent
statistics from the segment totals so the fixtures obey histogram invariants.
Sweep several combinations of child constraints and regularization. Compare the
complete returned `SplitInfo`, not only its gain.

The focused split-search regression covers all-two-bin global and compact local
layouts, mixed one-/two-/multi-bin features, unsorted direct feature order,
exact feature ties, non-finite gain filtering, and parent-count overflow. The
vectorized and scalar helpers match exactly on these fixtures and randomized
parity sweeps.

Include deliberately invalid direct-helper inputs in the existing OPT4 tests to
confirm the checked public wrapper still raises the same exception classes and
messages. The private vectorized kernel is not responsible for malformed input.

### Tree and estimator parity

Fit scalar-oracle and vectorized trees with identical binned data, gradients,
rows, features, and configuration. Require identical:

- node count and topology;
- original split feature IDs and threshold bins;
- default directions and sample counts;
- leaf values and split gains, exactly when possible and otherwise within a
  documented tight tolerance;
- raw tree output; and
- repeated-run determinism.

Repeat complete estimator comparisons for dense, CSR, and CSC logical inputs,
including feature subsampling, row bagging, sample weights, balanced class
weights, constant columns, and an all-filtered matrix.

## Direct-kernel benchmark record (local venv, 3-run medians)

The following excludes mapper fitting and full-tree orchestration. Each case
uses the same synthetic flattened histogram for the vectorized kernel and the
private scalar oracle:

| shape | vectorized | scalar oracle | scalar/vector speedup |
| --- | ---: | ---: | ---: |
| 5,000 features × 2 bins | 0.0106 s | 0.2413 s | 22.7× |
| 40,385 features × 2 bins | 0.1147 s | 2.6489 s | 23.1× |
| 5,000 features × 8 bins | 0.8533 s | 1.0643 s | 1.25× |
| 5,000 features × 32 bins | 0.9469 s | 3.9931 s | 4.22× |
| 5,000 mixed 2/8/32-bin features | 0.7515 s | 1.9394 s | 2.58× |

The pre-F2 local measurements were 0.993 s versus 0.314 s for the 5,000 ×
2-bin case and 8.04 s versus 2.32 s for 40,385 × 2 bins (vectorized versus
scalar); those measurements motivated the batched low-bin path. The current
records are direct-kernel evidence only: no full-tree or target-dataset timing
is claimed.

## Benchmark plan

Exclude mapper fitting and reuse identical binned data and histograms.

First benchmark scalar and vectorized split search directly on:

- the root histogram;
- representative medium and small leaves;
- `colsample_bytree=1.0`; and
- the tuned value near `0.52`.

Report selected feature count, local bin count, valid threshold count, scalar
time, vectorized time, speedup, and split parity.

Then benchmark the complete 31-leaf tree on the supplied 20,000 x 5,000 matrix
after optimizations 4-7, reporting:

- aggregate split-search time;
- histogram time;
- partition and prediction time;
- complete tree time;
- node and split parity; and
- peak process memory.

Treat any complete-tree speedup as a separate measurement, not a correctness
exemption. If a target-scale run is needed, profile its split/histogram balance
before attempting a more complex all-feature flattening design.

Finally, run a bounded root or small-tree benchmark on the 40,385-feature
representation before launching a complete fit.

## Files and change boundaries

Expected implementation ownership:

- `lite_lightgbm_dep/tree.py`: vectorize the trusted split kernel and retain or
  expose a private scalar oracle for tests;
- focused tests: scalar/vector split, tree, and estimator parity; and
- `lite_lightgbm_OPT8.md`: records the vectorized-kernel design and the private
  scalar-oracle role; direct-kernel benchmark evidence is recorded above, while
  full-tree and target-scale performance remain unmeasured.

No changes should be required in:

- the `LiteLightGBM` constructor or fitted attributes;
- the public façade exports;
- model serialization format;
- binning, histogram layout, tree queue, or prediction traversal; or
- application/notebook call sites.

## Acceptance criteria

- No Python loop iterates over candidate thresholds in the optimized kernel.
- One bounded Python loop over selected features remains acceptable.
- Public `find_best_split(...)` validation behavior remains intact.
- Scalar and vectorized candidate eligibility and deterministic tie-breaking
  agree on focused and randomized fixtures.
- Seeded tree topology and predictions remain equivalent.
- Sparse inputs remain sparse and no scikit-learn or LightGBM dependency is
  introduced.
- Low-bin and representative multi-bin direct-kernel performance is measured;
  full-tree and target-scale performance remain pending.
- The implementation does not add a public optimization switch or alter saved
  model structure.
