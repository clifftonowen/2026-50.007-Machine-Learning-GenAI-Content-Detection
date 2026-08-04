# LiteLightGBM optimization 1: vectorize bin-boundary selection

## Objective

Reduce the one-time cost of `fit_bin_mapper` without changing any learned cut
point, default bin, bin count, validation rule, or deterministic tie-break.
This is the first optimization because profiling the supplied 5,000-feature
matrix showed that mapper fitting took roughly 68-85 seconds, while applying an
already-fitted mapper took less than half a second.

This optimization must use only NumPy, SciPy, and the Python standard library.
It must not add a scikit-learn dependency.

## Current bottleneck

For each feature, `fit_bin_mapper` currently tries bin counts from the largest
feasible value down to one. For every desired boundary it then:

1. loops over every candidate boundary in Python;
2. appends feasible boundaries to a Python list;
3. constructs another list to calculate distances from the desired rank.

The result is correct, but the nested Python loops become expensive across
thousands of high-cardinality features. Sparse value extraction and
`np.unique` are not the main target of this change.

## Required behavior

Preserve the binning contract in `lite_lightgbm.md` exactly:

- implicit sparse zeros participate in value counts;
- the largest feasible number of bins is selected;
- each completed bin contains at least `min_data_in_bin` rows;
- enough rows and distinct values remain for unfinished bins;
- a boundary minimizes distance from `k * n_samples / candidate_bins`;
- equal distances choose the lower boundary;
- cut points are the largest training values in non-final bins;
- dense, CSR, CSC, and non-canonical sparse equivalents produce identical
  mappers.

Do not replace the documented algorithm with `np.quantile`. That would be
faster but would change frequency handling, minimum-bin-size behavior, and tie
resolution.

## Implementation design

Extract the boundary search into one private helper, for example:

```python
def _find_bin_boundaries(
    values: np.ndarray,
    counts: np.ndarray,
    n_samples: int,
    max_bin: int,
    min_data_in_bin: int,
) -> tuple[np.ndarray, int]:
    ...
```

`values` must already be sorted distinct `float64` values and `counts` must be
matching positive `int64` occurrence counts. The helper returns the cut points
and selected bin count. Keep input validation at the existing public/helper
boundary; do not repeat it inside the inner candidate loop.

For each candidate bin count, retain the short loop over desired boundaries,
because every chosen boundary changes the feasible range for the next one.
Replace the inner candidate scan with NumPy indexing:

```text
lower = previous_boundary + 1
upper = n_distinct - (candidate_bins - k) - 1
candidates = all_boundary_indices[lower : upper + 1]

left_counts = cumulative[candidates] - previous_count
remaining_counts = n_samples - cumulative[candidates]
feasible = (
    (left_counts >= min_data_in_bin)
    & (remaining_counts >= remaining_bins * min_data_in_bin)
)
feasible_candidates = candidates[feasible]
chosen = feasible_candidates[
    argmin(abs(cumulative[feasible_candidates] - desired_rank))
]
```

Create `all_boundary_indices = np.arange(n_distinct - 1, dtype=np.int64)` once
per feature. `np.argmin` returns the first minimum, and candidates are ascending,
so the existing lower-boundary tie-break is preserved. Continue decreasing the
candidate bin count when any desired boundary has no feasible candidate.

Process one feature through counting and boundary selection before moving to
the next if doing so is straightforward. The current `value_counts` list keeps
the distinct values and counts for every feature alive simultaneously. Removing
that list lowers peak memory, but it is secondary to the vectorized boundary
search and must not duplicate the dense/sparse counting logic unnecessarily.

Keep these mapper metadata arrays as `int64`:

- `default_bins`;
- `n_bins`;
- `bin_offsets`.

Compact sparse bin storage is optimization 2, not part of this change.

## Verification

Before replacing the current loop, record its outputs on deterministic fixtures
and use them as the correctness oracle. Tests must cover:

- a constant feature;
- exactly two distinct values;
- repeated values with highly unequal counts;
- a case where the initially requested bin count must be reduced;
- equal-distance candidate boundaries, verifying the lower boundary wins;
- zero as the smallest, middle, and largest distinct value;
- implicit sparse zeros;
- `max_bin` larger than the number of distinct values;
- `min_data_in_bin` equal to one and large enough to force one bin;
- logically identical dense, CSR, and CSC inputs;
- unsorted sparse indices, duplicate coordinates, and explicit sparse zeros.

For randomized small matrices, compare every element of `cut_points`,
`default_bins`, `n_bins`, and `bin_offsets` against a test-only copy of the old
reference algorithm. These results should be exactly equal, not merely close.

## Benchmark

Measure `fit_bin_mapper` separately from `transform_bins` on the supplied
5,000-feature representation. Run the old and new implementations in the same
process after one warm-up, using the same input and configuration. Report:

- wall-clock time;
- peak process memory if available;
- total number of bins;
- exact mapper equality.

## Acceptance criteria

- All mapper outputs are exactly unchanged on deterministic and randomized
  parity fixtures.
- Sparse input is never densified.
- The optimized mapper is deterministic across repeated runs.
- The 5,000-feature mapper benchmark is materially faster; target at least a
  2x reduction in mapper time before proceeding.
- No estimator API or fitted prediction changes as a result of this step.

