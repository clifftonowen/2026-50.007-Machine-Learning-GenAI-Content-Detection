# LiteLightGBM optimization 1: vectorize bin-boundary selection

## Status: completed

OPT1 is implemented in `src/lite_lightgbm_dep/binning.py` and re-exported through
the stable `src.lite_lightgbm` facade. `fit_bin_mapper` now delegates
per-feature boundary selection to `_find_bin_boundaries`; its inner candidate
scan uses NumPy cumulative counts, slices, masks, and stable `argmin`, while
the sequential outer boundary loop remains intact. Independent verification
confirmed exact mapper parity on deterministic and randomized reference cases,
including dense, CSR, CSC, and non-canonical sparse inputs, with deterministic
tie-breaking and no sparse densification.

Completion evidence (2026-08-05): on the supplied
`data/raw/train_features.csv` representation (20,000 rows x 5,000 features),
loaded as the project CSR matrix (`float32`, 1,356,986 stored values,
density 0.01356986), the git-HEAD mapper and current mapper were run in one
process with one full-input warm-up each and the same
`LiteLightGBMConfig(max_bin=255, min_data_in_bin=3)`. The timed runs were
64.1251 s (git HEAD) versus 10.7056 s (current), a 5.9899x speedup. Peak
process RSS sampled during the runs was 151,875,584 B (git HEAD; +38,932,480 B
over its timed baseline) and 155,062,272 B (current; +36,073,472 B). Both
mappers had exactly 376,209 total bins (per-feature range 5..255), and
`cut_points`, `default_bins`, `n_bins`, and `bin_offsets` were all
`np.array_equal`. Reproduction used a PowerShell here-string piped to
`uv run python -`, loading with `src.data.load_train_features(sparse=True)`;
the git-HEAD source was read in-memory with `git show
HEAD:src/lite_lightgbm.py` (no workspace file replacement), timed with
`time.perf_counter`, and RSS sampled via `psutil` every 2 ms.

Deliberately deferred: the secondary per-feature streaming suggested under
"Implementation design" was not implemented. `fit_bin_mapper` still builds the
complete `value_counts` list before selecting boundaries, so every feature's
distinct values and counts stay alive simultaneously. On the supplied
5,000-feature representation that list holds 1,357,223 distinct values in total
(mean 271.4, maximum 9,354 per feature), which is roughly 21.7 MB of array
payload plus about 1.1 MB of ndarray object overhead; streaming would hold
about 0.15 MB at a time. That accounts for most of the +36,073,472 B RSS delta
recorded above. It was left out because the mapper runs once per fit, the
saving is roughly 15% of process RSS, and no memory limit has been reached,
whereas restructuring would touch a routine whose exact parity is now verified
without improving mapper time at all. Revisit it only if profiling the
40,385-feature representation shows mapper peak memory to be a real constraint;
the payload scales roughly with total stored values. The clean form is a
generator yielding `(values, counts)` per feature that the existing selection
loop consumes, which keeps peak memory low without duplicating the dense and
sparse counting branches. Compact bin dtypes remain optimization 2.

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
