# LiteLightGBM optimization 5: vectorized flattened histogram aggregation

## Objective

Replace the per-feature Python loop in histogram construction with aggregation
over flattened tree-local bin keys. This is expected to provide the largest
per-tree speedup because histogram construction is repeated for many leaves and
currently searches selected rows separately in each CSC column.

This optimization assumes the local `HistogramLayout` from optimization 3 and
the trusted internal kernel from optimization 4. It must retain a direct
reference implementation for tests until histogram subtraction is verified.
It must not use scikit-learn or densify the feature matrix.

## Status

Implemented and verified on the supplied 20,000 x 5,000 representation.
Histogram construction now uses deterministic, bounded CSR row blocks and
flattened tree-local bin keys with `np.bincount` aggregation. The previous
per-feature CSC implementation remains as the private `_build_histogram_direct`
reference oracle for parity tests. The checked public helper and the trusted
`fit_tree` context validate CSR encoded-bin storage before entering the
optimized kernel.

The required 40,385-feature root-memory benchmark remains pending. Do not call
the full OPT5 acceptance gate complete until that measurement confirms the
temporary-memory bound on the chosen representation.

## Verification evidence (2026-08-06)

### Differential model sweep

A 525-fit comparison against the direct histogram path produced identical
predicted labels in every fit. The maximum raw-score difference was `2.7e-15`.
Two fits selected different tree splits because the CSR/`bincount` accumulation
order perturbed nearly equal gains:

| Case | Vectorized split | Direct split | Interpretation |
| --- | --- | --- | --- |
| `t14/c0` | feature 1, bin 0, gain `0.9303916629429135` | feature 0, bin 1, gain `0.9303916629429133` | Difference `2.2e-16`, or one ULP. The gains are not exactly equal after accumulation, so the exact lower-feature tie-break does not apply. |
| `t23/c6` | feature 2, bin 0, gain about `2.66e-15` | feature 0, bin 0, gain about `4.44e-16` | Both gains are floating-point noise above `min_split_gain=0`; OPT5 changes which noise-level split wins, not whether the scalar path can accept one. |

These cases satisfy the numerical-behavior contract: report genuine near-ties
rather than rounding histograms or adding a hidden gain tolerance. A caller that
wants to suppress noise-level splits may set a positive `min_split_gain`
explicitly. OPT5 does not replace the documented strict gain rule with an
implicit floor.

### Supplied-matrix benchmark

The benchmark used the full 20,000 x 5,000 sparse representation with
1,356,986 stored values. Histogram results had exactly equal counts, maximum
absolute gradient difference `5.7e-14`, and maximum absolute Hessian difference
`0.0`.

| Leaf | Vectorized | Direct CSC | Speedup |
| --- | ---: | ---: | ---: |
| Root, 20,000 rows | 0.0817 s | 0.3490 s | 4.27x |
| One quarter, 5,000 rows | 0.0237 s | 0.3025 s | 12.79x |
| One thirty-second, 625 rows | 0.0098 s | 0.2713 s | 27.76x |

The direct path loops over selected features regardless of leaf size, whereas
the vectorized path scales primarily with the leaf's stored values. The benefit
therefore grows for deeper, smaller leaves.

Three interleaved 31-leaf tree rounds took `58.6 / 54.2 / 53.6` seconds with
OPT5 and `67.7 / 65.5 / 68.9` seconds with the direct path. Median times were
54.16 and 67.65 seconds respectively, a 1.25x complete-tree speedup. That fixed
benchmark produced identical 61-node tree structures.

### Temporary memory

The full root histogram's traced peak temporary allocation was 79.7 MB for the
vectorized path and 10.3 MB for the direct path. This is bounded working memory,
not retained model state or a leak. At the current
`_HISTOGRAM_BLOCK_NNZ = 1_000_000`, the observed vectorized peak is roughly 80
bytes per permitted source nonzero in a block. Lowering the private block budget
should reduce that component approximately linearly while increasing the number
of aggregation blocks.

The 40,385-feature check is still required because its complete matrix contains
about 34.8 million stored values. The design strongly suggests that peak memory
will remain governed by the one-million-entry block budget, local total-bin
arrays, and row metadata rather than total matrix nonzeros, but that argument is
not a substitute for the specified measurement.

## Flattened key definition

For a stored non-default entry with original feature `j`, local feature slot
`q`, and decoded bin `b`, define:

```text
key = layout.bin_offsets[q] + b
```

Keys range from zero through `layout.bin_offsets[-1] - 1`. Three `np.bincount`
operations produce gradient sums, Hessian sums, and counts over this shared
layout.

## Recommended sparse algorithm

Build `layout.feature_to_local` once per tree. It maps every original feature
index to its local feature slot, or `-1` when that feature is not in the tree.

For a leaf row array:

1. calculate `leaf_gradients = gradients[rows]` and corresponding Hessians;
2. take row blocks from `data.csr`, preserving row order and repetitions;
3. derive one row-position value per stored entry from the sliced CSR `indptr`;
4. map each stored original column through `feature_to_local`;
5. discard entries whose mapped slot is `-1`;
6. widen encoded bin data to `int64`, subtract one, and validate bounds in the
   checked wrapper rather than in every block;
7. construct flattened keys from local offsets and decoded bins;
8. aggregate block contributions with `np.bincount(..., minlength=total_bins)`
   and add them to the output arrays.

Conceptually, the kernel is:

```python
local_columns = layout.feature_to_local[block.indices]
keep = local_columns >= 0
row_positions = repeat(arange(block_rows), diff(block.indptr))
keys = (
    layout.bin_offsets[local_columns[keep]]
    + decoded_bins[keep]
)

gradient_hist += np.bincount(
    keys,
    weights=block_gradients[row_positions[keep]],
    minlength=layout.total_bins,
)
```

Apply the same pattern for Hessians. Counts use unweighted `np.bincount` and are
stored as exact `int64` values.

## Bound temporary memory

Do not create row-position, key, and weight arrays for the entire root of the
40,385-feature matrix at once. Root-level temporary arrays can otherwise be
hundreds of megabytes.

Process consecutive leaf-row blocks under a fixed private temporary nonzero
budget. A starting target of about one million source CSR entries per block is
reasonable, but choose the final constant from measured peak memory. Determine
block boundaries from CSR row nonzero counts; a single unusually dense row may
exceed the target and should be processed alone.

The budget is an internal implementation constant, not a new estimator
hyperparameter. Every block must be processed in deterministic row order.

`np.bincount(minlength=total_bins)` allocates a full local histogram per block.
Reuse or promptly release block temporaries. If repeated full-length allocations
dominate, compare a fixed output with `np.add.at`; choose based on profiling, not
assumption. The correctness oracle remains the existing direct implementation.

## Add implicit default-bin contributions

Sparse data stores only bins different from each feature's default. After all
explicit entries are aggregated:

1. compute the leaf's total gradient, Hessian, and count;
2. reduce each feature's local histogram segment to its represented totals;
3. calculate each local default offset as
   `layout.bin_offsets[:-1] + layout.default_bins`;
4. add `leaf_total - represented_total` to every feature's default bin.

Segment totals can be calculated with `np.add.reduceat` because every feature
has at least one bin and local offsets are a valid prefix sum. Handle the
zero-feature layout before calling `reduceat`.

Repeated row indices must retain their existing multiplicity semantics. CSR row
slicing with a repeated row array naturally repeats its stored entries, while
`gradients[rows]`, `hessians[rows]`, and the leaf count repeat the corresponding
statistics.

## Numerical behavior

`np.bincount` may accumulate floating-point contributions in a different order
from the old CSC/`np.add.at` implementation. Requirements are:

- counts are exactly equal;
- gradient and Hessian bins agree within a tight scale-aware tolerance;
- repeated runs of the new algorithm are exactly deterministic;
- split tie-breaking remains unchanged;
- fixed non-tie fixtures produce identical trees.

Do not round histograms to force equality. Report genuine near-ties where a
changed accumulation order selects a different split.

## Reference path and tests

Keep the old per-feature algorithm as a private test oracle or move an exact
copy into the test suite. It need not remain selectable through the estimator
API.

Test:

- empty rows and empty feature layouts;
- all-default sparse matrices;
- one feature and one bin;
- several selected and excluded features;
- default bins at the first, middle, and last positions;
- repeated and unsorted leaf-row arrays;
- compact encoded dtypes from optimization 2;
- row blocks small enough to force multiple aggregation passes;
- dense/CSR/CSC logical parity after binning;
- randomized sparse fixtures against the direct oracle.

For every comparison, verify per-feature segment totals equal the leaf's total
gradient, Hessian, and row count.

## Benchmark

With mapper fitting excluded, benchmark root and smaller-leaf histograms on the
5,000-feature data using the same sampled features. Report:

- rows and source/selected nonzeros per leaf;
- local total bins;
- direct and vectorized histogram times;
- maximum absolute gradient/Hessian difference;
- peak temporary memory;
- complete 31-leaf tree time.

Repeat a root benchmark on the 40,385-feature matrix before running a complete
tree, to confirm the temporary-memory bound.

## Acceptance criteria

- No Python loop iterates over selected features or individual samples in the
  optimized kernel; a bounded loop over row blocks is allowed.
- Counts exactly match the direct oracle and floating statistics meet the
  documented tolerance.
- Implicit default bins and repeated rows retain their semantics.
- Peak temporary memory remains bounded on the full chosen representation.
- The vectorized path materially improves histogram and 31-leaf tree time.
- The sparse matrix is never densified.
