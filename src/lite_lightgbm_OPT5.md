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

