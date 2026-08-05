# LiteLightGBM optimization 4: validated internal hot paths

## Status

Complete. `fit_tree` now validates immutable tree inputs once into a private
trusted context and uses private histogram, split-search, and row-partition
kernels during leaf growth. The directly callable public helpers retain their
checked wrappers, including finite derivative, sparse-storage, index, split,
and histogram validation.

## Objective

Stop repeating expensive structural validation inside every leaf operation while
preserving the strict behavior of the directly callable helper functions. The
current `build_histogram`, `find_best_split`, and `partition_rows` routines
revalidate mapper metadata, sparse storage, indices, and numeric arrays on every
call. During a 31-leaf tree these checks are repeated dozens of times over data
that was already validated by `fit_tree`.

This optimization changes internal call structure, not public behavior. It must
not add a scikit-learn dependency.

## Design principle

Use a validating wrapper and a private trusted implementation:

```text
direct caller -> public function -> normalize and validate -> private kernel
fit_tree      -> validate once at entry -------------------> private kernel
```

Do not delete validation merely to improve a benchmark. Direct calls with
malformed inputs must continue raising the documented `TypeError`, `ValueError`,
or fitted-state error instead of causing out-of-bounds writes or obscure SciPy
failures.

## Normalized tree context

Create one internal context after `fit_tree` validates its arguments. It should
contain only data reused by leaf operations, such as:

- canonical CSR and CSC references;
- sample and feature counts;
- normalized `float64` gradient and Hessian arrays;
- normalized `int64` tree row indices;
- the tree's `HistogramLayout` from optimization 3;
- validated mapper defaults and bin counts used for routing;
- normalized tree constraints and regularization values.

A private dataclass is appropriate if it replaces repeated argument plumbing.
Do not copy the sparse matrices or full gradient arrays into the context.

## Functions to split

### Histogram construction

Keep the existing `build_histogram(...)` signature as the checked entry point.
After validation, call a private kernel such as:

```python
def _build_histogram_validated(
    context: _TreeContext,
    rows: np.ndarray,
) -> Histogram:
    ...
```

The private function may assume that rows are signed integer indices in range,
the layout is valid, encoded sparse bins are valid, and gradients/Hessians are
finite. It must still implement the same aggregation and default-bin arithmetic.

### Split search

Keep `find_best_split(...)` as the validating wrapper. Move prefix sums, child
constraint masks, gain calculation, and deterministic tie-breaking into
`_find_best_split_validated(...)`. Histograms produced internally may go directly
to that kernel.

### Row partitioning

Keep `partition_rows(...)` validating direct calls. Tree growth should use
`_partition_rows_validated(...)`, with a validated split and canonical CSC
storage. Preserve row order and duplicate-row behavior.

Do not optimize `predict_tree_raw` in this step unless profiling shows its
validation is material after the three training kernels above are fixed. If it
is optimized, retain the same wrapper/kernel pattern and use the private path
only for freshly built or already validated trees.

## Validation ownership

Each invariant must have one clear owner on the internal training path:

| Invariant | Validate at |
|---|---|
| Matrix shapes and canonical sparse storage | binned-data creation / tree entry |
| Mapper and local-layout consistency | layout construction / tree entry |
| Gradients and Hessians are finite and row-aligned | tree entry |
| Root rows and sampled features are valid | tree entry |
| Child rows | produced by the trusted partition kernel |
| Histogram shape and finiteness | guaranteed by trusted histogram kernel |
| Split belongs to the layout and threshold range | trusted split kernel output |
| User-supplied direct helper arguments | each public wrapper |

Use comments to mark private functions as accepting validated internal state.
Their leading underscore is not a substitute for establishing the invariants
at the caller.

## Required non-changes

- Do not relax validation of public helper functions.
- Do not change exception classes or primary error messages without a reason.
- Do not change histogram accumulation order in this optimization.
- Do not change split tie-breaking, sampling, tree growth, or prediction math.
- Do not add configuration flags for validation.
- Do not use assertions for user-facing validation; Python can disable them.

## Verification

First run or create malformed-input tests against each public helper. Include:

- fractional, negative, duplicate, and out-of-range indices;
- incompatible mapper arrays and offsets;
- encoded sparse zero and out-of-range bins;
- non-finite gradients and Hessians;
- invalid split feature, threshold, and default direction;
- histogram arrays with inconsistent lengths or invalid counts.

These tests must pass unchanged after introducing private kernels.

Then compare the public and private paths on valid dense and sparse fixtures:

- histogram arrays are exactly equal;
- selected splits and their sufficient statistics are equal;
- row partitions preserve identical contents and order;
- complete seeded trees and raw predictions are equal.

Add a focused call-count test or lightweight instrumentation showing that
mapper/sparse metadata normalization occurs once per tree on the internal path,
not once per leaf.

## Benchmark

Benchmark one 31-leaf tree with the same binned data, rows, features, gradients,
and Hessians before and after. Measure the entire `fit_tree` call and, if
possible, separate histogram, split-search, and partition times. Do not include
mapper fitting.

## Acceptance criteria

- All existing checked helper behavior remains available to direct callers.
- The estimator's training path validates immutable tree inputs once and uses
  trusted kernels afterward.
- Histograms, split sequence, tree nodes, and predictions remain identical.
- Profiling confirms that repeated metadata validation is removed from the
  leaf-growth loop.
- The implementation remains deterministic and uses only NumPy, SciPy, and the
  standard library.
