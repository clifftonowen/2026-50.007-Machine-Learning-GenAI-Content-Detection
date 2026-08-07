# LiteLightGBM optimization 7: bounded histogram caching

## Status

Implemented and verified. `fit_tree` now creates a private per-tree,
byte-bounded LRU cache for queued live-leaf histograms. Cache hits retain the
OPT6 subtraction path; evicted, oversized, and zero-budget entries fall back
to direct construction of both child histograms without changing candidate
eligibility or the public estimator API.

Focused cache checks covered zero, exact-fit, oversized, replacement, LRU
eviction, and `take` accounting. A fixed non-tie 8-leaf tree produced the same
topology in checks with zero, one-entry, and generous cache budgets; predictions
matched within a tight numerical tolerance. Tie-prone cases may differ across
budgets because of floating-point accumulation order.
The 256 MiB default remains a provisional cap: the required target-scale
budget sweep has not yet selected a runtime/memory plateau. A fixed budget is
deterministic, but changing it can alter tie-prone trees because direct child
construction and histogram subtraction accumulate floating statistics in a
different order.

## Objective

Bound the memory consumed by live-leaf histograms introduced for histogram
subtraction. Leaf-wise growth can keep many candidates in its priority queue;
an unbounded `node_index -> Histogram` dictionary can therefore retain one
large histogram for every live leaf.

The cache is an internal training optimization. Cache residency must not change
the fitted model format or estimator API, and it must not decide candidate
eligibility. Repeated fits with the same cache budget are exactly deterministic;
changing budgets may alter floating fitted statistics and tie-prone trees. The
module must not use scikit-learn.

## Cache contents and key

Use the tree node index as the cache key. A cache entry contains the histogram
used to calculate that node's queued `SplitInfo`. All entries within one tree
share the same `HistogramLayout`, so the node index is sufficient; never share
entries between trees or boosting iterations.

The exact entry size is:

```text
histogram.gradient_sums.nbytes
+ histogram.hessian_sums.nbytes
+ histogram.counts.nbytes
```

Do not use `sys.getsizeof` for the main accounting because it does not represent
NumPy buffer size reliably. Layout metadata is shared per tree and does not need
to be charged to every entry. The optional scalar accumulation-scale metadata
used by OPT6 subtraction is likewise construction-only and has no material
per-entry buffer charge.

## Internal cache type

Implement a small private LRU cache with `collections.OrderedDict`, for example:

```python
class _HistogramCache:
    def __init__(self, max_bytes: int): ...
    def put(self, node_index: int, histogram: Histogram) -> bool: ...
    def take(self, node_index: int) -> Histogram | None: ...
    @property
    def current_bytes(self) -> int: ...
```

Required behavior:

- `put` evicts least-recently-used entries until the new entry fits;
- an entry larger than the complete budget is not stored;
- replacing an existing key first removes its old byte charge;
- `take` returns and removes an entry, immediately freeing its charge;
- `0 <= current_bytes <= max_bytes` always holds;
- the cache is created inside `fit_tree` and discarded after that tree;
- byte arithmetic uses Python integers to avoid NumPy integer overflow.

Tree growth normally consumes entries only once, so insertion order and LRU
order will often be similar. LRU remains preferable because tests or later tree
logic may inspect a cached candidate before it is selected.

## Budget

Start with a private module constant rather than expanding the estimator's
public hyperparameters. A value such as 256 MiB is a reasonable initial
benchmark candidate, but choose the committed value after measuring the target
environment. Give tests a private way to instantiate the cache with a small
budget.

The byte limit bounds retained cache buffers, not every transient allocation in
histogram construction. Peak process memory may temporarily include the cache,
one directly built histogram, one derived histogram, and vectorization block
buffers. Measure this explicitly.

## Integration with tree growth

When a child has a valid split, enqueue its `SplitInfo` regardless of whether
its histogram fits in the cache. Cache residency must never decide whether a
leaf is eligible to split.

When a queued candidate is popped:

### Cache hit

1. `take` its parent histogram;
2. build the smaller child directly;
3. derive the larger child by subtraction;
4. search both children and attempt to cache their histograms.

### Cache miss

1. partition the rows normally;
2. build both child histograms directly;
3. search both children and attempt to cache their histograms.

Do not rebuild the missing parent plus one child: that still requires two
histogram builds and scans the complete parent rows in one of them. Building
the two children directly processes no more row membership than the parent in
total and uses the established correctness path.

Remove any cached parent before constructing children so its memory is
available. Histograms for terminal children or children at the depth/leaf limit
must not enter the cache.

## Correctness and determinism

Eviction changes only how a child histogram is obtained. Cache residency must
not change candidate eligibility, the public estimator API, or fitted model
format. Direct construction and subtraction can
have slightly different floating-point accumulation order: repeated runs with
the same cache budget must be exactly deterministic, while changing budgets may
alter floating fitted statistics and tie-prone tree topology or predictions.
The test suite should use a fixed non-tie fixture for exact split topology and
compare predictions within a tight numerical tolerance.

Optional private counters for hits, misses, evictions, and avoided histogram
builds are useful for tests and benchmarks. Do not publish them as fitted model
attributes unless project reporting actually requires them.

## Verification

Unit-test the cache independently:

- zero-byte budget;
- one entry exactly equal to the budget;
- an entry larger than the budget;
- eviction of the least-recently-used key;
- replacement of an existing key;
- `take` removes the byte charge;
- byte accounting never exceeds the limit.

Tree-level tests should fit the same fixed tree with:

- a budget large enough for all live histograms;
- a budget that holds exactly one histogram;
- a zero-byte budget, forcing direct construction after every miss.

On a fixed non-tie fixture, all three should produce equivalent tree topology
and predictions within a tight numerical tolerance. Tie-prone fixtures may
differ across budgets, so record floating-statistic and prediction differences;
repeated fits for each fixed budget must remain deterministic. Instrumented call
counts should show that a zero budget behaves like direct child building, while
a generous budget receives the subtraction benefit.

## Benchmark

On a fixed 31-leaf tree, sweep a small set of internal budgets such as 0, 64,
128, and 256 MiB. Report:

- cache hits, misses, and evictions;
- direct histogram build count;
- peak `current_bytes`;
- peak process memory;
- total tree time;
- tree/prediction parity on a fixed non-tie fixture within a tight numerical
  tolerance, with any tie-prone differences recorded.

Repeat the chosen budget on the supplied 5,000-feature representation and one
representative root/tree run from the 40,385-feature representation. Select the
smallest budget after which extra memory produces little additional runtime
benefit.

## Acceptance criteria

- Cache-accounted bytes never exceed the configured private limit.
- Cache misses fall back to correct direct construction and never remove a
  valid split candidate.
- Repeated seeded fits with the same budget are deterministic; across budgets,
  cache misses preserve candidate eligibility and the public estimator API even
  when floating statistics or tie-prone trees differ.
- Cache state is released after each tree and is absent from fitted models.
- Target-scale budget selection remains pending; the 256 MiB default is
  provisional until runtime and memory measurements identify an acceptable
  plateau on the target dataset.
- The module remains deterministic, sparse, and free of scikit-learn imports.

