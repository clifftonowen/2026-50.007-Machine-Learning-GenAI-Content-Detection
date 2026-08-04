# LiteLightGBM optimization 7: bounded histogram caching

## Objective

Bound the memory consumed by live-leaf histograms introduced for histogram
subtraction. Leaf-wise growth can keep many candidates in its priority queue;
an unbounded `node_index -> Histogram` dictionary can therefore retain one
large histogram for every live leaf.

The cache is an internal training optimization. It must not change the fitted
model format, estimator API, split ordering, or predictions, and it must not use
scikit-learn.

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
to be charged to every entry.

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

Eviction changes only how a child histogram is obtained. It must not change:

- the priority queue and its tie fields;
- candidate eligibility;
- sampled rows or features;
- original feature indices;
- leaf values or shrinkage;
- prediction traversal.

Direct construction and subtraction can have slightly different floating-point
accumulation order. The test suite should use non-tie fixtures for exact split
topology and tight tolerances for floating statistics. Repeated runs with the
same cache budget must be exactly deterministic.

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

All three must produce equivalent tree topology and predictions. Instrumented
call counts should show that a zero budget behaves like direct child building,
while a generous budget receives the subtraction benefit.

## Benchmark

On a fixed 31-leaf tree, sweep a small set of internal budgets such as 0, 64,
128, and 256 MiB. Report:

- cache hits, misses, and evictions;
- direct histogram build count;
- peak `current_bytes`;
- peak process memory;
- total tree time;
- tree/prediction parity.

Repeat the chosen budget on the supplied 5,000-feature representation and one
representative root/tree run from the 40,385-feature representation. Select the
smallest budget after which extra memory produces little additional runtime
benefit.

## Acceptance criteria

- Cache-accounted bytes never exceed the configured private limit.
- Cache misses fall back to correct direct construction and never remove a
  valid split candidate.
- Generous, constrained, and zero budgets produce equivalent seeded models.
- Cache state is released after each tree and is absent from fitted models.
- The selected default gives a measured runtime benefit without unacceptable
  peak memory on the target dataset.
- The module remains deterministic, sparse, and free of scikit-learn imports.

