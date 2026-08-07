# LiteLightGBM optimization 6: histogram subtraction

## Status

**Algorithmic implementation and project-scale parity complete; total-speed gain
modest.** `fit_tree`
retains construction-only histograms for queued live leaves, directly builds
the smaller child (breaking equal sizes toward the left), and derives the
sibling by layout-safe subtraction. Derived counts are exact. Every
zero-count bin is normalized to exactly zero gradient and Hessian statistics,
but only after subtraction validates that its raw residual is within the
documented scale-aware `128 * eps` tolerance; material positive or negative
inconsistencies still raise. Builder-created histograms carry optional finite
absolute accumulation scales (the sum of per-row absolute contributions), so
default-bin residuals are bounded by operand histories rather than signed leaf
totals. Derived histograms conservatively propagate both operand scales;
four-field hand-built histograms retain the per-bin fallback. Histograms are
released during construction and are never retained by `DecisionTree`.

The private `_fit_tree(..., use_histogram_subtraction=False)` path preserves a
direct-both-child oracle for regression checks without changing the public
`fit_tree` API. Focused direct-versus-subtraction checks verify exact counts,
close floating statistics, deterministic repeated fits, and one direct
child-histogram build per expandable accepted split. Synthetic near-tie cases
differed in topology in 116 of 300 cases and had six label flips; the dedicated
128-leaf and 1,200-leaf tests showed no accumulated histogram-statistic drift.
The implementation
deliberately retains its exact existing gain comparison and tie-break policy.

An independently reproduced first-tree diagnostic on the supplied 20,000 x 5,000
CSR representation (`num_leaves=149`, `colsample_bytree=0.5198042692950159`,
`min_child_samples=7`, `reg_alpha=0.34104824737458306`,
`reg_lambda=0.13010318597055903`, `learning_rate=0.034494`, balanced weights,
`random_state=42`) completed all 297 nodes and 149 leaves with 138 subtraction
operations. Exactly one zero-count gradient residual required the wider bound:
it measured `117.75 * eps` with a unit scale floor. This is measured first-tree
evidence for the subtraction guard, not a full-ensemble runtime estimate.

### Project-scale benchmark record (reported 2026-08-06)

On the supplied 20,000 by 5,000 representation with `num_leaves=31`, three
tree fits reported 30 histogram builds with subtraction versus 59 directly.
Total tree times were 60.58, 58.93, and 58.60 seconds with subtraction
(median 58.93 seconds), versus 61.17, 60.85, and 59.75 seconds directly
(median 60.85 seconds, 1.03x). Histogram construction was therefore halved,
but the total-runtime improvement is modest. The project-scale direct and
subtraction trees were bit-identical. Each local-layout histogram used about
9.03 MB and the observed peak was 17 live histograms (about 153 MB). The
unbounded OPT6 mapping has a worst case of about 271 MB at 31 leaves, with an
estimated 2.3 GB at 255 leaves; it must be bounded by OPT7 before larger-scale
use.

## Objective

Build only one child histogram after a split and derive the other from its
parent:

```text
larger_child = parent - smaller_child
```

The current tree builder constructs both child histograms directly. For a
31-leaf tree that means one root histogram followed by nearly two histogram
builds per accepted split. Subtraction should reduce the expensive construction
work to approximately one root plus one child build per split.

This optimization assumes tree-local layouts and vectorized histogram building
from optimizations 3-5. It must not use scikit-learn.

## Required invariant

For every bin in the tree-local layout:

```text
parent statistics = left statistics + right statistics
```

This includes explicit non-default entries and implicit default-bin
contributions. Both children must use the exact same `HistogramLayout` as their
parent. Never subtract histograms created with different sampled features,
offsets, or bin ordering.

## Retain histograms for live leaves

The current priority queue retains only `SplitInfo`. Add an internal mapping:

```text
node_index -> histogram used to calculate that node's SplitInfo
```

Optimization 6 may use an unbounded dictionary as the first correct
implementation. Optimization 7 replaces it with a byte-bounded cache. Do not
put histograms in trained `DecisionTree` objects; they are construction-only
state and must be released when `fit_tree` returns.

The root histogram is stored when its split is enqueued. A child histogram is
stored only if the child has a valid split and can therefore later be popped
from the queue. Histograms for terminal children can be released after split
search.

## Split procedure

When a candidate leaf is popped:

1. retrieve and remove its parent histogram from the live mapping;
2. partition its rows using the chosen `SplitInfo`;
3. choose the child with fewer rows as the direct-build child; break equal row
   counts toward the left child for determinism;
4. build that child's histogram with the normal optimized kernel;
5. derive the other child's histogram by subtraction;
6. find the best split for each eligible child;
7. retain each child's histogram only when its split is enqueued.

Row count is the initial definition of "smaller." Estimating selected nonzeros
could sometimes choose a cheaper sparse build, but it adds another scan and is
outside the first implementation. Profile before changing this rule.

Do not build either child when the leaf budget has been reached or the children
are already at the depth limit, matching the current early exits.

## Subtraction helper

Add one private helper:

```python
def _subtract_histograms(
    parent: Histogram,
    child: Histogram,
) -> Histogram:
    ...
```

It must verify or rely on the trusted caller to guarantee identical layout and
array lengths. Calculate:

- gradients with `float64` subtraction;
- Hessians with `float64` subtraction;
- counts with signed `int64` subtraction.

Counts must never be negative. A negative count indicates a logic/layout error
and must raise rather than be clipped.

Gradients may legitimately be negative, so never clamp them. Hessians are
non-negative, but roundoff can produce a very small negative difference when
the directly built child nearly equals the parent. For zero-count bins, use a
scale-aware tolerance based on the corresponding parent/child magnitudes plus
their optional absolute accumulation scales, with a unit floor; this
implementation uses `128 * eps`. Clamp only Hessians in `[-tolerance, 0)` to
zero; raise for a more negative result. For populated bins, retain the local
operand scale for the negativity check so a large default-bin history cannot
mask a materially inconsistent parent/child relationship. Document the chosen
tolerance in code.

After subtraction, verify in tests rather than the production hot path that:

```text
derived + direct_child ~= parent
```

for gradients and Hessians, and exactly for counts.

## Fallback behavior

In optimization 6, every enqueued candidate is expected to have a retained
histogram. If that invariant is broken, fail clearly during development rather
than silently using an unrelated histogram. Optimization 7 deliberately adds a
cache-miss fallback.

Keep a direct-both-children tree path in tests so complete tree behavior can be
compared. A private test switch or separate test helper is preferable to a new
public estimator parameter.

## Verification

Histogram-level tests must cover:

- left smaller, right smaller, and equal child counts;
- all-default sparse features;
- bins populated only by one child;
- zero-gradient and zero-Hessian bins;
- mixed positive and negative gradient sums;
- compact bin dtypes;
- a deliberate negative count, which must fail;
- harmless negative Hessian roundoff and a materially negative Hessian.

For randomized leaves:

1. build parent, left, and right directly;
2. derive each side in turn from the parent and its sibling;
3. require exact count equality and close floating arrays;
4. compare the resulting `SplitInfo` objects.

At the tree level, compare direct and subtraction growth with fixed rows,
features, gradients, and Hessians. Require identical node topology, features,
threshold bins, default directions, and sample counts. Leaf values, gains, and
raw predictions should be equal within a tight tolerance. Repeated subtraction
runs must be deterministic.

## Benchmark

Instrument histogram kernel calls for one 31-leaf tree:

- direct construction should use approximately `1 + 2 * accepted_splits`
  builds, subject to the existing leaf/depth early exits;
- subtraction should use approximately `1 + accepted_splits` builds.

Report call counts, histogram time, total tree time, and peak live histogram
bytes. Use the same binned data, gradients, rows, features, and configuration.

## Acceptance criteria

- Exactly one child histogram is directly constructed for every expandable
  accepted split.
- Derived counts are exact and floating statistics agree with direct
  construction within the specified tolerance.
- Split ordering and predictions remain deterministic.
- Direct and subtraction tree paths agree on non-tie fixtures.
- Construction histograms do not become part of the fitted model.
- The 31-leaf benchmark shows a material reduction in histogram work; total
  tree time is measured and reported, with the current project-scale gain
  remaining modest.

