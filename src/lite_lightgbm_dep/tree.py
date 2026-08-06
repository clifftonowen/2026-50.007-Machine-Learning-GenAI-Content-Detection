"""Histogram tree building and traversal for LiteLightGBM."""

from __future__ import annotations

import operator
import heapq
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .binning import BinnedDataset, BinMapper
from .core import EPSILON, LiteLightGBMConfig, soft_threshold


# Histogram construction is bounded by source CSR entries rather than by the
# number of rows or features.  A single unusually dense row is still handled
# as one block, so this is only a target for temporary arrays, not a hard
# restriction on input rows.
_HISTOGRAM_BLOCK_NNZ = 1_000_000

# The cache retains construction-only histogram buffers for queued candidates.
# This is intentionally private so the estimator's public configuration and
# fitted model format remain unchanged.
# The supplied 20,000 x 5,000 OPT6 profile measured about 9.03 MB per local
# histogram and 17 live histograms (about 153 MB); no target budget sweep
# justifies a smaller default, so retain this conservative provisional 256 MiB
# private cap pending a target-scale OPT7 budget measurement. Because direct-
# child aggregation and parent subtraction use different floating-point
# accumulation orders, changing this private budget can change tie-prone trees;
# repeated fits with one fixed budget remain deterministic.
_HISTOGRAM_CACHE_MAX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class HistogramLayout:
    """Immutable tree-local flattened histogram layout.

    ``feature_indices`` contains sorted original feature numbers while the
    remaining arrays describe the compact local histogram segments.  The
    reverse lookup uses ``-1`` for features not selected for this tree.
    Arrays are copied and marked read-only so every histogram in a tree can
    safely share one layout object without accidental mutation.
    """

    feature_indices: np.ndarray
    n_bins: np.ndarray
    default_bins: np.ndarray
    bin_offsets: np.ndarray
    feature_to_local: np.ndarray

    def __post_init__(self) -> None:
        try:
            features = np.asarray(self.feature_indices)
            n_bins = np.asarray(self.n_bins)
            defaults = np.asarray(self.default_bins)
            offsets = np.asarray(self.bin_offsets)
            feature_to_local = np.asarray(self.feature_to_local)
        except (TypeError, ValueError) as exc:
            raise ValueError("histogram layout must contain numeric arrays") from exc

        arrays = (
            (features, "feature_indices"),
            (n_bins, "n_bins"),
            (defaults, "default_bins"),
            (offsets, "bin_offsets"),
            (feature_to_local, "feature_to_local"),
        )
        for values, name in arrays:
            if values.ndim != 1:
                raise ValueError(f"histogram layout {name} must be one-dimensional")
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"histogram layout {name} must be integer-valued")

        if features.size and np.any(features < 0):
            raise ValueError("histogram layout feature indices must be non-negative")
        if features.size > 1 and np.any(features[1:] <= features[:-1]):
            raise ValueError(
                "histogram layout feature indices must be sorted and unique"
            )
        if n_bins.size != features.size or defaults.size != features.size:
            raise ValueError("histogram layout feature arrays have incompatible sizes")
        if offsets.size != features.size + 1 or int(offsets[0]) != 0:
            raise ValueError("histogram layout bin offsets have incompatible size")
        if np.any(offsets[1:] < offsets[:-1]) or np.any(n_bins <= 0):
            raise ValueError("histogram layout bins must be a positive prefix sum")
        if np.any(np.diff(offsets) != n_bins):
            raise ValueError("histogram layout offsets must match n_bins")
        if np.any(defaults < 0) or np.any(defaults >= n_bins):
            raise ValueError("histogram layout default bins are outside bin ranges")
        if feature_to_local.size == 0 and features.size:
            raise ValueError("histogram layout reverse lookup is too small")
        if features.size and np.any(features >= feature_to_local.size):
            raise ValueError("histogram layout feature index is out of range")

        # Preserve a valid signed dtype (the factory uses int32 below the
        # int32 feature-count limit) while ensuring unsigned inputs are
        # normalized to a signed dtype before checking the ``-1`` sentinel.
        # Unselected entries must remain ``-1``.
        if np.issubdtype(feature_to_local.dtype, np.signedinteger):
            reverse = np.array(feature_to_local, copy=True)
        else:
            if np.any(feature_to_local > np.iinfo(np.int64).max):
                raise ValueError(
                    "histogram layout reverse lookup must fit signed int64"
                )
            reverse = np.asarray(feature_to_local, dtype=np.int64)
        if np.any(reverse < -1) or np.any(reverse >= features.size):
            raise ValueError("histogram layout reverse lookup contains invalid slots")
        expected_reverse = np.full(reverse.size, -1, dtype=np.int64)
        expected_reverse[features] = np.arange(features.size, dtype=np.int64)
        if not np.array_equal(reverse, expected_reverse):
            raise ValueError("histogram layout reverse lookup does not match features")

        normalized = (
            np.asarray(features, dtype=np.int64),
            np.asarray(n_bins, dtype=np.int64),
            np.asarray(defaults, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
            reverse,
        )
        for name, values in zip(
            ("feature_indices", "n_bins", "default_bins", "bin_offsets", "feature_to_local"),
            normalized,
        ):
            values = np.array(values, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def _make_histogram_layout(
    mapper: BinMapper,
    feature_indices: np.ndarray,
) -> HistogramLayout:
    """Construct one compact immutable layout for a tree's feature set."""
    try:
        n_features = len(mapper.n_bins)
        raw_features = np.asarray(feature_indices)
        raw_n_bins = np.asarray(mapper.n_bins)
        raw_defaults = np.asarray(mapper.default_bins)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("mapper and feature indices must contain valid arrays") from exc
    if raw_features.ndim != 1 or (
        raw_features.size and not np.issubdtype(raw_features.dtype, np.integer)
    ):
        raise ValueError("feature_indices must be a one-dimensional integer array")
    features = np.asarray(raw_features, dtype=np.int64)
    if features.size and (np.any(features < 0) or np.any(features >= n_features)):
        raise ValueError("feature_indices contains an out-of-range feature")
    if features.size > 1 and np.any(features[1:] <= features[:-1]):
        raise ValueError("feature_indices must be sorted and unique")

    try:
        n_bins = np.asarray(raw_n_bins, dtype=np.int64)
        defaults = np.asarray(raw_defaults, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mapper bin metadata must be integer-valued") from exc
    if n_bins.size != n_features or defaults.size != n_features:
        raise ValueError("mapper metadata has incompatible feature dimensions")
    selected_n_bins = np.asarray(n_bins[features], dtype=np.int64)
    selected_defaults = np.asarray(defaults[features], dtype=np.int64)
    local_offsets = np.zeros(features.size + 1, dtype=np.int64)
    if selected_n_bins.size:
        local_offsets[1:] = np.cumsum(selected_n_bins, dtype=np.int64)

    reverse_dtype = np.int32 if n_features < np.iinfo(np.int32).max else np.int64
    reverse = np.full(n_features, -1, dtype=reverse_dtype)
    reverse[features] = np.arange(features.size, dtype=reverse_dtype)
    return HistogramLayout(
        feature_indices=features,
        n_bins=selected_n_bins,
        default_bins=selected_defaults,
        bin_offsets=local_offsets,
        feature_to_local=reverse,
    )


def _validate_layout_for_mapper(
    layout: HistogramLayout,
    n_features: int,
    mapper_n_bins: np.ndarray,
    mapper_defaults: np.ndarray,
) -> None:
    """Validate that a local layout is compatible with one mapper."""
    if not isinstance(layout, HistogramLayout):
        raise ValueError("histogram layout must be a HistogramLayout")
    features = layout.feature_indices
    if layout.feature_to_local.size != n_features:
        raise ValueError("histogram layout reverse lookup has wrong feature count")
    if features.size and np.any(features >= n_features):
        raise ValueError("histogram layout contains an out-of-range feature")
    if not np.array_equal(layout.n_bins, mapper_n_bins[features]):
        raise ValueError("histogram layout n_bins do not match mapper")
    if not np.array_equal(layout.default_bins, mapper_defaults[features]):
        raise ValueError("histogram layout default bins do not match mapper")
    if int(layout.bin_offsets[-1]) != sum(int(value) for value in layout.n_bins):
        raise ValueError("histogram layout final offset is inconsistent")


@dataclass(slots=True)
class Histogram:
    """Flattened per-bin gradient, Hessian, and exact-count statistics."""

    gradient_sums: np.ndarray
    hessian_sums: np.ndarray
    counts: np.ndarray
    layout: HistogramLayout | None = None


class _HistogramCache:
    """Private byte-bounded LRU cache for tree-local histograms."""

    def __init__(self, max_bytes: int):
        if isinstance(max_bytes, (bool, np.bool_)):
            raise ValueError("histogram cache byte budget must be a non-negative integer")
        try:
            normalized_max_bytes = operator.index(max_bytes)
        except TypeError as exc:
            raise ValueError(
                "histogram cache byte budget must be a non-negative integer"
            ) from exc
        if normalized_max_bytes < 0:
            raise ValueError(
                "histogram cache byte budget must be a non-negative integer"
            )
        self._max_bytes = int(normalized_max_bytes)
        self._entries: OrderedDict[int, tuple[Histogram, int]] = OrderedDict()
        self._current_bytes = 0

    @staticmethod
    def _entry_bytes(histogram: Histogram) -> int:
        """Return the Python-int size of the retained NumPy buffers."""
        return int(
            int(histogram.gradient_sums.nbytes)
            + int(histogram.hessian_sums.nbytes)
            + int(histogram.counts.nbytes)
        )

    @property
    def current_bytes(self) -> int:
        """Return the total bytes charged to retained histogram buffers."""
        return int(self._current_bytes)

    def put(self, node_index: int, histogram: Histogram) -> bool:
        """Retain ``histogram`` if it fits, evicting least-recent entries."""
        try:
            key = operator.index(node_index)
        except TypeError as exc:
            raise ValueError("histogram cache node index must be an integer") from exc
        entry_bytes = self._entry_bytes(histogram)

        # Replacing a key first releases its old charge, even when the new
        # histogram is too large to retain.
        old_entry = self._entries.pop(key, None)
        if old_entry is not None:
            self._current_bytes -= int(old_entry[1])

        if entry_bytes > self._max_bytes:
            return False

        while self._entries and self._current_bytes + entry_bytes > self._max_bytes:
            _, (_, evicted_bytes) = self._entries.popitem(last=False)
            self._current_bytes -= int(evicted_bytes)

        self._entries[key] = (histogram, entry_bytes)
        self._current_bytes += entry_bytes
        return True

    def take(self, node_index: int) -> Histogram | None:
        """Remove and return a cached histogram, or ``None`` on a miss."""
        try:
            key = operator.index(node_index)
        except TypeError as exc:
            raise ValueError("histogram cache node index must be an integer") from exc
        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        histogram, entry_bytes = entry
        self._current_bytes -= int(entry_bytes)
        return histogram


def _subtract_histograms(
    parent: Histogram,
    child: Histogram,
) -> Histogram:
    """Derive a sibling histogram by subtracting ``child`` from ``parent``.

    Histograms are tree-local, so subtraction is only valid when both objects
    point at the exact same immutable layout and contain equally shaped
    flattened arrays.  Floating statistics are widened before subtraction;
    exact row counts use signed ``int64`` arithmetic.  Empty-bin residuals are
    checked against scale-aware ``64 * eps`` bounds, while materially negative
    Hessian residuals in populated bins are rejected as a broken parent/child
    relationship.
    """
    if not isinstance(parent, Histogram) or not isinstance(child, Histogram):
        raise ValueError("histogram subtraction requires Histogram operands")
    if parent.layout is not child.layout:
        raise ValueError("histograms must share the exact same layout")

    try:
        parent_gradients = np.asarray(parent.gradient_sums)
        child_gradients = np.asarray(child.gradient_sums)
        parent_hessians = np.asarray(parent.hessian_sums)
        child_hessians = np.asarray(child.hessian_sums)
        parent_counts = np.asarray(parent.counts)
        child_counts = np.asarray(child.counts)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("histograms must contain numeric arrays") from exc

    arrays = (
        (parent_gradients, "parent gradient sums"),
        (child_gradients, "child gradient sums"),
        (parent_hessians, "parent Hessian sums"),
        (child_hessians, "child Hessian sums"),
        (parent_counts, "parent counts"),
        (child_counts, "child counts"),
    )
    for values, name in arrays:
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")

    expected_shape = parent_gradients.shape
    if (
        parent_hessians.shape != expected_shape
        or parent_counts.shape != expected_shape
        or child_gradients.shape != expected_shape
        or child_hessians.shape != expected_shape
        or child_counts.shape != expected_shape
    ):
        raise ValueError("histogram arrays must have identical one-dimensional shapes")

    try:
        parent_gradient_values = np.asarray(parent_gradients, dtype=np.float64)
        child_gradient_values = np.asarray(child_gradients, dtype=np.float64)
        parent_hessian_values = np.asarray(parent_hessians, dtype=np.float64)
        child_hessian_values = np.asarray(child_hessians, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("histogram gradients and Hessians must be numeric") from exc
    if (
        not np.isfinite(parent_gradient_values).all()
        or not np.isfinite(child_gradient_values).all()
        or not np.isfinite(parent_hessian_values).all()
        or not np.isfinite(child_hessian_values).all()
    ):
        raise ValueError("histogram gradients and Hessians must be finite")

    # Counts are sufficient statistics, not weights: require integer-valued
    # storage and widen compact unsigned/signed dtypes without allowing a
    # uint64 value to wrap into a negative int64 count.
    count_arrays = (
        (parent_counts, "parent counts"),
        (child_counts, "child counts"),
    )
    normalized_counts: list[np.ndarray] = []
    for values, name in count_arrays:
        if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.bool_
        ):
            raise ValueError(f"{name} must use an integer dtype")
        if np.issubdtype(values.dtype, np.unsignedinteger) and np.any(
            values > np.iinfo(np.int64).max
        ):
            raise ValueError(f"{name} must be representable as signed int64")
        normalized = np.asarray(values, dtype=np.int64)
        if np.any(normalized < 0):
            raise ValueError(f"{name} must not contain negative counts")
        normalized_counts.append(normalized)

    parent_count_values, child_count_values = normalized_counts
    derived_gradients = np.subtract(
        parent_gradient_values, child_gradient_values, dtype=np.float64
    )
    derived_hessians = np.subtract(
        parent_hessian_values, child_hessian_values, dtype=np.float64
    )
    derived_counts = np.subtract(
        parent_count_values, child_count_values, dtype=np.int64
    )
    if not np.isfinite(derived_gradients).all() or not np.isfinite(
        derived_hessians
    ).all():
        raise ValueError("histogram subtraction produced non-finite statistics")
    if np.any(derived_counts < 0):
        raise ValueError("histogram subtraction produced negative counts")
    empty_bins = derived_counts == 0

    # Directly built child and parent sums can differ by a few ulps even when
    # the child contains every row in a bin.  The optimized kernel computes
    # implicit default bins by subtracting represented sums from a leaf total,
    # so a nominally empty sibling can inherit roundoff at the scale of that
    # total rather than its tiny per-bin value.  Account for that accumulated
    # summation error with a modest epsilon multiple.  Empty-bin gradient and
    # Hessian residuals must remain within this bound before their statistics
    # are normalized away.  For populated bins, only a materially negative
    # Hessian signals a broken layout/partition relationship.
    # A unit floor is intentional: when both bin sums are near zero, the
    # residual came from subtracting leaf totals whose absolute scale is not
    # represented by this bin's tiny value.
    gradient_scale = np.maximum(
        np.abs(parent_gradient_values) + np.abs(child_gradient_values), 1.0
    )
    gradient_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * gradient_scale
    )
    hessian_scale = np.maximum(
        np.abs(parent_hessian_values) + np.abs(child_hessian_values), 1.0
    )
    hessian_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * hessian_scale
    )

    if np.any(
        empty_bins
        & (np.abs(derived_gradients) > gradient_tolerance)
    ):
        raise ValueError(
            "histogram subtraction produced materially non-zero gradients "
            "in zero-count bins"
        )
    if np.any(
        empty_bins
        & (np.abs(derived_hessians) > hessian_tolerance)
    ):
        raise ValueError(
            "histogram subtraction produced materially inconsistent Hessians "
            "in zero-count bins"
        )

    # Zero-count bins are required to have no statistics after the raw
    # residuals above have been validated.  This normalization is intentionally
    # count-aware; gradients in nonempty bins are never clamped.
    derived_gradients[empty_bins] = 0.0
    derived_hessians[empty_bins] = 0.0

    small_negative = (derived_hessians < 0.0) & (
        derived_hessians >= -hessian_tolerance
    )
    if np.any(small_negative):
        derived_hessians[small_negative] = 0.0
    if np.any(derived_hessians < -hessian_tolerance):
        raise ValueError("histogram subtraction produced materially negative Hessians")

    return Histogram(
        gradient_sums=derived_gradients,
        hessian_sums=derived_hessians,
        counts=derived_counts,
        layout=parent.layout,
    )


@dataclass(frozen=True, slots=True)
class SplitInfo:
    """Best valid split and its left/right sufficient statistics."""

    gain: float
    feature: int
    threshold_bin: int
    default_left: bool
    left_count: int
    right_count: int
    left_gradient: float
    left_hessian: float
    right_gradient: float
    right_hessian: float


@dataclass(slots=True)
class TreeNode:
    """One node in an array-backed decision tree."""

    depth: int
    value: float = 0.0
    feature: int = -1
    threshold_bin: int = -1
    default_left: bool = True
    left_child: int = -1
    right_child: int = -1
    sample_count: int = 0
    gradient_sum: float = 0.0
    hessian_sum: float = 0.0
    split_gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        """Return whether this node has no children."""
        return self.left_child < 0 and self.right_child < 0


@dataclass(slots=True)
class DecisionTree:
    """A trained tree and the feature subset used to build it."""

    nodes: list[TreeNode] = field(default_factory=list)
    feature_indices: np.ndarray | None = None


@dataclass(slots=True)
class _TreeContext:
    """Validated immutable inputs shared by one tree's trusted kernels.

    This is intentionally private: public helper calls continue to perform
    their full argument checks, while :func:`fit_tree` creates one context and
    reuses its canonical sparse views and normalized arrays for every leaf.
    """

    data: BinnedDataset
    csr: sp.csr_matrix
    csc: sp.csc_matrix
    mapper: BinMapper
    n_samples: int
    n_features: int
    n_bins: np.ndarray
    defaults: np.ndarray
    offsets: np.ndarray
    gradients: np.ndarray
    hessians: np.ndarray
    rows: np.ndarray
    features: np.ndarray
    layout: HistogramLayout
    num_leaves: int
    max_depth: int
    min_child_samples: int
    min_child_weight: float
    min_split_gain: float
    reg_alpha: float
    reg_lambda: float


def _normalize_tree_integer_metadata(values: np.ndarray, name: str) -> np.ndarray:
    """Normalize integer-valued metadata for the trusted tree context."""
    values = np.asarray(values)
    dtype = values.dtype
    if (
        not np.issubdtype(dtype, np.number)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.isfinite(values).all()
        or (
            np.issubdtype(dtype, np.floating)
            and np.any(values != np.trunc(values))
        )
    ):
        raise ValueError(f"{name} must be finite integer-valued numeric values")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as signed int64 values") from exc
    if np.issubdtype(dtype, np.unsignedinteger) and np.any(normalized < 0):
        raise ValueError(f"{name} must be representable as signed int64 values")
    if np.issubdtype(dtype, np.floating):
        with np.errstate(over="ignore", invalid="ignore"):
            round_trip = normalized.astype(dtype)
        if np.any(round_trip != values):
            raise ValueError(f"{name} must be representable as signed int64 values")
    return normalized


def _validate_tree_storage(
    data: BinnedDataset,
    n_samples: int,
    n_features: int,
) -> tuple[sp.csr_matrix, sp.csc_matrix, BinMapper, np.ndarray, np.ndarray, np.ndarray]:
    """Validate sparse storage and mapper metadata once at tree entry."""
    try:
        mapper = data.mapper
        csr = data.csr
        csc = data.csc
        raw_n_bins = np.asarray(mapper.n_bins)
        raw_defaults = np.asarray(mapper.default_bins)
        raw_offsets = np.asarray(mapper.bin_offsets)
        cut_points = mapper.cut_points
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("data must contain a valid binned dataset") from exc
    if (
        not sp.isspmatrix_csr(csr)
        or not sp.isspmatrix_csc(csc)
        or tuple(csr.shape) != (n_samples, n_features)
        or tuple(csc.shape) != (n_samples, n_features)
    ):
        raise ValueError("data sparse views must match their declared shape")
    if (
        raw_n_bins.ndim != 1
        or raw_n_bins.size != n_features
        or raw_defaults.ndim != 1
        or raw_defaults.size != n_features
        or raw_offsets.ndim != 1
        or raw_offsets.size != n_features + 1
    ):
        raise ValueError("data mapper has incompatible feature metadata")
    try:
        cut_count = len(cut_points)
    except (TypeError, ValueError) as exc:
        raise ValueError("mapper must contain one cut-point array per feature") from exc
    if cut_count != n_features:
        raise ValueError("mapper must contain one cut-point array per feature")

    n_bins = _normalize_tree_integer_metadata(raw_n_bins, "data mapper n_bins")
    defaults = _normalize_tree_integer_metadata(raw_defaults, "data mapper default bins")
    offsets = _normalize_tree_integer_metadata(raw_offsets, "data mapper bin offsets")
    if int(offsets[0]) != 0 or np.any(offsets < 0) or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("data mapper bin offsets must be a non-negative prefix sum")
    if np.any(n_bins <= 0) or np.any(np.diff(offsets) != n_bins):
        raise ValueError("data mapper bin offsets must match positive n_bins")
    if np.any(defaults < 0) or np.any(defaults >= n_bins):
        raise ValueError("data mapper default bins are outside feature bin ranges")
    expected_total = sum(int(value) for value in n_bins)
    if int(offsets[-1]) != expected_total:
        raise ValueError("data mapper final bin offset is inconsistent")
    for feature, cuts in enumerate(cut_points):
        try:
            normalized_cuts = np.asarray(cuts, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("mapper cut points must be numeric arrays") from exc
        if (
            normalized_cuts.ndim != 1
            or normalized_cuts.size != int(n_bins[feature]) - 1
            or not np.isfinite(normalized_cuts).all()
            or (
                normalized_cuts.size > 1
                and np.any(normalized_cuts[1:] < normalized_cuts[:-1])
            )
        ):
            raise ValueError("mapper cut points do not match n_bins")

    # Validate the row-oriented view's structural/canonical invariants once.
    # Histogram and routing kernels use CSC for indexed feature access, but a
    # malformed CSR companion must not be allowed to pass tree entry silently.
    try:
        csr_indptr = np.asarray(csr.indptr)
        csr_indices = np.asarray(csr.indices)
        csr_data = np.asarray(csr.data)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("data CSR view has invalid sparse storage") from exc
    if (
        csr_indptr.ndim != 1
        or csr_indptr.size != n_samples + 1
        or not np.issubdtype(csr_indptr.dtype, np.integer)
        or csr_indices.ndim != 1
        or csr_data.ndim != 1
        or csr_indices.size != csr_data.size
        or not np.issubdtype(csr_indices.dtype, np.integer)
    ):
        raise ValueError("data CSR view has invalid sparse storage")
    if csr_indptr.size and (
        np.any(csr_indptr < 0)
        or np.any(csr_indptr[1:] < csr_indptr[:-1])
        or int(csr_indptr[-1]) != int(csr_data.size)
    ):
        raise ValueError("data CSR row pointers are invalid")
    if csr_indices.size and (
        np.any(csr_indices < 0) or np.any(csr_indices >= n_features)
    ):
        raise ValueError("data CSR column indices are outside the dataset range")
    csr_encoded = _normalize_tree_integer_metadata(
        csr_data, "data CSR encoded bins"
    )
    if csr_encoded.size:
        csr_bin_limits = n_bins[csr_indices]
        if np.any(csr_encoded < 1) or np.any(csr_encoded > csr_bin_limits):
            raise ValueError("data CSR encoded bins are outside mapper ranges")
    # Compare adjacent stored columns in one pass, masking the boundaries
    # between rows so entries from separate rows are never compared.  Row
    # labels are looked up only for stored positions, keeping temporary
    # allocations bounded by nnz rather than by the number of (possibly empty)
    # rows.
    if csr_indices.size > 1:
        positions = np.arange(csr_indices.size, dtype=np.intp)
        row_labels = np.searchsorted(csr_indptr, positions, side="right")
        if np.any(
            (csr_indices[1:] <= csr_indices[:-1])
            & (row_labels[1:] == row_labels[:-1])
        ):
            raise ValueError("data CSR rows must have sorted unique columns")

    try:
        indptr = np.asarray(csc.indptr)
        indices = np.asarray(csc.indices)
        encoded_values = np.asarray(csc.data)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("data CSC view has invalid sparse storage") from exc
    if (
        indptr.ndim != 1
        or indptr.size != n_features + 1
        or not np.issubdtype(indptr.dtype, np.integer)
        or indices.ndim != 1
        or encoded_values.ndim != 1
        or indices.size != encoded_values.size
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("data CSC view has invalid sparse storage")
    if indptr.size and (
        np.any(indptr < 0)
        or np.any(indptr[1:] < indptr[:-1])
        or int(indptr[-1]) != int(encoded_values.size)
    ):
        raise ValueError("data CSC column pointers are invalid")
    if indices.size and (np.any(indices < 0) or np.any(indices >= n_samples)):
        raise ValueError("data CSC row indices are outside the dataset range")
    encoded = _normalize_tree_integer_metadata(encoded_values, "data CSC encoded bins")
    for feature in range(n_features):
        start, end = int(indptr[feature]), int(indptr[feature + 1])
        column_rows = indices[start:end]
        if column_rows.size > 1 and np.any(column_rows[1:] <= column_rows[:-1]):
            raise ValueError("data CSC columns must have sorted unique rows")
    # ``indptr`` describes one feature segment per column.  Label only the
    # stored values with their containing feature, then index the corresponding
    # bin limits.  This keeps temporaries bounded by nnz instead of allocating
    # an intermediate repeat-count array across all features (empty segments
    # naturally contribute no values).
    if encoded.size:
        encoded_positions = np.arange(encoded.size, dtype=np.intp)
        feature_labels = (
            np.searchsorted(indptr, encoded_positions, side="right") - 1
        )
        segment_bin_limits = n_bins[feature_labels]
        if np.any(encoded < 1) or np.any(encoded > segment_bin_limits):
            raise ValueError("data CSC encoded bins are outside mapper ranges")
    return csr, csc, mapper, n_bins, defaults, offsets


def build_histogram(
    data: BinnedDataset,
    row_indices: np.ndarray,
    feature_indices: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    layout: HistogramLayout | None = None,
) -> Histogram:
    """Aggregate one leaf's per-bin gradients, Hessians, and row counts."""
    # By default preserve the public/global mapper layout.  A tree supplies one
    # immutable local layout so each leaf's arrays contain only its sampled
    # features while all direct helper callers retain the original contract.
    try:
        n_samples, n_features = (int(data.shape[0]), int(data.shape[1]))
        csr = data.csr
        mapper = data.mapper
        raw_offsets = np.asarray(mapper.bin_offsets)
        raw_n_bins = np.asarray(mapper.n_bins)
        raw_defaults = np.asarray(mapper.default_bins)
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("data must contain a valid binned dataset") from exc

    if (
        raw_offsets.ndim != 1
        or raw_offsets.size != n_features + 1
        or raw_n_bins.ndim != 1
        or raw_n_bins.size != n_features
        or raw_defaults.ndim != 1
        or raw_defaults.size != n_features
    ):
        raise ValueError("data mapper has incompatible feature metadata")

    # Mapper metadata is part of the flattened histogram layout, so validate it
    # before converting values or allocating any output arrays.  In particular,
    # ``astype(int64)`` would silently truncate fractions and wrap large
    # unsigned values into negative indices.
    def _normalize_integer_metadata(values: np.ndarray, name: str) -> np.ndarray:
        dtype = values.dtype
        if not np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.complexfloating
        ):
            raise ValueError(
                f"data mapper {name} must be finite integer-valued numeric values"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"data mapper {name} must be finite integer-valued numeric values"
            )
        if np.issubdtype(dtype, np.floating) and np.any(
            values != np.trunc(values)
        ):
            raise ValueError(
                f"data mapper {name} must be finite integer-valued numeric values"
            )

        # Normalize only after the checks above.  Compare a round-trip for
        # floating values and reject unsigned values that exceed int64, both of
        # which would otherwise be silently changed by the cast.
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = np.asarray(values, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"data mapper {name} must be representable as signed int64 values"
            ) from exc
        if np.issubdtype(dtype, np.unsignedinteger) and np.any(normalized < 0):
            raise ValueError(
                f"data mapper {name} must be representable as signed int64 values"
            )
        if np.issubdtype(dtype, np.floating):
            with np.errstate(over="ignore", invalid="ignore"):
                round_trip = normalized.astype(dtype)
            if np.any(round_trip != values):
                raise ValueError(
                    f"data mapper {name} must be representable as signed int64 values"
                )
        return normalized

    raw_offsets = _normalize_integer_metadata(raw_offsets, "bin offsets")
    raw_n_bins = _normalize_integer_metadata(raw_n_bins, "n_bins")
    raw_defaults = _normalize_integer_metadata(raw_defaults, "default bins")

    if int(raw_offsets[0]) != 0 or np.any(raw_offsets[1:] < raw_offsets[:-1]):
        raise ValueError("data mapper bin offsets must be a non-negative prefix sum")
    if np.any(raw_n_bins <= 0):
        raise ValueError("data mapper n_bins must be positive")
    if np.any(np.diff(raw_offsets) != raw_n_bins):
        raise ValueError("data mapper bin offsets must match n_bins")
    if np.any(raw_defaults < 0) or np.any(raw_defaults >= raw_n_bins):
        raise ValueError("data mapper default bins are outside feature bin ranges")

    # The optimized kernel reads the row-oriented sparse view directly.  Keep
    # all CSR structural and encoded-bin checks in this public wrapper so the
    # trusted kernel can process blocks without repeating per-entry guards.
    if not sp.isspmatrix_csr(csr) or tuple(csr.shape) != (n_samples, n_features):
        raise ValueError("data CSR view must match its declared shape")
    try:
        csr_indptr = np.asarray(csr.indptr)
        csr_indices = np.asarray(csr.indices)
        csr_data = np.asarray(csr.data)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("data CSR view has invalid sparse storage") from exc
    if (
        csr_indptr.ndim != 1
        or csr_indptr.size != n_samples + 1
        or not np.issubdtype(csr_indptr.dtype, np.integer)
        or csr_indices.ndim != 1
        or csr_data.ndim != 1
        or csr_indices.size != csr_data.size
        or not np.issubdtype(csr_indices.dtype, np.integer)
    ):
        raise ValueError("data CSR view has invalid sparse storage")
    if csr_indptr.size and (
        np.any(csr_indptr < 0)
        or np.any(csr_indptr[1:] < csr_indptr[:-1])
        or int(csr_indptr[-1]) != int(csr_data.size)
    ):
        raise ValueError("data CSR row pointers are invalid")
    if csr_indices.size and (
        np.any(csr_indices < 0) or np.any(csr_indices >= n_features)
    ):
        raise ValueError("data CSR column indices are outside the dataset range")
    if csr_indices.size > 1:
        csr_positions = np.arange(csr_indices.size, dtype=np.intp)
        csr_row_labels = np.searchsorted(
            csr_indptr, csr_positions, side="right"
        )
        if np.any(
            (csr_indices[1:] <= csr_indices[:-1])
            & (csr_row_labels[1:] == csr_row_labels[:-1])
        ):
            raise ValueError("data CSR rows must have sorted unique columns")
    encoded_csr = _normalize_integer_metadata(csr_data, "CSR encoded bins")
    if encoded_csr.size:
        csr_bin_limits = raw_n_bins[csr_indices]
        if np.any(encoded_csr < 1) or np.any(encoded_csr > csr_bin_limits):
            raise ValueError("data CSR encoded bins are outside mapper ranges")

    # The adjacent-difference check implies this equality, but keep the final
    # prefix total explicit and compute it in Python integers to avoid int64
    # summation overflow on malformed metadata.
    expected_total = sum(int(value) for value in raw_n_bins)
    if int(raw_offsets[-1]) != expected_total:
        raise ValueError("data mapper final bin offset is inconsistent")
    if layout is not None:
        _validate_layout_for_mapper(layout, n_features, raw_n_bins, raw_defaults)
        local_features = layout.feature_indices
        total_bins = int(layout.bin_offsets[-1])
    else:
        local_features = None
        total_bins = expected_total

    # Normalize index vectors without silently truncating fractional values.
    # Empty Python lists are accepted as a convenience, while non-empty index
    # arrays must use an integer dtype and stay within the dataset bounds.
    def _normalize_indices(
        values: np.ndarray,
        upper_bound: int,
        name: str,
    ) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a one-dimensional integer array") from exc
        if raw.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if raw.size == 0:
            return np.empty(0, dtype=np.int64)
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        # Bounds are checked before conversion so very large unsigned values
        # cannot wrap around into valid signed indices.
        if np.any(raw < 0) or np.any(raw >= upper_bound):
            raise ValueError(f"{name} contains an out-of-range index")
        return np.asarray(raw, dtype=np.int64)

    rows = _normalize_indices(row_indices, n_samples, "row_indices")
    features = _normalize_indices(feature_indices, n_features, "feature_indices")
    if features.size > 1 and np.unique(features).size != features.size:
        raise ValueError("feature_indices must not contain duplicate features")
    if layout is not None:
        sorted_features = np.sort(features)
        if not np.array_equal(sorted_features, local_features):
            raise ValueError("feature_indices do not match histogram layout")
        # Local segments are always addressed in immutable sorted layout order.
        features = local_features

    try:
        gradient_values = np.asarray(gradients, dtype=np.float64)
        hessian_values = np.asarray(hessians, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("gradients and hessians must be numeric one-dimensional arrays") from exc
    if (
        gradient_values.ndim != 1
        or hessian_values.ndim != 1
        or gradient_values.size != n_samples
        or hessian_values.size != n_samples
    ):
        raise ValueError(
            "gradients and hessians must each have one value per dataset row"
        )
    if not np.isfinite(gradient_values).all() or not np.isfinite(
        hessian_values
    ).all():
        raise ValueError("gradients and hessians must be finite")

    return _build_histogram_validated(
        data=data,
        rows=rows,
        features=features,
        gradients=gradient_values,
        hessians=hessian_values,
        layout=layout,
        offsets=raw_offsets,
        n_bins=raw_n_bins,
        defaults=raw_defaults,
        total_bins=total_bins,
    )


def _build_histogram_direct(
    *,
    data: BinnedDataset,
    rows: np.ndarray,
    features: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    layout: HistogramLayout | None,
    offsets: np.ndarray,
    n_bins: np.ndarray,
    defaults: np.ndarray,
    total_bins: int,
) -> Histogram:
    """Reference histogram implementation using one CSC feature at a time.

    ``build_histogram`` owns all user-facing checks.  Callers of this kernel
    must provide signed integer indices, finite float arrays, canonical CSC
    storage, and mapper/layout metadata that already passed those checks.
    """
    gradient_sums = np.zeros(total_bins, dtype=np.float64)
    hessian_sums = np.zeros(total_bins, dtype=np.float64)
    counts = np.zeros(total_bins, dtype=np.int64)
    if rows.size == 0 or features.size == 0:
        return Histogram(
            gradient_sums=gradient_sums,
            hessian_sums=hessian_sums,
            counts=counts,
            layout=layout,
        )

    # Unique rows plus multiplicities preserves direct-helper duplicate-row
    # behavior while keeping the trusted tree path free of repeated checks.
    unique_rows, row_multiplicity = np.unique(rows, return_counts=True)
    leaf_gradient = float(np.sum(gradients[rows], dtype=np.float64))
    leaf_hessian = float(np.sum(hessians[rows], dtype=np.float64))
    csc = data.csc

    for local_slot, feature in enumerate(features):
        feature_index = int(feature)
        if layout is None:
            start = int(offsets[feature_index])
            end = int(offsets[feature_index + 1])
            default_bin = int(defaults[feature_index])
        else:
            start = int(layout.bin_offsets[local_slot])
            end = int(layout.bin_offsets[local_slot + 1])
            default_bin = int(layout.default_bins[local_slot])
        n_feature_bins = end - start
        if n_feature_bins <= 0:  # pragma: no cover - guarded by validation
            raise ValueError("data mapper must assign at least one bin per feature")

        column_start = int(csc.indptr[feature_index])
        column_end = int(csc.indptr[feature_index + 1])
        column_rows = np.asarray(csc.indices[column_start:column_end], dtype=np.int64)
        if column_rows.size:
            positions = np.searchsorted(unique_rows, column_rows, side="left")
            in_range = positions < unique_rows.size
            if np.any(in_range):
                candidate_positions = np.flatnonzero(in_range)
                matched = candidate_positions[
                    unique_rows[positions[candidate_positions]]
                    == column_rows[candidate_positions]
                ]
            else:
                matched = np.empty(0, dtype=np.int64)

            if matched.size:
                matched_rows = column_rows[matched]
                multiplicities = row_multiplicity[positions[matched]]
                encoded_bins = np.asarray(
                    csc.data[column_start:column_end][matched], dtype=np.int64
                )
                bin_ids = encoded_bins - np.int64(1)
                if np.any(bin_ids < 0) or np.any(bin_ids >= n_feature_bins):
                    raise ValueError("binned sparse values are outside mapper bin range")
                gradient_contrib = gradients[matched_rows] * multiplicities
                hessian_contrib = hessians[matched_rows] * multiplicities
                np.add.at(gradient_sums, start + bin_ids, gradient_contrib)
                np.add.at(hessian_sums, start + bin_ids, hessian_contrib)
                np.add.at(counts, start + bin_ids, multiplicities)

        # Sparse storage omits rows in the feature's default bin.  Whatever
        # selected rows were not represented explicitly contribute the
        # sufficient-statistic remainder to that one bin.
        segment = slice(start, end)
        represented_gradient = float(np.sum(gradient_sums[segment], dtype=np.float64))
        represented_hessian = float(np.sum(hessian_sums[segment], dtype=np.float64))
        represented_count = int(np.sum(counts[segment], dtype=np.int64))
        if default_bin < 0 or default_bin >= n_feature_bins:  # pragma: no cover
            raise ValueError("data mapper default bin is outside feature bin range")
        default_offset = start + default_bin
        gradient_sums[default_offset] += leaf_gradient - represented_gradient
        hessian_sums[default_offset] += leaf_hessian - represented_hessian
        counts[default_offset] += np.int64(rows.size - represented_count)

    # A zero-count bin cannot contain any sufficient statistics.  In
    # particular, default-bin remainder arithmetic can leave a tiny residual
    # in an otherwise empty bin; normalize it away so subtraction and split
    # comparisons never observe impossible statistics.
    empty_bins = counts == 0
    gradient_sums[empty_bins] = 0.0
    hessian_sums[empty_bins] = 0.0

    return Histogram(
        gradient_sums=gradient_sums,
        hessian_sums=hessian_sums,
        counts=counts,
        layout=layout,
    )


def _build_histogram_validated(
    *,
    data: BinnedDataset,
    rows: np.ndarray,
    features: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    layout: HistogramLayout | None,
    offsets: np.ndarray,
    n_bins: np.ndarray,
    defaults: np.ndarray,
    total_bins: int,
) -> Histogram:
    """Build one histogram by aggregating flattened CSR bin keys.

    The public ``build_histogram`` wrapper and ``fit_tree`` context validate
    sparse storage and mapper metadata before entering this trusted kernel.
    In particular, CSR encoded values are already known to be one-based and
    within their feature's bin range, so blocks only need an inexpensive
    widening/subtraction before key construction.
    """
    gradient_sums = np.zeros(total_bins, dtype=np.float64)
    hessian_sums = np.zeros(total_bins, dtype=np.float64)
    counts = np.zeros(total_bins, dtype=np.int64)
    if rows.size == 0 or features.size == 0:
        return Histogram(
            gradient_sums=gradient_sums,
            hessian_sums=hessian_sums,
            counts=counts,
            layout=layout,
        )

    csr = data.csr
    if layout is None:
        # Direct helper calls retain the historical global mapper layout.  The
        # reverse map stores output segment starts in the original feature
        # slots, while selected local slots are used only for row-block keys.
        feature_to_local = np.full(csr.shape[1], -1, dtype=np.int64)
        feature_to_local[features] = np.arange(features.size, dtype=np.int64)
        selected_offsets = np.asarray(offsets[features], dtype=np.int64)
        selected_defaults = np.asarray(defaults[features], dtype=np.int64)
        local_offsets = selected_offsets
    else:
        feature_to_local = layout.feature_to_local
        local_offsets = layout.bin_offsets
        selected_defaults = layout.default_bins

    leaf_gradient = float(np.sum(gradients[rows], dtype=np.float64))
    leaf_hessian = float(np.sum(hessians[rows], dtype=np.float64))

    # Prefix sums over source nonzeros determine deterministic consecutive
    # row-position blocks.  ``searchsorted`` avoids a Python loop over rows;
    # the only loop below is over bounded CSR row blocks.
    row_nnz = np.diff(np.asarray(csr.indptr, dtype=np.int64))
    selected_row_nnz = row_nnz[rows]
    row_prefix = np.empty(rows.size + 1, dtype=np.int64)
    row_prefix[0] = 0
    np.cumsum(selected_row_nnz, dtype=np.int64, out=row_prefix[1:])

    block_start = 0
    while block_start < rows.size:
        target = int(row_prefix[block_start]) + _HISTOGRAM_BLOCK_NNZ
        block_end = int(np.searchsorted(row_prefix, target, side="right") - 1)
        # A single row can exceed the target; process that row alone.  The
        # max also guarantees progress for zero-nnz rows and repeated rows.
        if block_end <= block_start:
            block_end = block_start + 1

        block_rows = rows[block_start:block_end]
        block = csr[block_rows]
        block_indptr = np.asarray(block.indptr, dtype=np.int64)
        block_nnz = int(block_indptr[-1])
        if block_nnz:
            row_positions = np.repeat(
                np.arange(block_rows.size, dtype=np.intp),
                np.diff(block_indptr),
            )
            original_columns = np.asarray(block.indices, dtype=np.int64)
            local_columns = feature_to_local[original_columns]
            keep = local_columns >= 0
            if np.any(keep):
                decoded_bins = np.asarray(block.data, dtype=np.int64) - np.int64(1)
                kept_rows = row_positions[keep]
                kept_slots = local_columns[keep]
                keys = local_offsets[kept_slots] + decoded_bins[keep]
                block_gradients = gradients[block_rows]
                block_hessians = hessians[block_rows]
                gradient_sums += np.bincount(
                    keys,
                    weights=block_gradients[kept_rows],
                    minlength=total_bins,
                )
                hessian_sums += np.bincount(
                    keys,
                    weights=block_hessians[kept_rows],
                    minlength=total_bins,
                )
                counts += np.bincount(keys, minlength=total_bins)

        block_start = block_end

    # Add implicit default-bin entries after all explicit CSR entries have
    # been aggregated.  Each represented-total reduction is vectorized across
    # selected features; no per-feature Python loop remains in this kernel.
    if layout is None:
        all_feature_gradient = np.add.reduceat(gradient_sums, offsets[:-1])
        all_feature_hessian = np.add.reduceat(hessian_sums, offsets[:-1])
        all_feature_counts = np.add.reduceat(counts, offsets[:-1])
        represented_gradient = all_feature_gradient[features]
        represented_hessian = all_feature_hessian[features]
        represented_counts = all_feature_counts[features]
        default_offsets = selected_offsets + selected_defaults
    else:
        represented_gradient = np.add.reduceat(
            gradient_sums, local_offsets[:-1]
        )
        represented_hessian = np.add.reduceat(
            hessian_sums, local_offsets[:-1]
        )
        represented_counts = np.add.reduceat(counts, local_offsets[:-1])
        default_offsets = local_offsets[:-1] + selected_defaults

    gradient_sums[default_offsets] += leaf_gradient - represented_gradient
    hessian_sums[default_offsets] += leaf_hessian - represented_hessian
    counts[default_offsets] += np.asarray(
        rows.size - represented_counts, dtype=np.int64
    )

    # Keep the optimized kernel's output physically consistent: bins with no
    # represented rows have exactly zero sufficient statistics.
    empty_bins = counts == 0
    gradient_sums[empty_bins] = 0.0
    hessian_sums[empty_bins] = 0.0

    return Histogram(
        gradient_sums=gradient_sums,
        hessian_sums=hessian_sums,
        counts=counts,
        layout=layout,
    )


def find_best_split(
    histogram: Histogram,
    feature_indices: np.ndarray,
    parent_gradient: float,
    parent_hessian: float,
    parent_count: int,
    mapper: BinMapper,
    config: LiteLightGBMConfig,
    layout: HistogramLayout | None = None,
) -> SplitInfo | None:
    """Return the highest-gain valid split, or ``None`` for a terminal leaf."""
    # Validate the flattened histogram and mapper metadata before indexing any
    # feature segment.  The tree builder normally supplies arrays created by
    # ``build_histogram``; keeping these checks here makes direct calls fail
    # clearly instead of silently evaluating malformed offsets.
    try:
        raw_gradients = np.asarray(histogram.gradient_sums)
        raw_hessians = np.asarray(histogram.hessian_sums)
        raw_counts = np.asarray(histogram.counts)
        raw_n_bins = np.asarray(mapper.n_bins)
        raw_offsets = np.asarray(mapper.bin_offsets)
        raw_defaults = np.asarray(mapper.default_bins)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("histogram and mapper must contain valid arrays") from exc

    if (
        raw_gradients.ndim != 1
        or raw_hessians.ndim != 1
        or raw_counts.ndim != 1
        or raw_gradients.size != raw_hessians.size
        or raw_gradients.size != raw_counts.size
    ):
        raise ValueError(
            "histogram gradient, Hessian, and count arrays must be one-dimensional"
        )

    # Gradients and Hessians are accumulated in floating point.  Rejecting
    # non-finite values keeps gain comparisons deterministic and avoids a NaN
    # silently winning/losing a tie.
    try:
        gradients = np.asarray(raw_gradients, dtype=np.float64)
        hessians = np.asarray(raw_hessians, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("histogram gradients and Hessians must be numeric") from exc
    if not np.isfinite(gradients).all() or not np.isfinite(hessians).all():
        raise ValueError("histogram gradients and Hessians must be finite")

    # Counts represent exact row multiplicities.  Normalize only after checking
    # that values are integral so fractional counts cannot be truncated into a
    # seemingly valid child-size constraint.
    if (
        not np.issubdtype(raw_counts.dtype, np.number)
        or np.issubdtype(raw_counts.dtype, np.complexfloating)
    ):
        raise ValueError("histogram counts must be finite integer-valued numbers")
    if not np.isfinite(raw_counts).all():
        raise ValueError("histogram counts must be finite integer-valued numbers")
    if np.issubdtype(raw_counts.dtype, np.floating) and np.any(
        raw_counts != np.trunc(raw_counts)
    ):
        raise ValueError("histogram counts must be finite integer-valued numbers")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            counts = np.asarray(raw_counts, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("histogram counts must be representable as int64") from exc
    if np.issubdtype(raw_counts.dtype, np.unsignedinteger) and np.any(counts < 0):
        raise ValueError("histogram counts must be representable as int64")
    if np.issubdtype(raw_counts.dtype, np.floating):
        with np.errstate(over="ignore", invalid="ignore"):
            if np.any(counts.astype(raw_counts.dtype) != raw_counts):
                raise ValueError("histogram counts must be representable as int64")
    if np.any(counts < 0):
        raise ValueError("histogram counts must be non-negative")

    # Mapper integer metadata determines the flattened layout.  As with counts,
    # reject fractions and values that would wrap when converted to int64.
    mapper_arrays = (raw_n_bins, raw_offsets, raw_defaults)
    mapper_names = ("n_bins", "bin_offsets", "default_bins")
    normalized_mapper: list[np.ndarray] = []
    for values, name in zip(mapper_arrays, mapper_names):
        if values.ndim != 1:
            raise ValueError(f"mapper {name} must be a one-dimensional integer array")
        if (
            not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.complexfloating)
        ):
            raise ValueError(f"mapper {name} must be finite integer-valued numbers")
        if not np.isfinite(values).all():
            raise ValueError(f"mapper {name} must be finite integer-valued numbers")
        if np.issubdtype(values.dtype, np.floating) and np.any(
            values != np.trunc(values)
        ):
            raise ValueError(f"mapper {name} must be finite integer-valued numbers")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = np.asarray(values, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"mapper {name} must be representable as int64") from exc
        if np.issubdtype(values.dtype, np.unsignedinteger) and np.any(
            normalized < 0
        ):
            raise ValueError(f"mapper {name} must be representable as int64")
        if np.issubdtype(values.dtype, np.floating):
            with np.errstate(over="ignore", invalid="ignore"):
                round_trip = normalized.astype(values.dtype)
            if np.any(round_trip != values):
                raise ValueError(f"mapper {name} must be representable as int64")
        normalized_mapper.append(normalized)

    n_bins, offsets, defaults = normalized_mapper
    n_features = int(n_bins.size)
    if offsets.size != n_features + 1 or defaults.size != n_features:
        raise ValueError("mapper metadata has incompatible feature dimensions")
    if offsets.size == 0 or int(offsets[0]) != 0 or np.any(offsets < 0):
        raise ValueError("mapper bin offsets must be a non-negative prefix sum")
    if np.any(n_bins <= 0) or np.any(np.diff(offsets) != n_bins):
        raise ValueError("mapper bin offsets must match positive n_bins")
    if np.any(defaults < 0) or np.any(defaults >= n_bins):
        raise ValueError("mapper default bins are outside feature bin ranges")

    histogram_layout = histogram.layout
    if layout is not None:
        if histogram_layout is not None and layout is not histogram_layout:
            raise ValueError("histogram and split layout objects do not match")
        histogram_layout = layout
    layout = histogram_layout
    if layout is not None:
        _validate_layout_for_mapper(layout, n_features, n_bins, defaults)
        total_bins = int(layout.bin_offsets[-1])
    else:
        total_bins = int(offsets[-1])
    if total_bins != gradients.size:
        raise ValueError("histogram arrays do not match mapper bin layout")

    # Normalize selected feature indices without allowing a fractional value to
    # be truncated into a different feature.  Empty selections simply have no
    # candidate split and return ``None`` after metadata validation above.
    try:
        raw_features = np.asarray(feature_indices)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature_indices must be a one-dimensional integer array") from exc
    if raw_features.ndim != 1:
        raise ValueError("feature_indices must be a one-dimensional integer array")
    if raw_features.size and not np.issubdtype(raw_features.dtype, np.integer):
        raise ValueError("feature_indices must be a one-dimensional integer array")
    if raw_features.size:
        if np.any(raw_features < 0) or np.any(raw_features >= n_features):
            raise ValueError("feature_indices contains an out-of-range feature")
        features = np.asarray(raw_features, dtype=np.int64)
        if np.unique(features).size != features.size:
            raise ValueError("feature_indices must not contain duplicate features")
    else:
        features = np.empty(0, dtype=np.int64)
    if layout is not None:
        sorted_features = np.sort(features)
        if not np.array_equal(sorted_features, layout.feature_indices):
            raise ValueError("feature_indices do not match histogram layout")
        features = layout.feature_indices

    # Configuration values are expected to be validated by ``fit`` as well,
    # but direct split-search calls should not silently accept negative
    # constraints or non-finite regularization values.
    try:
        raw_min_samples = float(config.min_child_samples)
        min_child_weight = float(config.min_child_weight)
        min_split_gain = float(config.min_split_gain)
        reg_alpha = float(config.reg_alpha)
        reg_lambda = float(config.reg_lambda)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("split configuration values must be numeric") from exc
    if (
        not np.isfinite(raw_min_samples)
        or raw_min_samples != np.trunc(raw_min_samples)
        or raw_min_samples < 0
    ):
        raise ValueError("min_child_samples must be a non-negative integer")
    if (
        not np.isfinite(min_child_weight)
        or not np.isfinite(min_split_gain)
        or not np.isfinite(reg_alpha)
        or not np.isfinite(reg_lambda)
        or min_child_weight < 0
        or min_split_gain < 0
        or reg_alpha < 0
        or reg_lambda < 0
    ):
        raise ValueError(
            "child constraints, gain, and regularization values must be finite and non-negative"
        )
    min_child_samples = int(raw_min_samples)

    parent_values = (parent_gradient, parent_hessian, parent_count)
    parent_names = ("parent_gradient", "parent_hessian", "parent_count")
    normalized_parent: list[float | int] = []
    for value, name in zip(parent_values, parent_names):
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if raw.ndim != 0:
            raise ValueError(f"{name} must be a finite scalar")
        try:
            normalized = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if not np.isfinite(normalized):
            raise ValueError(f"{name} must be finite")
        if name == "parent_count":
            if normalized != np.trunc(normalized) or normalized < 0:
                raise ValueError("parent_count must be a non-negative integer")
            normalized_parent.append(int(normalized))
        else:
            normalized_parent.append(normalized)
    parent_gradient_value = float(normalized_parent[0])
    parent_hessian_value = float(normalized_parent[1])
    parent_count_value = int(normalized_parent[2])

    return _find_best_split_validated(
        gradients=gradients,
        hessians=hessians,
        counts=counts,
        features=features,
        parent_gradient=parent_gradient_value,
        parent_hessian=parent_hessian_value,
        parent_count=parent_count_value,
        n_bins=n_bins,
        offsets=offsets,
        defaults=defaults,
        layout=layout,
        min_child_samples=min_child_samples,
        min_child_weight=min_child_weight,
        min_split_gain=min_split_gain,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
    )


def _find_best_split_validated(
    *,
    gradients: np.ndarray,
    hessians: np.ndarray,
    counts: np.ndarray,
    features: np.ndarray,
    parent_gradient: float,
    parent_hessian: float,
    parent_count: int,
    n_bins: np.ndarray,
    offsets: np.ndarray,
    defaults: np.ndarray,
    layout: HistogramLayout | None,
    min_child_samples: int,
    min_child_weight: float,
    min_split_gain: float,
    reg_alpha: float,
    reg_lambda: float,
) -> SplitInfo | None:
    """Search split candidates from normalized histogram/context state."""
    parent_denominator = parent_hessian + reg_lambda
    if parent_denominator > EPSILON:
        parent_thresholded = float(soft_threshold(parent_gradient, reg_alpha))
        parent_score = (parent_thresholded * parent_thresholded) / parent_denominator
    else:
        parent_score = 0.0

    best: SplitInfo | None = None
    best_gain = -np.inf
    best_feature = int(n_bins.size)
    best_threshold = int(np.max(n_bins)) if n_bins.size else 1

    if layout is None:
        feature_segments = (
            (
                int(feature),
                int(offsets[int(feature)]),
                int(offsets[int(feature) + 1]),
                int(defaults[int(feature)]),
            )
            for feature in features
        )
    else:
        feature_segments = (
            (
                int(feature),
                int(layout.bin_offsets[local_slot]),
                int(layout.bin_offsets[local_slot + 1]),
                int(layout.default_bins[local_slot]),
            )
            for local_slot, feature in enumerate(layout.feature_indices)
        )

    for feature, start, end, default_bin in feature_segments:
        feature_bin_count = end - start
        if feature_bin_count <= 1:
            continue

        # Keep prefix accumulation and threshold visitation in the exact
        # order used by the checked implementation.
        gradient_prefix = np.cumsum(gradients[start:end], dtype=np.float64)
        hessian_prefix = np.cumsum(hessians[start:end], dtype=np.float64)
        count_prefix = np.cumsum(counts[start:end], dtype=np.int64)
        for threshold in range(feature_bin_count - 1):
            left_count = int(count_prefix[threshold])
            right_count = parent_count - left_count
            if left_count < min_child_samples or right_count < min_child_samples:
                continue

            left_gradient = float(gradient_prefix[threshold])
            left_hessian = float(hessian_prefix[threshold])
            right_gradient = parent_gradient - left_gradient
            right_hessian = parent_hessian - left_hessian
            if left_hessian < min_child_weight or right_hessian < min_child_weight:
                continue

            left_denominator = left_hessian + reg_lambda
            right_denominator = right_hessian + reg_lambda
            if left_denominator > EPSILON:
                left_thresholded = float(soft_threshold(left_gradient, reg_alpha))
                left_score = (left_thresholded * left_thresholded) / left_denominator
            else:
                left_score = 0.0
            if right_denominator > EPSILON:
                right_thresholded = float(soft_threshold(right_gradient, reg_alpha))
                right_score = (right_thresholded * right_thresholded) / right_denominator
            else:
                right_score = 0.0

            gain = float(left_score + right_score - parent_score)
            if not np.isfinite(gain) or gain <= min_split_gain:
                continue
            is_better = (
                best is None
                or gain > best_gain
                or (
                    gain == best_gain
                    and (
                        feature < best_feature
                        or (feature == best_feature and threshold < best_threshold)
                    )
                )
            )
            if not is_better:
                continue
            best_gain = gain
            best_feature = feature
            best_threshold = threshold
            best = SplitInfo(
                gain=gain,
                feature=feature,
                threshold_bin=threshold,
                default_left=bool(default_bin <= threshold),
                left_count=left_count,
                right_count=right_count,
                left_gradient=left_gradient,
                left_hessian=left_hessian,
                right_gradient=right_gradient,
                right_hessian=right_hessian,
            )

    return best


def partition_rows(
    data: BinnedDataset,
    row_indices: np.ndarray,
    split: SplitInfo,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition a leaf's rows according to a selected binned split."""
    # Keep all routing in sparse/indexed form.  In particular, avoid extracting
    # a dense column for the selected rows: the project matrix can be very wide
    # and implicit entries are represented by the mapper's default bin.
    try:
        n_samples, n_features = (int(data.shape[0]), int(data.shape[1]))
        mapper = data.mapper
        raw_n_bins = np.asarray(mapper.n_bins)
        raw_defaults = np.asarray(mapper.default_bins)
        raw_offsets = np.asarray(mapper.bin_offsets)
        csc = data.csc
        csr = data.csr
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("data must contain a valid binned dataset") from exc

    if n_samples < 0 or n_features < 0:
        raise ValueError("data must have non-negative dimensions")
    if not sp.issparse(csr) or tuple(csr.shape) != (n_samples, n_features):
        raise ValueError("data CSR view has incompatible shape")
    if not sp.issparse(csc) or tuple(csc.shape) != (n_samples, n_features):
        raise ValueError("data CSC view has incompatible shape")
    # ``partition_rows`` only reads the CSC column, but accept any SciPy sparse
    # format by converting sparsely.  This never materializes a dense matrix.
    if not sp.isspmatrix_csc(csc):
        try:
            csc = csc.tocsc(copy=False)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("data CSC view must be a sparse matrix") from exc

    if (
        raw_n_bins.ndim != 1
        or raw_n_bins.size != n_features
        or raw_defaults.ndim != 1
        or raw_defaults.size != n_features
        or raw_offsets.ndim != 1
        or raw_offsets.size != n_features + 1
    ):
        raise ValueError("data mapper has incompatible feature metadata")

    # Mapper integer metadata determines the legal encoded-bin ranges.  Check
    # integrality and representability before converting, so malformed arrays
    # cannot silently truncate or wrap into valid indices.
    def _normalize_integer_metadata(values: np.ndarray, name: str) -> np.ndarray:
        dtype = values.dtype
        if not np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.complexfloating
        ):
            raise ValueError(
                f"data mapper {name} must be finite integer-valued numeric values"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"data mapper {name} must be finite integer-valued numeric values"
            )
        if np.issubdtype(dtype, np.floating) and np.any(
            values != np.trunc(values)
        ):
            raise ValueError(
                f"data mapper {name} must be finite integer-valued numeric values"
            )
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = np.asarray(values, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"data mapper {name} must be representable as signed int64 values"
            ) from exc
        if np.issubdtype(dtype, np.unsignedinteger) and np.any(normalized < 0):
            raise ValueError(
                f"data mapper {name} must be representable as signed int64 values"
            )
        if np.issubdtype(dtype, np.floating):
            with np.errstate(over="ignore", invalid="ignore"):
                round_trip = normalized.astype(dtype)
            if np.any(round_trip != values):
                raise ValueError(
                    f"data mapper {name} must be representable as signed int64 values"
                )
        return normalized

    n_bins = _normalize_integer_metadata(raw_n_bins, "n_bins")
    defaults = _normalize_integer_metadata(raw_defaults, "default bins")
    offsets = _normalize_integer_metadata(raw_offsets, "bin offsets")
    if (
        int(offsets[0]) != 0
        or np.any(offsets < 0)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("data mapper bin offsets must be a non-negative prefix sum")
    if np.any(n_bins <= 0) or np.any(np.diff(offsets) != n_bins):
        raise ValueError("data mapper bin offsets must match positive n_bins")
    if np.any(defaults < 0) or np.any(defaults >= n_bins):
        raise ValueError("data mapper default bins are outside feature bin ranges")
    expected_total = sum(int(value) for value in n_bins)
    if int(offsets[-1]) != expected_total:
        raise ValueError("data mapper final bin offset is inconsistent")

    # Normalize row and split indices without allowing fractional values or
    # unsigned values that would wrap into valid signed indices.
    try:
        raw_rows = np.asarray(row_indices)
    except (TypeError, ValueError) as exc:
        raise ValueError("row_indices must be a one-dimensional integer array") from exc
    if raw_rows.ndim != 1:
        raise ValueError("row_indices must be a one-dimensional integer array")
    if raw_rows.size and not np.issubdtype(raw_rows.dtype, np.integer):
        raise ValueError("row_indices must be a one-dimensional integer array")
    if raw_rows.size and (np.any(raw_rows < 0) or np.any(raw_rows >= n_samples)):
        raise ValueError("row_indices contains an out-of-range index")
    rows = np.asarray(raw_rows, dtype=np.int64)

    def _normalize_split_integer(value: Any, name: str) -> int:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"split {name} must be a scalar integer") from exc
        if raw.ndim != 0:
            raise ValueError(f"split {name} must be a scalar integer")
        dtype = raw.dtype
        if not np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.complexfloating
        ):
            raise ValueError(f"split {name} must be a scalar integer")
        if not np.isfinite(raw).all() or (
            np.issubdtype(dtype, np.floating) and raw != np.trunc(raw)
        ):
            raise ValueError(f"split {name} must be a scalar integer")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = np.asarray(raw, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"split {name} must be a representable integer") from exc
        if np.issubdtype(dtype, np.unsignedinteger) and int(normalized) < 0:
            raise ValueError(f"split {name} must be a representable integer")
        if np.issubdtype(dtype, np.floating):
            with np.errstate(over="ignore", invalid="ignore"):
                round_trip = normalized.astype(dtype)
            if round_trip != raw:
                raise ValueError(f"split {name} must be a representable integer")
        return int(normalized)

    # Read all required split fields up front so absent/malformed split
    # objects fail through the routine's established ValueError validation
    # path instead of leaking AttributeError from direct attribute access.
    try:
        split_feature = split.feature
        split_threshold_bin = split.threshold_bin
        split_default_left = split.default_left
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "split must contain feature, threshold_bin, and default_left fields"
        ) from exc

    feature = _normalize_split_integer(split_feature, "feature")
    threshold = _normalize_split_integer(split_threshold_bin, "threshold_bin")
    if feature < 0 or feature >= n_features:
        raise ValueError("split feature is outside the dataset feature range")
    feature_bin_count = int(n_bins[feature])
    if threshold < 0 or threshold >= feature_bin_count:
        raise ValueError("split threshold_bin is outside the feature bin range")

    raw_default_left = np.asarray(split_default_left)
    if raw_default_left.ndim != 0 or raw_default_left.dtype != np.dtype(bool):
        raise ValueError("split default_left must be a boolean scalar")
    default_left = bool(raw_default_left)
    default_bin = int(defaults[feature])
    expected_default_left = default_bin <= threshold
    if default_left != expected_default_left:
        raise ValueError(
            "split default_left is inconsistent with the mapper default bin"
        )

    # Validate the selected CSC segment before routing.  Stored entries use
    # ``bin_id + 1`` and must be integral and within this feature's bin range.
    try:
        indptr = np.asarray(csc.indptr)
        indices = np.asarray(csc.indices)
        csc_data = np.asarray(csc.data)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("data CSC view has invalid sparse storage") from exc
    if (
        indptr.ndim != 1
        or indptr.size != n_features + 1
        or not np.issubdtype(indptr.dtype, np.integer)
        or indices.ndim != 1
        or csc_data.ndim != 1
        or indices.size != csc_data.size
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("data CSC view has invalid sparse storage")
    if indptr.size and (
        np.any(indptr < 0)
        or np.any(indptr[1:] < indptr[:-1])
        or int(indptr[-1]) != int(csc_data.size)
    ):
        raise ValueError("data CSC column pointers are invalid")

    column_start = int(indptr[feature])
    column_end = int(indptr[feature + 1])
    if (
        column_start < 0
        or column_end < column_start
        or column_end > csc_data.size
    ):
        raise ValueError("data CSC view has invalid column pointers")
    stored_rows = indices[column_start:column_end]
    raw_encoded = csc_data[column_start:column_end]
    if stored_rows.ndim != 1 or raw_encoded.ndim != 1:
        raise ValueError("data CSC view has invalid sparse storage")
    if stored_rows.size != raw_encoded.size:
        raise ValueError("data CSC view has inconsistent sparse storage")
    if stored_rows.size and (
        not np.issubdtype(stored_rows.dtype, np.integer)
        or np.any(stored_rows < 0)
        or np.any(stored_rows >= n_samples)
    ):
        raise ValueError("data CSC row indices are outside the dataset range")
    encoded = _normalize_integer_metadata(raw_encoded, "encoded bins")
    if encoded.size:
        if np.any(encoded <= 0) or np.any(encoded > feature_bin_count):
            raise ValueError("binned sparse values are outside mapper bin range")
        # Duplicate coordinates would make the logical bin ambiguous.  Normal
        # transform_bins output is canonical, so reject malformed segments.
        if np.unique(stored_rows).size != stored_rows.size:
            raise ValueError("data CSC view contains duplicate feature entries")

    return _partition_rows_validated(
        csc=csc,
        rows=rows,
        feature=feature,
        threshold=threshold,
        default_left=default_left,
    )


def _partition_rows_validated(
    *,
    csc: sp.csc_matrix,
    rows: np.ndarray,
    feature: int,
    threshold: int,
    default_left: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Route rows using canonical CSC/split state after wrapper validation."""
    if rows.size == 0:
        return rows.copy(), rows.copy()

    column_start = int(csc.indptr[feature])
    column_end = int(csc.indptr[feature + 1])
    stored_rows = np.asarray(csc.indices[column_start:column_end], dtype=np.int64)
    encoded = np.asarray(csc.data[column_start:column_end], dtype=np.int64)

    # Map each distinct selected row once, then expand through ``inverse``.
    # This preserves ordering and duplicate-row behavior without a Python loop.
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    go_left_unique = np.full(unique_rows.size, default_left, dtype=bool)
    if stored_rows.size and unique_rows.size:
        positions = np.searchsorted(unique_rows, stored_rows, side="left")
        in_range = positions < unique_rows.size
        candidate_positions = np.flatnonzero(in_range)
        if candidate_positions.size:
            matched = candidate_positions[
                unique_rows[positions[candidate_positions]]
                == stored_rows[candidate_positions]
            ]
            if matched.size:
                go_left_unique[positions[matched]] = (
                    encoded[matched] - np.int64(1) <= threshold
                )

    go_left = go_left_unique[inverse]
    return rows[go_left], rows[~go_left]


def _fit_tree(
    data: BinnedDataset,
    gradients: np.ndarray,
    hessians: np.ndarray,
    row_indices: np.ndarray,
    feature_indices: np.ndarray,
    config: LiteLightGBMConfig,
    *,
    use_histogram_subtraction: bool,
    _histogram_cache_bytes: int | None = None,
) -> DecisionTree:
    """Fit one tree, optionally using histogram subtraction for child leaves."""
    # Keep the tree builder's validation local to the arguments it consumes.
    # ``build_histogram`` and ``find_best_split`` perform the detailed mapper
    # and histogram checks; these checks ensure root statistics can be computed
    # safely before either helper is called.
    try:
        n_samples, n_features = (int(data.shape[0]), int(data.shape[1]))
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("data must contain a valid binned dataset") from exc
    if n_samples < 0 or n_features < 0:
        raise ValueError("data must have non-negative dimensions")
    # Validate canonical sparse storage and mapper metadata once.  Every
    # trusted leaf kernel below receives this normalized context and therefore
    # does not repeat structural checks for each child.
    csr, csc, mapper, n_bins, defaults, offsets = _validate_tree_storage(
        data, n_samples, n_features
    )

    try:
        raw_rows = np.asarray(row_indices)
    except (TypeError, ValueError) as exc:
        raise ValueError("row_indices must be a one-dimensional integer array") from exc
    if raw_rows.ndim != 1:
        raise ValueError("row_indices must be a one-dimensional integer array")
    if raw_rows.size and not np.issubdtype(raw_rows.dtype, np.integer):
        raise ValueError("row_indices must be a one-dimensional integer array")
    rows = np.asarray(raw_rows, dtype=np.int64)
    if rows.size and (np.any(rows < 0) or np.any(rows >= n_samples)):
        raise ValueError("row_indices contains an out-of-range index")

    try:
        raw_features = np.asarray(feature_indices)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "feature_indices must be a one-dimensional integer array"
        ) from exc
    if raw_features.ndim != 1:
        raise ValueError("feature_indices must be a one-dimensional integer array")
    if raw_features.size and not np.issubdtype(raw_features.dtype, np.integer):
        raise ValueError("feature_indices must be a one-dimensional integer array")
    features = np.asarray(raw_features, dtype=np.int64)
    if features.size and (
        np.any(features < 0) or np.any(features >= n_features)
    ):
        raise ValueError("feature_indices contains an out-of-range feature")
    if features.size > 1 and np.unique(features).size != features.size:
        raise ValueError("feature_indices must not contain duplicate features")
    # The estimator samples and records features in ascending original-index
    # order.  Sorting here also makes direct fit_tree calls obey that invariant
    # without changing which features are eligible for split search.
    features = np.sort(features.copy())
    # Build one immutable compact histogram layout for this tree.  All leaf
    # histograms below share this object; no per-leaf reverse lookup is built.
    layout = _make_histogram_layout(mapper, features)

    try:
        gradient_values = np.asarray(gradients, dtype=np.float64)
        hessian_values = np.asarray(hessians, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "gradients and hessians must be numeric one-dimensional arrays"
        ) from exc
    if (
        gradient_values.ndim != 1
        or hessian_values.ndim != 1
        or gradient_values.size != n_samples
        or hessian_values.size != n_samples
    ):
        raise ValueError(
            "gradients and hessians must each have one value per dataset row"
        )
    if not np.isfinite(gradient_values).all() or not np.isfinite(
        hessian_values
    ).all():
        raise ValueError("gradients and hessians must be finite")

    # Values used by the Newton correction and split search.  The explicit
    # checks here make malformed direct calls fail clearly even when the root
    # has no eligible split (in which case find_best_split would not inspect
    # every configuration field).
    try:
        raw_num_leaves = float(config.num_leaves)
        raw_max_depth = float(config.max_depth)
        min_child_samples = float(config.min_child_samples)
        min_child_weight = float(config.min_child_weight)
        min_split_gain = float(config.min_split_gain)
        reg_alpha = float(config.reg_alpha)
        reg_lambda = float(config.reg_lambda)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tree configuration values must be numeric") from exc
    if (
        not np.isfinite(raw_num_leaves)
        or raw_num_leaves != np.trunc(raw_num_leaves)
        or raw_num_leaves < 2
    ):
        raise ValueError("num_leaves must be an integer of at least two")
    if (
        not np.isfinite(raw_max_depth)
        or raw_max_depth != np.trunc(raw_max_depth)
    ):
        raise ValueError("max_depth must be an integer")
    if (
        not np.isfinite(min_child_samples)
        or min_child_samples != np.trunc(min_child_samples)
        or min_child_samples < 0
        or not np.isfinite(min_child_weight)
        or not np.isfinite(min_split_gain)
        or not np.isfinite(reg_alpha)
        or not np.isfinite(reg_lambda)
        or min_child_weight < 0
        or min_split_gain < 0
        or reg_alpha < 0
        or reg_lambda < 0
    ):
        raise ValueError(
            "child constraints, gain, and regularization values must be finite and non-negative"
        )
    num_leaves = int(raw_num_leaves)
    max_depth = int(raw_max_depth)
    min_child_samples = int(min_child_samples)

    context = _TreeContext(
        data=data,
        csr=csr,
        csc=csc,
        mapper=mapper,
        n_samples=n_samples,
        n_features=n_features,
        n_bins=n_bins,
        defaults=defaults,
        offsets=offsets,
        gradients=gradient_values,
        hessians=hessian_values,
        rows=rows,
        features=features,
        layout=layout,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        min_child_weight=float(min_child_weight),
        min_split_gain=float(min_split_gain),
        reg_alpha=float(reg_alpha),
        reg_lambda=float(reg_lambda),
    )

    if _histogram_cache_bytes is None:
        cache_bytes = _HISTOGRAM_CACHE_MAX_BYTES
    else:
        if isinstance(_histogram_cache_bytes, (bool, np.bool_)):
            raise ValueError(
                "histogram cache byte budget must be a non-negative integer"
            )
        try:
            cache_bytes = operator.index(_histogram_cache_bytes)
        except TypeError as exc:
            raise ValueError(
                "histogram cache byte budget must be a non-negative integer"
            ) from exc
        if cache_bytes < 0:
            raise ValueError(
                "histogram cache byte budget must be a non-negative integer"
            )
        cache_bytes = int(cache_bytes)

    root_gradient = float(np.sum(gradient_values[rows], dtype=np.float64))
    root_hessian = float(np.sum(hessian_values[rows], dtype=np.float64))
    root_denominator = root_hessian + reg_lambda
    if root_denominator > EPSILON:
        root_value = -float(soft_threshold(root_gradient, reg_alpha)) / root_denominator
    else:
        root_value = 0.0

    root = TreeNode(
        depth=0,
        value=float(root_value),
        sample_count=int(rows.size),
        gradient_sum=root_gradient,
        hessian_sum=root_hessian,
    )
    nodes: list[TreeNode] = [root]
    # Row assignments are construction-only state.  They are indexed in the
    # same stable creation order as ``nodes`` and discarded with this scope.
    node_rows: list[np.ndarray | None] = [rows]
    leaves = 1

    # A candidate entry starts with ``-gain`` for heapq's min-heap, then uses
    # deterministic split fields and finally node creation order as a stable
    # tie-break.  The SplitInfo object is never compared because node indices
    # are unique for every live candidate.
    queue: list[tuple[float, int, int, int, SplitInfo]] = []
    # Construction-only state: every queued candidate may have one histogram
    # retained here, and entries are removed as soon as candidates are popped.
    # Histograms are never attached to the fitted tree nodes.  The direct
    # oracle intentionally skips retention because it does not use subtraction.
    histogram_cache = _HistogramCache(cache_bytes) if use_histogram_subtraction else None
    root_histogram = _build_histogram_validated(
        data=context.data,
        rows=rows,
        features=context.features,
        gradients=context.gradients,
        hessians=context.hessians,
        layout=context.layout,
        offsets=context.offsets,
        n_bins=context.n_bins,
        defaults=context.defaults,
        total_bins=int(context.layout.bin_offsets[-1]),
    )
    if max_depth <= 0 or 0 < max_depth:
        root_split = _find_best_split_validated(
            gradients=root_histogram.gradient_sums,
            hessians=root_histogram.hessian_sums,
            counts=root_histogram.counts,
            features=context.features,
            parent_gradient=root_gradient,
            parent_hessian=root_hessian,
            parent_count=int(rows.size),
            n_bins=context.n_bins,
            offsets=context.offsets,
            defaults=context.defaults,
            layout=context.layout,
            min_child_samples=context.min_child_samples,
            min_child_weight=context.min_child_weight,
            min_split_gain=context.min_split_gain,
            reg_alpha=context.reg_alpha,
            reg_lambda=context.reg_lambda,
        )
        if root_split is not None:
            heapq.heappush(
                queue,
                (
                    -float(root_split.gain),
                    int(root_split.feature),
                    int(root_split.threshold_bin),
                    0,
                    root_split,
                ),
            )
            if histogram_cache is not None:
                histogram_cache.put(0, root_histogram)
    # If the root is terminal its histogram is no longer needed.  When it is
    # queued, the cache above is the sole remaining owner after this scope.
    del root_histogram

    while queue and leaves < num_leaves:
        _, _, _, node_index, split = heapq.heappop(queue)
        parent_histogram = (
            histogram_cache.take(node_index)
            if histogram_cache is not None
            else None
        )
        current_rows = node_rows[node_index]
        if current_rows is None:
            # Defensive guard for malformed/stale candidates; normal growth
            # enqueues each leaf exactly once and never reaches this branch.
            del parent_histogram
            continue
        parent = nodes[node_index]
        if max_depth > 0 and parent.depth >= max_depth:
            del parent_histogram
            continue

        left_rows, right_rows = _partition_rows_validated(
            csc=context.csc,
            rows=current_rows,
            feature=int(split.feature),
            threshold=int(split.threshold_bin),
            default_left=bool(split.default_left),
        )
        left_gradient = float(split.left_gradient)
        left_hessian = float(split.left_hessian)
        right_gradient = float(split.right_gradient)
        right_hessian = float(split.right_hessian)
        left_denominator = left_hessian + reg_lambda
        right_denominator = right_hessian + reg_lambda
        if left_denominator > EPSILON:
            left_value = -float(soft_threshold(left_gradient, reg_alpha)) / left_denominator
        else:
            left_value = 0.0
        if right_denominator > EPSILON:
            right_value = -float(soft_threshold(right_gradient, reg_alpha)) / right_denominator
        else:
            right_value = 0.0

        parent.feature = int(split.feature)
        parent.threshold_bin = int(split.threshold_bin)
        parent.default_left = bool(split.default_left)
        parent.split_gain = float(split.gain)
        parent.left_child = len(nodes)
        left_index = parent.left_child
        left_node = TreeNode(
            depth=parent.depth + 1,
            value=float(left_value),
            sample_count=int(left_rows.size),
            gradient_sum=left_gradient,
            hessian_sum=left_hessian,
        )
        nodes.append(left_node)
        node_rows.append(left_rows)
        parent.right_child = len(nodes)
        right_index = parent.right_child
        right_node = TreeNode(
            depth=parent.depth + 1,
            value=float(right_value),
            sample_count=int(right_rows.size),
            gradient_sum=right_gradient,
            hessian_sum=right_hessian,
        )
        nodes.append(right_node)
        node_rows.append(right_rows)
        node_rows[node_index] = None
        leaves += 1

        # Once the requested terminal-leaf budget is exhausted, there is no
        # reason to build child histograms.  Otherwise, only children below
        # the depth cap can produce queue candidates.
        if leaves >= num_leaves:
            del parent_histogram
            continue
        child_depth = parent.depth + 1
        if max_depth > 0 and child_depth >= max_depth:
            del parent_histogram
            continue

        if use_histogram_subtraction and parent_histogram is not None:
            # Build only the smaller child directly (ties deterministically
            # choose the left child), then derive the sibling from the popped
            # parent.  The private oracle below retains the direct-both-child
            # construction for parity checks without changing public APIs.
            if left_rows.size <= right_rows.size:
                left_histogram = _build_histogram_validated(
                    data=context.data,
                    rows=left_rows,
                    features=context.features,
                    gradients=context.gradients,
                    hessians=context.hessians,
                    layout=context.layout,
                    offsets=context.offsets,
                    n_bins=context.n_bins,
                    defaults=context.defaults,
                    total_bins=int(context.layout.bin_offsets[-1]),
                )
                right_histogram = _subtract_histograms(
                    parent_histogram, left_histogram
                )
            else:
                right_histogram = _build_histogram_validated(
                    data=context.data,
                    rows=right_rows,
                    features=context.features,
                    gradients=context.gradients,
                    hessians=context.hessians,
                    layout=context.layout,
                    offsets=context.offsets,
                    n_bins=context.n_bins,
                    defaults=context.defaults,
                    total_bins=int(context.layout.bin_offsets[-1]),
                )
                left_histogram = _subtract_histograms(
                    parent_histogram, right_histogram
                )
        else:
            # Cache misses and the private direct-both-child oracle both use
            # the established direct construction path.
            left_histogram = _build_histogram_validated(
                data=context.data,
                rows=left_rows,
                features=context.features,
                gradients=context.gradients,
                hessians=context.hessians,
                layout=context.layout,
                offsets=context.offsets,
                n_bins=context.n_bins,
                defaults=context.defaults,
                total_bins=int(context.layout.bin_offsets[-1]),
            )
            right_histogram = _build_histogram_validated(
                data=context.data,
                rows=right_rows,
                features=context.features,
                gradients=context.gradients,
                hessians=context.hessians,
                layout=context.layout,
                offsets=context.offsets,
                n_bins=context.n_bins,
                defaults=context.defaults,
                total_bins=int(context.layout.bin_offsets[-1]),
            )
        del parent_histogram

        left_split = _find_best_split_validated(
            gradients=left_histogram.gradient_sums,
            hessians=left_histogram.hessian_sums,
            counts=left_histogram.counts,
            features=context.features,
            parent_gradient=left_gradient,
            parent_hessian=left_hessian,
            parent_count=int(left_rows.size),
            n_bins=context.n_bins,
            offsets=context.offsets,
            defaults=context.defaults,
            layout=context.layout,
            min_child_samples=context.min_child_samples,
            min_child_weight=context.min_child_weight,
            min_split_gain=context.min_split_gain,
            reg_alpha=context.reg_alpha,
            reg_lambda=context.reg_lambda,
        )
        if left_split is not None:
            heapq.heappush(
                queue,
                (
                    -float(left_split.gain),
                    int(left_split.feature),
                    int(left_split.threshold_bin),
                    left_index,
                    left_split,
                ),
            )
            if histogram_cache is not None:
                histogram_cache.put(left_index, left_histogram)

        right_split = _find_best_split_validated(
            gradients=right_histogram.gradient_sums,
            hessians=right_histogram.hessian_sums,
            counts=right_histogram.counts,
            features=context.features,
            parent_gradient=right_gradient,
            parent_hessian=right_hessian,
            parent_count=int(right_rows.size),
            n_bins=context.n_bins,
            offsets=context.offsets,
            defaults=context.defaults,
            layout=context.layout,
            min_child_samples=context.min_child_samples,
            min_child_weight=context.min_child_weight,
            min_split_gain=context.min_split_gain,
            reg_alpha=context.reg_alpha,
            reg_lambda=context.reg_lambda,
        )
        if right_split is not None:
            heapq.heappush(
                queue,
                (
                    -float(right_split.gain),
                    int(right_split.feature),
                    int(right_split.threshold_bin),
                    right_index,
                    right_split,
                ),
            )
            if histogram_cache is not None:
                histogram_cache.put(right_index, right_histogram)
        # Any histogram not retained for a queued candidate can be released at
        # the end of this iteration.  ``histogram_cache`` owns retained ones.
        del left_histogram, right_histogram

    return DecisionTree(nodes=nodes, feature_indices=features)


def fit_tree(
    data: BinnedDataset,
    gradients: np.ndarray,
    hessians: np.ndarray,
    row_indices: np.ndarray,
    feature_indices: np.ndarray,
    config: LiteLightGBMConfig,
) -> DecisionTree:
    """Fit one histogram tree using leaf-wise best-first growth."""
    return _fit_tree(
        data,
        gradients,
        hessians,
        row_indices,
        feature_indices,
        config,
        use_histogram_subtraction=True,
    )


def predict_tree_raw(tree: DecisionTree, data: BinnedDataset) -> np.ndarray:
    """Return one tree's unscaled leaf output for every row."""
    # Prediction walks the array-backed tree one node at a time while keeping
    # row assignments as NumPy index arrays.  In particular, CSC columns are
    # queried with ``searchsorted``; converting the binned matrix to dense form
    # would defeat the sparse-data contract of this module.
    def _scalar_integer(value: Any, name: str) -> int:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer scalar") from exc
        if raw.ndim != 0:
            raise ValueError(f"{name} must be an integer scalar")
        dtype = raw.dtype
        if np.issubdtype(dtype, np.bool_) or not np.issubdtype(
            dtype, np.number
        ) or np.issubdtype(dtype, np.complexfloating):
            raise ValueError(f"{name} must be an integer scalar")
        if np.issubdtype(dtype, np.integer):
            number = int(raw.item())
        else:
            try:
                number = float(raw.item())
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be an integer scalar") from exc
            if not np.isfinite(number) or number != np.trunc(number):
                raise ValueError(f"{name} must be an integer scalar")
            number = int(number)
        # All integer metadata is ultimately used as a NumPy index or stored
        # in an int64 array. Reject values outside that representation before
        # any assignment can wrap or raise an uncaught OverflowError.
        int64_info = np.iinfo(np.int64)
        if number < int64_info.min or number > int64_info.max:
            raise ValueError(f"{name} must be representable as int64")
        return number

    def _scalar_float(value: Any, name: str) -> float:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite numeric scalar") from exc
        if raw.ndim != 0:
            raise ValueError(f"{name} must be a finite numeric scalar")
        dtype = raw.dtype
        if np.issubdtype(dtype, np.bool_) or not np.issubdtype(
            dtype, np.number
        ) or np.issubdtype(dtype, np.complexfloating):
            raise ValueError(f"{name} must be a finite numeric scalar")
        try:
            number = float(raw.item())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite numeric scalar") from exc
        if not np.isfinite(number):
            raise ValueError(f"{name} must be a finite numeric scalar")
        return number

    def _scalar_bool(value: Any, name: str) -> bool:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a boolean scalar") from exc
        if raw.ndim != 0 or raw.dtype != np.dtype(bool):
            raise ValueError(f"{name} must be a boolean scalar")
        return bool(raw.item())

    # Read and validate the matrix shape and sparse storage before touching
    # any tree split.  The predictor only needs the CSC view, but checking the
    # CSR shape catches a mismatched BinnedDataset early and does not densify.
    try:
        n_samples, n_features = (int(data.shape[0]), int(data.shape[1]))
        csr = data.csr
        csc = data.csc
        mapper = data.mapper
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("data must contain a valid binned dataset") from exc
    if n_samples < 0 or n_features < 0:
        raise ValueError("data must have non-negative dimensions")
    if not sp.isspmatrix_csr(csr) or not sp.isspmatrix_csc(csc):
        raise ValueError("data must contain CSR and CSC sparse views")
    if tuple(csr.shape) != (n_samples, n_features) or tuple(
        csc.shape
    ) != (n_samples, n_features):
        raise ValueError("data sparse views must match their declared shape")

    # Normalize mapper metadata.  Stored sparse values are one-based encoded
    # bins, while absent entries use ``default_bins`` (zero-based).
    try:
        raw_n_bins = np.asarray(mapper.n_bins)
        raw_defaults = np.asarray(mapper.default_bins)
        raw_cut_points = mapper.cut_points
        raw_offsets = np.asarray(mapper.bin_offsets)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("data mapper is missing bin metadata") from exc
    if (
        raw_n_bins.ndim != 1
        or raw_n_bins.size != n_features
        or raw_defaults.ndim != 1
        or raw_defaults.size != n_features
    ):
        raise ValueError("mapper must contain one bin/default entry per feature")
    try:
        cut_point_count = len(raw_cut_points)
    except (TypeError, ValueError) as exc:
        raise ValueError("mapper must contain one cut-point array per feature") from exc
    if cut_point_count != n_features:
        raise ValueError("mapper must contain one cut-point array per feature")
    if raw_offsets.ndim != 1 or raw_offsets.size != n_features + 1:
        raise ValueError("mapper bin_offsets has an invalid shape")

    n_bins = np.empty(n_features, dtype=np.int64)
    defaults = np.empty(n_features, dtype=np.int64)
    for feature in range(n_features):
        n_bins[feature] = _scalar_integer(
            raw_n_bins[feature], f"mapper n_bins[{feature}]"
        )
        if n_bins[feature] < 1:
            raise ValueError("mapper n_bins entries must be positive")
        defaults[feature] = _scalar_integer(
            raw_defaults[feature], f"mapper default_bins[{feature}]"
        )
        if defaults[feature] < 0 or defaults[feature] >= n_bins[feature]:
            raise ValueError("mapper default_bins entry is outside its bin range")
        try:
            cuts = np.asarray(raw_cut_points[feature], dtype=np.float64)
        except (TypeError, ValueError, OverflowError, IndexError) as exc:
            raise ValueError("mapper cut points must be numeric arrays") from exc
        if cuts.ndim != 1 or cuts.size != n_bins[feature] - 1:
            raise ValueError("mapper cut points do not match n_bins")
        if not np.isfinite(cuts).all() or (
            cuts.size > 1 and np.any(cuts[1:] < cuts[:-1])
        ):
            raise ValueError("mapper cut points must be finite and sorted")

    offsets = np.empty(n_features + 1, dtype=np.int64)
    for index in range(n_features + 1):
        offsets[index] = _scalar_integer(
            raw_offsets[index], f"mapper bin_offsets[{index}]"
        )
    expected_offsets = np.zeros(n_features + 1, dtype=np.int64)
    running_offset = 0
    int64_max = int(np.iinfo(np.int64).max)
    for feature in range(n_features):
        bin_count = int(n_bins[feature])
        if running_offset > int64_max - bin_count:
            raise ValueError("mapper bin offsets exceed int64 range")
        running_offset += bin_count
        expected_offsets[feature + 1] = np.int64(running_offset)
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError("mapper bin_offsets do not match n_bins")

    # Validate the CSC index arrays once.  Column row indices must be sorted
    # and unique because routing uses binary search; malformed duplicates would
    # otherwise make a row's bin ambiguous.
    try:
        indptr = np.asarray(csc.indptr)
        indices = np.asarray(csc.indices)
        encoded_values = np.asarray(csc.data)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("data CSC view has invalid sparse storage") from exc
    if (
        indptr.ndim != 1
        or indptr.size != n_features + 1
        or not np.issubdtype(indptr.dtype, np.integer)
        or indices.ndim != 1
        or encoded_values.ndim != 1
        or indices.size != encoded_values.size
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("data CSC view has invalid sparse storage")
    if indptr.size and (
        np.any(indptr < 0)
        or np.any(indptr[1:] < indptr[:-1])
        or int(indptr[-1]) != int(encoded_values.size)
    ):
        raise ValueError("data CSC column pointers are invalid")
    if encoded_values.size:
        value_dtype = encoded_values.dtype
        if (
            np.issubdtype(value_dtype, np.bool_)
            or not np.issubdtype(value_dtype, np.number)
            or np.issubdtype(value_dtype, np.complexfloating)
        ):
            raise ValueError("data CSC encoded bins must be numeric integers")
        if np.issubdtype(value_dtype, np.floating):
            if not np.isfinite(encoded_values).all() or np.any(
                encoded_values != np.trunc(encoded_values)
            ):
                raise ValueError("data CSC encoded bins must be numeric integers")
    if indices.size and (
        np.any(indices < 0) or np.any(indices >= n_samples)
    ):
        raise ValueError("data CSC row indices are outside the dataset range")

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            encoded = np.asarray(encoded_values, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "data CSC encoded bins must be representable as int64"
        ) from exc
    if np.issubdtype(encoded_values.dtype, np.unsignedinteger) and np.any(
        encoded < 0
    ):
        raise ValueError("data CSC encoded bins must be representable as int64")
    if np.issubdtype(encoded_values.dtype, np.floating):
        with np.errstate(over="ignore", invalid="ignore"):
            round_trip = encoded.astype(encoded_values.dtype)
        if np.any(round_trip != encoded_values):
            raise ValueError("data CSC encoded bins must be representable as int64")
    for feature in range(n_features):
        start, end = int(indptr[feature]), int(indptr[feature + 1])
        column_rows = indices[start:end]
        column_bins = encoded[start:end]
        if column_rows.size > 1 and np.any(column_rows[1:] <= column_rows[:-1]):
            raise ValueError("data CSC columns must have sorted unique rows")
        if column_bins.size and (
            np.any(column_bins < 1) or np.any(column_bins > n_bins[feature])
        ):
            raise ValueError("data CSC encoded bins are outside mapper ranges")

    # Normalize every node before traversal, then validate the graph as a true
    # rooted tree.  This prevents malformed child indices or cycles from
    # turning prediction into an infinite loop or an out-of-bounds write.
    try:
        nodes = tree.nodes
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("tree must contain a non-empty nodes sequence") from exc
    try:
        node_count = len(nodes)
    except (TypeError, ValueError) as exc:
        raise ValueError("tree must contain a non-empty nodes sequence") from exc
    if node_count == 0:
        raise ValueError("tree must contain at least one node")

    normalized_nodes: list[tuple[int, float, int, int, bool, int, int]] = []
    for index, node in enumerate(nodes):
        try:
            depth = _scalar_integer(node.depth, f"tree node {index} depth")
            value = _scalar_float(node.value, f"tree node {index} value")
            feature = _scalar_integer(node.feature, f"tree node {index} feature")
            threshold = _scalar_integer(
                node.threshold_bin, f"tree node {index} threshold_bin"
            )
            default_left = _scalar_bool(
                node.default_left, f"tree node {index} default_left"
            )
            left_child = _scalar_integer(
                node.left_child, f"tree node {index} left_child"
            )
            right_child = _scalar_integer(
                node.right_child, f"tree node {index} right_child"
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"tree node {index} has invalid metadata") from exc
        if depth < 0:
            raise ValueError(f"tree node {index} depth must be non-negative")
        if (left_child < 0) != (right_child < 0):
            raise ValueError(f"tree node {index} must have both or neither child")
        if left_child < -1 or right_child < -1:
            raise ValueError(f"tree node {index} child index is invalid")
        if left_child >= node_count or right_child >= node_count:
            raise ValueError(f"tree node {index} child index is out of range")
        if left_child == index or right_child == index:
            raise ValueError(f"tree node {index} cannot be its own child")
        if left_child >= 0:
            if feature < 0 or feature >= n_features:
                raise ValueError(f"tree node {index} split feature is out of range")
            if threshold < 0 or threshold >= n_bins[feature]:
                raise ValueError(
                    f"tree node {index} threshold_bin is out of range"
                )
            if default_left != (defaults[feature] <= threshold):
                raise ValueError(
                    f"tree node {index} default_left disagrees with mapper default"
                )
        normalized_nodes.append(
            (depth, value, feature, threshold, default_left, left_child, right_child)
        )

    root_depth = normalized_nodes[0][0]
    if root_depth != 0:
        raise ValueError("tree root depth must be zero")
    parent_count = np.zeros(node_count, dtype=np.int64)
    for index, (_, _, _, _, _, left_child, right_child) in enumerate(
        normalized_nodes
    ):
        if left_child >= 0:
            if left_child == 0 or right_child == 0:
                raise ValueError("tree root cannot have a parent")
            parent_count[left_child] += 1
            parent_count[right_child] += 1
            if (
                normalized_nodes[left_child][0] != normalized_nodes[index][0] + 1
                or normalized_nodes[right_child][0]
                != normalized_nodes[index][0] + 1
            ):
                raise ValueError("tree child depths must increase by one")
    if parent_count[0] != 0 or np.any(parent_count[1:] != 1):
        raise ValueError("tree nodes must form one rooted tree")
    reachable = np.zeros(node_count, dtype=bool)
    pending_nodes = [0]
    while pending_nodes:
        index = pending_nodes.pop()
        if reachable[index]:
            raise ValueError("tree contains a cycle")
        reachable[index] = True
        left_child, right_child = normalized_nodes[index][5:7]
        if left_child >= 0:
            pending_nodes.extend((left_child, right_child))
    if not np.all(reachable):
        raise ValueError("tree contains unreachable nodes")

    predictions = np.empty(n_samples, dtype=np.float64)
    initial_rows = np.arange(n_samples, dtype=np.int64)
    pending: list[tuple[int, np.ndarray]] = [(0, initial_rows)]
    while pending:
        node_index, rows = pending.pop()
        depth, value, feature, threshold, default_left, left_child, right_child = (
            normalized_nodes[node_index]
        )
        del depth  # depth was validated above; prediction does not otherwise use it.
        if left_child < 0:
            predictions[rows] = value
            continue
        if rows.size == 0:
            continue

        column_start, column_end = int(indptr[feature]), int(indptr[feature + 1])
        stored_rows = indices[column_start:column_end]
        stored_bins = encoded[column_start:column_end]
        go_left = np.full(rows.size, default_left, dtype=bool)
        if stored_rows.size:
            positions = np.searchsorted(stored_rows, rows, side="left")
            in_range = positions < stored_rows.size
            if np.any(in_range):
                safe_positions = np.minimum(positions, stored_rows.size - 1)
                matches = in_range & (stored_rows[safe_positions] == rows)
                if np.any(matches):
                    go_left[matches] = (
                        stored_bins[positions[matches]] - np.int64(1) <= threshold
                    )
        left_rows = rows[go_left]
        right_rows = rows[~go_left]
        # Push the right branch first so the left branch is processed next;
        # branch order has no numerical effect but keeps traversal deterministic.
        if right_rows.size:
            pending.append((right_child, right_rows))
        if left_rows.size:
            pending.append((left_child, left_rows))

    return predictions
