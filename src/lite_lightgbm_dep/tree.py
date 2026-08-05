"""Histogram tree building and traversal for LiteLightGBM."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .binning import BinnedDataset, BinMapper
from .core import EPSILON, LiteLightGBMConfig, soft_threshold

@dataclass(slots=True)
class Histogram:
    """Flattened per-bin gradient, Hessian, and exact-count statistics."""

    gradient_sums: np.ndarray
    hessian_sums: np.ndarray
    counts: np.ndarray


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


def build_histogram(
    data: BinnedDataset,
    row_indices: np.ndarray,
    feature_indices: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
) -> Histogram:
    """Aggregate one leaf's per-bin gradients, Hessians, and row counts."""
    # Keep the flattened allocation independent of the selected row/feature
    # subsets.  In particular, an empty subset still returns one zero-filled
    # segment for every bin in the mapper, which lets callers reuse the same
    # histogram layout throughout tree growth.
    try:
        n_samples, n_features = (int(data.shape[0]), int(data.shape[1]))
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

    # The adjacent-difference check implies this equality, but keep the final
    # prefix total explicit and compute it in Python integers to avoid int64
    # summation overflow on malformed metadata.
    expected_total = sum(int(value) for value in raw_n_bins)
    if int(raw_offsets[-1]) != expected_total:
        raise ValueError("data mapper final bin offset is inconsistent")
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

    gradient_sums = np.zeros(total_bins, dtype=np.float64)
    hessian_sums = np.zeros(total_bins, dtype=np.float64)
    counts = np.zeros(total_bins, dtype=np.int64)
    if rows.size == 0 or features.size == 0:
        return Histogram(
            gradient_sums=gradient_sums,
            hessian_sums=hessian_sums,
            counts=counts,
        )

    # Unique rows plus multiplicities lets this routine remain well-defined
    # even if a caller supplies a repeated row index.  Normal tree partitions
    # are unique, so this preprocessing is linear in the selected leaf size and
    # avoids any Python loop over samples.
    unique_rows, row_multiplicity = np.unique(rows, return_counts=True)
    leaf_gradient = float(np.sum(gradient_values[rows], dtype=np.float64))
    leaf_hessian = float(np.sum(hessian_values[rows], dtype=np.float64))
    csc = data.csc

    for feature in features:
        feature_index = int(feature)
        start = int(raw_offsets[feature_index])
        end = int(raw_offsets[feature_index + 1])
        n_feature_bins = end - start
        if n_feature_bins <= 0:
            # A malformed mapper should not make indexed writes wrap around.
            raise ValueError("data mapper must assign at least one bin per feature")

        column_start = int(csc.indptr[feature_index])
        column_end = int(csc.indptr[feature_index + 1])
        column_rows = np.asarray(csc.indices[column_start:column_end], dtype=np.int64)
        if column_rows.size:
            # CSC rows are sorted by transform_bins.  searchsorted against the
            # sorted unique selected rows provides vectorized membership and
            # naturally preserves multiplicity for repeated row_indices.
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
                gradient_contrib = gradient_values[matched_rows] * multiplicities
                hessian_contrib = hessian_values[matched_rows] * multiplicities
                np.add.at(
                    gradient_sums,
                    start + bin_ids,
                    gradient_contrib,
                )
                np.add.at(
                    hessian_sums,
                    start + bin_ids,
                    hessian_contrib,
                )
                np.add.at(counts, start + bin_ids, multiplicities)

        # Sparse storage omits rows in the feature's default bin.  Whatever
        # selected rows were not represented explicitly therefore contribute
        # their sufficient-statistic remainder to that one bin.
        segment = slice(start, end)
        represented_gradient = float(np.sum(gradient_sums[segment], dtype=np.float64))
        represented_hessian = float(np.sum(hessian_sums[segment], dtype=np.float64))
        represented_count = int(np.sum(counts[segment], dtype=np.int64))
        default_bin = int(raw_defaults[feature_index])
        if default_bin < 0 or default_bin >= n_feature_bins:
            raise ValueError("data mapper default bin is outside feature bin range")
        default_offset = start + default_bin
        gradient_sums[default_offset] += leaf_gradient - represented_gradient
        hessian_sums[default_offset] += leaf_hessian - represented_hessian
        counts[default_offset] += np.int64(rows.size - represented_count)

    return Histogram(
        gradient_sums=gradient_sums,
        hessian_sums=hessian_sums,
        counts=counts,
    )


def find_best_split(
    histogram: Histogram,
    feature_indices: np.ndarray,
    parent_gradient: float,
    parent_hessian: float,
    parent_count: int,
    mapper: BinMapper,
    config: LiteLightGBMConfig,
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
    total_bins = int(offsets[-1])
    if total_bins != gradients.size:
        raise ValueError("histogram arrays do not match mapper bin layout")
    if np.any(defaults < 0) or np.any(defaults >= n_bins):
        raise ValueError("mapper default bins are outside feature bin ranges")

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

    # Score uses the same L1/L2 regularized sufficient-statistic formula as leaf
    # values.  A non-positive denominator is deliberately assigned score zero,
    # including saturated zero-curvature leaves.
    parent_denominator = parent_hessian_value + reg_lambda
    if parent_denominator > EPSILON:
        parent_thresholded = float(soft_threshold(parent_gradient_value, reg_alpha))
        parent_score = (parent_thresholded * parent_thresholded) / parent_denominator
    else:
        parent_score = 0.0

    best: SplitInfo | None = None
    best_gain = -np.inf
    best_feature = n_features
    best_threshold = int(np.max(n_bins)) if n_bins.size else 1

    for feature_value in features:
        feature = int(feature_value)
        start = int(offsets[feature])
        end = int(offsets[feature + 1])
        feature_bin_count = end - start
        if feature_bin_count <= 1:
            continue

        # Prefix sums turn each boundary into O(1) sufficient-statistic lookup;
        # the final bin is intentionally never used as a threshold.
        gradient_prefix = np.cumsum(gradients[start:end], dtype=np.float64)
        hessian_prefix = np.cumsum(hessians[start:end], dtype=np.float64)
        count_prefix = np.cumsum(counts[start:end], dtype=np.int64)
        for threshold in range(feature_bin_count - 1):
            left_count = int(count_prefix[threshold])
            right_count = parent_count_value - left_count
            if left_count < min_child_samples or right_count < min_child_samples:
                continue

            left_gradient = float(gradient_prefix[threshold])
            left_hessian = float(hessian_prefix[threshold])
            right_gradient = parent_gradient_value - left_gradient
            right_hessian = parent_hessian_value - left_hessian
            if (
                left_hessian < min_child_weight
                or right_hessian < min_child_weight
            ):
                continue

            left_denominator = left_hessian + reg_lambda
            right_denominator = right_hessian + reg_lambda
            if left_denominator > EPSILON:
                left_thresholded = float(soft_threshold(left_gradient, reg_alpha))
                left_score = (
                    left_thresholded * left_thresholded
                ) / left_denominator
            else:
                left_score = 0.0
            if right_denominator > EPSILON:
                right_thresholded = float(soft_threshold(right_gradient, reg_alpha))
                right_score = (
                    right_thresholded * right_thresholded
                ) / right_denominator
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
                        or (
                            feature == best_feature
                            and threshold < best_threshold
                        )
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
                default_left=bool(int(defaults[feature]) <= threshold),
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

    if rows.size == 0:
        return rows.copy(), rows.copy()

    # Map each distinct selected row once, then expand through ``inverse``.
    # This preserves the caller's ordering (and any repeated row indices)
    # without a Python loop over samples.
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


def fit_tree(
    data: BinnedDataset,
    gradients: np.ndarray,
    hessians: np.ndarray,
    row_indices: np.ndarray,
    feature_indices: np.ndarray,
    config: LiteLightGBMConfig,
) -> DecisionTree:
    """Fit one histogram tree using leaf-wise best-first growth."""
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
    root_histogram = build_histogram(
        data,
        rows,
        features,
        gradient_values,
        hessian_values,
    )
    if max_depth <= 0 or 0 < max_depth:
        root_split = find_best_split(
            root_histogram,
            features,
            root_gradient,
            root_hessian,
            int(rows.size),
            data.mapper,
            config,
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

    while queue and leaves < num_leaves:
        _, _, _, node_index, split = heapq.heappop(queue)
        current_rows = node_rows[node_index]
        if current_rows is None:
            # Defensive guard for malformed/stale candidates; normal growth
            # enqueues each leaf exactly once and never reaches this branch.
            continue
        parent = nodes[node_index]
        if max_depth > 0 and parent.depth >= max_depth:
            continue

        left_rows, right_rows = partition_rows(data, current_rows, split)
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
            continue
        child_depth = parent.depth + 1
        if max_depth > 0 and child_depth >= max_depth:
            continue

        left_histogram = build_histogram(
            data,
            left_rows,
            features,
            gradient_values,
            hessian_values,
        )
        left_split = find_best_split(
            left_histogram,
            features,
            left_gradient,
            left_hessian,
            int(left_rows.size),
            data.mapper,
            config,
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

        right_histogram = build_histogram(
            data,
            right_rows,
            features,
            gradient_values,
            hessian_values,
        )
        right_split = find_best_split(
            right_histogram,
            features,
            right_gradient,
            right_hessian,
            int(right_rows.size),
            data.mapper,
            config,
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

    return DecisionTree(nodes=nodes, feature_indices=features)


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
