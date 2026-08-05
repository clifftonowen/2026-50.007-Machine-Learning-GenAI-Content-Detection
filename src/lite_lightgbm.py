"""From-scratch, LightGBM-like binary classifier for dense and sparse data.

The completed module depends only on NumPy, SciPy, and the Python standard library.
Numerical, binning, training, and prediction routines implement the documented
scope and validation contract.

See ``src/lite_lightgbm_docs.md`` for the API reference and
``src/lite_lightgbm.md`` for the implementation contract.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.sparse as sp


Matrix = np.ndarray | sp.spmatrix
ClassWeight = str | dict[int, float] | None
EPSILON = 1e-15


@dataclass(frozen=True, slots=True)
class LiteLightGBMConfig:
    """Immutable snapshot of the estimator's supported hyperparameters."""

    n_estimators: int = 100
    learning_rate: float = 0.1
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    min_child_weight: float = 1e-3
    min_split_gain: float = 0.0
    max_bin: int = 255
    min_data_in_bin: int = 3
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    colsample_bytree: float = 1.0
    subsample: float = 1.0
    subsample_freq: int = 0
    class_weight: ClassWeight = None
    random_state: int | None = None


@dataclass(frozen=True, slots=True)
class BinMapper:
    """Learned cut points and flattened histogram layout for all features."""

    cut_points: tuple[np.ndarray, ...]
    default_bins: np.ndarray
    n_bins: np.ndarray
    bin_offsets: np.ndarray


@dataclass(frozen=True, slots=True)
class BinnedDataset:
    """Quantized data stored as equivalent row- and column-oriented sparse views.

    Stored values are ``bin_id + 1``; implicit sparse zeros use each feature's
    ``mapper.default_bins`` entry.
    """

    csr: sp.csr_matrix
    csc: sp.csc_matrix
    mapper: BinMapper

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(n_samples, n_features)``."""
        return self.csr.shape


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


def sigmoid(raw_scores: np.ndarray) -> np.ndarray:
    """Compute a numerically stable element-wise logistic sigmoid."""
    # Evaluate positive and negative inputs separately so that neither
    # exponential receives a positive argument large enough to overflow.
    scores = np.asarray(raw_scores)
    result = np.empty_like(scores, dtype=np.float64)

    positive = scores >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))

    negative = ~positive
    exp_scores = np.exp(scores[negative])
    result[negative] = exp_scores / (1.0 + exp_scores)
    return result


def soft_threshold(values: np.ndarray | float, reg_alpha: float):
    """Apply the L1 soft-threshold operator used by leaf values and gains."""
    return np.sign(values) * np.maximum(np.abs(values) - reg_alpha, 0.0)


def binary_gradients_hessians(
    raw_scores: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted binary-log-loss gradients and Hessians."""
    probabilities = sigmoid(raw_scores)
    gradients = sample_weight * (probabilities - labels)
    hessians = sample_weight * probabilities * (1.0 - probabilities)
    return gradients, hessians


def _find_bin_boundaries(
    values: np.ndarray,
    counts: np.ndarray,
    n_samples: int,
    max_bin: int,
    min_data_in_bin: int,
) -> tuple[np.ndarray, int]:
    """Select deterministic cut points for one sorted feature.

    ``values`` and ``counts`` are produced by :func:`fit_bin_mapper`, so they
    are already sorted distinct values and matching positive occurrence counts.
    The candidate-boundary scan is expressed with NumPy indexing while the
    outer boundary loop remains sequential because each chosen boundary limits
    the feasible range for the next one.
    """
    n_distinct = int(values.size)
    # The upper bound is deliberately frequency based: each row, including
    # implicit sparse zeros, contributes one occurrence to n_samples.
    max_feasible_bins = min(
        max_bin,
        n_distinct,
        max(1, n_samples // min_data_in_bin),
    )
    cumulative = np.cumsum(counts, dtype=np.int64)
    all_boundary_indices = np.arange(n_distinct - 1, dtype=np.int64)

    selected_boundaries: list[int] | None = None
    selected_bin_count = 1
    # Start at the largest allowed B.  If a complete set of B-1 boundaries
    # cannot be selected under the child-size constraints, retry with one
    # fewer bin.  A single-bin feature is always valid.
    for candidate_bins in range(max_feasible_bins, 0, -1):
        if candidate_bins == 1:
            selected_boundaries = []
            selected_bin_count = 1
            break

        boundaries: list[int] = []
        previous_boundary = -1
        previous_count = 0
        success = True
        for k in range(1, candidate_bins):
            desired_rank = (float(k) * float(n_samples)) / float(candidate_bins)
            # Leave enough distinct values for the unfinished bins.  The
            # final permissible boundary is therefore n_distinct-2, while
            # earlier boundaries must leave progressively more values.
            last_boundary = n_distinct - (candidate_bins - k) - 1
            if last_boundary < previous_boundary + 1:
                success = False
                break

            lower = previous_boundary + 1
            candidates = all_boundary_indices[lower : last_boundary + 1]
            left_counts = cumulative[candidates] - previous_count
            remaining_count = n_samples - cumulative[candidates]
            remaining_bins = candidate_bins - k
            feasible = (
                (left_counts >= min_data_in_bin)
                & (remaining_count >= remaining_bins * min_data_in_bin)
            )
            feasible_candidates = candidates[feasible]
            if feasible_candidates.size == 0:
                success = False
                break

            # np.argmin is stable for equal values; candidates are in
            # ascending value/count order, so ties go to the lower-valued
            # boundary as required by the contract.
            distances = np.abs(
                np.asarray(cumulative[feasible_candidates], dtype=np.float64)
                - desired_rank
            )
            chosen = int(feasible_candidates[int(np.argmin(distances))])
            boundaries.append(chosen)
            previous_boundary = chosen
            previous_count = int(cumulative[chosen])

        if success and len(boundaries) == candidate_bins - 1:
            selected_boundaries = boundaries
            selected_bin_count = candidate_bins
            break

    # max_feasible_bins is at least one for a non-empty feature, and B=1
    # above always succeeds.  Keep this guard for defensive clarity if the
    # implementation is later changed to permit empty features.
    if selected_boundaries is None:  # pragma: no cover
        selected_boundaries = []
        selected_bin_count = 1

    if selected_boundaries:
        cuts = np.asarray(
            [values[index] for index in selected_boundaries],
            dtype=np.float64,
        )
    else:
        cuts = np.empty(0, dtype=np.float64)
    return cuts, selected_bin_count


def fit_bin_mapper(X: Matrix, config: LiteLightGBMConfig) -> BinMapper:
    """Learn deterministic numeric bins without densifying sparse input."""
    # Keep this routine independent of the estimator's validation path.  The
    # mapper is also useful in focused tests, and accepting only finite real
    # numeric matrices here prevents subtle object/complex sorting behaviour.
    sparse_input = sp.issparse(X)
    if sparse_input:
        if len(X.shape) != 2:  # pragma: no cover - scipy matrices are 2-D
            raise ValueError("X must be a two-dimensional matrix")
        n_samples, n_features = (int(X.shape[0]), int(X.shape[1]))
        if n_samples <= 0 or n_features <= 0:
            raise ValueError("X must be non-empty")

        dtype = np.dtype(X.dtype)
        is_numeric = np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.bool_
        )
        if not is_numeric or np.issubdtype(dtype, np.complexfloating):
            raise TypeError("X must contain real numeric values")

        # Work on a private CSC copy.  In particular, summing duplicate
        # coordinates before counting values makes CSR/CSC input equivalent to
        # its logical dense representation without mutating the caller's data.
        matrix = X.tocsc(copy=True)
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64, copy=False)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.isfinite(matrix.data).all():
            raise ValueError("X must contain only finite values")
    else:
        raw = np.asarray(X)
        if raw.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")
        n_samples, n_features = (int(raw.shape[0]), int(raw.shape[1]))
        if n_samples <= 0 or n_features <= 0:
            raise ValueError("X must be non-empty")
        dtype = np.dtype(raw.dtype)
        is_numeric = np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.bool_
        )
        if not is_numeric or np.issubdtype(dtype, np.complexfloating):
            raise TypeError("X must contain real numeric values")
        matrix = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("X must contain only finite values")

    try:
        max_bin = int(config.max_bin)
        min_data_in_bin = int(config.min_data_in_bin)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_bin and min_data_in_bin must be integers") from exc
    if max_bin < 1:
        raise ValueError("max_bin must be positive")
    if min_data_in_bin < 1:
        raise ValueError("min_data_in_bin must be positive")

    # A feature's values are represented by sorted distinct values and integer
    # occurrence counts.  Sparse matrices need implicit zero occurrences added
    # explicitly; those zeros are part of the quantile population, not missing
    # values.
    value_counts: list[tuple[np.ndarray, np.ndarray]] = []
    if sparse_input:
        csc = matrix
        for feature in range(n_features):
            start, end = int(csc.indptr[feature]), int(csc.indptr[feature + 1])
            stored = np.asarray(csc.data[start:end], dtype=np.float64)
            values, counts = np.unique(stored, return_counts=True)
            values = np.asarray(values, dtype=np.float64)
            counts = np.asarray(counts, dtype=np.int64)

            implicit_zeros = n_samples - stored.size
            if implicit_zeros:
                zero_at = int(np.searchsorted(values, 0.0, side="left"))
                if zero_at < values.size and values[zero_at] == 0.0:
                    counts[zero_at] += np.int64(implicit_zeros)
                else:
                    values = np.insert(values, zero_at, 0.0)
                    counts = np.insert(
                        counts, zero_at, np.int64(implicit_zeros)
                    ).astype(np.int64, copy=False)
            value_counts.append((values, counts))
    else:
        dense = matrix
        for feature in range(n_features):
            values, counts = np.unique(dense[:, feature], return_counts=True)
            value_counts.append(
                (
                    np.asarray(values, dtype=np.float64),
                    np.asarray(counts, dtype=np.int64),
                )
            )

    cut_points: list[np.ndarray] = []
    default_bins = np.empty(n_features, dtype=np.int64)
    n_bins = np.empty(n_features, dtype=np.int64)

    for feature, (values, counts) in enumerate(value_counts):
        cuts, selected_bin_count = _find_bin_boundaries(
            values,
            counts,
            n_samples,
            max_bin,
            min_data_in_bin,
        )
        cut_points.append(cuts)
        n_bins[feature] = np.int64(selected_bin_count)
        default_bins[feature] = np.int64(
            np.searchsorted(cuts, 0.0, side="left")
        )

    # Histogram segments use an explicitly signed 64-bit prefix layout.  This
    # remains stable even when the total number of bins exceeds 2**31.
    bin_offsets = np.zeros(n_features + 1, dtype=np.int64)
    bin_offsets[1:] = np.cumsum(n_bins, dtype=np.int64)
    return BinMapper(
        cut_points=tuple(cut_points),
        default_bins=default_bins,
        n_bins=n_bins,
        bin_offsets=bin_offsets,
    )


def transform_bins(X: Matrix, mapper: BinMapper) -> BinnedDataset:
    """Quantize input with a fitted mapper and return sparse CSR/CSC views."""
    # Keep the validation here independent from the estimator so this helper can
    # also be used directly in focused tests.  In particular, sparse values are
    # checked before converting to floating point and are never densified.
    sparse_input = sp.issparse(X)
    if sparse_input:
        if len(X.shape) != 2:  # pragma: no cover - scipy matrices are 2-D
            raise ValueError("X must be a two-dimensional matrix")
        n_samples, n_features = (int(X.shape[0]), int(X.shape[1]))
        dtype = np.dtype(X.dtype)
        is_numeric = np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.bool_
        )
        if not is_numeric or np.issubdtype(dtype, np.complexfloating):
            raise TypeError("X must contain real numeric values")

        # Work on a private canonical CSC copy.  Summing duplicates before
        # mapping is important: binning is defined over logical matrix values,
        # not over individual sparse entries.  Explicit zeros are removed so
        # that an input's storage details do not leak into the output.
        matrix = X.tocsc(copy=True)
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64, copy=False)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.isfinite(matrix.data).all():
            raise ValueError("X must contain only finite values")
    else:
        raw = np.asarray(X)
        if raw.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")
        n_samples, n_features = (int(raw.shape[0]), int(raw.shape[1]))
        dtype = np.dtype(raw.dtype)
        is_numeric = np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.bool_
        )
        if not is_numeric or np.issubdtype(dtype, np.complexfloating):
            raise TypeError("X must contain real numeric values")
        matrix = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("X must contain only finite values")

    # A mapper is fitted for a fixed feature count.  Keep the feature-count
    # check explicit so malformed inputs fail before any sparse allocation.
    try:
        mapper_cut_points = mapper.cut_points
        mapper_n_features = len(mapper_cut_points)
        raw_defaults = np.asarray(mapper.default_bins)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("mapper must contain one default bin per feature") from exc
    if (
        n_features != mapper_n_features
        or raw_defaults.ndim != 1
        or raw_defaults.size != n_features
    ):
        raise ValueError("X has a different number of features than mapper")

    # Validate defaults before coercing to int64.  In particular, direct
    # coercion would silently truncate fractional values and can wrap very
    # large unsigned values.  Defaults are real, finite, integer-valued bin
    # indices; their per-feature bounds are checked after normalizing cuts.
    default_dtype = raw_defaults.dtype
    is_numeric = np.issubdtype(default_dtype, np.number)
    if not is_numeric or np.issubdtype(default_dtype, np.complexfloating):
        raise ValueError(
            "mapper default bins must be finite integer-valued numeric values"
        )
    if not np.isfinite(raw_defaults).all():
        raise ValueError(
            "mapper default bins must be finite integer-valued numeric values"
        )
    if np.issubdtype(default_dtype, np.floating) and np.any(
        raw_defaults != np.trunc(raw_defaults)
    ):
        raise ValueError(
            "mapper default bins must be finite integer-valued numeric values"
        )

    # Validate/normalize cut arrays once.  Fitted mappers always provide finite,
    # ascending one-dimensional arrays; these checks make an incompatible
    # hand-built mapper fail clearly rather than producing subtly invalid bins.
    cuts_by_feature: list[np.ndarray] = []
    for cuts in mapper_cut_points:
        try:
            normalized = np.asarray(cuts, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("mapper cut points must be real numeric arrays") from exc
        if normalized.ndim != 1:
            raise ValueError("mapper cut points must be one-dimensional")
        if not np.isfinite(normalized).all():
            raise ValueError("mapper cut points must be finite")
        if normalized.size > 1 and np.any(normalized[1:] < normalized[:-1]):
            raise ValueError("mapper cut points must be sorted")
        cuts_by_feature.append(normalized)

    defaults = np.empty(n_features, dtype=np.int64)
    for feature, (default, cuts) in enumerate(
        zip(raw_defaults, cuts_by_feature)
    ):
        if default < 0 or default > cuts.size:
            raise ValueError(
                "mapper default bins must be within each feature's bin range"
            )
        # The bounds above guarantee that conversion is representable in the
        # signed 64-bit array used by the sparse mapping paths.
        defaults[feature] = int(default)

    if sparse_input:
        # Build the output directly in CSC form, preserving sparsity even when
        # the source has millions of rows/features.  Only entries whose mapped
        # bin differs from the implicit-zero default are represented.
        output_indptr = np.empty(n_features + 1, dtype=np.int64)
        output_indptr[0] = 0
        output_indices_parts: list[np.ndarray] = []
        output_data_parts: list[np.ndarray] = []

        for feature in range(n_features):
            start = int(matrix.indptr[feature])
            end = int(matrix.indptr[feature + 1])
            values = np.asarray(matrix.data[start:end], dtype=np.float64)
            bins = np.asarray(
                np.searchsorted(
                    cuts_by_feature[feature], values, side="left"
                ),
                dtype=np.int64,
            )
            keep = bins != defaults[feature]
            if np.any(keep):
                output_indices_parts.append(
                    np.asarray(matrix.indices[start:end][keep], dtype=np.int64)
                )
                output_data_parts.append(bins[keep] + np.int64(1))
            output_indptr[feature + 1] = np.int64(
                output_indptr[feature]
                + int(np.count_nonzero(keep))
            )

        if output_data_parts:
            output_indices = np.concatenate(output_indices_parts)
            output_data = np.concatenate(output_data_parts)
        else:
            output_indices = np.empty(0, dtype=np.int64)
            output_data = np.empty(0, dtype=np.int64)

        csc = sp.csc_matrix(
            (output_data, output_indices, output_indptr),
            shape=(n_samples, n_features),
        )
        # The construction above is already duplicate-free and sorted because
        # the source CSC was canonicalized.  Keep these calls as defensive
        # guarantees for all SciPy sparse implementations.
        csc.sum_duplicates()
        csc.eliminate_zeros()
        csc.sort_indices()
    else:
        # Dense input may be mapped in a temporary dense integer array; sparse
        # inputs take the direct CSC path above and are never converted here.
        bins_matrix = np.empty((n_samples, n_features), dtype=np.int64)
        for feature in range(n_features):
            bins_matrix[:, feature] = np.asarray(
                np.searchsorted(
                    cuts_by_feature[feature],
                    matrix[:, feature],
                    side="left",
                ),
                dtype=np.int64,
            )

        rows, columns = np.nonzero(bins_matrix != defaults[np.newaxis, :])
        if rows.size:
            values = bins_matrix[rows, columns] + np.int64(1)
        else:
            values = np.empty(0, dtype=np.int64)
        csc = sp.csc_matrix(
            (values, (rows, columns)),
            shape=(n_samples, n_features),
            dtype=np.int64,
        )
        csc.sum_duplicates()
        csc.eliminate_zeros()
        csc.sort_indices()

    csr = csc.tocsr(copy=True)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    return BinnedDataset(csr=csr, csc=csc, mapper=mapper)


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


class LiteLightGBM:
    """From-scratch histogram gradient-boosted binary classifier.

    See ``src/lite_lightgbm_docs.md`` for parameters, learned attributes, input
    validation, and prediction semantics. Fitting and prediction implement the
    documented training and inference flow.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        min_child_weight: float = 1e-3,
        min_split_gain: float = 0.0,
        max_bin: int = 255,
        min_data_in_bin: int = 3,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        colsample_bytree: float = 1.0,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        class_weight: ClassWeight = None,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.max_bin = max_bin
        self.min_data_in_bin = min_data_in_bin
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.class_weight = class_weight
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all constructor parameters for cloning and tuning."""
        del deep
        return {
            name: getattr(self, name)
            for name in LiteLightGBMConfig.__dataclass_fields__
        }

    def set_params(self, **params: Any) -> LiteLightGBM:
        """Set known constructor parameters and return ``self``."""
        valid = set(self.get_params())
        unknown = set(params) - valid
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown parameter(s): {names}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def __sklearn_tags__(self) -> SimpleNamespace:
        """Return local metadata for optional external CV utilities."""
        return SimpleNamespace(
            estimator_type="classifier",
            input_tags=SimpleNamespace(pairwise=False),
        )

    def fit(
        self,
        X: Matrix,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> LiteLightGBM:
        """Fit the bin mapper and sequential leaf-wise boosted trees."""
        config = self._config()

        # Validate the random seed before any mapper or binned-data allocation.
        # ``random_state`` follows its declared ``int | None`` type strictly:
        # booleans, non-integer scalars, arrays, and invalid seed values are
        # rejected with a clear error.  Normalize NumPy integer scalars so the
        # local generator receives a plain Python integer (or ``None``).
        raw_random_state = config.random_state
        if raw_random_state is None:
            normalized_random_state: int | None = None
        else:
            # Check the scalar type directly instead of routing through
            # ``np.asarray``: arbitrary-size Python integers (for example,
            # ``2**100``) become object arrays and would otherwise be rejected.
            # Arrays, booleans, and non-integer scalars remain invalid.
            if isinstance(raw_random_state, (bool, np.bool_)) or not isinstance(
                raw_random_state, (int, np.integer)
            ):
                raise ValueError("random_state must be None or an integer scalar")
            try:
                normalized_random_state = int(raw_random_state)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "random_state must be None or an integer scalar"
                ) from exc
            if normalized_random_state < 0:
                raise ValueError(
                    "random_state must be None or a non-negative integer seed"
                )
        try:
            rng = np.random.default_rng(normalized_random_state)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "random_state must be None or a valid non-negative integer seed"
            ) from exc

        # Validate every configuration field before the mapper (and therefore
        # any flattened bin/histogram allocation) is constructed.  Integer
        # hyperparameters accept integer-valued real scalars, but not strings,
        # booleans, complex values, arrays, or non-finite numbers.
        integer_parameters = (
            ("n_estimators", config.n_estimators, 1),
            ("num_leaves", config.num_leaves, 2),
            ("max_bin", config.max_bin, 2),
            ("min_data_in_bin", config.min_data_in_bin, 1),
            ("min_child_samples", config.min_child_samples, 0),
            ("subsample_freq", config.subsample_freq, 0),
        )
        normalized_integers: dict[str, int] = {}
        for name, value, lower_bound in integer_parameters:
            try:
                raw_value = np.asarray(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be an integer scalar") from exc
            if raw_value.ndim != 0:
                raise ValueError(f"{name} must be an integer scalar")
            dtype = raw_value.dtype
            if (
                np.issubdtype(dtype, np.bool_)
                or not np.issubdtype(dtype, np.number)
                or np.issubdtype(dtype, np.complexfloating)
            ):
                raise ValueError(f"{name} must be an integer scalar")
            try:
                numeric_value = float(raw_value.item())
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be an integer scalar") from exc
            if (
                not np.isfinite(numeric_value)
                or numeric_value != np.trunc(numeric_value)
                or numeric_value < lower_bound
            ):
                raise ValueError(
                    f"{name} must be an integer at least {lower_bound}"
                )
            normalized_integers[name] = int(numeric_value)

        # max_depth uses non-positive values for unlimited growth; a positive
        # value is still required to be an integer depth.
        try:
            raw_max_depth = np.asarray(config.max_depth)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_depth must be an integer scalar") from exc
        if raw_max_depth.ndim != 0:
            raise ValueError("max_depth must be an integer scalar")
        max_depth_dtype = raw_max_depth.dtype
        if (
            np.issubdtype(max_depth_dtype, np.bool_)
            or not np.issubdtype(max_depth_dtype, np.number)
            or np.issubdtype(max_depth_dtype, np.complexfloating)
        ):
            raise ValueError("max_depth must be an integer scalar")
        try:
            numeric_max_depth = float(raw_max_depth.item())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_depth must be an integer scalar") from exc
        if (
            not np.isfinite(numeric_max_depth)
            or numeric_max_depth != np.trunc(numeric_max_depth)
        ):
            raise ValueError("max_depth must be an integer scalar")
        normalized_max_depth = int(numeric_max_depth)
        # Continuous configuration fields must be finite and non-negative (or,
        # for the two sampling fractions, strictly positive and at most one).
        continuous_parameters = (
            ("learning_rate", config.learning_rate, 0.0, None),
            ("min_child_weight", config.min_child_weight, 0.0, None),
            ("min_split_gain", config.min_split_gain, 0.0, None),
            ("reg_alpha", config.reg_alpha, 0.0, None),
            ("reg_lambda", config.reg_lambda, 0.0, None),
            ("colsample_bytree", config.colsample_bytree, 0.0, 1.0),
            ("subsample", config.subsample, 0.0, 1.0),
        )
        normalized_continuous: dict[str, float] = {}
        for name, value, lower_bound, upper_bound in continuous_parameters:
            try:
                raw_value = np.asarray(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite numeric scalar") from exc
            if raw_value.ndim != 0:
                raise ValueError(f"{name} must be a finite numeric scalar")
            dtype = raw_value.dtype
            if (
                np.issubdtype(dtype, np.bool_)
                or not np.issubdtype(dtype, np.number)
                or np.issubdtype(dtype, np.complexfloating)
            ):
                raise ValueError(f"{name} must be a finite numeric scalar")
            try:
                numeric_value = float(raw_value.item())
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite numeric scalar") from exc
            if not np.isfinite(numeric_value):
                raise ValueError(f"{name} must be a finite numeric scalar")
            if name in ("colsample_bytree", "subsample"):
                if not (numeric_value > lower_bound and numeric_value <= upper_bound):
                    raise ValueError(f"{name} must satisfy 0 < value <= 1")
            elif name == "learning_rate":
                if numeric_value <= lower_bound:
                    raise ValueError("learning_rate must be positive and finite")
            elif numeric_value < lower_bound:
                raise ValueError(f"{name} must be finite and non-negative")
            normalized_continuous[name] = numeric_value

        class_weight = config.class_weight
        if class_weight is not None and class_weight != "balanced":
            if not isinstance(class_weight, dict) or set(class_weight) != {0, 1}:
                raise ValueError(
                    "class_weight must be None, 'balanced', or a dictionary with keys 0 and 1"
                )
            for class_label in (0, 1):
                try:
                    raw_weight = np.asarray(class_weight[class_label])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "class_weight values must be finite and non-negative"
                    ) from exc
                if raw_weight.ndim != 0:
                    raise ValueError(
                        "class_weight values must be finite and non-negative"
                    )
                weight_dtype = raw_weight.dtype
                if (
                    np.issubdtype(weight_dtype, np.bool_)
                    or not np.issubdtype(weight_dtype, np.number)
                    or np.issubdtype(weight_dtype, np.complexfloating)
                ):
                    raise ValueError(
                        "class_weight values must be finite and non-negative"
                    )
                try:
                    numeric_weight = float(raw_weight.item())
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "class_weight values must be finite and non-negative"
                    ) from exc
                if not np.isfinite(numeric_weight) or numeric_weight < 0:
                    raise ValueError(
                        "class_weight values must be finite and non-negative"
                    )
        elif class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None, 'balanced', or a dictionary")

        # Validate dense/sparse feature storage without ever densifying sparse
        # input.  The mapper performs the same checks defensively, but these are
        # intentionally done first so invalid data cannot trigger its allocations.
        sparse_input = sp.issparse(X)
        if sparse_input:
            try:
                n_samples = int(X.shape[0])
                n_features = int(X.shape[1])
                feature_dtype = np.dtype(X.dtype)
                stored_values = np.asarray(X.data)
            except (AttributeError, TypeError, ValueError, IndexError) as exc:
                raise ValueError("X must be a two-dimensional numeric matrix") from exc
            if n_samples <= 0 or n_features <= 0:
                raise ValueError("X must be non-empty")
            if (
                not np.issubdtype(feature_dtype, np.number)
                and not np.issubdtype(feature_dtype, np.bool_)
            ) or np.issubdtype(feature_dtype, np.complexfloating):
                raise TypeError("X must contain real numeric values")
            try:
                finite_values = np.asarray(stored_values, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("X must contain real numeric values") from exc
            if not np.isfinite(finite_values).all():
                raise ValueError("X must contain only finite values")
        else:
            try:
                dense_values = np.asarray(X)
            except (TypeError, ValueError) as exc:
                raise ValueError("X must be a two-dimensional numeric matrix") from exc
            if dense_values.ndim != 2:
                raise ValueError("X must be a two-dimensional matrix")
            n_samples, n_features = (
                int(dense_values.shape[0]),
                int(dense_values.shape[1]),
            )
            if n_samples <= 0 or n_features <= 0:
                raise ValueError("X must be non-empty")
            feature_dtype = dense_values.dtype
            if (
                not np.issubdtype(feature_dtype, np.number)
                and not np.issubdtype(feature_dtype, np.bool_)
            ) or np.issubdtype(feature_dtype, np.complexfloating):
                raise TypeError("X must contain real numeric values")
            try:
                finite_values = np.asarray(dense_values, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("X must contain real numeric values") from exc
            if not np.isfinite(finite_values).all():
                raise ValueError("X must contain only finite values")

        # Labels are normalized only after shape, dtype, finiteness, and binary
        # membership checks.  Both classes are required by the binary objective.
        try:
            labels = np.asarray(y)
        except (TypeError, ValueError) as exc:
            raise ValueError("y must be a one-dimensional binary array") from exc
        if labels.ndim != 1 or labels.size != n_samples:
            raise ValueError("y must be a one-dimensional array matching X")
        label_dtype = labels.dtype
        if (
            not np.issubdtype(label_dtype, np.number)
            and not np.issubdtype(label_dtype, np.bool_)
        ) or np.issubdtype(label_dtype, np.complexfloating):
            raise TypeError("y must contain real binary labels 0 and 1")
        try:
            labels_float = np.asarray(labels, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("y must contain real binary labels 0 and 1") from exc
        if not np.isfinite(labels_float).all() or not np.all(
            (labels_float == 0.0) | (labels_float == 1.0)
        ):
            raise ValueError("y must contain exactly binary labels 0 and 1")
        if not np.any(labels_float == 0.0) or not np.any(labels_float == 1.0):
            raise ValueError("y must contain both classes 0 and 1")
        labels_float = np.asarray(labels_float, dtype=np.float64)

        if sample_weight is None:
            effective_weights = np.ones(n_samples, dtype=np.float64)
        else:
            try:
                raw_sample_weight = np.asarray(sample_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "sample_weight must be a one-dimensional numeric array"
                ) from exc
            if raw_sample_weight.ndim != 1 or raw_sample_weight.size != n_samples:
                raise ValueError("sample_weight must match X rows")
            weight_dtype = raw_sample_weight.dtype
            if (
                not np.issubdtype(weight_dtype, np.number)
                and not np.issubdtype(weight_dtype, np.bool_)
            ) or np.issubdtype(weight_dtype, np.complexfloating):
                raise TypeError("sample_weight must contain real numeric values")
            try:
                effective_weights = np.asarray(raw_sample_weight, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("sample_weight must contain real numeric values") from exc
            if not np.isfinite(effective_weights).all() or np.any(
                effective_weights < 0
            ):
                raise ValueError("sample_weight must be finite and non-negative")

        # Apply class weighting after validating the target and sample weights.
        if class_weight == "balanced":
            class_zero_count = int(np.count_nonzero(labels_float == 0.0))
            class_one_count = int(np.count_nonzero(labels_float == 1.0))
            class_zero_weight = float(n_samples) / (2.0 * class_zero_count)
            class_one_weight = float(n_samples) / (2.0 * class_one_count)
            effective_weights = effective_weights * np.where(
                labels_float == 0.0, class_zero_weight, class_one_weight
            )
        elif isinstance(class_weight, dict):
            class_zero_weight = float(class_weight[0])
            class_one_weight = float(class_weight[1])
            effective_weights = effective_weights * np.where(
                labels_float == 0.0, class_zero_weight, class_one_weight
            )
        if not np.isfinite(effective_weights).all() or np.any(effective_weights < 0):
            raise ValueError("effective training weights must be finite and non-negative")
        total_weight = float(np.sum(effective_weights, dtype=np.float64))
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("effective training weights must have a positive total")

        # Binning and all subsequent tree work start only after every validation
        # path above has succeeded.
        mapper = fit_bin_mapper(X, config)
        binned = transform_bins(X, mapper)
        positive_rate = float(
            np.sum(effective_weights * labels_float, dtype=np.float64)
        ) / total_weight
        positive_rate = float(np.clip(positive_rate, EPSILON, 1.0 - EPSILON))
        init_score = float(np.log(positive_rate / (1.0 - positive_rate)))
        raw_scores = np.full(n_samples, init_score, dtype=np.float64)
        trees: list[DecisionTree] = []
        feature_importances = np.zeros(n_features, dtype=np.int64)
        all_rows = np.arange(n_samples, dtype=np.int64)
        previous_rows: np.ndarray | None = None
        n_estimators = normalized_integers["n_estimators"]
        feature_fraction = normalized_continuous["colsample_bytree"]
        subsample_fraction = normalized_continuous["subsample"]
        subsample_frequency = normalized_integers["subsample_freq"]
        learning_rate = normalized_continuous["learning_rate"]
        for iteration in range(n_estimators):
            if feature_fraction < 1.0:
                feature_count = max(1, int(np.floor(feature_fraction * n_features)))
                feature_indices = np.sort(
                    np.asarray(
                        rng.choice(n_features, size=feature_count, replace=False),
                        dtype=np.int64,
                    )
                )
            else:
                feature_indices = np.arange(n_features, dtype=np.int64)

            if subsample_fraction < 1.0 and subsample_frequency > 0:
                if iteration % subsample_frequency == 0 or previous_rows is None:
                    row_count = max(1, int(np.floor(subsample_fraction * n_samples)))
                    previous_rows = np.sort(
                        np.asarray(
                            rng.choice(n_samples, size=row_count, replace=False),
                            dtype=np.int64,
                        )
                    )
                row_indices = previous_rows
            else:
                row_indices = all_rows

            gradients, hessians = binary_gradients_hessians(
                raw_scores, labels_float, effective_weights
            )
            tree = fit_tree(
                binned,
                gradients,
                hessians,
                row_indices,
                feature_indices,
                config,
            )
            tree_output = predict_tree_raw(tree, binned)
            if not np.isfinite(tree_output).all():
                raise ValueError("tree predictions must remain finite")
            raw_scores = raw_scores + learning_rate * tree_output
            if not np.isfinite(raw_scores).all():
                raise ValueError("training scores must remain finite")
            trees.append(tree)
            for node in tree.nodes:
                if node.feature >= 0:
                    feature_importances[int(node.feature)] += np.int64(1)

        # Publish fitted state atomically after the complete boosting sequence
        # succeeds, leaving any prior fitted state untouched on validation or
        # numerical failure.
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = n_features
        self.mapper_ = mapper
        self.trees_ = trees
        self.init_score_ = init_score
        self.learning_rate_ = learning_rate
        self.feature_importances_ = feature_importances
        return self

    def predict_raw(self, X: Matrix) -> np.ndarray:
        """Return additive raw logits before the logistic transform."""
        # ``fit`` publishes all learned attributes together after training
        # succeeds.  Check that complete state before inspecting ``X`` so a
        # prediction attempted on a fresh estimator fails predictably.
        learned = (
            "classes_",
            "n_features_in_",
            "mapper_",
            "trees_",
            "init_score_",
            "learning_rate_",
        )
        if any(not hasattr(self, name) for name in learned):
            raise RuntimeError("LiteLightGBM instance is not fitted")

        try:
            fitted_features = int(self.n_features_in_)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("LiteLightGBM fitted state is invalid") from exc
        if fitted_features < 0:
            raise RuntimeError("LiteLightGBM fitted state is invalid")

        # Validate dense and sparse inputs without ever materializing a sparse
        # matrix as dense.  Empty prediction batches are valid; unlike ``fit``,
        # prediction only requires a two-dimensional matrix with the fitted
        # feature count.
        sparse_input = sp.issparse(X)
        if sparse_input:
            try:
                if len(X.shape) != 2:  # pragma: no cover - SciPy matrices are 2-D
                    raise ValueError("X must be a two-dimensional matrix")
                n_samples, n_features = (int(X.shape[0]), int(X.shape[1]))
                feature_dtype = np.dtype(X.dtype)
                stored_values = np.asarray(X.data)
            except (AttributeError, TypeError, ValueError, IndexError) as exc:
                raise ValueError("X must be a two-dimensional numeric matrix") from exc
            if n_samples < 0 or n_features < 0:  # pragma: no cover - invalid SciPy shape
                raise ValueError("X must have non-negative dimensions")
            if (
                not np.issubdtype(feature_dtype, np.number)
                and not np.issubdtype(feature_dtype, np.bool_)
            ) or np.issubdtype(feature_dtype, np.complexfloating):
                raise TypeError("X must contain real numeric values")
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    finite_values = np.asarray(stored_values, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("X must contain real numeric values") from exc
            if not np.isfinite(finite_values).all():
                raise ValueError("X must contain only finite values")
        else:
            try:
                dense_values = np.asarray(X)
            except (TypeError, ValueError) as exc:
                raise ValueError("X must be a two-dimensional numeric matrix") from exc
            if dense_values.ndim != 2:
                raise ValueError("X must be a two-dimensional matrix")
            n_samples, n_features = (
                int(dense_values.shape[0]),
                int(dense_values.shape[1]),
            )
            feature_dtype = dense_values.dtype
            if (
                not np.issubdtype(feature_dtype, np.number)
                and not np.issubdtype(feature_dtype, np.bool_)
            ) or np.issubdtype(feature_dtype, np.complexfloating):
                raise TypeError("X must contain real numeric values")
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    finite_values = np.asarray(dense_values, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("X must contain real numeric values") from exc
            if not np.isfinite(finite_values).all():
                raise ValueError("X must contain only finite values")

        if n_features != fitted_features:
            raise ValueError("X has a different number of features than fitted estimator")

        # ``transform_bins`` retains sparse inputs as CSC/CSR views, and its
        # mapper checks ensure the learned representation is compatible with
        # this batch before any tree traversal begins.
        data = transform_bins(X, self.mapper_)
        try:
            init_score = float(self.init_score_)
            learning_rate = float(self.learning_rate_)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("LiteLightGBM fitted state is invalid") from exc
        if not np.isfinite(init_score) or not np.isfinite(learning_rate):
            raise RuntimeError("LiteLightGBM fitted state is invalid")

        tree_sum = np.zeros(n_samples, dtype=np.float64)
        try:
            trees = iter(self.trees_)
        except TypeError as exc:
            raise RuntimeError("LiteLightGBM fitted state is invalid") from exc
        for tree in trees:
            try:
                tree_output = np.asarray(
                    predict_tree_raw(tree, data), dtype=np.float64
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("tree predictions must be finite numeric values") from exc
            if tree_output.ndim != 1 or tree_output.size != n_samples:
                raise ValueError("tree predictions must have one value per row")
            if not np.isfinite(tree_output).all():
                raise ValueError("tree predictions must remain finite")
            with np.errstate(over="ignore", invalid="ignore"):
                tree_sum += tree_output
            if not np.isfinite(tree_sum).all():
                raise ValueError("tree predictions must remain finite")

        with np.errstate(over="ignore", invalid="ignore"):
            raw_scores = init_score + learning_rate * tree_sum
        if not np.isfinite(raw_scores).all():
            raise ValueError("predictions must remain finite")
        return raw_scores

    def decision_function(self, X: Matrix) -> np.ndarray:
        """Return raw logits through a conventional classifier interface."""
        return self.predict_raw(X)

    def predict_proba(self, X: Matrix) -> np.ndarray:
        """Return probabilities with columns ``[P(y=0), P(y=1)]``."""
        probabilities_one = sigmoid(self.predict_raw(X))
        return np.column_stack((1.0 - probabilities_one, probabilities_one))

    def predict(self, X: Matrix) -> np.ndarray:
        """Return class 1 only when its probability is strictly above 0.5."""
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities > 0.5).astype(np.int64)

    def _config(self) -> LiteLightGBMConfig:
        """Snapshot current parameters as an immutable configuration."""
        return LiteLightGBMConfig(**self.get_params())
