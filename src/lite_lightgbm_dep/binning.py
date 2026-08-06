"""Binning and sparse encoded dataset definitions for LiteLightGBM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .core import Matrix, LiteLightGBMConfig, _normalize_integer_scalar

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

    # Match the estimator's scalar validation without silently truncating
    # fractional values.  Direct mapper calls retain their historical lower
    # bound of one for both fields.
    normalized_config: dict[str, int] = {}
    for name, value in (
        ("max_bin", config.max_bin),
        ("min_data_in_bin", config.min_data_in_bin),
    ):
        normalized_value = _normalize_integer_scalar(value, name)
        if normalized_value < 1:
            raise ValueError(f"{name} must be positive")
        normalized_config[name] = normalized_value

    max_bin = normalized_config["max_bin"]
    min_data_in_bin = normalized_config["min_data_in_bin"]

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


def _encoded_bin_dtype(n_bins: np.ndarray) -> np.dtype:
    """Return the smallest unsigned dtype for validated bin counts."""
    largest_encoded_bin = int(np.max(n_bins))
    if largest_encoded_bin <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if largest_encoded_bin <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if largest_encoded_bin <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


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
        raw_n_bins = np.asarray(mapper.n_bins)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("mapper must contain one default bin per feature") from exc
    if (
        n_features != mapper_n_features
        or raw_defaults.ndim != 1
        or raw_defaults.size != n_features
        or raw_n_bins.ndim != 1
        or raw_n_bins.size != n_features
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

    # Validate n_bins before selecting a compact encoded dtype.  The mapper's
    # metadata remains signed int64; each feature's count must agree with the
    # number of cut points used by this transform.
    n_bins_dtype = raw_n_bins.dtype
    if (
        np.issubdtype(n_bins_dtype, np.bool_)
        or not np.issubdtype(n_bins_dtype, np.number)
        or np.issubdtype(n_bins_dtype, np.complexfloating)
    ):
        raise ValueError("mapper n_bins must be finite integer-valued numeric values")
    if not np.isfinite(raw_n_bins).all():
        raise ValueError("mapper n_bins must be finite integer-valued numeric values")
    if np.issubdtype(n_bins_dtype, np.floating) and np.any(
        raw_n_bins != np.trunc(raw_n_bins)
    ):
        raise ValueError("mapper n_bins must be finite integer-valued numeric values")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            n_bins = np.asarray(raw_n_bins, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mapper n_bins must be representable as signed int64 values") from exc
    if np.issubdtype(n_bins_dtype, np.unsignedinteger) and np.any(n_bins < 0):
        raise ValueError("mapper n_bins must be representable as signed int64 values")
    if np.issubdtype(n_bins_dtype, np.floating):
        with np.errstate(over="ignore", invalid="ignore"):
            round_trip = n_bins.astype(n_bins_dtype)
        if np.any(round_trip != raw_n_bins):
            raise ValueError("mapper n_bins must be representable as signed int64 values")
    if np.any(n_bins <= 0):
        raise ValueError("mapper n_bins must be positive")
    expected_n_bins = np.asarray(
        [cuts.size + 1 for cuts in cuts_by_feature], dtype=np.int64
    )
    if np.any(n_bins != expected_n_bins):
        raise ValueError("mapper n_bins must match cut-point bin counts")

    encoded_dtype = _encoded_bin_dtype(n_bins)
    encoded_max = int(np.iinfo(encoded_dtype).max)

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
                # Keep the addition in signed int64, then narrow only after
                # checking that the encoded value fits the selected dtype.
                encoded_values = bins[keep] + np.int64(1)
                if np.any(encoded_values > encoded_max):
                    raise ValueError("encoded bins exceed selected dtype range")
                output_data_parts.append(
                    np.asarray(encoded_values, dtype=encoded_dtype)
                )
            output_indptr[feature + 1] = np.int64(
                output_indptr[feature]
                + int(np.count_nonzero(keep))
            )

        if output_data_parts:
            output_indices = np.concatenate(output_indices_parts)
            output_data = np.concatenate(output_data_parts)
        else:
            output_indices = np.empty(0, dtype=np.int64)
            output_data = np.empty(0, dtype=encoded_dtype)

        csc = sp.csc_matrix(
            (output_data, output_indices, output_indptr),
            shape=(n_samples, n_features),
            dtype=encoded_dtype,
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
        bins_matrix = np.empty((n_samples, n_features), dtype=encoded_dtype)
        for feature in range(n_features):
            bins_matrix[:, feature] = np.asarray(
                np.searchsorted(
                    cuts_by_feature[feature],
                    matrix[:, feature],
                    side="left",
                ),
                dtype=encoded_dtype,
            )

        rows, columns = np.nonzero(bins_matrix != defaults[np.newaxis, :])
        if rows.size:
            # Compute the one-based value in a signed type before narrowing;
            # this avoids unsigned overflow when a bin equals 255, 65,535,
            # or 4,294,967,295 at a dtype boundary.
            bin_ids = np.asarray(bins_matrix[rows, columns], dtype=np.int64)
            encoded_values = bin_ids + np.int64(1)
            if np.any(encoded_values > encoded_max):
                raise ValueError("encoded bins exceed selected dtype range")
            values = np.asarray(encoded_values, dtype=encoded_dtype)
        else:
            values = np.empty(0, dtype=encoded_dtype)
        csc = sp.csc_matrix(
            (values, (rows, columns)),
            shape=(n_samples, n_features),
            dtype=encoded_dtype,
        )
        csc.sum_duplicates()
        csc.eliminate_zeros()
        csc.sort_indices()

    csr = csc.tocsr(copy=True)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    if csc.data.dtype != encoded_dtype or csr.data.dtype != encoded_dtype:
        raise ValueError("sparse encoded-bin dtype was not preserved")
    return BinnedDataset(csr=csr, csc=csc, mapper=mapper)
