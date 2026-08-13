"""Public LiteLightGBM estimator façade.

See ``src/lite_lightgbm_docs.md`` for the API reference and
``src/lite_lightgbm.md`` for the implementation contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.sparse as sp

from .lite_lightgbm_dep.core import (
    Matrix,
    ClassWeight,
    EPSILON,
    LiteLightGBMConfig,
    _normalize_integer_scalar,
    sigmoid,
    soft_threshold,
    binary_gradients_hessians,
)
from .lite_lightgbm_dep.binning import (
    BinMapper,
    BinnedDataset,
    _find_bin_boundaries,
    _encoded_bin_dtype,
    fit_bin_mapper,
    transform_bins,
)
from .lite_lightgbm_dep.tree import (
    Histogram,
    HistogramLayout,
    SplitInfo,
    TreeNode,
    DecisionTree,
    build_histogram,
    find_best_split,
    partition_rows,
    fit_tree,
    predict_tree_raw,
    _fit_tree,
    _predict_tree_raw_validated,
    _validate_tree_storage,
)
from .lite_lightgbm_dep.lightgbm_import import import_lightgbm_dump


def _active_feature_indices(
    data: BinnedDataset,
    min_child_samples: int,
) -> np.ndarray:
    """Return count-splittable original features from one canonical CSC view."""
    try:
        n_samples, n_features = (int(data.shape[0]), int(data.shape[1]))
        csc = data.csc
        mapper = data.mapper
        n_bins = np.asarray(mapper.n_bins, dtype=np.int64)
        defaults = np.asarray(mapper.default_bins, dtype=np.int64)
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("binned data must contain valid feature metadata") from exc
    if (
        n_bins.ndim != 1
        or defaults.ndim != 1
        or n_bins.size != n_features
        or defaults.size != n_features
        or tuple(csc.shape) != (n_samples, n_features)
    ):
        raise ValueError("binned data mapper has incompatible feature metadata")

    # ``transform_bins`` stores only non-default entries.  Decode those bins,
    # count them exactly, then add all implicit rows to the feature's default
    # bin.  This is a one-time count-only pass and never allocates a dense
    # sample-by-feature matrix or computes gradient/Hessian statistics.
    active: list[int] = []
    for feature in range(n_features):
        feature_bin_count = int(n_bins[feature])
        if feature_bin_count <= 1:
            continue
        start = int(csc.indptr[feature])
        end = int(csc.indptr[feature + 1])
        encoded = np.asarray(csc.data[start:end], dtype=np.int64)
        if encoded.size and (np.any(encoded <= 0) or np.any(encoded > feature_bin_count)):
            raise ValueError("binned sparse values are outside mapper bin range")
        explicit_bins = encoded - np.int64(1)
        counts = np.bincount(explicit_bins, minlength=feature_bin_count)
        if counts.size != feature_bin_count:
            # ``encoded`` was range-checked above; retain this guard for
            # malformed sparse implementations that return an unexpected dtype.
            raise ValueError("binned sparse values are outside mapper bin range")
        default_bin = int(defaults[feature])
        if default_bin < 0 or default_bin >= feature_bin_count:
            raise ValueError("mapper default bins are outside feature bin ranges")
        counts[default_bin] += np.int64(n_samples - encoded.size)
        prefix = np.cumsum(counts[:-1], dtype=np.int64)
        if np.any(
            (prefix >= int(min_child_samples))
            & ((n_samples - prefix) >= int(min_child_samples))
        ):
            active.append(feature)

    return np.asarray(active, dtype=np.int64)


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

    @classmethod
    def from_lightgbm_dump(
        cls, model_dump: Mapping[str, Any]
    ) -> LiteLightGBM:
        """Create a predictor from an official ``Booster.dump_model()`` result."""
        imported = import_lightgbm_dump(model_dump)
        model = cls(n_estimators=len(imported.trees), learning_rate=1.0)
        model.classes_ = np.array([0, 1])
        model.n_features_in_ = len(imported.mapper.n_bins)
        model.mapper_ = imported.mapper
        model.trees_ = imported.trees
        model.init_score_ = 0.0
        model.learning_rate_ = 1.0
        model.feature_importances_ = imported.feature_importances
        model.active_features_ = imported.active_features
        return model

    @classmethod
    def from_lightgbm_json(
        cls, path: str | PathLike[str]
    ) -> LiteLightGBM:
        """Load a JSON-serialized official ``Booster.dump_model()`` result."""
        with open(path, "r", encoding="utf-8") as model_file:
            model_dump = json.load(model_file)
        return cls.from_lightgbm_dump(model_dump)

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
            normalized_value = _normalize_integer_scalar(value, name)
            if normalized_value < lower_bound:
                raise ValueError(
                    f"{name} must be an integer at least {lower_bound}"
                )
            normalized_integers[name] = normalized_value

        # max_depth uses non-positive values for unlimited growth; a positive
        # value is still required to be an integer depth.
        normalized_max_depth = _normalize_integer_scalar(
            config.max_depth, "max_depth"
        )
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
        tree_storage = _validate_tree_storage(binned, n_samples, n_features)
        active_features = _active_feature_indices(
            binned,
            normalized_integers["min_child_samples"],
        )
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
                sampled_features = np.sort(
                    np.asarray(
                        rng.choice(n_features, size=feature_count, replace=False),
                        dtype=np.int64,
                    )
                )
                # Draw from the original feature range first to preserve the
                # seeded RNG stream, then intersect with the one-time safe
                # active-feature set.  Both inputs are sorted and unique.
                feature_indices = np.intersect1d(
                    sampled_features,
                    active_features,
                    assume_unique=True,
                )
            else:
                feature_indices = np.asarray(active_features, dtype=np.int64).copy()

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
            tree = _fit_tree(
                binned,
                gradients,
                hessians,
                row_indices,
                feature_indices,
                config,
                use_histogram_subtraction=True,
                _validated_storage=tree_storage,
            )
            tree_output = _predict_tree_raw_validated(
                tree,
                n_samples=n_samples,
                n_features=n_features,
                csc=tree_storage[1],
                n_bins=tree_storage[3],
                defaults=tree_storage[4],
            )
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
        self.active_features_ = np.asarray(active_features, dtype=np.int64)
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
        tree_storage = _validate_tree_storage(data, n_samples, n_features)
        try:
            init_score = float(self.init_score_)
            learning_rate = float(self.learning_rate_)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("LiteLightGBM fitted state is invalid") from exc
        if not np.isfinite(init_score) or not np.isfinite(learning_rate):
            raise RuntimeError("LiteLightGBM fitted state is invalid")

        raw_scores = np.full(n_samples, init_score, dtype=np.float64)
        try:
            trees = iter(self.trees_)
        except TypeError as exc:
            raise RuntimeError("LiteLightGBM fitted state is invalid") from exc
        for tree in trees:
            try:
                tree_output = np.asarray(
                    _predict_tree_raw_validated(
                        tree,
                        n_samples=n_samples,
                        n_features=n_features,
                        csc=tree_storage[1],
                        n_bins=tree_storage[3],
                        defaults=tree_storage[4],
                    ),
                    dtype=np.float64,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("tree predictions must be finite numeric values") from exc
            if tree_output.ndim != 1 or tree_output.size != n_samples:
                raise ValueError("tree predictions must have one value per row")
            if not np.isfinite(tree_output).all():
                raise ValueError("tree predictions must remain finite")
            with np.errstate(over="ignore", invalid="ignore"):
                raw_scores = raw_scores + learning_rate * tree_output
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

    def score(
        self,
        X: Matrix,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Return mean classification accuracy, optionally sample-weighted."""
        predictions = self.predict(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.size != predictions.size:
            raise ValueError("y must be one-dimensional with one label per row")
        if labels.size == 0:
            raise ValueError("y must be non-empty")
        correct = predictions == labels
        if sample_weight is None:
            return float(np.mean(correct, dtype=np.float64))

        raw_weights = np.asarray(sample_weight)
        if raw_weights.ndim != 1 or raw_weights.size != predictions.size:
            raise ValueError(
                "sample_weight must be one-dimensional with one value per row"
            )
        weight_dtype = raw_weights.dtype
        if (
            not np.issubdtype(weight_dtype, np.number)
            and not np.issubdtype(weight_dtype, np.bool_)
        ) or np.issubdtype(weight_dtype, np.complexfloating):
            raise TypeError("sample_weight must contain real numeric values")
        try:
            weights = np.asarray(raw_weights, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("sample_weight must contain real numeric values") from exc
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("sample_weight must be finite and non-negative")
        total_weight = float(np.sum(weights, dtype=np.float64))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("sample_weight must have a positive total")
        return float(
            np.sum(weights * correct, dtype=np.float64) / total_weight
        )

    def _config(self) -> LiteLightGBMConfig:
        """Snapshot current parameters as an immutable configuration."""
        return LiteLightGBMConfig(**self.get_params())
