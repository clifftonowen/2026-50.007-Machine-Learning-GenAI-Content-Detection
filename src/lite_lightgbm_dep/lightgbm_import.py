"""Convert a supported official LightGBM model dump for local prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .binning import BinMapper
from .tree import DecisionTree, TreeNode


@dataclass(frozen=True, slots=True)
class ImportedLightGBM:
    """Validated fitted state produced from ``Booster.dump_model()``."""

    mapper: BinMapper
    trees: list[DecisionTree]
    feature_importances: np.ndarray
    active_features: np.ndarray


def _integer(value: Any, name: str, *, minimum: int) -> int:
    """Return one non-boolean integer field from an official model dump."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"LightGBM dump {name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"LightGBM dump {name} must be at least {minimum}")
    return result


def _finite_float(value: Any, name: str) -> float:
    """Return one finite real scalar from an official model dump."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"LightGBM dump {name} must be finite and numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"LightGBM dump {name} must be finite and numeric"
        ) from exc
    if not np.isfinite(result):
        raise ValueError(f"LightGBM dump {name} must be finite and numeric")
    return result


def _validate_objective(model_dump: Mapping[str, Any]) -> None:
    """Reject objectives whose prediction transform LiteLightGBM cannot match."""
    objective = model_dump.get("objective")
    if not isinstance(objective, str) or not objective.split():
        raise ValueError("LightGBM dump must describe a binary objective")
    tokens = objective.split()
    if tokens[0] != "binary":
        raise ValueError("only the LightGBM binary objective is supported")
    for token in tokens[1:]:
        if token.startswith("sigmoid:"):
            sigmoid_scale = _finite_float(
                token.removeprefix("sigmoid:"), "objective sigmoid"
            )
            if sigmoid_scale != 1.0:
                raise ValueError("only LightGBM sigmoid=1 is supported")


def import_lightgbm_dump(model_dump: Mapping[str, Any]) -> ImportedLightGBM:
    """Convert the JSON-compatible result of ``Booster.dump_model()``.

    Official dumped leaf values already contain shrinkage, and the first tree
    contains the initial binary bias. The returned trees are therefore used
    with a zero initial score and a fitted learning rate of one.
    """
    if not isinstance(model_dump, Mapping):
        raise TypeError("model_dump must be a mapping from Booster.dump_model()")
    _validate_objective(model_dump)
    if _integer(model_dump.get("num_class"), "num_class", minimum=1) != 1:
        raise ValueError("only binary one-class-score LightGBM dumps are supported")
    if (
        _integer(
            model_dump.get("num_tree_per_iteration"),
            "num_tree_per_iteration",
            minimum=1,
        )
        != 1
    ):
        raise ValueError("only one LightGBM tree per boosting iteration is supported")
    if bool(model_dump.get("average_output", False)):
        raise ValueError("LightGBM models that average tree outputs are unsupported")
    if bool(model_dump.get("linear_tree", False)):
        raise ValueError("LightGBM linear trees are unsupported")

    max_feature = _integer(
        model_dump.get("max_feature_idx"), "max_feature_idx", minimum=0
    )
    n_features = max_feature + 1
    tree_info = model_dump.get("tree_info")
    if not isinstance(tree_info, list) or not tree_info:
        raise ValueError("LightGBM dump tree_info must be a non-empty list")

    thresholds: list[set[float]] = [set() for _ in range(n_features)]

    def collect(node: Any, tree_index: int) -> None:
        if not isinstance(node, Mapping):
            raise ValueError(f"LightGBM tree {tree_index} contains an invalid node")
        if "leaf_value" in node:
            if "leaf_const" in node or "leaf_coeff" in node:
                raise ValueError("LightGBM linear leaves are unsupported")
            _finite_float(node["leaf_value"], f"tree {tree_index} leaf_value")
            return

        if node.get("decision_type") != "<=":
            raise ValueError("only numerical LightGBM <= splits are supported")
        missing_type = node.get("missing_type", "None")
        if missing_type == "Zero":
            raise ValueError("LightGBM zero_as_missing models are unsupported")
        if missing_type not in ("None", "NaN"):
            raise ValueError(
                f"LightGBM missing type {missing_type!r} is unsupported"
            )
        feature = _integer(
            node.get("split_feature"),
            f"tree {tree_index} split_feature",
            minimum=0,
        )
        if feature >= n_features:
            raise ValueError("LightGBM split feature is outside the model feature range")
        threshold = _finite_float(
            node.get("threshold"), f"tree {tree_index} threshold"
        )
        thresholds[feature].add(threshold)
        if "left_child" not in node or "right_child" not in node:
            raise ValueError(f"LightGBM tree {tree_index} split is missing a child")
        collect(node["left_child"], tree_index)
        collect(node["right_child"], tree_index)

    structures: list[Mapping[str, Any]] = []
    for tree_index, info in enumerate(tree_info):
        if not isinstance(info, Mapping) or not isinstance(
            info.get("tree_structure"), Mapping
        ):
            raise ValueError(f"LightGBM tree_info[{tree_index}] is invalid")
        structure = info["tree_structure"]
        collect(structure, tree_index)
        structures.append(structure)

    cut_points = tuple(
        np.asarray(sorted(feature_thresholds), dtype=np.float64)
        for feature_thresholds in thresholds
    )
    n_bins = np.asarray(
        [cuts.size + 1 for cuts in cut_points], dtype=np.int64
    )
    default_bins = np.asarray(
        [
            np.searchsorted(cuts, 0.0, side="left")
            for cuts in cut_points
        ],
        dtype=np.int64,
    )
    bin_offsets = np.zeros(n_features + 1, dtype=np.int64)
    bin_offsets[1:] = np.cumsum(n_bins, dtype=np.int64)
    mapper = BinMapper(
        cut_points=cut_points,
        default_bins=default_bins,
        n_bins=n_bins,
        bin_offsets=bin_offsets,
    )

    feature_importances = np.zeros(n_features, dtype=np.int64)
    trees: list[DecisionTree] = []
    for tree_index, structure in enumerate(structures):
        nodes: list[TreeNode] = []
        used_features: set[int] = set()

        def convert(node: Mapping[str, Any], depth: int) -> int:
            node_index = len(nodes)
            nodes.append(TreeNode(depth=depth))
            if "leaf_value" in node:
                nodes[node_index].value = _finite_float(
                    node["leaf_value"], f"tree {tree_index} leaf_value"
                )
                if "leaf_count" in node:
                    nodes[node_index].sample_count = _integer(
                        node["leaf_count"],
                        f"tree {tree_index} leaf_count",
                        minimum=0,
                    )
                return node_index

            feature = _integer(
                node["split_feature"],
                f"tree {tree_index} split_feature",
                minimum=0,
            )
            threshold_value = _finite_float(
                node["threshold"], f"tree {tree_index} threshold"
            )
            threshold_bin = int(
                np.searchsorted(
                    cut_points[feature], threshold_value, side="left"
                )
            )
            current = nodes[node_index]
            current.feature = feature
            current.threshold_bin = threshold_bin
            current.default_left = bool(default_bins[feature] <= threshold_bin)
            current.split_gain = _finite_float(
                node.get("split_gain", 0.0), f"tree {tree_index} split_gain"
            )
            if "internal_count" in node:
                current.sample_count = _integer(
                    node["internal_count"],
                    f"tree {tree_index} internal_count",
                    minimum=0,
                )
            used_features.add(feature)
            feature_importances[feature] += np.int64(1)
            current.left_child = convert(node["left_child"], depth + 1)
            current.right_child = convert(node["right_child"], depth + 1)
            return node_index

        convert(structure, 0)
        trees.append(
            DecisionTree(
                nodes=nodes,
                feature_indices=np.asarray(
                    sorted(used_features), dtype=np.int64
                ),
            )
        )

    active_features = np.flatnonzero(feature_importances).astype(
        np.int64, copy=False
    )
    return ImportedLightGBM(
        mapper=mapper,
        trees=trees,
        feature_importances=feature_importances,
        active_features=active_features,
    )
