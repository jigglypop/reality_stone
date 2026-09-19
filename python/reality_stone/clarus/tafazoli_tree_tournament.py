"""Nested, label-blind tree-family tournament for the Tafazoli PFC snapshot.

The tournament compares restricted predictive model families on exactly the
same session-local, non-overlapping forecast anchors.  Model-family and
hyperparameter selection happen inside each outer training fold.  The outer
test fold is evaluated once.

This is an observational model tournament.  A winning family is not evidence
that the brain literally executes that algorithm, that a biological optimizer
has been identified, or that a programming language has been recovered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from math import factorial, log2, pi
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .tafazoli_call_graph_probe import (
    CodelengthResult,
    TrajectoryDesign,
    apply_switching_predictor,
    apply_var_predictor,
    assign_past_only_states,
    build_trajectory_design,
    centroid_parameter_count,
    fit_past_only_gate,
    fit_switching_predictor,
    fit_var_predictor,
    score_heldout_gaussian_codelength,
    whiten_prepared_latent_fold,
)
from .tafazoli_session_operator_probe import (
    OFFICIAL_CLASSIFIER_MD5,
    PreparedLatentFold,
    SessionSpec,
    WholeTrialFold,
    load_tafazoli_train_dimensions,
    make_whole_trial_folds,
    prepare_session_latent_fold,
    recovered_session_specs,
    verify_official_classifier_checksum,
)


SCHEMA_VERSION = "clarus-tafazoli-tree-tournament/v1"
PROBE_SCOPE = "nested_label_blind_session_local_tree_family_tournament"

YES = "YES"
NO = "NO"
PENDING = "PENDING"
TEST_UNAVAILABLE = "TEST_UNAVAILABLE"

FAMILY_MATCHED_VAR = "matched_var"
FAMILY_FLAT_SWITCHING = "flat_kmeans_switching"
FAMILY_AXIS_TREE = "axis_aligned_model_tree"
FAMILY_OBLIQUE_TREE = "fixed_two_sparse_oblique_model_tree"
FAMILY_PARENT_TREE = "parent_rank1_contrast_tree"

IMPLEMENTED_FAMILIES = (
    FAMILY_MATCHED_VAR,
    FAMILY_FLAT_SWITCHING,
    FAMILY_AXIS_TREE,
    FAMILY_OBLIQUE_TREE,
    FAMILY_PARENT_TREE,
)
TREE_FAMILIES = (
    FAMILY_AXIS_TREE,
    FAMILY_OBLIQUE_TREE,
    FAMILY_PARENT_TREE,
)

CATALOG_IMPLEMENTED = "IMPLEMENTED"
CATALOG_PENDING = PENDING
CATALOG_PARTIAL = "PARTIAL"
CATALOG_UNAVAILABLE = TEST_UNAVAILABLE


@dataclass(frozen=True)
class TreeTournamentConfig:
    """Predeclared nested-CV and model-search protocol."""

    seed: int = 20260730
    families: tuple[str, ...] = IMPLEMENTED_FAMILIES
    leaf_counts: tuple[int, ...] = (2, 3)
    history_depths: tuple[int, ...] = (1, 2, 3)
    global_anchor_depth: int = 3
    rank_cap: int = 3
    lag_bins: int = 10
    primary_stride_bins: int = 10
    outer_fold_count: int = 6
    inner_fold_count: int = 3
    ridge_alpha: float = 1.0
    kmeans_restarts: int = 4
    kmeans_max_iterations: int = 100
    split_quantiles: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8)
    minimum_leaf_samples: int = 12
    minimum_leaf_trials: int = 4
    run_event_mean_removed_sensitivity: bool = True
    run_reverse_descriptive_control: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("seed", self.seed),
            ("global_anchor_depth", self.global_anchor_depth),
            ("rank_cap", self.rank_cap),
            ("lag_bins", self.lag_bins),
            ("primary_stride_bins", self.primary_stride_bins),
            ("outer_fold_count", self.outer_fold_count),
            ("inner_fold_count", self.inner_fold_count),
            ("kmeans_restarts", self.kmeans_restarts),
            ("kmeans_max_iterations", self.kmeans_max_iterations),
            ("minimum_leaf_samples", self.minimum_leaf_samples),
            ("minimum_leaf_trials", self.minimum_leaf_trials),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.outer_fold_count < 2 or self.inner_fold_count < 2:
            raise ValueError("outer and inner fold counts must be at least two")
        if self.primary_stride_bins < self.lag_bins:
            raise ValueError("primary stride must not reuse overlapping count windows")
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("families must be a non-empty unique tuple")
        if any(family not in IMPLEMENTED_FAMILIES for family in self.families):
            raise ValueError("families contains an unknown or unimplemented family")
        if (
            FAMILY_MATCHED_VAR not in self.families
            or FAMILY_FLAT_SWITCHING not in self.families
        ):
            raise ValueError("matched VAR and flat switching baselines are required")
        if not self.leaf_counts or len(set(self.leaf_counts)) != len(self.leaf_counts):
            raise ValueError("leaf_counts must be a non-empty unique tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in self.leaf_counts
        ):
            raise ValueError("leaf counts must be integers of at least two")
        if not self.history_depths or len(set(self.history_depths)) != len(
            self.history_depths
        ):
            raise ValueError("history_depths must be a non-empty unique tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.history_depths
        ):
            raise ValueError("history depths must be positive integers")
        required_depth = max((*self.leaf_counts, *self.history_depths))
        if self.global_anchor_depth < required_depth:
            raise ValueError(
                "global_anchor_depth must cover every VAR order and gate history"
            )
        if isinstance(self.ridge_alpha, bool) or not np.isfinite(
            float(self.ridge_alpha)
        ):
            raise ValueError("ridge_alpha must be finite")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must not be negative")
        if not self.split_quantiles or len(set(self.split_quantiles)) != len(
            self.split_quantiles
        ):
            raise ValueError("split_quantiles must be a non-empty unique tuple")
        if any(
            not np.isfinite(float(value)) or not 0.0 < float(value) < 1.0
            for value in self.split_quantiles
        ):
            raise ValueError("split quantiles must lie strictly between zero and one")


@dataclass(frozen=True)
class CandidateSpec:
    """One predeclared family, operator-budget, and gate-history choice."""

    family: str
    leaf_count: int
    history_depth: int

    @property
    def key(self) -> str:
        return (
            f"{self.family}:leaves={self.leaf_count}:history={self.history_depth}"
        )


@dataclass(frozen=True)
class SplitRule:
    """One frozen hard split; application has no successor argument."""

    kind: str
    feature_indices: tuple[int, ...]
    weights: tuple[float, ...]
    threshold: float
    search_cost_bits: float


@dataclass(frozen=True)
class TreeNode:
    """Flat, serializable binary-tree node."""

    node_id: int
    depth: int
    parent_id: int | None
    left_child_id: int | None
    right_child_id: int | None
    leaf_index: int | None
    split: SplitRule | None

    @property
    def is_leaf(self) -> bool:
        return self.split is None


@dataclass(frozen=True)
class ContrastMap:
    """One zero-mean sibling contrast stored at an internal node."""

    node_id: int
    matrix: np.ndarray
    left_multiplier: float
    right_multiplier: float


@dataclass(frozen=True)
class HardTreePredictor:
    """Hard-routed common-intercept plus leaf-map predictor."""

    family: str
    latent_rank: int
    history_depth: int
    nodes: tuple[TreeNode, ...]
    coefficients: np.ndarray
    dynamic_parameter_count: int
    gate_parameter_count: int
    discrete_search_bits: float

    @property
    def leaf_count(self) -> int:
        return sum(node.is_leaf for node in self.nodes)


@dataclass(frozen=True)
class ParentContrastPredictor:
    """Root affine operator plus rank-one, zero-mean sibling contrasts."""

    family: str
    latent_rank: int
    history_depth: int
    nodes: tuple[TreeNode, ...]
    root_coefficients: np.ndarray
    contrasts: tuple[ContrastMap, ...]
    dynamic_parameter_count: int
    gate_parameter_count: int
    discrete_search_bits: float

    @property
    def leaf_count(self) -> int:
        return sum(node.is_leaf for node in self.nodes)


@dataclass(frozen=True)
class TreeStructureAudit:
    """Small topology/search audit without storing fitted numeric arrays."""

    family: str
    requested_leaf_count: int
    fitted_leaf_count: int
    internal_node_count: int
    maximum_depth: int
    split_kinds: tuple[str, ...]
    split_features: tuple[tuple[int, ...], ...]
    gate_parameter_count: int
    dynamic_parameter_count: int
    discrete_search_bits: float
    all_leaves_meet_support_floor: bool


@dataclass(frozen=True)
class InnerSelectionAudit:
    """Nested selection result from outer-training rows only."""

    family: str
    candidate_count: int
    available_candidate_count: int
    selected_spec: CandidateSpec
    selected_inner_bits_per_scalar: float
    runner_up_margin_bits_per_scalar: float | None
    inner_fold_count: int
    outer_test_used_for_selection: bool


@dataclass(frozen=True)
class FamilyOuterFoldResult:
    """One family evaluated once on one untouched outer fold."""

    fold_index_zero_based: int
    family: str
    selection: InnerSelectionAudit
    outer_score: CodelengthResult
    matched_var_score: CodelengthResult
    matched_flat_switching_score: CodelengthResult
    reverse_outer_score: CodelengthResult | None
    structure: TreeStructureAudit | None
    codelength_advantage_over_var_bits_per_scalar: float
    codelength_advantage_over_flat_switching_bits_per_scalar: float
    forward_advantage_over_reverse_bits_per_scalar: float | None
    global_anchor_depth: int
    same_outer_targets_for_all_competitors: bool
    baselines_independently_nested_selected: bool
    gate_uses_current_and_past_only: bool
    outer_test_target_used_for_selection_or_gate: bool
    d1_d3_rows_treated_as_paired_trials: bool


@dataclass(frozen=True)
class SessionFamilyResult:
    """Outer-fold aggregate for one physical session, dimension, and family."""

    analysis_key: str
    session_index_one_based: int
    animal: str
    neuron_count: int
    dimension: int
    event_mean_removed: bool
    family: str
    fold_results: tuple[FamilyOuterFoldResult, ...]
    selected_spec_keys: tuple[str, ...]
    test_scalar_count: int
    model_bits_per_test_scalar: float
    codelength_advantage_over_var_bits_per_scalar: float
    codelength_advantage_over_flat_switching_bits_per_scalar: float
    forward_advantage_over_reverse_bits_per_scalar: float | None
    complete_outer_folds: bool


@dataclass(frozen=True)
class TreeAggregateResult:
    """Session x dimension aggregation; anchors and neurons are not replicates."""

    family: str
    event_mean_removed: bool
    animal: str
    unit_count: int
    all_units_complete: bool
    median_model_bits_per_test_scalar: float
    median_advantage_over_var_bits_per_scalar: float
    median_advantage_over_flat_switching_bits_per_scalar: float
    median_forward_advantage_over_reverse_bits_per_scalar: float | None
    session_unit_win_fraction_over_var: float
    session_unit_win_fraction_over_flat_switching: float


@dataclass(frozen=True)
class CandidateCatalogEntry:
    """Implementation status for a candidate requested by the search program."""

    family: str
    status: str
    reason: str


@dataclass(frozen=True)
class ClaimVerdict:
    """Claim-local answer with a stable key."""

    key: str
    answer: str
    reason: str


@dataclass(frozen=True)
class TreeTournamentClaimLocks:
    """Claims that this processed observational tournament cannot unlock."""

    labels_or_responses_used: bool = False
    all_factors_used: bool = False
    dimension_two_used: bool = False
    saved_classifier_test_set_used: bool = False
    full_pseudopopulation_fit: bool = False
    outer_test_used_for_model_selection: bool = False
    test_future_used_for_gate_or_split: bool = False
    d1_d3_rows_treated_as_paired_trials: bool = False
    biological_tree_algorithm_identified: bool = False
    task_inheritance_tree_identified: bool = False
    optimizer_mechanism_identified: bool = False
    region_size_speed_throughput_tradeoff_identified: bool = False
    brain_programming_language_identified: bool = False


@dataclass(frozen=True)
class TafazoliTreeTournamentReport:
    """Serializable result of the nested model-family tournament."""

    schema_version: str
    scope: str
    method_status: str
    source_file_md5: str | None
    official_checksum_verified: bool
    config: TreeTournamentConfig
    session_specs: tuple[SessionSpec, ...]
    fields_used_for_fitting: tuple[str, ...]
    blind_fields_used: tuple[str, ...]
    saved_test_role: str
    train_only_preprocessing: bool
    primary_inference_unit: str
    codelength_name: str
    catalog: tuple[CandidateCatalogEntry, ...]
    results: tuple[SessionFamilyResult, ...]
    aggregates: tuple[TreeAggregateResult, ...]
    screening_survivors: tuple[str, ...]
    model_relative_winner: str | None
    verdicts: tuple[ClaimVerdict, ...]
    claim_locks: TreeTournamentClaimLocks
    limitations: tuple[str, ...]
    conclusion: str

    def verdict(self, key: str) -> ClaimVerdict:
        matches = tuple(item for item in self.verdicts if item.key == key)
        if len(matches) != 1:
            raise KeyError(key)
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _MutableNode:
    node_id: int
    depth: int
    parent_id: int | None
    left_child_id: int | None = None
    right_child_id: int | None = None
    split: SplitRule | None = None


@dataclass(frozen=True)
class _CandidateEvaluation:
    score: CodelengthResult
    structure: TreeStructureAudit | None


class _CandidateUnavailable(RuntimeError):
    """Raised when a training fold cannot support a declared candidate."""


def _derived_seed(base_seed: int, *tokens: Any) -> int:
    payload = json.dumps(
        (SCHEMA_VERSION, int(base_seed), *tokens),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:16],
        byteorder="little",
        signed=False,
    )


def enumerate_candidate_specs(
    config: TreeTournamentConfig,
    family: str,
) -> tuple[CandidateSpec, ...]:
    """Return the fixed inner-selection grid for one family."""

    if family not in config.families:
        raise ValueError("family is not enabled")
    if family == FAMILY_MATCHED_VAR:
        return tuple(
            CandidateSpec(family, leaf_count, leaf_count)
            for leaf_count in config.leaf_counts
        )
    return tuple(
        CandidateSpec(family, leaf_count, history_depth)
        for leaf_count in config.leaf_counts
        for history_depth in config.history_depths
    )


def _family_selection_bits(config: TreeTournamentConfig, family: str) -> float:
    family_bits = log2(float(len(config.families)))
    grid_bits = log2(float(len(enumerate_candidate_specs(config, family))))
    return family_bits + grid_bits


def _ridge_coefficients(
    design: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    penalize_intercept: bool,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge design and target must be aligned matrices")
    gram = x.T @ x
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    return np.linalg.pinv(gram + penalty) @ x.T @ y


def _sse(observed: np.ndarray, predicted: np.ndarray) -> float:
    difference = np.asarray(observed, dtype=np.float64) - np.asarray(
        predicted,
        dtype=np.float64,
    )
    return float(np.sum(np.square(difference), dtype=np.float64))


def _build_design(
    latent: np.ndarray,
    *,
    history_depth: int,
    config: TreeTournamentConfig,
    reverse: bool,
) -> TrajectoryDesign:
    return build_trajectory_design(
        latent,
        history_depth=history_depth,
        anchor_history_depth=config.global_anchor_depth,
        lag_bins=config.lag_bins,
        stride_bins=config.primary_stride_bins,
        reverse=reverse,
    )


def _flatten_design(
    design: TrajectoryDesign,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trials, anchors, rank = design.current.shape
    return (
        design.history.reshape(trials * anchors, -1),
        design.current.reshape(trials * anchors, rank),
        design.successor.reshape(trials * anchors, rank),
        np.repeat(np.arange(trials, dtype=np.int64), anchors),
    )


def _split_projection(history: np.ndarray, split: SplitRule) -> np.ndarray:
    indices = np.asarray(split.feature_indices, dtype=np.int64)
    weights = np.asarray(split.weights, dtype=np.float64)
    return np.asarray(history, dtype=np.float64)[:, indices] @ weights


def _freeze_nodes(nodes: Mapping[int, _MutableNode]) -> tuple[TreeNode, ...]:
    leaf_ids = sorted(node_id for node_id, node in nodes.items() if node.split is None)
    leaf_lookup = {node_id: index for index, node_id in enumerate(leaf_ids)}
    return tuple(
        TreeNode(
            node_id=node.node_id,
            depth=node.depth,
            parent_id=node.parent_id,
            left_child_id=node.left_child_id,
            right_child_id=node.right_child_id,
            leaf_index=leaf_lookup[node.node_id] if node.split is None else None,
            split=node.split,
        )
        for node in sorted(nodes.values(), key=lambda item: item.node_id)
    )


def _route_leaf_indices(
    nodes: Sequence[TreeNode],
    history: np.ndarray,
) -> np.ndarray:
    items = {node.node_id: node for node in nodes}
    if 0 not in items:
        raise ValueError("tree has no root")
    values = np.asarray(history, dtype=np.float64)
    labels = np.empty(values.shape[0], dtype=np.int64)
    for sample_index in range(values.shape[0]):
        node = items[0]
        while not node.is_leaf:
            if (
                node.split is None
                or node.left_child_id is None
                or node.right_child_id is None
            ):
                raise ValueError("internal tree node is incomplete")
            projection = float(_split_projection(values[sample_index : sample_index + 1], node.split)[0])
            next_id = (
                node.left_child_id
                if projection <= node.split.threshold
                else node.right_child_id
            )
            node = items[next_id]
        if node.leaf_index is None:
            raise ValueError("leaf has no canonical index")
        labels[sample_index] = node.leaf_index
    return labels


def _node_membership(
    nodes: Sequence[TreeNode],
    history: np.ndarray,
    target_node_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return samples reaching a node and their left/right branch at that node."""

    items = {node.node_id: node for node in nodes}
    reached = np.zeros(history.shape[0], dtype=bool)
    goes_left = np.zeros(history.shape[0], dtype=bool)
    for sample_index in range(history.shape[0]):
        node = items[0]
        while True:
            if node.node_id == target_node_id:
                reached[sample_index] = True
                if node.split is not None:
                    projection = float(
                        _split_projection(
                            history[sample_index : sample_index + 1],
                            node.split,
                        )[0]
                    )
                    goes_left[sample_index] = projection <= node.split.threshold
                break
            if (
                node.split is None
                or node.left_child_id is None
                or node.right_child_id is None
            ):
                break
            projection = float(
                _split_projection(
                    history[sample_index : sample_index + 1],
                    node.split,
                )[0]
            )
            next_id = (
                node.left_child_id
                if projection <= node.split.threshold
                else node.right_child_id
            )
            node = items[next_id]
    return reached, goes_left


def _thresholds(values: np.ndarray, config: TreeTournamentConfig) -> tuple[float, ...]:
    projected = np.asarray(values, dtype=np.float64).reshape(-1)
    if projected.size < 2 or float(np.min(projected)) == float(np.max(projected)):
        return ()
    candidates = np.quantile(
        projected,
        np.asarray(config.split_quantiles, dtype=np.float64),
    )
    lower = float(np.min(projected))
    upper = float(np.max(projected))
    return tuple(
        sorted(
            {
                float(value)
                for value in candidates
                if np.isfinite(value) and lower < float(value) < upper
            }
        )
    )


def _oblique_directions(feature_count: int) -> tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]:
    if feature_count < 2:
        return ()
    scale = float(1.0 / np.sqrt(2.0))
    directions = []
    for left in range(feature_count):
        for right in range(left + 1, feature_count):
            directions.append(((left, right), (scale, scale)))
            directions.append(((left, right), (scale, -scale)))
    return tuple(directions)


def _candidate_split_rules(
    history: np.ndarray,
    member_mask: np.ndarray,
    *,
    kind: str,
    current_leaf_count: int,
    config: TreeTournamentConfig,
) -> tuple[SplitRule, ...]:
    values = np.asarray(history, dtype=np.float64)
    subset = values[np.asarray(member_mask, dtype=bool)]
    if kind == "axis":
        directions = tuple(
            ((feature,), (1.0,)) for feature in range(values.shape[1])
        )
    elif kind == "fixed_two_sparse_oblique":
        directions = _oblique_directions(values.shape[1])
    else:
        raise ValueError("unknown split kind")
    if not directions:
        return ()
    leaf_bits = log2(float(max(current_leaf_count, 1)))
    direction_bits = log2(float(len(directions)))
    threshold_bits = log2(float(len(config.split_quantiles)))
    rules = []
    for indices, weights in directions:
        projection = subset[:, np.asarray(indices)] @ np.asarray(weights)
        for threshold in _thresholds(projection, config):
            rules.append(
                SplitRule(
                    kind=kind,
                    feature_indices=tuple(int(value) for value in indices),
                    weights=tuple(float(value) for value in weights),
                    threshold=float(threshold),
                    search_cost_bits=leaf_bits + direction_bits + threshold_bits,
                )
            )
    return tuple(rules)


def _leaf_support_is_valid(
    member_mask: np.ndarray,
    trial_ids: np.ndarray,
    *,
    config: TreeTournamentConfig,
) -> bool:
    mask = np.asarray(member_mask, dtype=bool)
    return (
        int(np.count_nonzero(mask)) >= config.minimum_leaf_samples
        and np.unique(np.asarray(trial_ids, dtype=np.int64)[mask]).size
        >= config.minimum_leaf_trials
    )


def _rank_one(matrix: np.ndarray) -> np.ndarray:
    left, singular, right = np.linalg.svd(
        np.asarray(matrix, dtype=np.float64),
        full_matrices=False,
    )
    if singular.size == 0:
        return np.zeros_like(matrix, dtype=np.float64)
    return (left[:, :1] * singular[:1]) @ right[:1, :]


def _fit_parent_contrast_given_tree(
    nodes: Sequence[TreeNode],
    history: np.ndarray,
    current: np.ndarray,
    successor: np.ndarray,
    *,
    history_depth: int,
    ridge_alpha: float,
    discrete_search_bits: float,
) -> ParentContrastPredictor:
    rank = current.shape[1]
    root_design = np.column_stack((np.ones(current.shape[0]), current))
    root = _ridge_coefficients(
        root_design,
        successor,
        alpha=ridge_alpha,
        penalize_intercept=False,
    )
    prediction = root_design @ root
    contrasts = []
    internal_nodes = sorted(
        (node for node in nodes if not node.is_leaf),
        key=lambda node: (node.depth, node.node_id),
    )
    for node in internal_nodes:
        reached, goes_left = _node_membership(nodes, history, node.node_id)
        left_mask = reached & goes_left
        right_mask = reached & ~goes_left
        if not np.any(left_mask) or not np.any(right_mask):
            raise _CandidateUnavailable("contrast split has an empty child")
        residual = successor - prediction
        left_map = _ridge_coefficients(
            current[left_mask],
            residual[left_mask],
            alpha=ridge_alpha,
            penalize_intercept=True,
        )
        right_map = _ridge_coefficients(
            current[right_mask],
            residual[right_mask],
            alpha=ridge_alpha,
            penalize_intercept=True,
        )
        contrast = _rank_one(left_map - right_map)
        left_count = int(np.count_nonzero(left_mask))
        right_count = int(np.count_nonzero(right_mask))
        total = left_count + right_count
        left_multiplier = right_count / total
        right_multiplier = -left_count / total
        prediction[left_mask] += (
            current[left_mask] @ contrast
        ) * left_multiplier
        prediction[right_mask] += (
            current[right_mask] @ contrast
        ) * right_multiplier
        contrasts.append(
            ContrastMap(
                node_id=node.node_id,
                matrix=np.asarray(contrast, dtype=np.float64),
                left_multiplier=float(left_multiplier),
                right_multiplier=float(right_multiplier),
            )
        )
    dynamic_count = rank * (rank + 1) + len(internal_nodes) * (2 * rank - 1)
    return ParentContrastPredictor(
        family=FAMILY_PARENT_TREE,
        latent_rank=rank,
        history_depth=history_depth,
        nodes=tuple(nodes),
        root_coefficients=np.asarray(root, dtype=np.float64),
        contrasts=tuple(contrasts),
        dynamic_parameter_count=dynamic_count,
        gate_parameter_count=len(internal_nodes),
        discrete_search_bits=float(discrete_search_bits),
    )


def _apply_parent_contrast(
    predictor: ParentContrastPredictor,
    history: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    root_design = np.column_stack((np.ones(current.shape[0]), current))
    prediction = root_design @ predictor.root_coefficients
    for contrast in predictor.contrasts:
        reached, goes_left = _node_membership(
            predictor.nodes,
            history,
            contrast.node_id,
        )
        left_mask = reached & goes_left
        right_mask = reached & ~goes_left
        prediction[left_mask] += (
            current[left_mask] @ contrast.matrix
        ) * contrast.left_multiplier
        prediction[right_mask] += (
            current[right_mask] @ contrast.matrix
        ) * contrast.right_multiplier
    return prediction


def _fit_hard_experts_given_tree(
    family: str,
    nodes: Sequence[TreeNode],
    history: np.ndarray,
    current: np.ndarray,
    successor: np.ndarray,
    *,
    history_depth: int,
    ridge_alpha: float,
    discrete_search_bits: float,
) -> HardTreePredictor:
    labels = _route_leaf_indices(nodes, history)
    rank = current.shape[1]
    leaf_count = sum(node.is_leaf for node in nodes)
    occupied = tuple(int(value) for value in np.unique(labels))
    if occupied != tuple(range(leaf_count)):
        raise _CandidateUnavailable(
            "a declared tree leaf has no outer-training support"
        )
    fitted = fit_switching_predictor(
        current.reshape(1, current.shape[0], rank),
        successor.reshape(1, successor.shape[0], rank),
        labels.reshape(1, labels.size),
        state_count=leaf_count,
        ridge_alpha=ridge_alpha,
    )
    return HardTreePredictor(
        family=family,
        latent_rank=rank,
        history_depth=history_depth,
        nodes=tuple(nodes),
        coefficients=np.asarray(fitted.coefficients, dtype=np.float64),
        dynamic_parameter_count=rank + leaf_count * rank * rank,
        gate_parameter_count=leaf_count - 1,
        discrete_search_bits=float(discrete_search_bits),
    )


def _apply_hard_tree(
    predictor: HardTreePredictor,
    history: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    labels = _route_leaf_indices(predictor.nodes, history)
    rank = current.shape[1]
    from .tafazoli_call_graph_probe import LinearPredictor

    wrapped = LinearPredictor(
        family="past_gated_switching_current_map",
        coefficients=predictor.coefficients,
        parameter_count=predictor.dynamic_parameter_count,
        state_count=predictor.leaf_count,
    )
    return apply_switching_predictor(
        wrapped,
        current.reshape(1, current.shape[0], rank),
        labels.reshape(1, labels.size),
    ).reshape(current.shape)


def _training_structure_cost_bits(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    residual_variance: float,
    dynamic_parameter_count: int,
    gate_parameter_count: int,
    discrete_search_bits: float,
) -> float:
    scalar_count = int(observed.size)
    heldin = 0.5 * scalar_count * log2(2.0 * pi * residual_variance)
    heldin += _sse(observed, predicted) / (
        2.0 * residual_variance * np.log(2.0)
    )
    parameter_count = dynamic_parameter_count + gate_parameter_count + 1
    parameter_bits = 0.5 * parameter_count * log2(
        float(max(scalar_count, 2))
    )
    return float(heldin + parameter_bits + discrete_search_bits)


def _tree_predictions(
    predictor: HardTreePredictor | ParentContrastPredictor,
    history: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    if isinstance(predictor, HardTreePredictor):
        return _apply_hard_tree(predictor, history, current)
    return _apply_parent_contrast(predictor, history, current)


def _fit_predictor_for_nodes(
    family: str,
    nodes: Sequence[TreeNode],
    history: np.ndarray,
    current: np.ndarray,
    successor: np.ndarray,
    *,
    history_depth: int,
    config: TreeTournamentConfig,
    discrete_search_bits: float,
) -> HardTreePredictor | ParentContrastPredictor:
    if family in (FAMILY_AXIS_TREE, FAMILY_OBLIQUE_TREE):
        return _fit_hard_experts_given_tree(
            family,
            nodes,
            history,
            current,
            successor,
            history_depth=history_depth,
            ridge_alpha=config.ridge_alpha,
            discrete_search_bits=discrete_search_bits,
        )
    if family == FAMILY_PARENT_TREE:
        return _fit_parent_contrast_given_tree(
            nodes,
            history,
            current,
            successor,
            history_depth=history_depth,
            ridge_alpha=config.ridge_alpha,
            discrete_search_bits=discrete_search_bits,
        )
    raise ValueError("family is not a tree family")


def _fit_tree(
    family: str,
    train_design: TrajectoryDesign,
    *,
    leaf_count: int,
    history_depth: int,
    config: TreeTournamentConfig,
) -> HardTreePredictor | ParentContrastPredictor:
    history, current, successor, trial_ids = _flatten_design(train_design)
    if family == FAMILY_OBLIQUE_TREE and history.shape[1] < 2:
        raise _CandidateUnavailable("two-sparse split requires two history features")
    split_kind = (
        "fixed_two_sparse_oblique"
        if family == FAMILY_OBLIQUE_TREE
        else "axis"
    )
    nodes: dict[int, _MutableNode] = {0: _MutableNode(0, 0, None)}
    frozen_root = _freeze_nodes(nodes)
    root_predictor = _fit_predictor_for_nodes(
        family,
        frozen_root,
        history,
        current,
        successor,
        history_depth=history_depth,
        config=config,
        discrete_search_bits=0.0,
    )
    root_prediction = _tree_predictions(
        root_predictor,
        history,
        current,
    )
    root_degrees = max(
        successor.size - root_predictor.dynamic_parameter_count,
        1,
    )
    residual_variance = max(
        _sse(successor, root_prediction) / root_degrees,
        np.finfo(np.float64).eps,
    )
    search_bits = 0.0

    while sum(node.split is None for node in nodes.values()) < leaf_count:
        frozen = _freeze_nodes(nodes)
        current_labels = _route_leaf_indices(frozen, history)
        leaf_nodes = tuple(node for node in frozen if node.is_leaf)
        best: tuple[
            float,
            int,
            SplitRule,
            dict[int, _MutableNode],
            HardTreePredictor | ParentContrastPredictor,
        ] | None = None
        for leaf_node in leaf_nodes:
            if leaf_node.leaf_index is None:
                raise RuntimeError("leaf index is missing")
            member_mask = current_labels == leaf_node.leaf_index
            rules = _candidate_split_rules(
                history,
                member_mask,
                kind=split_kind,
                current_leaf_count=len(leaf_nodes),
                config=config,
            )
            for rule in rules:
                projection = _split_projection(history, rule)
                left_mask = member_mask & (projection <= rule.threshold)
                right_mask = member_mask & ~left_mask
                if not _leaf_support_is_valid(
                    left_mask,
                    trial_ids,
                    config=config,
                ) or not _leaf_support_is_valid(
                    right_mask,
                    trial_ids,
                    config=config,
                ):
                    continue
                candidate_nodes = {
                    node_id: _MutableNode(
                        node.node_id,
                        node.depth,
                        node.parent_id,
                        node.left_child_id,
                        node.right_child_id,
                        node.split,
                    )
                    for node_id, node in nodes.items()
                }
                next_id = max(candidate_nodes) + 1
                left_id = next_id
                right_id = next_id + 1
                parent = candidate_nodes[leaf_node.node_id]
                parent.split = rule
                parent.left_child_id = left_id
                parent.right_child_id = right_id
                candidate_nodes[left_id] = _MutableNode(
                    left_id,
                    parent.depth + 1,
                    parent.node_id,
                )
                candidate_nodes[right_id] = _MutableNode(
                    right_id,
                    parent.depth + 1,
                    parent.node_id,
                )
                frozen_candidate = _freeze_nodes(candidate_nodes)
                candidate_search_bits = search_bits + rule.search_cost_bits
                try:
                    predictor = _fit_predictor_for_nodes(
                        family,
                        frozen_candidate,
                        history,
                        current,
                        successor,
                        history_depth=history_depth,
                        config=config,
                        discrete_search_bits=candidate_search_bits,
                    )
                except _CandidateUnavailable:
                    continue
                prediction = _tree_predictions(
                    predictor,
                    history,
                    current,
                )
                criterion = _training_structure_cost_bits(
                    successor,
                    prediction,
                    residual_variance=residual_variance,
                    dynamic_parameter_count=predictor.dynamic_parameter_count,
                    gate_parameter_count=predictor.gate_parameter_count,
                    discrete_search_bits=candidate_search_bits,
                )
                tie_key = (
                    criterion,
                    leaf_node.node_id,
                    rule.feature_indices,
                    rule.weights,
                    rule.threshold,
                )
                if best is None or tie_key < (
                    best[0],
                    best[1],
                    best[2].feature_indices,
                    best[2].weights,
                    best[2].threshold,
                ):
                    best = (
                        criterion,
                        leaf_node.node_id,
                        rule,
                        candidate_nodes,
                        predictor,
                    )
        if best is None:
            raise _CandidateUnavailable("no split satisfies the leaf support floors")
        _, _, selected_rule, nodes, _ = best
        search_bits += selected_rule.search_cost_bits

    frozen = _freeze_nodes(nodes)
    return _fit_predictor_for_nodes(
        family,
        frozen,
        history,
        current,
        successor,
        history_depth=history_depth,
        config=config,
        discrete_search_bits=search_bits,
    )


def _structure_audit(
    predictor: HardTreePredictor | ParentContrastPredictor,
    *,
    requested_leaf_count: int,
    train_design: TrajectoryDesign,
    config: TreeTournamentConfig,
) -> TreeStructureAudit:
    history, _, _, trial_ids = _flatten_design(train_design)
    labels = _route_leaf_indices(predictor.nodes, history)
    support_ok = all(
        _leaf_support_is_valid(
            labels == leaf_index,
            trial_ids,
            config=config,
        )
        for leaf_index in range(predictor.leaf_count)
    )
    internal = tuple(node for node in predictor.nodes if not node.is_leaf)
    return TreeStructureAudit(
        family=predictor.family,
        requested_leaf_count=requested_leaf_count,
        fitted_leaf_count=predictor.leaf_count,
        internal_node_count=len(internal),
        maximum_depth=max(node.depth for node in predictor.nodes),
        split_kinds=tuple(
            node.split.kind for node in internal if node.split is not None
        ),
        split_features=tuple(
            node.split.feature_indices
            for node in internal
            if node.split is not None
        ),
        gate_parameter_count=predictor.gate_parameter_count,
        dynamic_parameter_count=predictor.dynamic_parameter_count,
        discrete_search_bits=predictor.discrete_search_bits,
        all_leaves_meet_support_floor=support_ok,
    )


def _score_flat_switching(
    train_design: TrajectoryDesign,
    test_design: TrajectoryDesign,
    *,
    spec: CandidateSpec,
    config: TreeTournamentConfig,
    seed_tokens: tuple[Any, ...],
) -> _CandidateEvaluation:
    rank = train_design.current.shape[-1]
    gate = fit_past_only_gate(
        train_design.history,
        state_count=spec.leaf_count,
        history_depth=spec.history_depth,
        latent_rank=rank,
        seed=_derived_seed(config.seed, *seed_tokens, spec.key),
        restarts=config.kmeans_restarts,
        max_iterations=config.kmeans_max_iterations,
    )
    train_states = assign_past_only_states(gate, train_design.history)
    test_states = assign_past_only_states(gate, test_design.history)
    predictor = fit_switching_predictor(
        train_design.current,
        train_design.successor,
        train_states,
        state_count=spec.leaf_count,
        ridge_alpha=config.ridge_alpha,
    )
    train_prediction = apply_switching_predictor(
        predictor,
        train_design.current,
        train_states,
    )
    test_prediction = apply_switching_predictor(
        predictor,
        test_design.current,
        test_states,
    )
    selection_bits = _family_selection_bits(config, FAMILY_FLAT_SWITCHING)
    selection_bits += log2(float(factorial(spec.leaf_count)))
    score = score_heldout_gaussian_codelength(
        family=FAMILY_FLAT_SWITCHING,
        train_observed=train_design.successor,
        train_predicted=train_prediction,
        test_observed=test_design.successor,
        test_predicted=test_prediction,
        dynamic_parameter_count=predictor.parameter_count,
        gate_parameter_count=centroid_parameter_count(
            spec.leaf_count,
            spec.history_depth,
            rank,
        ),
        model_selection_cost_bits=selection_bits,
    )
    return _CandidateEvaluation(score=score, structure=None)


def _score_var(
    train_design: TrajectoryDesign,
    test_design: TrajectoryDesign,
    *,
    spec: CandidateSpec,
    config: TreeTournamentConfig,
) -> _CandidateEvaluation:
    rank = train_design.current.shape[-1]
    predictor = fit_var_predictor(
        train_design.history,
        train_design.successor,
        history_order=spec.leaf_count,
        latent_rank=rank,
        ridge_alpha=config.ridge_alpha,
    )
    train_prediction = apply_var_predictor(predictor, train_design.history)
    test_prediction = apply_var_predictor(predictor, test_design.history)
    score = score_heldout_gaussian_codelength(
        family=f"{FAMILY_MATCHED_VAR}_order_{spec.leaf_count}",
        train_observed=train_design.successor,
        train_predicted=train_prediction,
        test_observed=test_design.successor,
        test_predicted=test_prediction,
        dynamic_parameter_count=predictor.parameter_count,
        gate_parameter_count=0,
        model_selection_cost_bits=_family_selection_bits(
            config,
            FAMILY_MATCHED_VAR,
        ),
    )
    return _CandidateEvaluation(score=score, structure=None)


def _score_tree(
    train_design: TrajectoryDesign,
    test_design: TrajectoryDesign,
    *,
    spec: CandidateSpec,
    config: TreeTournamentConfig,
) -> _CandidateEvaluation:
    predictor = _fit_tree(
        spec.family,
        train_design,
        leaf_count=spec.leaf_count,
        history_depth=spec.history_depth,
        config=config,
    )
    train_history, train_current, train_successor, _ = _flatten_design(
        train_design
    )
    test_history, test_current, test_successor, _ = _flatten_design(test_design)
    train_prediction = _tree_predictions(
        predictor,
        train_history,
        train_current,
    )
    test_prediction = _tree_predictions(
        predictor,
        test_history,
        test_current,
    )
    score = score_heldout_gaussian_codelength(
        family=spec.family,
        train_observed=train_successor,
        train_predicted=train_prediction,
        test_observed=test_successor,
        test_predicted=test_prediction,
        dynamic_parameter_count=predictor.dynamic_parameter_count,
        gate_parameter_count=predictor.gate_parameter_count,
        model_selection_cost_bits=(
            _family_selection_bits(config, spec.family)
            + predictor.discrete_search_bits
        ),
    )
    return _CandidateEvaluation(
        score=score,
        structure=_structure_audit(
            predictor,
            requested_leaf_count=spec.leaf_count,
            train_design=train_design,
            config=config,
        ),
    )


def _evaluate_candidate(
    prepared: PreparedLatentFold,
    *,
    spec: CandidateSpec,
    config: TreeTournamentConfig,
    reverse: bool,
    seed_tokens: tuple[Any, ...],
) -> _CandidateEvaluation:
    whitened = whiten_prepared_latent_fold(prepared)
    history_depth = (
        spec.leaf_count if spec.family == FAMILY_MATCHED_VAR else spec.history_depth
    )
    train_design = _build_design(
        whitened.source_train,
        history_depth=history_depth,
        config=config,
        reverse=reverse,
    )
    test_design = _build_design(
        whitened.target_test,
        history_depth=history_depth,
        config=config,
        reverse=reverse,
    )
    if spec.family == FAMILY_MATCHED_VAR:
        return _score_var(
            train_design,
            test_design,
            spec=spec,
            config=config,
        )
    if spec.family == FAMILY_FLAT_SWITCHING:
        return _score_flat_switching(
            train_design,
            test_design,
            spec=spec,
            config=config,
            seed_tokens=seed_tokens,
        )
    return _score_tree(
        train_design,
        test_design,
        spec=spec,
        config=config,
    )


def _sum_scores(
    scores: Sequence[CodelengthResult],
) -> tuple[float, float, int]:
    items = tuple(scores)
    if not items:
        raise ValueError("at least one score is required")
    return (
        float(sum(item.test_sse for item in items)),
        float(sum(item.total_codelength_bits for item in items)),
        int(sum(item.test_scalar_count for item in items)),
    )


def _select_family_inside_outer_train(
    inner_prepared: Sequence[PreparedLatentFold],
    *,
    family: str,
    config: TreeTournamentConfig,
    seed_tokens: tuple[Any, ...],
) -> InnerSelectionAudit:
    candidates = enumerate_candidate_specs(config, family)
    available: list[tuple[float, CandidateSpec]] = []
    for spec in candidates:
        fold_scores = []
        try:
            for prepared in inner_prepared:
                evaluation = _evaluate_candidate(
                    prepared,
                    spec=spec,
                    config=config,
                    reverse=False,
                    seed_tokens=(
                        *seed_tokens,
                        "inner",
                        prepared.fold.index_zero_based,
                    ),
                )
                fold_scores.append(evaluation.score)
        except _CandidateUnavailable:
            continue
        _, bits, scalars = _sum_scores(fold_scores)
        available.append((bits / max(scalars, 1), spec))
    if not available:
        raise _CandidateUnavailable(
            f"no nested candidate is available for family {family}"
        )
    available.sort(key=lambda item: (item[0], item[1].key))
    best_bits, best_spec = available[0]
    margin = (
        float(available[1][0] - best_bits) if len(available) > 1 else None
    )
    return InnerSelectionAudit(
        family=family,
        candidate_count=len(candidates),
        available_candidate_count=len(available),
        selected_spec=best_spec,
        selected_inner_bits_per_scalar=float(best_bits),
        runner_up_margin_bits_per_scalar=margin,
        inner_fold_count=len(inner_prepared),
        outer_test_used_for_selection=False,
    )


def _code_advantage(
    model: CodelengthResult,
    baseline: CodelengthResult,
) -> float:
    if model.test_scalar_count != baseline.test_scalar_count:
        raise RuntimeError("competing scores used different held-out scalars")
    return float(
        (baseline.total_codelength_bits - model.total_codelength_bits)
        / max(model.test_scalar_count, 1)
    )


def _prepare_inner_folds(
    outer_training_population: np.ndarray,
    *,
    config: TreeTournamentConfig,
    event_mean_removed: bool,
    seed_tokens: tuple[Any, ...],
) -> tuple[PreparedLatentFold, ...]:
    inner_folds = make_whole_trial_folds(
        outer_training_population.shape[0],
        fold_count=config.inner_fold_count,
        seed=_derived_seed(config.seed, *seed_tokens, "inner_folds"),
    )
    return tuple(
        prepare_session_latent_fold(
            outer_training_population,
            outer_training_population,
            fold,
            rank_cap=config.rank_cap,
            event_mean_removed=event_mean_removed,
        )
        for fold in inner_folds
    )


def _assemble_outer_fold_result(
    outer_fold: WholeTrialFold,
    *,
    family: str,
    selection: InnerSelectionAudit,
    primary: _CandidateEvaluation,
    independently_selected_var: _CandidateEvaluation,
    independently_selected_flat: _CandidateEvaluation,
    reverse: _CandidateEvaluation | None,
    config: TreeTournamentConfig,
) -> FamilyOuterFoldResult:
    if reverse is not None:
        reverse_score = reverse.score
        forward_reverse = _code_advantage(primary.score, reverse_score)
    else:
        reverse_score = None
        forward_reverse = None
    equal_targets = (
        primary.score.test_scalar_count
        == independently_selected_var.score.test_scalar_count
        == independently_selected_flat.score.test_scalar_count
    )
    if not equal_targets:
        raise RuntimeError("global anchor contract failed")
    return FamilyOuterFoldResult(
        fold_index_zero_based=outer_fold.index_zero_based,
        family=family,
        selection=selection,
        outer_score=primary.score,
        matched_var_score=independently_selected_var.score,
        matched_flat_switching_score=independently_selected_flat.score,
        reverse_outer_score=reverse_score,
        structure=primary.structure,
        codelength_advantage_over_var_bits_per_scalar=_code_advantage(
            primary.score,
            independently_selected_var.score,
        ),
        codelength_advantage_over_flat_switching_bits_per_scalar=_code_advantage(
            primary.score,
            independently_selected_flat.score,
        ),
        forward_advantage_over_reverse_bits_per_scalar=forward_reverse,
        global_anchor_depth=config.global_anchor_depth,
        same_outer_targets_for_all_competitors=True,
        baselines_independently_nested_selected=True,
        gate_uses_current_and_past_only=True,
        outer_test_target_used_for_selection_or_gate=False,
        d1_d3_rows_treated_as_paired_trials=False,
    )


def _session_summary(
    *,
    session: SessionSpec,
    dimension: int,
    event_mean_removed: bool,
    family: str,
    fold_results: Sequence[FamilyOuterFoldResult],
) -> SessionFamilyResult:
    folds = tuple(fold_results)
    model_sse, model_bits, scalars = _sum_scores(
        tuple(item.outer_score for item in folds)
    )
    _ = model_sse
    _, var_bits, var_scalars = _sum_scores(
        tuple(item.matched_var_score for item in folds)
    )
    _, flat_bits, flat_scalars = _sum_scores(
        tuple(item.matched_flat_switching_score for item in folds)
    )
    if scalars != var_scalars or scalars != flat_scalars:
        raise RuntimeError("session competitors used different held-out scalars")
    reverse_scores = tuple(
        item.reverse_outer_score
        for item in folds
        if item.reverse_outer_score is not None
    )
    if len(reverse_scores) == len(folds):
        _, reverse_bits, reverse_scalars = _sum_scores(reverse_scores)
        if reverse_scalars != scalars:
            raise RuntimeError("reverse control used different held-out scalars")
        forward_reverse = (reverse_bits - model_bits) / max(scalars, 1)
    else:
        forward_reverse = None
    return SessionFamilyResult(
        analysis_key=(
            f"session{session.index_one_based}:dim{dimension}:"
            f"eventmean={int(event_mean_removed)}:{family}"
        ),
        session_index_one_based=session.index_one_based,
        animal=session.animal,
        neuron_count=session.neuron_count,
        dimension=dimension,
        event_mean_removed=event_mean_removed,
        family=family,
        fold_results=folds,
        selected_spec_keys=tuple(
            item.selection.selected_spec.key for item in folds
        ),
        test_scalar_count=scalars,
        model_bits_per_test_scalar=model_bits / max(scalars, 1),
        codelength_advantage_over_var_bits_per_scalar=(
            var_bits - model_bits
        )
        / max(scalars, 1),
        codelength_advantage_over_flat_switching_bits_per_scalar=(
            flat_bits - model_bits
        )
        / max(scalars, 1),
        forward_advantage_over_reverse_bits_per_scalar=forward_reverse,
        complete_outer_folds=all(
            not item.selection.outer_test_used_for_selection
            and not item.outer_test_target_used_for_selection_or_gate
            and item.same_outer_targets_for_all_competitors
            for item in folds
        ),
    )


def _run_session_dimension(
    session_population: np.ndarray,
    *,
    session: SessionSpec,
    dimension: int,
    event_mean_removed: bool,
    config: TreeTournamentConfig,
) -> tuple[SessionFamilyResult, ...]:
    outer_folds = make_whole_trial_folds(
        session_population.shape[0],
        fold_count=config.outer_fold_count,
        seed=_derived_seed(
            config.seed,
            "outer_folds",
            session.index_one_based,
            dimension,
            event_mean_removed,
        ),
    )
    family_folds: dict[str, list[FamilyOuterFoldResult]] = {
        family: [] for family in config.families
    }
    for fold in outer_folds:
        outer_prepared = prepare_session_latent_fold(
            session_population,
            session_population,
            fold,
            rank_cap=config.rank_cap,
            event_mean_removed=event_mean_removed,
        )
        outer_train_indices = np.asarray(fold.train_indices, dtype=np.int64)
        inner_prepared = _prepare_inner_folds(
            session_population[outer_train_indices],
            config=config,
            event_mean_removed=event_mean_removed,
            seed_tokens=(
                session.index_one_based,
                dimension,
                event_mean_removed,
                fold.index_zero_based,
            ),
        )
        selections = {
            family: _select_family_inside_outer_train(
                inner_prepared,
                family=family,
                config=config,
                seed_tokens=(
                    session.index_one_based,
                    dimension,
                    event_mean_removed,
                    fold.index_zero_based,
                    family,
                ),
            )
            for family in config.families
        }
        primary_evaluations = {
            family: _evaluate_candidate(
                outer_prepared,
                spec=selections[family].selected_spec,
                config=config,
                reverse=False,
                seed_tokens=(
                    session.index_one_based,
                    dimension,
                    event_mean_removed,
                    fold.index_zero_based,
                    family,
                    "outer",
                ),
            )
            for family in config.families
        }
        if config.run_reverse_descriptive_control:
            reverse_evaluations: dict[str, _CandidateEvaluation | None] = {
                family: _evaluate_candidate(
                    outer_prepared,
                    spec=selections[family].selected_spec,
                    config=config,
                    reverse=True,
                    seed_tokens=(
                        session.index_one_based,
                        dimension,
                        event_mean_removed,
                        fold.index_zero_based,
                        family,
                        "reverse",
                    ),
                )
                for family in config.families
            }
        else:
            reverse_evaluations = {
                family: None for family in config.families
            }
        independently_selected_var = primary_evaluations[FAMILY_MATCHED_VAR]
        independently_selected_flat = primary_evaluations[
            FAMILY_FLAT_SWITCHING
        ]
        for family in config.families:
            family_folds[family].append(
                _assemble_outer_fold_result(
                    fold,
                    family=family,
                    selection=selections[family],
                    primary=primary_evaluations[family],
                    independently_selected_var=independently_selected_var,
                    independently_selected_flat=independently_selected_flat,
                    reverse=reverse_evaluations[family],
                    config=config,
                )
            )
    results = []
    for family in config.families:
        results.append(
            _session_summary(
                session=session,
                dimension=dimension,
                event_mean_removed=event_mean_removed,
                family=family,
                fold_results=tuple(family_folds[family]),
            )
        )
    return tuple(results)


def _median(values: Sequence[float]) -> float:
    items = tuple(float(value) for value in values)
    if not items:
        raise ValueError("median requires at least one value")
    return float(np.median(np.asarray(items, dtype=np.float64)))


def _optional_median(values: Sequence[float | None]) -> float | None:
    items = tuple(float(value) for value in values if value is not None)
    return _median(items) if items else None


def aggregate_tree_results(
    results: Sequence[SessionFamilyResult],
    *,
    config: TreeTournamentConfig,
) -> tuple[TreeAggregateResult, ...]:
    items = tuple(results)
    event_modes = (
        (False, True)
        if config.run_event_mean_removed_sensitivity
        else (False,)
    )
    aggregates = []
    for family in config.families:
        for event_mean_removed in event_modes:
            for animal in ("all", "Chico", "Silas"):
                group = tuple(
                    item
                    for item in items
                    if item.family == family
                    and item.event_mean_removed == event_mean_removed
                    and (animal == "all" or item.animal == animal)
                )
                if not group:
                    continue
                aggregates.append(
                    TreeAggregateResult(
                        family=family,
                        event_mean_removed=event_mean_removed,
                        animal=animal,
                        unit_count=len(group),
                        all_units_complete=all(
                            item.complete_outer_folds for item in group
                        ),
                        median_model_bits_per_test_scalar=_median(
                            tuple(
                                item.model_bits_per_test_scalar
                                for item in group
                            )
                        ),
                        median_advantage_over_var_bits_per_scalar=_median(
                            tuple(
                                item.codelength_advantage_over_var_bits_per_scalar
                                for item in group
                            )
                        ),
                        median_advantage_over_flat_switching_bits_per_scalar=_median(
                            tuple(
                                item.codelength_advantage_over_flat_switching_bits_per_scalar
                                for item in group
                            )
                        ),
                        median_forward_advantage_over_reverse_bits_per_scalar=_optional_median(
                            tuple(
                                item.forward_advantage_over_reverse_bits_per_scalar
                                for item in group
                            )
                        ),
                        session_unit_win_fraction_over_var=float(
                            np.mean(
                                np.asarray(
                                    tuple(
                                        item.codelength_advantage_over_var_bits_per_scalar
                                        > 0.0
                                        for item in group
                                    ),
                                    dtype=np.float64,
                                )
                            )
                        ),
                        session_unit_win_fraction_over_flat_switching=float(
                            np.mean(
                                np.asarray(
                                    tuple(
                                        item.codelength_advantage_over_flat_switching_bits_per_scalar
                                        > 0.0
                                        for item in group
                                    ),
                                    dtype=np.float64,
                                )
                            )
                        ),
                    )
                )
    return tuple(aggregates)


def _catalog() -> tuple[CandidateCatalogEntry, ...]:
    return (
        CandidateCatalogEntry(
            FAMILY_MATCHED_VAR,
            CATALOG_IMPLEMENTED,
            "Stationary VAR order is matched to the requested leaf/operator budget.",
        ),
        CandidateCatalogEntry(
            FAMILY_FLAT_SWITCHING,
            CATALOG_IMPLEMENTED,
            "Existing deterministic past-only K=2/3 centroid gate wrapper.",
        ),
        CandidateCatalogEntry(
            FAMILY_AXIS_TREE,
            CATALOG_IMPLEMENTED,
            "Greedy hard model tree over deterministic axis/quantile candidates.",
        ),
        CandidateCatalogEntry(
            FAMILY_OBLIQUE_TREE,
            CATALOG_IMPLEMENTED,
            "Greedy hard tree over every fixed equal-weight two-sparse direction.",
        ),
        CandidateCatalogEntry(
            FAMILY_PARENT_TREE,
            CATALOG_IMPLEMENTED,
            "Axis gate with a root affine map and rank-one sibling contrasts.",
        ),
        CandidateCatalogEntry(
            "hierarchical_mixture_of_linear_experts",
            CATALOG_PENDING,
            "Soft-gate EM adds convergence and likelihood choices not admitted to v1.",
        ),
        CandidateCatalogEntry(
            "duration_state_tree",
            CATALOG_PENDING,
            "A causal dwell-time definition and matched state-only control remain required.",
        ),
        CandidateCatalogEntry(
            "hidden_markov_model",
            CATALOG_PENDING,
            "A low-power observational HMM proxy is testable in a later tournament round.",
        ),
        CandidateCatalogEntry(
            "hidden_semi_markov_model",
            CATALOG_PARTIAL,
            "A duration proxy is partly testable, but biological dwell-time claims need raw trials.",
        ),
    )


def _aggregate_lookup(
    aggregates: Sequence[TreeAggregateResult],
    *,
    family: str,
    event_mean_removed: bool,
    animal: str,
) -> TreeAggregateResult | None:
    matches = tuple(
        item
        for item in aggregates
        if item.family == family
        and item.event_mean_removed == event_mean_removed
        and item.animal == animal
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError("aggregate key is not unique")
    return matches[0]


def _tree_family_passes(
    aggregates: Sequence[TreeAggregateResult],
    *,
    family: str,
    sensitivity_complete: bool,
    reverse_complete: bool,
) -> bool:
    if not sensitivity_complete or not reverse_complete:
        return False
    groups = tuple(
        _aggregate_lookup(
            aggregates,
            family=family,
            event_mean_removed=event_mean_removed,
            animal=animal,
        )
        for event_mean_removed in (False, True)
        for animal in ("all", "Chico", "Silas")
    )
    if any(group is None for group in groups):
        return False
    return all(
        group.all_units_complete
        and group.median_advantage_over_var_bits_per_scalar > 0.0
        and group.median_advantage_over_flat_switching_bits_per_scalar > 0.0
        and group.median_forward_advantage_over_reverse_bits_per_scalar is not None
        and group.median_forward_advantage_over_reverse_bits_per_scalar > 0.0
        and group.session_unit_win_fraction_over_var > 0.5
        and group.session_unit_win_fraction_over_flat_switching > 0.5
        for group in groups
        if group is not None
    )


def _screening_survivors(
    aggregates: Sequence[TreeAggregateResult],
    *,
    config: TreeTournamentConfig,
) -> tuple[str, ...]:
    return tuple(
        family
        for family in TREE_FAMILIES
        if family in config.families
        and _tree_family_passes(
            aggregates,
            family=family,
            sensitivity_complete=config.run_event_mean_removed_sensitivity,
            reverse_complete=config.run_reverse_descriptive_control,
        )
    )


def _model_relative_winner(
    survivors: Sequence[str],
) -> str | None:
    """Return a leader only when the declared screen leaves one family."""

    items = tuple(survivors)
    return items[0] if len(items) == 1 else None


def _build_verdicts(
    survivors: Sequence[str],
    winner: str | None,
    *,
    config: TreeTournamentConfig,
) -> tuple[ClaimVerdict, ...]:
    controls_complete = (
        config.run_event_mean_removed_sensitivity
        and config.run_reverse_descriptive_control
    )
    survivor_items = tuple(survivors)
    proxy_answer = (
        YES if survivor_items else (NO if controls_complete else PENDING)
    )
    unique_answer = (
        YES if winner is not None else (NO if controls_complete else PENDING)
    )
    return (
        ClaimVerdict(
            "nested_tree_tournament_completed",
            YES,
            "Every outer-test score follows outer-train-only three-fold selection.",
        ),
        ClaimVerdict(
            "tested_tree_family_outperformed_flat_baselines",
            proxy_answer,
            (
                f"{survivor_items} pass independently selected VAR, flat switching, "
                "reverse, event-mean, and both-animal gates."
                if survivor_items
                else (
                    "No implemented tree family passes every declared gate."
                    if controls_complete
                    else "Reverse or event-mean sensitivity was not completed."
                )
            ),
        ),
        ClaimVerdict(
            "unique_model_relative_tree_winner",
            unique_answer,
            (
                f"{winner} is the sole screening survivor."
                if winner is not None
                else (
                    "The complete screen leaves no survivor or a multi-family "
                    "Rashomon set; it does not force a unique winner."
                    if controls_complete
                    else "Reverse or event-mean sensitivity was not completed."
                )
            ),
        ),
        ClaimVerdict(
            "brain_executes_winning_tree_algorithm",
            NO,
            "Predictive compression is not an implementation-level mechanism trace.",
        ),
        ClaimVerdict(
            "task_inheritance_tree_identified",
            NO,
            "Within-dimension state routing is not task inheritance, and D1/D3 rows are unpaired.",
        ),
        ClaimVerdict(
            "task_inheritance_architecture_exists_or_is_absent",
            TEST_UNAVAILABLE,
            "Two processed task dimensions cannot decide universal inheritance architecture.",
        ),
        ClaimVerdict(
            "optimizer_mechanism_identified",
            NO,
            "The tournament compares predictors; it does not observe plasticity or optimization.",
        ),
        ClaimVerdict(
            "optimizer_mechanism_exists_or_is_absent",
            TEST_UNAVAILABLE,
            "No learning trajectory or selective optimizer intervention is present.",
        ),
        ClaimVerdict(
            "region_size_speed_throughput_tradeoff_identified",
            TEST_UNAVAILABLE,
            "One processed LPFC snapshot has no multi-region size, latency, and throughput intervention.",
        ),
        ClaimVerdict(
            "brain_programming_language_identified",
            NO,
            "A model-relative dynamics winner is not an opcode set, grammar, or causal semantics.",
        ),
    )


def validate_tree_tournament_claim_locks(
    locks: TreeTournamentClaimLocks,
) -> None:
    if not isinstance(locks, TreeTournamentClaimLocks):
        raise TypeError("locks must be TreeTournamentClaimLocks")
    unlocked = tuple(
        name for name, value in asdict(locks).items() if value is not False
    )
    if unlocked:
        raise ValueError(f"tree tournament claim locks must remain false: {unlocked}")


def _validate_population(population: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(population, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{label} must be trial x neuron x time")
    if min(values.shape) < 1:
        raise ValueError(f"{label} axes must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains NaN or infinity")
    if np.any(values < 0.0):
        raise ValueError(f"{label} contains negative count-like values")
    return values


def _validate_session_specs(
    specs: Sequence[SessionSpec],
    *,
    neuron_count: int,
) -> tuple[SessionSpec, ...]:
    items = tuple(specs)
    if not items:
        raise ValueError("at least one session spec is required")
    expected_start = 0
    for expected_index, item in enumerate(items, start=1):
        if not isinstance(item, SessionSpec):
            raise TypeError("session_specs must contain SessionSpec")
        if item.index_one_based != expected_index:
            raise ValueError("session indices must be consecutive and one-based")
        if item.column_start_zero_based != expected_start:
            raise ValueError("session column ranges must be contiguous")
        if item.column_stop_exclusive <= item.column_start_zero_based:
            raise ValueError("session column range must not be empty")
        expected_start = item.column_stop_exclusive
    if expected_start != neuron_count:
        raise ValueError("session column ranges do not cover the population")
    return items


def run_tafazoli_tree_tournament_from_arrays(
    dimension_one_train: np.ndarray,
    dimension_three_train: np.ndarray,
    *,
    config: TreeTournamentConfig = TreeTournamentConfig(),
    session_specs: Sequence[SessionSpec] | None = None,
) -> TafazoliTreeTournamentReport:
    """Run the strict NumPy-only tournament on the two allowed train tensors."""

    if not isinstance(config, TreeTournamentConfig):
        raise TypeError("config must be TreeTournamentConfig")
    dim1 = _validate_population(
        dimension_one_train,
        label="dimension_one_train",
    )
    dim3 = _validate_population(
        dimension_three_train,
        label="dimension_three_train",
    )
    if dim1.shape != dim3.shape:
        raise ValueError("dimension 1 and 3 tensors must have one shape")
    if dim1.shape[0] < config.outer_fold_count:
        raise ValueError("not enough trials for outer whole-trial CV")
    minimum_outer_train = dim1.shape[0] - int(
        np.ceil(dim1.shape[0] / config.outer_fold_count)
    )
    if minimum_outer_train < config.inner_fold_count:
        raise ValueError("not enough outer-training rows for nested CV")
    if dim1.shape[2] <= config.global_anchor_depth * config.lag_bins:
        raise ValueError("not enough timepoints for the global anchor contract")
    specs = (
        recovered_session_specs()
        if session_specs is None
        else tuple(session_specs)
    )
    specs = _validate_session_specs(specs, neuron_count=dim1.shape[1])
    dimensions = {1: dim1, 3: dim3}
    event_modes = (
        (False, True)
        if config.run_event_mean_removed_sensitivity
        else (False,)
    )
    results = []
    for event_mean_removed in event_modes:
        for dimension, population in dimensions.items():
            for session in specs:
                session_population = population[
                    :,
                    session.column_start_zero_based : session.column_stop_exclusive,
                    :,
                ]
                results.extend(
                    _run_session_dimension(
                        session_population,
                        session=session,
                        dimension=dimension,
                        event_mean_removed=event_mean_removed,
                        config=config,
                    )
                )
    aggregates = aggregate_tree_results(results, config=config)
    survivors = _screening_survivors(aggregates, config=config)
    winner = _model_relative_winner(survivors)
    locks = TreeTournamentClaimLocks()
    validate_tree_tournament_claim_locks(locks)
    report = TafazoliTreeTournamentReport(
        schema_version=SCHEMA_VERSION,
        scope=PROBE_SCOPE,
        method_status="NESTED_TREE_TOURNAMENT_COMPLETE",
        source_file_md5=None,
        official_checksum_verified=False,
        config=config,
        session_specs=specs,
        fields_used_for_fitting=(
            "ClassifierOpts.Dimpredictors[dimension=1].train",
            "ClassifierOpts.Dimpredictors[dimension=3].train",
        ),
        blind_fields_used=(),
        saved_test_role="not_used",
        train_only_preprocessing=True,
        primary_inference_unit="recording_session_x_dimension",
        codelength_name="heldout Gaussian codelength/BIC proxy",
        catalog=_catalog(),
        results=tuple(results),
        aggregates=aggregates,
        screening_survivors=survivors,
        model_relative_winner=winner,
        verdicts=_build_verdicts(survivors, winner, config=config),
        claim_locks=locks,
        limitations=(
            "The 36 rows are saved pseudotrials from one overwritten classifier fold.",
            "All preprocessing, topology search, and grid selection are nested inside outer training rows.",
            "VAR and flat-switching baselines independently select their own inner-CV specifications.",
            "The 27 sessions, not anchors or neurons, are the observational units.",
            "D1 and D3 use independently seeded folds and are never treated as paired trials.",
            "The primary 100 ms stride prevents 90%-overlapping windows from becoming sample replicates.",
            "Tree thresholds, supports, and topology search pay explicit proxy code costs.",
            "The codelength is a held-out Gaussian/BIC proxy, not strict prequential MDL.",
            "Greedy depth-two search is a restricted model class, not an exhaustive tree theorem.",
            "Reverse time is descriptive, not a causal permutation test.",
            "No perturbation, rescue, learning trajectory, or multi-region throughput experiment exists.",
        ),
        conclusion=(
            "The report can name a model-relative winner among five implemented "
            "restricted predictor families only when one family survives. Multiple "
            "passing families remain a Rashomon set. It cannot identify a biological "
            "tree algorithm, optimizer, task-inheritance mechanism, region scaling "
            "law, or brain programming language."
        ),
    )
    validate_tree_tournament_report(report)
    return report


def run_tafazoli_tree_tournament(
    classifier_file: str | Path,
    *,
    config: TreeTournamentConfig = TreeTournamentConfig(),
) -> TafazoliTreeTournamentReport:
    """Checksum-lock the official MAT snapshot and run the strict core."""

    observed_md5 = verify_official_classifier_checksum(classifier_file)
    dim1, dim3 = load_tafazoli_train_dimensions(classifier_file)
    report = run_tafazoli_tree_tournament_from_arrays(
        dim1,
        dim3,
        config=config,
    )
    verified = replace(
        report,
        source_file_md5=observed_md5,
        official_checksum_verified=(observed_md5 == OFFICIAL_CLASSIFIER_MD5),
    )
    validate_tree_tournament_report(verified)
    return verified


def validate_tree_tournament_report(
    report: TafazoliTreeTournamentReport,
) -> None:
    """Reject leakage, unit-count drift, and scientific overclaiming."""

    if not isinstance(report, TafazoliTreeTournamentReport):
        raise TypeError("report must be TafazoliTreeTournamentReport")
    if report.schema_version != SCHEMA_VERSION or report.scope != PROBE_SCOPE:
        raise ValueError("unexpected tree tournament schema or scope")
    if report.method_status != "NESTED_TREE_TOURNAMENT_COMPLETE":
        raise ValueError("unexpected method status")
    if report.codelength_name != "heldout Gaussian codelength/BIC proxy":
        raise ValueError("codelength must not be called strict or prequential MDL")
    if report.blind_fields_used or report.saved_test_role != "not_used":
        raise ValueError("blind fields or saved classifier test entered the probe")
    if not report.train_only_preprocessing:
        raise ValueError("preprocessing was not train-only")
    validate_tree_tournament_claim_locks(report.claim_locks)
    expected_events = (
        2 if report.config.run_event_mean_removed_sensitivity else 1
    )
    expected_result_count = (
        len(report.session_specs)
        * 2
        * expected_events
        * len(report.config.families)
    )
    if len(report.results) != expected_result_count:
        raise ValueError("session-family result count drifted")
    session_lookup = {
        item.index_one_based: item for item in report.session_specs
    }
    seen_keys = set()
    for item in report.results:
        if item.analysis_key in seen_keys:
            raise ValueError("session-family analysis key is duplicated")
        seen_keys.add(item.analysis_key)
        expected_session = session_lookup.get(item.session_index_one_based)
        if expected_session is None:
            raise ValueError("result references an unknown session")
        if (
            item.animal != expected_session.animal
            or item.neuron_count != expected_session.neuron_count
        ):
            raise ValueError("session-family result crossed a session boundary")
        if item.dimension not in (1, 3):
            raise ValueError("dimension 2 entered the tournament")
        if item.family not in report.config.families:
            raise ValueError("result family was not predeclared")
        if len(item.fold_results) != report.config.outer_fold_count:
            raise ValueError("outer fold silently disappeared")
        if not item.complete_outer_folds:
            raise ValueError("outer selection or target-separation contract failed")
        for fold in item.fold_results:
            if (
                fold.selection.outer_test_used_for_selection
                or fold.outer_test_target_used_for_selection_or_gate
                or fold.d1_d3_rows_treated_as_paired_trials
            ):
                raise ValueError("outer-test leakage or D1/D3 pairing was declared")
            if (
                not fold.same_outer_targets_for_all_competitors
                or not fold.baselines_independently_nested_selected
                or not fold.gate_uses_current_and_past_only
                or fold.global_anchor_depth != report.config.global_anchor_depth
            ):
                raise ValueError("common-anchor or past-only gate contract failed")
            scalar_count = fold.outer_score.test_scalar_count
            if (
                fold.matched_var_score.test_scalar_count != scalar_count
                or fold.matched_flat_switching_score.test_scalar_count
                != scalar_count
            ):
                raise ValueError("outer competitors used different targets")
            if fold.structure is not None and (
                fold.structure.fitted_leaf_count
                != fold.structure.requested_leaf_count
                or not fold.structure.all_leaves_meet_support_floor
            ):
                raise ValueError("tree structure violated its declared budget")
    for family in report.config.families:
        for event_mean_removed in (
            (False, True)
            if report.config.run_event_mean_removed_sensitivity
            else (False,)
        ):
            all_group = _aggregate_lookup(
                report.aggregates,
                family=family,
                event_mean_removed=event_mean_removed,
                animal="all",
            )
            if all_group is None or all_group.unit_count != len(
                report.session_specs
            ) * 2:
                raise ValueError("aggregate session x dimension unit count drifted")
    verdict_map = {item.key: item.answer for item in report.verdicts}
    required_verdicts = {
        "nested_tree_tournament_completed",
        "tested_tree_family_outperformed_flat_baselines",
        "unique_model_relative_tree_winner",
        "brain_executes_winning_tree_algorithm",
        "task_inheritance_tree_identified",
        "task_inheritance_architecture_exists_or_is_absent",
        "optimizer_mechanism_identified",
        "optimizer_mechanism_exists_or_is_absent",
        "region_size_speed_throughput_tradeoff_identified",
        "brain_programming_language_identified",
    }
    if set(verdict_map) != required_verdicts:
        raise ValueError("verdict key set drifted")
    if any(family not in TREE_FAMILIES for family in report.screening_survivors):
        raise ValueError("a non-tree family entered the screening survivor set")
    if len(set(report.screening_survivors)) != len(report.screening_survivors):
        raise ValueError("screening survivor family is duplicated")
    expected_winner = _model_relative_winner(report.screening_survivors)
    if report.model_relative_winner != expected_winner:
        raise ValueError("unique-winner and screening-survivor fields disagree")
    if verdict_map["brain_executes_winning_tree_algorithm"] != NO:
        raise ValueError("model winner was promoted to a biological algorithm")
    if verdict_map["task_inheritance_tree_identified"] != NO:
        raise ValueError("state tree was promoted to task inheritance")
    if verdict_map["optimizer_mechanism_identified"] != NO:
        raise ValueError("predictive tournament was promoted to an optimizer")
    if verdict_map["brain_programming_language_identified"] != NO:
        raise ValueError("predictive tournament was promoted to a language")


__all__ = [
    "CATALOG_IMPLEMENTED",
    "CATALOG_PARTIAL",
    "CATALOG_PENDING",
    "CATALOG_UNAVAILABLE",
    "CandidateCatalogEntry",
    "CandidateSpec",
    "ClaimVerdict",
    "FAMILY_AXIS_TREE",
    "FAMILY_FLAT_SWITCHING",
    "FAMILY_MATCHED_VAR",
    "FAMILY_OBLIQUE_TREE",
    "FAMILY_PARENT_TREE",
    "FamilyOuterFoldResult",
    "HardTreePredictor",
    "IMPLEMENTED_FAMILIES",
    "InnerSelectionAudit",
    "NO",
    "PENDING",
    "PROBE_SCOPE",
    "ParentContrastPredictor",
    "SCHEMA_VERSION",
    "SessionFamilyResult",
    "SplitRule",
    "TEST_UNAVAILABLE",
    "TREE_FAMILIES",
    "TafazoliTreeTournamentReport",
    "TreeAggregateResult",
    "TreeNode",
    "TreeStructureAudit",
    "TreeTournamentClaimLocks",
    "TreeTournamentConfig",
    "YES",
    "aggregate_tree_results",
    "enumerate_candidate_specs",
    "run_tafazoli_tree_tournament",
    "run_tafazoli_tree_tournament_from_arrays",
    "validate_tree_tournament_claim_locks",
    "validate_tree_tournament_report",
]
