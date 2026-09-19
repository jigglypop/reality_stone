"""Finite tree-algorithm census for neural reverse engineering.

The census collapses named algorithms into behavioral equivalence families.
It prevents an open-ended literature search from silently expanding the
hypothesis universe after results are known.  ``TESTABLE`` means that the
processed Tafazoli snapshot can screen a predictive fingerprint; it never
means that a biological tree implementation has been identified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "clarus-neural-tree-algorithm-census/v1"
CENSUS_SCOPE = (
    "finite_behavioral_equivalence_census_for_neural_reverse_engineering"
)
TESTABLE = "TESTABLE"
PARTIAL = "PARTIAL"
UNAVAILABLE = "UNAVAILABLE"
STATUSES = (TESTABLE, PARTIAL, UNAVAILABLE)
CENSUS_LOCKED_STATUS = "FINITE_TREE_ALGORITHM_CENSUS_LOCKED"

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "description",
        "snapshot_constraints",
        "status_definitions",
        "required_controls",
        "families",
        "screening_order",
        "claim_locks",
    }
)
_SNAPSHOT_KEYS = frozenset(
    {
        "recording_session_count",
        "pseudo_trial_row_count",
        "timepoint_count",
        "count_window_seconds",
        "time_step_seconds",
        "primary_nonoverlap_stride_bins",
        "simultaneous_403_neuron_population",
        "allowed_dimensions_one_based",
        "dimension_rows_are_paired_trials",
    }
)
_FAMILY_KEYS = frozenset(
    {
        "family_id",
        "equivalence_class",
        "observable_fingerprint",
        "status",
        "current_round",
        "required_observations",
        "reference_url",
    }
)
_REQUIRED_FAMILY_IDS = frozenset(
    {
        "continuous_var",
        "cart_m5_model_tree",
        "oblique_model_tree",
        "soft_decision_tree",
        "hierarchical_mixture_of_experts",
        "flat_or_recurrent_slds",
        "tree_structured_rslds",
        "tree_sparse_routing",
        "hidden_markov_model",
        "hidden_semi_markov_model",
        "hierarchical_hidden_markov_model",
        "context_tree_vlmc",
        "options_semimarkov_control",
        "behavior_tree",
        "hierarchical_task_network",
        "pushdown_call_return",
        "hierarchical_clustering_tree_metric",
        "chow_liu_dependency_tree",
    }
)
_REQUIRED_EQUIVALENCE_CLASSES = frozenset(
    {
        "state_space_routing",
        "latent_temporal_segmentation",
        "control_semantics",
        "recursive_memory",
        "relational_tree",
    }
)
_REQUIRED_CLAIM_LOCKS = frozenset(
    {
        "tree_shaped_predictor_is_not_a_biological_tree",
        "state_hierarchy_is_not_task_inheritance",
        "fixed_depth_hierarchy_is_not_recursion",
        "predictive_routing_is_not_call_return",
        "processed_time_bins_do_not_measure_latency_or_throughput",
        "screening_winner_is_not_confirmed_on_independent_data",
        "optimizer_identity_is_not_available",
    }
)


@dataclass(frozen=True)
class SnapshotConstraints:
    """Processed-data limits that determine which families are screenable."""

    recording_session_count: int
    pseudo_trial_row_count: int
    timepoint_count: int
    count_window_seconds: float
    time_step_seconds: float
    primary_nonoverlap_stride_bins: int
    simultaneous_403_neuron_population: bool
    allowed_dimensions_one_based: tuple[int, ...]
    dimension_rows_are_paired_trials: bool


@dataclass(frozen=True)
class AlgorithmFamily:
    """One behavioral equivalence family in the finite hypothesis universe."""

    family_id: str
    equivalence_class: str
    observable_fingerprint: str
    status: str
    current_round: str
    required_observations: str
    reference_url: str


@dataclass(frozen=True)
class NeuralTreeAlgorithmCensus:
    """Strict, serializable tree-algorithm census."""

    schema_version: str
    scope: str
    method_status: str
    description: str
    snapshot_constraints: SnapshotConstraints
    status_definitions: Mapping[str, str]
    required_controls: tuple[str, ...]
    families: tuple[AlgorithmFamily, ...]
    screening_order: tuple[str, ...]
    claim_locks: tuple[str, ...]

    def family(self, family_id: str) -> AlgorithmFamily:
        """Return one family by stable identifier."""

        matches = tuple(
            item for item in self.families if item.family_id == family_id
        )
        if len(matches) != 1:
            raise KeyError(family_id)
        return matches[0]

    def families_with_status(
        self,
        status: str,
    ) -> tuple[AlgorithmFamily, ...]:
        """Return the families in one declared testability class."""

        if status not in STATUSES:
            raise ValueError(f"unsupported status: {status!r}")
        return tuple(item for item in self.families if item.status == status)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return asdict(self)


def _exact_mapping(
    value: Any,
    *,
    keys: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    observed = frozenset(value)
    missing = sorted(keys - observed)
    unknown = sorted(observed - keys)
    if missing:
        raise ValueError(f"{label} is missing keys: {missing}")
    if unknown:
        raise ValueError(f"{label} has unknown keys: {unknown}")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _text_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty array")
    result = tuple(
        _text(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"{label} must be a positive integer")
    return value


def _positive_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _parse_snapshot(value: Any) -> SnapshotConstraints:
    raw = _exact_mapping(
        value,
        keys=_SNAPSHOT_KEYS,
        label="snapshot_constraints",
    )
    allowed_raw = raw["allowed_dimensions_one_based"]
    if not isinstance(allowed_raw, list):
        raise TypeError("allowed_dimensions_one_based must be an array")
    allowed = tuple(
        _positive_int(item, label="allowed_dimensions_one_based item")
        for item in allowed_raw
    )
    if allowed != (1, 3):
        raise ValueError("only dimensions 1 and 3 may enter the census")
    for key in (
        "simultaneous_403_neuron_population",
        "dimension_rows_are_paired_trials",
    ):
        if type(raw[key]) is not bool:
            raise TypeError(f"snapshot_constraints.{key} must be boolean")
    snapshot = SnapshotConstraints(
        recording_session_count=_positive_int(
            raw["recording_session_count"],
            label="recording_session_count",
        ),
        pseudo_trial_row_count=_positive_int(
            raw["pseudo_trial_row_count"],
            label="pseudo_trial_row_count",
        ),
        timepoint_count=_positive_int(
            raw["timepoint_count"],
            label="timepoint_count",
        ),
        count_window_seconds=_positive_float(
            raw["count_window_seconds"],
            label="count_window_seconds",
        ),
        time_step_seconds=_positive_float(
            raw["time_step_seconds"],
            label="time_step_seconds",
        ),
        primary_nonoverlap_stride_bins=_positive_int(
            raw["primary_nonoverlap_stride_bins"],
            label="primary_nonoverlap_stride_bins",
        ),
        simultaneous_403_neuron_population=raw[
            "simultaneous_403_neuron_population"
        ],
        allowed_dimensions_one_based=allowed,
        dimension_rows_are_paired_trials=raw[
            "dimension_rows_are_paired_trials"
        ],
    )
    if snapshot.recording_session_count != 27:
        raise ValueError("the census must remain locked to 27 sessions")
    if snapshot.pseudo_trial_row_count != 36:
        raise ValueError("the census must remain locked to 36 pseudo-trials")
    if snapshot.timepoint_count != 81:
        raise ValueError("the census must remain locked to 81 timepoints")
    if snapshot.simultaneous_403_neuron_population:
        raise ValueError("the stitched pseudopopulation is not simultaneous")
    if snapshot.dimension_rows_are_paired_trials:
        raise ValueError("D1 and D3 rows must remain unpaired")
    overlap_ratio = (
        snapshot.count_window_seconds / snapshot.time_step_seconds
    )
    if abs(overlap_ratio - snapshot.primary_nonoverlap_stride_bins) > 1e-12:
        raise ValueError("nonoverlap stride must match the count-window width")
    return snapshot


def _parse_family(value: Any, *, index: int) -> AlgorithmFamily:
    raw = _exact_mapping(
        value,
        keys=_FAMILY_KEYS,
        label=f"families[{index}]",
    )
    status = _text(raw["status"], label=f"families[{index}].status")
    if status not in STATUSES:
        raise ValueError(f"unsupported family status: {status!r}")
    reference = _text(
        raw["reference_url"],
        label=f"families[{index}].reference_url",
    )
    if not reference.startswith("https://"):
        raise ValueError("family reference URLs must use HTTPS")
    return AlgorithmFamily(
        family_id=_text(
            raw["family_id"],
            label=f"families[{index}].family_id",
        ),
        equivalence_class=_text(
            raw["equivalence_class"],
            label=f"families[{index}].equivalence_class",
        ),
        observable_fingerprint=_text(
            raw["observable_fingerprint"],
            label=f"families[{index}].observable_fingerprint",
        ),
        status=status,
        current_round=_text(
            raw["current_round"],
            label=f"families[{index}].current_round",
        ),
        required_observations=_text(
            raw["required_observations"],
            label=f"families[{index}].required_observations",
        ),
        reference_url=reference,
    )


def load_neural_tree_algorithm_census(
    path: str | Path,
) -> NeuralTreeAlgorithmCensus:
    """Load and strictly validate the preregistered census."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    top = _exact_mapping(payload, keys=_TOP_LEVEL_KEYS, label="census")
    schema = _text(top["schema_version"], label="schema_version")
    scope = _text(top["scope"], label="scope")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}")
    if scope != CENSUS_SCOPE:
        raise ValueError(f"scope must equal {CENSUS_SCOPE!r}")
    snapshot = _parse_snapshot(top["snapshot_constraints"])

    definitions_raw = top["status_definitions"]
    if not isinstance(definitions_raw, Mapping):
        raise TypeError("status_definitions must be an object")
    if frozenset(definitions_raw) != frozenset(STATUSES):
        raise ValueError("status_definitions must define the three statuses")
    definitions = {
        status: _text(
            definitions_raw[status],
            label=f"status_definitions.{status}",
        )
        for status in STATUSES
    }

    families_raw = top["families"]
    if not isinstance(families_raw, list) or not families_raw:
        raise TypeError("families must be a non-empty array")
    families = tuple(
        _parse_family(value, index=index)
        for index, value in enumerate(families_raw)
    )
    identifiers = tuple(item.family_id for item in families)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("family identifiers must be unique")
    if frozenset(identifiers) != _REQUIRED_FAMILY_IDS:
        missing = sorted(_REQUIRED_FAMILY_IDS - frozenset(identifiers))
        extra = sorted(frozenset(identifiers) - _REQUIRED_FAMILY_IDS)
        raise ValueError(
            f"finite family universe changed; missing={missing}, extra={extra}"
        )
    classes = frozenset(item.equivalence_class for item in families)
    if classes != _REQUIRED_EQUIVALENCE_CLASSES:
        raise ValueError("behavioral equivalence classes changed")
    status_counts = {
        status: sum(item.status == status for item in families)
        for status in STATUSES
    }
    if status_counts != {TESTABLE: 9, PARTIAL: 4, UNAVAILABLE: 5}:
        raise ValueError("family testability counts changed")

    screening_order = _text_tuple(
        top["screening_order"],
        label="screening_order",
    )
    family_map = {item.family_id: item for item in families}
    unknown_screen = sorted(set(screening_order) - set(family_map))
    if unknown_screen:
        raise ValueError(f"screening order has unknown families: {unknown_screen}")
    if any(
        family_map[family_id].status == UNAVAILABLE
        for family_id in screening_order
    ):
        raise ValueError("unavailable families cannot enter screening order")

    locks = _text_tuple(top["claim_locks"], label="claim_locks")
    if frozenset(locks) != _REQUIRED_CLAIM_LOCKS:
        raise ValueError("scientific claim locks changed")
    return NeuralTreeAlgorithmCensus(
        schema_version=schema,
        scope=scope,
        method_status=CENSUS_LOCKED_STATUS,
        description=_text(top["description"], label="description"),
        snapshot_constraints=snapshot,
        status_definitions=definitions,
        required_controls=_text_tuple(
            top["required_controls"],
            label="required_controls",
        ),
        families=families,
        screening_order=screening_order,
        claim_locks=locks,
    )


__all__ = [
    "AlgorithmFamily",
    "CENSUS_LOCKED_STATUS",
    "CENSUS_SCOPE",
    "NeuralTreeAlgorithmCensus",
    "PARTIAL",
    "SCHEMA_VERSION",
    "STATUSES",
    "SnapshotConstraints",
    "TESTABLE",
    "UNAVAILABLE",
    "load_neural_tree_algorithm_census",
]
