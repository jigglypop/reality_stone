"""Oracle-labelled method controls for a neural-language hypothesis.

The gate in this module asks whether a deliberately generated finite-state
system has a few *necessary-looking* signatures of reusable computation:

* a declared assembly-boundary state is predictively sufficient for the next
  transition;
* the same primitive transition is reusable across nuisance contexts and
  repeated invocations;
* learned primitive transitions predict held-out ``A`` then ``B`` tuples;
* shuffled transition targets and a tuple-lookup model fail on those same
  held-out targets while the lookup model memorizes seen tuples; and
* self-loop and cross-assembly feedback topologies survive open-loop rollout
  and fail an edge-ablation control.

The candidate assembly is a finite-state stochastic transducer.  Its state and
message token are visible at the synthetic boundary, while context labels and
microstate labels are nuisance variables by construction.  This makes the
benchmark useful for checking the *logic and implementation* of a proposed
reverse-engineering gate.  Because the discrete state, token, and primitive
labels are supplied by an oracle, it is a forward method control rather than
an inverse recovery of hidden neural structure.  It is not evidence that
biological neurons use
this representation, that a neural Clarus assembly exists, that its operations
are causal instructions, or that a brain programming language has been
identified.

Accordingly, ``full_brain_language_identified`` and
``neural_clarus_assembly_validated`` and
``causal_instruction_set_validated`` are hard-coded false in every report.  A
benchmark that attempts to set any claim true is rejected by the strict loader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "clarus-neural-language-synthetic/v1"
REPORT_SCHEMA_VERSION = "clarus-neural-language-report/v1"
SYNTHETIC_SCOPE = (
    "oracle_labeled_synthetic_finite_state_method_control_only"
)
SYNTHETIC_PASS = "SYNTHETIC_ORACLE_LABELED_METHOD_CONTROL_PASS"
SYNTHETIC_FAIL = "SYNTHETIC_ORACLE_LABELED_METHOD_CONTROL_FAIL"

MODEL_CLASS = "finite_state_stochastic_transducer"
BOUNDARY_STATE = "discrete_assembly_state_z"
INPUT_TOKEN = "discrete_boundary_message_x"
PRIMITIVE_OPERATIONS = ("A", "B")
CONTEXT_ROLE = "held_out_nuisance_label_not_used_by_transition"
MICROSTATE_ROLE = "irrelevant_interior_nuisance_control"

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "description",
        "transducer",
        "generator",
        "thresholds",
        "claim_locks",
        "excluded_inferences",
    }
)
_TRANSDUCER_KEYS = frozenset(
    {
        "model_class",
        "boundary_state",
        "input_token",
        "primitive_operations",
        "context_role",
        "microstate_role",
    }
)
_GENERATOR_KEYS = frozenset(
    {
        "seed",
        "state_count",
        "token_count",
        "context_count",
        "microstate_count",
        "repetitions_per_context",
        "transition_noise",
        "laplace_alpha",
        "composition_trials_per_case",
        "composition_holdout_modulus",
        "recursion_trajectories",
        "recursion_depth",
    }
)
_THRESHOLD_KEYS = frozenset(
    {
        "boundary_accuracy_min",
        "context_transition_tv_max",
        "nuisance_accuracy_gain_max",
        "reuse_accuracy_min",
        "composition_accuracy_min",
        "negative_control_accuracy_max",
        "recursion_step_accuracy_min",
        "feedback_edge_ablation_gap_min",
    }
)
_CLAIM_LOCK_KEYS = frozenset(
    {
        "full_brain_language_identified",
        "neural_clarus_assembly_validated",
        "causal_instruction_set_validated",
    }
)


@dataclass(frozen=True)
class TransducerSpecification:
    """Exact semantic declarations for the synthetic candidate assembly."""

    model_class: str
    boundary_state: str
    input_token: str
    primitive_operations: tuple[str, ...]
    context_role: str
    microstate_role: str


@dataclass(frozen=True)
class SyntheticGeneratorConfig:
    """Deterministic data-generation settings."""

    seed: int
    state_count: int
    token_count: int
    context_count: int
    microstate_count: int
    repetitions_per_context: int
    transition_noise: float
    laplace_alpha: float
    composition_trials_per_case: int
    composition_holdout_modulus: int
    recursion_trajectories: int
    recursion_depth: int


@dataclass(frozen=True)
class NeuralLanguageThresholds:
    """Preregistered pass/fail thresholds for the synthetic controls."""

    boundary_accuracy_min: float
    context_transition_tv_max: float
    nuisance_accuracy_gain_max: float
    reuse_accuracy_min: float
    composition_accuracy_min: float
    negative_control_accuracy_max: float
    recursion_step_accuracy_min: float
    feedback_edge_ablation_gap_min: float


@dataclass(frozen=True)
class NeuralLanguageSyntheticBenchmark:
    """Strictly loaded benchmark and locked epistemic scope."""

    schema_version: str
    scope: str
    description: str
    transducer: TransducerSpecification
    generator: SyntheticGeneratorConfig
    thresholds: NeuralLanguageThresholds
    full_brain_language_identified: bool
    neural_clarus_assembly_validated: bool
    causal_instruction_set_validated: bool
    excluded_inferences: tuple[str, ...]


@dataclass(frozen=True)
class ContextAccuracy:
    """Leave-one-context-out transition accuracy."""

    context: int
    accuracy: float


@dataclass(frozen=True)
class OperationAccuracy:
    """Late-repetition accuracy for one reusable primitive."""

    operation: str
    accuracy: float


@dataclass(frozen=True)
class BoundaryClosureAudit:
    """Predictive sufficiency of the declared cell boundary."""

    sample_count: int
    boundary_variables: tuple[str, ...]
    nuisance_variables: tuple[str, ...]
    leave_one_context_out_accuracy: tuple[ContextAccuracy, ...]
    mean_heldout_context_accuracy: float
    minimum_heldout_context_accuracy: float
    maximum_context_transition_total_variation: float
    boundary_late_repetition_accuracy: float
    nuisance_augmented_late_repetition_accuracy: float
    nuisance_accuracy_gain: float
    predictive_sufficiency_pass: bool
    empirical_closure_pass: bool


@dataclass(frozen=True)
class ReuseAudit:
    """Context and repetition reuse of the same learned operations."""

    operation_accuracy: tuple[OperationAccuracy, ...]
    minimum_operation_accuracy: float
    early_to_late_repetition_accuracy: float
    minimum_heldout_context_accuracy: float
    same_operation_reused_across_contexts: bool
    same_operation_reused_across_repetitions: bool
    structural_pass: bool


@dataclass(frozen=True)
class CompositionAudit:
    """Primitive composition and two mandatory negative controls."""

    sequence: tuple[str, ...]
    evaluation_count: int
    heldout_case_count: int
    seen_evaluation_count: int
    primitive_composition_accuracy: float
    shuffled_target_composition_accuracy: float
    noncompositional_lookup_accuracy: float
    lookup_seen_memorization_accuracy: float
    true_composition_pass: bool
    shuffled_targets_passed_composition_gate: bool
    noncompositional_lookup_passed_composition_gate: bool
    shuffled_target_control_rejected: bool
    noncompositional_lookup_control_rejected: bool
    negative_controls_rejected: bool
    structural_pass: bool


@dataclass(frozen=True)
class FeedbackTopologyAudit:
    """One open-loop feedback topology, not a recursive-syntax claim."""

    topology: str
    evaluation_mode: str
    directed_edges: tuple[tuple[str, str], ...]
    self_loop_count: int
    cross_assembly_edge_count: int
    cycle_closed: bool
    trajectory_count: int
    recursion_depth: int
    predicted_transition_count: int
    step_accuracy: float
    severed_edge_step_accuracy: float
    edge_ablation_gap: float
    edge_dependency_pass: bool
    structural_pass: bool


@dataclass(frozen=True)
class NeuralLanguageGateReport:
    """Serializable result with biological claims permanently locked false."""

    schema_version: str
    scope: str
    structural_status: str
    synthetic_oracle_labeled_method_control_pass: bool
    boundary_closure_audit: BoundaryClosureAudit
    reuse_audit: ReuseAudit
    composition_audit: CompositionAudit
    self_feedback_audit: FeedbackTopologyAudit
    cross_assembly_feedback_audit: FeedbackTopologyAudit
    shuffled_target_control_rejected: bool
    noncompositional_lookup_control_rejected: bool
    real_neural_data_used: bool
    full_brain_language_identified: bool
    neural_clarus_assembly_validated: bool
    causal_instruction_set_validated: bool
    excluded_inferences: tuple[str, ...]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


@dataclass(frozen=True)
class _SyntheticSamples:
    state: np.ndarray
    token: np.ndarray
    operation: np.ndarray
    context: np.ndarray
    repetition: np.ndarray
    microstate: np.ndarray
    next_state: np.ndarray

    @property
    def size(self) -> int:
        return int(self.state.size)


def _strict_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    parent: str,
) -> None:
    actual = frozenset(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{parent} missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{parent} has unknown keys: {unknown}")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = tuple(_string(item, name=f"{name}[]") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _integer(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _real(
    value: Any,
    *,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    lower_bad = result < lower if lower_inclusive else result <= lower
    upper_bad = result > upper if upper_inclusive else result >= upper
    if lower_bad or upper_bad:
        left = "[" if lower_inclusive else "("
        right = "]" if upper_inclusive else ")"
        raise ValueError(f"{name} must be in {left}{lower}, {upper}{right}")
    return result


def _claim_lock(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    if value:
        raise ValueError(f"{name} must be false for this synthetic benchmark")
    return False


def _parse_transducer(payload: Mapping[str, Any]) -> TransducerSpecification:
    _strict_keys(payload, _TRANSDUCER_KEYS, parent="transducer")
    specification = TransducerSpecification(
        model_class=_string(payload["model_class"], name="transducer.model_class"),
        boundary_state=_string(
            payload["boundary_state"],
            name="transducer.boundary_state",
        ),
        input_token=_string(
            payload["input_token"],
            name="transducer.input_token",
        ),
        primitive_operations=_string_tuple(
            payload["primitive_operations"],
            name="transducer.primitive_operations",
        ),
        context_role=_string(
            payload["context_role"],
            name="transducer.context_role",
        ),
        microstate_role=_string(
            payload["microstate_role"],
            name="transducer.microstate_role",
        ),
    )
    expected = TransducerSpecification(
        model_class=MODEL_CLASS,
        boundary_state=BOUNDARY_STATE,
        input_token=INPUT_TOKEN,
        primitive_operations=PRIMITIVE_OPERATIONS,
        context_role=CONTEXT_ROLE,
        microstate_role=MICROSTATE_ROLE,
    )
    if specification != expected:
        raise ValueError(
            "transducer declarations must exactly match the synthetic control "
            "conventions"
        )
    return specification


def _parse_generator(payload: Mapping[str, Any]) -> SyntheticGeneratorConfig:
    _strict_keys(payload, _GENERATOR_KEYS, parent="generator")
    config = SyntheticGeneratorConfig(
        seed=_integer(payload["seed"], name="generator.seed", minimum=0),
        state_count=_integer(
            payload["state_count"],
            name="generator.state_count",
            minimum=3,
        ),
        token_count=_integer(
            payload["token_count"],
            name="generator.token_count",
            minimum=2,
        ),
        context_count=_integer(
            payload["context_count"],
            name="generator.context_count",
            minimum=3,
        ),
        microstate_count=_integer(
            payload["microstate_count"],
            name="generator.microstate_count",
            minimum=2,
        ),
        repetitions_per_context=_integer(
            payload["repetitions_per_context"],
            name="generator.repetitions_per_context",
            minimum=10,
        ),
        transition_noise=_real(
            payload["transition_noise"],
            name="generator.transition_noise",
            lower=0.0,
            upper=0.5,
            upper_inclusive=False,
        ),
        laplace_alpha=_real(
            payload["laplace_alpha"],
            name="generator.laplace_alpha",
            lower=0.0,
            upper=10.0,
            lower_inclusive=False,
        ),
        composition_trials_per_case=_integer(
            payload["composition_trials_per_case"],
            name="generator.composition_trials_per_case",
            minimum=1,
        ),
        composition_holdout_modulus=_integer(
            payload["composition_holdout_modulus"],
            name="generator.composition_holdout_modulus",
            minimum=2,
        ),
        recursion_trajectories=_integer(
            payload["recursion_trajectories"],
            name="generator.recursion_trajectories",
            minimum=1,
        ),
        recursion_depth=_integer(
            payload["recursion_depth"],
            name="generator.recursion_depth",
            minimum=2,
        ),
    )
    if config.token_count != config.state_count:
        raise ValueError(
            "generator.token_count must equal generator.state_count because "
            "feedback feeds emitted states back as boundary messages"
        )
    return config


def _parse_thresholds(payload: Mapping[str, Any]) -> NeuralLanguageThresholds:
    _strict_keys(payload, _THRESHOLD_KEYS, parent="thresholds")

    def probability(key: str) -> float:
        return _real(
            payload[key],
            name=f"thresholds.{key}",
            lower=0.0,
            upper=1.0,
        )

    thresholds = NeuralLanguageThresholds(
        boundary_accuracy_min=probability("boundary_accuracy_min"),
        context_transition_tv_max=probability(
            "context_transition_tv_max"
        ),
        nuisance_accuracy_gain_max=probability(
            "nuisance_accuracy_gain_max"
        ),
        reuse_accuracy_min=probability("reuse_accuracy_min"),
        composition_accuracy_min=probability("composition_accuracy_min"),
        negative_control_accuracy_max=probability(
            "negative_control_accuracy_max"
        ),
        recursion_step_accuracy_min=probability(
            "recursion_step_accuracy_min"
        ),
        feedback_edge_ablation_gap_min=probability(
            "feedback_edge_ablation_gap_min"
        ),
    )
    if (
        thresholds.negative_control_accuracy_max
        >= thresholds.composition_accuracy_min
    ):
        raise ValueError(
            "thresholds.negative_control_accuracy_max must be lower than "
            "thresholds.composition_accuracy_min"
        )
    if (
        thresholds.feedback_edge_ablation_gap_min
        >= thresholds.recursion_step_accuracy_min
    ):
        raise ValueError(
            "thresholds.feedback_edge_ablation_gap_min must be lower than "
            "thresholds.recursion_step_accuracy_min"
        )
    return thresholds


def load_neural_language_benchmark(
    path: str | Path,
) -> NeuralLanguageSyntheticBenchmark:
    """Load and strictly validate one deterministic synthetic benchmark."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load benchmark {source}: {error}") from error
    root = _mapping(payload, name="benchmark")
    _strict_keys(root, _TOP_LEVEL_KEYS, parent="benchmark")

    schema_version = _string(
        root["schema_version"],
        name="benchmark.schema_version",
    )
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"benchmark.schema_version must be {SCHEMA_VERSION!r}")
    scope = _string(root["scope"], name="benchmark.scope")
    if scope != SYNTHETIC_SCOPE:
        raise ValueError(f"benchmark.scope must be {SYNTHETIC_SCOPE!r}")

    claims = _mapping(root["claim_locks"], name="claim_locks")
    _strict_keys(claims, _CLAIM_LOCK_KEYS, parent="claim_locks")
    excluded = _string_tuple(
        root["excluded_inferences"],
        name="benchmark.excluded_inferences",
    )
    required_exclusions = {
        "biological_clarus_cell_existence",
        "brain_programming_language_identification",
        "neural_spike_semantics",
        "causal_neural_mechanism",
    }
    if not required_exclusions.issubset(excluded):
        raise ValueError(
            "benchmark.excluded_inferences must declare every biological "
            "and semantic exclusion"
        )

    return NeuralLanguageSyntheticBenchmark(
        schema_version=schema_version,
        scope=scope,
        description=_string(
            root["description"],
            name="benchmark.description",
        ),
        transducer=_parse_transducer(
            _mapping(root["transducer"], name="transducer")
        ),
        generator=_parse_generator(
            _mapping(root["generator"], name="generator")
        ),
        thresholds=_parse_thresholds(
            _mapping(root["thresholds"], name="thresholds")
        ),
        full_brain_language_identified=_claim_lock(
            claims["full_brain_language_identified"],
            name="claim_locks.full_brain_language_identified",
        ),
        neural_clarus_assembly_validated=_claim_lock(
            claims["neural_clarus_assembly_validated"],
            name="claim_locks.neural_clarus_assembly_validated",
        ),
        causal_instruction_set_validated=_claim_lock(
            claims["causal_instruction_set_validated"],
            name="claim_locks.causal_instruction_set_validated",
        ),
        excluded_inferences=excluded,
    )


def _intended_transition(
    operation: np.ndarray | int,
    state: np.ndarray | int,
    token: np.ndarray | int,
    state_count: int,
) -> np.ndarray:
    operation_array = np.asarray(operation, dtype=np.int64)
    state_array = np.asarray(state, dtype=np.int64)
    token_array = np.asarray(token, dtype=np.int64)
    operation_a = (state_array + token_array + 1) % state_count
    operation_b = (2 * state_array + token_array + 2) % state_count
    return np.where(operation_array == 0, operation_a, operation_b)


def _noisy_transition(
    *,
    operation: int,
    state: int,
    token: int,
    config: SyntheticGeneratorConfig,
    rng: np.random.Generator,
) -> int:
    intended = int(
        _intended_transition(
            operation,
            state,
            token,
            config.state_count,
        )
    )
    if rng.random() >= config.transition_noise:
        return intended
    offset = int(rng.integers(1, config.state_count))
    return (intended + offset) % config.state_count


def _generate_samples(
    config: SyntheticGeneratorConfig,
    rng: np.random.Generator,
) -> _SyntheticSamples:
    rows = (
        config.context_count
        * config.repetitions_per_context
        * len(PRIMITIVE_OPERATIONS)
        * config.state_count
        * config.token_count
    )
    state = np.empty(rows, dtype=np.int64)
    token = np.empty(rows, dtype=np.int64)
    operation = np.empty(rows, dtype=np.int64)
    context = np.empty(rows, dtype=np.int64)
    repetition = np.empty(rows, dtype=np.int64)
    microstate = rng.integers(
        0,
        config.microstate_count,
        size=rows,
        dtype=np.int64,
    )

    cursor = 0
    for context_index in range(config.context_count):
        for repetition_index in range(config.repetitions_per_context):
            for operation_index in range(len(PRIMITIVE_OPERATIONS)):
                for state_index in range(config.state_count):
                    for token_index in range(config.token_count):
                        state[cursor] = state_index
                        token[cursor] = token_index
                        operation[cursor] = operation_index
                        context[cursor] = context_index
                        repetition[cursor] = repetition_index
                        cursor += 1

    intended = _intended_transition(
        operation,
        state,
        token,
        config.state_count,
    )
    noisy = rng.random(rows) < config.transition_noise
    offsets = rng.integers(1, config.state_count, size=rows, dtype=np.int64)
    next_state = np.where(
        noisy,
        (intended + offsets) % config.state_count,
        intended,
    ).astype(np.int64)
    return _SyntheticSamples(
        state=state,
        token=token,
        operation=operation,
        context=context,
        repetition=repetition,
        microstate=microstate,
        next_state=next_state,
    )


def _transition_model(
    samples: _SyntheticSamples,
    mask: np.ndarray,
    config: SyntheticGeneratorConfig,
    *,
    shuffled_targets: np.ndarray | None = None,
) -> np.ndarray:
    counts = np.full(
        (
            len(PRIMITIVE_OPERATIONS),
            config.state_count,
            config.token_count,
            config.state_count,
        ),
        config.laplace_alpha,
        dtype=np.float64,
    )
    targets = samples.next_state if shuffled_targets is None else shuffled_targets
    np.add.at(
        counts,
        (
            samples.operation[mask],
            samples.state[mask],
            samples.token[mask],
            targets[mask],
        ),
        1.0,
    )
    return counts / counts.sum(axis=-1, keepdims=True)


def _augmented_transition_model(
    samples: _SyntheticSamples,
    mask: np.ndarray,
    config: SyntheticGeneratorConfig,
) -> np.ndarray:
    counts = np.full(
        (
            config.context_count,
            config.microstate_count,
            len(PRIMITIVE_OPERATIONS),
            config.state_count,
            config.token_count,
            config.state_count,
        ),
        config.laplace_alpha,
        dtype=np.float64,
    )
    np.add.at(
        counts,
        (
            samples.context[mask],
            samples.microstate[mask],
            samples.operation[mask],
            samples.state[mask],
            samples.token[mask],
            samples.next_state[mask],
        ),
        1.0,
    )
    return counts / counts.sum(axis=-1, keepdims=True)


def _model_accuracy(
    model: np.ndarray,
    samples: _SyntheticSamples,
    mask: np.ndarray,
) -> float:
    predictions = np.argmax(
        model[
            samples.operation[mask],
            samples.state[mask],
            samples.token[mask],
        ],
        axis=-1,
    )
    return float(np.mean(predictions == samples.next_state[mask]))


def _augmented_model_accuracy(
    model: np.ndarray,
    samples: _SyntheticSamples,
    mask: np.ndarray,
) -> float:
    predictions = np.argmax(
        model[
            samples.context[mask],
            samples.microstate[mask],
            samples.operation[mask],
            samples.state[mask],
            samples.token[mask],
        ],
        axis=-1,
    )
    return float(np.mean(predictions == samples.next_state[mask]))


def _audit_boundary_and_reuse(
    samples: _SyntheticSamples,
    config: SyntheticGeneratorConfig,
    thresholds: NeuralLanguageThresholds,
) -> tuple[BoundaryClosureAudit, ReuseAudit, np.ndarray]:
    all_rows = np.ones(samples.size, dtype=bool)
    pooled_model = _transition_model(samples, all_rows, config)
    context_results: list[ContextAccuracy] = []
    transition_divergences: list[float] = []
    for context_index in range(config.context_count):
        heldout = samples.context == context_index
        leave_one_out = _transition_model(samples, ~heldout, config)
        context_results.append(
            ContextAccuracy(
                context=context_index,
                accuracy=_model_accuracy(leave_one_out, samples, heldout),
            )
        )
        context_model = _transition_model(samples, heldout, config)
        total_variation = 0.5 * np.abs(
            context_model - leave_one_out
        ).sum(axis=-1)
        transition_divergences.append(float(np.mean(total_variation)))

    split = config.repetitions_per_context // 2
    early = samples.repetition < split
    late = ~early
    early_model = _transition_model(samples, early, config)
    augmented_model = _augmented_transition_model(samples, early, config)
    boundary_late_accuracy = _model_accuracy(early_model, samples, late)
    augmented_late_accuracy = _augmented_model_accuracy(
        augmented_model,
        samples,
        late,
    )
    nuisance_gain = augmented_late_accuracy - boundary_late_accuracy
    heldout_accuracies = tuple(item.accuracy for item in context_results)
    mean_heldout_accuracy = float(np.mean(heldout_accuracies))
    minimum_heldout_accuracy = min(heldout_accuracies)
    maximum_context_tv = max(transition_divergences)
    predictive_sufficiency = (
        mean_heldout_accuracy >= thresholds.boundary_accuracy_min
        and boundary_late_accuracy >= thresholds.boundary_accuracy_min
        and nuisance_gain <= thresholds.nuisance_accuracy_gain_max
    )
    closure_pass = (
        predictive_sufficiency
        and maximum_context_tv <= thresholds.context_transition_tv_max
    )
    boundary_audit = BoundaryClosureAudit(
        sample_count=samples.size,
        boundary_variables=(BOUNDARY_STATE, INPUT_TOKEN, "primitive_operation"),
        nuisance_variables=("context_label", "interior_microstate_label"),
        leave_one_context_out_accuracy=tuple(context_results),
        mean_heldout_context_accuracy=mean_heldout_accuracy,
        minimum_heldout_context_accuracy=minimum_heldout_accuracy,
        maximum_context_transition_total_variation=maximum_context_tv,
        boundary_late_repetition_accuracy=boundary_late_accuracy,
        nuisance_augmented_late_repetition_accuracy=augmented_late_accuracy,
        nuisance_accuracy_gain=nuisance_gain,
        predictive_sufficiency_pass=predictive_sufficiency,
        empirical_closure_pass=closure_pass,
    )

    operation_results: list[OperationAccuracy] = []
    for operation_index, operation_name in enumerate(PRIMITIVE_OPERATIONS):
        operation_mask = late & (samples.operation == operation_index)
        operation_results.append(
            OperationAccuracy(
                operation=operation_name,
                accuracy=_model_accuracy(
                    early_model,
                    samples,
                    operation_mask,
                ),
            )
        )
    minimum_operation_accuracy = min(
        item.accuracy for item in operation_results
    )
    context_reuse = (
        minimum_heldout_accuracy >= thresholds.reuse_accuracy_min
    )
    repetition_reuse = (
        boundary_late_accuracy >= thresholds.reuse_accuracy_min
        and minimum_operation_accuracy >= thresholds.reuse_accuracy_min
    )
    reuse_audit = ReuseAudit(
        operation_accuracy=tuple(operation_results),
        minimum_operation_accuracy=minimum_operation_accuracy,
        early_to_late_repetition_accuracy=boundary_late_accuracy,
        minimum_heldout_context_accuracy=minimum_heldout_accuracy,
        same_operation_reused_across_contexts=context_reuse,
        same_operation_reused_across_repetitions=repetition_reuse,
        structural_pass=context_reuse and repetition_reuse,
    )
    return boundary_audit, reuse_audit, pooled_model


def _composition_predictions(
    model: np.ndarray,
    state: np.ndarray,
    first_token: np.ndarray,
    second_token: np.ndarray,
) -> np.ndarray:
    predictions = np.empty(state.size, dtype=np.int64)
    for index in range(state.size):
        intermediate = model[0, state[index], first_token[index], :]
        final_distribution = (
            intermediate[:, None]
            * model[1, :, second_token[index], :]
        ).sum(axis=0)
        predictions[index] = int(np.argmax(final_distribution))
    return predictions


def _audit_composition(
    samples: _SyntheticSamples,
    model: np.ndarray,
    config: SyntheticGeneratorConfig,
    thresholds: NeuralLanguageThresholds,
    rng: np.random.Generator,
) -> CompositionAudit:
    states: list[int] = []
    first_tokens: list[int] = []
    second_tokens: list[int] = []
    true_targets: list[int] = []
    heldout_rows: list[bool] = []
    total_case_count = (
        config.state_count * config.token_count * config.token_count
    )
    holdout_case_target = max(
        1,
        total_case_count // config.composition_holdout_modulus,
    )
    split_rng = np.random.default_rng(config.seed + 104_729)
    heldout_case_indices = frozenset(
        int(index)
        for index in split_rng.choice(
            total_case_count,
            size=holdout_case_target,
            replace=False,
        )
    )
    case_index = 0
    for state in range(config.state_count):
        for first_token in range(config.token_count):
            for second_token in range(config.token_count):
                is_heldout = case_index in heldout_case_indices
                case_index += 1
                for _ in range(config.composition_trials_per_case):
                    middle = _noisy_transition(
                        operation=0,
                        state=state,
                        token=first_token,
                        config=config,
                        rng=rng,
                    )
                    final = _noisy_transition(
                        operation=1,
                        state=middle,
                        token=second_token,
                        config=config,
                        rng=rng,
                    )
                    states.append(state)
                    first_tokens.append(first_token)
                    second_tokens.append(second_token)
                    true_targets.append(final)
                    heldout_rows.append(is_heldout)

    state_array = np.asarray(states, dtype=np.int64)
    first_array = np.asarray(first_tokens, dtype=np.int64)
    second_array = np.asarray(second_tokens, dtype=np.int64)
    true_array = np.asarray(true_targets, dtype=np.int64)
    heldout = np.asarray(heldout_rows, dtype=bool)
    seen = ~heldout
    primitive_predictions = _composition_predictions(
        model,
        state_array,
        first_array,
        second_array,
    )
    primitive_accuracy = float(
        np.mean(primitive_predictions[heldout] == true_array[heldout])
    )

    shuffled_targets = samples.next_state.copy()
    rng.shuffle(shuffled_targets)
    all_rows = np.ones(samples.size, dtype=bool)
    shuffled_model = _transition_model(
        samples,
        all_rows,
        config,
        shuffled_targets=shuffled_targets,
    )
    shuffled_predictions = _composition_predictions(
        shuffled_model,
        state_array,
        first_array,
        second_array,
    )
    shuffled_accuracy = float(
        np.mean(shuffled_predictions[heldout] == true_array[heldout])
    )

    lookup_counts = np.full(
        (
            config.state_count,
            config.token_count,
            config.token_count,
            config.state_count,
        ),
        config.laplace_alpha,
        dtype=np.float64,
    )
    np.add.at(
        lookup_counts,
        (
            state_array[seen],
            first_array[seen],
            second_array[seen],
            true_array[seen],
        ),
        1.0,
    )
    lookup_predictions = np.argmax(lookup_counts, axis=-1)
    predicted_lookup_targets = lookup_predictions[
        state_array,
        first_array,
        second_array,
    ]
    noncompositional_accuracy = float(
        np.mean(
            predicted_lookup_targets[heldout]
            == true_array[heldout]
        )
    )
    lookup_seen_accuracy = float(
        np.mean(predicted_lookup_targets[seen] == true_array[seen])
    )

    true_pass = primitive_accuracy >= thresholds.composition_accuracy_min
    shuffled_pass = (
        shuffled_accuracy >= thresholds.composition_accuracy_min
    )
    lookup_pass = (
        noncompositional_accuracy >= thresholds.composition_accuracy_min
    )
    shuffled_rejected = (
        shuffled_accuracy <= thresholds.negative_control_accuracy_max
        and not shuffled_pass
    )
    lookup_rejected = (
        noncompositional_accuracy
        <= thresholds.negative_control_accuracy_max
        and not lookup_pass
        and lookup_seen_accuracy >= thresholds.composition_accuracy_min
    )
    controls_rejected = shuffled_rejected and lookup_rejected
    return CompositionAudit(
        sequence=PRIMITIVE_OPERATIONS,
        evaluation_count=int(np.sum(heldout)),
        heldout_case_count=len(heldout_case_indices),
        seen_evaluation_count=int(np.sum(seen)),
        primitive_composition_accuracy=primitive_accuracy,
        shuffled_target_composition_accuracy=shuffled_accuracy,
        noncompositional_lookup_accuracy=noncompositional_accuracy,
        lookup_seen_memorization_accuracy=lookup_seen_accuracy,
        true_composition_pass=true_pass,
        shuffled_targets_passed_composition_gate=shuffled_pass,
        noncompositional_lookup_passed_composition_gate=lookup_pass,
        shuffled_target_control_rejected=shuffled_rejected,
        noncompositional_lookup_control_rejected=lookup_rejected,
        negative_controls_rejected=controls_rejected,
        structural_pass=true_pass and controls_rejected,
    )


def _has_directed_cycle(edges: tuple[tuple[str, str], ...]) -> bool:
    adjacency: dict[str, set[str]] = {}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for target in adjacency[node]:
            if visit(target):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in adjacency)


def _audit_self_feedback(
    model: np.ndarray,
    config: SyntheticGeneratorConfig,
    thresholds: NeuralLanguageThresholds,
    rng: np.random.Generator,
) -> FeedbackTopologyAudit:
    correct = 0
    severed_correct = 0
    total = 0
    predictor = np.argmax(model, axis=-1)
    for _ in range(config.recursion_trajectories):
        true_state = int(rng.integers(config.state_count))
        predicted_state = true_state
        severed_state = true_state
        for depth_index in range(config.recursion_depth):
            operation = depth_index % len(PRIMITIVE_OPERATIONS)
            true_next = int(
                _intended_transition(
                    operation,
                    true_state,
                    true_state,
                    config.state_count,
                )
            )
            predicted_next = int(
                predictor[operation, predicted_state, predicted_state]
            )
            severed_token = (severed_state + 1) % config.token_count
            severed_next = int(
                predictor[operation, severed_state, severed_token]
            )
            correct += int(predicted_next == true_next)
            severed_correct += int(severed_next == true_next)
            total += 1
            true_state = true_next
            predicted_state = predicted_next
            severed_state = severed_next
    accuracy = correct / total
    severed_accuracy = severed_correct / total
    ablation_gap = accuracy - severed_accuracy
    edges = (("assembly_0", "assembly_0"),)
    cycle_closed = _has_directed_cycle(edges)
    edge_dependency_pass = (
        ablation_gap >= thresholds.feedback_edge_ablation_gap_min
    )
    return FeedbackTopologyAudit(
        topology="single_assembly_self_feedback",
        evaluation_mode="noiseless_open_loop_depth_extrapolation",
        directed_edges=edges,
        self_loop_count=sum(source == target for source, target in edges),
        cross_assembly_edge_count=sum(
            source != target for source, target in edges
        ),
        cycle_closed=cycle_closed,
        trajectory_count=config.recursion_trajectories,
        recursion_depth=config.recursion_depth,
        predicted_transition_count=total,
        step_accuracy=accuracy,
        severed_edge_step_accuracy=severed_accuracy,
        edge_ablation_gap=ablation_gap,
        edge_dependency_pass=edge_dependency_pass,
        structural_pass=(
            cycle_closed
            and accuracy >= thresholds.recursion_step_accuracy_min
            and edge_dependency_pass
        ),
    )


def _audit_cross_assembly_feedback(
    model: np.ndarray,
    config: SyntheticGeneratorConfig,
    thresholds: NeuralLanguageThresholds,
    rng: np.random.Generator,
) -> FeedbackTopologyAudit:
    correct = 0
    severed_correct = 0
    total = 0
    predictor = np.argmax(model, axis=-1)
    for _ in range(config.recursion_trajectories):
        true_a = int(rng.integers(config.state_count))
        true_b = int(rng.integers(config.state_count))
        predicted_a, predicted_b = true_a, true_b
        severed_a, severed_b = true_a, true_b
        for _ in range(config.recursion_depth):
            true_next_a = int(
                _intended_transition(
                    0,
                    true_a,
                    true_b,
                    config.state_count,
                )
            )
            true_next_b = int(
                _intended_transition(
                    1,
                    true_b,
                    true_a,
                    config.state_count,
                )
            )
            predicted_next_a = int(
                predictor[0, predicted_a, predicted_b]
            )
            predicted_next_b = int(
                predictor[1, predicted_b, predicted_a]
            )
            severed_next_a = int(
                predictor[0, severed_a, severed_a]
            )
            severed_next_b = int(
                predictor[1, severed_b, severed_b]
            )
            correct += int(predicted_next_a == true_next_a)
            correct += int(predicted_next_b == true_next_b)
            severed_correct += int(severed_next_a == true_next_a)
            severed_correct += int(severed_next_b == true_next_b)
            total += 2
            true_a, true_b = true_next_a, true_next_b
            predicted_a, predicted_b = predicted_next_a, predicted_next_b
            severed_a, severed_b = severed_next_a, severed_next_b
    accuracy = correct / total
    severed_accuracy = severed_correct / total
    ablation_gap = accuracy - severed_accuracy
    edges = (
        ("assembly_0", "assembly_1"),
        ("assembly_1", "assembly_0"),
    )
    cycle_closed = _has_directed_cycle(edges)
    edge_dependency_pass = (
        ablation_gap >= thresholds.feedback_edge_ablation_gap_min
    )
    return FeedbackTopologyAudit(
        topology="two_assembly_mutual_feedback",
        evaluation_mode="noiseless_open_loop_depth_extrapolation",
        directed_edges=edges,
        self_loop_count=sum(source == target for source, target in edges),
        cross_assembly_edge_count=sum(
            source != target for source, target in edges
        ),
        cycle_closed=cycle_closed,
        trajectory_count=config.recursion_trajectories,
        recursion_depth=config.recursion_depth,
        predicted_transition_count=total,
        step_accuracy=accuracy,
        severed_edge_step_accuracy=severed_accuracy,
        edge_ablation_gap=ablation_gap,
        edge_dependency_pass=edge_dependency_pass,
        structural_pass=(
            cycle_closed
            and accuracy >= thresholds.recursion_step_accuracy_min
            and edge_dependency_pass
        ),
    )


def neural_language_gate_report(
    benchmark: NeuralLanguageSyntheticBenchmark,
) -> NeuralLanguageGateReport:
    """Run all synthetic controls without promoting biological claims."""
    if not isinstance(benchmark, NeuralLanguageSyntheticBenchmark):
        raise TypeError(
            "benchmark must be a NeuralLanguageSyntheticBenchmark"
        )
    if (
        benchmark.full_brain_language_identified
        or benchmark.neural_clarus_assembly_validated
        or benchmark.causal_instruction_set_validated
    ):
        raise ValueError("synthetic benchmark claim locks must remain false")

    rng = np.random.default_rng(benchmark.generator.seed)
    samples = _generate_samples(benchmark.generator, rng)
    boundary, reuse, model = _audit_boundary_and_reuse(
        samples,
        benchmark.generator,
        benchmark.thresholds,
    )
    composition = _audit_composition(
        samples,
        model,
        benchmark.generator,
        benchmark.thresholds,
        rng,
    )
    self_feedback = _audit_self_feedback(
        model,
        benchmark.generator,
        benchmark.thresholds,
        rng,
    )
    cross_feedback = _audit_cross_assembly_feedback(
        model,
        benchmark.generator,
        benchmark.thresholds,
        rng,
    )
    synthetic_pass = (
        boundary.empirical_closure_pass
        and reuse.structural_pass
        and composition.structural_pass
        and self_feedback.structural_pass
        and cross_feedback.structural_pass
    )
    if synthetic_pass:
        conclusion = (
            "The oracle-labelled finite-state transducer passed the "
            "manifest-declared method controls, including fair held-out "
            "composition and open-loop feedback edge ablations. This "
            "validates only the forward synthetic pipeline; no neural "
            "Clarus assembly, causal instruction set, or brain programming "
            "language has been identified."
        )
    else:
        conclusion = (
            "At least one oracle-labelled synthetic method control failed. "
            "No inference about neural Clarus assemblies, causal "
            "instructions, or a brain programming language is available."
        )
    return NeuralLanguageGateReport(
        schema_version=REPORT_SCHEMA_VERSION,
        scope=benchmark.scope,
        structural_status=SYNTHETIC_PASS if synthetic_pass else SYNTHETIC_FAIL,
        synthetic_oracle_labeled_method_control_pass=synthetic_pass,
        boundary_closure_audit=boundary,
        reuse_audit=reuse,
        composition_audit=composition,
        self_feedback_audit=self_feedback,
        cross_assembly_feedback_audit=cross_feedback,
        shuffled_target_control_rejected=(
            composition.shuffled_target_control_rejected
        ),
        noncompositional_lookup_control_rejected=(
            composition.noncompositional_lookup_control_rejected
        ),
        real_neural_data_used=False,
        full_brain_language_identified=False,
        neural_clarus_assembly_validated=False,
        causal_instruction_set_validated=False,
        excluded_inferences=benchmark.excluded_inferences,
        conclusion=conclusion,
    )
