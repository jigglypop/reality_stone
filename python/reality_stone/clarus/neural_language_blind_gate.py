"""Partial-blind synthetic inverse control for neural-language hypotheses.

This module deliberately removes the strongest oracle assumptions made by the
forward synthetic neural-language gate.  The inverse procedure sees
neuron-like population observations and is told only:

* the external input token;
* the task/operation cue;
* a partition of observed neurons into candidate groups, including the
  cross-session identity of those supplied groups;
* the latent-state cardinality; and
* paired pre/post transition boundaries, shared cue semantics, the calibration
  split, and which sessions are training sessions versus one held-out session.

The latent state labels and the identity of the generator-target candidate
group
are hidden from the inverse procedure.  Candidate selection uses training
sessions only.  In the held-out session, an early calibration split is used to
fit clusters and align their arbitrary labels; all reported predictive
evaluation uses the untouched late split.  Synthetic ground truth is passed
only to a separate scoring function after inference has completed.

Each session applies a different latent-code permutation, neuron permutation,
linear mixing, neuron dropout mask, and observation noise.  Deterministic
k-means recovers local state clusters, while exhaustive label-permutation
alignment connects those local labels to a training-session transition model.
Distractor groups contain equally clusterable but session-specific dynamics.

This remains a partial-blind *synthetic method control*.  Tokens, operation
cues, candidate partitions, session boundaries, and latent cardinality are
still supplied.  It neither analyzes real neural data nor validates a neural
programming language.  Every biological and fully blind discovery claim is a
strict false lock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "clarus-neural-language-partial-blind-synthetic/v1"
REPORT_SCHEMA_VERSION = (
    "clarus-neural-language-partial-blind-synthetic-report/v1"
)
PARTIAL_BLIND_SCOPE = "partial_blind_synthetic_inverse_method_control_only"
PARTIAL_BLIND_SYNTHETIC_PASS = "PARTIAL_BLIND_SYNTHETIC_PASS"
PARTIAL_BLIND_SYNTHETIC_FAIL = "PARTIAL_BLIND_SYNTHETIC_FAIL"
PARTIAL_BLIND_SYNTHETIC_AMBIGUOUS = (
    "PARTIAL_BLIND_SYNTHETIC_AMBIGUOUS"
)

KNOWN_ITEMS = (
    "external_input_token",
    "task_operation_cue",
    "candidate_group_partition",
    "cross_session_candidate_identity",
    "latent_cardinality",
    "operation_label_and_cardinality",
    "paired_pre_post_transition_boundary",
    "cross_session_operation_token_semantics",
    "calibration_split_location",
    "context_session_identity",
)
HIDDEN_ITEMS = (
    "latent_state_labels",
    "generator_target_candidate",
)
OPERATION_NAMES = ("A", "B")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "description",
        "known_to_inverse",
        "hidden_from_inverse",
        "generator",
        "inference",
        "thresholds",
        "claim_locks",
        "excluded_inferences",
    }
)
_KNOWN_KEYS = frozenset(KNOWN_ITEMS)
_GENERATOR_KEYS = frozenset(
    {
        "seed",
        "state_count",
        "token_count",
        "context_count",
        "candidate_group_count",
        "neurons_per_candidate",
        "samples_per_context",
        "transition_noise",
        "observation_noise",
        "neuron_dropout_fraction",
        "mixing_strength",
        "prototype_scale",
        "stable_monolithic_distractor_count",
    }
)
_INFERENCE_KEYS = frozenset(
    {
        "train_context_count",
        "early_calibration_fraction",
        "kmeans_max_iterations",
    }
)
_THRESHOLD_KEYS = frozenset(
    {
        "selected_candidate_train_accuracy_min",
        "selected_candidate_margin_min",
        "distractor_train_accuracy_max",
        "heldout_late_transition_accuracy_min",
        "late_state_recovery_accuracy_min",
        "alignment_over_permutation_null_gain_min",
    }
)
_CLAIM_LOCK_KEYS = frozenset(
    {
        "real_neural_data_used",
        "full_brain_language_identified",
        "neural_clarus_assembly_validated",
        "causal_instruction_set_validated",
        "fully_blind_inverse_recovery_validated",
    }
)


@dataclass(frozen=True)
class KnownToInverse:
    """Information intentionally supplied to the inverse procedure."""

    external_input_token: bool
    task_operation_cue: bool
    candidate_group_partition: bool
    cross_session_candidate_identity: bool
    latent_cardinality: bool
    operation_label_and_cardinality: bool
    paired_pre_post_transition_boundary: bool
    cross_session_operation_token_semantics: bool
    calibration_split_location: bool
    context_session_identity: bool


@dataclass(frozen=True)
class PartialBlindGeneratorConfig:
    """Deterministic neuron-like synthetic generator settings."""

    seed: int
    state_count: int
    token_count: int
    context_count: int
    candidate_group_count: int
    neurons_per_candidate: int
    samples_per_context: int
    transition_noise: float
    observation_noise: float
    neuron_dropout_fraction: float
    mixing_strength: float
    prototype_scale: float
    stable_monolithic_distractor_count: int


@dataclass(frozen=True)
class PartialBlindInferenceConfig:
    """Manifest-declared split and deterministic clustering settings."""

    train_context_count: int
    early_calibration_fraction: float
    kmeans_max_iterations: int


@dataclass(frozen=True)
class PartialBlindThresholds:
    """Pass/fail thresholds for this synthetic inverse method control."""

    selected_candidate_train_accuracy_min: float
    selected_candidate_margin_min: float
    distractor_train_accuracy_max: float
    heldout_late_transition_accuracy_min: float
    late_state_recovery_accuracy_min: float
    alignment_over_permutation_null_gain_min: float


@dataclass(frozen=True)
class NeuralLanguagePartialBlindBenchmark:
    """Strict benchmark with all discovery claims locked false."""

    schema_version: str
    scope: str
    description: str
    known_to_inverse: KnownToInverse
    hidden_from_inverse: tuple[str, ...]
    generator: PartialBlindGeneratorConfig
    inference: PartialBlindInferenceConfig
    thresholds: PartialBlindThresholds
    real_neural_data_used: bool
    full_brain_language_identified: bool
    neural_clarus_assembly_validated: bool
    causal_instruction_set_validated: bool
    fully_blind_inverse_recovery_validated: bool
    excluded_inferences: tuple[str, ...]


@dataclass(frozen=True)
class CandidateBoundaryScore:
    """Training-session-only score for one supplied candidate group."""

    candidate_group: int
    train_context_late_accuracies: tuple[float, ...]
    mean_train_context_late_accuracy: float
    minimum_train_context_late_accuracy: float


@dataclass(frozen=True)
class InformationBoundaryAudit:
    """Explicit inventory of known, hidden, and scoring-only information."""

    known_to_inverse: tuple[str, ...]
    hidden_from_inverse: tuple[str, ...]
    state_labels_used_for_inference: bool
    generator_target_used_for_selection: bool
    ground_truth_used_only_after_inference_for_scoring: bool


@dataclass(frozen=True)
class ObservationTransformationAudit:
    """Session transforms present in every candidate observation group."""

    context_specific_latent_code_permutation: bool
    context_specific_neuron_permutation: bool
    context_specific_linear_mixing: bool
    context_specific_neuron_dropout: bool
    observation_noise_present: bool
    neuron_dropout_fraction: float
    mixing_strength: float
    observation_noise: float


@dataclass(frozen=True)
class HeldoutContextAudit:
    """Calibration-adapted transfer into a held-out session's late split."""

    heldout_context: int
    candidate_group_evaluated: int
    diagnostic_only_after_abstention: bool
    early_calibration_count: int
    late_evaluation_count: int
    clusters_fit_on_early_calibration_only: bool
    label_alignment_fit_on_early_calibration_only: bool
    late_transition_accuracy_permutation_null_mean: float
    late_transition_accuracy_with_alignment: float
    alignment_over_permutation_null_gain: float
    late_latent_state_recovery_accuracy: float


@dataclass(frozen=True)
class NeuralLanguagePartialBlindReport:
    """Serializable result with a deliberately narrow status label."""

    schema_version: str
    scope: str
    method_status: str
    information_boundary_audit: InformationBoundaryAudit
    observation_transformation_audit: ObservationTransformationAudit
    candidate_scores: tuple[CandidateBoundaryScore, ...]
    top_scoring_candidate_group: int
    selected_candidate_group: int | None
    selection_abstained: bool
    abstention_reason: str | None
    scoring_only_generator_target_group: int
    selected_candidate_matches_generator_target: bool
    top_candidate_train_accuracy: float
    second_best_candidate_train_accuracy: float
    top_candidate_margin: float
    scoring_only_maximum_distractor_train_accuracy: float
    heldout_context_audit: HeldoutContextAudit
    partial_blind_synthetic_pass: bool
    real_neural_data_used: bool
    full_brain_language_identified: bool
    neural_clarus_assembly_validated: bool
    causal_instruction_set_validated: bool
    fully_blind_inverse_recovery_validated: bool
    excluded_inferences: tuple[str, ...]
    limitations: tuple[str, ...]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True)
class _ContextObservations:
    """One observed session; no latent or true-candidate fields exist here."""

    context: int
    operation: np.ndarray
    token: np.ndarray
    candidate_pre: tuple[np.ndarray, ...]
    candidate_post: tuple[np.ndarray, ...]
    early_count: int


@dataclass(frozen=True)
class _ObservedExperiment:
    """The complete input accepted by the inverse procedure."""

    state_count: int
    token_count: int
    candidate_group_count: int
    train_context_count: int
    kmeans_max_iterations: int
    contexts: tuple[_ContextObservations, ...]


@dataclass(frozen=True)
class _SyntheticTruth:
    """Ground truth kept outside the inverse API and used only for scoring."""

    generator_target_group: int
    latent_pre: tuple[np.ndarray, ...]
    latent_post: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _CandidateFit:
    score: CandidateBoundaryScore
    transition_model: np.ndarray


@dataclass(frozen=True)
class _InverseResult:
    candidate_fits: tuple[_CandidateFit, ...]
    top_scoring_candidate_group: int
    selected_candidate_group: int | None
    selection_abstained: bool
    abstention_reason: str | None
    selected_transition_model: np.ndarray
    heldout_pre_labels: np.ndarray
    heldout_post_labels: np.ndarray
    heldout_permutation_null_mean_accuracy: float
    heldout_with_alignment_accuracy: float
    heldout_early_count: int
    heldout_late_count: int


def _require_exact_keys(
    value: Any,
    *,
    required: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    keys = frozenset(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{label} has unknown keys: {unknown}")
    return value


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    return value


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _strict_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _strict_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _strict_string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty JSON array")
    result = tuple(_strict_string(item, label=f"{label} item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def load_neural_language_partial_blind_benchmark(
    path: str | Path,
) -> NeuralLanguagePartialBlindBenchmark:
    """Strictly load the fixed partial-blind synthetic benchmark."""

    benchmark_path = Path(path)
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    top = _require_exact_keys(
        payload,
        required=_TOP_LEVEL_KEYS,
        label="benchmark",
    )

    schema_version = _strict_string(
        top["schema_version"],
        label="schema_version",
    )
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}")
    scope = _strict_string(top["scope"], label="scope")
    if scope != PARTIAL_BLIND_SCOPE:
        raise ValueError(f"scope must equal {PARTIAL_BLIND_SCOPE!r}")

    known_raw = _require_exact_keys(
        top["known_to_inverse"],
        required=_KNOWN_KEYS,
        label="known_to_inverse",
    )
    known_values = {
        key: _strict_bool(known_raw[key], label=f"known_to_inverse.{key}")
        for key in KNOWN_ITEMS
    }
    if not all(known_values.values()):
        raise ValueError("every declared known_to_inverse item must be true")
    known = KnownToInverse(**known_values)

    hidden = _strict_string_tuple(
        top["hidden_from_inverse"],
        label="hidden_from_inverse",
    )
    if hidden != HIDDEN_ITEMS:
        raise ValueError(f"hidden_from_inverse must equal {HIDDEN_ITEMS!r}")

    generator_raw = _require_exact_keys(
        top["generator"],
        required=_GENERATOR_KEYS,
        label="generator",
    )
    generator = PartialBlindGeneratorConfig(
        seed=_strict_int(generator_raw["seed"], label="generator.seed"),
        state_count=_strict_int(
            generator_raw["state_count"],
            label="generator.state_count",
        ),
        token_count=_strict_int(
            generator_raw["token_count"],
            label="generator.token_count",
        ),
        context_count=_strict_int(
            generator_raw["context_count"],
            label="generator.context_count",
        ),
        candidate_group_count=_strict_int(
            generator_raw["candidate_group_count"],
            label="generator.candidate_group_count",
        ),
        neurons_per_candidate=_strict_int(
            generator_raw["neurons_per_candidate"],
            label="generator.neurons_per_candidate",
        ),
        samples_per_context=_strict_int(
            generator_raw["samples_per_context"],
            label="generator.samples_per_context",
        ),
        transition_noise=_strict_float(
            generator_raw["transition_noise"],
            label="generator.transition_noise",
        ),
        observation_noise=_strict_float(
            generator_raw["observation_noise"],
            label="generator.observation_noise",
        ),
        neuron_dropout_fraction=_strict_float(
            generator_raw["neuron_dropout_fraction"],
            label="generator.neuron_dropout_fraction",
        ),
        mixing_strength=_strict_float(
            generator_raw["mixing_strength"],
            label="generator.mixing_strength",
        ),
        prototype_scale=_strict_float(
            generator_raw["prototype_scale"],
            label="generator.prototype_scale",
        ),
        stable_monolithic_distractor_count=_strict_int(
            generator_raw["stable_monolithic_distractor_count"],
            label="generator.stable_monolithic_distractor_count",
        ),
    )

    inference_raw = _require_exact_keys(
        top["inference"],
        required=_INFERENCE_KEYS,
        label="inference",
    )
    inference = PartialBlindInferenceConfig(
        train_context_count=_strict_int(
            inference_raw["train_context_count"],
            label="inference.train_context_count",
        ),
        early_calibration_fraction=_strict_float(
            inference_raw["early_calibration_fraction"],
            label="inference.early_calibration_fraction",
        ),
        kmeans_max_iterations=_strict_int(
            inference_raw["kmeans_max_iterations"],
            label="inference.kmeans_max_iterations",
        ),
    )

    thresholds_raw = _require_exact_keys(
        top["thresholds"],
        required=_THRESHOLD_KEYS,
        label="thresholds",
    )
    thresholds = PartialBlindThresholds(
        selected_candidate_train_accuracy_min=_strict_float(
            thresholds_raw["selected_candidate_train_accuracy_min"],
            label="thresholds.selected_candidate_train_accuracy_min",
        ),
        selected_candidate_margin_min=_strict_float(
            thresholds_raw["selected_candidate_margin_min"],
            label="thresholds.selected_candidate_margin_min",
        ),
        distractor_train_accuracy_max=_strict_float(
            thresholds_raw["distractor_train_accuracy_max"],
            label="thresholds.distractor_train_accuracy_max",
        ),
        heldout_late_transition_accuracy_min=_strict_float(
            thresholds_raw["heldout_late_transition_accuracy_min"],
            label="thresholds.heldout_late_transition_accuracy_min",
        ),
        late_state_recovery_accuracy_min=_strict_float(
            thresholds_raw["late_state_recovery_accuracy_min"],
            label="thresholds.late_state_recovery_accuracy_min",
        ),
        alignment_over_permutation_null_gain_min=_strict_float(
            thresholds_raw[
                "alignment_over_permutation_null_gain_min"
            ],
            label=(
                "thresholds."
                "alignment_over_permutation_null_gain_min"
            ),
        ),
    )

    locks_raw = _require_exact_keys(
        top["claim_locks"],
        required=_CLAIM_LOCK_KEYS,
        label="claim_locks",
    )
    locks = {
        key: _strict_bool(locks_raw[key], label=f"claim_locks.{key}")
        for key in sorted(_CLAIM_LOCK_KEYS)
    }
    for key, value in locks.items():
        if value:
            raise ValueError(f"claim_locks.{key} must be false")

    _validate_configuration(generator, inference, thresholds)
    return NeuralLanguagePartialBlindBenchmark(
        schema_version=schema_version,
        scope=scope,
        description=_strict_string(top["description"], label="description"),
        known_to_inverse=known,
        hidden_from_inverse=hidden,
        generator=generator,
        inference=inference,
        thresholds=thresholds,
        excluded_inferences=_strict_string_tuple(
            top["excluded_inferences"],
            label="excluded_inferences",
        ),
        **locks,
    )


def _validate_configuration(
    generator: PartialBlindGeneratorConfig,
    inference: PartialBlindInferenceConfig,
    thresholds: PartialBlindThresholds,
) -> None:
    if generator.state_count < 3 or generator.state_count > 7:
        raise ValueError("generator.state_count must be between 3 and 7")
    if generator.token_count != generator.state_count:
        raise ValueError("generator.token_count must equal generator.state_count")
    if generator.context_count < 3:
        raise ValueError("generator.context_count must be at least 3")
    if generator.candidate_group_count < 3:
        raise ValueError("generator.candidate_group_count must be at least 3")
    if not (
        0
        <= generator.stable_monolithic_distractor_count
        < generator.candidate_group_count
    ):
        raise ValueError(
            "generator.stable_monolithic_distractor_count must be "
            "non-negative and below candidate_group_count"
        )
    if generator.neurons_per_candidate < 2 * generator.state_count:
        raise ValueError(
            "generator.neurons_per_candidate must be at least twice state_count"
        )
    if generator.samples_per_context < 400:
        raise ValueError("generator.samples_per_context must be at least 400")
    if not 0.0 <= generator.transition_noise < 0.25:
        raise ValueError("generator.transition_noise must be in [0, 0.25)")
    if not 0.0 < generator.observation_noise < generator.prototype_scale:
        raise ValueError(
            "generator.observation_noise must be positive and below "
            "prototype_scale"
        )
    if not 0.0 < generator.neuron_dropout_fraction < 0.5:
        raise ValueError(
            "generator.neuron_dropout_fraction must be in (0, 0.5)"
        )
    if not 0.0 < generator.mixing_strength <= 1.0:
        raise ValueError("generator.mixing_strength must be in (0, 1]")
    if generator.prototype_scale <= 0.0:
        raise ValueError("generator.prototype_scale must be positive")
    if inference.train_context_count != generator.context_count - 1:
        raise ValueError(
            "inference.train_context_count must leave exactly one context out"
        )
    if not 0.25 <= inference.early_calibration_fraction <= 0.5:
        raise ValueError(
            "inference.early_calibration_fraction must be in [0.25, 0.5]"
        )
    if inference.kmeans_max_iterations < 10:
        raise ValueError(
            "inference.kmeans_max_iterations must be at least 10"
        )
    threshold_values = asdict(thresholds)
    for key, value in threshold_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"thresholds.{key} must be in [0, 1]")
    if (
        thresholds.distractor_train_accuracy_max
        >= thresholds.selected_candidate_train_accuracy_min
    ):
        raise ValueError(
            "distractor_train_accuracy_max must be lower than "
            "selected_candidate_train_accuracy_min"
        )


def _true_transition(
    state: np.ndarray,
    operation: np.ndarray,
    token: np.ndarray,
    state_count: int,
) -> np.ndarray:
    operation_a = (state + token + 1) % state_count
    operation_b = (token - state) % state_count
    return np.where(operation == 0, operation_a, operation_b).astype(
        np.int64
    )


def _session_codebook(
    rng: np.random.Generator,
    base_prototypes: np.ndarray,
    config: PartialBlindGeneratorConfig,
) -> np.ndarray:
    state_count, neuron_count = base_prototypes.shape
    latent_code_permutation = rng.permutation(state_count)
    neuron_permutation = rng.permutation(neuron_count)
    mixing = np.eye(neuron_count) + (
        config.mixing_strength
        * rng.normal(size=(neuron_count, neuron_count))
        / np.sqrt(neuron_count)
    )
    dropout_count = max(
        1,
        int(round(config.neuron_dropout_fraction * neuron_count)),
    )
    active = np.ones(neuron_count, dtype=np.float64)
    active[rng.choice(neuron_count, size=dropout_count, replace=False)] = 0.0
    codebook = base_prototypes[latent_code_permutation] @ mixing
    codebook = codebook[:, neuron_permutation] * active
    return codebook


def _observe_labels(
    rng: np.random.Generator,
    labels: np.ndarray,
    codebook: np.ndarray,
    noise: float,
) -> np.ndarray:
    return codebook[labels] + rng.normal(
        scale=noise,
        size=(labels.size, codebook.shape[1]),
    )


def _generate_partial_blind_experiment(
    benchmark: NeuralLanguagePartialBlindBenchmark,
) -> tuple[_ObservedExperiment, _SyntheticTruth]:
    """Generate observed inputs and a strictly separate scoring truth."""

    config = benchmark.generator
    inference = benchmark.inference
    rng = np.random.default_rng(config.seed)
    generator_target_group = int(
        rng.integers(0, config.candidate_group_count)
    )
    base_prototypes = tuple(
        rng.normal(
            scale=config.prototype_scale,
            size=(config.state_count, config.neurons_per_candidate),
        )
        for _ in range(config.candidate_group_count)
    )
    distractor_candidates = tuple(
        candidate
        for candidate in range(config.candidate_group_count)
        if candidate != generator_target_group
    )
    stable_distractor_candidates = frozenset(
        distractor_candidates[
            : config.stable_monolithic_distractor_count
        ]
    )
    stable_distractor_tables = {
        candidate: rng.integers(
            0,
            config.state_count,
            size=(
                config.state_count,
                len(OPERATION_NAMES),
                config.token_count,
            ),
            dtype=np.int64,
        )
        for candidate in stable_distractor_candidates
    }

    contexts: list[_ContextObservations] = []
    truth_pre: list[np.ndarray] = []
    truth_post: list[np.ndarray] = []
    early_count = int(
        round(
            config.samples_per_context
            * inference.early_calibration_fraction
        )
    )
    for context in range(config.context_count):
        operation = rng.integers(
            0,
            len(OPERATION_NAMES),
            size=config.samples_per_context,
            dtype=np.int64,
        )
        token = rng.integers(
            0,
            config.token_count,
            size=config.samples_per_context,
            dtype=np.int64,
        )
        latent_pre = rng.integers(
            0,
            config.state_count,
            size=config.samples_per_context,
            dtype=np.int64,
        )
        latent_post = _true_transition(
            latent_pre,
            operation,
            token,
            config.state_count,
        )
        transition_flip = (
            rng.random(config.samples_per_context)
            < config.transition_noise
        )
        latent_post[transition_flip] = rng.integers(
            0,
            config.state_count,
            size=int(np.count_nonzero(transition_flip)),
        )

        group_pre: list[np.ndarray] = []
        group_post: list[np.ndarray] = []
        for candidate in range(config.candidate_group_count):
            codebook = _session_codebook(
                rng,
                base_prototypes[candidate],
                config,
            )
            if candidate == generator_target_group:
                source_pre = latent_pre
                source_post = latent_post
            else:
                source_pre = rng.integers(
                    0,
                    config.state_count,
                    size=config.samples_per_context,
                    dtype=np.int64,
                )
                distractor_table = stable_distractor_tables.get(candidate)
                if distractor_table is None:
                    distractor_table = rng.integers(
                        0,
                        config.state_count,
                        size=(
                            config.state_count,
                            len(OPERATION_NAMES),
                            config.token_count,
                        ),
                        dtype=np.int64,
                    )
                source_post = distractor_table[
                    source_pre,
                    operation,
                    token,
                ]
                distractor_flip = (
                    rng.random(config.samples_per_context)
                    < config.transition_noise
                )
                source_post[distractor_flip] = rng.integers(
                    0,
                    config.state_count,
                    size=int(np.count_nonzero(distractor_flip)),
                )
            group_pre.append(
                _observe_labels(
                    rng,
                    source_pre,
                    codebook,
                    config.observation_noise,
                )
            )
            group_post.append(
                _observe_labels(
                    rng,
                    source_post,
                    codebook,
                    config.observation_noise,
                )
            )

        contexts.append(
            _ContextObservations(
                context=context,
                operation=operation,
                token=token,
                candidate_pre=tuple(group_pre),
                candidate_post=tuple(group_post),
                early_count=early_count,
            )
        )
        truth_pre.append(latent_pre.copy())
        truth_post.append(latent_post.copy())

    observed = _ObservedExperiment(
        state_count=config.state_count,
        token_count=config.token_count,
        candidate_group_count=config.candidate_group_count,
        train_context_count=inference.train_context_count,
        kmeans_max_iterations=inference.kmeans_max_iterations,
        contexts=tuple(contexts),
    )
    truth = _SyntheticTruth(
        generator_target_group=generator_target_group,
        latent_pre=tuple(truth_pre),
        latent_post=tuple(truth_post),
    )
    return observed, truth


def _squared_distances(
    samples: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    delta = samples[:, None, :] - centers[None, :, :]
    return np.einsum("nkd,nkd->nk", delta, delta)


def _fit_deterministic_kmeans(
    samples: np.ndarray,
    cluster_count: int,
    max_iterations: int,
) -> np.ndarray:
    if samples.ndim != 2 or samples.shape[0] < cluster_count:
        raise ValueError("k-means samples must be a sufficiently tall matrix")
    centered = samples - samples.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.einsum("nd,nd->n", centered, centered)))
    center_indices = [first]
    minimum_distance = _squared_distances(
        samples,
        samples[[first]],
    )[:, 0]
    for _ in range(1, cluster_count):
        next_index = int(np.argmax(minimum_distance))
        center_indices.append(next_index)
        candidate_distance = _squared_distances(
            samples,
            samples[[next_index]],
        )[:, 0]
        minimum_distance = np.minimum(
            minimum_distance,
            candidate_distance,
        )
    centers = samples[np.asarray(center_indices)].copy()

    previous_labels: np.ndarray | None = None
    for _ in range(max_iterations):
        labels = np.argmin(_squared_distances(samples, centers), axis=1)
        if previous_labels is not None and np.array_equal(
            labels,
            previous_labels,
        ):
            break
        previous_labels = labels.copy()
        nearest_distance = np.min(
            _squared_distances(samples, centers),
            axis=1,
        )
        for cluster in range(cluster_count):
            members = samples[labels == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)
            else:
                centers[cluster] = samples[int(np.argmax(nearest_distance))]
    return centers


def _assign_clusters(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(_squared_distances(samples, centers), axis=1).astype(
        np.int64
    )


def _cluster_context_candidate(
    context: _ContextObservations,
    candidate: int,
    state_count: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = context.early_count
    early_samples = np.vstack(
        (
            context.candidate_pre[candidate][:split],
            context.candidate_post[candidate][:split],
        )
    )
    centers = _fit_deterministic_kmeans(
        early_samples,
        state_count,
        max_iterations,
    )
    early_pre = _assign_clusters(
        context.candidate_pre[candidate][:split],
        centers,
    )
    early_post = _assign_clusters(
        context.candidate_post[candidate][:split],
        centers,
    )
    late_pre = _assign_clusters(
        context.candidate_pre[candidate][split:],
        centers,
    )
    late_post = _assign_clusters(
        context.candidate_post[candidate][split:],
        centers,
    )
    return early_pre, early_post, late_pre, late_post


def _fit_transition_model(
    pre_labels: np.ndarray,
    operation: np.ndarray,
    token: np.ndarray,
    post_labels: np.ndarray,
    state_count: int,
    token_count: int,
) -> np.ndarray:
    counts = np.zeros(
        (
            state_count,
            len(OPERATION_NAMES),
            token_count,
            state_count,
        ),
        dtype=np.int64,
    )
    np.add.at(
        counts,
        (pre_labels, operation, token, post_labels),
        1,
    )
    return np.argmax(counts, axis=-1).astype(np.int64)


def _transition_accuracy(
    model: np.ndarray,
    pre_labels: np.ndarray,
    operation: np.ndarray,
    token: np.ndarray,
    post_labels: np.ndarray,
) -> float:
    predicted = model[pre_labels, operation, token]
    return float(np.mean(predicted == post_labels))


def _best_label_permutation(
    model: np.ndarray,
    local_pre: np.ndarray,
    operation: np.ndarray,
    token: np.ndarray,
    local_post: np.ndarray,
    state_count: int,
) -> np.ndarray:
    best_accuracy = -1.0
    best = np.arange(state_count, dtype=np.int64)
    for candidate_tuple in permutations(range(state_count)):
        candidate = np.asarray(candidate_tuple, dtype=np.int64)
        accuracy = _transition_accuracy(
            model,
            candidate[local_pre],
            operation,
            token,
            candidate[local_post],
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best = candidate
    return best


def _permutation_null_mean_accuracy(
    model: np.ndarray,
    local_pre: np.ndarray,
    operation: np.ndarray,
    token: np.ndarray,
    local_post: np.ndarray,
    state_count: int,
) -> float:
    """Average over every arbitrary cluster-label alignment.

    Unlike an identity-permutation baseline, this null is invariant to a
    relabeling of the local clusters.
    """

    accuracies = [
        _transition_accuracy(
            model,
            candidate[local_pre],
            operation,
            token,
            candidate[local_post],
        )
        for candidate_tuple in permutations(range(state_count))
        for candidate in (np.asarray(candidate_tuple, dtype=np.int64),)
    ]
    return float(np.mean(accuracies))


def _fit_candidate_on_train_contexts(
    observed: _ObservedExperiment,
    candidate: int,
) -> _CandidateFit:
    reference = observed.contexts[0]
    (
        reference_early_pre,
        reference_early_post,
        reference_late_pre,
        reference_late_post,
    ) = _cluster_context_candidate(
        reference,
        candidate,
        observed.state_count,
        observed.kmeans_max_iterations,
    )
    split = reference.early_count
    model = _fit_transition_model(
        reference_early_pre,
        reference.operation[:split],
        reference.token[:split],
        reference_early_post,
        observed.state_count,
        observed.token_count,
    )
    late_accuracies = [
        _transition_accuracy(
            model,
            reference_late_pre,
            reference.operation[split:],
            reference.token[split:],
            reference_late_post,
        )
    ]
    for context in observed.contexts[1 : observed.train_context_count]:
        early_pre, early_post, late_pre, late_post = (
            _cluster_context_candidate(
                context,
                candidate,
                observed.state_count,
                observed.kmeans_max_iterations,
            )
        )
        split = context.early_count
        alignment = _best_label_permutation(
            model,
            early_pre,
            context.operation[:split],
            context.token[:split],
            early_post,
            observed.state_count,
        )
        late_accuracies.append(
            _transition_accuracy(
                model,
                alignment[late_pre],
                context.operation[split:],
                context.token[split:],
                alignment[late_post],
            )
        )
    score = CandidateBoundaryScore(
        candidate_group=candidate,
        train_context_late_accuracies=tuple(
            float(value) for value in late_accuracies
        ),
        mean_train_context_late_accuracy=float(np.mean(late_accuracies)),
        minimum_train_context_late_accuracy=float(np.min(late_accuracies)),
    )
    return _CandidateFit(score=score, transition_model=model)


def _run_partial_blind_inverse(
    observed: _ObservedExperiment,
    *,
    candidate_accuracy_min: float,
    candidate_margin_min: float,
) -> _InverseResult:
    """Infer a boundary without accepting any synthetic truth argument."""

    candidate_fits = tuple(
        _fit_candidate_on_train_contexts(observed, candidate)
        for candidate in range(observed.candidate_group_count)
    )
    selected_fit = max(
        candidate_fits,
        key=lambda item: (
            item.score.mean_train_context_late_accuracy,
            -item.score.candidate_group,
        ),
    )
    ranked_scores = sorted(
        (
            fit.score.mean_train_context_late_accuracy
            for fit in candidate_fits
        ),
        reverse=True,
    )
    top_score = ranked_scores[0]
    selection_margin = top_score - ranked_scores[1]
    if top_score < candidate_accuracy_min:
        selected_candidate_group = None
        abstention_reason = "no_candidate_meets_accuracy_threshold"
    elif selection_margin < candidate_margin_min:
        selected_candidate_group = None
        abstention_reason = "candidate_margin_below_threshold"
    else:
        selected_candidate_group = selected_fit.score.candidate_group
        abstention_reason = None
    heldout = observed.contexts[observed.train_context_count]
    early_pre, early_post, late_pre, late_post = (
        _cluster_context_candidate(
            heldout,
            selected_fit.score.candidate_group,
            observed.state_count,
            observed.kmeans_max_iterations,
        )
    )
    split = heldout.early_count
    alignment = _best_label_permutation(
        selected_fit.transition_model,
        early_pre,
        heldout.operation[:split],
        heldout.token[:split],
        early_post,
        observed.state_count,
    )
    permutation_null_mean = _permutation_null_mean_accuracy(
        selected_fit.transition_model,
        late_pre,
        heldout.operation[split:],
        heldout.token[split:],
        late_post,
        observed.state_count,
    )
    aligned_pre = alignment[late_pre]
    aligned_post = alignment[late_post]
    with_alignment = _transition_accuracy(
        selected_fit.transition_model,
        aligned_pre,
        heldout.operation[split:],
        heldout.token[split:],
        aligned_post,
    )
    return _InverseResult(
        candidate_fits=candidate_fits,
        top_scoring_candidate_group=selected_fit.score.candidate_group,
        selected_candidate_group=selected_candidate_group,
        selection_abstained=selected_candidate_group is None,
        abstention_reason=abstention_reason,
        selected_transition_model=selected_fit.transition_model,
        heldout_pre_labels=aligned_pre,
        heldout_post_labels=aligned_post,
        heldout_permutation_null_mean_accuracy=permutation_null_mean,
        heldout_with_alignment_accuracy=with_alignment,
        heldout_early_count=split,
        heldout_late_count=late_pre.size,
    )


def _scoring_only_state_recovery_accuracy(
    inferred_pre: np.ndarray,
    inferred_post: np.ndarray,
    truth_pre: np.ndarray,
    truth_post: np.ndarray,
    state_count: int,
) -> float:
    """Use latent truth only here, after all inverse outputs are frozen."""

    inferred = np.concatenate((inferred_pre, inferred_post))
    truth = np.concatenate((truth_pre, truth_post))
    best = 0.0
    for candidate_tuple in permutations(range(state_count)):
        candidate = np.asarray(candidate_tuple, dtype=np.int64)
        best = max(best, float(np.mean(candidate[inferred] == truth)))
    return best


def neural_language_partial_blind_gate_report(
    benchmark: NeuralLanguagePartialBlindBenchmark,
) -> NeuralLanguagePartialBlindReport:
    """Run the partial-blind synthetic inverse and scoring-only audit."""

    if not isinstance(benchmark, NeuralLanguagePartialBlindBenchmark):
        raise TypeError(
            "benchmark must be NeuralLanguagePartialBlindBenchmark"
        )
    lock_values = (
        benchmark.real_neural_data_used,
        benchmark.full_brain_language_identified,
        benchmark.neural_clarus_assembly_validated,
        benchmark.causal_instruction_set_validated,
        benchmark.fully_blind_inverse_recovery_validated,
    )
    if any(lock_values):
        raise ValueError("all discovery claim locks must remain false")

    observed, truth = _generate_partial_blind_experiment(benchmark)
    inverse = _run_partial_blind_inverse(
        observed,
        candidate_accuracy_min=(
            benchmark.thresholds.selected_candidate_train_accuracy_min
        ),
        candidate_margin_min=(
            benchmark.thresholds.selected_candidate_margin_min
        ),
    )
    scores = tuple(fit.score for fit in inverse.candidate_fits)
    ranked = sorted(
        (
            score.mean_train_context_late_accuracy
            for score in scores
        ),
        reverse=True,
    )
    selected_score = ranked[0]
    second_score = ranked[1]
    selected_margin = selected_score - second_score
    distractor_scores = [
        score.mean_train_context_late_accuracy
        for score in scores
        if score.candidate_group != truth.generator_target_group
    ]
    maximum_distractor = float(max(distractor_scores))

    heldout_index = benchmark.inference.train_context_count
    heldout_truth_pre = truth.latent_pre[heldout_index][
        inverse.heldout_early_count :
    ]
    heldout_truth_post = truth.latent_post[heldout_index][
        inverse.heldout_early_count :
    ]
    recovery_accuracy = _scoring_only_state_recovery_accuracy(
        inverse.heldout_pre_labels,
        inverse.heldout_post_labels,
        heldout_truth_pre,
        heldout_truth_post,
        benchmark.generator.state_count,
    )
    alignment_over_null_gain = (
        inverse.heldout_with_alignment_accuracy
        - inverse.heldout_permutation_null_mean_accuracy
    )
    selected_is_true = (
        not inverse.selection_abstained
        and inverse.selected_candidate_group == truth.generator_target_group
    )
    threshold = benchmark.thresholds
    method_pass = bool(
        selected_is_true
        and selected_score
        >= threshold.selected_candidate_train_accuracy_min
        and selected_margin >= threshold.selected_candidate_margin_min
        and maximum_distractor
        <= threshold.distractor_train_accuracy_max
        and inverse.heldout_with_alignment_accuracy
        >= threshold.heldout_late_transition_accuracy_min
        and recovery_accuracy
        >= threshold.late_state_recovery_accuracy_min
        and alignment_over_null_gain
        >= threshold.alignment_over_permutation_null_gain_min
    )

    heldout_audit = HeldoutContextAudit(
        heldout_context=heldout_index,
        candidate_group_evaluated=inverse.top_scoring_candidate_group,
        diagnostic_only_after_abstention=inverse.selection_abstained,
        early_calibration_count=inverse.heldout_early_count,
        late_evaluation_count=inverse.heldout_late_count,
        clusters_fit_on_early_calibration_only=True,
        label_alignment_fit_on_early_calibration_only=True,
        late_transition_accuracy_permutation_null_mean=(
            inverse.heldout_permutation_null_mean_accuracy
        ),
        late_transition_accuracy_with_alignment=(
            inverse.heldout_with_alignment_accuracy
        ),
        alignment_over_permutation_null_gain=alignment_over_null_gain,
        late_latent_state_recovery_accuracy=recovery_accuracy,
    )
    if method_pass:
        status = PARTIAL_BLIND_SYNTHETIC_PASS
        conclusion = (
            "A calibration-adapted partial-blind synthetic inverse selected "
            "the generator-target group from supplied candidate boundaries "
            "and "
            "aligned arbitrary latent labels after session drift. This is "
            "only a pipeline control because token, operation, candidate "
            "partition, session identity, and latent cardinality were "
            "supplied."
        )
    elif inverse.abstention_reason == "candidate_margin_below_threshold":
        status = PARTIAL_BLIND_SYNTHETIC_AMBIGUOUS
        conclusion = (
            "The inverse abstained because multiple supplied candidate "
            "groups were too close under the manifest-declared score. In "
            "the stable-monolithic negative control they are "
            "observationally indistinguishable, so behavioral output, "
            "interface, or causal evidence is required."
        )
    else:
        status = PARTIAL_BLIND_SYNTHETIC_FAIL
        conclusion = (
            "The partial-blind synthetic inverse did not satisfy its "
            "manifest-declared method-control thresholds. All biological "
            "and fully blind claims remain locked false."
        )
    return NeuralLanguagePartialBlindReport(
        schema_version=REPORT_SCHEMA_VERSION,
        scope=PARTIAL_BLIND_SCOPE,
        method_status=status,
        information_boundary_audit=InformationBoundaryAudit(
            known_to_inverse=KNOWN_ITEMS,
            hidden_from_inverse=HIDDEN_ITEMS,
            state_labels_used_for_inference=False,
            generator_target_used_for_selection=False,
            ground_truth_used_only_after_inference_for_scoring=True,
        ),
        observation_transformation_audit=ObservationTransformationAudit(
            context_specific_latent_code_permutation=True,
            context_specific_neuron_permutation=True,
            context_specific_linear_mixing=True,
            context_specific_neuron_dropout=True,
            observation_noise_present=True,
            neuron_dropout_fraction=(
                benchmark.generator.neuron_dropout_fraction
            ),
            mixing_strength=benchmark.generator.mixing_strength,
            observation_noise=benchmark.generator.observation_noise,
        ),
        candidate_scores=scores,
        top_scoring_candidate_group=inverse.top_scoring_candidate_group,
        selected_candidate_group=inverse.selected_candidate_group,
        selection_abstained=inverse.selection_abstained,
        abstention_reason=inverse.abstention_reason,
        scoring_only_generator_target_group=truth.generator_target_group,
        selected_candidate_matches_generator_target=selected_is_true,
        top_candidate_train_accuracy=selected_score,
        second_best_candidate_train_accuracy=second_score,
        top_candidate_margin=selected_margin,
        scoring_only_maximum_distractor_train_accuracy=maximum_distractor,
        heldout_context_audit=heldout_audit,
        partial_blind_synthetic_pass=method_pass,
        real_neural_data_used=False,
        full_brain_language_identified=False,
        neural_clarus_assembly_validated=False,
        causal_instruction_set_validated=False,
        fully_blind_inverse_recovery_validated=False,
        excluded_inferences=benchmark.excluded_inferences,
        limitations=(
            "external input tokens are supplied",
            "task/operation cues are supplied",
            "candidate neuron-group partitions are supplied",
            "latent-state cardinality is supplied",
            "session identities and the calibration boundary are supplied",
            "the generator is finite-state, stationary within each session, "
            "and easier than biological neural dynamics",
            "the reference pass uses session-specific distractors; an "
            "equally stable monolithic transition is intentionally "
            "indistinguishable and causes the method gate to fail",
            "synthetic truth is used only for post-inference scoring",
        ),
        conclusion=conclusion,
    )
