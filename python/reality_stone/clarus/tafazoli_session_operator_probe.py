"""Label-blind, session-local stationary operator probe for Tafazoli data.

The released ``PFC_ClassifierData.mat`` contains one saved pseudo-trial
snapshot, not a raw-trial archive.  Its 403 neuron columns stitch together 27
recording sessions.  This module therefore:

* reads only the *training* tensors for dimensions 1 and 3;
* slices the tensors into recovered simultaneous-population sessions before
  fitting anything;
* holds out complete pseudo-trial rows;
* fits every transform on the training fold only; and
* treats reverse-time and deranged-successor fits as descriptive controls, not
  causal permutation tests.

The probe can detect short-lived population predictability.  It cannot identify
an opcode, a causal instruction set, a shared call graph, or a brain
programming language.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "clarus-tafazoli-session-operator-probe/v1"
PROBE_SCOPE = "label_blind_session_local_stationary_linear_operator_probe"
OFFICIAL_CLASSIFIER_MD5 = "04e42460ccae245ec16fcd9801ddea4a"

YES = "YES"
NO = "NO"
PENDING = "PENDING"
TEST_UNAVAILABLE = "TEST_UNAVAILABLE"

SESSION_NEURON_COUNTS = (
    21,
    17,
    14,
    8,
    12,
    7,
    9,
    7,
    2,
    21,
    30,
    27,
    19,
    29,
    23,
    25,
    25,
    15,
    9,
    9,
    15,
    6,
    14,
    8,
    6,
    12,
    13,
)
SESSION_ANIMALS = ("Chico",) * 9 + ("Silas",) * 18
ALLOWED_DIMENSIONS = (1, 3)
RANK_STABILITY_CAPS = (1, 2, 3, 5)

VERDICT_SESSION_LOCAL_SHORT_MEMORY = "session_local_short_memory"
VERDICT_SHARED_STATIONARY_DIRECTED_OPERATOR = (
    "shared_stationary_directed_operator"
)
VERDICT_STATE_DEPENDENT_SWITCHING_OPERATOR = (
    "state_dependent_switching_operator"
)
VERDICT_SHARED_CALL_GRAPH_HIERARCHICAL_OPERATOR = (
    "shared_call_graph_or_hierarchical_operator"
)
VERDICT_BRAIN_PROGRAMMING_LANGUAGE = "brain_programming_language_identified"

NEXT_TEST_SWITCHING_K2_K3 = "label_free_switching_operator_k2_k3"
NEXT_TEST_EVENT_TIME_BOTTLENECK = "event_time_convergence_bottleneck"
NEXT_TEST_PARENT_PLUS_RESIDUAL = "parent_operator_plus_session_task_residual"
NEXT_TEST_COMMON_SUCCESSOR = "common_successor_state"


@dataclass(frozen=True)
class ProbeConfig:
    """Deterministic protocol for the stationary operator probe."""

    seed: int = 20260730
    fold_count: int = 6
    lag_bins: int = 10
    transition_stride_bins: int = 1
    rank_cap: int = 5
    successor_shuffle_count: int = 100
    session_neuron_counts: tuple[int, ...] = SESSION_NEURON_COUNTS
    session_animals: tuple[str, ...] = SESSION_ANIMALS
    rank_stability_caps: tuple[int, ...] = RANK_STABILITY_CAPS
    run_event_mean_removed_sensitivity: bool = True
    minimum_source_grand_mean_r2: float = 0.05
    minimum_direction_specificity: float = 0.02
    minimum_frozen_vs_target_refit_skill: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name, value in (
            ("fold_count", self.fold_count),
            ("lag_bins", self.lag_bins),
            ("transition_stride_bins", self.transition_stride_bins),
            ("rank_cap", self.rank_cap),
            ("successor_shuffle_count", self.successor_shuffle_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.fold_count < 2:
            raise ValueError("fold_count must be at least two")
        if self.lag_bins < 1:
            raise ValueError("lag_bins must be positive")
        if self.transition_stride_bins < 1:
            raise ValueError("transition_stride_bins must be positive")
        if self.rank_cap < 1:
            raise ValueError("rank_cap must be positive")
        if self.successor_shuffle_count < 1:
            raise ValueError("successor_shuffle_count must be positive")
        if not self.session_neuron_counts:
            raise ValueError("at least one session is required")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in self.session_neuron_counts
        ):
            raise ValueError("session neuron counts must be positive integers")
        if len(self.session_neuron_counts) != len(self.session_animals):
            raise ValueError(
                "session_neuron_counts and session_animals must align"
            )
        if any(not animal for animal in self.session_animals):
            raise ValueError("session animal labels must be non-empty")
        if not self.rank_stability_caps:
            raise ValueError("rank_stability_caps must not be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in self.rank_stability_caps
        ):
            raise ValueError("rank stability caps must be positive integers")
        for name, value in (
            (
                "minimum_source_grand_mean_r2",
                self.minimum_source_grand_mean_r2,
            ),
            (
                "minimum_direction_specificity",
                self.minimum_direction_specificity,
            ),
            (
                "minimum_frozen_vs_target_refit_skill",
                self.minimum_frozen_vs_target_refit_skill,
            ),
        ):
            if isinstance(value, bool) or not np.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite real number")


@dataclass(frozen=True)
class SessionSpec:
    """One recovered simultaneous-recording column range."""

    index_one_based: int
    animal: str
    column_start_zero_based: int
    column_stop_exclusive: int

    @property
    def neuron_count(self) -> int:
        return self.column_stop_exclusive - self.column_start_zero_based


@dataclass(frozen=True)
class WholeTrialFold:
    """One outer fold; a trial row appears wholly in train or test."""

    index_zero_based: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class LatentTransform:
    """Training-only Anscombe, standardization, event mean, and PCA state."""

    neuron_mean: np.ndarray
    neuron_scale: np.ndarray
    active_neuron_mask: np.ndarray
    event_time_mean: np.ndarray
    components: np.ndarray
    event_mean_removed: bool

    @property
    def rank(self) -> int:
        return int(self.components.shape[1])

    @property
    def active_neuron_count(self) -> int:
        return int(np.count_nonzero(self.active_neuron_mask))

    def project(self, population: np.ndarray) -> np.ndarray:
        """Project ``trial x neuron x time`` counts without refitting."""

        values = _validate_population_tensor(
            population,
            label="population projected by a frozen transform",
        )
        if values.shape[1] != self.active_neuron_mask.size:
            raise ValueError("frozen transform neuron count does not match")
        if values.shape[2] != self.event_time_mean.shape[0]:
            raise ValueError("frozen transform time count does not match")
        transformed = _anscombe_trial_time_neuron(values)
        active = transformed[:, :, self.active_neuron_mask]
        active = (
            active - self.neuron_mean[self.active_neuron_mask]
        ) / self.neuron_scale[self.active_neuron_mask]
        if self.event_mean_removed:
            active = active - self.event_time_mean[None, :, :]
        flat = active.reshape(-1, active.shape[-1])
        latent = flat @ self.components
        return latent.reshape(
            values.shape[0],
            values.shape[2],
            self.rank,
        )


@dataclass(frozen=True)
class PreparedLatentFold:
    """Reusable train/test latent tensors for stationary or switching probes."""

    fold: WholeTrialFold
    transform: LatentTransform
    source_train: np.ndarray
    source_test: np.ndarray
    target_train_in_source_coordinates: np.ndarray
    target_test_in_source_coordinates: np.ndarray


@dataclass(frozen=True)
class PredictionMetrics:
    """Held-out trajectory errors and dimensionless baseline comparisons."""

    transition_count: int
    model_sse: float
    source_grand_mean_sse: float
    persistence_sse: float
    time_locked_mean_sse: float
    reverse_control_sse: float
    source_grand_mean_r2: float
    persistence_skill: float
    time_locked_mean_skill: float
    direction_specificity: float
    successor_shuffle_advantage: float
    successor_shuffle_fraction_worse: float


@dataclass(frozen=True)
class SessionOperatorResult:
    """Within-dimension result for one simultaneous population."""

    analysis_key: str
    session_index_one_based: int
    animal: str
    neuron_count: int
    dimension: int
    rank_cap: int
    effective_rank: int
    minimum_active_neurons: int
    event_mean_removed: bool
    forward_spectral_radius_median: float
    reverse_spectral_radius_median: float
    metrics: PredictionMetrics


@dataclass(frozen=True)
class SessionTransferResult:
    """Frozen source-to-target operator transfer within one session."""

    analysis_key: str
    session_index_one_based: int
    animal: str
    neuron_count: int
    source_dimension: int
    target_dimension: int
    rank_cap: int
    effective_rank: int
    minimum_active_neurons: int
    event_mean_removed: bool
    representation_fit_on_source_only: bool
    metrics: PredictionMetrics
    target_refit_source_grand_mean_r2: float
    frozen_vs_target_refit_skill: float
    source_target_operator_frobenius_median: float


@dataclass(frozen=True)
class AggregateResult:
    """Session-weighted aggregate; neurons and time bins are not replicates."""

    analysis_key: str
    animal: str
    session_count: int
    event_mean_removed: bool
    median_source_grand_mean_r2: float
    median_persistence_skill: float
    median_time_locked_mean_skill: float
    median_direction_specificity: float
    median_successor_shuffle_advantage: float
    median_frozen_vs_target_refit_skill: float | None


@dataclass(frozen=True)
class RankStabilityResult:
    """Median result at one predeclared latent-rank cap."""

    analysis_key: str
    rank_cap: int
    session_count: int
    median_source_grand_mean_r2: float
    median_persistence_skill: float
    median_direction_specificity: float
    median_frozen_vs_target_refit_skill: float | None


@dataclass(frozen=True)
class ClaimVerdict:
    """Stable claim-local answer for downstream gates."""

    key: str
    answer: str
    reason: str


@dataclass(frozen=True)
class ProbeClaimLocks:
    """Claims that this stationary, observational snapshot cannot unlock."""

    labels_or_responses_used: bool = False
    all_factors_used: bool = False
    saved_classifier_test_set_used: bool = False
    dimension_two_used: bool = False
    full_pseudopopulation_fit: bool = False
    heldout_session_weight_transfer_completed: bool = False
    unseen_composition_tested: bool = False
    causal_instruction_set_identified: bool = False
    fixed_neuron_opcode_identified: bool = False
    shared_call_graph_refuted: bool = False
    hierarchical_operator_refuted: bool = False
    brain_programming_language_identified: bool = False


@dataclass(frozen=True)
class NextTest:
    """A structured next gate that the stationary probe does not answer."""

    key: str
    status: str
    question: str
    required_control: str


@dataclass(frozen=True)
class TafazoliSessionOperatorProbeReport:
    """Serializable stationary probe report with explicit claim boundaries."""

    schema_version: str
    scope: str
    method_status: str
    source_file_md5: str | None
    official_checksum_verified: bool
    config: ProbeConfig
    session_specs: tuple[SessionSpec, ...]
    fields_used_for_fitting: tuple[str, ...]
    blind_fields_used: tuple[str, ...]
    train_only_preprocessing: bool
    saved_test_role: str
    within_results: tuple[SessionOperatorResult, ...]
    transfer_results: tuple[SessionTransferResult, ...]
    event_mean_removed_within_results: tuple[SessionOperatorResult, ...]
    event_mean_removed_transfer_results: tuple[SessionTransferResult, ...]
    aggregates: tuple[AggregateResult, ...]
    rank_stability: tuple[RankStabilityResult, ...]
    verdicts: tuple[ClaimVerdict, ...]
    claim_locks: ProbeClaimLocks
    next_tests: tuple[NextTest, ...]
    limitations: tuple[str, ...]
    conclusion: str

    def verdict(self, key: str) -> ClaimVerdict:
        """Return one claim-local verdict by its stable key."""

        matches = tuple(item for item in self.verdicts if item.key == key)
        if len(matches) != 1:
            raise KeyError(key)
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return asdict(self)


def recovered_session_specs(
    session_neuron_counts: Sequence[int] = SESSION_NEURON_COUNTS,
    session_animals: Sequence[str] = SESSION_ANIMALS,
) -> tuple[SessionSpec, ...]:
    """Return the recovered, non-overlapping recording-session ranges."""

    counts = tuple(int(value) for value in session_neuron_counts)
    animals = tuple(str(value) for value in session_animals)
    if len(counts) != len(animals):
        raise ValueError("session counts and animals must align")
    if not counts or any(value < 1 for value in counts):
        raise ValueError("session counts must be positive")
    start = 0
    specs = []
    for index, (count, animal) in enumerate(zip(counts, animals), start=1):
        specs.append(
            SessionSpec(
                index_one_based=index,
                animal=animal,
                column_start_zero_based=start,
                column_stop_exclusive=start + count,
            )
        )
        start += count
    return tuple(specs)


def make_whole_trial_folds(
    trial_count: int,
    *,
    fold_count: int = 6,
    seed: int = 20260730,
) -> tuple[WholeTrialFold, ...]:
    """Create deterministic folds without ever splitting a trial trajectory."""

    if isinstance(trial_count, bool) or not isinstance(trial_count, int):
        raise TypeError("trial_count must be an integer")
    if trial_count < fold_count or fold_count < 2:
        raise ValueError("fold_count must be in [2, trial_count]")
    permutation = np.random.default_rng(seed).permutation(trial_count)
    all_indices = np.arange(trial_count, dtype=np.int64)
    folds = []
    for index, test in enumerate(np.array_split(permutation, fold_count)):
        test = np.sort(test)
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        folds.append(
            WholeTrialFold(
                index_zero_based=index,
                train_indices=tuple(int(value) for value in train),
                test_indices=tuple(int(value) for value in test),
            )
        )
    return tuple(folds)


def transition_start_indices(
    timepoint_count: int,
    *,
    lag_bins: int,
    stride_bins: int = 1,
) -> np.ndarray:
    """Return within-trial transition starts; no trial boundary is crossed."""

    if timepoint_count <= lag_bins:
        raise ValueError("timepoint_count must exceed lag_bins")
    if lag_bins < 1 or stride_bins < 1:
        raise ValueError("lag_bins and stride_bins must be positive")
    return np.arange(
        0,
        timepoint_count - lag_bins,
        stride_bins,
        dtype=np.int64,
    )


def fit_session_latent_transform(
    training_population: np.ndarray,
    *,
    rank_cap: int,
    event_mean_removed: bool,
) -> tuple[LatentTransform, np.ndarray]:
    """Fit the label-free representation on training trials only."""

    values = _validate_population_tensor(
        training_population,
        label="training_population",
    )
    transformed = _anscombe_trial_time_neuron(values)
    flat = transformed.reshape(-1, transformed.shape[-1])
    neuron_mean = flat.mean(axis=0)
    neuron_scale = flat.std(axis=0)
    active = neuron_scale > 1e-10
    if not np.any(active):
        raise ValueError("session has no active neurons after Anscombe transform")
    standardized = (
        transformed[:, :, active] - neuron_mean[active]
    ) / neuron_scale[active]
    if event_mean_removed:
        event_time_mean = standardized.mean(axis=0)
        standardized = standardized - event_time_mean[None, :, :]
    else:
        event_time_mean = np.zeros(
            (values.shape[2], int(np.count_nonzero(active))),
            dtype=np.float64,
        )
    flat_standardized = standardized.reshape(-1, standardized.shape[-1])
    _, _, right = np.linalg.svd(flat_standardized, full_matrices=False)
    rank = min(int(rank_cap), right.shape[0])
    if rank < 1:
        raise ValueError("latent rank is zero")
    components = right[:rank].T
    transform = LatentTransform(
        neuron_mean=np.asarray(neuron_mean, dtype=np.float64),
        neuron_scale=np.asarray(neuron_scale, dtype=np.float64),
        active_neuron_mask=np.asarray(active, dtype=bool),
        event_time_mean=np.asarray(event_time_mean, dtype=np.float64),
        components=np.asarray(components, dtype=np.float64),
        event_mean_removed=bool(event_mean_removed),
    )
    return transform, transform.project(values)


def prepare_session_latent_fold(
    source_population: np.ndarray,
    target_population: np.ndarray,
    fold: WholeTrialFold,
    *,
    rank_cap: int,
    event_mean_removed: bool,
) -> PreparedLatentFold:
    """Prepare one reusable source-frozen latent fold."""

    source = _validate_population_tensor(
        source_population,
        label="source_population",
    )
    target = _validate_population_tensor(
        target_population,
        label="target_population",
    )
    if source.shape != target.shape:
        raise ValueError("source and target session tensors must have one shape")
    train = np.asarray(fold.train_indices, dtype=np.int64)
    test = np.asarray(fold.test_indices, dtype=np.int64)
    if np.intersect1d(train, test).size:
        raise ValueError("whole-trial train and test indices overlap")
    if np.union1d(train, test).size != source.shape[0]:
        raise ValueError("fold does not cover every trial exactly once")
    transform, source_train = fit_session_latent_transform(
        source[train],
        rank_cap=rank_cap,
        event_mean_removed=event_mean_removed,
    )
    return PreparedLatentFold(
        fold=fold,
        transform=transform,
        source_train=source_train,
        source_test=transform.project(source[test]),
        target_train_in_source_coordinates=transform.project(target[train]),
        target_test_in_source_coordinates=transform.project(target[test]),
    )


def extract_tafazoli_train_dimensions(
    classifier_options: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract only dimension-1/3 train tensors from ``ClassifierOpts``.

    No response label, AllFactors field, saved test tensor, or dimension 2 is
    accessed.  This narrow function also permits a spy mapping in tests.
    """

    predictors = classifier_options["Dimpredictors"]
    dimension_one = np.asarray(predictors[0][0], dtype=np.float64)
    dimension_three = np.asarray(predictors[2][0], dtype=np.float64)
    return dimension_one, dimension_three


def load_tafazoli_train_dimensions(
    classifier_file: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the two allowed train tensors; SciPy remains an optional import."""

    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required to load the Tafazoli MATLAB snapshot"
        ) from error
    payload = loadmat(Path(classifier_file), simplify_cells=True)
    options = payload["ClassifierOpts"]
    if not isinstance(options, Mapping):
        raise TypeError("ClassifierOpts must decode to a mapping")
    return extract_tafazoli_train_dimensions(options)


def verify_official_classifier_checksum(
    classifier_file: str | Path,
) -> str:
    """Verify the exact official snapshot before applying fixed column slices."""

    path = Path(classifier_file)
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != OFFICIAL_CLASSIFIER_MD5:
        raise ValueError(
            "official classifier checksum mismatch: "
            f"expected {OFFICIAL_CLASSIFIER_MD5}, observed {observed}"
        )
    return observed


def run_tafazoli_session_operator_probe_from_arrays(
    dimension_one_train: np.ndarray,
    dimension_three_train: np.ndarray,
    *,
    config: ProbeConfig = ProbeConfig(),
) -> TafazoliSessionOperatorProbeReport:
    """Run the complete label-blind stationary probe on allowed tensors."""

    dim1 = _validate_population_tensor(
        dimension_one_train,
        label="dimension_one_train",
    )
    dim3 = _validate_population_tensor(
        dimension_three_train,
        label="dimension_three_train",
    )
    if dim1.shape != dim3.shape:
        raise ValueError("dimension 1 and 3 train tensors must have one shape")
    if dim1.shape[0] < config.fold_count:
        raise ValueError("not enough trials for the requested whole-trial CV")
    if dim1.shape[2] <= config.lag_bins:
        raise ValueError("not enough timepoints for the requested lag")
    specs = recovered_session_specs(
        config.session_neuron_counts,
        config.session_animals,
    )
    if sum(spec.neuron_count for spec in specs) != dim1.shape[1]:
        raise ValueError(
            "recovered session column counts do not match the tensor"
        )
    folds = make_whole_trial_folds(
        dim1.shape[0],
        fold_count=config.fold_count,
        seed=config.seed,
    )
    arrays = {1: dim1, 3: dim3}

    within = _run_within_grid(
        arrays,
        specs,
        folds,
        config=config,
        rank_cap=config.rank_cap,
        event_mean_removed=False,
        include_shuffles=True,
    )
    transfer = _run_transfer_grid(
        arrays,
        specs,
        folds,
        config=config,
        rank_cap=config.rank_cap,
        event_mean_removed=False,
        include_shuffles=True,
    )
    if config.run_event_mean_removed_sensitivity:
        residual_within = _run_within_grid(
            arrays,
            specs,
            folds,
            config=config,
            rank_cap=config.rank_cap,
            event_mean_removed=True,
            include_shuffles=True,
        )
        residual_transfer = _run_transfer_grid(
            arrays,
            specs,
            folds,
            config=config,
            rank_cap=config.rank_cap,
            event_mean_removed=True,
            include_shuffles=True,
        )
    else:
        residual_within = ()
        residual_transfer = ()

    rank_stability = _run_rank_stability(
        arrays,
        specs,
        folds,
        config=config,
        primary_within=within,
        primary_transfer=transfer,
    )
    all_results: tuple[
        SessionOperatorResult | SessionTransferResult, ...
    ] = (
        tuple(within)
        + tuple(transfer)
        + tuple(residual_within)
        + tuple(residual_transfer)
    )
    aggregates = _aggregate_results(all_results)
    verdicts = _build_verdicts(within, transfer, config=config)
    locks = ProbeClaimLocks()
    validate_claim_locks(locks)
    report = TafazoliSessionOperatorProbeReport(
        schema_version=SCHEMA_VERSION,
        scope=PROBE_SCOPE,
        method_status="STATIONARY_SESSION_OPERATOR_PROBE_COMPLETE",
        source_file_md5=None,
        official_checksum_verified=False,
        config=config,
        session_specs=specs,
        fields_used_for_fitting=(
            "ClassifierOpts.Dimpredictors[dimension=1].train",
            "ClassifierOpts.Dimpredictors[dimension=3].train",
        ),
        blind_fields_used=(),
        train_only_preprocessing=True,
        saved_test_role="not_used",
        within_results=tuple(within),
        transfer_results=tuple(transfer),
        event_mean_removed_within_results=tuple(residual_within),
        event_mean_removed_transfer_results=tuple(residual_transfer),
        aggregates=aggregates,
        rank_stability=rank_stability,
        verdicts=verdicts,
        claim_locks=locks,
        next_tests=_next_tests(),
        limitations=(
            "The file is one saved fold-1 pseudo-trial snapshot, not raw trials.",
            "The 27 sessions, not neurons or time bins, are inference units.",
            "Stride 1 reuses adjacent event-aligned start bins as fitting weights; "
            "it does not make those overlapping transitions inference units.",
            "Two animals do not support broad population-level generalization.",
            "The stationary operator can average over unobserved switching states.",
            "Reverse-time similarity is a direction-specificity control, not causality.",
            "Deranged successors are a descriptive model control, not a formal permutation p-value.",
            "No unseen composition, intervention, call graph, or grammar is present.",
        ),
        conclusion=(
            "Short session-local memory is detectable, but a shared stationary "
            "directed operator is not supported; switching and hierarchical "
            "operator hypotheses remain pending."
        ),
    )
    validate_probe_report(report)
    return report


def run_tafazoli_session_operator_probe(
    classifier_file: str | Path,
    *,
    config: ProbeConfig = ProbeConfig(),
) -> TafazoliSessionOperatorProbeReport:
    """Load the official snapshot and run the strict stationary probe."""

    observed_md5 = verify_official_classifier_checksum(classifier_file)
    dim1, dim3 = load_tafazoli_train_dimensions(classifier_file)
    if config.session_neuron_counts == SESSION_NEURON_COUNTS:
        expected = (36, 403, 81)
        if dim1.shape != expected or dim3.shape != expected:
            raise ValueError(
                f"official snapshot train tensors must have shape {expected}"
            )
    report = run_tafazoli_session_operator_probe_from_arrays(
        dim1,
        dim3,
        config=config,
    )
    verified = replace(
        report,
        source_file_md5=observed_md5,
        official_checksum_verified=True,
    )
    validate_probe_report(verified)
    return verified


def validate_claim_locks(locks: ProbeClaimLocks) -> None:
    """Reject any attempt to unlock an inference outside this probe."""

    if not isinstance(locks, ProbeClaimLocks):
        raise TypeError("locks must be ProbeClaimLocks")
    unlocked = tuple(
        key for key, value in asdict(locks).items() if value is not False
    )
    if unlocked:
        raise ValueError(f"stationary-probe claim locks must remain false: {unlocked}")


def validate_probe_report(report: TafazoliSessionOperatorProbeReport) -> None:
    """Check report invariants used by downstream claim gates."""

    if not isinstance(report, TafazoliSessionOperatorProbeReport):
        raise TypeError("report must be TafazoliSessionOperatorProbeReport")
    if not isinstance(report.config, ProbeConfig):
        raise TypeError("report config must be ProbeConfig")
    if report.schema_version != SCHEMA_VERSION or report.scope != PROBE_SCOPE:
        raise ValueError("unexpected probe schema or scope")
    if report.blind_fields_used:
        raise ValueError("labels or factor metadata entered a blind probe")
    if not report.train_only_preprocessing:
        raise ValueError("all preprocessing must remain train-only")
    if report.saved_test_role != "not_used":
        raise ValueError("saved classifier test rows must not be used")
    if report.source_file_md5 is None:
        if report.official_checksum_verified:
            raise ValueError("array-only reports cannot claim checksum verification")
    elif (
        report.source_file_md5 != OFFICIAL_CLASSIFIER_MD5
        or not report.official_checksum_verified
    ):
        raise ValueError("official source checksum must be exact and verified")
    validate_claim_locks(report.claim_locks)
    verdict_map = {item.key: item.answer for item in report.verdicts}
    required = {
        VERDICT_SESSION_LOCAL_SHORT_MEMORY,
        VERDICT_SHARED_STATIONARY_DIRECTED_OPERATOR,
        VERDICT_STATE_DEPENDENT_SWITCHING_OPERATOR,
        VERDICT_SHARED_CALL_GRAPH_HIERARCHICAL_OPERATOR,
        VERDICT_BRAIN_PROGRAMMING_LANGUAGE,
    }
    if set(verdict_map) != required:
        raise ValueError("report verdict keys do not match the locked interface")
    if verdict_map[VERDICT_STATE_DEPENDENT_SWITCHING_OPERATOR] != PENDING:
        raise ValueError("switching operator must remain pending")
    if (
        verdict_map[VERDICT_SHARED_CALL_GRAPH_HIERARCHICAL_OPERATOR]
        != TEST_UNAVAILABLE
    ):
        raise ValueError("shared call graph must remain untested")
    if verdict_map[VERDICT_BRAIN_PROGRAMMING_LANGUAGE] != NO:
        raise ValueError("stationary probe cannot identify a brain language")
    session_keys = {
        (item.session_index_one_based, item.dimension)
        for item in report.within_results
    }
    expected_within = {
        (spec.index_one_based, dimension)
        for spec in report.session_specs
        for dimension in ALLOWED_DIMENSIONS
    }
    if session_keys != expected_within:
        raise ValueError("within results must contain every session and dim1/3")
    transfer_keys = {
        (
            item.session_index_one_based,
            item.source_dimension,
            item.target_dimension,
        )
        for item in report.transfer_results
    }
    expected_transfer = {
        (spec.index_one_based, 1, 3) for spec in report.session_specs
    } | {(spec.index_one_based, 3, 1) for spec in report.session_specs}
    if transfer_keys != expected_transfer:
        raise ValueError("transfer results must contain both frozen directions")
    if not all(item.representation_fit_on_source_only for item in report.transfer_results):
        raise ValueError("cross-dimension representation must remain source-frozen")
    payload = report.to_dict()
    json.dumps(payload, sort_keys=True, allow_nan=False)


def _validate_population_tensor(
    population: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    values = np.asarray(population, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"{label} must be trial x neuron x time")
    if min(values.shape) < 1:
        raise ValueError(f"{label} axes must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains NaN or infinity")
    if np.any(values < 0):
        raise ValueError(f"{label} contains negative spike counts")
    return values


def _anscombe_trial_time_neuron(population: np.ndarray) -> np.ndarray:
    return np.sqrt(np.transpose(population, (0, 2, 1)) + 3.0 / 8.0)


def _analysis_key(
    source_dimension: int,
    target_dimension: int,
) -> str:
    if source_dimension == target_dimension:
        return f"within_dim{source_dimension}"
    return f"frozen_dim{source_dimension}_to_dim{target_dimension}"


def _transition_pairs(
    latent: np.ndarray,
    *,
    lag_bins: int,
    stride_bins: int,
    reverse: bool = False,
    successor_order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    starts = transition_start_indices(
        latent.shape[1],
        lag_bins=lag_bins,
        stride_bins=stride_bins,
    )
    current = latent[:, starts, :]
    successor = latent[:, starts + lag_bins, :]
    if successor_order is not None:
        successor = successor[successor_order]
    if reverse:
        current, successor = successor, current
    return current, successor


def _fit_affine_map(current: np.ndarray, successor: np.ndarray) -> np.ndarray:
    x = current.reshape(-1, current.shape[-1])
    y = successor.reshape(-1, successor.shape[-1])
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _apply_affine_map(current: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    x = current.reshape(-1, current.shape[-1])
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    predicted = design @ coefficients
    return predicted.reshape(current.shape)


def _sse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum(np.square(observed - predicted), dtype=np.float64))


def _skill(model_sse: float, baseline_sse: float) -> float:
    epsilon = np.finfo(np.float64).eps
    if baseline_sse <= epsilon:
        if model_sse <= epsilon:
            return 0.0
        return float(1.0 - model_sse / epsilon)
    return float(1.0 - model_sse / baseline_sse)


def _spectral_radius(coefficients: np.ndarray) -> float:
    operator = coefficients[:-1, :]
    return float(np.max(np.abs(np.linalg.eigvals(operator))))


def _derived_rng(config: ProbeConfig, *tokens: Any) -> np.random.Generator:
    canonical = json.dumps(
        (SCHEMA_VERSION, config.seed, *tokens),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    seed = int.from_bytes(digest[:16], byteorder="little", signed=False)
    return np.random.Generator(np.random.PCG64(seed))


def _derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    identity = np.arange(size, dtype=np.int64)
    for _ in range(1_000):
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate
    return np.roll(identity, 1)


def _evaluate_session_direction(
    source: np.ndarray,
    target: np.ndarray,
    *,
    session: SessionSpec,
    source_dimension: int,
    target_dimension: int,
    folds: tuple[WholeTrialFold, ...],
    config: ProbeConfig,
    rank_cap: int,
    event_mean_removed: bool,
    include_shuffles: bool,
) -> SessionOperatorResult | SessionTransferResult:
    totals = {
        "model": 0.0,
        "source_grand_mean": 0.0,
        "persistence": 0.0,
        "time_mean": 0.0,
        "reverse": 0.0,
        "target_refit": 0.0,
    }
    shuffle_totals = np.zeros(
        config.successor_shuffle_count if include_shuffles else 0,
        dtype=np.float64,
    )
    transition_count = 0
    ranks = []
    active_counts = []
    forward_radii = []
    reverse_radii = []
    operator_distances = []

    for fold in folds:
        prepared = prepare_session_latent_fold(
            source,
            target,
            fold,
            rank_cap=rank_cap,
            event_mean_removed=event_mean_removed,
        )
        source_current, source_successor = _transition_pairs(
            prepared.source_train,
            lag_bins=config.lag_bins,
            stride_bins=config.transition_stride_bins,
        )
        source_reverse_current, source_reverse_successor = _transition_pairs(
            prepared.source_train,
            lag_bins=config.lag_bins,
            stride_bins=config.transition_stride_bins,
            reverse=True,
        )
        test_current, test_successor = _transition_pairs(
            prepared.target_test_in_source_coordinates,
            lag_bins=config.lag_bins,
            stride_bins=config.transition_stride_bins,
        )
        target_train_current, target_train_successor = _transition_pairs(
            prepared.target_train_in_source_coordinates,
            lag_bins=config.lag_bins,
            stride_bins=config.transition_stride_bins,
        )
        forward = _fit_affine_map(source_current, source_successor)
        reverse = _fit_affine_map(
            source_reverse_current,
            source_reverse_successor,
        )
        target_refit = _fit_affine_map(
            target_train_current,
            target_train_successor,
        )
        prediction = _apply_affine_map(test_current, forward)
        reverse_prediction = _apply_affine_map(test_current, reverse)
        target_refit_prediction = _apply_affine_map(
            test_current,
            target_refit,
        )
        time_mean = source_successor.mean(axis=0, keepdims=True)
        time_mean = np.broadcast_to(time_mean, test_successor.shape)
        zeros = np.zeros_like(test_successor)

        totals["model"] += _sse(test_successor, prediction)
        totals["source_grand_mean"] += _sse(test_successor, zeros)
        totals["persistence"] += _sse(test_successor, test_current)
        totals["time_mean"] += _sse(test_successor, time_mean)
        totals["reverse"] += _sse(test_successor, reverse_prediction)
        totals["target_refit"] += _sse(
            test_successor,
            target_refit_prediction,
        )
        transition_count += int(np.prod(test_successor.shape[:-1]))
        ranks.append(prepared.transform.rank)
        active_counts.append(prepared.transform.active_neuron_count)
        forward_radii.append(_spectral_radius(forward))
        reverse_radii.append(_spectral_radius(reverse))
        operator_distances.append(
            float(
                np.linalg.norm(
                    forward[:-1, :] - target_refit[:-1, :],
                    ord="fro",
                )
                / np.sqrt(forward.shape[1])
            )
        )

        for replicate in range(shuffle_totals.size):
            rng = _derived_rng(
                config,
                "deranged_successor",
                session.index_one_based,
                source_dimension,
                target_dimension,
                int(event_mean_removed),
                rank_cap,
                fold.index_zero_based,
                replicate,
            )
            order = _derangement(prepared.source_train.shape[0], rng)
            _, shuffled_successor = _transition_pairs(
                prepared.source_train,
                lag_bins=config.lag_bins,
                stride_bins=config.transition_stride_bins,
                successor_order=order,
            )
            shuffled = _fit_affine_map(source_current, shuffled_successor)
            shuffled_prediction = _apply_affine_map(test_current, shuffled)
            shuffle_totals[replicate] += _sse(
                test_successor,
                shuffled_prediction,
            )

    if shuffle_totals.size:
        shuffle_median = float(np.median(shuffle_totals))
        shuffle_advantage = _skill(totals["model"], shuffle_median)
        shuffle_fraction_worse = float(
            np.mean(shuffle_totals >= totals["model"])
        )
    else:
        shuffle_advantage = 0.0
        shuffle_fraction_worse = 0.0
    metrics = PredictionMetrics(
        transition_count=transition_count,
        model_sse=totals["model"],
        source_grand_mean_sse=totals["source_grand_mean"],
        persistence_sse=totals["persistence"],
        time_locked_mean_sse=totals["time_mean"],
        reverse_control_sse=totals["reverse"],
        source_grand_mean_r2=_skill(
            totals["model"],
            totals["source_grand_mean"],
        ),
        persistence_skill=_skill(
            totals["model"],
            totals["persistence"],
        ),
        time_locked_mean_skill=_skill(
            totals["model"],
            totals["time_mean"],
        ),
        direction_specificity=_skill(
            totals["model"],
            totals["reverse"],
        ),
        successor_shuffle_advantage=shuffle_advantage,
        successor_shuffle_fraction_worse=shuffle_fraction_worse,
    )
    common = {
        "analysis_key": _analysis_key(
            source_dimension,
            target_dimension,
        ),
        "session_index_one_based": session.index_one_based,
        "animal": session.animal,
        "neuron_count": session.neuron_count,
        "rank_cap": rank_cap,
        "effective_rank": int(min(ranks)),
        "minimum_active_neurons": int(min(active_counts)),
        "event_mean_removed": event_mean_removed,
    }
    if source_dimension == target_dimension:
        return SessionOperatorResult(
            **common,
            dimension=source_dimension,
            forward_spectral_radius_median=float(np.median(forward_radii)),
            reverse_spectral_radius_median=float(np.median(reverse_radii)),
            metrics=metrics,
        )
    return SessionTransferResult(
        **common,
        source_dimension=source_dimension,
        target_dimension=target_dimension,
        representation_fit_on_source_only=True,
        metrics=metrics,
        target_refit_source_grand_mean_r2=_skill(
            totals["target_refit"],
            totals["source_grand_mean"],
        ),
        frozen_vs_target_refit_skill=_skill(
            totals["model"],
            totals["target_refit"],
        ),
        source_target_operator_frobenius_median=float(
            np.median(operator_distances)
        ),
    )


def _run_within_grid(
    arrays: Mapping[int, np.ndarray],
    specs: tuple[SessionSpec, ...],
    folds: tuple[WholeTrialFold, ...],
    *,
    config: ProbeConfig,
    rank_cap: int,
    event_mean_removed: bool,
    include_shuffles: bool,
) -> tuple[SessionOperatorResult, ...]:
    results = []
    for dimension in ALLOWED_DIMENSIONS:
        array = arrays[dimension]
        for session in specs:
            columns = slice(
                session.column_start_zero_based,
                session.column_stop_exclusive,
            )
            result = _evaluate_session_direction(
                array[:, columns, :],
                array[:, columns, :],
                session=session,
                source_dimension=dimension,
                target_dimension=dimension,
                folds=folds,
                config=config,
                rank_cap=rank_cap,
                event_mean_removed=event_mean_removed,
                include_shuffles=include_shuffles,
            )
            if not isinstance(result, SessionOperatorResult):
                raise AssertionError("within-grid result type mismatch")
            results.append(result)
    return tuple(results)


def _run_transfer_grid(
    arrays: Mapping[int, np.ndarray],
    specs: tuple[SessionSpec, ...],
    folds: tuple[WholeTrialFold, ...],
    *,
    config: ProbeConfig,
    rank_cap: int,
    event_mean_removed: bool,
    include_shuffles: bool,
) -> tuple[SessionTransferResult, ...]:
    results = []
    for source_dimension, target_dimension in ((1, 3), (3, 1)):
        for session in specs:
            columns = slice(
                session.column_start_zero_based,
                session.column_stop_exclusive,
            )
            result = _evaluate_session_direction(
                arrays[source_dimension][:, columns, :],
                arrays[target_dimension][:, columns, :],
                session=session,
                source_dimension=source_dimension,
                target_dimension=target_dimension,
                folds=folds,
                config=config,
                rank_cap=rank_cap,
                event_mean_removed=event_mean_removed,
                include_shuffles=include_shuffles,
            )
            if not isinstance(result, SessionTransferResult):
                raise AssertionError("transfer-grid result type mismatch")
            results.append(result)
    return tuple(results)


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _aggregate_results(
    results: tuple[SessionOperatorResult | SessionTransferResult, ...],
) -> tuple[AggregateResult, ...]:
    keys = sorted(
        {
            (item.analysis_key, item.event_mean_removed)
            for item in results
        }
    )
    aggregates = []
    for analysis_key, event_mean_removed in keys:
        selected = tuple(
            item
            for item in results
            if item.analysis_key == analysis_key
            and item.event_mean_removed == event_mean_removed
        )
        animals = ("ALL",) + tuple(sorted({item.animal for item in selected}))
        for animal in animals:
            group = (
                selected
                if animal == "ALL"
                else tuple(item for item in selected if item.animal == animal)
            )
            transfer_values = tuple(
                item.frozen_vs_target_refit_skill
                for item in group
                if isinstance(item, SessionTransferResult)
            )
            aggregates.append(
                AggregateResult(
                    analysis_key=analysis_key,
                    animal=animal,
                    session_count=len(group),
                    event_mean_removed=event_mean_removed,
                    median_source_grand_mean_r2=_median(
                        [
                            item.metrics.source_grand_mean_r2
                            for item in group
                        ]
                    ),
                    median_persistence_skill=_median(
                        [item.metrics.persistence_skill for item in group]
                    ),
                    median_time_locked_mean_skill=_median(
                        [
                            item.metrics.time_locked_mean_skill
                            for item in group
                        ]
                    ),
                    median_direction_specificity=_median(
                        [
                            item.metrics.direction_specificity
                            for item in group
                        ]
                    ),
                    median_successor_shuffle_advantage=_median(
                        [
                            item.metrics.successor_shuffle_advantage
                            for item in group
                        ]
                    ),
                    median_frozen_vs_target_refit_skill=(
                        _median(transfer_values) if transfer_values else None
                    ),
                )
            )
    return tuple(aggregates)


def _run_rank_stability(
    arrays: Mapping[int, np.ndarray],
    specs: tuple[SessionSpec, ...],
    folds: tuple[WholeTrialFold, ...],
    *,
    config: ProbeConfig,
    primary_within: tuple[SessionOperatorResult, ...],
    primary_transfer: tuple[SessionTransferResult, ...],
) -> tuple[RankStabilityResult, ...]:
    results = []
    for rank_cap in config.rank_stability_caps:
        if rank_cap == config.rank_cap:
            within = primary_within
            transfer = primary_transfer
        else:
            within = _run_within_grid(
                arrays,
                specs,
                folds,
                config=config,
                rank_cap=rank_cap,
                event_mean_removed=False,
                include_shuffles=False,
            )
            transfer = _run_transfer_grid(
                arrays,
                specs,
                folds,
                config=config,
                rank_cap=rank_cap,
                event_mean_removed=False,
                include_shuffles=False,
            )
        combined: tuple[
            SessionOperatorResult | SessionTransferResult, ...
        ] = tuple(within) + tuple(transfer)
        for analysis_key in sorted({item.analysis_key for item in combined}):
            group = tuple(
                item for item in combined if item.analysis_key == analysis_key
            )
            transfer_values = tuple(
                item.frozen_vs_target_refit_skill
                for item in group
                if isinstance(item, SessionTransferResult)
            )
            results.append(
                RankStabilityResult(
                    analysis_key=analysis_key,
                    rank_cap=rank_cap,
                    session_count=len(group),
                    median_source_grand_mean_r2=_median(
                        [
                            item.metrics.source_grand_mean_r2
                            for item in group
                        ]
                    ),
                    median_persistence_skill=_median(
                        [item.metrics.persistence_skill for item in group]
                    ),
                    median_direction_specificity=_median(
                        [
                            item.metrics.direction_specificity
                            for item in group
                        ]
                    ),
                    median_frozen_vs_target_refit_skill=(
                        _median(transfer_values) if transfer_values else None
                    ),
                )
            )
    return tuple(results)


def _all_session_aggregate(
    results: Sequence[SessionOperatorResult | SessionTransferResult],
    analysis_key: str,
) -> tuple[SessionOperatorResult | SessionTransferResult, ...]:
    return tuple(
        item
        for item in results
        if item.analysis_key == analysis_key and not item.event_mean_removed
    )


def _build_verdicts(
    within: tuple[SessionOperatorResult, ...],
    transfer: tuple[SessionTransferResult, ...],
    *,
    config: ProbeConfig,
) -> tuple[ClaimVerdict, ...]:
    dim1 = _all_session_aggregate(within, "within_dim1")
    dim3 = _all_session_aggregate(within, "within_dim3")
    short_memory = all(
        _median(
            [item.metrics.successor_shuffle_advantage for item in group]
        )
        > 0.0
        for group in (dim1, dim3)
    )
    directional_groups = (
        dim1,
        dim3,
        _all_session_aggregate(transfer, "frozen_dim1_to_dim3"),
        _all_session_aggregate(transfer, "frozen_dim3_to_dim1"),
    )
    direction_gate = all(
        _median([item.metrics.direction_specificity for item in group])
        >= config.minimum_direction_specificity
        for group in directional_groups
    )
    absolute_gate = all(
        _median(
            [item.metrics.source_grand_mean_r2 for item in group]
        )
        >= config.minimum_source_grand_mean_r2
        for group in directional_groups
    )
    transfer_gate = all(
        _median(
            [
                item.frozen_vs_target_refit_skill
                for item in group
                if isinstance(item, SessionTransferResult)
            ]
        )
        >= config.minimum_frozen_vs_target_refit_skill
        for group in directional_groups[2:]
    )
    shared_supported = direction_gate and absolute_gate and transfer_gate
    return (
        ClaimVerdict(
            key=VERDICT_SESSION_LOCAL_SHORT_MEMORY,
            answer=YES if short_memory else NO,
            reason=(
                "Paired within-trial successors outperform deranged whole-trial "
                "successors in both allowed dimensions."
                if short_memory
                else "The deranged-successor control is not beaten in both dimensions."
            ),
        ),
        ClaimVerdict(
            key=VERDICT_SHARED_STATIONARY_DIRECTED_OPERATOR,
            answer=PENDING if shared_supported else NO,
            reason=(
                "Numerical stationary gates pass, but held-out-session neuron "
                "alignment and causal tests remain unavailable."
                if shared_supported
                else "Absolute prediction, direction specificity, and frozen "
                "cross-dimension transfer do not jointly pass."
            ),
        ),
        ClaimVerdict(
            key=VERDICT_STATE_DEPENDENT_SWITCHING_OPERATOR,
            answer=PENDING,
            reason="A one-operator model cannot adjudicate K=2/3 switching dynamics.",
        ),
        ClaimVerdict(
            key=VERDICT_SHARED_CALL_GRAPH_HIERARCHICAL_OPERATOR,
            answer=TEST_UNAVAILABLE,
            reason=(
                "No parent operator, task/session residual, call/return edge, "
                "or common-successor-state model is fitted here."
            ),
        ),
        ClaimVerdict(
            key=VERDICT_BRAIN_PROGRAMMING_LANGUAGE,
            answer=NO,
            reason=(
                "An observational stationary linear predictor is not an opcode, "
                "grammar, causal instruction set, or programming language."
            ),
        ),
    )


def _next_tests() -> tuple[NextTest, ...]:
    return (
        NextTest(
            key=NEXT_TEST_SWITCHING_K2_K3,
            status=PENDING,
            question=(
                "Do label-free K=2/3 state-conditioned operators beat the "
                "same-parameter stationary model on held-out whole trials?"
            ),
            required_control=(
                "Reuse the exact folds and train-only latent preparation; compare "
                "reverse-time and deranged-successor controls."
            ),
        ),
        NextTest(
            key=NEXT_TEST_EVENT_TIME_BOTTLENECK,
            status=PENDING,
            question=(
                "Do different trajectories converge on an event-time bottleneck "
                "beyond the train-only time-locked mean?"
            ),
            required_control=(
                "Predict held-out convergence against event-time mean, matched "
                "initial-state, and time-reversal baselines."
            ),
        ),
        NextTest(
            key=NEXT_TEST_PARENT_PLUS_RESIDUAL,
            status=PENDING,
            question=(
                "Can one parent operator plus session/task residuals predict "
                "better than unrelated session operators?"
            ),
            required_control=(
                "Hierarchical shrinkage must be fit without cross-session neuron "
                "weight alignment and evaluated by held-out sessions/animals."
            ),
        ),
        NextTest(
            key=NEXT_TEST_COMMON_SUCCESSOR,
            status=PENDING,
            question=(
                "Do distinct inferred states share a successor state consistent "
                "with dispatch, join, or return semantics?"
            ),
            required_control=(
                "Predeclare successor matching and compare against occupancy-, "
                "time-, and trajectory-shuffled transition graphs."
            ),
        ),
    )


__all__ = [
    "ALLOWED_DIMENSIONS",
    "NO",
    "OFFICIAL_CLASSIFIER_MD5",
    "PENDING",
    "PROBE_SCOPE",
    "ProbeClaimLocks",
    "ProbeConfig",
    "RANK_STABILITY_CAPS",
    "SCHEMA_VERSION",
    "SESSION_ANIMALS",
    "SESSION_NEURON_COUNTS",
    "TEST_UNAVAILABLE",
    "TafazoliSessionOperatorProbeReport",
    "VERDICT_BRAIN_PROGRAMMING_LANGUAGE",
    "VERDICT_SESSION_LOCAL_SHORT_MEMORY",
    "VERDICT_SHARED_CALL_GRAPH_HIERARCHICAL_OPERATOR",
    "VERDICT_SHARED_STATIONARY_DIRECTED_OPERATOR",
    "VERDICT_STATE_DEPENDENT_SWITCHING_OPERATOR",
    "WholeTrialFold",
    "YES",
    "extract_tafazoli_train_dimensions",
    "fit_session_latent_transform",
    "load_tafazoli_train_dimensions",
    "make_whole_trial_folds",
    "prepare_session_latent_fold",
    "recovered_session_specs",
    "run_tafazoli_session_operator_probe",
    "run_tafazoli_session_operator_probe_from_arrays",
    "transition_start_indices",
    "validate_claim_locks",
    "validate_probe_report",
    "verify_official_classifier_checksum",
]
