"""Locked G7-M episodic-LTM and constrained dream-like factorial gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REGISTERED_CONFIG_SHA256 = (
    "6487156371e4c42877fa0813dd170fb000ce11fe05e51f34bceb74653159fac0"
)
_SLOT_SLICES = (slice(0, 5), slice(5, 7), slice(7, 12))
_OBSERVED_EDGES = ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0))
_MISSING_EDGES = ((0, 2), (1, 0), (2, 1))


@dataclass(frozen=True)
class Provenance:
    source: str
    epistemic_status: str
    observed: bool
    recalled: bool


REAL_PROVENANCE = Provenance("real", "observed", True, False)
SYNTHETIC_PROVENANCE = Provenance("synthetic", "hypothetical", False, False)
FALLBACK_PROVENANCE = Provenance("schema_fallback", "inferred", False, False)
RECALLED_PROVENANCE = Provenance("real", "recalled", False, True)


@dataclass(frozen=True)
class EpisodicRecord:
    episode_id: str
    context_token: str
    prefix_token: str
    suffix_token: str
    trajectory: np.ndarray
    provenance: Provenance = REAL_PROVENANCE


@dataclass(frozen=True)
class PartialCue:
    context_token: str
    prefix_token: str
    suffix_token: str
    cue_values: np.ndarray
    cue_mask: np.ndarray


@dataclass(frozen=True)
class RecallResult:
    accepted: bool
    episode_id: str | None
    reconstruction: np.ndarray
    confidence: float
    iterations: int
    converged: bool
    extra_step_stable: bool
    clamp_max_error: float
    provenance: Provenance


@dataclass(frozen=True)
class CoordinateStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, trajectory: np.ndarray) -> np.ndarray:
        values = np.asarray(trajectory, dtype=float)
        if values.shape != (12, 8):
            raise ValueError("trajectory must have shape (12, 8)")
        return ((values.reshape(-1) - self.mean) / self.scale).reshape(12, 8)

    def inverse(self, standardized: np.ndarray) -> np.ndarray:
        values = np.asarray(standardized, dtype=float)
        if values.shape != (12, 8):
            raise ValueError("standardized trajectory must have shape (12, 8)")
        return (values.reshape(-1) * self.scale + self.mean).reshape(12, 8)


@dataclass(frozen=True)
class SchemaEntry:
    context_token: str
    prefix_token: str
    suffix_token: str
    standardized_trajectory: np.ndarray
    provenance: Provenance
    binding_specific: bool


@dataclass(frozen=True)
class DreamBinding:
    context_token: str
    prefix_token: str
    suffix_token: str
    standardized_trajectory: np.ndarray
    left_join_rms: float
    right_join_rms: float
    provenance: Provenance = SYNTHETIC_PROVENANCE


@dataclass(frozen=True)
class CooccurrenceComponents:
    prefix_component: dict[tuple[str, str], int]
    suffix_component: dict[tuple[str, str], int]

    def same_component(
        self, context: str, prefix: str, suffix: str
    ) -> bool:
        left = self.prefix_component.get((context, prefix))
        right = self.suffix_component.get((context, suffix))
        return left is not None and left == right


@dataclass(frozen=True)
class _QuerySpec:
    context_token: str
    prefix_token: str
    suffix_token: str
    target: np.ndarray
    masks: dict[int, np.ndarray]
    noise: np.ndarray
    target_episode_id: str | None


@dataclass(frozen=True)
class _ScoredQuery:
    cue: PartialCue
    target: np.ndarray
    target_episode_id: str | None


@dataclass(frozen=True)
class _SeedWorld:
    records_a: tuple[EpisodicRecord, ...]
    records_b: tuple[EpisodicRecord, ...]
    canonical_a: tuple[EpisodicRecord, ...]
    canonical_b: tuple[EpisodicRecord, ...]
    observed_bases: dict[tuple[str, str, str], np.ndarray]
    recall_specs: tuple[_QuerySpec, ...]
    novel_specs: tuple[_QuerySpec, ...]
    lure_specs: tuple[_QuerySpec, ...]
    invalid_specs: tuple[_QuerySpec, ...]


def fit_coordinate_standardizer(
    records: Sequence[EpisodicRecord], *, floor: float = 1e-8
) -> CoordinateStandardizer:
    if not records:
        raise ValueError("standardization requires real wake records")
    values = np.stack([np.asarray(item.trajectory, dtype=float).reshape(-1) for item in records])
    if values.shape[1] != 96 or not np.all(np.isfinite(values)):
        raise ValueError("wake records must contain finite 12 by 8 trajectories")
    mean = np.mean(values, axis=0)
    scale = np.maximum(np.std(values, axis=0, ddof=0), float(floor))
    return CoordinateStandardizer(mean=mean, scale=scale)


def _rng(master_seed: int, stream_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(master_seed), int(stream_id)]))


def _stationary_ar(
    rng: np.random.Generator, steps: int, dimension: int, rho: float
) -> np.ndarray:
    result = np.empty((steps, dimension), dtype=float)
    result[0] = rng.normal(size=dimension)
    innovation = math.sqrt(1.0 - rho * rho)
    for index in range(1, steps):
        result[index] = rho * result[index - 1] + innovation * rng.normal(
            size=dimension
        )
    return result


def _transform(value: np.ndarray, context: str) -> np.ndarray:
    if context == "A":
        permutation = np.arange(8)
        signs = np.ones(8)
    elif context == "B":
        permutation = np.asarray((1, 0, 3, 2, 5, 4, 7, 6))
        signs = np.asarray((1, -1, 1, -1, 1, -1, 1, -1))
    else:
        raise ValueError("context must be A or B")
    return signs * np.asarray(value)[permutation]


def _slot_mask(order_rng: np.random.Generator) -> dict[int, np.ndarray]:
    slot_sizes = (40, 16, 40)
    visible = {12: (5, 2, 5), 24: (10, 4, 10), 48: (20, 8, 20)}
    orders = [order_rng.permutation(size) for size in slot_sizes]
    offsets = (0, 40, 56)
    result: dict[int, np.ndarray] = {}
    for total, counts in visible.items():
        mask = np.zeros(96, dtype=bool)
        for order, offset, count in zip(orders, offsets, counts):
            mask[offset + order[:count]] = True
        result[total] = mask.reshape(12, 8)
    return result


def _instance_residual(
    rng: np.random.Generator, *, rho: float = 0.4
) -> np.ndarray:
    residual = _stationary_ar(rng, 12, 8, rho)
    scales = np.full(12, 0.2, dtype=float)
    scales[4:8] = 0.05
    return residual * scales[:, None]


def _generate_seed_world(master_seed: int) -> _SeedWorld:
    anchor_rng = _rng(master_seed, 0)
    primitive_rng = _rng(master_seed, 1)
    permutation_rng = _rng(master_seed, 2)
    residual_rng = _rng(master_seed, 3)
    order_rng = _rng(master_seed, 4)
    mask_rng = _rng(master_seed, 5)
    noise_rng = _rng(master_seed, 6)
    lure_rng = _rng(master_seed, 7)
    invalid_rng = _rng(master_seed, 8)

    matrix = anchor_rng.normal(size=(8, 8))
    q, r = np.linalg.qr(matrix)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    anchors = q * signs[None, :]
    anchors = anchors[:, :4].T

    prefixes: dict[tuple[str, int, int], np.ndarray] = {}
    suffixes: dict[tuple[str, int, int], np.ndarray] = {}
    opaque_prefix: dict[tuple[str, int, int], str] = {}
    opaque_suffix: dict[tuple[str, int, int], str] = {}
    context_tokens = {"A": "context-0", "B": "context-1"}
    for context in ("A", "B"):
        prefix_perm = permutation_rng.permutation(12)
        suffix_perm = permutation_rng.permutation(12)
        for port in range(4):
            for local in range(3):
                prefix = _stationary_ar(primitive_rng, 5, 8, 0.6)
                prefix[-1] = anchors[port]
                suffix = np.empty((5, 8), dtype=float)
                suffix[0] = _transform(anchors[port], context)
                innovation = math.sqrt(1.0 - 0.6**2)
                for index in range(1, 5):
                    suffix[index] = 0.6 * suffix[index - 1] + innovation * (
                        primitive_rng.normal(size=8)
                    )
                prefixes[(context, port, local)] = prefix
                suffixes[(context, port, local)] = suffix
                canonical = 3 * port + local
                opaque_prefix[(context, port, local)] = (
                    f"{context_tokens[context]}:prefix-{prefix_perm[canonical]}"
                )
                opaque_suffix[(context, port, local)] = (
                    f"{context_tokens[context]}:suffix-{suffix_perm[canonical]}"
                )

    canonical: dict[str, list[EpisodicRecord]] = {"A": [], "B": []}
    bases: dict[tuple[str, str, str], np.ndarray] = {}
    for context in ("A", "B"):
        context_token = context_tokens[context]
        for port in range(4):
            connector = np.stack((anchors[port], _transform(anchors[port], context)))
            for edge_index, (pre_local, suffix_local) in enumerate(_OBSERVED_EDGES):
                prefix_token = opaque_prefix[(context, port, pre_local)]
                suffix_token = opaque_suffix[(context, port, suffix_local)]
                base_trajectory = np.concatenate(
                    (
                        prefixes[(context, port, pre_local)],
                        connector,
                        suffixes[(context, port, suffix_local)],
                    ),
                    axis=0,
                )
                bases[(context_token, prefix_token, suffix_token)] = base_trajectory
                for instance in range(2):
                    ordinal = port * 12 + edge_index * 2 + instance
                    canonical[context].append(
                        EpisodicRecord(
                            episode_id=f"{context_token}:episode-{ordinal:02d}",
                            context_token=context_token,
                            prefix_token=prefix_token,
                            suffix_token=suffix_token,
                            trajectory=base_trajectory + _instance_residual(residual_rng),
                        )
                    )

    records_a = tuple(canonical["A"][index] for index in order_rng.permutation(48))
    records_b = tuple(canonical["B"][index] for index in order_rng.permutation(48))

    recall_specs: list[_QuerySpec] = []
    for record in canonical["A"]:
        recall_specs.append(
            _QuerySpec(
                record.context_token,
                record.prefix_token,
                record.suffix_token,
                record.trajectory,
                _slot_mask(mask_rng),
                noise_rng.normal(size=(12, 8)),
                record.episode_id,
            )
        )

    novel_specs: list[_QuerySpec] = []
    for context in ("A", "B"):
        context_token = context_tokens[context]
        for port in range(4):
            connector = np.stack((anchors[port], _transform(anchors[port], context)))
            for pre_local, suffix_local in _MISSING_EDGES:
                prefix_token = opaque_prefix[(context, port, pre_local)]
                suffix_token = opaque_suffix[(context, port, suffix_local)]
                target = np.concatenate(
                    (
                        prefixes[(context, port, pre_local)],
                        connector,
                        suffixes[(context, port, suffix_local)],
                    )
                )
                novel_specs.append(
                    _QuerySpec(
                        context_token,
                        prefix_token,
                        suffix_token,
                        target,
                        _slot_mask(mask_rng),
                        noise_rng.normal(size=(12, 8)),
                        None,
                    )
                )

    lure_specs: list[_QuerySpec] = []
    for context in ("A", "B"):
        for binding_index in range(0, 48, 2):
            record = canonical[context][binding_index]
            base_trajectory = bases[
                (record.context_token, record.prefix_token, record.suffix_token)
            ]
            lure_specs.append(
                _QuerySpec(
                    record.context_token,
                    record.prefix_token,
                    record.suffix_token,
                    base_trajectory + _instance_residual(lure_rng),
                    _slot_mask(mask_rng),
                    noise_rng.normal(size=(12, 8)),
                    None,
                )
            )

    invalid_specs: list[_QuerySpec] = []
    for context in ("A", "B"):
        context_token = context_tokens[context]
        for port in range(4):
            connector = np.stack((anchors[port], _transform(anchors[port], context)))
            other = (port + 1) % 4
            for local in range(3):
                target = np.concatenate(
                    (
                        prefixes[(context, port, local)],
                        connector,
                        suffixes[(context, other, local)],
                    )
                )
                invalid_specs.append(
                    _QuerySpec(
                        context_token,
                        opaque_prefix[(context, port, local)],
                        opaque_suffix[(context, other, local)],
                        target,
                        _slot_mask(mask_rng),
                        invalid_rng.normal(size=(12, 8)),
                        None,
                    )
                )
    return _SeedWorld(
        records_a=records_a,
        records_b=records_b,
        canonical_a=tuple(canonical["A"]),
        canonical_b=tuple(canonical["B"]),
        observed_bases=bases,
        recall_specs=tuple(recall_specs),
        novel_specs=tuple(novel_specs),
        lure_specs=tuple(lure_specs),
        invalid_specs=tuple(invalid_specs),
    )


def _materialize_query(
    specification: _QuerySpec,
    standardizer: CoordinateStandardizer,
    *,
    visible_count: int = 24,
    noise_standard_deviation: float = 0.1,
) -> _ScoredQuery:
    mask = specification.masks[visible_count]
    target = np.asarray(specification.target, dtype=float)
    values = np.zeros((12, 8), dtype=float)
    scaled_noise = (
        specification.noise.reshape(-1)
        * standardizer.scale
        * float(noise_standard_deviation)
    ).reshape(12, 8)
    values[mask] = target[mask] + scaled_noise[mask]
    return _ScoredQuery(
        cue=PartialCue(
            context_token=specification.context_token,
            prefix_token=specification.prefix_token,
            suffix_token=specification.suffix_token,
            cue_values=values,
            cue_mask=mask.copy(),
        ),
        target=np.asarray(specification.target, dtype=float).copy(),
        target_episode_id=specification.target_episode_id,
    )


def infer_cooccurrence_components(
    records: Sequence[EpisodicRecord],
) -> CooccurrenceComponents:
    adjacency: dict[tuple[str, str, str], set[tuple[str, str, str]]] = {}
    for record in records:
        prefix_node = (record.context_token, "prefix", record.prefix_token)
        suffix_node = (record.context_token, "suffix", record.suffix_token)
        adjacency.setdefault(prefix_node, set()).add(suffix_node)
        adjacency.setdefault(suffix_node, set()).add(prefix_node)
    prefix_components: dict[tuple[str, str], int] = {}
    suffix_components: dict[tuple[str, str], int] = {}
    component = 0
    for start in sorted(adjacency):
        key = (start[0], start[2])
        destination = prefix_components if start[1] == "prefix" else suffix_components
        if key in destination:
            continue
        stack = [start]
        seen: set[tuple[str, str, str]] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            node_key = (node[0], node[2])
            if node[1] == "prefix":
                prefix_components[node_key] = component
            else:
                suffix_components[node_key] = component
            stack.extend(adjacency.get(node, ()))
        component += 1
    return CooccurrenceComponents(prefix_components, suffix_components)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


class PersistentEpisodicStore:
    """Fixed-capacity index that accepts real wake records only."""

    def __init__(
        self,
        standardizer: CoordinateStandardizer,
        *,
        capacity: int = 96,
        threshold: float = math.inf,
        beta: float = 12.0,
        damping: float = 0.5,
        maximum_iterations: int = 20,
        tolerance: float = 1e-6,
    ) -> None:
        self.standardizer = standardizer
        self.capacity = int(capacity)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.damping = float(damping)
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance)
        self._records: list[EpisodicRecord] = []
        self.synthetic_insert_attempts = 0

    @property
    def records(self) -> tuple[EpisodicRecord, ...]:
        return tuple(self._records)

    @property
    def trace_bytes(self) -> int:
        return int(sum(item.trajectory.nbytes for item in self._records))

    def insert_real(self, record: EpisodicRecord) -> None:
        if record.provenance != REAL_PROVENANCE:
            self.synthetic_insert_attempts += 1
            raise ValueError("episodic LTM accepts real wake observations only")
        trajectory = np.asarray(record.trajectory, dtype=float)
        if trajectory.shape != (12, 8) or not np.all(np.isfinite(trajectory)):
            raise ValueError("episodic record trajectory must be finite and 12 by 8")
        if len(self._records) >= self.capacity:
            raise OverflowError("registered episodic capacity exceeded")
        if any(item.episode_id == record.episode_id for item in self._records):
            raise ValueError("episode identity must be unique")
        owned = trajectory.copy()
        owned.setflags(write=False)
        self._records.append(
            EpisodicRecord(
                episode_id=str(record.episode_id),
                context_token=str(record.context_token),
                prefix_token=str(record.prefix_token),
                suffix_token=str(record.suffix_token),
                trajectory=owned,
                provenance=record.provenance,
            )
        )

    def recurrent_clamped_recall(self, cue: PartialCue) -> RecallResult:
        return recurrent_clamped_recall(self, cue)


def _masked_cosines(
    cue: np.ndarray, mask: np.ndarray, traces: np.ndarray
) -> np.ndarray:
    observed = mask.reshape(-1)
    left = cue.reshape(-1)[observed]
    right = traces[:, observed]
    numerator = right @ left
    denominator = np.linalg.norm(right, axis=1) * np.linalg.norm(left)
    return numerator / np.maximum(denominator, 1e-12)


def _full_cosines(state: np.ndarray, traces: np.ndarray) -> np.ndarray:
    numerator = traces @ state
    denominator = np.linalg.norm(traces, axis=1) * np.linalg.norm(state)
    return numerator / np.maximum(denominator, 1e-12)


def recurrent_clamped_recall(
    store: PersistentEpisodicStore, cue: PartialCue
) -> RecallResult:
    """Reinstate a stored trace without accepting evaluator-only truth inputs."""

    values = np.asarray(cue.cue_values, dtype=float)
    mask = np.asarray(cue.cue_mask, dtype=bool)
    if values.shape != (12, 8) or mask.shape != (12, 8):
        raise ValueError("cue values and mask must have shape (12, 8)")
    standardized_cue = store.standardizer.transform(values).reshape(-1)
    mask_flat = mask.reshape(-1)
    empty_reconstruction = np.zeros((12, 8), dtype=float)
    if not store.records:
        return RecallResult(
            False, None, empty_reconstruction, -math.inf, 0, True, True, 0.0,
            FALLBACK_PROVENANCE,
        )
    components = infer_cooccurrence_components(store.records)
    if not components.same_component(
        cue.context_token, cue.prefix_token, cue.suffix_token
    ):
        return RecallResult(
            False, None, empty_reconstruction, -math.inf, 0, True, True, 0.0,
            FALLBACK_PROVENANCE,
        )
    traces = np.stack(
        [store.standardizer.transform(item.trajectory).reshape(-1) for item in store.records]
    )
    initial_scores = _masked_cosines(standardized_cue, mask, traces)
    confidence = float(np.max(initial_scores))
    attention = _softmax(store.beta * initial_scores)
    weighted = attention @ traces
    state = weighted.copy()
    state[mask_flat] = standardized_cue[mask_flat]
    converged = False
    iterations = 1
    for iteration in range(1, store.maximum_iterations):
        attention = _softmax(store.beta * _full_cosines(state, traces))
        weighted = attention @ traces
        following = state.copy()
        following[~mask_flat] = (
            store.damping * state[~mask_flat]
            + (1.0 - store.damping) * weighted[~mask_flat]
        )
        following[mask_flat] = standardized_cue[mask_flat]
        delta = float(np.sqrt(np.mean((following[~mask_flat] - state[~mask_flat]) ** 2)))
        state = following
        iterations = iteration + 1
        if delta <= store.tolerance:
            converged = True
            break
    final_attention = _softmax(store.beta * _full_cosines(state, traces))
    identity_index = int(np.argmax(final_attention))
    accepted = bool(confidence > store.threshold)
    episode_id = store.records[identity_index].episode_id if accepted else None

    next_attention = _softmax(store.beta * _full_cosines(state, traces))
    next_weighted = next_attention @ traces
    extra = state.copy()
    extra[~mask_flat] = (
        store.damping * state[~mask_flat]
        + (1.0 - store.damping) * next_weighted[~mask_flat]
    )
    extra[mask_flat] = standardized_cue[mask_flat]
    extra_delta = float(np.sqrt(np.mean((extra[~mask_flat] - state[~mask_flat]) ** 2)))
    extra_identity = int(
        np.argmax(_softmax(store.beta * _full_cosines(extra, traces)))
    )
    extra_stable = bool(extra_identity == identity_index and extra_delta <= store.tolerance)
    clamp_error = float(
        np.max(np.abs(state[mask_flat] - standardized_cue[mask_flat]))
    )
    reconstruction = store.standardizer.inverse(state.reshape(12, 8))
    return RecallResult(
        accepted=accepted,
        episode_id=episode_id,
        reconstruction=reconstruction,
        confidence=confidence,
        iterations=iterations,
        converged=converged,
        extra_step_stable=extra_stable,
        clamp_max_error=clamp_error,
        provenance=RECALLED_PROVENANCE if accepted else FALLBACK_PROVENANCE,
    )


class SlowSchemaTable:
    """Observed binding means plus provenance-separated missing entries."""

    def __init__(
        self,
        records: Sequence[EpisodicRecord],
        standardizer: CoordinateStandardizer,
    ) -> None:
        self.standardizer = standardizer
        self.components = infer_cooccurrence_components(records)
        grouped: dict[tuple[str, str, str], list[np.ndarray]] = {}
        for record in records:
            key = (record.context_token, record.prefix_token, record.suffix_token)
            grouped.setdefault(key, []).append(standardizer.transform(record.trajectory))
        self._observed = {
            key: SchemaEntry(
                *key,
                standardized_trajectory=np.mean(values, axis=0),
                provenance=REAL_PROVENANCE,
                binding_specific=True,
            )
            for key, values in grouped.items()
        }
        self._synthetic: dict[tuple[str, str, str], SchemaEntry] = {}
        self.observed_overwrite_count = 0

    @property
    def observed_entries(self) -> Mapping[tuple[str, str, str], SchemaEntry]:
        return dict(self._observed)

    @property
    def synthetic_entries(self) -> Mapping[tuple[str, str, str], SchemaEntry]:
        return dict(self._synthetic)

    def lookup(self, context: str, prefix: str, suffix: str) -> SchemaEntry | None:
        key = (context, prefix, suffix)
        if key in self._observed:
            return self._observed[key]
        if key in self._synthetic:
            return self._synthetic[key]
        if not self.components.same_component(context, prefix, suffix):
            return None
        component = self.components.prefix_component[(context, prefix)]
        candidates = [
            value.standardized_trajectory
            for key, value in self._observed.items()
            if key[0] == context
            and self.components.prefix_component[(key[0], key[1])] == component
        ]
        return SchemaEntry(
            context,
            prefix,
            suffix,
            np.mean(candidates, axis=0),
            FALLBACK_PROVENANCE,
            False,
        )


def observed_binding_hash(table: SlowSchemaTable) -> str:
    digest = hashlib.sha256()
    for key in sorted(table._observed):
        for token in key:
            digest.update(token.encode("utf-8"))
            digest.update(b"\0")
        values = np.ascontiguousarray(
            table._observed[key].standardized_trajectory, dtype=np.float64
        )
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def constrained_missing_binding_dream(
    real_records: Sequence[EpisodicRecord],
    standardizer: CoordinateStandardizer,
    join_threshold: float,
) -> tuple[DreamBinding, ...]:
    """Purely propose constrained missing bindings from a real-record snapshot."""

    records = tuple(real_records)
    if any(record.provenance != REAL_PROVENANCE for record in records):
        raise ValueError("dream workspace accepts real wake records only")
    components = infer_cooccurrence_components(records)
    grouped: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for record in records:
        key = (record.context_token, record.prefix_token, record.suffix_token)
        grouped.setdefault(key, []).append(standardizer.transform(record.trajectory))
    binding_means = {key: np.mean(values, axis=0) for key, values in grouped.items()}
    observed_keys = set(binding_means)
    prefixes: dict[tuple[str, str], list[np.ndarray]] = {}
    suffixes: dict[tuple[str, str], list[np.ndarray]] = {}
    connectors: dict[tuple[str, int], list[np.ndarray]] = {}
    for key, trajectory in binding_means.items():
        context, prefix, suffix = key
        prefixes.setdefault((context, prefix), []).append(trajectory[_SLOT_SLICES[0]])
        suffixes.setdefault((context, suffix), []).append(trajectory[_SLOT_SLICES[2]])
        component = components.prefix_component[(context, prefix)]
        connectors.setdefault((context, component), []).append(
            trajectory[_SLOT_SLICES[1]]
        )
    prefix_means = {key: np.mean(value, axis=0) for key, value in prefixes.items()}
    suffix_means = {key: np.mean(value, axis=0) for key, value in suffixes.items()}
    connector_means = {key: np.mean(value, axis=0) for key, value in connectors.items()}

    proposals: list[DreamBinding] = []
    contexts = sorted({key[0] for key in prefix_means})
    for context in contexts:
        context_prefixes = sorted(key[1] for key in prefix_means if key[0] == context)
        context_suffixes = sorted(key[1] for key in suffix_means if key[0] == context)
        for prefix in context_prefixes:
            component = components.prefix_component[(context, prefix)]
            for suffix in context_suffixes:
                if not components.same_component(context, prefix, suffix):
                    continue
                key = (context, prefix, suffix)
                if key in observed_keys:
                    continue
                prefix_slot = prefix_means[(context, prefix)]
                connector = connector_means[(context, component)]
                suffix_slot = suffix_means[(context, suffix)]
                left = float(np.linalg.norm(prefix_slot[-1] - connector[0]) / math.sqrt(8))
                right = float(np.linalg.norm(connector[-1] - suffix_slot[0]) / math.sqrt(8))
                if left <= join_threshold and right <= join_threshold:
                    proposals.append(
                        DreamBinding(
                            context,
                            prefix,
                            suffix,
                            np.concatenate((prefix_slot, connector, suffix_slot)),
                            left,
                            right,
                        )
                    )
    return tuple(proposals)


def update_missing_slow_binding(
    table: SlowSchemaTable, binding: DreamBinding
) -> bool:
    key = (binding.context_token, binding.prefix_token, binding.suffix_token)
    if binding.provenance != SYNTHETIC_PROVENANCE:
        raise ValueError("missing schema updates require hypothetical provenance")
    if key in table._observed:
        table.observed_overwrite_count += 1
        return False
    if key in table._synthetic:
        return False
    if not table.components.same_component(*key):
        return False
    table._synthetic[key] = SchemaEntry(
        *key,
        standardized_trajectory=np.asarray(binding.standardized_trajectory).copy(),
        provenance=SYNTHETIC_PROVENANCE,
        binding_specific=True,
    )
    return True


@dataclass(frozen=True)
class TrainCalibration:
    standardizer: CoordinateStandardizer
    threshold_pre_48: float
    threshold_post_96: float
    join_threshold: float
    sha256: str


def _initial_confidence(
    store: PersistentEpisodicStore, cue: PartialCue
) -> float:
    if not store.records:
        return -math.inf
    components = infer_cooccurrence_components(store.records)
    if not components.same_component(
        cue.context_token, cue.prefix_token, cue.suffix_token
    ):
        return -math.inf
    traces = np.stack(
        [store.standardizer.transform(item.trajectory).reshape(-1) for item in store.records]
    )
    standardized = store.standardizer.transform(cue.cue_values)
    return float(np.max(_masked_cosines(standardized, cue.cue_mask, traces)))


def _select_threshold(
    positive_confidence: Sequence[float],
    positive_identity_correct: Sequence[bool],
    lure_confidence: Sequence[float],
) -> float:
    positives = np.asarray(positive_confidence, dtype=float)
    correct = np.asarray(positive_identity_correct, dtype=bool)
    lures = np.asarray(lure_confidence, dtype=float)
    candidates = np.concatenate((np.unique(np.concatenate((positives, lures))), [math.inf]))
    best_key: tuple[float, float, float] | None = None
    best_threshold = math.inf
    for threshold in candidates:
        lure_rate = float(np.mean(lures > threshold))
        if lure_rate > 0.025:
            continue
        identity_accuracy = float(np.mean((positives > threshold) & correct))
        key = (identity_accuracy, -lure_rate, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    if best_key is None:
        raise RuntimeError("no registered abstention threshold is feasible")
    return best_threshold


def _calibration_payload(
    standardizer: CoordinateStandardizer,
    threshold_pre: float,
    threshold_post: float,
    join_threshold: float,
) -> dict:
    return {
        "coordinate_mean": standardizer.mean.tolist(),
        "coordinate_scale": standardizer.scale.tolist(),
        "threshold_pre_48": threshold_pre,
        "threshold_post_96": threshold_post,
        "join_threshold": join_threshold,
    }


def _payload_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _train_calibration_artifact_payload(
    calibration: TrainCalibration,
    registration_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict:
    return {
        "schema_version": 1,
        "experiment": "episodic_ltm_dream_factorial_v1",
        "source_split": "train_only",
        "mu": calibration.standardizer.mean.tolist(),
        "sigma": calibration.standardizer.scale.tolist(),
        "tau_pre": calibration.threshold_pre_48,
        "tau_post": calibration.threshold_post_96,
        "join_threshold": calibration.join_threshold,
        "registration_sha256": registration_sha256,
        "implementation_sha256": dict(implementation_sha256),
    }


def _calibration_from_artifact(raw: bytes) -> TrainCalibration:
    payload = json.loads(raw)
    mean = np.asarray(payload["mu"], dtype=float)
    scale = np.asarray(payload["sigma"], dtype=float)
    if mean.shape != (96,) or scale.shape != (96,):
        raise PermissionError("frozen train calibration coordinate shape changed")
    numeric = np.concatenate(
        (
            mean,
            scale,
            np.asarray(
                [payload["tau_pre"], payload["tau_post"], payload["join_threshold"]],
                dtype=float,
            ),
        )
    )
    if not np.all(np.isfinite(numeric)) or np.any(scale < 1e-8):
        raise PermissionError("frozen train calibration is nonfinite or unscaled")
    return TrainCalibration(
        CoordinateStandardizer(mean, scale),
        float(payload["tau_pre"]),
        float(payload["tau_post"]),
        float(payload["join_threshold"]),
        hashlib.sha256(raw).hexdigest(),
    )


def _join_discontinuities(
    records: Sequence[EpisodicRecord], standardizer: CoordinateStandardizer
) -> list[float]:
    table = SlowSchemaTable(records, standardizer)
    values: list[float] = []
    for entry in table.observed_entries.values():
        trajectory = entry.standardized_trajectory
        values.append(
            float(np.linalg.norm(trajectory[4] - trajectory[5]) / math.sqrt(8))
        )
        values.append(
            float(np.linalg.norm(trajectory[6] - trajectory[7]) / math.sqrt(8))
        )
    return values


def calibrate_train_worlds(worlds: Sequence[_SeedWorld]) -> TrainCalibration:
    """Fit the registered calibration from caller-supplied train worlds only."""

    worlds = tuple(worlds)
    if not worlds:
        raise ValueError("train calibration requires at least one generated world")
    all_records = [
        record
        for world in worlds
        for record in (*world.records_a, *world.records_b)
    ]
    standardizer = fit_coordinate_standardizer(all_records)
    pre_positive_confidence: list[float] = []
    post_positive_confidence: list[float] = []
    pre_correct: list[bool] = []
    post_correct: list[bool] = []
    pre_lure_confidence: list[float] = []
    post_lure_confidence: list[float] = []
    joins: list[float] = []
    for world in worlds:
        pre = PersistentEpisodicStore(standardizer, threshold=-math.inf)
        for record in world.records_a:
            pre.insert_real(record)
        positives = [
            _materialize_query(spec, standardizer) for spec in world.recall_specs
        ]
        for query in positives:
            result = pre.recurrent_clamped_recall(query.cue)
            pre_positive_confidence.append(result.confidence)
            pre_correct.append(result.episode_id == query.target_episode_id)
        for spec in world.lure_specs[:24]:
            query = _materialize_query(spec, standardizer)
            pre_lure_confidence.append(_initial_confidence(pre, query.cue))

        post = PersistentEpisodicStore(standardizer, threshold=-math.inf)
        for record in (*world.records_a, *world.records_b):
            post.insert_real(record)
        for query in positives:
            result = post.recurrent_clamped_recall(query.cue)
            post_positive_confidence.append(result.confidence)
            post_correct.append(result.episode_id == query.target_episode_id)
        for spec in world.lure_specs:
            query = _materialize_query(spec, standardizer)
            post_lure_confidence.append(_initial_confidence(post, query.cue))
        joins.extend(
            _join_discontinuities(
                (*world.records_a, *world.records_b), standardizer
            )
        )
    threshold_pre = _select_threshold(
        pre_positive_confidence, pre_correct, pre_lure_confidence
    )
    threshold_post = _select_threshold(
        post_positive_confidence, post_correct, post_lure_confidence
    )
    join_threshold = float(np.quantile(np.asarray(joins), 0.99, method="linear"))
    payload = _calibration_payload(
        standardizer, threshold_pre, threshold_post, join_threshold
    )
    return TrainCalibration(
        standardizer,
        threshold_pre,
        threshold_post,
        join_threshold,
        _payload_sha256(payload),
    )


def _pooled_hidden_nrmse(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> float:
    squared = 0.0
    count = 0
    for prediction, target, mask in zip(predictions, targets, masks):
        hidden = ~np.asarray(mask, dtype=bool)
        difference = np.asarray(prediction)[hidden] - np.asarray(target)[hidden]
        squared += float(difference @ difference)
        count += int(len(difference))
    return float(math.sqrt(squared / count))


def _schema_observed_nrmse(
    records: Sequence[EpisodicRecord], table: SlowSchemaTable
) -> float:
    squared = 0.0
    count = 0
    for record in records:
        entry = table.lookup(
            record.context_token, record.prefix_token, record.suffix_token
        )
        if entry is None:
            raise RuntimeError("observed binding disappeared from slow schema")
        target = table.standardizer.transform(record.trajectory)
        difference = entry.standardized_trajectory - target
        squared += float(np.sum(difference**2))
        count += int(difference.size)
    return float(math.sqrt(squared / count))


def _recall_metrics(
    store: PersistentEpisodicStore | None,
    queries: Sequence[_ScoredQuery],
    table: SlowSchemaTable,
) -> dict[str, float]:
    correct = 0
    accepted = 0
    wrong = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    convergence: list[bool] = []
    extra_stability: list[bool] = []
    clamp_errors: list[float] = []
    for query in queries:
        result = None if store is None else store.recurrent_clamped_recall(query.cue)
        is_accepted = bool(result is not None and result.accepted)
        is_correct = bool(
            is_accepted and result.episode_id == query.target_episode_id
        )
        accepted += int(is_accepted)
        correct += int(is_correct)
        wrong += int(is_accepted and not is_correct)
        if is_accepted:
            prediction = table.standardizer.transform(result.reconstruction)
            convergence.append(result.converged)
            extra_stability.append(result.extra_step_stable)
            clamp_errors.append(result.clamp_max_error)
        else:
            entry = table.lookup(
                query.cue.context_token,
                query.cue.prefix_token,
                query.cue.suffix_token,
            )
            if entry is None:
                raise RuntimeError("positive recall binding missing from slow schema")
            prediction = entry.standardized_trajectory
            if result is not None:
                convergence.append(result.converged)
                extra_stability.append(result.extra_step_stable)
                clamp_errors.append(result.clamp_max_error)
        predictions.append(prediction)
        targets.append(table.standardizer.transform(query.target))
        masks.append(query.cue.cue_mask)
    total = len(queries)
    return {
        "identity_accuracy": correct / total,
        "positive_coverage": accepted / total,
        "accepted_wrong_rate": wrong / total,
        "hidden_nrmse": _pooled_hidden_nrmse(predictions, targets, masks),
        "convergence_rate": float(np.mean(convergence)) if convergence else 1.0,
        "extra_step_stability_rate": (
            float(np.mean(extra_stability)) if extra_stability else 1.0
        ),
        "clamp_max_error": max(clamp_errors, default=0.0),
    }


def evaluate_factorial_seed(
    master_seed: int,
    calibration: TrainCalibration,
    registration: Mapping[str, object],
) -> dict[str, dict[str, float | dict[str, int]]]:
    """Evaluate all four paired cells for one caller-selected master seed."""

    world = _generate_seed_world(int(master_seed))
    standardizer = calibration.standardizer
    positives = [
        _materialize_query(spec, standardizer) for spec in world.recall_specs
    ]
    novel = [_materialize_query(spec, standardizer) for spec in world.novel_specs]
    lures = [_materialize_query(spec, standardizer) for spec in world.lure_specs]
    invalid = [_materialize_query(spec, standardizer) for spec in world.invalid_specs]
    real_records = (*world.records_a, *world.records_b)
    cells = registration["factorial_design"]["cells"]
    result: dict[str, dict[str, float | dict[str, int]]] = {}
    for label in ("M00", "M10", "M01", "M11"):
        cell = cells[label]
        use_ltm = bool(cell["persistent_ltm"])
        use_dream = bool(cell["dream_update"])
        pre_table = SlowSchemaTable(world.records_a, standardizer)
        store: PersistentEpisodicStore | None = None
        if use_ltm:
            store = PersistentEpisodicStore(
                standardizer, threshold=calibration.threshold_pre_48
            )
            for record in world.records_a:
                store.insert_real(record)
        pre = _recall_metrics(store, positives, pre_table)
        if store is not None:
            for record in world.records_b:
                store.insert_real(record)
            store.threshold = calibration.threshold_post_96

        table = SlowSchemaTable(real_records, standardizer)
        observed_hash_before = observed_binding_hash(table)

        proposals: tuple[DreamBinding, ...] = ()
        accepted_dreams = 0
        if use_dream:
            proposals = constrained_missing_binding_dream(
                real_records, standardizer, calibration.join_threshold
            )
            for proposal in proposals:
                accepted_dreams += int(update_missing_slow_binding(table, proposal))
        observed_hash_after = observed_binding_hash(table)
        post = _recall_metrics(store, positives, table)

        lure_false = 0
        if store is not None:
            lure_false = sum(
                store.recurrent_clamped_recall(query.cue).accepted for query in lures
            )
        novel_predictions: list[np.ndarray] = []
        novel_targets: list[np.ndarray] = []
        novel_masks: list[np.ndarray] = []
        provenance_counts: dict[str, int] = {}
        covered = 0
        novel_recalled = 0
        for query in novel:
            entry = table.lookup(
                query.cue.context_token,
                query.cue.prefix_token,
                query.cue.suffix_token,
            )
            if entry is None:
                raise RuntimeError("valid novel query lacks schema fallback")
            covered += int(entry.binding_specific)
            provenance_counts[entry.provenance.source] = (
                provenance_counts.get(entry.provenance.source, 0) + 1
            )
            novel_predictions.append(entry.standardized_trajectory)
            novel_targets.append(standardizer.transform(query.target))
            novel_masks.append(query.cue.cue_mask)
            novel_recalled += int(entry.provenance.recalled)

        invalid_nonabstain = 0
        for query in invalid:
            schema_output = table.lookup(
                query.cue.context_token,
                query.cue.prefix_token,
                query.cue.suffix_token,
            )
            recall_output = (
                None if store is None else store.recurrent_clamped_recall(query.cue)
            )
            invalid_nonabstain += int(
                schema_output is not None
                or (recall_output is not None and recall_output.accepted)
            )

        component_audit = infer_cooccurrence_components(real_records)
        port_violations = sum(
            not component_audit.same_component(
                item.context_token, item.prefix_token, item.suffix_token
            )
            for item in proposals
        )
        context_violations = sum(
            item.prefix_token.split(":", 1)[0] != item.context_token
            or item.suffix_token.split(":", 1)[0] != item.context_token
            for item in proposals
        )
        join_violations = sum(
            item.left_join_rms > calibration.join_threshold
            or item.right_join_rms > calibration.join_threshold
            for item in proposals
        )
        metrics: dict[str, float | dict[str, int]] = {
            "post_old_A_identity_accuracy": post["identity_accuracy"],
            "post_old_A_positive_coverage": post["positive_coverage"],
            "post_old_A_accepted_wrong_rate": post["accepted_wrong_rate"],
            "post_old_A_hidden_nrmse": post["hidden_nrmse"],
            "pre_to_post_identity_drop": pre["identity_accuracy"]
            - post["identity_accuracy"],
            "pre_to_post_hidden_nrmse_increase": post["hidden_nrmse"]
            - pre["hidden_nrmse"],
            "convergence_rate": post["convergence_rate"],
            "extra_step_stability_rate": post["extra_step_stability_rate"],
            "clamp_max_error": post["clamp_max_error"],
            "valid_output_coverage": covered / len(novel),
            "noise_free_base_hidden_nrmse": _pooled_hidden_nrmse(
                novel_predictions, novel_targets, novel_masks
            ),
            "output_provenance": provenance_counts,
            "unstored_lure_false_episode_recall_rate": lure_false / len(lures),
            "novel_valid_tagged_recalled_rate": novel_recalled / len(novel),
            "slow_model_only_old_A_schema_nrmse": _schema_observed_nrmse(
                world.canonical_a, table
            ),
            "current_B_observed_nrmse": _schema_observed_nrmse(
                world.canonical_b, table
            ),
            "observed_binding_overwrite_count": float(table.observed_overwrite_count),
            "observed_binding_hash_change_count": float(
                observed_hash_before != observed_hash_after
            ),
            "accepted_dream_port_violation_count": float(port_violations),
            "accepted_dream_context_violation_count": float(context_violations),
            "accepted_dream_join_violation_count": float(join_violations),
            "accepted_synthetic_bindings": float(accepted_dreams),
            "invalid_query_nonabstain_rate": invalid_nonabstain / len(invalid),
            "synthetic_to_ltm_insert_count": float(
                0 if store is None else store.synthetic_insert_attempts
            ),
            "heldout_target_read_count": 0.0,
            "persistent_observed_items": float(0 if store is None else len(store.records)),
            "persistent_trace_bytes": float(0 if store is None else store.trace_bytes),
        }
        numeric = [value for value in metrics.values() if isinstance(value, float)]
        metrics["nonfinite_metric_or_prediction_count"] = float(
            len(numeric) - np.count_nonzero(np.isfinite(numeric))
        )
        result[label] = metrics
    return result


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _ci(values: Sequence[float], critical: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    center = float(np.mean(array))
    if len(array) < 2:
        return center, center
    margin = float(critical * np.std(array, ddof=1) / math.sqrt(len(array)))
    return center - margin, center + margin


def _aggregate_cells(seed_results: Sequence[dict]) -> tuple[dict, dict]:
    numeric_keys = [
        key
        for key, value in seed_results[0]["M00"].items()
        if isinstance(value, float)
    ]
    means = {
        cell: {
            key: _mean([float(seed[cell][key]) for seed in seed_results])
            for key in numeric_keys
        }
        for cell in ("M00", "M10", "M01", "M11")
    }
    provenance = {
        cell: {
            source: int(
                sum(
                    seed[cell]["output_provenance"].get(source, 0)
                    for seed in seed_results
                )
            )
            for source in ("real", "synthetic", "schema_fallback")
        }
        for cell in ("M00", "M10", "M01", "M11")
    }
    return means, provenance


def _factorial_arrays(seed_results: Sequence[dict], metric: str) -> dict[str, np.ndarray]:
    values = {
        cell: np.asarray([seed[cell][metric] for seed in seed_results], dtype=float)
        for cell in ("M00", "M10", "M01", "M11")
    }
    return {
        "L_main": 0.5 * ((values["M10"] - values["M00"]) + (values["M11"] - values["M01"])),
        "D_main": 0.5 * ((values["M01"] - values["M00"]) + (values["M11"] - values["M10"])),
        "interaction": values["M11"] - values["M10"] - values["M01"] + values["M00"],
    }


def _error_effect_arrays(seed_results: Sequence[dict], metric: str) -> dict[str, np.ndarray]:
    effects = _factorial_arrays(seed_results, metric)
    return {key: -value for key, value in effects.items()}


def _effect_report(values: Mapping[str, np.ndarray], critical: float) -> dict:
    return {
        key: {
            "seed_values": array.tolist(),
            "mean": float(np.mean(array)),
            "ci95_lower": _ci(array, critical)[0],
            "ci95_upper": _ci(array, critical)[1],
        }
        for key, array in values.items()
    }


def _paired_upper(
    seed_results: Sequence[dict], left: str, right: str, metric: str, critical: float
) -> float:
    differences = [
        float(seed[left][metric]) - float(seed[right][metric])
        for seed in seed_results
    ]
    return _ci(differences, critical)[1]


def _build_gate_checks(
    registration: dict,
    seed_results: Sequence[dict],
    means: dict,
    effects: dict,
    critical: float,
) -> dict[str, bool]:
    gate = registration["all_of_gate"]
    recall = gate["ltm_recall"]
    dream = gate["dream_novel_combination"]
    integration = gate["integration_no_antagonism"]
    false_memory = gate["false_memory_and_provenance"]
    integrity = gate["forgetting_schema_and_integrity"]
    checks: dict[str, bool] = {}
    for cell in ("M10", "M11"):
        checks[f"{cell}.post_identity"] = means[cell]["post_old_A_identity_accuracy"] >= recall[
            "M10_and_M11_post_identity_accuracy_min"
        ]
        checks[f"{cell}.post_coverage"] = means[cell]["post_old_A_positive_coverage"] >= recall[
            "M10_and_M11_post_positive_coverage_min"
        ]
        checks[f"{cell}.post_hidden_nrmse"] = means[cell]["post_old_A_hidden_nrmse"] <= recall[
            "M10_and_M11_post_hidden_nrmse_max"
        ]
        checks[f"{cell}.accepted_wrong"] = means[cell]["post_old_A_accepted_wrong_rate"] <= recall[
            "M10_and_M11_post_accepted_wrong_rate_max"
        ]
        checks[f"{cell}.identity_drop"] = means[cell]["pre_to_post_identity_drop"] <= recall[
            "M10_and_M11_pre_to_post_identity_drop_max"
        ]
        checks[f"{cell}.nrmse_increase"] = means[cell]["pre_to_post_hidden_nrmse_increase"] <= recall[
            "M10_and_M11_pre_to_post_hidden_nrmse_increase_max"
        ]
        checks[f"{cell}.convergence"] = means[cell]["convergence_rate"] >= recall[
            "convergence_rate_min"
        ]
        checks[f"{cell}.extra_stability"] = means[cell]["extra_step_stability_rate"] >= recall[
            "extra_step_stability_rate_min"
        ]
        checks[f"{cell}.clamp"] = means[cell]["clamp_max_error"] <= recall[
            "clamp_max_error_max"
        ]
        checks[f"{cell}.lure_false"] = means[cell]["unstored_lure_false_episode_recall_rate"] <= false_memory[
            "M10_and_M11_unstored_lure_false_episode_recall_rate_max"
        ]

    identity_effect = effects["recall_identity"]["L_main"]
    recall_error_effect = effects["recall_error"]["L_main"]
    checks["L_main.identity_gain"] = identity_effect["mean"] >= recall[
        "L_main_identity_gain_min"
    ]
    checks["L_main.identity_ci"] = identity_effect["ci95_lower"] >= recall[
        "L_main_identity_paired_ci95_lower_min"
    ]
    denominator = means["M00"]["post_old_A_hidden_nrmse"] + means["M01"]["post_old_A_hidden_nrmse"]
    if denominator <= 0.0:
        raise RuntimeError("zero LTM NRMSE denominator is hard invalid")
    l_reduction = 1.0 - (
        means["M10"]["post_old_A_hidden_nrmse"]
        + means["M11"]["post_old_A_hidden_nrmse"]
    ) / denominator
    checks["L_main.hidden_reduction"] = l_reduction >= recall[
        "L_main_hidden_nrmse_reduction_min"
    ]
    checks["L_main.hidden_ci"] = recall_error_effect["ci95_lower"] >= recall[
        "L_main_hidden_nrmse_paired_ci95_lower_min"
    ]

    for cell in ("M01", "M11"):
        checks[f"{cell}.novel_coverage"] = means[cell]["valid_output_coverage"] >= dream[
            "M01_and_M11_valid_output_coverage_min"
        ]
        checks[f"{cell}.novel_nrmse"] = means[cell]["noise_free_base_hidden_nrmse"] <= dream[
            "M01_and_M11_hidden_nrmse_max"
        ]
        checks[f"{cell}.synthetic_count"] = all(
            int(seed[cell]["accepted_synthetic_bindings"])
            == int(dream["accepted_synthetic_bindings_per_seed_required"])
            for seed in seed_results
        )
    dream_error_effect = effects["novel_error"]["D_main"]
    dream_coverage_effect = effects["novel_coverage"]["D_main"]
    denominator = means["M00"]["noise_free_base_hidden_nrmse"] + means["M10"]["noise_free_base_hidden_nrmse"]
    if denominator <= 0.0:
        raise RuntimeError("zero dream NRMSE denominator is hard invalid")
    d_reduction = 1.0 - (
        means["M01"]["noise_free_base_hidden_nrmse"]
        + means["M11"]["noise_free_base_hidden_nrmse"]
    ) / denominator
    checks["D_main.hidden_reduction"] = d_reduction >= dream[
        "D_main_hidden_nrmse_reduction_min"
    ]
    checks["D_main.hidden_ci"] = dream_error_effect["ci95_lower"] >= dream[
        "D_main_hidden_nrmse_paired_ci95_lower_min"
    ]
    d_error_values = np.asarray(dream_error_effect["seed_values"])
    checks["D_main.strict_seed_win"] = float(np.mean(d_error_values > 0.0)) >= dream[
        "D_main_strict_seed_win_fraction_min"
    ]
    checks["D_main.coverage_gain"] = dream_coverage_effect["mean"] >= dream[
        "D_main_valid_coverage_gain_min"
    ]
    checks["D_main.coverage_ci"] = dream_coverage_effect["ci95_lower"] >= dream[
        "D_main_valid_coverage_paired_ci95_lower_min"
    ]
    violation_total = max(
        means[cell]["accepted_dream_port_violation_count"]
        + means[cell]["accepted_dream_context_violation_count"]
        + means[cell]["accepted_dream_join_violation_count"]
        for cell in ("M01", "M11")
    )
    checks["dream.constraint_violations"] = violation_total <= dream[
        "accepted_constraint_violation_count_max"
    ]

    checks["integration.recall_identity"] = (
        means["M11"]["post_old_A_identity_accuracy"]
        - means["M10"]["post_old_A_identity_accuracy"]
        >= integration["M11_recall_identity_minus_M10_min"]
    )
    checks["integration.recall_nrmse"] = _paired_upper(
        seed_results, "M11", "M10", "post_old_A_hidden_nrmse", critical
    ) <= integration["M11_minus_M10_recall_nrmse_paired_ci95_upper_max"]
    checks["integration.novel_coverage"] = (
        means["M11"]["valid_output_coverage"]
        - means["M01"]["valid_output_coverage"]
        >= integration["M11_novel_coverage_minus_M01_min"]
    )
    checks["integration.novel_nrmse"] = _paired_upper(
        seed_results, "M11", "M01", "noise_free_base_hidden_nrmse", critical
    ) <= integration["M11_minus_M01_novel_nrmse_paired_ci95_upper_max"]

    for cell in ("M00", "M10", "M01", "M11"):
        checks[f"{cell}.invalid_abstain"] = means[cell]["invalid_query_nonabstain_rate"] <= false_memory[
            "all_cells_invalid_query_nonabstain_rate_max"
        ]
        checks[f"{cell}.novel_not_recalled"] = means[cell]["novel_valid_tagged_recalled_rate"] <= false_memory[
            "all_cells_novel_valid_tagged_recalled_rate_max"
        ]
        checks[f"{cell}.synthetic_not_ltm"] = means[cell]["synthetic_to_ltm_insert_count"] <= false_memory[
            "synthetic_to_ltm_insert_count_max"
        ]
        checks[f"{cell}.no_observed_overwrite"] = means[cell]["observed_binding_overwrite_count"] <= integrity[
            "observed_binding_overwrite_count_max"
        ]
        checks[f"{cell}.observed_hash_stable"] = means[cell]["observed_binding_hash_change_count"] <= integrity[
            "observed_binding_hash_change_count_max"
        ]
        checks[f"{cell}.no_heldout_read"] = means[cell]["heldout_target_read_count"] <= integrity[
            "heldout_target_read_count_max"
        ]
        checks[f"{cell}.finite"] = means[cell]["nonfinite_metric_or_prediction_count"] <= integrity[
            "nonfinite_metric_or_prediction_count_max"
        ]
    for left, right in (("M01", "M00"), ("M11", "M10")):
        checks[f"{left}-{right}.B_schema"] = _paired_upper(
            seed_results, left, right, "current_B_observed_nrmse", critical
        ) <= integrity[
            "dream_minus_matched_no_dream_current_B_nrmse_paired_ci95_upper_max"
        ]
        checks[f"{left}-{right}.A_schema"] = _paired_upper(
            seed_results, left, right, "slow_model_only_old_A_schema_nrmse", critical
        ) <= integrity[
            "dream_minus_matched_no_dream_old_A_slow_schema_nrmse_paired_ci95_upper_max"
        ]
    cross_violations = max(
        means[cell]["accepted_dream_port_violation_count"]
        + means[cell]["accepted_dream_context_violation_count"]
        for cell in ("M01", "M11")
    )
    checks["dream.no_cross_context_component"] = cross_violations <= integrity[
        "cross_context_or_component_dream_accept_count_max"
    ]
    return checks


def _load_registration(config_path: Path) -> tuple[dict, bytes]:
    raw = config_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTERED_CONFIG_SHA256:
        raise PermissionError("G7-M raw preregistration SHA256 changed")
    registration = json.loads(raw)
    if registration.get("runner") != "episodic_ltm_dream_factorial":
        raise ValueError("G7-M factorial registration required")
    if registration.get("status") != "locked_pre_implementation":
        raise ValueError("G7-M registration must remain locked")
    if registration.get("extends") is not None or not registration.get("standalone"):
        raise ValueError("G7-M V1 must remain standalone")
    return registration, raw


def _implementation_hashes(config_path: Path) -> dict[str, str]:
    root = config_path.resolve().parents[2]
    relative = (
        "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge.py",
        "examples/agi/episodic_ltm_dream_bridge_gate.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in relative}


def _train_calibration_artifact_path(
    config_path: Path, registration: Mapping[str, object]
) -> Path:
    root = config_path.resolve().parents[2]
    return root / registration["test_lock"]["train_calibration_artifact"]


def _assert_test_unlocked(
    config_path: Path, registration: dict, config_sha: str
) -> str:
    root = config_path.resolve().parents[2]
    path = root / registration["test_lock"]["validation_artifact"]
    if not path.exists():
        raise PermissionError("G7-M test requires a saved passing validation artifact")
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("experiment") != registration["experiment"]:
        raise PermissionError("G7-M validation experiment changed")
    if report.get("split") != "validation" or report.get("passed") is not True:
        raise PermissionError("G7-M validation all-of gate did not pass")
    if report.get("registration_sha256") != config_sha:
        raise PermissionError("G7-M registration changed after validation")
    if report.get("implementation_sha256") != _implementation_hashes(config_path):
        raise PermissionError("G7-M implementation changed after validation")
    calibration_sha = report.get("train_calibration_sha256")
    if not isinstance(calibration_sha, str) or len(calibration_sha) != 64:
        raise PermissionError("G7-M validation lacks a frozen train calibration")
    calibration_path = _train_calibration_artifact_path(config_path, registration)
    if not calibration_path.exists():
        raise PermissionError("G7-M frozen train calibration artifact is missing")
    calibration_raw = calibration_path.read_bytes()
    if hashlib.sha256(calibration_raw).hexdigest() != calibration_sha:
        raise PermissionError("G7-M frozen train calibration bytes changed")
    payload = json.loads(calibration_raw)
    if payload.get("registration_sha256") != config_sha:
        raise PermissionError("G7-M train calibration registration lock changed")
    if payload.get("implementation_sha256") != _implementation_hashes(config_path):
        raise PermissionError("G7-M train calibration implementation lock changed")
    return calibration_sha


def run_episodic_ltm_dream_gate(
    config_path: Path, *, split: str = "validation"
) -> dict:
    """Run the registered split without opening test before validation passes."""

    started = time.perf_counter()
    registration, raw = _load_registration(config_path)
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    config_sha = hashlib.sha256(raw).hexdigest()
    expected_calibration_sha = None
    validation_artifact_sha = None
    if split == "test":
        expected_calibration_sha = _assert_test_unlocked(
            config_path, registration, config_sha
        )
        validation_path = (
            config_path.resolve().parents[2]
            / registration["test_lock"]["validation_artifact"]
        )
        validation_artifact_sha = hashlib.sha256(validation_path.read_bytes()).hexdigest()

    implementation_sha = _implementation_hashes(config_path)
    calibration_path = _train_calibration_artifact_path(config_path, registration)
    if split == "validation":
        if calibration_path.exists():
            calibration_raw = calibration_path.read_bytes()
            calibration_payload = json.loads(calibration_raw)
            if calibration_payload.get("registration_sha256") != config_sha:
                raise PermissionError(
                    "existing train calibration registration lock changed"
                )
            if calibration_payload.get("implementation_sha256") != implementation_sha:
                raise PermissionError(
                    "existing train calibration implementation lock changed"
                )
            calibration = _calibration_from_artifact(calibration_raw)
            train_seed_count = 0
        else:
            train_worlds = [
                _generate_seed_world(int(seed))
                for seed in registration["data_roles"]["train"]["seeds"]
            ]
            calibration = calibrate_train_worlds(train_worlds)
            calibration_payload = _train_calibration_artifact_payload(
                calibration, config_sha, implementation_sha
            )
            _write_json_lf(calibration_path, calibration_payload)
            calibration_raw = calibration_path.read_bytes()
            calibration = _calibration_from_artifact(calibration_raw)
            train_seed_count = len(train_worlds)
    else:
        calibration_raw = calibration_path.read_bytes()
        if hashlib.sha256(calibration_raw).hexdigest() != expected_calibration_sha:
            raise PermissionError("G7-M train calibration changed after validation")
        calibration = _calibration_from_artifact(calibration_raw)
        train_seed_count = 0

    role = registration["data_roles"][split]
    seed_results = [
        evaluate_factorial_seed(int(seed), calibration, registration)
        for seed in role["seeds"]
    ]
    means, provenance = _aggregate_cells(seed_results)
    critical = float(
        registration["paired_inference"][
            "validation_critical_value_n40"
            if split == "validation"
            else "test_critical_value_n60"
        ]
    )
    effects = {
        "recall_identity": _effect_report(
            _factorial_arrays(seed_results, "post_old_A_identity_accuracy"), critical
        ),
        "recall_error": _effect_report(
            _error_effect_arrays(seed_results, "post_old_A_hidden_nrmse"), critical
        ),
        "novel_coverage": _effect_report(
            _factorial_arrays(seed_results, "valid_output_coverage"), critical
        ),
        "novel_error": _effect_report(
            _error_effect_arrays(seed_results, "noise_free_base_hidden_nrmse"), critical
        ),
    }
    checks = _build_gate_checks(
        registration, seed_results, means, effects, critical
    )
    elapsed = time.perf_counter() - started
    resources = registration["resources"]
    resource_checks = {
        "ltm_cells_have_96_observed_items": all(
            int(seed[cell]["persistent_observed_items"])
            == int(resources["persistent_observed_items_M10_M11"])
            for seed in seed_results
            for cell in ("M10", "M11")
        ),
        "no_ltm_cells_have_zero_items": all(
            int(seed[cell]["persistent_observed_items"])
            == int(resources["persistent_observed_items_M00_M01"])
            for seed in seed_results
            for cell in ("M00", "M01")
        ),
        "persistent_trace_bytes": all(
            float(seed[cell]["persistent_trace_bytes"])
            <= float(resources["persistent_observed_trace_bytes_max"])
            for seed in seed_results
            for cell in ("M10", "M11")
        ),
        "zero_download": int(resources["external_download_bytes"]) == 0,
        "zero_raw_trajectory_files": not bool(resources["write_raw_trajectory_files"]),
        "numpy_cpu": resources["backend"] == "numpy_cpu",
    }
    report = {
        "experiment": registration["experiment"],
        "roadmap_stage": registration["roadmap_stage"],
        "split": split,
        "registration_sha256": config_sha,
        "implementation_sha256": implementation_sha,
        "train_calibration": _calibration_payload(
            calibration.standardizer,
            calibration.threshold_pre_48,
            calibration.threshold_post_96,
            calibration.join_threshold,
        ),
        "train_calibration_sha256": calibration.sha256,
        "test_lock": {
            "validation_artifact_sha256": validation_artifact_sha,
            "test_opened_after_validation_pass": split == "test",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "cell_means": means,
        "novel_output_provenance_counts": provenance,
        "factorial_effects": effects,
        "seed_results": seed_results,
        "checks": checks,
        "performance_passed": bool(all(checks.values())),
        "resource_checks": resource_checks,
        "resource_passed": bool(all(resource_checks.values())),
        "resource_usage": {
            "wall_seconds": elapsed,
            "external_download_bytes": 0,
            "raw_trajectory_files_written": 0,
            "train_seeds": train_seed_count,
            "evaluation_seeds": len(seed_results),
            "backend": "numpy_cpu",
        },
    }
    report["passed"] = bool(
        report["performance_passed"] and report["resource_passed"]
    )
    return report


def _default_output(config_path: Path, split: str, registration: dict) -> Path:
    root = config_path.resolve().parents[2]
    key = "validation_artifact" if split == "validation" else "test_artifact"
    return root / registration["test_lock"][key]


def _write_json_lf(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    registration, _ = _load_registration(args.config)
    report = run_episodic_ltm_dream_gate(args.config, split=args.split)
    output = args.output or _default_output(args.config, args.split, registration)
    _write_json_lf(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"artifact: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
