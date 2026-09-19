"""Locked G7-M/V2 hard episodic reinstatement and dream factorial gate."""

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

from reality_stone.clarus import episodic_ltm_dream_bridge as v1


REGISTERED_CONFIG_SHA256 = (
    "973e90111ee98862a5c9ffc3f86509b46ee4e263b5a977e7e1504e00109092b9"
)
_V1_REGISTERED_SEEDS = frozenset(
    (*range(77100, 77140), *range(78100, 78140), *range(79100, 79160))
)

# V2 deliberately imports the immutable V1 generator, store, schema, and dream
# implementation.  These aliases make that dependency explicit and give the
# off-range equivalence tests one surface to inspect.
Provenance = v1.Provenance
EpisodicRecord = v1.EpisodicRecord
PartialCue = v1.PartialCue
RecallResult = v1.RecallResult
CoordinateStandardizer = v1.CoordinateStandardizer
SchemaEntry = v1.SchemaEntry
DreamBinding = v1.DreamBinding
CooccurrenceComponents = v1.CooccurrenceComponents
SlowSchemaTable = v1.SlowSchemaTable

REAL_PROVENANCE = v1.REAL_PROVENANCE
SYNTHETIC_PROVENANCE = v1.SYNTHETIC_PROVENANCE
FALLBACK_PROVENANCE = v1.FALLBACK_PROVENANCE
RECALLED_PROVENANCE = v1.RECALLED_PROVENANCE

fit_coordinate_standardizer = v1.fit_coordinate_standardizer
infer_cooccurrence_components = v1.infer_cooccurrence_components
constrained_missing_binding_dream = v1.constrained_missing_binding_dream
update_missing_slow_binding = v1.update_missing_slow_binding
observed_binding_hash = v1.observed_binding_hash

_generate_seed_world = v1._generate_seed_world
_materialize_query = v1._materialize_query


class PersistentEpisodicStore(v1.PersistentEpisodicStore):
    """Immutable V1 real-only store with the registered V2 recall entry point."""

    def hard_cue_anchored_recall(self, cue: PartialCue) -> RecallResult:
        return hard_cue_anchored_recall(self, cue)


def _empty_recall() -> RecallResult:
    return RecallResult(
        accepted=False,
        episode_id=None,
        reconstruction=np.zeros((12, 8), dtype=float),
        confidence=-math.inf,
        iterations=0,
        converged=True,
        extra_step_stable=True,
        clamp_max_error=0.0,
        provenance=FALLBACK_PROVENANCE,
    )


def hard_cue_anchored_recall(
    store: PersistentEpisodicStore, cue: PartialCue
) -> RecallResult:
    """Complete hidden coordinates from the exact initial masked-cosine winner."""

    values = np.asarray(cue.cue_values)
    mask = np.asarray(cue.cue_mask, dtype=bool)
    if values.shape != (12, 8) or mask.shape != (12, 8):
        raise ValueError("cue values and mask must have shape (12, 8)")
    if not store.records:
        return _empty_recall()
    components = infer_cooccurrence_components(store.records)
    if not components.same_component(
        cue.context_token, cue.prefix_token, cue.suffix_token
    ):
        return _empty_recall()

    observed = mask.reshape(-1)
    values_flat = values.reshape(-1)
    mean = np.asarray(store.standardizer.mean, dtype=float)
    scale = np.maximum(np.asarray(store.standardizer.scale, dtype=float), 1e-8)
    # Do not transform, validate, or otherwise inspect values outside the mask.
    standardized_observed = (
        np.asarray(values_flat[observed], dtype=float) - mean[observed]
    ) / scale[observed]
    if not np.all(np.isfinite(standardized_observed)):
        raise ValueError("observed cue coordinates must be finite")

    traces = np.stack(
        [
            (np.asarray(record.trajectory, dtype=float).reshape(-1) - mean) / scale
            for record in store.records
        ]
    )
    observed_traces = traces[:, observed]
    numerator = observed_traces @ standardized_observed
    denominator = np.linalg.norm(observed_traces, axis=1) * np.linalg.norm(
        standardized_observed
    )
    scores = numerator / np.maximum(denominator, 1e-12)
    winner = int(np.argmax(scores))
    confidence = float(scores[winner])

    state = traces[winner].copy()
    state[observed] = standardized_observed
    clamp_error = float(
        np.max(np.abs(state[observed] - standardized_observed), initial=0.0)
    )
    accepted = bool(confidence > store.threshold)
    reconstruction = np.zeros((12, 8), dtype=float)
    if accepted:
        reconstruction = np.asarray(
            store.records[winner].trajectory, dtype=float
        ).copy()
        reconstruction.reshape(-1)[observed] = values_flat[observed]
    return RecallResult(
        accepted=accepted,
        episode_id=store.records[winner].episode_id if accepted else None,
        reconstruction=reconstruction,
        confidence=confidence,
        iterations=1,
        converged=True,
        extra_step_stable=True,
        clamp_max_error=clamp_error,
        provenance=RECALLED_PROVENANCE if accepted else FALLBACK_PROVENANCE,
    )


def frozen_v1_soft_recall(
    store: PersistentEpisodicStore, cue: PartialCue
) -> RecallResult:
    """Call the byte-locked V1 soft recurrent comparator without modification."""

    return v1.recurrent_clamped_recall(store, cue)


@dataclass(frozen=True)
class TrainCalibrationV2:
    standardizer: CoordinateStandardizer
    v2_threshold_pre_48: float
    v2_threshold_post_96: float
    v1_threshold_pre_48: float
    v1_threshold_post_96: float
    join_threshold: float
    comparator_equivalence_sha256: str
    sha256: str
    off_range_shared_equivalence_sha256: str = ""


def _calibration_values(calibration: TrainCalibrationV2) -> dict[str, object]:
    return {
        "coordinate_mean": calibration.standardizer.mean.tolist(),
        "coordinate_scale": calibration.standardizer.scale.tolist(),
        "v2_threshold_pre_48": calibration.v2_threshold_pre_48,
        "v2_threshold_post_96": calibration.v2_threshold_post_96,
        "v1_threshold_pre_48": calibration.v1_threshold_pre_48,
        "v1_threshold_post_96": calibration.v1_threshold_post_96,
        "join_threshold": calibration.join_threshold,
        "frozen_v1_comparator_equivalence_sha256": (
            calibration.comparator_equivalence_sha256
        ),
        "off_range_shared_equivalence_sha256": (
            calibration.off_range_shared_equivalence_sha256
        ),
    }


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hard_threshold_pools(
    worlds: Sequence[object], standardizer: CoordinateStandardizer
) -> tuple[list[float], list[bool], list[float], list[float], list[bool], list[float]]:
    pre_confidence: list[float] = []
    pre_correct: list[bool] = []
    pre_lure: list[float] = []
    post_confidence: list[float] = []
    post_correct: list[bool] = []
    post_lure: list[float] = []
    for world in worlds:
        positives = [
            _materialize_query(specification, standardizer)
            for specification in world.recall_specs
        ]
        pre = PersistentEpisodicStore(standardizer, threshold=-math.inf)
        for record in world.records_a:
            pre.insert_real(record)
        for query in positives:
            result = hard_cue_anchored_recall(pre, query.cue)
            pre_confidence.append(result.confidence)
            pre_correct.append(result.episode_id == query.target_episode_id)
        for specification in world.lure_specs[:24]:
            query = _materialize_query(specification, standardizer)
            pre_lure.append(hard_cue_anchored_recall(pre, query.cue).confidence)

        post = PersistentEpisodicStore(standardizer, threshold=-math.inf)
        for record in (*world.records_a, *world.records_b):
            post.insert_real(record)
        for query in positives:
            result = hard_cue_anchored_recall(post, query.cue)
            post_confidence.append(result.confidence)
            post_correct.append(result.episode_id == query.target_episode_id)
        for specification in world.lure_specs:
            query = _materialize_query(specification, standardizer)
            post_lure.append(hard_cue_anchored_recall(post, query.cue).confidence)
    return (
        pre_confidence,
        pre_correct,
        pre_lure,
        post_confidence,
        post_correct,
        post_lure,
    )


def calibrate_train_worlds_v2(
    worlds: Sequence[object],
    *,
    comparator_equivalence_sha256: str = "",
    off_range_shared_equivalence_sha256: str = "",
) -> TrainCalibrationV2:
    """Fit common coordinates, separate V2/V1 thresholds, and the V1 dream join."""

    worlds = tuple(worlds)
    if not worlds:
        raise ValueError("train calibration requires at least one generated world")
    v1_calibration = v1.calibrate_train_worlds(worlds)
    standardizer = v1_calibration.standardizer
    pools = _hard_threshold_pools(worlds, standardizer)
    v2_pre = v1._select_threshold(pools[0], pools[1], pools[2])
    v2_post = v1._select_threshold(pools[3], pools[4], pools[5])
    values = {
        "coordinate_mean": standardizer.mean.tolist(),
        "coordinate_scale": standardizer.scale.tolist(),
        "v2_threshold_pre_48": v2_pre,
        "v2_threshold_post_96": v2_post,
        "v1_threshold_pre_48": v1_calibration.threshold_pre_48,
        "v1_threshold_post_96": v1_calibration.threshold_post_96,
        "join_threshold": v1_calibration.join_threshold,
        "frozen_v1_comparator_equivalence_sha256": comparator_equivalence_sha256,
        "off_range_shared_equivalence_sha256": off_range_shared_equivalence_sha256,
    }
    return TrainCalibrationV2(
        standardizer=standardizer,
        v2_threshold_pre_48=v2_pre,
        v2_threshold_post_96=v2_post,
        v1_threshold_pre_48=v1_calibration.threshold_pre_48,
        v1_threshold_post_96=v1_calibration.threshold_post_96,
        join_threshold=v1_calibration.join_threshold,
        comparator_equivalence_sha256=comparator_equivalence_sha256,
        off_range_shared_equivalence_sha256=(
            off_range_shared_equivalence_sha256
        ),
        sha256=_canonical_sha256(values),
    )


def _f64le_hex(value: float) -> str:
    return np.asarray(value, dtype="<f8").tobytes().hex()


def _array_f64le_sha256(value: np.ndarray) -> str:
    raw = np.ascontiguousarray(value, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _same_recall_result(left: RecallResult, right: RecallResult) -> bool:
    return bool(
        left.accepted == right.accepted
        and left.episode_id == right.episode_id
        and left.confidence == right.confidence
        and left.iterations == right.iterations
        and left.converged == right.converged
        and left.extra_step_stable == right.extra_step_stable
        and left.clamp_max_error == right.clamp_max_error
        and left.provenance == right.provenance
        and np.array_equal(left.reconstruction, right.reconstruction)
    )


def _equivalence_result_record(
    seed: int,
    stage: str,
    family: str,
    query_index: int,
    result: RecallResult,
) -> dict[str, object]:
    return {
        "seed": int(seed),
        "stage": stage,
        "family": family,
        "query_index": int(query_index),
        "accepted": bool(result.accepted),
        "episode_id_or_null": result.episode_id,
        "provenance_source": result.provenance.source,
        "provenance_epistemic_status": result.provenance.epistemic_status,
        "provenance_observed": bool(result.provenance.observed),
        "provenance_recalled": bool(result.provenance.recalled),
        "confidence_f64le_hex": _f64le_hex(result.confidence),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "extra_step_stable": bool(result.extra_step_stable),
        "clamp_max_error_f64le_hex": _f64le_hex(result.clamp_max_error),
        "returned_reconstruction_f64le_c_order_sha256": (
            _array_f64le_sha256(result.reconstruction)
        ),
    }


def frozen_v1_comparator_equivalence(
    registration: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    """Build the exact preregistered off-range V1 comparator golden payload."""

    recipe = registration["frozen_v1_soft_comparator"]["equivalence_hash_recipe"]
    seeds = tuple(int(seed) for seed in recipe["off_range_seeds_in_order"])
    worlds = [_generate_seed_world(seed) for seed in seeds]
    records = [
        record
        for world in worlds
        for record in (*world.records_a, *world.records_b)
    ]
    standardizer = fit_coordinate_standardizer(records)
    fingerprint_bytes = (
        np.ascontiguousarray(standardizer.mean, dtype="<f8").tobytes(order="C")
        + np.ascontiguousarray(standardizer.scale, dtype="<f8").tobytes(order="C")
    )
    result_records: list[dict[str, object]] = []
    threshold = float(recipe["threshold_for_equivalence_only"])
    for seed, world in zip(seeds, worlds):
        positives = [
            _materialize_query(specification, standardizer)
            for specification in world.recall_specs
        ]
        stages = (
            ("pre48", world.records_a, world.lure_specs[:24]),
            ("post96", (*world.records_a, *world.records_b), world.lure_specs),
        )
        for stage, stage_records, lure_specs in stages:
            store = PersistentEpisodicStore(standardizer, threshold=threshold)
            for record in stage_records:
                store.insert_real(record)
            families = (
                ("positive", positives),
                (
                    "lure",
                    [
                        _materialize_query(specification, standardizer)
                        for specification in lure_specs
                    ],
                ),
            )
            for family, queries in families:
                for query_index, query in enumerate(queries):
                    direct = v1.recurrent_clamped_recall(store, query.cue)
                    wrapped = frozen_v1_soft_recall(store, query.cue)
                    if not _same_recall_result(direct, wrapped):
                        raise RuntimeError("frozen V1 comparator wrapper changed output")
                    result_records.append(
                        _equivalence_result_record(
                            seed, stage, family, query_index, direct
                        )
                    )
    dependencies = dict(
        registration["test_lock"]["require_immutable_v1_dependency_sha256"]
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "frozen_v1_dependency_hashes": dependencies,
        "off_range_seeds": list(seeds),
        "standardizer_fingerprint": hashlib.sha256(
            fingerprint_bytes
        ).hexdigest(),
        "threshold_for_equivalence_only": threshold,
        "result_records": result_records,
    }
    return _canonical_sha256(payload), payload


def _digest_token(digest: object, value: object) -> None:
    digest.update(str(value).encode("utf-8"))
    digest.update(b"\0")


def _digest_array(digest: object, value: np.ndarray) -> None:
    array = np.asarray(value)
    _digest_token(digest, array.shape)
    _digest_token(digest, array.dtype.str)
    if array.dtype == np.bool_:
        digest.update(np.ascontiguousarray(array, dtype=np.uint8).tobytes(order="C"))
    else:
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes(order="C"))


def _world_fingerprint(world: object) -> str:
    digest = hashlib.sha256()
    for family in ("records_a", "records_b", "canonical_a", "canonical_b"):
        _digest_token(digest, family)
        for record in getattr(world, family):
            for value in (
                record.episode_id,
                record.context_token,
                record.prefix_token,
                record.suffix_token,
                record.provenance,
            ):
                _digest_token(digest, value)
            _digest_array(digest, record.trajectory)
    _digest_token(digest, "observed_bases")
    for key in sorted(world.observed_bases):
        _digest_token(digest, key)
        _digest_array(digest, world.observed_bases[key])
    for family in ("recall_specs", "novel_specs", "lure_specs", "invalid_specs"):
        _digest_token(digest, family)
        for specification in getattr(world, family):
            for value in (
                specification.context_token,
                specification.prefix_token,
                specification.suffix_token,
                specification.target_episode_id,
            ):
                _digest_token(digest, value)
            _digest_array(digest, specification.target)
            _digest_array(digest, specification.noise)
            for visible in sorted(specification.masks):
                _digest_token(digest, visible)
                _digest_array(digest, specification.masks[visible])
    return digest.hexdigest()


def _components_fingerprint(components: CooccurrenceComponents) -> str:
    payload = {
        "prefix": [
            [context, token, int(component)]
            for (context, token), component in sorted(
                components.prefix_component.items()
            )
        ],
        "suffix": [
            [context, token, int(component)]
            for (context, token), component in sorted(
                components.suffix_component.items()
            )
        ],
    }
    return _canonical_sha256(payload)


def _dream_fingerprint(bindings: Sequence[DreamBinding]) -> str:
    digest = hashlib.sha256()
    for binding in bindings:
        for value in (
            binding.context_token,
            binding.prefix_token,
            binding.suffix_token,
            binding.provenance,
        ):
            _digest_token(digest, value)
        _digest_array(digest, binding.standardized_trajectory)
        digest.update(np.asarray(binding.left_join_rms, dtype="<f8").tobytes())
        digest.update(np.asarray(binding.right_join_rms, dtype="<f8").tobytes())
    return digest.hexdigest()


def off_range_shared_equivalence(
    registration: Mapping[str, object],
    *,
    comparator_equivalence_sha256: str | None = None,
) -> tuple[str, dict[str, object]]:
    """Audit every preregistered V1-shared path on development-only seeds."""

    seeds = tuple(
        int(seed)
        for seed in registration["implementation_equivalence_before_registered_train"][
            "off_range_seeds"
        ]
    )
    v2_registered = {
        int(seed)
        for role in ("train", "validation", "test")
        for seed in registration["data_roles"][role]["seeds"]
    }
    registered_seed_count = len(
        set(seeds) & (set(_V1_REGISTERED_SEEDS) | v2_registered)
    )
    if registered_seed_count:
        raise PermissionError("registered seed entered prelock equivalence")
    v1_worlds = [v1._generate_seed_world(seed) for seed in seeds]
    v2_worlds = [_generate_seed_world(seed) for seed in seeds]
    world_pairs = [
        (_world_fingerprint(left), _world_fingerprint(right))
        for left, right in zip(v1_worlds, v2_worlds)
    ]
    v1_records = [
        record
        for world in v1_worlds
        for record in (*world.records_a, *world.records_b)
    ]
    v2_records = [
        record
        for world in v2_worlds
        for record in (*world.records_a, *world.records_b)
    ]
    v1_standardizer = v1.fit_coordinate_standardizer(v1_records)
    v2_standardizer = fit_coordinate_standardizer(v2_records)
    standardizer_pair = (
        _array_f64le_sha256(
            np.concatenate((v1_standardizer.mean, v1_standardizer.scale))
        ),
        _array_f64le_sha256(
            np.concatenate((v2_standardizer.mean, v2_standardizer.scale))
        ),
    )
    per_seed: list[dict[str, object]] = []
    for seed, left, right, world_pair in zip(
        seeds, v1_worlds, v2_worlds, world_pairs
    ):
        left_records = (*left.records_a, *left.records_b)
        right_records = (*right.records_a, *right.records_b)
        left_components = _components_fingerprint(
            v1.infer_cooccurrence_components(left_records)
        )
        right_components = _components_fingerprint(
            infer_cooccurrence_components(right_records)
        )
        left_table = v1.SlowSchemaTable(left_records, v1_standardizer)
        right_table = SlowSchemaTable(right_records, v2_standardizer)
        left_schema = v1.observed_binding_hash(left_table)
        right_schema = observed_binding_hash(right_table)
        left_dream = _dream_fingerprint(
            v1.constrained_missing_binding_dream(
                left_records, v1_standardizer, math.inf
            )
        )
        right_dream = _dream_fingerprint(
            constrained_missing_binding_dream(
                right_records, v2_standardizer, math.inf
            )
        )
        per_seed.append(
            {
                "seed": seed,
                "world_v1_sha256": world_pair[0],
                "world_v2_sha256": world_pair[1],
                "world_and_queries_equal": world_pair[0] == world_pair[1],
                "components_v1_sha256": left_components,
                "components_v2_sha256": right_components,
                "components_equal": left_components == right_components,
                "slow_schema_v1_sha256": left_schema,
                "slow_schema_v2_sha256": right_schema,
                "slow_schema_equal": left_schema == right_schema,
                "dream_v1_sha256": left_dream,
                "dream_v2_sha256": right_dream,
                "dream_equal": left_dream == right_dream,
            }
        )

    if comparator_equivalence_sha256 is None:
        comparator_equivalence_sha256, _ = frozen_v1_comparator_equivalence(
            registration
        )
    calibration = calibrate_train_worlds_v2(
        v2_worlds,
        comparator_equivalence_sha256=comparator_equivalence_sha256,
    )
    v1_calibration = v1.TrainCalibration(
        standardizer=calibration.standardizer,
        threshold_pre_48=calibration.v1_threshold_pre_48,
        threshold_post_96=calibration.v1_threshold_post_96,
        join_threshold=calibration.join_threshold,
        sha256="off-range-shared-equivalence",
    )
    for item in per_seed:
        evaluated = evaluate_factorial_seed_v2(
            int(item["seed"]), calibration, registration
        )
        v1_cells = v1.evaluate_factorial_seed(
            int(item["seed"]), v1_calibration, registration
        )
        item.update(_compare_shared_cells_off_range(evaluated["cells"], v1_cells))

    report: dict[str, object] = {
        "schema_version": 1,
        "recipe_identifier": "v1_shared_paths_v2_recipe_1",
        "scope": "prelock_off_range_comprehensive",
        "off_range_seeds": list(seeds),
        "registered_seed_used_for_prelock_equivalence": registered_seed_count,
        "registered_seed_count": registered_seed_count,
        "standardizer_v1_sha256": standardizer_pair[0],
        "standardizer_v2_sha256": standardizer_pair[1],
        "standardizer_input_and_values_equal": (
            standardizer_pair[0] == standardizer_pair[1]
        ),
        "per_seed": per_seed,
    }
    required = [
        bool(report["standardizer_input_and_values_equal"]),
        *[
            bool(item[key])
            for item in per_seed
            for key in (
                "world_and_queries_equal",
                "components_equal",
                "slow_schema_equal",
                "dream_equal",
                "M00_shared_metrics_equal",
                "M01_shared_metrics_equal",
                "M10_nonrecall_metrics_equal",
                "M11_nonrecall_metrics_equal",
            )
        ],
    ]
    report["all_required_equal"] = bool(all(required))
    report["all_passed"] = report["all_required_equal"]
    return _canonical_sha256(report), report


def _hard_recall_metrics(
    store: PersistentEpisodicStore | None,
    queries: Sequence[object],
    table: SlowSchemaTable,
) -> dict[str, float]:
    correct = 0
    accepted = 0
    wrong = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    accepted_idempotent: list[bool] = []
    repeated_stable: list[bool] = []
    hidden_deltas: list[float] = []
    clamp_errors: list[float] = []
    for query in queries:
        result = None if store is None else hard_cue_anchored_recall(store, query.cue)
        is_accepted = bool(result is not None and result.accepted)
        is_correct = bool(
            is_accepted and result.episode_id == query.target_episode_id
        )
        accepted += int(is_accepted)
        correct += int(is_correct)
        wrong += int(is_accepted and not is_correct)
        if result is not None:
            repeated = hard_cue_anchored_recall(store, query.cue)
            first_standardized = table.standardizer.transform(result.reconstruction)
            repeated_standardized = table.standardizer.transform(
                repeated.reconstruction
            )
            hidden = ~np.asarray(query.cue.cue_mask, dtype=bool)
            hidden_delta = float(
                np.sqrt(
                    np.mean(
                        (
                            first_standardized[hidden]
                            - repeated_standardized[hidden]
                        )
                        ** 2
                    )
                )
            )
            hidden_deltas.append(hidden_delta)
            repeated_stable.append(_same_recall_result(result, repeated))
            if is_accepted:
                accepted_idempotent.append(
                    bool(
                        result.episode_id == repeated.episode_id
                        and hidden_delta <= 1e-12
                    )
                )
            clamp_errors.append(result.clamp_max_error)
        if is_accepted:
            prediction = table.standardizer.transform(result.reconstruction)
        else:
            entry = table.lookup(
                query.cue.context_token,
                query.cue.prefix_token,
                query.cue.suffix_token,
            )
            if entry is None:
                raise RuntimeError("positive recall binding missing from slow schema")
            prediction = entry.standardized_trajectory
        predictions.append(prediction)
        targets.append(table.standardizer.transform(query.target))
        masks.append(query.cue.cue_mask)
    total = len(queries)
    return {
        "identity_accuracy": correct / total,
        "positive_coverage": accepted / total,
        "accepted_wrong_rate": wrong / total,
        "hidden_nrmse": v1._pooled_hidden_nrmse(predictions, targets, masks),
        "one_step_idempotence_rate": (
            float(np.mean(accepted_idempotent)) if accepted_idempotent else 1.0
        ),
        "repeat_identity_stability_rate": (
            float(np.mean(repeated_stable)) if repeated_stable else 1.0
        ),
        "hidden_idempotence_rms_max": max(hidden_deltas, default=0.0),
        "clamp_max_error": max(clamp_errors, default=0.0),
    }


_SOFT_ONLY_CELL_FIELDS = {"convergence_rate", "extra_step_stability_rate"}
_RECALL_OR_LURE_FIELDS = {
    "post_old_A_identity_accuracy",
    "post_old_A_positive_coverage",
    "post_old_A_accepted_wrong_rate",
    "post_old_A_hidden_nrmse",
    "pre_to_post_identity_drop",
    "pre_to_post_hidden_nrmse_increase",
    "convergence_rate",
    "extra_step_stability_rate",
    "clamp_max_error",
    "unstored_lure_false_episode_recall_rate",
    "nonfinite_metric_or_prediction_count",
}


def _v2_cell_from_v1(value: Mapping[str, object]) -> dict[str, object]:
    result = {
        key: item for key, item in value.items() if key not in _SOFT_ONLY_CELL_FIELDS
    }
    result.update(
        {
            "one_step_idempotence_rate": 1.0,
            "repeat_identity_stability_rate": 1.0,
            "hidden_idempotence_rms_max": 0.0,
        }
    )
    return result


def _same_values_on_keys(
    left: Mapping[str, object], right: Mapping[str, object], keys: set[str]
) -> bool:
    return all(left[key] == right[key] for key in keys)


def _compare_shared_cells_off_range(
    cells: Mapping[str, Mapping[str, object]],
    v1_cells: Mapping[str, Mapping[str, object]],
) -> dict[str, bool]:
    """Compare shared outputs only inside the prelock off-range harness."""

    shared_no_ltm_keys = set(v1_cells["M00"]) - _SOFT_ONLY_CELL_FIELDS
    independent_ltm_keys = set(v1_cells["M10"]) - _RECALL_OR_LURE_FIELDS
    return {
        "M00_shared_metrics_equal": _same_values_on_keys(
            cells["M00"], v1_cells["M00"], shared_no_ltm_keys
        ),
        "M01_shared_metrics_equal": _same_values_on_keys(
            cells["M01"], v1_cells["M01"], shared_no_ltm_keys
        ),
        "M10_nonrecall_metrics_equal": _same_values_on_keys(
            cells["M10"], v1_cells["M10"], independent_ltm_keys
        ),
        "M11_nonrecall_metrics_equal": _same_values_on_keys(
            cells["M11"], v1_cells["M11"], independent_ltm_keys
        ),
    }


def evaluate_factorial_seed_v2(
    master_seed: int,
    calibration: TrainCalibrationV2,
    registration: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the four V2 cells and the frozen V1 comparator on one world."""

    v1_calibration = v1.TrainCalibration(
        standardizer=calibration.standardizer,
        threshold_pre_48=calibration.v1_threshold_pre_48,
        threshold_post_96=calibration.v1_threshold_post_96,
        join_threshold=calibration.join_threshold,
        sha256="fresh-v2-world-comparator",
    )
    v1_cells = v1.evaluate_factorial_seed(
        int(master_seed), v1_calibration, registration
    )
    cells = {label: _v2_cell_from_v1(v1_cells[label]) for label in v1_cells}

    world = _generate_seed_world(int(master_seed))
    standardizer = calibration.standardizer
    positives = [
        _materialize_query(specification, standardizer)
        for specification in world.recall_specs
    ]
    lures = [
        _materialize_query(specification, standardizer)
        for specification in world.lure_specs
    ]
    pre_table = SlowSchemaTable(world.records_a, standardizer)
    post_table = SlowSchemaTable(
        (*world.records_a, *world.records_b), standardizer
    )
    hard_pre = PersistentEpisodicStore(
        standardizer, threshold=calibration.v2_threshold_pre_48
    )
    for record in world.records_a:
        hard_pre.insert_real(record)
    pre = _hard_recall_metrics(hard_pre, positives, pre_table)
    for record in world.records_b:
        hard_pre.insert_real(record)
    hard_pre.threshold = calibration.v2_threshold_post_96
    post = _hard_recall_metrics(hard_pre, positives, post_table)
    lure_false = sum(
        hard_cue_anchored_recall(hard_pre, query.cue).accepted for query in lures
    ) / len(lures)

    hard_updates = {
        "post_old_A_identity_accuracy": post["identity_accuracy"],
        "post_old_A_positive_coverage": post["positive_coverage"],
        "post_old_A_accepted_wrong_rate": post["accepted_wrong_rate"],
        "post_old_A_hidden_nrmse": post["hidden_nrmse"],
        "pre_to_post_identity_drop": (
            pre["identity_accuracy"] - post["identity_accuracy"]
        ),
        "pre_to_post_hidden_nrmse_increase": (
            post["hidden_nrmse"] - pre["hidden_nrmse"]
        ),
        "one_step_idempotence_rate": post["one_step_idempotence_rate"],
        "repeat_identity_stability_rate": post[
            "repeat_identity_stability_rate"
        ],
        "hidden_idempotence_rms_max": post["hidden_idempotence_rms_max"],
        "clamp_max_error": post["clamp_max_error"],
        "unstored_lure_false_episode_recall_rate": lure_false,
    }
    for label in ("M10", "M11"):
        cells[label].update(hard_updates)
        numeric = [
            value for value in cells[label].values() if isinstance(value, float)
        ]
        cells[label]["nonfinite_metric_or_prediction_count"] = float(
            len(numeric) - np.count_nonzero(np.isfinite(numeric))
        )

    v1_soft = v1_cells["M10"]
    comparator: dict[str, float] = {
        "v1_soft_post_identity_accuracy": float(
            v1_soft["post_old_A_identity_accuracy"]
        ),
        "v1_soft_post_positive_coverage": float(
            v1_soft["post_old_A_positive_coverage"]
        ),
        "v1_soft_post_accepted_wrong_rate": float(
            v1_soft["post_old_A_accepted_wrong_rate"]
        ),
        "v1_soft_post_hidden_nrmse": float(v1_soft["post_old_A_hidden_nrmse"]),
        "v1_soft_lure_false_episode_recall_rate": float(
            v1_soft["unstored_lure_false_episode_recall_rate"]
        ),
        "v1_soft_convergence_rate": float(v1_soft["convergence_rate"]),
        "v1_soft_extra_step_stability_rate": float(
            v1_soft["extra_step_stability_rate"]
        ),
    }
    comparator.update(
        {
            "paired_hidden_nrmse_benefit": (
                comparator["v1_soft_post_hidden_nrmse"]
                - float(cells["M10"]["post_old_A_hidden_nrmse"])
            ),
            "paired_identity_difference": (
                float(cells["M10"]["post_old_A_identity_accuracy"])
                - comparator["v1_soft_post_identity_accuracy"]
            ),
            "paired_coverage_difference": (
                float(cells["M10"]["post_old_A_positive_coverage"])
                - comparator["v1_soft_post_positive_coverage"]
            ),
            "paired_accepted_wrong_difference": (
                float(cells["M10"]["post_old_A_accepted_wrong_rate"])
                - comparator["v1_soft_post_accepted_wrong_rate"]
            ),
            "paired_lure_false_difference": (
                float(cells["M10"]["unstored_lure_false_episode_recall_rate"])
                - comparator["v1_soft_lure_false_episode_recall_rate"]
            ),
        }
    )

    return {
        "cells": cells,
        "v1_soft_comparator": comparator,
    }


def _ci(values: Sequence[float], critical: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    center = float(np.mean(array))
    if len(array) < 2:
        return center, center
    margin = float(critical * np.std(array, ddof=1) / math.sqrt(len(array)))
    return center - margin, center + margin


def _interval_report(values: Sequence[float], critical: float) -> dict[str, object]:
    lower, upper = _ci(values, critical)
    return {
        "mean": float(np.mean(np.asarray(values, dtype=float))),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "seed_values": [float(value) for value in values],
    }


def _aggregate_cells(
    seed_results: Sequence[Mapping[str, object]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    means: dict[str, dict[str, float]] = {}
    provenance: dict[str, dict[str, int]] = {}
    for label in ("M00", "M10", "M01", "M11"):
        cells = [result["cells"][label] for result in seed_results]
        numeric_keys = [
            key for key, value in cells[0].items() if isinstance(value, float)
        ]
        means[label] = {
            key: float(np.mean([float(cell[key]) for cell in cells]))
            for key in numeric_keys
        }
        counts: dict[str, int] = {}
        for cell in cells:
            for source, count in cell["output_provenance"].items():
                counts[source] = counts.get(source, 0) + int(count)
        provenance[label] = counts
    return means, provenance


def _aggregate_comparator(
    seed_results: Sequence[Mapping[str, object]], critical: float
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    keys = tuple(seed_results[0]["v1_soft_comparator"])
    means = {
        key: float(
            np.mean(
                [float(result["v1_soft_comparator"][key]) for result in seed_results]
            )
        )
        for key in keys
    }
    paired_keys = (
        "paired_hidden_nrmse_benefit",
        "paired_identity_difference",
        "paired_coverage_difference",
        "paired_accepted_wrong_difference",
        "paired_lure_false_difference",
    )
    intervals = {
        key: _interval_report(
            [float(result["v1_soft_comparator"][key]) for result in seed_results],
            critical,
        )
        for key in paired_keys
    }
    return means, intervals


def _cell_seed_results(seed_results: Sequence[Mapping[str, object]]) -> list[dict]:
    return [dict(result["cells"]) for result in seed_results]


def _paired_upper(
    cell_results: Sequence[dict], left: str, right: str, metric: str, critical: float
) -> float:
    values = [
        float(result[left][metric]) - float(result[right][metric])
        for result in cell_results
    ]
    return _ci(values, critical)[1]


def _build_gate_checks(
    registration: Mapping[str, object],
    seed_results: Sequence[Mapping[str, object]],
    means: Mapping[str, Mapping[str, float]],
    effects: Mapping[str, Mapping[str, Mapping[str, object]]],
    comparator_means: Mapping[str, float],
    remediation: Mapping[str, Mapping[str, object]],
    critical: float,
    prelock_equivalence_report: Mapping[str, object],
) -> dict[str, bool]:
    gate = registration["all_of_gate"]
    recall = gate["ltm_recall"]
    dream = gate["dream_novel_combination"]
    integration = gate["integration_no_antagonism"]
    false_memory = gate["false_memory_and_provenance"]
    integrity = gate["forgetting_schema_and_integrity"]
    repair = gate["remediation_vs_frozen_v1_on_fresh_seeds"]
    cell_results = _cell_seed_results(seed_results)
    checks: dict[str, bool] = {}
    for cell in ("M10", "M11"):
        checks[f"{cell}.post_identity"] = means[cell][
            "post_old_A_identity_accuracy"
        ] >= recall["M10_and_M11_post_identity_accuracy_min"]
        checks[f"{cell}.post_coverage"] = means[cell][
            "post_old_A_positive_coverage"
        ] >= recall["M10_and_M11_post_positive_coverage_min"]
        checks[f"{cell}.post_hidden_nrmse"] = means[cell][
            "post_old_A_hidden_nrmse"
        ] <= recall["M10_and_M11_post_hidden_nrmse_max"]
        checks[f"{cell}.accepted_wrong"] = means[cell][
            "post_old_A_accepted_wrong_rate"
        ] <= recall["M10_and_M11_post_accepted_wrong_rate_max"]
        checks[f"{cell}.identity_drop"] = means[cell][
            "pre_to_post_identity_drop"
        ] <= recall["M10_and_M11_pre_to_post_identity_drop_max"]
        checks[f"{cell}.nrmse_increase"] = means[cell][
            "pre_to_post_hidden_nrmse_increase"
        ] <= recall["M10_and_M11_pre_to_post_hidden_nrmse_increase_max"]
        checks[f"{cell}.idempotence"] = means[cell][
            "one_step_idempotence_rate"
        ] >= recall["one_step_idempotence_rate_min"]
        checks[f"{cell}.repeat_stability"] = means[cell][
            "repeat_identity_stability_rate"
        ] >= recall["repeat_identity_stability_rate_min"]
        checks[f"{cell}.hidden_idempotence"] = all(
            float(result["cells"][cell]["hidden_idempotence_rms_max"])
            <= recall["hidden_idempotence_rms_max"]
            for result in seed_results
        )
        checks[f"{cell}.clamp"] = all(
            float(result["cells"][cell]["clamp_max_error"])
            <= recall["clamp_max_error_max"]
            for result in seed_results
        )
        checks[f"{cell}.lure_false"] = means[cell][
            "unstored_lure_false_episode_recall_rate"
        ] <= false_memory[
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
    denominator = (
        means["M00"]["post_old_A_hidden_nrmse"]
        + means["M01"]["post_old_A_hidden_nrmse"]
    )
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

    hidden = remediation["paired_hidden_nrmse_benefit"]
    checks["remediation.hidden_ci"] = hidden["ci95_lower"] >= repair[
        "paired_hidden_nrmse_benefit_ci95_lower_min"
    ]
    checks["remediation.hidden_strict_win"] = float(
        np.mean(np.asarray(hidden["seed_values"], dtype=float) > 0.0)
    ) >= repair["paired_hidden_nrmse_strict_win_fraction_min"]
    v1_error = comparator_means["v1_soft_post_hidden_nrmse"]
    if v1_error <= 0.0:
        raise RuntimeError("zero frozen V1 NRMSE denominator is hard invalid")
    checks["remediation.hidden_relative_reduction"] = (
        1.0 - means["M10"]["post_old_A_hidden_nrmse"] / v1_error
    ) >= repair["mean_hidden_nrmse_relative_reduction_min"]
    checks["remediation.identity_ci"] = remediation[
        "paired_identity_difference"
    ]["ci95_lower"] >= repair["paired_identity_difference_ci95_lower_min"]
    checks["remediation.coverage_ci"] = remediation[
        "paired_coverage_difference"
    ]["ci95_lower"] >= repair[
        "paired_positive_coverage_difference_ci95_lower_min"
    ]
    checks["remediation.accepted_wrong_ci"] = remediation[
        "paired_accepted_wrong_difference"
    ]["ci95_upper"] <= repair["paired_accepted_wrong_difference_ci95_upper_max"]
    checks["remediation.lure_false_ci"] = remediation[
        "paired_lure_false_difference"
    ]["ci95_upper"] <= repair["paired_lure_false_difference_ci95_upper_max"]

    for cell in ("M01", "M11"):
        checks[f"{cell}.novel_coverage"] = means[cell][
            "valid_output_coverage"
        ] >= dream["M01_and_M11_valid_output_coverage_min"]
        checks[f"{cell}.novel_nrmse"] = means[cell][
            "noise_free_base_hidden_nrmse"
        ] <= dream["M01_and_M11_hidden_nrmse_max"]
        checks[f"{cell}.synthetic_count"] = all(
            int(result["cells"][cell]["accepted_synthetic_bindings"])
            == int(dream["accepted_synthetic_bindings_per_seed_required"])
            for result in seed_results
        )
    dream_error_effect = effects["novel_error"]["D_main"]
    dream_coverage_effect = effects["novel_coverage"]["D_main"]
    denominator = (
        means["M00"]["noise_free_base_hidden_nrmse"]
        + means["M10"]["noise_free_base_hidden_nrmse"]
    )
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
    checks["D_main.strict_seed_win"] = float(
        np.mean(np.asarray(dream_error_effect["seed_values"], dtype=float) > 0.0)
    ) >= dream["D_main_strict_seed_win_fraction_min"]
    checks["D_main.coverage_gain"] = dream_coverage_effect["mean"] >= dream[
        "D_main_valid_coverage_gain_min"
    ]
    checks["D_main.coverage_ci"] = dream_coverage_effect["ci95_lower"] >= dream[
        "D_main_valid_coverage_paired_ci95_lower_min"
    ]
    violations = max(
        means[cell]["accepted_dream_port_violation_count"]
        + means[cell]["accepted_dream_context_violation_count"]
        + means[cell]["accepted_dream_join_violation_count"]
        for cell in ("M01", "M11")
    )
    checks["dream.constraint_violations"] = violations <= dream[
        "accepted_constraint_violation_count_max"
    ]

    checks["integration.recall_identity"] = (
        means["M11"]["post_old_A_identity_accuracy"]
        - means["M10"]["post_old_A_identity_accuracy"]
    ) >= integration["M11_recall_identity_minus_M10_min"]
    checks["integration.recall_nrmse"] = _paired_upper(
        cell_results, "M11", "M10", "post_old_A_hidden_nrmse", critical
    ) <= integration["M11_minus_M10_recall_nrmse_paired_ci95_upper_max"]
    checks["integration.novel_coverage"] = (
        means["M11"]["valid_output_coverage"]
        - means["M01"]["valid_output_coverage"]
    ) >= integration["M11_novel_coverage_minus_M01_min"]
    checks["integration.novel_nrmse"] = _paired_upper(
        cell_results, "M11", "M01", "noise_free_base_hidden_nrmse", critical
    ) <= integration["M11_minus_M01_novel_nrmse_paired_ci95_upper_max"]

    for cell in ("M00", "M10", "M01", "M11"):
        checks[f"{cell}.invalid_abstain"] = means[cell][
            "invalid_query_nonabstain_rate"
        ] <= false_memory["all_cells_invalid_query_nonabstain_rate_max"]
        checks[f"{cell}.novel_not_recalled"] = means[cell][
            "novel_valid_tagged_recalled_rate"
        ] <= false_memory["all_cells_novel_valid_tagged_recalled_rate_max"]
        checks[f"{cell}.synthetic_not_ltm"] = means[cell][
            "synthetic_to_ltm_insert_count"
        ] <= false_memory["synthetic_to_ltm_insert_count_max"]
        checks[f"{cell}.no_observed_overwrite"] = means[cell][
            "observed_binding_overwrite_count"
        ] <= integrity["observed_binding_overwrite_count_max"]
        checks[f"{cell}.observed_hash_stable"] = means[cell][
            "observed_binding_hash_change_count"
        ] <= integrity["observed_binding_hash_change_count_max"]
        checks[f"{cell}.no_heldout_read"] = means[cell][
            "heldout_target_read_count"
        ] <= integrity["heldout_target_read_count_max"]
        checks[f"{cell}.finite"] = means[cell][
            "nonfinite_metric_or_prediction_count"
        ] <= integrity["nonfinite_metric_or_prediction_count_max"]
    for left, right in (("M01", "M00"), ("M11", "M10")):
        checks[f"{left}-{right}.B_schema"] = _paired_upper(
            cell_results, left, right, "current_B_observed_nrmse", critical
        ) <= integrity[
            "dream_minus_matched_no_dream_current_B_nrmse_paired_ci95_upper_max"
        ]
        checks[f"{left}-{right}.A_schema"] = _paired_upper(
            cell_results,
            left,
            right,
            "slow_model_only_old_A_schema_nrmse",
            critical,
        ) <= integrity[
            "dream_minus_matched_no_dream_old_A_slow_schema_nrmse_paired_ci95_upper_max"
        ]
    checks["dream.no_cross_context_component"] = max(
        means[cell]["accepted_dream_port_violation_count"]
        + means[cell]["accepted_dream_context_violation_count"]
        for cell in ("M01", "M11")
    ) <= integrity["cross_context_or_component_dream_accept_count_max"]

    checks["equivalence.prelock_comprehensive"] = bool(
        prelock_equivalence_report["all_required_equal"]
    )
    checks["equivalence.prelock_no_registered_seed"] = int(
        prelock_equivalence_report[
            "registered_seed_used_for_prelock_equivalence"
        ]
    ) <= gate["implementation_equivalence"][
        "registered_seed_used_for_equivalence_max"
    ]
    return checks


def _load_registration(config_path: Path) -> tuple[dict, bytes]:
    raw = config_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTERED_CONFIG_SHA256:
        raise PermissionError("G7-M/V2 raw preregistration SHA256 changed")
    registration = json.loads(raw)
    if registration.get("runner") != "episodic_ltm_dream_factorial_v2":
        raise ValueError("G7-M/V2 factorial registration required")
    if registration.get("status") != "locked_pre_implementation":
        raise ValueError("G7-M/V2 registration must remain locked")
    if registration.get("extends") is not None or not registration.get("standalone"):
        raise ValueError("G7-M/V2 must remain standalone")
    return registration, raw


def _root(config_path: Path) -> Path:
    return config_path.resolve().parents[2]


def _implementation_hashes(config_path: Path) -> dict[str, str]:
    root = _root(config_path)
    relative = (
        "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge_v2.py",
        "examples/agi/episodic_ltm_dream_bridge_v2_gate.py",
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in relative
    }


def _assert_immutable_v1_dependencies(
    config_path: Path, registration: Mapping[str, object]
) -> dict[str, str]:
    root = _root(config_path)
    forbidden_test = (
        root / "artifacts/agi/episodic_ltm_dream_factorial_test_v1.json"
    )
    if forbidden_test.exists():
        raise PermissionError("locked V1 test artifact must remain unopened")
    expected = dict(
        registration["test_lock"]["require_immutable_v1_dependency_sha256"]
    )
    actual = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in expected
    }
    if actual != expected:
        raise PermissionError("immutable V1 dependency SHA256 changed")
    return actual


def _write_json_lf(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def _locked_json(raw: bytes, label: str) -> object:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw:
        raise PermissionError(f"{label} must use one LF transport")
    value = json.loads(raw)
    canonical = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise PermissionError(f"{label} bytes are not canonical")
    return value


def _implementation_lock_path(
    config_path: Path, registration: Mapping[str, object]
) -> Path:
    return _root(config_path) / registration["test_lock"][
        "implementation_lock_artifact"
    ]


def _prepare_implementation_lock(
    config_path: Path,
    registration: Mapping[str, object],
    registration_sha256: str,
) -> tuple[dict[str, object], str]:
    dependencies = _assert_immutable_v1_dependencies(config_path, registration)
    implementation = _implementation_hashes(config_path)
    comparator_sha, _ = frozen_v1_comparator_equivalence(registration)
    shared_sha, shared_report = off_range_shared_equivalence(
        registration, comparator_equivalence_sha256=comparator_sha
    )
    if shared_report.get("all_required_equal") is not True:
        raise PermissionError("off-range V1-shared implementation equivalence failed")
    recipe = registration["frozen_v1_soft_comparator"]["equivalence_hash_recipe"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": registration["experiment"],
        "registration_sha256": registration_sha256,
        "implementation_sha256": implementation,
        "immutable_v1_dependency_sha256": dependencies,
        "frozen_v1_comparator_equivalence_sha256": comparator_sha,
        "off_range_shared_equivalence_sha256": shared_sha,
        "off_range_shared_equivalence_report": shared_report,
        "equivalence_recipe_identifier": recipe["recipe_identifier"],
        "off_range_seeds": list(recipe["off_range_seeds_in_order"]),
        "registered_seed_used_for_prelock_equivalence": shared_report[
            "registered_seed_used_for_prelock_equivalence"
        ],
    }
    path = _implementation_lock_path(config_path, registration)
    if path.exists():
        raw = path.read_bytes()
        if _locked_json(raw, "G7-M/V2 implementation lock") != payload:
            raise PermissionError("G7-M/V2 implementation lock changed")
    else:
        _write_json_lf(path, payload)
        raw = path.read_bytes()
    return payload, hashlib.sha256(raw).hexdigest()


def _train_calibration_path(
    config_path: Path, registration: Mapping[str, object]
) -> Path:
    return _root(config_path) / registration["test_lock"][
        "train_calibration_artifact"
    ]


def _calibration_artifact_payload(
    calibration: TrainCalibrationV2,
    registration: Mapping[str, object],
    registration_sha256: str,
    implementation_lock_sha256: str,
    implementation_sha256: Mapping[str, str],
    dependencies: Mapping[str, str],
    off_range_shared_equivalence_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "experiment": registration["experiment"],
        "source_split": "train_only",
        "mu": calibration.standardizer.mean.tolist(),
        "sigma": calibration.standardizer.scale.tolist(),
        "tau_v2_pre": calibration.v2_threshold_pre_48,
        "tau_v2_post": calibration.v2_threshold_post_96,
        "tau_v1_pre": calibration.v1_threshold_pre_48,
        "tau_v1_post": calibration.v1_threshold_post_96,
        "join_threshold": calibration.join_threshold,
        "registration_sha256": registration_sha256,
        "implementation_lock_artifact_sha256": implementation_lock_sha256,
        "implementation_sha256": dict(implementation_sha256),
        "immutable_v1_dependency_sha256": dict(dependencies),
        "frozen_v1_comparator_equivalence_sha256": (
            calibration.comparator_equivalence_sha256
        ),
        "off_range_shared_equivalence_sha256": (
            calibration.off_range_shared_equivalence_sha256
        ),
        "off_range_shared_equivalence_report": dict(
            off_range_shared_equivalence_report
        ),
    }


def _calibration_from_artifact(raw: bytes) -> TrainCalibrationV2:
    payload = _locked_json(raw, "G7-M/V2 train calibration")
    mean = np.asarray(payload["mu"], dtype=float)
    scale = np.asarray(payload["sigma"], dtype=float)
    if mean.shape != (96,) or scale.shape != (96,):
        raise PermissionError("frozen V2 calibration coordinate shape changed")
    scalars = np.asarray(
        [
            payload["tau_v2_pre"],
            payload["tau_v2_post"],
            payload["tau_v1_pre"],
            payload["tau_v1_post"],
            payload["join_threshold"],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(np.concatenate((mean, scale, scalars)))):
        raise PermissionError("frozen V2 calibration is nonfinite")
    if np.any(scale < 1e-8):
        raise PermissionError("frozen V2 calibration scale changed")
    comparator_sha = payload["frozen_v1_comparator_equivalence_sha256"]
    if not isinstance(comparator_sha, str) or len(comparator_sha) != 64:
        raise PermissionError("frozen V1 comparator equivalence hash missing")
    shared_sha = payload["off_range_shared_equivalence_sha256"]
    if not isinstance(shared_sha, str) or len(shared_sha) != 64:
        raise PermissionError("off-range shared equivalence hash missing")
    return TrainCalibrationV2(
        standardizer=CoordinateStandardizer(mean, scale),
        v2_threshold_pre_48=float(payload["tau_v2_pre"]),
        v2_threshold_post_96=float(payload["tau_v2_post"]),
        v1_threshold_pre_48=float(payload["tau_v1_pre"]),
        v1_threshold_post_96=float(payload["tau_v1_post"]),
        join_threshold=float(payload["join_threshold"]),
        comparator_equivalence_sha256=comparator_sha,
        sha256=hashlib.sha256(raw).hexdigest(),
        off_range_shared_equivalence_sha256=shared_sha,
    )


def _assert_validation_exists(
    config_path: Path, registration: Mapping[str, object]
) -> Path:
    path = _root(config_path) / registration["test_lock"]["validation_artifact"]
    if not path.exists():
        raise PermissionError("G7-M/V2 test requires a saved passing validation artifact")
    report = _locked_json(path.read_bytes(), "G7-M/V2 validation artifact")
    if report.get("split") != "validation" or report.get("passed") is not True:
        raise PermissionError("G7-M/V2 validation all-of gate did not pass")
    checks = report.get("checks")
    resource_checks = report.get("resource_checks")
    if not isinstance(checks, dict) or not checks:
        raise PermissionError("G7-M/V2 validation artifact checks are incomplete")
    if not isinstance(resource_checks, dict) or not resource_checks:
        raise PermissionError("G7-M/V2 validation resource checks are incomplete")
    performance = bool(all(value is True for value in checks.values()))
    resources = bool(all(value is True for value in resource_checks.values()))
    if (
        report.get("performance_passed") is not performance
        or report.get("resource_passed") is not resources
        or report.get("passed") is not (performance and resources)
        or report.get("passed") is not True
    ):
        raise PermissionError("G7-M/V2 validation artifact is not self-consistent")
    return path


def _assert_test_unlocked(
    config_path: Path,
    registration: Mapping[str, object],
    registration_sha256: str,
    implementation_lock_sha256: str,
    implementation_lock: Mapping[str, object],
) -> tuple[str, str]:
    validation_path = _assert_validation_exists(config_path, registration)
    report = _locked_json(
        validation_path.read_bytes(), "G7-M/V2 validation artifact"
    )
    if report.get("experiment") != registration["experiment"]:
        raise PermissionError("G7-M/V2 validation experiment changed")
    if report.get("registration_sha256") != registration_sha256:
        raise PermissionError("G7-M/V2 registration changed after validation")
    if report.get("implementation_sha256") != implementation_lock[
        "implementation_sha256"
    ]:
        raise PermissionError("G7-M/V2 implementation changed after validation")
    if report.get("implementation_lock_artifact_sha256") != (
        implementation_lock_sha256
    ):
        raise PermissionError("G7-M/V2 implementation lock changed after validation")
    if (
        not isinstance(
            implementation_lock.get("immutable_v1_dependency_sha256"), dict
        )
        or not isinstance(
            implementation_lock.get("frozen_v1_comparator_equivalence_sha256"),
            str,
        )
        or not isinstance(
            implementation_lock.get("off_range_shared_equivalence_sha256"), str
        )
    ):
        raise PermissionError("G7-M/V2 implementation dependency lock is incomplete")
    report_locks = {
        "immutable_v1_dependency_sha256": implementation_lock.get(
            "immutable_v1_dependency_sha256"
        ),
        "frozen_v1_comparator_equivalence_sha256": implementation_lock.get(
            "frozen_v1_comparator_equivalence_sha256"
        ),
    }
    if any(report.get(key) != value for key, value in report_locks.items()):
        raise PermissionError("G7-M/V2 validation dependency lock changed")
    prelock = report.get("prelock_implementation_equivalence")
    if not isinstance(prelock, dict) or prelock.get(
        "off_range_shared_equivalence_sha256"
    ) != implementation_lock.get("off_range_shared_equivalence_sha256"):
        raise PermissionError("G7-M/V2 validation shared equivalence lock changed")
    calibration_sha = report.get("train_calibration_sha256")
    if not isinstance(calibration_sha, str) or len(calibration_sha) != 64:
        raise PermissionError("G7-M/V2 validation lacks frozen train calibration")
    calibration_path = _train_calibration_path(config_path, registration)
    if not calibration_path.exists():
        raise PermissionError("G7-M/V2 frozen train calibration is missing")
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != calibration_sha:
        raise PermissionError("G7-M/V2 train calibration changed after validation")
    return calibration_sha, hashlib.sha256(validation_path.read_bytes()).hexdigest()


def run_episodic_ltm_dream_v2_gate(
    config_path: Path, *, split: str = "validation"
) -> dict[str, object]:
    """Run a locked V2 split; registered test remains closed until validation passes."""

    started = time.perf_counter()
    registration, raw = _load_registration(config_path)
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    _assert_immutable_v1_dependencies(config_path, registration)
    if split == "test":
        _assert_validation_exists(config_path, registration)
    registration_sha = hashlib.sha256(raw).hexdigest()
    implementation_lock, implementation_lock_sha = _prepare_implementation_lock(
        config_path, registration, registration_sha
    )
    implementation_sha = dict(implementation_lock["implementation_sha256"])
    dependencies = dict(implementation_lock["immutable_v1_dependency_sha256"])
    comparator_sha = str(
        implementation_lock["frozen_v1_comparator_equivalence_sha256"]
    )
    shared_equivalence_sha = str(
        implementation_lock["off_range_shared_equivalence_sha256"]
    )
    shared_equivalence_report = dict(
        implementation_lock["off_range_shared_equivalence_report"]
    )
    expected_calibration_sha = None
    validation_artifact_sha = None
    if split == "test":
        expected_calibration_sha, validation_artifact_sha = _assert_test_unlocked(
            config_path,
            registration,
            registration_sha,
            implementation_lock_sha,
            implementation_lock,
        )

    calibration_path = _train_calibration_path(config_path, registration)
    if split == "validation" and not calibration_path.exists():
        train_worlds = [
            _generate_seed_world(int(seed))
            for seed in registration["data_roles"]["train"]["seeds"]
        ]
        calibration = calibrate_train_worlds_v2(
            train_worlds,
            comparator_equivalence_sha256=comparator_sha,
            off_range_shared_equivalence_sha256=shared_equivalence_sha,
        )
        payload = _calibration_artifact_payload(
            calibration,
            registration,
            registration_sha,
            implementation_lock_sha,
            implementation_sha,
            dependencies,
            shared_equivalence_report,
        )
        _write_json_lf(calibration_path, payload)
        calibration_raw = calibration_path.read_bytes()
        calibration = _calibration_from_artifact(calibration_raw)
        train_seed_count = len(train_worlds)
    else:
        if not calibration_path.exists():
            raise PermissionError("G7-M/V2 train calibration artifact is missing")
        calibration_raw = calibration_path.read_bytes()
        if expected_calibration_sha is not None and (
            hashlib.sha256(calibration_raw).hexdigest()
            != expected_calibration_sha
        ):
            raise PermissionError("G7-M/V2 train calibration changed")
        payload = _locked_json(calibration_raw, "G7-M/V2 train calibration")
        expected_locks = {
            "registration_sha256": registration_sha,
            "implementation_lock_artifact_sha256": implementation_lock_sha,
            "implementation_sha256": implementation_sha,
            "immutable_v1_dependency_sha256": dependencies,
            "frozen_v1_comparator_equivalence_sha256": comparator_sha,
            "off_range_shared_equivalence_sha256": shared_equivalence_sha,
            "off_range_shared_equivalence_report": shared_equivalence_report,
        }
        if any(payload.get(key) != value for key, value in expected_locks.items()):
            raise PermissionError("G7-M/V2 calibration lock changed")
        calibration = _calibration_from_artifact(calibration_raw)
        train_seed_count = 0

    # Reverify immutable dependencies and the golden comparator immediately before
    # any validation/test seed is generated, as the preregistration requires.
    if _assert_immutable_v1_dependencies(config_path, registration) != dependencies:
        raise PermissionError("immutable V1 dependencies changed before evaluation")
    repeated_comparator_sha, _ = frozen_v1_comparator_equivalence(registration)
    if repeated_comparator_sha != comparator_sha:
        raise PermissionError("frozen V1 comparator equivalence changed")
    if calibration.comparator_equivalence_sha256 != comparator_sha:
        raise PermissionError("calibration comparator equivalence lock changed")
    if calibration.off_range_shared_equivalence_sha256 != shared_equivalence_sha:
        raise PermissionError("calibration shared equivalence lock changed")
    repeated_shared_sha, repeated_shared_report = off_range_shared_equivalence(
        registration, comparator_equivalence_sha256=comparator_sha
    )
    if (
        repeated_shared_sha != shared_equivalence_sha
        or repeated_shared_report != shared_equivalence_report
        or repeated_shared_report.get("all_required_equal") is not True
    ):
        raise PermissionError("off-range shared equivalence changed")

    role = registration["data_roles"][split]
    seed_results = [
        evaluate_factorial_seed_v2(int(seed), calibration, registration)
        for seed in role["seeds"]
    ]
    critical = float(
        registration["paired_inference"][
            "validation_critical_value_n40"
            if split == "validation"
            else "test_critical_value_n60"
        ]
    )
    means, provenance = _aggregate_cells(seed_results)
    worst_case = {
        cell: {
            "hidden_idempotence_rms_max": max(
                float(result["cells"][cell]["hidden_idempotence_rms_max"])
                for result in seed_results
            ),
            "clamp_max_error": max(
                float(result["cells"][cell]["clamp_max_error"])
                for result in seed_results
            ),
        }
        for cell in ("M10", "M11")
    }
    comparator_means, remediation = _aggregate_comparator(seed_results, critical)
    cell_results = _cell_seed_results(seed_results)
    effects = {
        "recall_identity": v1._effect_report(
            v1._factorial_arrays(cell_results, "post_old_A_identity_accuracy"),
            critical,
        ),
        "recall_error": v1._effect_report(
            v1._error_effect_arrays(cell_results, "post_old_A_hidden_nrmse"),
            critical,
        ),
        "novel_coverage": v1._effect_report(
            v1._factorial_arrays(cell_results, "valid_output_coverage"), critical
        ),
        "novel_error": v1._effect_report(
            v1._error_effect_arrays(
                cell_results, "noise_free_base_hidden_nrmse"
            ),
            critical,
        ),
    }
    checks = _build_gate_checks(
        registration,
        seed_results,
        means,
        effects,
        comparator_means,
        remediation,
        critical,
        shared_equivalence_report,
    )
    resources = registration["resources"]
    resource_checks = {
        "ltm_cells_have_96_observed_items": all(
            int(result["cells"][cell]["persistent_observed_items"])
            == int(resources["persistent_observed_items_M10_M11"])
            for result in seed_results
            for cell in ("M10", "M11")
        ),
        "no_ltm_cells_have_zero_items": all(
            int(result["cells"][cell]["persistent_observed_items"])
            == int(resources["persistent_observed_items_M00_M01"])
            for result in seed_results
            for cell in ("M00", "M01")
        ),
        "persistent_trace_bytes": all(
            float(result["cells"][cell]["persistent_trace_bytes"])
            <= float(resources["persistent_observed_trace_bytes_max"])
            for result in seed_results
            for cell in ("M10", "M11")
        ),
        "comparator_adds_no_persistent_records": bool(
            resources[
                "frozen_v1_comparator_runs_sequentially_and_does_not_add_persistent_records"
            ]
        ),
        "zero_download": int(resources["external_download_bytes"]) == 0,
        "zero_raw_trajectory_files": not bool(
            resources["write_raw_trajectory_files"]
        ),
        "numpy_cpu": resources["backend"] == "numpy_cpu",
    }
    report: dict[str, object] = {
        "experiment": registration["experiment"],
        "roadmap_stage": registration["roadmap_stage"],
        "split": split,
        "registration_sha256": registration_sha,
        "implementation_sha256": implementation_sha,
        "implementation_lock_artifact_sha256": implementation_lock_sha,
        "immutable_v1_dependency_sha256": dependencies,
        "frozen_v1_comparator_equivalence_sha256": comparator_sha,
        "prelock_implementation_equivalence": {
            "recipe_identifier": implementation_lock[
                "equivalence_recipe_identifier"
            ],
            "off_range_seeds": implementation_lock["off_range_seeds"],
            "registered_seed_used_for_prelock_equivalence": implementation_lock[
                "registered_seed_used_for_prelock_equivalence"
            ],
            "off_range_shared_equivalence_sha256": shared_equivalence_sha,
            "off_range_shared_equivalence_report": shared_equivalence_report,
        },
        "train_calibration": _calibration_values(calibration),
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
        "cell_worst_case": worst_case,
        "novel_output_provenance_counts": provenance,
        "v1_soft_comparator_means": comparator_means,
        "fresh_v1_remediation": remediation,
        "factorial_effects": effects,
        "seed_results": seed_results,
        "checks": checks,
        "performance_passed": bool(all(checks.values())),
        "resource_checks": resource_checks,
        "resource_passed": bool(all(resource_checks.values())),
        "resource_usage": {
            "wall_seconds": time.perf_counter() - started,
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


def _default_output(
    config_path: Path, split: str, registration: Mapping[str, object]
) -> Path:
    key = "validation_artifact" if split == "validation" else "test_artifact"
    return _root(config_path) / registration["test_lock"][key]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    registration, _ = _load_registration(args.config)
    report = run_episodic_ltm_dream_v2_gate(args.config, split=args.split)
    output = args.output or _default_output(args.config, args.split, registration)
    _write_json_lf(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"artifact: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
