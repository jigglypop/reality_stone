"""Canonical payload hashes and immutable split IDs for the dimension execution chain."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

if __package__:
    from .conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate
    from .verified_dimension_preprocessing_execution import (
        VerifiedDimensionPreprocessingExecution,
        verified_dimension_preprocessing_execution,
    )
else:
    from conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate  # type: ignore[no-redef]
    from verified_dimension_preprocessing_execution import (  # type: ignore[no-redef]
        VerifiedDimensionPreprocessingExecution,
        verified_dimension_preprocessing_execution,
    )


@dataclass(frozen=True)
class VerifiedDimensionObservationManifest:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    preprocessing_execution: VerifiedDimensionPreprocessingExecution | None
    development_payload_sha256: str
    heldout_payload_sha256: str
    preprocessing_payload_sha256: str
    eigenbasis_payload_sha256: str
    bootstrap_payload_sha256: str
    execution_contract_sha256: str
    expected_hashes_match_computed_payloads: bool
    observation_id_shapes_match_payloads: bool
    observation_ids_globally_unique_across_splits: bool
    canonical_exact_payload_encoding_verified: bool
    content_addressed_dimension_chain_composed: bool
    external_source_signature_verified: bool
    source_locked_empirical_dimension_result: bool
    consciousness_dimension_claim_admitted: bool


def _canonical(value: object):
    if type(value) is bool or isinstance(value, float):
        raise ValueError("canonical exact payloads forbid booleans and floats")
    if isinstance(value, (int, Fraction)):
        number = Fraction(value)
        return ["q", str(number.numerator), str(number.denominator)]
    if isinstance(value, str):
        return ["s", value]
    if isinstance(value, (tuple, list)):
        return ["a", *(_canonical(item) for item in value)]
    if isinstance(value, dict):
        items = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        items.sort(key=lambda pair: json.dumps(pair[0], ensure_ascii=False, separators=(",", ":")))
        return ["m", *([key, item] for key, item in items)]
    raise ValueError(f"unsupported canonical exact payload type: {type(value).__name__}")


def canonical_exact_bytes(value: object) -> bytes:
    return json.dumps(_canonical(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def canonical_exact_sha256(value: object) -> str:
    return hashlib.sha256(canonical_exact_bytes(value)).hexdigest()


def _decode_canonical(node: object) -> object:
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError("canonical exact node must be a tagged JSON array")
    tag = node[0]
    if tag == "q":
        if len(node) != 3 or not all(isinstance(item, str) for item in node[1:]):
            raise ValueError("canonical rational node is malformed")
        try:
            numerator = int(node[1])
            denominator = int(node[2])
        except ValueError as error:
            raise ValueError("canonical rational tokens must be decimal integers") from error
        if denominator <= 0 or str(numerator) != node[1] or str(denominator) != node[2]:
            raise ValueError("canonical rational tokens are not normalized")
        value = Fraction(numerator, denominator)
        if value.numerator != numerator or value.denominator != denominator:
            raise ValueError("canonical rational must be reduced")
        return value
    if tag == "s":
        if len(node) != 2 or not isinstance(node[1], str):
            raise ValueError("canonical string node is malformed")
        return node[1]
    if tag == "a":
        return tuple(_decode_canonical(item) for item in node[1:])
    if tag == "m":
        result = {}
        for pair in node[1:]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("canonical mapping pair is malformed")
            key = _decode_canonical(pair[0])
            try:
                if key in result:
                    raise ValueError("canonical mapping contains a duplicate key")
                result[key] = _decode_canonical(pair[1])
            except TypeError as error:
                raise ValueError("canonical mapping key must be hashable") from error
        return result
    raise ValueError(f"unknown canonical exact tag: {tag}")


def canonical_exact_from_bytes(encoded: object) -> object:
    if not isinstance(encoded, bytes):
        raise ValueError("canonical exact archive payload must be bytes")
    try:
        node = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("canonical exact archive payload is not valid UTF-8 JSON") from error
    value = _decode_canonical(node)
    if canonical_exact_bytes(value) != encoded:
        raise ValueError("archive payload is semantically valid but not canonical byte-for-byte")
    return value


def _expected_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _flat_ids_for_rows(ids: object, observations: object, name: str, sessions: int):
    if not isinstance(ids, (tuple, list)) or len(ids) != sessions:
        raise ValueError(f"{name} must match session count")
    flat = []
    valid = True
    for session_ids, rows in zip(ids, observations, strict=True):
        if not isinstance(session_ids, (tuple, list)) or len(session_ids) != len(rows):
            valid = False
            continue
        for value in session_ids:
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError("observation IDs must be nonempty canonical strings")
            flat.append(value)
    return tuple(flat), valid


def _flat_ids_for_windows(ids: object, observations: object, name: str, sessions: int):
    if not isinstance(ids, (tuple, list)) or len(ids) != sessions:
        raise ValueError(f"{name} must match session count")
    flat = []
    valid = True
    for session_ids, windows in zip(ids, observations, strict=True):
        if not isinstance(session_ids, (tuple, list)) or len(session_ids) != len(windows):
            valid = False
            continue
        for window_ids, rows in zip(session_ids, windows, strict=True):
            if not isinstance(window_ids, (tuple, list)) or len(window_ids) != len(rows):
                valid = False
                continue
            for value in window_ids:
                if not isinstance(value, str) or not value or value.strip() != value:
                    raise ValueError("observation IDs must be nonempty canonical strings")
                flat.append(value)
    return tuple(flat), valid


def verified_dimension_observation_manifest(
    *,
    base_dimension_certificate: ConsciousMomentDimensionProtocolCertificate,
    session_ids: object,
    development_observation_ids_by_session: object,
    conscious_heldout_observation_ids_by_session: object,
    control_heldout_observation_ids_by_session: object,
    raw_development_observations_by_session: object,
    raw_conscious_heldout_observations_by_session: object,
    raw_control_heldout_observations_by_session: object,
    coordinate_reference_scales_by_session: object,
    covariance_eigenbasis_witnesses_by_session: object,
    complexity_penalty: object,
    bootstrap_window_indices_by_session: object,
    frozen_block_length: int,
    window_partition_kind: str,
    block_bootstrap_kind: str,
    minimum_conscious_band_window_fraction: object,
    minimum_selected_rank_window_fraction: object,
    maximum_conscious_rank_transition_fraction: object,
    minimum_selected_rank_longest_dwell_fraction: object,
    minimum_conscious_control_band_gap: object,
    minimum_bootstrap_band_fraction: object,
    minimum_bootstrap_selected_rank_fraction: object,
    expected_development_payload_sha256: object,
    expected_heldout_payload_sha256: object,
    expected_preprocessing_payload_sha256: object,
    expected_eigenbasis_payload_sha256: object,
    expected_bootstrap_payload_sha256: object,
    expected_execution_contract_sha256: object,
) -> VerifiedDimensionObservationManifest:
    if not isinstance(session_ids, (tuple, list)) or len(session_ids) < 3:
        raise ValueError("at least three session IDs are required")
    ids = tuple(session_ids)
    development_ids, development_shape = _flat_ids_for_rows(
        development_observation_ids_by_session, raw_development_observations_by_session,
        "development_observation_ids_by_session", len(ids),
    )
    conscious_ids, conscious_shape = _flat_ids_for_windows(
        conscious_heldout_observation_ids_by_session, raw_conscious_heldout_observations_by_session,
        "conscious_heldout_observation_ids_by_session", len(ids),
    )
    control_ids, control_shape = _flat_ids_for_windows(
        control_heldout_observation_ids_by_session, raw_control_heldout_observations_by_session,
        "control_heldout_observation_ids_by_session", len(ids),
    )
    shape_ok = development_shape and conscious_shape and control_shape
    all_observation_ids = development_ids + conscious_ids + control_ids
    unique = len(set(all_observation_ids)) == len(all_observation_ids)

    development_hash = canonical_exact_sha256((
        "CE-DIM-DEVELOPMENT-v1", ids, development_observation_ids_by_session,
        raw_development_observations_by_session,
    ))
    heldout_hash = canonical_exact_sha256((
        "CE-DIM-HELDOUT-v1", ids,
        conscious_heldout_observation_ids_by_session, raw_conscious_heldout_observations_by_session,
        control_heldout_observation_ids_by_session, raw_control_heldout_observations_by_session,
    ))
    preprocessing_hash = canonical_exact_sha256((
        "CE-DIM-PREPROCESSING-v1", ids, coordinate_reference_scales_by_session,
    ))
    eigenbasis_hash = canonical_exact_sha256((
        "CE-DIM-EIGENBASIS-v1", ids, covariance_eigenbasis_witnesses_by_session,
    ))
    bootstrap_hash = canonical_exact_sha256((
        "CE-DIM-BOOTSTRAP-v1", ids, frozen_block_length, bootstrap_window_indices_by_session,
    ))
    contract_hash = canonical_exact_sha256((
        "CE-DIM-EXECUTION-CONTRACT-v1",
        base_dimension_certificate.frozen_dimension_menu,
        base_dimension_certificate.candidate_band,
        base_dimension_certificate.selected_signal_rank,
        complexity_penalty, window_partition_kind, block_bootstrap_kind,
        minimum_conscious_band_window_fraction, minimum_selected_rank_window_fraction,
        maximum_conscious_rank_transition_fraction, minimum_selected_rank_longest_dwell_fraction,
        minimum_conscious_control_band_gap, minimum_bootstrap_band_fraction,
        minimum_bootstrap_selected_rank_fraction,
    ))
    expected = tuple(_expected_hash(value, name) for value, name in (
        (expected_development_payload_sha256, "expected_development_payload_sha256"),
        (expected_heldout_payload_sha256, "expected_heldout_payload_sha256"),
        (expected_preprocessing_payload_sha256, "expected_preprocessing_payload_sha256"),
        (expected_eigenbasis_payload_sha256, "expected_eigenbasis_payload_sha256"),
        (expected_bootstrap_payload_sha256, "expected_bootstrap_payload_sha256"),
        (expected_execution_contract_sha256, "expected_execution_contract_sha256"),
    ))
    computed = (development_hash, heldout_hash, preprocessing_hash, eigenbasis_hash, bootstrap_hash, contract_hash)
    hashes_match = computed == expected
    failures: list[str] = []
    if not shape_ok:
        failures.append("DIMENSION_MANIFEST_OBSERVATION_ID_SHAPE_MISMATCH")
    if not unique:
        failures.append("DIMENSION_MANIFEST_OBSERVATION_ID_SPLIT_OVERLAP")
    if not hashes_match:
        failures.append("DIMENSION_MANIFEST_EXPECTED_HASH_MISMATCH")
    preprocessing = None
    if not failures:
        preprocessing = verified_dimension_preprocessing_execution(
            base_dimension_certificate=base_dimension_certificate, session_ids=ids,
            raw_development_observations_by_session=raw_development_observations_by_session,
            raw_conscious_heldout_observations_by_session=raw_conscious_heldout_observations_by_session,
            raw_control_heldout_observations_by_session=raw_control_heldout_observations_by_session,
            coordinate_reference_scales_by_session=coordinate_reference_scales_by_session,
            covariance_eigenbasis_witnesses_by_session=covariance_eigenbasis_witnesses_by_session,
            complexity_penalty=complexity_penalty,
            bootstrap_window_indices_by_session=bootstrap_window_indices_by_session,
            frozen_block_length=frozen_block_length,
            window_partition_kind=window_partition_kind, block_bootstrap_kind=block_bootstrap_kind,
            minimum_conscious_band_window_fraction=minimum_conscious_band_window_fraction,
            minimum_selected_rank_window_fraction=minimum_selected_rank_window_fraction,
            maximum_conscious_rank_transition_fraction=maximum_conscious_rank_transition_fraction,
            minimum_selected_rank_longest_dwell_fraction=minimum_selected_rank_longest_dwell_fraction,
            minimum_conscious_control_band_gap=minimum_conscious_control_band_gap,
            minimum_bootstrap_band_fraction=minimum_bootstrap_band_fraction,
            minimum_bootstrap_selected_rank_fraction=minimum_bootstrap_selected_rank_fraction,
            raw_development_sha256=development_hash, raw_heldout_sha256=heldout_hash,
            preprocessing_contract_sha256=preprocessing_hash,
            eigenbasis_witness_sha256=eigenbasis_hash,
            bootstrap_schedule_sha256=bootstrap_hash,
            execution_contract_sha256=contract_hash,
        )
        if preprocessing.validation_level is None:
            failures.extend(preprocessing.failure_codes)
    success = not failures and preprocessing is not None
    status = "VERIFIED_CONTENT_ADDRESSED_IMMUTABLE_DIMENSION_SPLIT_CHAIN" if success else failures[0]
    return VerifiedDimensionObservationManifest(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), preprocessing_execution=preprocessing,
        development_payload_sha256=development_hash, heldout_payload_sha256=heldout_hash,
        preprocessing_payload_sha256=preprocessing_hash,
        eigenbasis_payload_sha256=eigenbasis_hash, bootstrap_payload_sha256=bootstrap_hash,
        execution_contract_sha256=contract_hash,
        expected_hashes_match_computed_payloads=hashes_match,
        observation_id_shapes_match_payloads=shape_ok,
        observation_ids_globally_unique_across_splits=unique,
        canonical_exact_payload_encoding_verified=True,
        content_addressed_dimension_chain_composed=success,
        external_source_signature_verified=False,
        source_locked_empirical_dimension_result=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = [
    "VerifiedDimensionObservationManifest",
    "canonical_exact_bytes",
    "canonical_exact_from_bytes",
    "canonical_exact_sha256",
    "verified_dimension_observation_manifest",
]
