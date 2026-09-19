"""Fail-closed measurement bridge from neural envelopes to graph certificates.

The bridge consumes already computed simultaneous confidence envelopes.  It
does not estimate them from spikes and does not treat hashes as evidence that
their contents are scientifically valid.  Its job is narrower: enforce one
frozen parameter family, specimen-disjoint splits, complete held-out envelope
coverage, and exact routing into the full global/local/matched theorem chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import re

if __package__:
    from .e1_metadata_receipt import E1MetadataReceipt
    from .quantitative_graph_transform import _exact_fraction
    from .quantitative_coupled_graph_transform import quantitative_coupled_graph_transform
    from .quantitative_coupled_arbitrary_order_graph_transform import quantitative_coupled_arbitrary_order_graph_transform
    from .quantitative_local_coupled_arbitrary_order_graph_transform import (
        quantitative_local_coupled_arbitrary_order_graph_transform,
        quantitative_matched_local_coupled_arbitrary_order_graph_transform,
    )
    from .quantitative_full_coupled_arbitrary_order_graph_transform import (
        quantitative_full_coupled_arbitrary_order_graph_transform,
        quantitative_full_local_coupled_arbitrary_order_graph_transform,
        quantitative_full_matched_local_coupled_arbitrary_order_graph_transform,
    )
    from .conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate
else:
    from e1_metadata_receipt import E1MetadataReceipt  # type: ignore[no-redef]
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from quantitative_coupled_graph_transform import quantitative_coupled_graph_transform  # type: ignore[no-redef]
    from quantitative_coupled_arbitrary_order_graph_transform import quantitative_coupled_arbitrary_order_graph_transform  # type: ignore[no-redef]
    from quantitative_local_coupled_arbitrary_order_graph_transform import (  # type: ignore[no-redef]
        quantitative_local_coupled_arbitrary_order_graph_transform,
        quantitative_matched_local_coupled_arbitrary_order_graph_transform,
    )
    from quantitative_full_coupled_arbitrary_order_graph_transform import (  # type: ignore[no-redef]
        quantitative_full_coupled_arbitrary_order_graph_transform,
        quantitative_full_local_coupled_arbitrary_order_graph_transform,
        quantitative_full_matched_local_coupled_arbitrary_order_graph_transform,
    )
    from conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate  # type: ignore[no-redef]


SOURCE_LOCKED_EMPIRICAL = "SOURCE_LOCKED_HELD_OUT_EMPIRICAL"
SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
SIMULTANEOUS_COVERAGE_KIND = "FAMILYWISE_SIMULTANEOUS_INTERVAL"
GLOBAL_LEVEL = "GLOBAL"
LOCAL_LEVEL = "LOCAL"
MATCHED_LEVEL = "MATCHED"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPLITS = ("calibration", "development", "held_out")


@dataclass(frozen=True)
class FrozenNeuralGraphTransformMeasurementContract:
    base_dimension: int
    maximum_order: int
    requested_theorem_level: str
    parameters: tuple[tuple[str, Fraction], ...]
    required_upper_envelope_names: tuple[str, ...]
    required_lower_envelope_names: tuple[str, ...]


@dataclass(frozen=True)
class NeuralMapSessionEnvelope:
    ecephys_session_id: str
    specimen_id: str
    split: str
    upper_envelopes: tuple[tuple[str, Fraction], ...]
    lower_envelopes: tuple[tuple[str, Fraction], ...]


@dataclass(frozen=True)
class NeuralGraphTransformMeasurementBridgeCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    provenance_status: str
    metadata_table_sha256: str
    assignment_sha256: str
    neural_observation_sha256: str
    analysis_contract_sha256: str
    preprocessing_sha256: str
    model_family_sha256: str
    simultaneous_coverage_kind: str
    familywise_error_upper: Fraction
    coverage_family_size: int
    expected_coverage_family_size: int
    session_count: int
    session_assignments: tuple[tuple[str, str, str], ...]
    specimen_count_by_split: tuple[tuple[str, int], ...]
    frozen_contract: FrozenNeuralGraphTransformMeasurementContract
    requested_theorem_level: str
    highest_verified_theorem_level: str | None
    c0_certificate: object
    derivative_certificate: object
    full_global_certificate: object
    derivative_local_certificate: object
    full_local_certificate: object
    derivative_matched_certificate: object
    full_matched_certificate: object
    all_session_envelopes_within_frozen_family: bool
    external_remote_metadata_receipt_verified: bool
    source_locked_empirical_neural_graph_result: bool
    consciousness_claim_admitted: bool
    dimension_4_6_claim_admitted: bool


@dataclass(frozen=True)
class NeuralGraphAndSignalRankJointCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    measurement_certificate: NeuralGraphTransformMeasurementBridgeCertificate
    dimension_certificate: ConsciousMomentDimensionProtocolCertificate
    heldout_session_ids: tuple[str, ...]
    graph_base_dimension: int
    selected_neural_signal_rank: int | None
    graph_dimension_equals_signal_rank_numerically: bool
    graph_dimension_signal_rank_identity_claim_admitted: bool
    source_locked_joint_empirical_result: bool
    consciousness_claim_admitted: bool


def _canonical_mapping(values: object, name: str) -> tuple[tuple[str, Fraction], ...]:
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{name} must be a nonempty dictionary")
    result = []
    for key in sorted(values):
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be nonempty strings")
        value = _exact_fraction(values[key], f"{name}[{key}]")
        if value < 0:
            raise ValueError(f"{name} values must be nonnegative")
        result.append((key, value))
    return tuple(result)


def _parameter_names(maximum_order: int, requested_theorem_level: str) -> tuple[set[str], set[str], set[str]]:
    fixed = {
        "base_reference_scale", "fiber_reference_scale", "fiber_radius",
    }
    upper = {
        "forcing_at_zero_upper", "base_inverse_lipschitz",
        "fiber_linear_norm_upper", "base_self_lipschitz",
        "fiber_to_base_lipschitz", "base_to_fiber_lipschitz",
        "fiber_self_lipschitz", "graph_slope_upper",
        "preimage_value_coupling_upper",
    }
    lower = {"base_invertibility_lower"}
    for order in range(1, maximum_order + 1):
        upper.update({f"base_K{order}", f"base_H{order}", f"fiber_K{order}", f"fiber_H{order}"})
    for order in range(1, maximum_order + 2):
        upper.add(f"graph_D{order}")
    if requested_theorem_level in (LOCAL_LEVEL, MATCHED_LEVEL):
        fixed.update({
            "input_base_domain_radius", "output_base_domain_radius",
            "input_extension_collar_radius", "output_extension_collar_radius",
        })
        upper.update({
            "uniform_inverse_base_image_radius_upper",
            "uniform_inverse_collar_image_radius_upper",
        })
    if requested_theorem_level == MATCHED_LEVEL:
        fixed.add("uniform_inverse_base_image_radius_exact")
        upper.update({
            "uniform_forward_base_image_radius_exact",
            "graph_boundary_value_upper", "fiber_boundary_forcing_upper",
        })
    return fixed, upper, lower


def frozen_neural_graph_transform_measurement_contract(
    *, base_dimension: int, maximum_order: int,
    requested_theorem_level: str, parameters: object,
) -> FrozenNeuralGraphTransformMeasurementContract:
    if type(base_dimension) is not int or base_dimension < 1:
        raise ValueError("base_dimension must be a positive built-in integer")
    if type(maximum_order) is not int or maximum_order < 2:
        raise ValueError("maximum_order must be a built-in integer at least two")
    if requested_theorem_level not in (GLOBAL_LEVEL, LOCAL_LEVEL, MATCHED_LEVEL):
        raise ValueError("requested_theorem_level must be GLOBAL, LOCAL, or MATCHED")
    canonical = _canonical_mapping(parameters, "parameters")
    supplied = {key for key, _ in canonical}
    fixed, upper, lower = _parameter_names(maximum_order, requested_theorem_level)
    required = fixed | upper | lower
    missing = sorted(required - supplied)
    extra = sorted(supplied - required)
    if missing or extra:
        raise ValueError(f"parameter schema mismatch; missing={missing}; extra={extra}")
    values = dict(canonical)
    if min(values[name] for name in ("base_reference_scale", "fiber_reference_scale", "fiber_radius")) <= 0:
        raise ValueError("reference scales and fiber radius must be positive")
    return FrozenNeuralGraphTransformMeasurementContract(
        base_dimension=base_dimension,
        maximum_order=maximum_order,
        requested_theorem_level=requested_theorem_level,
        parameters=canonical,
        required_upper_envelope_names=tuple(sorted(upper)),
        required_lower_envelope_names=tuple(sorted(lower)),
    )


def neural_map_session_envelope(
    *, ecephys_session_id: object, specimen_id: object, split: str,
    upper_envelopes: object, lower_envelopes: object,
) -> NeuralMapSessionEnvelope:
    for value, name in ((ecephys_session_id, "ecephys_session_id"), (specimen_id, "specimen_id")):
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal() or (len(value) > 1 and value.startswith("0")):
            raise ValueError(f"{name} must be a canonical nonnegative decimal string")
    if split not in _SPLITS:
        raise ValueError("split must be calibration, development, or held_out")
    return NeuralMapSessionEnvelope(
        ecephys_session_id=ecephys_session_id,
        specimen_id=specimen_id,
        split=split,
        upper_envelopes=_canonical_mapping(upper_envelopes, "upper_envelopes"),
        lower_envelopes=_canonical_mapping(lower_envelopes, "lower_envelopes"),
    )


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _metadata_assignments(receipt: E1MetadataReceipt) -> dict[str, tuple[str, str]]:
    assignments = {}
    for line in receipt.assignment_jsonl.splitlines():
        row = json.loads(line)
        assignments[row["ecephys_session_id"]] = (row["specimen_id"], row["split"])
    return assignments


def _build_theorem_chain(contract: FrozenNeuralGraphTransformMeasurementContract):
    p = dict(contract.parameters)
    n = contract.maximum_order
    c0 = quantitative_coupled_graph_transform(
        base_dimension=contract.base_dimension,
        base_reference_scale=p["base_reference_scale"], fiber_reference_scale=p["fiber_reference_scale"],
        fiber_radius=p["fiber_radius"], forcing_at_zero_upper=p["forcing_at_zero_upper"],
        base_inverse_lipschitz=p["base_inverse_lipschitz"], fiber_linear_norm_upper=p["fiber_linear_norm_upper"],
        base_self_lipschitz=p["base_self_lipschitz"], fiber_to_base_lipschitz=p["fiber_to_base_lipschitz"],
        base_to_fiber_lipschitz=p["base_to_fiber_lipschitz"], fiber_self_lipschitz=p["fiber_self_lipschitz"],
        graph_slope_upper=p["graph_slope_upper"],
    )
    derivative = quantitative_coupled_arbitrary_order_graph_transform(
        base_dimension=contract.base_dimension,
        base_invertibility_lower=p["base_invertibility_lower"],
        preimage_value_coupling_upper=p["preimage_value_coupling_upper"],
        normalized_graph_derivative_bounds_with_point_modulus=tuple(p[f"graph_D{j}"] for j in range(1, n + 2)),
        base_map_derivative_bounds=tuple(p[f"base_K{j}"] for j in range(1, n + 1)),
        base_map_derivative_point_lipschitz_bounds=tuple(p[f"base_H{j}"] for j in range(1, n + 1)),
        fiber_map_derivative_bounds=tuple(p[f"fiber_K{j}"] for j in range(1, n + 1)),
        fiber_map_derivative_point_lipschitz_bounds=tuple(p[f"fiber_H{j}"] for j in range(1, n + 1)),
    )
    full = quantitative_full_coupled_arbitrary_order_graph_transform(
        c0_certificate=c0, derivative_certificate=derivative
    )
    if contract.requested_theorem_level == GLOBAL_LEVEL:
        return c0, derivative, full, None, None, None, None
    local = quantitative_local_coupled_arbitrary_order_graph_transform(
        global_certificate=derivative,
        base_reference_scale=p["base_reference_scale"],
        collar_moduli_covered_through_order=n + 1,
        input_base_domain_radius=p["input_base_domain_radius"],
        output_base_domain_radius=p["output_base_domain_radius"],
        uniform_inverse_base_image_radius_upper=p["uniform_inverse_base_image_radius_upper"],
        input_extension_collar_radius=p["input_extension_collar_radius"],
        output_extension_collar_radius=p["output_extension_collar_radius"],
        uniform_inverse_collar_image_radius_upper=p["uniform_inverse_collar_image_radius_upper"],
    )
    full_local = quantitative_full_local_coupled_arbitrary_order_graph_transform(
        global_certificate=full, derivative_local_certificate=local
    )
    if contract.requested_theorem_level == LOCAL_LEVEL:
        return c0, derivative, full, local, full_local, None, None
    matched = quantitative_matched_local_coupled_arbitrary_order_graph_transform(
        local_certificate=local,
        fiber_reference_scale=p["fiber_reference_scale"],
        uniform_inverse_base_image_radius_exact=p["uniform_inverse_base_image_radius_exact"],
        uniform_forward_base_image_radius_exact=p["uniform_forward_base_image_radius_exact"],
        graph_boundary_value_upper=p["graph_boundary_value_upper"],
        fiber_boundary_forcing_upper=p["fiber_boundary_forcing_upper"],
    )
    full_matched = quantitative_full_matched_local_coupled_arbitrary_order_graph_transform(
        local_certificate=full_local, derivative_matched_certificate=matched
    )
    return c0, derivative, full, local, full_local, matched, full_matched


def neural_graph_transform_measurement_bridge(
    *, metadata_receipt: E1MetadataReceipt,
    frozen_contract: FrozenNeuralGraphTransformMeasurementContract,
    session_envelopes: object,
    neural_observation_sha256: object,
    analysis_contract_sha256: object,
    preprocessing_sha256: object,
    model_family_sha256: object,
    simultaneous_coverage_kind: str,
    familywise_error_upper: object,
    coverage_family_size: int,
    provenance_status: str,
) -> NeuralGraphTransformMeasurementBridgeCertificate:
    if not isinstance(metadata_receipt, E1MetadataReceipt) or not metadata_receipt.status.accepted:
        raise ValueError("metadata_receipt must be an accepted E1 receipt")
    if not isinstance(frozen_contract, FrozenNeuralGraphTransformMeasurementContract):
        raise ValueError("frozen_contract must be a frozen neural measurement contract")
    if provenance_status not in (SYNTHETIC_FIXTURE, SOURCE_LOCKED_EMPIRICAL):
        raise ValueError("provenance_status must be a frozen bridge value")
    if not isinstance(session_envelopes, (tuple, list)) or not session_envelopes:
        raise ValueError("session_envelopes must be a nonempty tuple or list")
    sessions = tuple(session_envelopes)
    if any(not isinstance(row, NeuralMapSessionEnvelope) for row in sessions):
        raise ValueError("every session envelope must be canonical")
    if type(coverage_family_size) is not int or coverage_family_size < 1:
        raise ValueError("coverage_family_size must be a positive built-in integer")
    error = _exact_fraction(familywise_error_upper, "familywise_error_upper")
    if error <= 0 or error >= Fraction(1, 20):
        raise ValueError("familywise_error_upper must be strictly between zero and 1/20")
    observation_hash = _hash(neural_observation_sha256, "neural_observation_sha256")
    contract_hash = _hash(analysis_contract_sha256, "analysis_contract_sha256")
    preprocessing_hash = _hash(preprocessing_sha256, "preprocessing_sha256")
    family_hash = _hash(model_family_sha256, "model_family_sha256")

    failures: list[str] = []
    assignments = _metadata_assignments(metadata_receipt)
    by_session = {row.ecephys_session_id: row for row in sessions}
    if len(by_session) != len(sessions):
        failures.append("NEURAL_BRIDGE_DUPLICATE_SESSION_ENVELOPE")
    if set(by_session) != set(assignments):
        failures.append("NEURAL_BRIDGE_SESSION_SET_DOES_NOT_MATCH_METADATA_RECEIPT")
    upper_names = set(frozen_contract.required_upper_envelope_names)
    lower_names = set(frozen_contract.required_lower_envelope_names)
    p = dict(frozen_contract.parameters)
    specimen_by_split = {split: set() for split in _SPLITS}
    all_within = True
    for row in sessions:
        assigned = assignments.get(row.ecephys_session_id)
        if assigned != (row.specimen_id, row.split):
            failures.append(f"NEURAL_BRIDGE_SESSION_{row.ecephys_session_id}_ASSIGNMENT_MISMATCH")
            all_within = False
        else:
            specimen_by_split[row.split].add(row.specimen_id)
        upper = dict(row.upper_envelopes)
        lower = dict(row.lower_envelopes)
        if set(upper) != upper_names or set(lower) != lower_names:
            failures.append(f"NEURAL_BRIDGE_SESSION_{row.ecephys_session_id}_ENVELOPE_SCHEMA_MISMATCH")
            all_within = False
            continue
        for name in sorted(upper_names):
            if upper[name] > p[name]:
                failures.append(f"NEURAL_BRIDGE_SESSION_{row.ecephys_session_id}_{name}_UPPER_VIOLATION")
                all_within = False
        for name in sorted(lower_names):
            if lower[name] < p[name]:
                failures.append(f"NEURAL_BRIDGE_SESSION_{row.ecephys_session_id}_{name}_LOWER_VIOLATION")
                all_within = False
    for split in _SPLITS:
        if len(specimen_by_split[split]) < 2:
            failures.append(f"NEURAL_BRIDGE_{split.upper()}_SPECIMEN_COUNT_BELOW_TWO")
    expected_family_size = len(sessions) * (len(upper_names) + len(lower_names))
    if simultaneous_coverage_kind != SIMULTANEOUS_COVERAGE_KIND:
        failures.append("NEURAL_BRIDGE_COVERAGE_NOT_FAMILYWISE_SIMULTANEOUS")
    if coverage_family_size != expected_family_size:
        failures.append("NEURAL_BRIDGE_COVERAGE_FAMILY_SIZE_MISMATCH")

    chain = _build_theorem_chain(frozen_contract)
    for certificate in chain:
        if certificate is None:
            continue
        if certificate.validation_level is None:
            failures.extend(certificate.failure_codes or (certificate.status,))
    c0, derivative, full, local, full_local, matched, full_matched = chain
    split_counts = tuple((split, len(specimen_by_split[split])) for split in _SPLITS)
    highest_verified = None
    if full.validation_level is not None:
        highest_verified = GLOBAL_LEVEL
    if full_local is not None and full_local.validation_level is not None:
        highest_verified = LOCAL_LEVEL
    if full_matched is not None and full_matched.validation_level is not None:
        highest_verified = MATCHED_LEVEL
    scope = (
        "CONDITIONAL_SOURCE_LOCKED_NEURAL_ENVELOPE_TO_FULL_GRAPH_TRANSFORM_BRIDGE; "
        "INPUT_INTERVALS_ARE_MEASUREMENT_PREMISES_AND_DO_NOT_IDENTIFY_CONSCIOUSNESS"
    )
    # E1MetadataReceipt is deliberately a local canonicalization receipt.  It
    # contains no authenticated remote-response execution evidence, so a
    # caller-supplied provenance label can never promote this apparatus to an
    # empirical result.
    external_remote_verified = False
    empirical = False
    common = dict(
        claim_scope=scope, failure_codes=tuple(failures),
        provenance_status=provenance_status,
        metadata_table_sha256=metadata_receipt.canonical_table_sha256,
        assignment_sha256=metadata_receipt.assignment_sha256,
        neural_observation_sha256=observation_hash,
        analysis_contract_sha256=contract_hash,
        preprocessing_sha256=preprocessing_hash,
        model_family_sha256=family_hash,
        simultaneous_coverage_kind=simultaneous_coverage_kind,
        familywise_error_upper=error,
        coverage_family_size=coverage_family_size,
        expected_coverage_family_size=expected_family_size,
        session_count=len(sessions),
        session_assignments=tuple(sorted(
            ((row.ecephys_session_id, row.specimen_id, row.split) for row in sessions),
            key=lambda value: int(value[0]),
        )),
        specimen_count_by_split=split_counts,
        frozen_contract=frozen_contract,
        requested_theorem_level=frozen_contract.requested_theorem_level,
        highest_verified_theorem_level=highest_verified,
        c0_certificate=c0, derivative_certificate=derivative,
        full_global_certificate=full, derivative_local_certificate=local,
        full_local_certificate=full_local, derivative_matched_certificate=matched,
        full_matched_certificate=full_matched,
        all_session_envelopes_within_frozen_family=all_within,
        external_remote_metadata_receipt_verified=external_remote_verified,
        source_locked_empirical_neural_graph_result=empirical,
        consciousness_claim_admitted=False,
        dimension_4_6_claim_admitted=False,
    )
    if failures:
        return NeuralGraphTransformMeasurementBridgeCertificate(
            status=failures[0], validation_level=None, robust_interior=False, **common
        )
    status = (
        "VALIDATED_DECLARED_SOURCE_LOCKED_NEURAL_GRAPH_SUMMARY_APPARATUS_ONLY"
        if provenance_status == SOURCE_LOCKED_EMPIRICAL
        else "VALIDATED_SYNTHETIC_NEURAL_GRAPH_MEASUREMENT_BRIDGE"
    )
    return NeuralGraphTransformMeasurementBridgeCertificate(
        status=status, validation_level=status, robust_interior=False, **common
    )


def neural_graph_and_signal_rank_joint_certificate(
    *,
    measurement_certificate: NeuralGraphTransformMeasurementBridgeCertificate,
    dimension_certificate: ConsciousMomentDimensionProtocolCertificate,
    dimension_session_ids: object,
) -> NeuralGraphAndSignalRankJointCertificate:
    """Join the two apparatuses while forbidding graph/rank identity claims."""
    if not isinstance(measurement_certificate, NeuralGraphTransformMeasurementBridgeCertificate):
        raise ValueError("measurement_certificate must be a neural measurement bridge")
    if not isinstance(dimension_certificate, ConsciousMomentDimensionProtocolCertificate):
        raise ValueError("dimension_certificate must be a conscious-moment dimension certificate")
    if not isinstance(dimension_session_ids, (tuple, list)):
        raise ValueError("dimension_session_ids must be a tuple or list")
    session_ids = tuple(dimension_session_ids)
    if any(not isinstance(value, str) or not value.isdecimal() for value in session_ids):
        raise ValueError("dimension session IDs must be canonical decimal strings")
    failures = list(measurement_certificate.failure_codes)
    failures.extend(dimension_certificate.failure_codes)
    if measurement_certificate.validation_level is None and not measurement_certificate.failure_codes:
        failures.append("NEURAL_JOINT_MEASUREMENT_CERTIFICATE_NOT_VERIFIED")
    if dimension_certificate.validation_level is None and not dimension_certificate.failure_codes:
        failures.append("NEURAL_JOINT_DIMENSION_CERTIFICATE_NOT_VERIFIED")
    if len(set(session_ids)) != len(session_ids):
        failures.append("NEURAL_JOINT_DIMENSION_SESSION_IDS_NOT_UNIQUE")
    expected = tuple(
        session for session, _specimen, split in measurement_certificate.session_assignments
        if split == "held_out"
    )
    if tuple(sorted(session_ids, key=int)) != expected:
        failures.append("NEURAL_JOINT_DIMENSION_SESSIONS_DO_NOT_MATCH_HELDOUT_SPLIT")
    if len(session_ids) != len(dimension_certificate.session_metrics):
        failures.append("NEURAL_JOINT_DIMENSION_SESSION_COUNT_MISMATCH")
    if measurement_certificate.provenance_status != dimension_certificate.provenance_status:
        failures.append("NEURAL_JOINT_PROVENANCE_STATUS_MISMATCH")

    graph_dimension = measurement_certificate.frozen_contract.base_dimension
    selected = dimension_certificate.selected_signal_rank
    equal_numerically = selected is not None and graph_dimension == selected
    scope = (
        "JOINT_GRAPH_TRANSFORM_AND_HELDOUT_NEURAL_SIGNAL_RANK_APPARATUS; "
        "NUMERICAL_DIMENSION_EQUALITY_DOES_NOT_ESTABLISH OBJECT_IDENTITY_OR_CONSCIOUSNESS"
    )
    common = dict(
        claim_scope=scope, failure_codes=tuple(failures),
        measurement_certificate=measurement_certificate,
        dimension_certificate=dimension_certificate,
        heldout_session_ids=tuple(sorted(session_ids, key=int)),
        graph_base_dimension=graph_dimension,
        selected_neural_signal_rank=selected,
        graph_dimension_equals_signal_rank_numerically=equal_numerically,
        graph_dimension_signal_rank_identity_claim_admitted=False,
        source_locked_joint_empirical_result=False,
        consciousness_claim_admitted=False,
    )
    if failures:
        return NeuralGraphAndSignalRankJointCertificate(
            status=failures[0], validation_level=None, **common
        )
    status = "VALIDATED_JOINT_NEURAL_GRAPH_AND_SIGNAL_RANK_APPARATUS_ONLY"
    return NeuralGraphAndSignalRankJointCertificate(
        status=status, validation_level=status, **common
    )


__all__ = [
    "FrozenNeuralGraphTransformMeasurementContract",
    "GLOBAL_LEVEL",
    "LOCAL_LEVEL",
    "MATCHED_LEVEL",
    "NeuralGraphTransformMeasurementBridgeCertificate",
    "NeuralGraphAndSignalRankJointCertificate",
    "NeuralMapSessionEnvelope",
    "SIMULTANEOUS_COVERAGE_KIND",
    "SOURCE_LOCKED_EMPIRICAL",
    "SYNTHETIC_FIXTURE",
    "frozen_neural_graph_transform_measurement_contract",
    "neural_graph_transform_measurement_bridge",
    "neural_graph_and_signal_rank_joint_certificate",
    "neural_map_session_envelope",
]
