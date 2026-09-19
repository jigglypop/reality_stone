"""Canonical archive-byte receipt for the complete dimension execution chain."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib

if __package__:
    from .conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate
    from .verified_dimension_observation_manifest import (
        VerifiedDimensionObservationManifest,
        canonical_exact_from_bytes,
        verified_dimension_observation_manifest,
    )
else:
    from conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate  # type: ignore[no-redef]
    from verified_dimension_observation_manifest import (  # type: ignore[no-redef]
        VerifiedDimensionObservationManifest,
        canonical_exact_from_bytes,
        verified_dimension_observation_manifest,
    )


_DOMAIN_TAGS = (
    "CE-DIM-DEVELOPMENT-v1",
    "CE-DIM-HELDOUT-v1",
    "CE-DIM-PREPROCESSING-v1",
    "CE-DIM-EIGENBASIS-v1",
    "CE-DIM-BOOTSTRAP-v1",
    "CE-DIM-EXECUTION-CONTRACT-v1",
)


@dataclass(frozen=True)
class VerifiedDimensionSourceBytesReceipt:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    observation_manifest: VerifiedDimensionObservationManifest | None
    domain_payload_sha256: tuple[str, ...]
    source_bundle_sha256: str
    expected_source_bundle_sha256_matches: bool
    canonical_archive_bytes_verified: bool
    domain_tags_and_shapes_verified: bool
    session_identity_consistency_verified: bool
    execution_contract_matches_base_certificate: bool
    canonical_source_archive_receipt_verified: bool
    external_acquisition_signature_verified: bool
    original_device_byte_parser_verified: bool
    source_locked_empirical_dimension_result: bool
    consciousness_dimension_claim_admitted: bool


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_dimension_source_bundle_sha256(*payloads: bytes) -> str:
    if len(payloads) != 6 or not all(isinstance(payload, bytes) for payload in payloads):
        raise ValueError("dimension source bundle requires exactly six byte payloads")
    digest = hashlib.sha256()
    digest.update(b"CE-DIM-SOURCE-BUNDLE-v1\x00")
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _expected_hash(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected_source_bundle_sha256 must be a lowercase hexadecimal SHA-256")
    return value


def _exact_integer(value: object, name: str) -> int:
    if not isinstance(value, (int, Fraction)) or isinstance(value, bool):
        raise ValueError(f"{name} must be an exact integer")
    exact = Fraction(value)
    if exact.denominator != 1:
        raise ValueError(f"{name} must be an exact integer")
    return exact.numerator


def _exact_integer_tree(value: object, name: str) -> object:
    if isinstance(value, (tuple, list)):
        return tuple(_exact_integer_tree(item, name) for item in value)
    return _exact_integer(value, name)


def verified_dimension_source_bytes_receipt(
    *,
    base_dimension_certificate: ConsciousMomentDimensionProtocolCertificate,
    development_archive_bytes: bytes,
    heldout_archive_bytes: bytes,
    preprocessing_archive_bytes: bytes,
    eigenbasis_archive_bytes: bytes,
    bootstrap_archive_bytes: bytes,
    execution_contract_archive_bytes: bytes,
    expected_source_bundle_sha256: object,
) -> VerifiedDimensionSourceBytesReceipt:
    payload_bytes = (
        development_archive_bytes,
        heldout_archive_bytes,
        preprocessing_archive_bytes,
        eigenbasis_archive_bytes,
        bootstrap_archive_bytes,
        execution_contract_archive_bytes,
    )
    if not all(isinstance(payload, bytes) for payload in payload_bytes):
        raise ValueError("all dimension archive payloads must be bytes")
    decoded = tuple(canonical_exact_from_bytes(payload) for payload in payload_bytes)
    hashes = tuple(_sha256(payload) for payload in payload_bytes)
    bundle_hash = canonical_dimension_source_bundle_sha256(*payload_bytes)
    bundle_match = bundle_hash == _expected_hash(expected_source_bundle_sha256)

    expected_lengths = (4, 6, 3, 3, 4, 14)
    domain_ok = all(
        isinstance(payload, tuple)
        and len(payload) == length
        and payload[0] == tag
        for payload, tag, length in zip(decoded, _DOMAIN_TAGS, expected_lengths, strict=True)
    )
    failures: list[str] = []
    if not bundle_match:
        failures.append("DIMENSION_SOURCE_BUNDLE_HASH_MISMATCH")
    if not domain_ok:
        failures.append("DIMENSION_SOURCE_DOMAIN_OR_SHAPE_MISMATCH")

    session_ok = False
    contract_ok = False
    manifest = None
    if domain_ok:
        development, heldout, preprocessing, eigenbasis, bootstrap, contract = decoded
        session_ids = development[1]
        session_ok = session_ids == heldout[1] == preprocessing[1] == eigenbasis[1] == bootstrap[1]
        if not session_ok:
            failures.append("DIMENSION_SOURCE_SESSION_ID_MISMATCH")
        contract_ok = (
            contract[1] == base_dimension_certificate.frozen_dimension_menu
            and contract[2] == base_dimension_certificate.candidate_band
            and contract[3] == base_dimension_certificate.selected_signal_rank
        )
        if not contract_ok:
            failures.append("DIMENSION_SOURCE_BASE_CONTRACT_MISMATCH")
        if not failures:
            manifest = verified_dimension_observation_manifest(
                base_dimension_certificate=base_dimension_certificate,
                session_ids=session_ids,
                development_observation_ids_by_session=development[2],
                conscious_heldout_observation_ids_by_session=heldout[2],
                control_heldout_observation_ids_by_session=heldout[4],
                raw_development_observations_by_session=development[3],
                raw_conscious_heldout_observations_by_session=heldout[3],
                raw_control_heldout_observations_by_session=heldout[5],
                coordinate_reference_scales_by_session=preprocessing[2],
                covariance_eigenbasis_witnesses_by_session=eigenbasis[2],
                complexity_penalty=contract[4],
                bootstrap_window_indices_by_session=_exact_integer_tree(
                    bootstrap[3], "bootstrap_window_index"
                ),
                frozen_block_length=_exact_integer(bootstrap[2], "frozen_block_length"),
                window_partition_kind=contract[5],
                block_bootstrap_kind=contract[6],
                minimum_conscious_band_window_fraction=contract[7],
                minimum_selected_rank_window_fraction=contract[8],
                maximum_conscious_rank_transition_fraction=contract[9],
                minimum_selected_rank_longest_dwell_fraction=contract[10],
                minimum_conscious_control_band_gap=contract[11],
                minimum_bootstrap_band_fraction=contract[12],
                minimum_bootstrap_selected_rank_fraction=contract[13],
                expected_development_payload_sha256=hashes[0],
                expected_heldout_payload_sha256=hashes[1],
                expected_preprocessing_payload_sha256=hashes[2],
                expected_eigenbasis_payload_sha256=hashes[3],
                expected_bootstrap_payload_sha256=hashes[4],
                expected_execution_contract_sha256=hashes[5],
            )
            if manifest.validation_level is None:
                failures.extend(manifest.failure_codes)

    success = not failures and manifest is not None
    status = "VERIFIED_CANONICAL_DIMENSION_SOURCE_BYTES_CHAIN" if success else failures[0]
    return VerifiedDimensionSourceBytesReceipt(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        observation_manifest=manifest,
        domain_payload_sha256=hashes,
        source_bundle_sha256=bundle_hash,
        expected_source_bundle_sha256_matches=bundle_match,
        canonical_archive_bytes_verified=True,
        domain_tags_and_shapes_verified=domain_ok,
        session_identity_consistency_verified=session_ok,
        execution_contract_matches_base_certificate=contract_ok,
        canonical_source_archive_receipt_verified=success,
        external_acquisition_signature_verified=False,
        original_device_byte_parser_verified=False,
        source_locked_empirical_dimension_result=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = [
    "VerifiedDimensionSourceBytesReceipt",
    "canonical_dimension_source_bundle_sha256",
    "verified_dimension_source_bytes_receipt",
]
