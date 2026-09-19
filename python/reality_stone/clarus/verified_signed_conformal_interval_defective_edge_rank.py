"""Canonical-byte and strict Ed25519 receipt for conformal interval edge rank."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib

if __package__:
    from .quantitative_graph_transform import _exact_fraction
    from .verified_conformal_interval_defective_edge_disconnection_spectral_rank import (
        VerifiedConformalIntervalDefectiveEdgeDisconnectionSpectralRank,
        verified_conformal_interval_defective_edge_disconnection_spectral_rank,
    )
    from .verified_dimension_signed_acquisition_receipt import ed25519_verify_strict
else:
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]
    from verified_conformal_interval_defective_edge_disconnection_spectral_rank import (  # type: ignore[no-redef]
        VerifiedConformalIntervalDefectiveEdgeDisconnectionSpectralRank,
        verified_conformal_interval_defective_edge_disconnection_spectral_rank,
    )
    from verified_dimension_signed_acquisition_receipt import ed25519_verify_strict  # type: ignore[no-redef]


_VECTOR_DOMAIN = b"CE-CIDISC-EXACT-ERROR-VECTORS-v1\x00"
_SIGNATURE_DOMAIN = b"CE-CIDISC-SIGNED-COVERAGE-v1\x00"


def _u64(value: int) -> bytes:
    if type(value) is not int or value < 0 or value >= 1 << 64:
        raise ValueError("canonical length/count must be a uint64")
    return value.to_bytes(8, "big")


def _field(payload: bytes) -> bytes:
    return _u64(len(payload)) + payload


def _canonical_text(value: object, name: str) -> bytes:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    encoded = value.encode("utf-8")
    if b"\x00" in encoded:
        raise ValueError(f"{name} may not contain NUL")
    return encoded


def _fraction_bytes(value: object, name: str) -> bytes:
    fraction = _exact_fraction(value, name)
    return f"{fraction.numerator}/{fraction.denominator}".encode("ascii")


def canonical_exact_error_vectors_bytes(
    *,
    vector_role: str,
    component_labels: object,
    component_scales: object,
    absolute_error_vectors: object,
) -> bytes:
    """Encode labels, scales and exact nonnegative error vectors without JSON ambiguity."""
    role = _canonical_text(vector_role, "vector_role")
    if vector_role not in ("calibration", "heldout"):
        raise ValueError("vector_role must be calibration or heldout")
    if not isinstance(component_labels, (tuple, list)) or not component_labels:
        raise ValueError("component_labels must be a nonempty sequence")
    labels = tuple(_canonical_text(value, f"component_labels[{i}]") for i, value in enumerate(component_labels))
    if len(set(labels)) != len(labels):
        raise ValueError("component labels must be unique")
    if not isinstance(component_scales, (tuple, list)) or len(component_scales) != len(labels):
        raise ValueError("component_scales must match component_labels")
    scales = tuple(_fraction_bytes(value, f"component_scales[{i}]") for i, value in enumerate(component_scales))
    if not isinstance(absolute_error_vectors, (tuple, list)) or not absolute_error_vectors:
        raise ValueError("absolute_error_vectors must be a nonempty sequence")
    rows: list[tuple[bytes, ...]] = []
    for i, row in enumerate(absolute_error_vectors):
        if not isinstance(row, (tuple, list)) or len(row) != len(labels):
            raise ValueError("absolute error vector width must match component labels")
        parsed = []
        for j, value in enumerate(row):
            fraction = _exact_fraction(value, f"absolute_error_vectors[{i}][{j}]")
            if fraction < 0:
                raise ValueError("absolute errors must be nonnegative")
            parsed.append(f"{fraction.numerator}/{fraction.denominator}".encode("ascii"))
        rows.append(tuple(parsed))
    payload = bytearray(_VECTOR_DOMAIN)
    payload.extend(_field(role))
    payload.extend(_u64(len(labels)))
    for label, scale in zip(labels, scales, strict=True):
        payload.extend(_field(label)); payload.extend(_field(scale))
    payload.extend(_u64(len(rows)))
    for row in rows:
        for value in row:
            payload.extend(_field(value))
    return bytes(payload)


def canonical_signed_coverage_message(
    *,
    calibration_data_sha256: str,
    heldout_data_sha256: str,
    coverage_contract_sha256: str,
    signer_key_id: str,
) -> bytes:
    digests = []
    for value, name in (
        (calibration_data_sha256, "calibration_data_sha256"),
        (heldout_data_sha256, "heldout_data_sha256"),
        (coverage_contract_sha256, "coverage_contract_sha256"),
    ):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
        digests.append(bytes.fromhex(value))
    key_id = _canonical_text(signer_key_id, "signer_key_id")
    return _SIGNATURE_DOMAIN + _field(key_id) + b"".join(digests)


@dataclass(frozen=True)
class VerifiedSignedConformalIntervalDefectiveEdgeRank:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    conformal_rank_certificate: VerifiedConformalIntervalDefectiveEdgeDisconnectionSpectralRank | None
    calibration_canonical_bytes_sha256: str
    heldout_canonical_bytes_sha256: str
    coverage_contract_sha256: str
    signed_message_sha256: str
    signer_key_id: str
    signer_public_key_sha256: str
    content_hashes_match_expected: bool
    signer_key_fingerprint_matches: bool
    strict_ed25519_signature_verified: bool
    cryptographic_coverage_bundle_verified: bool
    externally_frozen_trust_anchor_verified: bool
    device_native_error_converter_verified: bool
    source_locked_empirical_rank_result: bool
    consciousness_dimension_claim_admitted: bool


def verified_signed_conformal_interval_defective_edge_rank(
    nominal_disconnected_transition: object,
    *,
    component_labels: object,
    component_scales: object,
    calibration_absolute_error_vectors: object,
    heldout_absolute_error_vectors: object,
    coverage_contract_bytes: object,
    expected_calibration_data_sha256: object,
    expected_heldout_data_sha256: object,
    expected_coverage_contract_sha256: object,
    signer_key_id: object,
    signer_ed25519_public_key: object,
    expected_signer_public_key_sha256: object,
    detached_ed25519_signature: object,
    **conformal_rank_arguments: object,
) -> VerifiedSignedConformalIntervalDefectiveEdgeRank:
    calibration_bytes = canonical_exact_error_vectors_bytes(
        vector_role="calibration", component_labels=component_labels,
        component_scales=component_scales,
        absolute_error_vectors=calibration_absolute_error_vectors,
    )
    heldout_bytes = canonical_exact_error_vectors_bytes(
        vector_role="heldout", component_labels=component_labels,
        component_scales=component_scales,
        absolute_error_vectors=heldout_absolute_error_vectors,
    )
    if not isinstance(coverage_contract_bytes, bytes) or not coverage_contract_bytes:
        raise ValueError("coverage_contract_bytes must be nonempty bytes")
    calibration_hash = hashlib.sha256(calibration_bytes).hexdigest()
    heldout_hash = hashlib.sha256(heldout_bytes).hexdigest()
    contract_hash = hashlib.sha256(coverage_contract_bytes).hexdigest()
    expected = tuple(
        value if isinstance(value, str) else ""
        for value in (
            expected_calibration_data_sha256,
            expected_heldout_data_sha256,
            expected_coverage_contract_sha256,
        )
    )
    message = canonical_signed_coverage_message(
        calibration_data_sha256=calibration_hash,
        heldout_data_sha256=heldout_hash,
        coverage_contract_sha256=contract_hash,
        signer_key_id=signer_key_id,
    )
    if not isinstance(signer_ed25519_public_key, bytes) or not isinstance(detached_ed25519_signature, bytes):
        raise ValueError("Ed25519 public key and detached signature must be bytes")
    key_hash = hashlib.sha256(signer_ed25519_public_key).hexdigest()
    hashes_match = expected == (calibration_hash, heldout_hash, contract_hash)
    key_match = key_hash == expected_signer_public_key_sha256
    signature_ok = ed25519_verify_strict(
        signer_ed25519_public_key, message, detached_ed25519_signature
    )
    failures: list[str] = []
    if not hashes_match: failures.append("SIGNED_CIDISC_CONTENT_HASH_MISMATCH")
    if not key_match: failures.append("SIGNED_CIDISC_SIGNER_KEY_FINGERPRINT_MISMATCH")
    if not signature_ok: failures.append("SIGNED_CIDISC_ED25519_SIGNATURE_INVALID")
    certificate = None
    if not failures:
        certificate = verified_conformal_interval_defective_edge_disconnection_spectral_rank(
            nominal_disconnected_transition,
            component_labels=component_labels, component_scales=component_scales,
            calibration_absolute_error_vectors=calibration_absolute_error_vectors,
            heldout_absolute_error_vectors=heldout_absolute_error_vectors,
            calibration_data_sha256=calibration_hash,
            heldout_data_sha256=heldout_hash,
            coverage_contract_sha256=contract_hash,
            **conformal_rank_arguments,
        )
        if certificate.validation_level is None:
            failures.extend(certificate.failure_codes)
    success = not failures and certificate is not None
    status = "VERIFIED_SIGNED_CONFORMAL_INTERVAL_DEFECTIVE_EDGE_RANK_BUNDLE" if success else failures[0]
    return VerifiedSignedConformalIntervalDefectiveEdgeRank(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), conformal_rank_certificate=certificate,
        calibration_canonical_bytes_sha256=calibration_hash,
        heldout_canonical_bytes_sha256=heldout_hash,
        coverage_contract_sha256=contract_hash,
        signed_message_sha256=hashlib.sha256(message).hexdigest(),
        signer_key_id=signer_key_id, signer_public_key_sha256=key_hash,
        content_hashes_match_expected=hashes_match,
        signer_key_fingerprint_matches=key_match,
        strict_ed25519_signature_verified=signature_ok,
        cryptographic_coverage_bundle_verified=success,
        externally_frozen_trust_anchor_verified=False,
        device_native_error_converter_verified=False,
        source_locked_empirical_rank_result=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = [
    "VerifiedSignedConformalIntervalDefectiveEdgeRank",
    "canonical_exact_error_vectors_bytes", "canonical_signed_coverage_message",
    "verified_signed_conformal_interval_defective_edge_rank",
]
