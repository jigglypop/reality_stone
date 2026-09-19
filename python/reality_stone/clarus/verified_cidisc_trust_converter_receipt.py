"""Root-key trust manifest and native-to-canonical converter content receipt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import re

if __package__:
    from .verified_dimension_signed_acquisition_receipt import ed25519_verify_strict
    from .verified_signed_conformal_interval_defective_edge_rank import (
        VerifiedSignedConformalIntervalDefectiveEdgeRank,
    )
else:
    from verified_dimension_signed_acquisition_receipt import ed25519_verify_strict  # type: ignore[no-redef]
    from verified_signed_conformal_interval_defective_edge_rank import (  # type: ignore[no-redef]
        VerifiedSignedConformalIntervalDefectiveEdgeRank,
    )


_ANCHOR_DOMAIN = b"CE-CIDISC-INSTITUTIONAL-TRUST-ANCHOR-v1\x00"
_CONVERTER_DOMAIN = b"CE-CIDISC-NATIVE-CONVERTER-RECEIPT-v1\x00"
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _field(value: bytes) -> bytes:
    if len(value) >= 1 << 64:
        raise ValueError("canonical field is too long")
    return len(value).to_bytes(8, "big") + value


def _text(value: object, name: str) -> bytes:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    encoded = value.encode("utf-8")
    if b"\x00" in encoded:
        raise ValueError(f"{name} may not contain NUL")
    return encoded


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _calendar_date(value: object, name: str) -> date:
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        raise ValueError(f"{name} must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a real Gregorian calendar date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must use canonical zero-padded YYYY-MM-DD")
    return parsed


def canonical_cidisc_trust_anchor_message(
    *,
    institution_id: str,
    root_key_id: str,
    coverage_signer_key_id: str,
    coverage_signer_public_key: bytes,
    valid_from_date: str,
    valid_until_date: str,
) -> bytes:
    if not isinstance(coverage_signer_public_key, bytes) or len(coverage_signer_public_key) != 32:
        raise ValueError("coverage signer public key must be 32 bytes")
    valid_from = _calendar_date(valid_from_date, "valid_from_date")
    valid_until = _calendar_date(valid_until_date, "valid_until_date")
    if valid_from > valid_until:
        raise ValueError("trust-anchor validity interval must be ordered")
    return (
        _ANCHOR_DOMAIN
        + _field(_text(institution_id, "institution_id"))
        + _field(_text(root_key_id, "root_key_id"))
        + _field(_text(coverage_signer_key_id, "coverage_signer_key_id"))
        + _field(coverage_signer_public_key)
        + valid_from_date.encode("ascii")
        + valid_until_date.encode("ascii")
    )


def canonical_cidisc_converter_receipt_message(
    *,
    trust_anchor_message_sha256: str,
    native_calibration_sha256: str,
    native_heldout_sha256: str,
    converter_binary_sha256: str,
    converter_contract_sha256: str,
    canonical_calibration_output_sha256: str,
    canonical_heldout_output_sha256: str,
) -> bytes:
    digests = []
    for value in (
        trust_anchor_message_sha256, native_calibration_sha256,
        native_heldout_sha256, converter_binary_sha256, converter_contract_sha256,
        canonical_calibration_output_sha256, canonical_heldout_output_sha256,
    ):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("converter receipt digests must be lowercase SHA-256")
        digests.append(bytes.fromhex(value))
    return _CONVERTER_DOMAIN + b"".join(digests)


@dataclass(frozen=True)
class VerifiedCidiscTrustConverterReceipt:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    signed_cidisc_certificate: VerifiedSignedConformalIntervalDefectiveEdgeRank
    trust_anchor_message_sha256: str
    converter_receipt_message_sha256: str
    native_calibration_sha256: str
    native_heldout_sha256: str
    converter_binary_sha256: str
    converter_contract_sha256: str
    evaluation_date: str
    trust_anchor_time_window_verified: bool
    root_key_fingerprint_matches: bool
    root_authorization_signature_verified: bool
    coverage_key_matches_signed_cidisc: bool
    converter_content_signature_verified: bool
    two_signature_content_chain_verified: bool
    externally_distributed_root_fingerprint_verified: bool
    converter_execution_attested_by_hardware: bool
    source_locked_empirical_rank_result: bool
    consciousness_dimension_claim_admitted: bool


def verified_cidisc_trust_converter_receipt(
    *,
    signed_cidisc_certificate: object,
    institution_id: object,
    root_key_id: object,
    coverage_signer_key_id: object,
    coverage_signer_public_key: object,
    valid_from_date: object,
    valid_until_date: object,
    evaluation_date: object,
    root_ed25519_public_key: object,
    expected_root_public_key_sha256: object,
    root_detached_signature: object,
    native_calibration_bytes: object,
    native_heldout_bytes: object,
    converter_binary_bytes: object,
    converter_contract_bytes: object,
    converter_detached_signature: object,
) -> VerifiedCidiscTrustConverterReceipt:
    if not isinstance(signed_cidisc_certificate, VerifiedSignedConformalIntervalDefectiveEdgeRank):
        raise ValueError("signed_cidisc_certificate must be a verified runtime receipt")
    byte_fields = {
        "coverage_signer_public_key": coverage_signer_public_key,
        "root_ed25519_public_key": root_ed25519_public_key,
        "root_detached_signature": root_detached_signature,
        "native_calibration_bytes": native_calibration_bytes,
        "native_heldout_bytes": native_heldout_bytes,
        "converter_binary_bytes": converter_binary_bytes,
        "converter_contract_bytes": converter_contract_bytes,
        "converter_detached_signature": converter_detached_signature,
    }
    if any(not isinstance(value, bytes) or not value for value in byte_fields.values()):
        raise ValueError("trust/converter receipt byte fields must be nonempty bytes")
    anchor_message = canonical_cidisc_trust_anchor_message(
        institution_id=institution_id, root_key_id=root_key_id,
        coverage_signer_key_id=coverage_signer_key_id,
        coverage_signer_public_key=coverage_signer_public_key,
        valid_from_date=valid_from_date, valid_until_date=valid_until_date,
    )
    evaluation = _calendar_date(evaluation_date, "evaluation_date")
    in_time = date.fromisoformat(valid_from_date) <= evaluation <= date.fromisoformat(valid_until_date)
    root_hash = _sha256(root_ed25519_public_key)
    root_match = root_hash == expected_root_public_key_sha256
    root_signature_ok = ed25519_verify_strict(
        root_ed25519_public_key, anchor_message, root_detached_signature
    )
    key_matches = bool(
        signed_cidisc_certificate.cryptographic_coverage_bundle_verified
        and signed_cidisc_certificate.signer_key_id == coverage_signer_key_id
        and signed_cidisc_certificate.signer_public_key_sha256 == _sha256(coverage_signer_public_key)
    )
    native_cal_hash = _sha256(native_calibration_bytes)
    native_hold_hash = _sha256(native_heldout_bytes)
    converter_hash = _sha256(converter_binary_bytes)
    converter_contract_hash = _sha256(converter_contract_bytes)
    anchor_hash = _sha256(anchor_message)
    converter_message = canonical_cidisc_converter_receipt_message(
        trust_anchor_message_sha256=anchor_hash,
        native_calibration_sha256=native_cal_hash,
        native_heldout_sha256=native_hold_hash,
        converter_binary_sha256=converter_hash,
        converter_contract_sha256=converter_contract_hash,
        canonical_calibration_output_sha256=signed_cidisc_certificate.calibration_canonical_bytes_sha256,
        canonical_heldout_output_sha256=signed_cidisc_certificate.heldout_canonical_bytes_sha256,
    )
    converter_signature_ok = ed25519_verify_strict(
        coverage_signer_public_key, converter_message, converter_detached_signature
    )
    failures: list[str] = []
    if signed_cidisc_certificate.validation_level is None:
        failures.append("CIDISC_TRUST_BASE_SIGNED_CERTIFICATE_FAILED")
    if not in_time: failures.append("CIDISC_TRUST_ANCHOR_OUTSIDE_VALIDITY_WINDOW")
    if not root_match: failures.append("CIDISC_TRUST_ROOT_FINGERPRINT_MISMATCH")
    if not root_signature_ok: failures.append("CIDISC_TRUST_ROOT_SIGNATURE_INVALID")
    if not key_matches: failures.append("CIDISC_TRUST_COVERAGE_KEY_CERTIFICATE_MISMATCH")
    if not converter_signature_ok: failures.append("CIDISC_CONVERTER_CONTENT_SIGNATURE_INVALID")
    success = not failures
    status = "VERIFIED_CIDISC_TWO_SIGNATURE_TRUST_AND_CONVERTER_CONTENT_CHAIN" if success else failures[0]
    return VerifiedCidiscTrustConverterReceipt(
        status=status, validation_level=status if success else None,
        failure_codes=tuple(failures), signed_cidisc_certificate=signed_cidisc_certificate,
        trust_anchor_message_sha256=anchor_hash,
        converter_receipt_message_sha256=_sha256(converter_message),
        native_calibration_sha256=native_cal_hash,
        native_heldout_sha256=native_hold_hash,
        converter_binary_sha256=converter_hash,
        converter_contract_sha256=converter_contract_hash,
        evaluation_date=evaluation_date,
        trust_anchor_time_window_verified=in_time,
        root_key_fingerprint_matches=root_match,
        root_authorization_signature_verified=root_signature_ok,
        coverage_key_matches_signed_cidisc=key_matches,
        converter_content_signature_verified=converter_signature_ok,
        two_signature_content_chain_verified=success,
        externally_distributed_root_fingerprint_verified=False,
        converter_execution_attested_by_hardware=False,
        source_locked_empirical_rank_result=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = [
    "VerifiedCidiscTrustConverterReceipt",
    "canonical_cidisc_trust_anchor_message",
    "canonical_cidisc_converter_receipt_message",
    "verified_cidisc_trust_converter_receipt",
]
