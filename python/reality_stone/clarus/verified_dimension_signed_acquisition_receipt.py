"""Strict Ed25519 signature gate for a canonical dimension source bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

if __package__:
    from .conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate
    from .verified_dimension_source_bytes_receipt import (
        VerifiedDimensionSourceBytesReceipt,
        canonical_dimension_source_bundle_sha256,
        verified_dimension_source_bytes_receipt,
    )
else:
    from conscious_moment_dimension_protocol import ConsciousMomentDimensionProtocolCertificate  # type: ignore[no-redef]
    from verified_dimension_source_bytes_receipt import (  # type: ignore[no-redef]
        VerifiedDimensionSourceBytesReceipt,
        canonical_dimension_source_bundle_sha256,
        verified_dimension_source_bytes_receipt,
    )


_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)
_IDENTITY = (0, 1, 1, 0)


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x_squared = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x_squared == 0:
        return None if sign else 0
    x = pow(x_squared, (_P + 3) // 8, _P)
    if (x * x - x_squared) % _P:
        x = x * _SQRT_M1 % _P
    if (x * x - x_squared) % _P:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_BASE_Y = 4 * pow(5, _P - 2, _P) % _P
_BASE_X = _recover_x(_BASE_Y, 0)
assert _BASE_X is not None
_BASE = (_BASE_X, _BASE_Y, 1, _BASE_X * _BASE_Y % _P)


def _point_add(left, right):
    a = (left[1] - left[0]) * (right[1] - right[0]) % _P
    b = (left[1] + left[0]) * (right[1] + right[0]) % _P
    c = 2 * left[3] * right[3] * _D % _P
    d = 2 * left[2] * right[2] % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return e * f, g * h, f * g, e * h


def _point_mul(scalar: int, point):
    result = _IDENTITY
    while scalar > 0:
        if scalar & 1:
            result = _point_add(result, point)
        point = _point_add(point, point)
        scalar >>= 1
    return result


def _point_equal(left, right) -> bool:
    return (
        (left[0] * right[2] - right[0] * left[2]) % _P == 0
        and (left[1] * right[2] - right[1] * left[2]) % _P == 0
    )


def _point_compress(point) -> bytes:
    inverse_z = pow(point[2], _P - 2, _P)
    x = point[0] * inverse_z % _P
    y = point[1] * inverse_z % _P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _point_decompress(encoded: bytes):
    if len(encoded) != 32:
        return None
    packed = int.from_bytes(encoded, "little")
    sign = packed >> 255
    y = packed & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    if x is None:
        return None
    point = (x, y, 1, x * y % _P)
    if _point_compress(point) != encoded:
        return None
    return point


def ed25519_verify_strict(public_key: object, message: object, signature: object) -> bool:
    """Verify pure Ed25519 with canonical encodings and prime-subgroup checks."""
    if not isinstance(public_key, bytes) or not isinstance(message, bytes) or not isinstance(signature, bytes):
        return False
    if len(public_key) != 32 or len(signature) != 64:
        return False
    public_point = _point_decompress(public_key)
    r_encoded = signature[:32]
    r_point = _point_decompress(r_encoded)
    scalar = int.from_bytes(signature[32:], "little")
    if public_point is None or r_point is None or scalar >= _L:
        return False
    if _point_equal(public_point, _IDENTITY):
        return False
    if not _point_equal(_point_mul(_L, public_point), _IDENTITY):
        return False
    if not _point_equal(_point_mul(_L, r_point), _IDENTITY):
        return False
    challenge = int.from_bytes(hashlib.sha512(r_encoded + public_key + message).digest(), "little") % _L
    return _point_equal(
        _point_mul(scalar, _BASE),
        _point_add(r_point, _point_mul(challenge, public_point)),
    )


def _lower_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def canonical_acquisition_signature_message(
    *, source_bundle_sha256: object, acquisition_contract_sha256: object, signer_key_id: object
) -> bytes:
    bundle = bytes.fromhex(_lower_sha256(source_bundle_sha256, "source_bundle_sha256"))
    contract = bytes.fromhex(_lower_sha256(acquisition_contract_sha256, "acquisition_contract_sha256"))
    if not isinstance(signer_key_id, str) or not signer_key_id or signer_key_id.strip() != signer_key_id:
        raise ValueError("signer_key_id must be a nonempty canonical string")
    identifier = signer_key_id.encode("utf-8")
    if len(identifier) > 65535 or b"\x00" in identifier:
        raise ValueError("signer_key_id UTF-8 encoding is inadmissible")
    return (
        b"CE-DIM-ACQUISITION-SIGNATURE-v1\x00"
        + len(identifier).to_bytes(2, "big") + identifier
        + contract + bundle
    )


@dataclass(frozen=True)
class VerifiedDimensionSignedAcquisitionReceipt:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    source_bytes_receipt: VerifiedDimensionSourceBytesReceipt | None
    source_bundle_sha256: str
    acquisition_contract_sha256: str
    signer_key_id: str
    trusted_public_key_sha256: str
    signed_message_sha256: str
    trusted_public_key_fingerprint_matches: bool
    strict_ed25519_signature_verified: bool
    cryptographic_bundle_signature_verified: bool
    externally_frozen_trust_anchor_verified: bool
    device_native_converter_receipt_verified: bool
    source_locked_empirical_dimension_result: bool
    consciousness_dimension_claim_admitted: bool


def verified_dimension_signed_acquisition_receipt(
    *,
    base_dimension_certificate: ConsciousMomentDimensionProtocolCertificate,
    development_archive_bytes: bytes,
    heldout_archive_bytes: bytes,
    preprocessing_archive_bytes: bytes,
    eigenbasis_archive_bytes: bytes,
    bootstrap_archive_bytes: bytes,
    execution_contract_archive_bytes: bytes,
    expected_source_bundle_sha256: object,
    acquisition_contract_sha256: object,
    signer_key_id: object,
    trusted_ed25519_public_key: object,
    expected_trusted_public_key_sha256: object,
    detached_ed25519_signature: object,
) -> VerifiedDimensionSignedAcquisitionReceipt:
    payloads = (
        development_archive_bytes, heldout_archive_bytes, preprocessing_archive_bytes,
        eigenbasis_archive_bytes, bootstrap_archive_bytes, execution_contract_archive_bytes,
    )
    bundle_hash = canonical_dimension_source_bundle_sha256(*payloads)
    expected_bundle = _lower_sha256(expected_source_bundle_sha256, "expected_source_bundle_sha256")
    contract_hash = _lower_sha256(acquisition_contract_sha256, "acquisition_contract_sha256")
    expected_key_hash = _lower_sha256(
        expected_trusted_public_key_sha256, "expected_trusted_public_key_sha256"
    )
    if not isinstance(trusted_ed25519_public_key, bytes):
        raise ValueError("trusted_ed25519_public_key must be bytes")
    if not isinstance(detached_ed25519_signature, bytes):
        raise ValueError("detached_ed25519_signature must be bytes")
    key_hash = hashlib.sha256(trusted_ed25519_public_key).hexdigest()
    key_match = key_hash == expected_key_hash
    message = canonical_acquisition_signature_message(
        source_bundle_sha256=bundle_hash,
        acquisition_contract_sha256=contract_hash,
        signer_key_id=signer_key_id,
    )
    signature_ok = ed25519_verify_strict(
        trusted_ed25519_public_key, message, detached_ed25519_signature
    )
    failures: list[str] = []
    if bundle_hash != expected_bundle:
        failures.append("DIMENSION_SIGNED_RECEIPT_BUNDLE_HASH_MISMATCH")
    if not key_match:
        failures.append("DIMENSION_SIGNED_RECEIPT_TRUSTED_KEY_FINGERPRINT_MISMATCH")
    if not signature_ok:
        failures.append("DIMENSION_SIGNED_RECEIPT_ED25519_SIGNATURE_INVALID")
    source_receipt = None
    if not failures:
        source_receipt = verified_dimension_source_bytes_receipt(
            base_dimension_certificate=base_dimension_certificate,
            development_archive_bytes=development_archive_bytes,
            heldout_archive_bytes=heldout_archive_bytes,
            preprocessing_archive_bytes=preprocessing_archive_bytes,
            eigenbasis_archive_bytes=eigenbasis_archive_bytes,
            bootstrap_archive_bytes=bootstrap_archive_bytes,
            execution_contract_archive_bytes=execution_contract_archive_bytes,
            expected_source_bundle_sha256=expected_bundle,
        )
        if source_receipt.validation_level is None:
            failures.extend(source_receipt.failure_codes)
    success = not failures and source_receipt is not None
    status = "VERIFIED_SIGNED_CANONICAL_DIMENSION_ACQUISITION_BUNDLE" if success else failures[0]
    return VerifiedDimensionSignedAcquisitionReceipt(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        source_bytes_receipt=source_receipt,
        source_bundle_sha256=bundle_hash,
        acquisition_contract_sha256=contract_hash,
        signer_key_id=signer_key_id,
        trusted_public_key_sha256=key_hash,
        signed_message_sha256=hashlib.sha256(message).hexdigest(),
        trusted_public_key_fingerprint_matches=key_match,
        strict_ed25519_signature_verified=signature_ok,
        cryptographic_bundle_signature_verified=success,
        externally_frozen_trust_anchor_verified=False,
        device_native_converter_receipt_verified=False,
        source_locked_empirical_dimension_result=False,
        consciousness_dimension_claim_admitted=False,
    )


__all__ = [
    "VerifiedDimensionSignedAcquisitionReceipt",
    "canonical_acquisition_signature_message",
    "ed25519_verify_strict",
    "verified_dimension_signed_acquisition_receipt",
]
