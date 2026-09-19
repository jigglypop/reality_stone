"""Exact bit receipt for IEEE-754 binary64 approximate inverse witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re

if __package__:
    from .verified_rational_contour import QComplex
    from .verified_interval_residual import (
        VerifiedResidualIntervalCircle,
        verified_componentwise_residual_circle,
    )
else:
    from verified_rational_contour import QComplex  # type: ignore[no-redef]
    from verified_interval_residual import (  # type: ignore[no-redef]
        VerifiedResidualIntervalCircle,
        verified_componentwise_residual_circle,
    )


_BINARY64_HEX = re.compile(r"[0-9a-f]{16}")


@dataclass(frozen=True)
class Binary64ExactScalar:
    bits_hex: str
    classification: str
    exact_value: Fraction
    negative_zero: bool


@dataclass(frozen=True)
class Binary64ResidualWitnessReceipt:
    status: str
    validation_level: str | None
    claim_scope: str
    witness_bits_sha256: str
    witness_bits: tuple[tuple[tuple[tuple[str, str], ...], ...], ...]
    decoded_approximate_inverses: tuple[tuple[tuple[QComplex, ...], ...], ...]
    negative_zero_count: int
    subnormal_count: int
    residual_circle: VerifiedResidualIntervalCircle
    rank_preserved_for_entire_family: bool
    solver_algorithm_verified: bool
    binary64_stored_value_decoding_verified: bool
    solver_operation_rounding_mode_verified: bool
    empirical_matrix_provenance_verified: bool


def decode_binary64_hex(bits_hex: object) -> Binary64ExactScalar:
    """Decode one canonical finite binary64 bit pattern without float arithmetic."""
    if not isinstance(bits_hex, str) or _BINARY64_HEX.fullmatch(bits_hex) is None:
        raise ValueError("binary64 bits must be exactly sixteen lowercase hexadecimal digits")
    bits = int(bits_hex, 16)
    sign = -1 if bits >> 63 else 1
    exponent_bits = (bits >> 52) & 0x7FF
    fraction_bits = bits & ((1 << 52) - 1)
    if exponent_bits == 0x7FF:
        raise ValueError("NaN and infinity binary64 witnesses are forbidden")
    if exponent_bits == 0:
        if fraction_bits == 0:
            return Binary64ExactScalar(
                bits_hex=bits_hex, classification="zero", exact_value=Fraction(0),
                negative_zero=sign < 0,
            )
        mantissa = fraction_bits
        exponent = -1074
        classification = "subnormal"
    else:
        mantissa = (1 << 52) + fraction_bits
        exponent = exponent_bits - 1023 - 52
        classification = "normal"
    magnitude = (
        Fraction(mantissa << exponent)
        if exponent >= 0
        else Fraction(mantissa, 1 << (-exponent))
    )
    return Binary64ExactScalar(
        bits_hex=bits_hex, classification=classification,
        exact_value=sign * magnitude, negative_zero=False,
    )


def _decode_witness_bits(value: object):
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("binary64 witness bits must contain exactly four node matrices")
    canonical_nodes = []
    decoded_nodes = []
    negative_zeros = 0
    subnormals = 0
    size = None
    for node_index, matrix in enumerate(value):
        if not isinstance(matrix, (tuple, list)) or not matrix:
            raise ValueError(f"node {node_index} must be a nonempty square matrix")
        if size is None:
            size = len(matrix)
        if len(matrix) != size:
            raise ValueError("all binary64 witness matrices must have one common size")
        canonical_rows = []
        decoded_rows = []
        for row_index, row in enumerate(matrix):
            if not isinstance(row, (tuple, list)) or len(row) != size:
                raise ValueError(f"node {node_index} row {row_index} has invalid shape")
            canonical_row = []
            decoded_row = []
            for column_index, entry in enumerate(row):
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise ValueError(
                        f"node {node_index} entry {row_index},{column_index} must contain real/imag bits"
                    )
                real = decode_binary64_hex(entry[0])
                imag = decode_binary64_hex(entry[1])
                negative_zeros += int(real.negative_zero) + int(imag.negative_zero)
                subnormals += int(real.classification == "subnormal") + int(imag.classification == "subnormal")
                canonical_row.append((real.bits_hex, imag.bits_hex))
                decoded_row.append(QComplex(real.exact_value, imag.exact_value))
            canonical_rows.append(tuple(canonical_row))
            decoded_rows.append(tuple(decoded_row))
        canonical_nodes.append(tuple(canonical_rows))
        decoded_nodes.append(tuple(decoded_rows))
    return tuple(canonical_nodes), tuple(decoded_nodes), negative_zeros, subnormals


def binary64_residual_witness_circle_receipt(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    approximate_inverse_binary64_bits: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> Binary64ResidualWitnessReceipt:
    """Decode stored solver bits exactly, then run the unchanged residual gate."""
    canonical, decoded, negative_zeros, subnormals = _decode_witness_bits(
        approximate_inverse_binary64_bits
    )
    canonical_bytes = json.dumps(
        canonical, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    circle = verified_componentwise_residual_circle(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        approximate_inverses=decoded,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    scope = (
        "EXACT_DECODING_OF_STORED_FINITE_BINARY64_WITNESSES_AND_EXACT_RATIONAL_RESIDUAL; "
        "SOLVER_ALGORITHM_AND_EMPIRICAL_MATRIX_PROVENANCE_ARE_NOT_VERIFIED"
    )
    if circle.validation_level is None:
        return Binary64ResidualWitnessReceipt(
            status=circle.status, validation_level=None, claim_scope=scope,
            witness_bits_sha256=digest, witness_bits=canonical,
            decoded_approximate_inverses=decoded,
            negative_zero_count=negative_zeros, subnormal_count=subnormals,
            residual_circle=circle, rank_preserved_for_entire_family=False,
            solver_algorithm_verified=False,
            binary64_stored_value_decoding_verified=True,
            solver_operation_rounding_mode_verified=False,
            empirical_matrix_provenance_verified=False,
        )
    status = "VERIFIED_BINARY64_WITNESS_COMPONENTWISE_RESIDUAL_CONTOUR"
    return Binary64ResidualWitnessReceipt(
        status=status, validation_level=status, claim_scope=scope,
        witness_bits_sha256=digest, witness_bits=canonical,
        decoded_approximate_inverses=decoded,
        negative_zero_count=negative_zeros, subnormal_count=subnormals,
        residual_circle=circle, rank_preserved_for_entire_family=True,
        solver_algorithm_verified=False,
        binary64_stored_value_decoding_verified=True,
        solver_operation_rounding_mode_verified=False,
        empirical_matrix_provenance_verified=False,
    )


__all__ = [
    "Binary64ExactScalar",
    "Binary64ResidualWitnessReceipt",
    "binary64_residual_witness_circle_receipt",
    "decode_binary64_hex",
]
