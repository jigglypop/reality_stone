"""Exact mixed-unit scale map and PSD gate for finite tensor mobility."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import re


def _exact_fraction(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{name} must be an exact integer, Fraction, or canonical rational string")
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, str):
        decimal = re.fullmatch(r"-?(?:0|[1-9][0-9]*)\.[0-9]*[1-9]", value)
        rational = re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/(?:[1-9][0-9]*))?", value)
        if decimal is None and rational is None:
            raise ValueError(f"{name} is not a canonical rational string")
        try:
            parsed = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"{name} is not a canonical rational string") from error
        if parsed == 0 and value.startswith("-"):
            raise ValueError(f"{name} is not a canonical rational string")
        if rational is not None and str(parsed) != value:
            raise ValueError(f"{name} is not a canonical rational string")
        return parsed
    raise ValueError(f"{name} must be an exact integer, Fraction, or canonical rational string")


def _vector(value: object, name: str, *, length: int | None = None) -> tuple[Fraction, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a nonempty exact vector")
    parsed = tuple(_exact_fraction(entry, f"{name}[{i}]") for i, entry in enumerate(value))
    if length is not None and len(parsed) != length:
        raise ValueError(f"{name} must have length {length}")
    return parsed


def _matrix(value: object, name: str, n: int) -> tuple[tuple[Fraction, ...], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != n:
        raise ValueError(f"{name} must be a square {n} by {n} exact matrix")
    rows = []
    for i, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != n:
            raise ValueError(f"{name} must be a square {n} by {n} exact matrix")
        rows.append(tuple(_exact_fraction(entry, f"{name}[{i}][{j}]") for j, entry in enumerate(row)))
    return tuple(rows)


def _determinant(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    n = len(matrix)
    work = [list(row) for row in matrix]
    sign = 1
    result = Fraction(1)
    for col in range(n):
        pivot = next((row for row in range(col, n) if work[row][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
            sign = -sign
        pivot_value = work[col][col]
        result *= pivot_value
        for row in range(col + 1, n):
            factor = work[row][col] / pivot_value
            for j in range(col + 1, n):
                work[row][j] -= factor * work[col][j]
    return result * sign


@dataclass(frozen=True)
class PrincipalMinor:
    indices: tuple[int, ...]
    determinant: Fraction


@dataclass(frozen=True)
class PhysicalTensorMobilityCertificate:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    dimension: int
    state_reference_scales: tuple[Fraction, ...]
    energy_reference_scale: Fraction
    time_reference_scale: Fraction
    physical_mobility: tuple[tuple[Fraction, ...], ...]
    normalized_mobility: tuple[tuple[Fraction, ...], ...]
    symmetric: bool
    principal_minors: tuple[PrincipalMinor, ...]
    first_negative_principal_minor: PrincipalMinor | None
    positive_semidefinite: bool
    positive_definite: bool
    dimensionless_gradient: tuple[Fraction, ...]
    normalized_velocity: tuple[Fraction, ...] | None
    physical_coordinate_velocities: tuple[Fraction, ...] | None
    normalized_dissipation: Fraction | None
    physical_model_potential_dissipation: Fraction | None


def physical_tensor_mobility_certificate(
    *,
    state_reference_scales: object,
    energy_reference_scale: object,
    time_reference_scale: object,
    physical_mobility: object,
    dimensionless_gradient: object,
) -> PhysicalTensorMobilityCertificate:
    """Normalize a declared mixed-unit tensor and certify symmetric PSD exactly."""
    state_scales = _vector(state_reference_scales, "state_reference_scales")
    n = len(state_scales)
    energy = _exact_fraction(energy_reference_scale, "energy_reference_scale")
    time = _exact_fraction(time_reference_scale, "time_reference_scale")
    if any(scale <= 0 for scale in state_scales) or energy <= 0 or time <= 0:
        raise ValueError("all state, energy, and time reference scales must be positive")
    mobility = _matrix(physical_mobility, "physical_mobility", n)
    gradient = _vector(dimensionless_gradient, "dimensionless_gradient", length=n)
    normalized = tuple(
        tuple(energy * time * mobility[i][j] / (state_scales[i] * state_scales[j]) for j in range(n))
        for i in range(n)
    )
    symmetric = all(normalized[i][j] == normalized[j][i] for i in range(n) for j in range(n))
    common = dict(
        dimension=n,
        state_reference_scales=state_scales,
        energy_reference_scale=energy,
        time_reference_scale=time,
        physical_mobility=mobility,
        normalized_mobility=normalized,
        symmetric=symmetric,
        dimensionless_gradient=gradient,
    )
    if not symmetric:
        return PhysicalTensorMobilityCertificate(
            status="TENSOR_MOBILITY_NOT_SYMMETRIC",
            validation_level=None,
            failure_codes=("TENSOR_MOBILITY_NOT_SYMMETRIC",),
            principal_minors=(),
            first_negative_principal_minor=None,
            positive_semidefinite=False,
            positive_definite=False,
            normalized_velocity=None,
            physical_coordinate_velocities=None,
            normalized_dissipation=None,
            physical_model_potential_dissipation=None,
            **common,
        )
    minors = []
    for size in range(1, n + 1):
        for subset in combinations(range(n), size):
            principal = tuple(tuple(normalized[i][j] for j in subset) for i in subset)
            minors.append(PrincipalMinor(tuple(index + 1 for index in subset), _determinant(principal)))
    minor_tuple = tuple(minors)
    negative = next((minor for minor in minor_tuple if minor.determinant < 0), None)
    if negative is not None:
        return PhysicalTensorMobilityCertificate(
            status="TENSOR_MOBILITY_NOT_POSITIVE_SEMIDEFINITE",
            validation_level=None,
            failure_codes=("TENSOR_MOBILITY_NOT_POSITIVE_SEMIDEFINITE",),
            principal_minors=minor_tuple,
            first_negative_principal_minor=negative,
            positive_semidefinite=False,
            positive_definite=False,
            normalized_velocity=None,
            physical_coordinate_velocities=None,
            normalized_dissipation=None,
            physical_model_potential_dissipation=None,
            **common,
        )
    matrix_gradient = tuple(
        sum((normalized[i][j] * gradient[j] for j in range(n)), Fraction(0))
        for i in range(n)
    )
    normalized_velocity = tuple(-entry for entry in matrix_gradient)
    physical_velocities = tuple(
        state_scales[i] * normalized_velocity[i] / time for i in range(n)
    )
    dissipation = sum((gradient[i] * matrix_gradient[i] for i in range(n)), Fraction(0))
    if dissipation < 0:
        raise AssertionError("exact PSD principal-minor gate produced negative quadratic form")
    positive_definite = all(minor.determinant > 0 for minor in minor_tuple)
    return PhysicalTensorMobilityCertificate(
        status="VERIFIED_MIXED_UNIT_TENSOR_MOBILITY_MAP",
        validation_level="VERIFIED_MIXED_UNIT_TENSOR_MOBILITY_MAP",
        failure_codes=(),
        principal_minors=minor_tuple,
        first_negative_principal_minor=None,
        positive_semidefinite=True,
        positive_definite=positive_definite,
        normalized_velocity=normalized_velocity,
        physical_coordinate_velocities=physical_velocities,
        normalized_dissipation=dissipation,
        physical_model_potential_dissipation=energy * dissipation / time,
        **common,
    )


__all__ = [
    "PhysicalTensorMobilityCertificate",
    "PrincipalMinor",
    "physical_tensor_mobility_certificate",
]

