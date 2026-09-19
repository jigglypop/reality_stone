"""Exact finite dissipative/skew split for nonsymmetric mixed-unit mobility."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

if __package__:
    from .physical_tensor_mobility import (
        PhysicalTensorMobilityCertificate,
        PrincipalMinor,
        _determinant,
        _exact_fraction,
        _matrix,
        _vector,
        physical_tensor_mobility_certificate,
    )
else:
    from physical_tensor_mobility import (  # type: ignore[no-redef]
        PhysicalTensorMobilityCertificate,
        PrincipalMinor,
        _determinant,
        _exact_fraction,
        _matrix,
        _vector,
        physical_tensor_mobility_certificate,
    )


@dataclass(frozen=True)
class PhysicalDissipativeSkewMobilityCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    dimension: int
    state_reference_scales: tuple[Fraction, ...]
    energy_reference_scale: Fraction
    time_reference_scale: Fraction
    physical_mobility: tuple[tuple[Fraction, ...], ...]
    normalized_mobility: tuple[tuple[Fraction, ...], ...]
    normalized_symmetric_dissipative_part: tuple[tuple[Fraction, ...], ...]
    normalized_skew_drift_part: tuple[tuple[Fraction, ...], ...]
    skew_adjoint_exact: bool
    symmetric_part_certificate: PhysicalTensorMobilityCertificate
    coercivity_lower_bound: Fraction
    shifted_coercivity_principal_minors: tuple[PrincipalMinor, ...]
    coercivity_certified: bool
    dimensionless_gradient: tuple[Fraction, ...]
    dimensionless_external_forcing: tuple[Fraction, ...]
    normalized_symmetric_velocity: tuple[Fraction, ...] | None
    normalized_skew_velocity: tuple[Fraction, ...] | None
    normalized_total_velocity: tuple[Fraction, ...] | None
    physical_coordinate_velocities: tuple[Fraction, ...] | None
    normalized_symmetric_dissipation: Fraction | None
    normalized_skew_power: Fraction | None
    normalized_external_input_power: Fraction | None
    normalized_net_dissipation_margin: Fraction | None
    nonincreasing_model_potential_certified: bool
    polyak_lojasiewicz_constant: Fraction
    polyak_lojasiewicz_hypothesis_supplied: bool
    unforced_exponential_potential_rate_normalized: Fraction | None
    unforced_exponential_potential_rate_physical: Fraction | None
    infinite_dimensional_form_hypotheses_verified: bool


def _matvec(matrix, vector):
    return tuple(
        sum((matrix[i][j] * vector[j] for j in range(len(vector))), Fraction(0))
        for i in range(len(vector))
    )


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def _principal_minors(matrix):
    result = []
    for size in range(1, len(matrix) + 1):
        for subset in combinations(range(len(matrix)), size):
            principal = tuple(tuple(matrix[i][j] for j in subset) for i in subset)
            result.append(PrincipalMinor(tuple(index + 1 for index in subset), _determinant(principal)))
    return tuple(result)


def physical_dissipative_skew_mobility_certificate(
    *,
    state_reference_scales: object,
    energy_reference_scale: object,
    time_reference_scale: object,
    physical_mobility: object,
    dimensionless_gradient: object,
    dimensionless_external_forcing: object,
    coercivity_lower_bound: object,
    polyak_lojasiewicz_constant: object,
) -> PhysicalDissipativeSkewMobilityCertificate:
    """Split normalized M=S+K and certify dissipation by S exactly."""
    scales = _vector(state_reference_scales, "state_reference_scales")
    n = len(scales)
    energy = _exact_fraction(energy_reference_scale, "energy_reference_scale")
    time = _exact_fraction(time_reference_scale, "time_reference_scale")
    if any(value <= 0 for value in scales) or energy <= 0 or time <= 0:
        raise ValueError("state, energy, and time scales must be positive")
    mobility = _matrix(physical_mobility, "physical_mobility", n)
    gradient = _vector(dimensionless_gradient, "dimensionless_gradient", length=n)
    forcing = _vector(dimensionless_external_forcing, "dimensionless_external_forcing", length=n)
    coercivity = _exact_fraction(coercivity_lower_bound, "coercivity_lower_bound")
    pl = _exact_fraction(polyak_lojasiewicz_constant, "polyak_lojasiewicz_constant")
    if coercivity < 0 or pl < 0:
        raise ValueError("coercivity and Polyak-Lojasiewicz constants must be nonnegative")

    normalized = tuple(
        tuple(energy * time * mobility[i][j] / (scales[i] * scales[j]) for j in range(n))
        for i in range(n)
    )
    symmetric = tuple(
        tuple((normalized[i][j] + normalized[j][i]) / 2 for j in range(n))
        for i in range(n)
    )
    skew = tuple(
        tuple((normalized[i][j] - normalized[j][i]) / 2 for j in range(n))
        for i in range(n)
    )
    physical_symmetric = tuple(
        tuple((mobility[i][j] + mobility[j][i]) / 2 for j in range(n))
        for i in range(n)
    )
    symmetric_certificate = physical_tensor_mobility_certificate(
        state_reference_scales=scales,
        energy_reference_scale=energy,
        time_reference_scale=time,
        physical_mobility=physical_symmetric,
        dimensionless_gradient=gradient,
    )
    shifted = tuple(
        tuple(symmetric[i][j] - (coercivity if i == j else 0) for j in range(n))
        for i in range(n)
    )
    shifted_minors = _principal_minors(shifted)
    coercivity_ok = all(minor.determinant >= 0 for minor in shifted_minors)
    failures = list(symmetric_certificate.failure_codes)
    if not coercivity_ok:
        failures.append("DISSIPATIVE_SKEW_COERCIVITY_LOWER_BOUND_INVALID")

    common = dict(
        claim_scope=(
            "FINITE_EXACT_MIXED_UNIT_DISSIPATIVE_SKEW_MOBILITY; "
            "INFINITE_DIMENSIONAL_FORM_DOMAIN_AND_PL_HYPOTHESES_REMAIN_EXTERNAL"
        ),
        failure_codes=tuple(failures), dimension=n,
        state_reference_scales=scales, energy_reference_scale=energy,
        time_reference_scale=time, physical_mobility=mobility,
        normalized_mobility=normalized,
        normalized_symmetric_dissipative_part=symmetric,
        normalized_skew_drift_part=skew,
        skew_adjoint_exact=all(skew[i][j] == -skew[j][i] for i in range(n) for j in range(n)),
        symmetric_part_certificate=symmetric_certificate,
        coercivity_lower_bound=coercivity,
        shifted_coercivity_principal_minors=shifted_minors,
        coercivity_certified=coercivity_ok,
        dimensionless_gradient=gradient,
        dimensionless_external_forcing=forcing,
        polyak_lojasiewicz_constant=pl,
        polyak_lojasiewicz_hypothesis_supplied=pl > 0,
        infinite_dimensional_form_hypotheses_verified=False,
    )
    if failures:
        return PhysicalDissipativeSkewMobilityCertificate(
            status=failures[0], validation_level=None, robust_interior=False,
            normalized_symmetric_velocity=None, normalized_skew_velocity=None,
            normalized_total_velocity=None, physical_coordinate_velocities=None,
            normalized_symmetric_dissipation=None, normalized_skew_power=None,
            normalized_external_input_power=None,
            normalized_net_dissipation_margin=None,
            nonincreasing_model_potential_certified=False,
            unforced_exponential_potential_rate_normalized=None,
            unforced_exponential_potential_rate_physical=None,
            **common,
        )

    sg = _matvec(symmetric, gradient)
    kg = _matvec(skew, gradient)
    symmetric_velocity = tuple(-value for value in sg)
    skew_velocity = tuple(-value for value in kg)
    total_velocity = tuple(-sg[i] - kg[i] + forcing[i] for i in range(n))
    physical_velocities = tuple(scales[i] * total_velocity[i] / time for i in range(n))
    dissipation = _dot(gradient, sg)
    skew_power = _dot(gradient, kg)
    if skew_power != 0:
        raise AssertionError("exact skew matrix produced nonzero quadratic power")
    input_power = _dot(gradient, forcing)
    margin = dissipation - input_power
    unforced = all(value == 0 for value in forcing)
    rate = 2 * coercivity * pl if unforced and coercivity > 0 and pl > 0 else None
    status = "VERIFIED_FINITE_DISSIPATIVE_SKEW_MOBILITY_MAP"
    return PhysicalDissipativeSkewMobilityCertificate(
        status=status, validation_level=status,
        robust_interior=(coercivity_ok and all(minor.determinant > 0 for minor in shifted_minors)),
        normalized_symmetric_velocity=symmetric_velocity,
        normalized_skew_velocity=skew_velocity,
        normalized_total_velocity=total_velocity,
        physical_coordinate_velocities=physical_velocities,
        normalized_symmetric_dissipation=dissipation,
        normalized_skew_power=skew_power,
        normalized_external_input_power=input_power,
        normalized_net_dissipation_margin=margin,
        nonincreasing_model_potential_certified=margin >= 0,
        unforced_exponential_potential_rate_normalized=rate,
        unforced_exponential_potential_rate_physical=(None if rate is None else rate / time),
        **common,
    )


__all__ = [
    "PhysicalDissipativeSkewMobilityCertificate",
    "physical_dissipative_skew_mobility_certificate",
]
