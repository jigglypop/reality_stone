"""Exact best-of-Frobenius-and-induced tightening for interval families."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_interval_contour import (
        VerifiedIntervalCircleBridge,
        verified_interval_family_circle,
    )
    from .verified_rational_contour import (
        VerifiedRationalStrip,
        _dyadic_sqrt,
        _fraction,
        _matrix,
        verified_rational_analytic_strip,
    )
else:
    from verified_interval_contour import (  # type: ignore[no-redef]
        VerifiedIntervalCircleBridge,
        verified_interval_family_circle,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        VerifiedRationalStrip,
        _dyadic_sqrt,
        _fraction,
        _matrix,
        verified_rational_analytic_strip,
    )


@dataclass(frozen=True)
class VerifiedTightIntervalCircle:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    frobenius_bridge: VerifiedIntervalCircleBridge
    normalized_entry_magnitude_brackets: tuple[tuple[tuple[Fraction, Fraction], ...], ...]
    entry_magnitude_self_checks: tuple[tuple[tuple[bool, bool], ...], ...]
    normalized_induced_one_upper: Fraction
    normalized_induced_infinity_upper: Fraction
    normalized_induced_product_sqrt_bracket: tuple[Fraction, Fraction]
    induced_product_sqrt_self_checks: tuple[bool, bool]
    normalized_induced_uncertainty_upper: Fraction
    raw_induced_uncertainty_upper: Fraction
    selected_uncertainty_method: str
    normalized_selected_uncertainty_upper: Fraction
    raw_selected_uncertainty_upper: Fraction
    normalized_robust_delta_lower: Fraction | None
    raw_robust_delta_lower: Fraction | None
    normalized_robust_resolvent_upper: Fraction | None
    raw_robust_resolvent_upper: Fraction | None
    projector_perturbation_upper: Fraction | None
    spectral_reference_scale: Fraction
    sqrt_precision: int


@dataclass(frozen=True)
class VerifiedTightIntervalProjector:
    status: str
    validation_level: str | None
    circle: VerifiedTightIntervalCircle
    nominal_strip: VerifiedRationalStrip
    nominal_quadrature_error_upper: Fraction | None
    uncertainty_projector_error_upper: Fraction | None
    total_projector_error_upper: Fraction | None


def _entry_magnitude_enclosures(
    uncertainty_radii: object, *, n: int, scale: Fraction, precision: int
) -> tuple[
    tuple[tuple[tuple[Fraction, Fraction], ...], ...],
    tuple[tuple[tuple[bool, bool], ...], ...],
    tuple[tuple[Fraction, ...], ...],
]:
    if not isinstance(uncertainty_radii, (tuple, list)) or len(uncertainty_radii) != n:
        raise ValueError("uncertainty_radii must shape-match the nominal square matrix")
    brackets = []
    checks = []
    uppers = []
    for i, row in enumerate(uncertainty_radii):
        if not isinstance(row, (tuple, list)) or len(row) != n:
            raise ValueError("uncertainty_radii must shape-match the nominal square matrix")
        bracket_row = []
        check_row = []
        upper_row = []
        for j, entry in enumerate(row):
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError(
                    f"uncertainty_radii[{i}][{j}] must be a (real_radius, imag_radius) pair"
                )
            a = _fraction(entry[0], f"uncertainty_radii[{i}][{j}].real")
            b = _fraction(entry[1], f"uncertainty_radii[{i}][{j}].imag")
            if a < 0 or b < 0:
                raise ValueError("uncertainty radii must be nonnegative")
            a_normalized, b_normalized = a / scale, b / scale
            lower, upper, lower_ok, upper_ok = _dyadic_sqrt(
                a_normalized * a_normalized + b_normalized * b_normalized,
                precision,
            )
            bracket_row.append((lower, upper))
            check_row.append((lower_ok, upper_ok))
            upper_row.append(upper)
        brackets.append(tuple(bracket_row))
        checks.append(tuple(check_row))
        uppers.append(tuple(upper_row))
    return tuple(brackets), tuple(checks), tuple(uppers)


def verified_tight_interval_family_circle(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedTightIntervalCircle:
    matrix = _matrix(nominal_transition, "nominal_transition")
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    if scale <= 0:
        raise ValueError("spectral_reference_scale must be positive")
    if type(sqrt_precision) is not int or sqrt_precision < 0:
        raise ValueError("square-root precision must be a nonnegative built-in integer")
    brackets, checks, entry_uppers = _entry_magnitude_enclosures(
        uncertainty_radii, n=len(matrix), scale=scale, precision=sqrt_precision
    )
    base = verified_interval_family_circle(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        center=center,
        radius=radius,
        spectral_reference_scale=scale,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    n = len(entry_uppers)
    one_upper = max(sum(entry_uppers[i][j] for i in range(n)) for j in range(n))
    infinity_upper = max(sum(row, Fraction(0)) for row in entry_uppers)
    product_lower, product_upper, product_lower_ok, product_upper_ok = _dyadic_sqrt(
        one_upper * infinity_upper, sqrt_precision
    )
    frobenius_upper = base.normalized_uncertainty_upper
    if frobenius_upper <= product_upper:
        method = "FROBENIUS"
        selected = frobenius_upper
    else:
        method = "INDUCED_1_INFINITY"
        selected = product_upper
    common = dict(
        frobenius_bridge=base,
        normalized_entry_magnitude_brackets=brackets,
        entry_magnitude_self_checks=checks,
        normalized_induced_one_upper=one_upper,
        normalized_induced_infinity_upper=infinity_upper,
        normalized_induced_product_sqrt_bracket=(product_lower, product_upper),
        induced_product_sqrt_self_checks=(product_lower_ok, product_upper_ok),
        normalized_induced_uncertainty_upper=product_upper,
        raw_induced_uncertainty_upper=product_upper * scale,
        selected_uncertainty_method=method,
        normalized_selected_uncertainty_upper=selected,
        raw_selected_uncertainty_upper=selected * scale,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    nominal = base.nominal
    if nominal.validation_level is None or nominal.normalized_delta_lower is None:
        return VerifiedTightIntervalCircle(
            status="VERIFIED_NOMINAL_CIRCLE_CERTIFICATE_UNAVAILABLE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=None,
            raw_robust_delta_lower=None,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    nominal_delta = nominal.normalized_delta_lower
    robust_delta = nominal_delta - selected
    if robust_delta <= 0:
        return VerifiedTightIntervalCircle(
            status="VERIFIED_TIGHT_INTERVAL_UNCERTAINTY_NOT_BELOW_MARGIN",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=robust_delta,
            raw_robust_delta_lower=robust_delta * scale,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    normalized_radius = _fraction(radius, "radius") / scale
    normalized_resolvent = Fraction(1) / robust_delta
    projector_bound = normalized_radius * selected / (nominal_delta * robust_delta)
    return VerifiedTightIntervalCircle(
        status="VERIFIED_RATIONAL_TIGHT_INTERVAL_FAMILY_CONTOUR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_TIGHT_INTERVAL_FAMILY_CONTOUR_BRIDGE",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=robust_delta,
        raw_robust_delta_lower=robust_delta * scale,
        normalized_robust_resolvent_upper=normalized_resolvent,
        raw_robust_resolvent_upper=normalized_resolvent / scale,
        projector_perturbation_upper=projector_bound,
        **common,
    )


def verified_tight_interval_family_projector(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    expansion_factor: object,
    eigenvectors: object,
    eigenvalues: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedTightIntervalProjector:
    circle = verified_tight_interval_family_circle(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    strip = verified_rational_analytic_strip(
        nominal_transition,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        expansion_factor=expansion_factor,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    if circle.validation_level is None:
        return VerifiedTightIntervalProjector(
            circle.status, None, circle, strip, strip.error_upper, None, None
        )
    if strip.validation_level is None or strip.error_upper is None:
        return VerifiedTightIntervalProjector(
            "VERIFIED_NOMINAL_STRIP_CERTIFICATE_UNAVAILABLE",
            None,
            circle,
            strip,
            None,
            circle.projector_perturbation_upper,
            None,
        )
    assert circle.projector_perturbation_upper is not None
    total = circle.projector_perturbation_upper + strip.error_upper
    return VerifiedTightIntervalProjector(
        "VERIFIED_RATIONAL_TIGHT_INTERVAL_FAMILY_PROJECTOR_BRIDGE",
        "VERIFIED_RATIONAL_TIGHT_INTERVAL_FAMILY_PROJECTOR_BRIDGE",
        circle,
        strip,
        strip.error_upper,
        circle.projector_perturbation_upper,
        total,
    )


__all__ = [
    "VerifiedTightIntervalCircle",
    "VerifiedTightIntervalProjector",
    "verified_tight_interval_family_circle",
    "verified_tight_interval_family_projector",
]
