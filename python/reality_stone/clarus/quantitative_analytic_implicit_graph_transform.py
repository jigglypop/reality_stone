"""Quantitative analytic majorant for an implicit coupled graph transform."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial

if __package__:
    from .quantitative_graph_transform import _exact_fraction
else:
    from quantitative_graph_transform import _exact_fraction  # type: ignore[no-redef]


@dataclass(frozen=True)
class QuantitativeAnalyticImplicitGraphTransformCertificate:
    status: str
    validation_level: str | None
    claim_scope: str
    failure_codes: tuple[str, ...]
    robust_interior: bool
    base_dimension: int
    maximum_reported_derivative_order: int
    base_linear_conorm_lower: Fraction
    complex_input_radius: Fraction
    base_nonlinear_derivative_amplitude: Fraction
    base_nonlinear_derivative_rate: Fraction
    base_majorant_argument: Fraction
    base_majorant_convergence_margin: Fraction
    nonlinear_displacement_upper: Fraction | None
    nonlinear_derivative_upper: Fraction | None
    inverse_contraction_factor_upper: Fraction | None
    inverse_contraction_margin: Fraction | None
    available_complex_output_radius_lower: Fraction | None
    requested_complex_output_radius: Fraction
    output_radius_margin: Fraction | None
    fiber_value_at_center_upper: Fraction
    fiber_derivative_amplitude: Fraction
    fiber_derivative_rate: Fraction
    fiber_majorant_argument: Fraction
    fiber_majorant_convergence_margin: Fraction
    transformed_graph_sup_upper: Fraction | None
    analytic_graph_sup_upper: Fraction
    analytic_graph_class_margin: Fraction | None
    complex_c0_contraction_factor_upper: Fraction
    complex_c0_contraction_margin: Fraction
    frechet_derivative_bounds_upper: tuple[Fraction, ...] | None
    analytic_factorial_rate_upper: Fraction | None
    analytic_fixed_point_certified: bool
    c_infinity_certified: bool
    gevrey_s_for_every_real_s_at_least_one_certified: bool
    analytic_graph_real_dimension: int | None


@dataclass(frozen=True)
class QuantitativeIntegerGevreyEnvelope:
    gevrey_order: int
    amplitude_upper: Fraction
    rate_upper: Fraction
    derivative_bounds_upper: tuple[Fraction, ...]


@dataclass(frozen=True)
class AnalyticFixedPointIterationBound:
    steps: int
    sup_distance_upper: Fraction


def quantitative_analytic_implicit_graph_transform(
    *,
    base_dimension: int,
    maximum_reported_derivative_order: int,
    base_linear_conorm_lower: object,
    complex_input_radius: object,
    base_nonlinear_derivative_amplitude: object,
    base_nonlinear_derivative_rate: object,
    requested_complex_output_radius: object,
    fiber_value_at_center_upper: object,
    fiber_derivative_amplitude: object,
    fiber_derivative_rate: object,
    analytic_graph_sup_upper: object,
    complex_c0_contraction_factor_upper: object,
) -> QuantitativeAnalyticImplicitGraphTransformCertificate:
    """Certify a uniform holomorphic inverse/composition class on complex balls.

    The hypotheses mean ``||D^j N(0)|| <= A_F B_F^j j!`` for ``j>=2`` and
    ``||D^j Y(0)|| <= A_Y B_Y^j j!`` for ``j>=1``, uniformly over the graph
    class on the declared complex input ball.
    """
    if type(base_dimension) is not int or base_dimension < 1:
        raise ValueError("base_dimension must be a positive built-in integer")
    if type(maximum_reported_derivative_order) is not int or maximum_reported_derivative_order < 1:
        raise ValueError("maximum_reported_derivative_order must be positive")
    alpha = _exact_fraction(base_linear_conorm_lower, "base_linear_conorm_lower")
    radius = _exact_fraction(complex_input_radius, "complex_input_radius")
    af = _exact_fraction(base_nonlinear_derivative_amplitude, "base_nonlinear_derivative_amplitude")
    bf = _exact_fraction(base_nonlinear_derivative_rate, "base_nonlinear_derivative_rate")
    requested = _exact_fraction(requested_complex_output_radius, "requested_complex_output_radius")
    y0 = _exact_fraction(fiber_value_at_center_upper, "fiber_value_at_center_upper")
    ay = _exact_fraction(fiber_derivative_amplitude, "fiber_derivative_amplitude")
    by = _exact_fraction(fiber_derivative_rate, "fiber_derivative_rate")
    graph_sup = _exact_fraction(analytic_graph_sup_upper, "analytic_graph_sup_upper")
    q = _exact_fraction(complex_c0_contraction_factor_upper, "complex_c0_contraction_factor_upper")
    if alpha <= 0 or radius <= 0 or requested <= 0:
        raise ValueError("conorm and complex radii must be positive")
    if min(af, bf, y0, ay, by, graph_sup, q) < 0:
        raise ValueError("majorant amplitudes, rates, sup bounds, and contraction must be nonnegative")

    xf = bf * radius
    xy = by * radius
    xf_margin = 1 - xf
    xy_margin = 1 - xy
    q_margin = 1 - q
    failures: list[str] = []
    if xf_margin <= 0:
        failures.append("ANALYTIC_BASE_MAJORANT_RADIUS_NOT_STRICT")
    if xy_margin <= 0:
        failures.append("ANALYTIC_FIBER_MAJORANT_RADIUS_NOT_STRICT")
    if q_margin <= 0:
        failures.append("ANALYTIC_COMPLEX_C0_CONTRACTION_NOT_STRICT")

    displacement: Fraction | None = None
    derivative: Fraction | None = None
    inverse_factor: Fraction | None = None
    inverse_margin: Fraction | None = None
    available: Fraction | None = None
    radius_margin: Fraction | None = None
    transformed_sup: Fraction | None = None
    class_margin: Fraction | None = None
    derivative_bounds: tuple[Fraction, ...] | None = None
    factorial_rate: Fraction | None = None
    if xf_margin > 0:
        displacement = af * xf**2 / xf_margin
        derivative = af * bf * (Fraction(1) / xf_margin**2 - 1)
        inverse_factor = derivative / alpha
        inverse_margin = 1 - inverse_factor
        available = alpha * radius - displacement
        radius_margin = available - requested
        if inverse_margin <= 0:
            failures.append("ANALYTIC_NONLINEAR_INVERSE_CONTRACTION_NOT_STRICT")
        if available <= 0:
            failures.append("ANALYTIC_COMPLEX_INVERSE_OUTPUT_RADIUS_NOT_POSITIVE")
        elif radius_margin <= 0:
            failures.append("ANALYTIC_REQUESTED_OUTPUT_RADIUS_NOT_STRICTLY_COVERED")
    if xy_margin > 0:
        transformed_sup = y0 + ay * xy / xy_margin
        class_margin = graph_sup - transformed_sup
        if class_margin < 0:
            failures.append("ANALYTIC_GRAPH_SUP_CLASS_NOT_INVARIANT")
    if not failures:
        assert transformed_sup is not None
        factorial_rate = Fraction(3) / requested
        derivative_bounds = tuple(
            transformed_sup * factorial_rate**order * factorial(order)
            for order in range(1, maximum_reported_derivative_order + 1)
        )

    scope = (
        "CONDITIONAL_COMPLEX_BALL_ANALYTIC_FIXED_POINT_THEOREM; REQUIRES_"
        "UNIFORM_HOLOMORPHIC_MAJORANTS_AND_COMPLEX_C0_CONTRACTION"
    )
    common = dict(
        claim_scope=scope,
        base_dimension=base_dimension,
        maximum_reported_derivative_order=maximum_reported_derivative_order,
        base_linear_conorm_lower=alpha,
        complex_input_radius=radius,
        base_nonlinear_derivative_amplitude=af,
        base_nonlinear_derivative_rate=bf,
        base_majorant_argument=xf,
        base_majorant_convergence_margin=xf_margin,
        nonlinear_displacement_upper=displacement,
        nonlinear_derivative_upper=derivative,
        inverse_contraction_factor_upper=inverse_factor,
        inverse_contraction_margin=inverse_margin,
        available_complex_output_radius_lower=available,
        requested_complex_output_radius=requested,
        output_radius_margin=radius_margin,
        fiber_value_at_center_upper=y0,
        fiber_derivative_amplitude=ay,
        fiber_derivative_rate=by,
        fiber_majorant_argument=xy,
        fiber_majorant_convergence_margin=xy_margin,
        transformed_graph_sup_upper=transformed_sup,
        analytic_graph_sup_upper=graph_sup,
        analytic_graph_class_margin=class_margin,
        complex_c0_contraction_factor_upper=q,
        complex_c0_contraction_margin=q_margin,
        frechet_derivative_bounds_upper=derivative_bounds,
        analytic_factorial_rate_upper=factorial_rate,
    )
    if failures:
        return QuantitativeAnalyticImplicitGraphTransformCertificate(
            status=failures[0], validation_level=None,
            failure_codes=tuple(failures), robust_interior=False,
            analytic_fixed_point_certified=False, c_infinity_certified=False,
            gevrey_s_for_every_real_s_at_least_one_certified=False,
            analytic_graph_real_dimension=None, **common,
        )
    status = "VERIFIED_CONDITIONAL_ANALYTIC_IMPLICIT_GRAPH_TRANSFORM_FIXED_POINT"
    robust = all(
        margin is not None and margin > 0
        for margin in (xf_margin, xy_margin, q_margin, inverse_margin, radius_margin, class_margin)
    )
    return QuantitativeAnalyticImplicitGraphTransformCertificate(
        status=status, validation_level=status, failure_codes=(),
        robust_interior=robust, analytic_fixed_point_certified=True,
        c_infinity_certified=True,
        gevrey_s_for_every_real_s_at_least_one_certified=True,
        analytic_graph_real_dimension=base_dimension, **common,
    )


def integer_gevrey_envelope_from_analytic_certificate(
    certificate: QuantitativeAnalyticImplicitGraphTransformCertificate,
    *, gevrey_order: int,
) -> QuantitativeIntegerGevreyEnvelope:
    """Weaken the analytic n! envelope to an integer Gevrey-(s>=1) envelope."""
    if certificate.validation_level is None or certificate.analytic_factorial_rate_upper is None:
        raise ValueError("Gevrey weakening requires a verified analytic certificate")
    if type(gevrey_order) is not int or gevrey_order < 1:
        raise ValueError("gevrey_order must be a positive built-in integer")
    assert certificate.transformed_graph_sup_upper is not None
    amplitude = certificate.transformed_graph_sup_upper
    rate = certificate.analytic_factorial_rate_upper
    bounds = tuple(
        amplitude * rate**order * factorial(order) ** gevrey_order
        for order in range(1, certificate.maximum_reported_derivative_order + 1)
    )
    return QuantitativeIntegerGevreyEnvelope(
        gevrey_order=gevrey_order,
        amplitude_upper=amplitude,
        rate_upper=rate,
        derivative_bounds_upper=bounds,
    )


def analytic_fixed_point_iteration_bound(
    certificate: QuantitativeAnalyticImplicitGraphTransformCertificate,
    *, initial_sup_distance: object, steps: int,
) -> AnalyticFixedPointIterationBound:
    if certificate.validation_level is None:
        raise ValueError("iteration requires a verified analytic certificate")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative built-in integer")
    initial = _exact_fraction(initial_sup_distance, "initial_sup_distance")
    if initial < 0:
        raise ValueError("initial_sup_distance must be nonnegative")
    return AnalyticFixedPointIterationBound(
        steps=steps,
        sup_distance_upper=certificate.complex_c0_contraction_factor_upper**steps * initial,
    )


__all__ = [
    "AnalyticFixedPointIterationBound",
    "QuantitativeAnalyticImplicitGraphTransformCertificate",
    "QuantitativeIntegerGevreyEnvelope",
    "analytic_fixed_point_iteration_bound",
    "integer_gevrey_envelope_from_analytic_certificate",
    "quantitative_analytic_implicit_graph_transform",
]
