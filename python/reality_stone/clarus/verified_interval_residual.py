"""Exact componentwise residual/Krawczyk certificates for interval contours."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_interval_tightening import (
        VerifiedTightIntervalCircle,
        _entry_magnitude_enclosures,
        verified_tight_interval_family_circle,
    )
    from .verified_rational_contour import (
        ONE,
        ZERO,
        QComplex,
        VerifiedRationalStrip,
        _dyadic_sqrt,
        _fraction,
        _identity,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
        verified_rational_analytic_strip,
    )
else:
    from verified_interval_tightening import (  # type: ignore[no-redef]
        VerifiedTightIntervalCircle,
        _entry_magnitude_enclosures,
        verified_tight_interval_family_circle,
    )
    from verified_rational_contour import (  # type: ignore[no-redef]
        ONE,
        ZERO,
        QComplex,
        VerifiedRationalStrip,
        _dyadic_sqrt,
        _fraction,
        _identity,
        _inverse,
        _matmul,
        _matrix,
        parse_qcomplex,
        verified_rational_analytic_strip,
    )


QMatrix = tuple[tuple[QComplex, ...], ...]
RMatrix = tuple[tuple[Fraction, ...], ...]
BracketMatrix = tuple[tuple[tuple[Fraction, Fraction], ...], ...]
CheckMatrix = tuple[tuple[tuple[bool, bool], ...], ...]


@dataclass(frozen=True)
class ResidualNodeCertificate:
    status: str
    approximate_inverse: QMatrix
    inverse_magnitude_brackets: BracketMatrix
    inverse_magnitude_self_checks: CheckMatrix
    residual: QMatrix
    residual_magnitude_brackets: BracketMatrix
    residual_magnitude_self_checks: CheckMatrix
    interval_residual_upper: RMatrix
    contraction_one_upper: Fraction
    contraction_infinity_upper: Fraction
    inverse_witness_one_upper: Fraction
    inverse_witness_infinity_upper: Fraction
    inverse_one_upper: Fraction | None
    inverse_infinity_upper: Fraction | None
    inverse_two_sqrt_bracket: tuple[Fraction, Fraction] | None
    inverse_two_sqrt_self_checks: tuple[bool, bool] | None
    node_sigma_lower: Fraction | None


@dataclass(frozen=True)
class VerifiedResidualIntervalCircle:
    status: str
    validation_level: str | None
    rank_preserved_for_entire_family: bool
    tightening: VerifiedTightIntervalCircle
    nodes: tuple[ResidualNodeCertificate, ...]
    normalized_uncertainty_entry_brackets: BracketMatrix
    uncertainty_entry_self_checks: CheckMatrix
    sqrt2_bracket: tuple[Fraction, Fraction]
    sqrt2_self_checks: tuple[bool, bool]
    chord_factor_bracket: tuple[Fraction, Fraction]
    chord_factor_self_checks: tuple[bool, bool]
    normalized_chord_upper: Fraction
    raw_chord_upper: Fraction
    normalized_robust_delta_lower: Fraction | None
    raw_robust_delta_lower: Fraction | None
    normalized_robust_resolvent_upper: Fraction | None
    raw_robust_resolvent_upper: Fraction | None
    projector_perturbation_upper: Fraction | None
    spectral_reference_scale: Fraction
    sqrt_precision: int


@dataclass(frozen=True)
class VerifiedResidualIntervalProjector:
    status: str
    validation_level: str | None
    circle: VerifiedResidualIntervalCircle
    nominal_strip: VerifiedRationalStrip
    nominal_quadrature_error_upper: Fraction | None
    uncertainty_projector_error_upper: Fraction | None
    total_projector_error_upper: Fraction | None


@dataclass(frozen=True)
class ExactResidualWitnessConstruction:
    status: str
    validation_level: str | None
    normalized_node_matrices: tuple[QMatrix, ...]
    approximate_inverses: tuple[QMatrix, ...] | None
    inverse_identity_checks: tuple[tuple[bool, bool], ...]
    failing_node_index: int | None
    spectral_reference_scale: Fraction


@dataclass(frozen=True)
class ConstructedResidualIntervalCircle:
    status: str
    validation_level: str | None
    construction: ExactResidualWitnessConstruction
    circle: VerifiedResidualIntervalCircle | None


def _magnitude_enclosures(
    matrix: QMatrix, precision: int
) -> tuple[BracketMatrix, CheckMatrix, RMatrix]:
    brackets = []
    checks = []
    uppers = []
    for row in matrix:
        bracket_row = []
        check_row = []
        upper_row = []
        for entry in row:
            lower, upper, lower_ok, upper_ok = _dyadic_sqrt(
                entry.abs_squared(), precision
            )
            bracket_row.append((lower, upper))
            check_row.append((lower_ok, upper_ok))
            upper_row.append(upper)
        brackets.append(tuple(bracket_row))
        checks.append(tuple(check_row))
        uppers.append(tuple(upper_row))
    return tuple(brackets), tuple(checks), tuple(uppers)


def _nonnegative_matmul(left: RMatrix, right: RMatrix) -> RMatrix:
    n = len(left)
    return tuple(
        tuple(sum((left[i][k] * right[k][j] for k in range(n)), Fraction(0)) for j in range(n))
        for i in range(n)
    )


def _one_norm(matrix: RMatrix) -> Fraction:
    n = len(matrix)
    return max(sum((matrix[i][j] for i in range(n)), Fraction(0)) for j in range(n))


def _infinity_norm(matrix: RMatrix) -> Fraction:
    return max(sum(row, Fraction(0)) for row in matrix)


def _node_certificate(
    a0: QMatrix,
    witness: QMatrix,
    uncertainty_upper: RMatrix,
    precision: int,
) -> ResidualNodeCertificate:
    n = len(a0)
    ba0 = _matmul(witness, a0)
    identity = _identity(n)
    residual = tuple(
        tuple(identity[i][j] - ba0[i][j] for j in range(n)) for i in range(n)
    )
    b_brackets, b_checks, b_upper = _magnitude_enclosures(witness, precision)
    r_brackets, r_checks, r_upper = _magnitude_enclosures(residual, precision)
    product = _nonnegative_matmul(b_upper, uncertainty_upper)
    interval_residual = tuple(
        tuple(r_upper[i][j] + product[i][j] for j in range(n)) for i in range(n)
    )
    q_one = _one_norm(interval_residual)
    q_infinity = _infinity_norm(interval_residual)
    beta_one = _one_norm(b_upper)
    beta_infinity = _infinity_norm(b_upper)
    common = dict(
        approximate_inverse=witness,
        inverse_magnitude_brackets=b_brackets,
        inverse_magnitude_self_checks=b_checks,
        residual=residual,
        residual_magnitude_brackets=r_brackets,
        residual_magnitude_self_checks=r_checks,
        interval_residual_upper=interval_residual,
        contraction_one_upper=q_one,
        contraction_infinity_upper=q_infinity,
        inverse_witness_one_upper=beta_one,
        inverse_witness_infinity_upper=beta_infinity,
    )
    if q_one >= 1 or q_infinity >= 1:
        return ResidualNodeCertificate(
            status="VERIFIED_RESIDUAL_NODE_CONTRACTION_UNAVAILABLE",
            inverse_one_upper=None,
            inverse_infinity_upper=None,
            inverse_two_sqrt_bracket=None,
            inverse_two_sqrt_self_checks=None,
            node_sigma_lower=None,
            **common,
        )
    inverse_one = beta_one / (1 - q_one)
    inverse_infinity = beta_infinity / (1 - q_infinity)
    lower, upper, lower_ok, upper_ok = _dyadic_sqrt(
        inverse_one * inverse_infinity, precision
    )
    if upper == 0:
        raise AssertionError("a contracting approximate inverse cannot have zero norm")
    return ResidualNodeCertificate(
        status="VERIFIED_RESIDUAL_NODE_CONTRACTION",
        inverse_one_upper=inverse_one,
        inverse_infinity_upper=inverse_infinity,
        inverse_two_sqrt_bracket=(lower, upper),
        inverse_two_sqrt_self_checks=(lower_ok, upper_ok),
        node_sigma_lower=Fraction(1) / upper,
        **common,
    )


def exact_nominal_node_inverse_witnesses(
    nominal_transition: object,
    *,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    nodes: int = 4,
) -> ExactResidualWitnessConstruction:
    """Construct the four nominal resolvent inverses exactly over Q(i)."""
    if type(nodes) is not int or nodes != 4:
        raise ValueError("only the exact four-node mesh (nodes=4) is supported")
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    radius_q = _fraction(radius, "radius")
    if scale <= 0 or radius_q <= 0:
        raise ValueError("radius and spectral_reference_scale must be positive")
    u = tuple(tuple(entry / scale for entry in row) for row in matrix)
    c = parse_qcomplex(center, "center") / scale
    r = radius_q / scale
    directions = (
        ONE,
        QComplex(Fraction(0), Fraction(1)),
        QComplex(Fraction(-1)),
        QComplex(Fraction(0), Fraction(-1)),
    )
    node_matrices = tuple(
        tuple(
            tuple(((c + r * direction) if i == j else ZERO) - u[i][j] for j in range(n))
            for i in range(n)
        )
        for direction in directions
    )
    witnesses = []
    checks = []
    identity = _identity(n)
    for index, node in enumerate(node_matrices):
        inverse = _inverse(node)
        if inverse is None:
            return ExactResidualWitnessConstruction(
                status="EXACT_NOMINAL_NODE_INVERSE_UNAVAILABLE",
                validation_level=None,
                normalized_node_matrices=node_matrices,
                approximate_inverses=None,
                inverse_identity_checks=tuple(checks),
                failing_node_index=index,
                spectral_reference_scale=scale,
            )
        left_ok = _matmul(inverse, node) == identity
        right_ok = _matmul(node, inverse) == identity
        if not (left_ok and right_ok):
            raise AssertionError("exact inverse construction failed its two-sided identity check")
        witnesses.append(inverse)
        checks.append((left_ok, right_ok))
    return ExactResidualWitnessConstruction(
        status="VERIFIED_EXACT_NOMINAL_NODE_INVERSE_WITNESSES",
        validation_level="VERIFIED_EXACT_NOMINAL_NODE_INVERSE_WITNESSES",
        normalized_node_matrices=node_matrices,
        approximate_inverses=tuple(witnesses),
        inverse_identity_checks=tuple(checks),
        failing_node_index=None,
        spectral_reference_scale=scale,
    )


def verified_componentwise_residual_circle_with_exact_witnesses(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> ConstructedResidualIntervalCircle:
    """Construct exact nominal witnesses, then run the unchanged family gate."""
    construction = exact_nominal_node_inverse_witnesses(
        nominal_transition,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        nodes=nodes,
    )
    if construction.approximate_inverses is None:
        return ConstructedResidualIntervalCircle(
            status=construction.status,
            validation_level=None,
            construction=construction,
            circle=None,
        )
    circle = verified_componentwise_residual_circle(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        approximate_inverses=construction.approximate_inverses,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    if circle.validation_level is None:
        return ConstructedResidualIntervalCircle(
            status=circle.status,
            validation_level=None,
            construction=construction,
            circle=circle,
        )
    return ConstructedResidualIntervalCircle(
        status="VERIFIED_EXACT_CONSTRUCTED_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        validation_level="VERIFIED_EXACT_CONSTRUCTED_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        construction=construction,
        circle=circle,
    )


def verified_componentwise_residual_circle(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    approximate_inverses: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedResidualIntervalCircle:
    if type(nodes) is not int or nodes != 4:
        raise ValueError("only the exact four-node mesh (nodes=4) is supported")
    if type(sqrt_precision) is not int or sqrt_precision < 0:
        raise ValueError("square-root precision must be a nonnegative built-in integer")
    matrix = _matrix(nominal_transition, "nominal_transition")
    n = len(matrix)
    scale = _fraction(spectral_reference_scale, "spectral_reference_scale")
    radius_q = _fraction(radius, "radius")
    if scale <= 0 or radius_q <= 0:
        raise ValueError("radius and spectral_reference_scale must be positive")
    if not isinstance(approximate_inverses, (tuple, list)) or len(approximate_inverses) != 4:
        raise ValueError("approximate_inverses must contain exactly four matrices")
    witnesses = tuple(_matrix(value, f"approximate_inverses[{k}]") for k, value in enumerate(approximate_inverses))
    if any(len(witness) != n for witness in witnesses):
        raise ValueError("every approximate inverse must shape-match the nominal matrix")
    uncertainty_brackets, uncertainty_checks, uncertainty_upper = _entry_magnitude_enclosures(
        uncertainty_radii, n=n, scale=scale, precision=sqrt_precision
    )
    tightening = verified_tight_interval_family_circle(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        center=center,
        radius=radius,
        spectral_reference_scale=scale,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    u = tuple(tuple(entry / scale for entry in row) for row in matrix)
    c = parse_qcomplex(center, "center") / scale
    r = radius_q / scale
    directions = (
        ONE,
        QComplex(Fraction(0), Fraction(1)),
        QComplex(Fraction(-1)),
        QComplex(Fraction(0), Fraction(-1)),
    )
    node_results = []
    for direction, witness in zip(directions, witnesses):
        z = c + r * direction
        a0 = tuple(
            tuple((z if i == j else ZERO) - u[i][j] for j in range(n))
            for i in range(n)
        )
        node_results.append(
            _node_certificate(a0, witness, uncertainty_upper, sqrt_precision)
        )
    node_tuple = tuple(node_results)
    sqrt2_lower, sqrt2_upper, sqrt2_lower_ok, sqrt2_upper_ok = _dyadic_sqrt(
        Fraction(2), sqrt_precision
    )
    chord_lower, chord_upper, chord_lower_ok, chord_upper_ok = _dyadic_sqrt(
        Fraction(2) - sqrt2_lower, sqrt_precision
    )
    chord = r * chord_upper
    common = dict(
        tightening=tightening,
        nodes=node_tuple,
        normalized_uncertainty_entry_brackets=uncertainty_brackets,
        uncertainty_entry_self_checks=uncertainty_checks,
        sqrt2_bracket=(sqrt2_lower, sqrt2_upper),
        sqrt2_self_checks=(sqrt2_lower_ok, sqrt2_upper_ok),
        chord_factor_bracket=(chord_lower, chord_upper),
        chord_factor_self_checks=(chord_lower_ok, chord_upper_ok),
        normalized_chord_upper=chord,
        raw_chord_upper=chord * scale,
        spectral_reference_scale=scale,
        sqrt_precision=sqrt_precision,
    )
    if any(node.node_sigma_lower is None for node in node_tuple):
        return VerifiedResidualIntervalCircle(
            status="VERIFIED_RESIDUAL_NODE_CONTRACTION_UNAVAILABLE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=None,
            raw_robust_delta_lower=None,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    node_lowers = tuple(node.node_sigma_lower for node in node_tuple)
    assert all(value is not None for value in node_lowers)
    delta = min(value for value in node_lowers if value is not None) - chord
    if delta <= 0:
        return VerifiedResidualIntervalCircle(
            status="VERIFIED_RESIDUAL_FULL_CIRCLE_LOWER_NONPOSITIVE",
            validation_level=None,
            rank_preserved_for_entire_family=False,
            normalized_robust_delta_lower=delta,
            raw_robust_delta_lower=delta * scale,
            normalized_robust_resolvent_upper=None,
            raw_robust_resolvent_upper=None,
            projector_perturbation_upper=None,
            **common,
        )
    resolvent = Fraction(1) / delta
    projector_bound = (
        r
        * tightening.normalized_selected_uncertainty_upper
        * resolvent
        * resolvent
    )
    return VerifiedResidualIntervalCircle(
        status="VERIFIED_RATIONAL_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        validation_level="VERIFIED_RATIONAL_COMPONENTWISE_RESIDUAL_CONTOUR_BRIDGE",
        rank_preserved_for_entire_family=True,
        normalized_robust_delta_lower=delta,
        raw_robust_delta_lower=delta * scale,
        normalized_robust_resolvent_upper=resolvent,
        raw_robust_resolvent_upper=resolvent / scale,
        projector_perturbation_upper=projector_bound,
        **common,
    )


def verified_componentwise_residual_projector(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    approximate_inverses: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    expansion_factor: object,
    eigenvectors: object,
    eigenvalues: object,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedResidualIntervalProjector:
    circle = verified_componentwise_residual_circle(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        approximate_inverses=approximate_inverses,
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
        return VerifiedResidualIntervalProjector(
            circle.status, None, circle, strip, strip.error_upper, None, None
        )
    if strip.validation_level is None or strip.error_upper is None:
        return VerifiedResidualIntervalProjector(
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
    return VerifiedResidualIntervalProjector(
        "VERIFIED_RATIONAL_COMPONENTWISE_RESIDUAL_PROJECTOR_BRIDGE",
        "VERIFIED_RATIONAL_COMPONENTWISE_RESIDUAL_PROJECTOR_BRIDGE",
        circle,
        strip,
        strip.error_upper,
        circle.projector_perturbation_upper,
        total,
    )


__all__ = [
    "ConstructedResidualIntervalCircle",
    "ExactResidualWitnessConstruction",
    "ResidualNodeCertificate",
    "VerifiedResidualIntervalCircle",
    "VerifiedResidualIntervalProjector",
    "exact_nominal_node_inverse_witnesses",
    "verified_componentwise_residual_circle",
    "verified_componentwise_residual_circle_with_exact_witnesses",
    "verified_componentwise_residual_projector",
]
