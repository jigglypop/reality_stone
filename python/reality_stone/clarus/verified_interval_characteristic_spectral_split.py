"""Transfer an automatic exact nominal spectral split to an interval family."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

if __package__:
    from .verified_characteristic_spectral_split_discovery import (
        VerifiedCharacteristicSpectralSplitDiscovery,
        verified_characteristic_spectral_split_discovery,
    )
    from .verified_interval_contour import (
        VerifiedIntervalCircleBridge,
        verified_interval_family_circle,
    )
    from .verified_rational_contour import QComplex
else:
    from verified_characteristic_spectral_split_discovery import (  # type: ignore[no-redef]
        VerifiedCharacteristicSpectralSplitDiscovery,
        verified_characteristic_spectral_split_discovery,
    )
    from verified_interval_contour import (  # type: ignore[no-redef]
        VerifiedIntervalCircleBridge,
        verified_interval_family_circle,
    )
    from verified_rational_contour import QComplex  # type: ignore[no-redef]


QMatrix = tuple[tuple[QComplex, ...], ...]


@dataclass(frozen=True)
class VerifiedIntervalCharacteristicSpectralSplit:
    status: str
    validation_level: str | None
    failure_codes: tuple[str, ...]
    nominal_discovery: VerifiedCharacteristicSpectralSplitDiscovery
    interval_circle_bridge: VerifiedIntervalCircleBridge
    nominal_exact_projector: QMatrix | None
    nominal_projector_rank: int | None
    family_projector_rank: int | None
    all_family_members_have_same_projector_rank: bool
    normalized_uncertainty_upper: Fraction
    normalized_nominal_contour_delta_lower: Fraction | None
    normalized_robust_contour_delta_lower: Fraction | None
    exact_projector_perturbation_upper: Fraction | None
    interval_characteristic_polynomial_factorization_required: bool
    interval_root_tracking_required: bool
    homotopy_contour_crossing_excluded: bool
    empirical_uncertainty_provenance_verified: bool


def verified_interval_characteristic_spectral_split(
    nominal_transition: object,
    *,
    uncertainty_radii: object,
    center: object,
    radius: object,
    spectral_reference_scale: object,
    maximum_partitions: int = 1024,
    maximum_factor_candidate_value_tuples: int = 1_000_000,
    nodes: int = 4,
    sqrt_precision: int = 32,
) -> VerifiedIntervalCharacteristicSpectralSplit:
    """Discover nominal rank exactly and certify it on an entire matrix box."""
    nominal = verified_characteristic_spectral_split_discovery(
        nominal_transition,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        maximum_partitions=maximum_partitions,
        maximum_factor_candidate_value_tuples=maximum_factor_candidate_value_tuples,
        sqrt_precision=sqrt_precision,
    )
    interval = verified_interval_family_circle(
        nominal_transition,
        uncertainty_radii=uncertainty_radii,
        center=center,
        radius=radius,
        spectral_reference_scale=spectral_reference_scale,
        nodes=nodes,
        sqrt_precision=sqrt_precision,
    )
    failures: list[str] = []
    if nominal.validation_level is None or nominal.selected_construction is None:
        failures.append("INTERVAL_SPLIT_NOMINAL_CHARACTERISTIC_DISCOVERY_UNAVAILABLE")
    if interval.validation_level is None:
        failures.append("INTERVAL_SPLIT_UNIFORM_CONTOUR_BRIDGE_UNAVAILABLE")
    selected = nominal.selected_construction
    projector = None if selected is None else selected.constructed_projector
    rank = nominal.projector_rank
    if not failures and (projector is None or rank is None):
        failures.append("INTERVAL_SPLIT_NOMINAL_PROJECTOR_OR_RANK_UNAVAILABLE")
    success = not failures
    status = (
        "VERIFIED_INTERVAL_CHARACTERISTIC_SPECTRAL_SPLIT_AND_RANK"
        if success
        else failures[0]
    )
    return VerifiedIntervalCharacteristicSpectralSplit(
        status=status,
        validation_level=status if success else None,
        failure_codes=tuple(failures),
        nominal_discovery=nominal,
        interval_circle_bridge=interval,
        nominal_exact_projector=projector,
        nominal_projector_rank=rank,
        family_projector_rank=rank if success else None,
        all_family_members_have_same_projector_rank=success,
        normalized_uncertainty_upper=interval.normalized_uncertainty_upper,
        normalized_nominal_contour_delta_lower=interval.nominal.normalized_delta_lower,
        normalized_robust_contour_delta_lower=interval.normalized_robust_delta_lower,
        exact_projector_perturbation_upper=(
            interval.projector_perturbation_upper if success else None
        ),
        interval_characteristic_polynomial_factorization_required=False,
        interval_root_tracking_required=False,
        homotopy_contour_crossing_excluded=success,
        empirical_uncertainty_provenance_verified=False,
    )


__all__ = [
    "VerifiedIntervalCharacteristicSpectralSplit",
    "verified_interval_characteristic_spectral_split",
]
