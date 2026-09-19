"""Finite L0 host for the primordial hybrid map, plus a boxed P-H1 construction.

This module has no repository-local imports.  It copies the source hybrid map
``F_0`` locally and may apply the named growth modulation

    r(q) = r_0 (1 + kappa (2q - 1)),  r_0 = 9/2,

only for ``kappa`` in ``I_r ∪ {0}`` with ``I_r = (0, 86/315)``.  Default
``kappa = 0`` recovers ``F_0`` pointwise.

An optional ``drive`` in ``[0, 1]`` multiplies only the growth term.
Default ``drive = 1`` recovers the current growth bracket.  ``drive = 0``
removes that term.  Two copies may be stepped with independent routed
drives ``u = W E``.  This is a finite two-channel construction.  It does
not claim autonomy, C. elegans identity, or AGI.

A wash resets both copies to the same registered start between epochs.
After epoch alpha the left occupancy may be stored as a named bit that
gates the right drive in epoch beta.  No-store ignores that bit.
This is a finite wash construction.  It does not claim autonomy or a
brain.

A registered activity pair at fixed sigma may be stepped once.  The
readout is ``(m', b')``.  This is a finite pair construction.  It does
not claim autonomy or a brain.

A three-epoch wash may write sensor occupancy after alpha and a named
bit ``I`` from action occupancy after beta.  Epoch gamma may gate the
action drive by ``I``, freeze sigma, or overwrite sigma from action
occupancy on the same named slot.  ``I`` is a named bit, not a cube.
Finite construction only.  Does not claim autonomy or a brain.

A registered host tuple ``H = (t, E, Z^S, Z^A, σ, I)`` may be stepped
once.  The registered internal kernel is that one-step map, typed
``H → H``.  Finite pair construction.  Does not claim autonomy or a
brain.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence


NOMINAL_GROWTH = Fraction(9, 2)
NOMINAL_LEAK = Fraction(5, 2)
NOMINAL_BOUNDARY_PRODUCTION = Fraction(1, 5)
NOMINAL_BOUNDARY_DECAY = Fraction(1, 10)
NOMINAL_COPY_SELECTION = Fraction(1, 2)
NOMINAL_MUTATION = Fraction(3, 32)
NOMINAL_INHERITANCE_GAIN = Fraction(1)
NOMINAL_DIVISION_THRESHOLD = Fraction(3, 4)
NOMINAL_CAPACITY = Fraction(1)
DEFAULT_LEAKAGE = Fraction(0)
DEFAULT_TARGET_FLUX = Fraction(1)
DEFAULT_RESIDUAL = Fraction(0)
KAPPA_OPEN_RIGHT = Fraction(86, 315)
SOURCE_EXTINCTION_AREA = Fraction(1, 10)
EXTINCTION_AREA_FLOOR = Fraction(1, 20)
DIVIDING_DISCRIMINANT_AT_HALF = Fraction(-95, 64)
DIVIDING_DISCRIMINANT_AT_ONE = Fraction(-359, 16)


def _as_fraction(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValueError(f"{name} must be a Fraction or int")
    return Fraction(value)


def _unit_interval(value: object, name: str) -> Fraction:
    result = _as_fraction(value, name)
    if not Fraction(0) <= result <= Fraction(1):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative built-in integer")
    return value


def _clip_unit(value: Fraction) -> Fraction:
    if value < 0:
        return Fraction(0)
    if value > 1:
        return Fraction(1)
    return value


def admissible_kappa(value: object) -> Fraction:
    """Accept ``kappa = 0`` or ``kappa`` in the open interval ``I_r``.

    Values ``{1/2, 1}`` are killing tests for the parent reading of P-H1 and
    are not members of ``I_r``.  This function refuses them; it does not
    construct ``F_kappa`` there.
    """

    kappa = _as_fraction(value, "kappa")
    if kappa == 0:
        return kappa
    if Fraction(0) < kappa < KAPPA_OPEN_RIGHT:
        return kappa
    raise ValueError(
        f"kappa={kappa} is outside I_r ∪ {{0}} = (0, 86/315) ∪ {{0}}; "
        "kappa in {1/2, 1} are killing tests, not members of I_r"
    )


def growth_at_label(label: Fraction, kappa: Fraction) -> Fraction:
    """Boxed P-H1 construction: ``r(q) = r_0 (1 + kappa (2q - 1))``."""

    return NOMINAL_GROWTH * (1 + kappa * (2 * label - 1))


def dividing_mass_discriminant(growth: Fraction) -> Fraction:
    """Cited math-lane polynomial ``Δ_r = 9r^2 - 32r + 4``.  Not a new theorem."""

    return 9 * growth**2 - 32 * growth + 4


@dataclass(frozen=True)
class NominalParameters:
    """Nominal rationals of the source hybrid map."""

    growth: Fraction = NOMINAL_GROWTH
    leak: Fraction = NOMINAL_LEAK
    boundary_production: Fraction = NOMINAL_BOUNDARY_PRODUCTION
    boundary_decay: Fraction = NOMINAL_BOUNDARY_DECAY
    copy_selection: Fraction = NOMINAL_COPY_SELECTION
    mutation: Fraction = NOMINAL_MUTATION
    inheritance_gain: Fraction = NOMINAL_INHERITANCE_GAIN
    division_threshold: Fraction = NOMINAL_DIVISION_THRESHOLD
    capacity: Fraction = NOMINAL_CAPACITY


NOMINAL_PARAMETERS = NominalParameters()


@dataclass(frozen=True)
class HybridState:
    """Source cube state ``Z = (m, b, q)``."""

    mass: Fraction
    boundary: Fraction
    label: Fraction

    @classmethod
    def from_values(
        cls,
        mass: object,
        boundary: object,
        label: object,
    ) -> "HybridState":
        return cls(
            mass=_unit_interval(mass, "mass"),
            boundary=_unit_interval(boundary, "boundary"),
            label=_unit_interval(label, "label"),
        )


SOURCE_FIXED_POINTS: tuple[HybridState, ...] = tuple(
    HybridState.from_values(mass, boundary, label)
    for mass, boundary in ((0, 0), (Fraction(1, 2), Fraction(1, 2)))
    for label in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
)
CUBE_CORNERS: tuple[HybridState, ...] = tuple(
    HybridState.from_values(mass, boundary, label)
    for mass in (0, 1)
    for boundary in (0, 1)
    for label in (0, 1)
)


def source_hybrid_step(
    state: HybridState,
    *,
    parameters: NominalParameters = NOMINAL_PARAMETERS,
    kappa: Fraction = Fraction(0),
    drive: object = Fraction(1),
) -> HybridState:
    """Local copy of ``F_0``, or boxed ``F_kappa`` when ``kappa`` is in ``I_r``.

    The algebra is (P.1)--(P.4) with ``r`` replaced by ``r(q)`` when
    ``kappa > 0``.  ``q`` does not enter (P.1)--(P.3) at ``kappa = 0``.
    Optional ``drive`` multiplies only the growth term.  Default 1
    recovers the current bracket.  ``q``-maps stay uncoupled from drive.
    """

    if not isinstance(state, HybridState):
        raise TypeError("state must be a HybridState")
    kappa = admissible_kappa(kappa)
    drive = _unit_interval(drive, "drive")
    growth = growth_at_label(state.label, kappa)
    raw_predivision = state.mass * (
        1
        + drive * growth * (1 - state.mass / parameters.capacity)
        - parameters.leak * (1 - state.boundary)
    )
    predivision = max(Fraction(0), raw_predivision)
    next_mass = predivision / (2 if predivision >= parameters.division_threshold else 1)
    next_boundary = (1 - parameters.boundary_decay) * state.boundary + (
        parameters.boundary_production * state.mass * (1 - state.boundary)
    )
    copied_label = (
        state.label
        + parameters.copy_selection
        * state.label
        * (1 - state.label)
        * (2 * state.label - 1)
        + parameters.mutation * (1 - 2 * state.label)
    )
    next_label = Fraction(1, 2) + parameters.inheritance_gain * (
        copied_label - Fraction(1, 2)
    )
    return HybridState(next_mass, next_boundary, next_label)


def iterate_source(
    state: HybridState,
    ticks: int,
    *,
    parameters: NominalParameters = NOMINAL_PARAMETERS,
    kappa: Fraction = Fraction(0),
    drive: object = Fraction(1),
) -> HybridState:
    ticks = _nonnegative_int(ticks, "ticks")
    current = state
    for _ in range(ticks):
        current = source_hybrid_step(
            current,
            parameters=parameters,
            kappa=kappa,
            drive=drive,
        )
    return current


@dataclass(frozen=True)
class SourceHybridSubsystem:
    """One registered subsystem.  ``step(E)`` accepts flux and does not use it.

    Optional ``drive`` multiplies only the growth term.  Default 1
    recovers the current source step.
    """

    state: HybridState
    parameters: NominalParameters = NOMINAL_PARAMETERS
    kappa: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if not isinstance(self.state, HybridState):
            raise TypeError("state must be a HybridState")
        object.__setattr__(self, "kappa", admissible_kappa(self.kappa))

    def step(
        self,
        flux: object,
        *,
        drive: object = Fraction(1),
    ) -> "SourceHybridSubsystem":
        _unit_interval(flux, "flux")
        next_state = source_hybrid_step(
            self.state,
            parameters=self.parameters,
            kappa=self.kappa,
            drive=drive,
        )
        return SourceHybridSubsystem(
            state=next_state,
            parameters=self.parameters,
            kappa=self.kappa,
        )


@dataclass(frozen=True)
class UniverseState:
    """Registered L0 state ``U = (t, E, subsystems, phi)``."""

    tick: int
    flux: Fraction
    subsystems: tuple[SourceHybridSubsystem, ...]
    residual: Fraction = DEFAULT_RESIDUAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "tick", _nonnegative_int(self.tick, "tick"))
        object.__setattr__(self, "flux", _unit_interval(self.flux, "flux"))
        object.__setattr__(
            self, "residual", _unit_interval(self.residual, "residual")
        )
        if not self.subsystems:
            raise ValueError("subsystems must be a nonempty tuple")
        if any(
            not isinstance(subsystem, SourceHybridSubsystem)
            for subsystem in self.subsystems
        ):
            raise TypeError("every subsystem must be a SourceHybridSubsystem")
        object.__setattr__(self, "subsystems", tuple(self.subsystems))


@dataclass(frozen=True)
class UniverseKernel:
    """Finite chemostat host.  The same ``E`` is applied to every subsystem."""

    leakage: Fraction = DEFAULT_LEAKAGE
    target_flux: Fraction = DEFAULT_TARGET_FLUX
    parameters: NominalParameters = NOMINAL_PARAMETERS
    kappa: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "leakage", _unit_interval(self.leakage, "leakage"))
        object.__setattr__(
            self, "target_flux", _unit_interval(self.target_flux, "target_flux")
        )
        object.__setattr__(self, "kappa", admissible_kappa(self.kappa))

    def host(
        self,
        states: Sequence[HybridState] | HybridState,
        *,
        flux: object = DEFAULT_TARGET_FLUX,
        residual: object = DEFAULT_RESIDUAL,
        tick: int = 0,
    ) -> UniverseState:
        if isinstance(states, HybridState):
            hosted = (states,)
        else:
            hosted = tuple(states)
        if not hosted:
            raise ValueError("states must be nonempty")
        subsystems = tuple(
            SourceHybridSubsystem(
                state=state,
                parameters=self.parameters,
                kappa=self.kappa,
            )
            for state in hosted
        )
        return UniverseState(
            tick=tick,
            flux=_unit_interval(flux, "flux"),
            subsystems=subsystems,
            residual=_unit_interval(residual, "residual"),
        )

    def next_flux(self, flux: Fraction) -> Fraction:
        return _clip_unit(
            (1 - self.leakage) * flux + self.leakage * self.target_flux
        )

    def step(self, universe: UniverseState) -> UniverseState:
        if not isinstance(universe, UniverseState):
            raise TypeError("universe must be a UniverseState")
        flux = universe.flux
        next_subsystems = tuple(subsystem.step(flux) for subsystem in universe.subsystems)
        return UniverseState(
            tick=universe.tick + 1,
            flux=self.next_flux(flux),
            subsystems=next_subsystems,
            residual=universe.residual,
        )

    def iterate(self, universe: UniverseState, ticks: int) -> UniverseState:
        ticks = _nonnegative_int(ticks, "ticks")
        current = universe
        for _ in range(ticks):
            current = self.step(current)
        return current


def hosted_states(universe: UniverseState) -> tuple[HybridState, ...]:
    return tuple(subsystem.state for subsystem in universe.subsystems)


def source_one_step_extinction_area() -> Fraction:
    """Source-document area of the nominal one-step extinction wedge.

    The integral is the cited ``1/10`` formula
    ``ceiling/3 - 5 ceiling^2 / 18`` with ``ceiling = 3/5``.  This is not a
    new area theorem.
    """

    ceiling = Fraction(3, 5)
    return ceiling / 3 - 5 * ceiling**2 / 18


def registered_grid() -> tuple[HybridState, ...]:
    seen: dict[tuple[Fraction, Fraction, Fraction], HybridState] = {}
    for state in (*CUBE_CORNERS, *SOURCE_FIXED_POINTS):
        seen[(state.mass, state.boundary, state.label)] = state
    return tuple(seen.values())


def killing_low_label_growth(kappa: Fraction) -> Fraction:
    """``r(1/4)`` for a numeric kappa.  Used only to document killing tests."""

    return growth_at_label(Fraction(1, 4), _as_fraction(kappa, "kappa"))


IDENTITY_WEIGHT: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = (
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
)
COMPLETE_BINARY_ROUTER: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = (
    (Fraction(1, 2), Fraction(1, 2)),
    (Fraction(1, 2), Fraction(1, 2)),
)
REGISTERED_FLUX_E1: tuple[Fraction, Fraction] = (Fraction(1), Fraction(0))
REGISTERED_FLUX_E2: tuple[Fraction, Fraction] = (Fraction(0), Fraction(1))
R0_MASS: tuple[Fraction, Fraction] = (Fraction(2, 5), Fraction(3, 5))
R0_BOUNDARY: tuple[Fraction, Fraction] = (Fraction(4, 9), Fraction(6, 11))
ROUTING_HORIZON = 32


def _row_stochastic_2x2(
    weight: object,
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    if not isinstance(weight, Sequence) or len(weight) != 2:
        raise ValueError("weight must be a 2x2 row-stochastic matrix")
    rows: list[tuple[Fraction, Fraction]] = []
    for index, row in enumerate(weight):
        if not isinstance(row, Sequence) or len(row) != 2:
            raise ValueError("weight must be a 2x2 row-stochastic matrix")
        left = _as_fraction(row[0], f"weight[{index}][0]")
        right = _as_fraction(row[1], f"weight[{index}][1]")
        if left < 0 or right < 0:
            raise ValueError("weight entries must be nonnegative")
        if left + right != 1:
            raise ValueError("each weight row must sum to 1")
        rows.append((left, right))
    return (rows[0], rows[1])


def _flux_pair(flux: object) -> tuple[Fraction, Fraction]:
    if not isinstance(flux, Sequence) or len(flux) != 2:
        raise ValueError("flux must be a pair in [0, 1]^2")
    return (
        _unit_interval(flux[0], "flux[0]"),
        _unit_interval(flux[1], "flux[1]"),
    )


def routed_drives(
    weight: object,
    flux: object,
) -> tuple[Fraction, Fraction]:
    """Routed drives ``u = W E`` for a row-stochastic nonnegative ``W``."""

    rows = _row_stochastic_2x2(weight)
    first, second = _flux_pair(flux)
    return (
        rows[0][0] * first + rows[0][1] * second,
        rows[1][0] * first + rows[1][1] * second,
    )


def occupancy_bit(state: HybridState) -> int:
    """Occupancy ``1[(m, b) in R0]``.  Label is not an observable."""

    if not isinstance(state, HybridState):
        raise TypeError("state must be a HybridState")
    in_mass = R0_MASS[0] <= state.mass <= R0_MASS[1]
    in_boundary = R0_BOUNDARY[0] <= state.boundary <= R0_BOUNDARY[1]
    return int(in_mass and in_boundary)


@dataclass(frozen=True)
class RoutedTwoCopy:
    """Two copies of the boxed host with independent routed drives.

    ``q``-maps stay uncoupled.  Finite two-channel construction only.
    """

    left: HybridState
    right: HybridState
    parameters: NominalParameters = NOMINAL_PARAMETERS
    kappa: Fraction = Fraction(1, 4)

    def __post_init__(self) -> None:
        if not isinstance(self.left, HybridState):
            raise TypeError("left must be a HybridState")
        if not isinstance(self.right, HybridState):
            raise TypeError("right must be a HybridState")
        object.__setattr__(self, "kappa", admissible_kappa(self.kappa))

    def step(self, drive_left: object, drive_right: object) -> "RoutedTwoCopy":
        return RoutedTwoCopy(
            left=source_hybrid_step(
                self.left,
                parameters=self.parameters,
                kappa=self.kappa,
                drive=drive_left,
            ),
            right=source_hybrid_step(
                self.right,
                parameters=self.parameters,
                kappa=self.kappa,
                drive=drive_right,
            ),
            parameters=self.parameters,
            kappa=self.kappa,
        )

    def iterate(
        self,
        ticks: int,
        drive_left: object,
        drive_right: object,
    ) -> "RoutedTwoCopy":
        ticks = _nonnegative_int(ticks, "ticks")
        current = self
        for _ in range(ticks):
            current = current.step(drive_left, drive_right)
        return current

    def iterate_routed(
        self,
        ticks: int,
        weight: object,
        flux: object,
    ) -> "RoutedTwoCopy":
        drive_left, drive_right = routed_drives(weight, flux)
        return self.iterate(ticks, drive_left, drive_right)

    def occupancy_pair(self) -> tuple[int, int]:
        return occupancy_bit(self.left), occupancy_bit(self.right)


REGISTERED_START_MASS = Fraction(1, 2)
REGISTERED_START_BOUNDARY = Fraction(49, 99)
REGISTERED_START_LABEL = Fraction(3, 4)
WASH_TASK_TAU1: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = (
    REGISTERED_FLUX_E1,
    REGISTERED_FLUX_E2,
)
WASH_TASK_TAU2: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = (
    REGISTERED_FLUX_E2,
    REGISTERED_FLUX_E2,
)
LOOP_TASK_PHI1: tuple[
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
] = (
    REGISTERED_FLUX_E1,
    REGISTERED_FLUX_E2,
    REGISTERED_FLUX_E2,
)
LOOP_TASK_PHI2: tuple[
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
] = (
    REGISTERED_FLUX_E1,
    REGISTERED_FLUX_E1,
    REGISTERED_FLUX_E2,
)


def registered_start() -> HybridState:
    """Registered wash start: center of ``U0`` at ``q = 3/4``."""

    return HybridState.from_values(
        REGISTERED_START_MASS,
        REGISTERED_START_BOUNDARY,
        REGISTERED_START_LABEL,
    )


def _bit(value: object, name: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be a bit in {{0, 1}}")
    return value


def role_split_drives(
    sigma: object,
    flux: object,
    *,
    weight: object = IDENTITY_WEIGHT,
) -> tuple[Fraction, Fraction]:
    """Sensor drive is ``u_I(e)``. Action drive is ``σ * u_I(e)``.

    Body indexing is ``(S, A) = (L, R)``.  ``σ`` is a named bit, not a
    cube coordinate.
    """

    bit = _bit(sigma, "sigma")
    drive_sensor, drive_action = routed_drives(weight, flux)
    return drive_sensor, Fraction(bit) * drive_action


def loop_gate_drives(
    named_i: object,
    flux: object,
    *,
    weight: object = IDENTITY_WEIGHT,
) -> tuple[Fraction, Fraction]:
    """γ-gate (L7.3): action drive is ``I * u_I(e)``.

    Same product as ``role_split_drives`` with ``I`` in the σ slot.
    Construction equality on the two-cube body.  Not a third cube.
    """

    return role_split_drives(named_i, flux, weight=weight)


@dataclass(frozen=True)
class WashedRoleSplit:
    """Wash niche on the two-channel host. Sensor is left, action is right.

    Between epochs both copies reset to ``start``. After epoch α the
    sensor occupancy may be stored as a named bit σ that gates the
    action drive in epoch β. No-store ignores σ. Finite construction
    only. Does not claim autonomy or a brain.
    """

    start: HybridState
    body: RoutedTwoCopy
    sigma: int | None = None
    parameters: NominalParameters = NOMINAL_PARAMETERS
    kappa: Fraction = Fraction(1, 4)

    def __post_init__(self) -> None:
        if not isinstance(self.start, HybridState):
            raise TypeError("start must be a HybridState")
        if not isinstance(self.body, RoutedTwoCopy):
            raise TypeError("body must be a RoutedTwoCopy")
        if self.sigma is not None:
            object.__setattr__(self, "sigma", _bit(self.sigma, "sigma"))
        object.__setattr__(self, "kappa", admissible_kappa(self.kappa))

    @classmethod
    def washed(
        cls,
        start: HybridState | None = None,
        *,
        kappa: Fraction = Fraction(1, 4),
        parameters: NominalParameters = NOMINAL_PARAMETERS,
        sigma: int | None = None,
    ) -> "WashedRoleSplit":
        seed = registered_start() if start is None else start
        if not isinstance(seed, HybridState):
            raise TypeError("start must be a HybridState")
        return cls(
            start=seed,
            body=RoutedTwoCopy(
                left=seed,
                right=seed,
                parameters=parameters,
                kappa=kappa,
            ),
            sigma=sigma,
            parameters=parameters,
            kappa=kappa,
        )

    def wash(self) -> "WashedRoleSplit":
        """Reset both copies to start. Named σ is kept if present."""

        return WashedRoleSplit(
            start=self.start,
            body=RoutedTwoCopy(
                left=self.start,
                right=self.start,
                parameters=self.parameters,
                kappa=self.kappa,
            ),
            sigma=self.sigma,
            parameters=self.parameters,
            kappa=self.kappa,
        )

    def with_sigma(self, sigma: object) -> "WashedRoleSplit":
        return WashedRoleSplit(
            start=self.start,
            body=self.body,
            sigma=_bit(sigma, "sigma"),
            parameters=self.parameters,
            kappa=self.kappa,
        )

    def iterate_routed(
        self,
        ticks: int,
        flux: object,
        *,
        weight: object = IDENTITY_WEIGHT,
    ) -> "WashedRoleSplit":
        """Ordinary L4 epoch. Used for α and for no-store β."""

        next_body = self.body.iterate_routed(ticks, weight, flux)
        return WashedRoleSplit(
            start=self.start,
            body=next_body,
            sigma=self.sigma,
            parameters=self.parameters,
            kappa=self.kappa,
        )

    def iterate_role_split(
        self,
        ticks: int,
        flux: object,
        *,
        weight: object = IDENTITY_WEIGHT,
    ) -> "WashedRoleSplit":
        if self.sigma is None:
            raise ValueError("sigma must be a written bit before a role-split epoch")
        drive_left, drive_right = role_split_drives(self.sigma, flux, weight=weight)
        next_body = self.body.iterate(ticks, drive_left, drive_right)
        return WashedRoleSplit(
            start=self.start,
            body=next_body,
            sigma=self.sigma,
            parameters=self.parameters,
            kappa=self.kappa,
        )

    def iterate_loop_gate(
        self,
        ticks: int,
        flux: object,
        named_i: object,
        *,
        weight: object = IDENTITY_WEIGHT,
    ) -> "WashedRoleSplit":
        """Epoch γ under (L7.3). Named ``I`` gates the action drive. σ is kept."""

        drive_left, drive_right = loop_gate_drives(named_i, flux, weight=weight)
        next_body = self.body.iterate(ticks, drive_left, drive_right)
        return WashedRoleSplit(
            start=self.start,
            body=next_body,
            sigma=self.sigma,
            parameters=self.parameters,
            kappa=self.kappa,
        )

    def sensor_occupancy(self) -> int:
        return occupancy_bit(self.body.left)

    def action_occupancy(self) -> int:
        return occupancy_bit(self.body.right)


ACTIVITY_PAIR_CIRC_MASS = Fraction(7, 15)
ACTIVITY_PAIR_SIGMA = 1
ACTIVITY_NEXT_STAR = (Fraction(7187, 12672), Fraction(491, 990))
ACTIVITY_NEXT_CIRC = (Fraction(16891, 29700), Fraction(133, 270))
ACTIVITY_DELTA_MASS = Fraction(-1487, 950400)
ACTIVITY_DELTA_BOUNDARY = Fraction(1, 297)


def registered_activity_pair() -> tuple[HybridState, HybridState]:
    """Registered pair (L6.1): center and one other interior point of U0.

    Both carry the same registered bit ``σ = 1``.  Finite pair
    construction.  Does not claim autonomy or a brain.
    """

    star = registered_start()
    circ = HybridState.from_values(
        ACTIVITY_PAIR_CIRC_MASS,
        REGISTERED_START_BOUNDARY,
        REGISTERED_START_LABEL,
    )
    return star, circ


def activity_readout(
    state: HybridState,
    *,
    kappa: Fraction = Fraction(1, 4),
    drive: object = Fraction(1),
) -> tuple[Fraction, Fraction]:
    """One-step ``(m', b')``.  Label ``q'`` is discarded."""

    nxt = source_hybrid_step(state, kappa=kappa, drive=drive)
    return nxt.mass, nxt.boundary


def overwrite_sigma_from_action(host: WashedRoleSplit) -> WashedRoleSplit:
    """σ ← o^A on the same named slot. Not a third cube."""

    if not isinstance(host, WashedRoleSplit):
        raise TypeError("host must be a WashedRoleSplit")
    return host.with_sigma(host.action_occupancy())


@dataclass(frozen=True)
class HostTuple:
    """Registered L8 host ``H = (t, E, Z^S, Z^A, σ, I)``.

    Same slots the L7 construction already advances.  Finite pair
    construction.  Does not claim autonomy or a brain.
    """

    tick: int
    flux: tuple[Fraction, Fraction]
    sensor: HybridState
    action: HybridState
    sigma: int
    named_i: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "tick", _nonnegative_int(self.tick, "tick"))
        object.__setattr__(self, "flux", _flux_pair(self.flux))
        if not isinstance(self.sensor, HybridState):
            raise TypeError("sensor must be a HybridState")
        if not isinstance(self.action, HybridState):
            raise TypeError("action must be a HybridState")
        object.__setattr__(self, "sigma", _bit(self.sigma, "sigma"))
        object.__setattr__(self, "named_i", _bit(self.named_i, "named_i"))


def registered_host_pair() -> tuple[HostTuple, HostTuple]:
    """Registered set ``S = {H_★, H_◦}`` from the L6 activity pair.

    Both points use ``E = e^{(2)}``, ``σ = 1``, ``I = 1``, ``t = 0``,
    and ``Z^S = Z^A``.  Finite pair construction.  Does not claim
    autonomy or a brain.
    """

    star, circ = registered_activity_pair()
    flux = REGISTERED_FLUX_E2
    return (
        HostTuple(
            tick=0,
            flux=flux,
            sensor=star,
            action=star,
            sigma=ACTIVITY_PAIR_SIGMA,
            named_i=1,
        ),
        HostTuple(
            tick=0,
            flux=flux,
            sensor=circ,
            action=circ,
            sigma=ACTIVITY_PAIR_SIGMA,
            named_i=1,
        ),
    )


def internal_kernel(
    host: HostTuple,
    *,
    weight: object = IDENTITY_WEIGHT,
    kappa: Fraction = Fraction(1, 4),
) -> HostTuple:
    """One-step ``Φ`` typed ``H → H``.  Registered ``K`` is ``Φ`` itself.

    Each cube follows ``F_{1/4}`` with ``W = I`` gates from ``(σ, I, E)``
    as in L5--L7.  ``t ↦ t+1``.  Flux and bits are held.  No wash.
    Finite pair construction.  Does not claim autonomy or a brain.
    """

    if not isinstance(host, HostTuple):
        raise TypeError("host must be a HostTuple")
    drive_sensor, drive_action = loop_gate_drives(
        host.named_i,
        host.flux,
        weight=weight,
    )
    return HostTuple(
        tick=host.tick + 1,
        flux=host.flux,
        sensor=source_hybrid_step(host.sensor, kappa=kappa, drive=drive_sensor),
        action=source_hybrid_step(host.action, kappa=kappa, drive=drive_action),
        sigma=host.sigma,
        named_i=host.named_i,
    )
