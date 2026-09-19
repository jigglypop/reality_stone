"""Dimensionless bookkeeping for CE gates.

CE calculations should close first on dimensionless ratios.  This module
provides small exact-rational tools for checking that rule without pulling in
symbolic math dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import prod
from typing import Callable, Generic, Iterable, Sequence, TypeVar


DimVector = tuple[Fraction, ...]
T = TypeVar("T")
U = TypeVar("U")


def _frac(x: int | float | Fraction) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator()


def dim(*exponents: int | float | Fraction) -> DimVector:
    """Build a dimension vector, e.g. ``dim(1, 0, -2)`` for M L^0 T^-2."""

    return tuple(_frac(x) for x in exponents)


DIMENSIONLESS: DimVector = dim(0, 0, 0, 0)
MASS: DimVector = dim(1, 0, 0, 0)
LENGTH: DimVector = dim(0, 1, 0, 0)
TIME: DimVector = dim(0, 0, 1, 0)
TEMPERATURE: DimVector = dim(0, 0, 0, 1)
ENERGY: DimVector = MASS
CURVATURE: DimVector = dim(0, -2, 0, 0)
ACTION: DimVector = dim(1, 2, -1, 0)


@dataclass(frozen=True)
class Quantity:
    """Numeric value with a base-dimension exponent vector."""

    name: str
    value: float
    dims: DimVector = DIMENSIONLESS

    @property
    def dimensionless(self) -> bool:
        return is_dimensionless(self.dims)


@dataclass(frozen=True)
class GateResult(Generic[T]):
    """Small Result/Either value for composing CE gate checks."""

    value: T | None = None
    errors: tuple[str, ...] = ()

    @classmethod
    def ok(cls, value: T) -> "GateResult[T]":
        return cls(value=value)

    @classmethod
    def fail(cls, *errors: str) -> "GateResult[T]":
        return cls(errors=tuple(error for error in errors if error))

    @property
    def passed(self) -> bool:
        return not self.errors

    def map(self, transform: Callable[[T], U]) -> "GateResult[U]":
        if not self.passed:
            return GateResult.fail(*self.errors)
        return GateResult.ok(transform(self.unwrap()))

    def bind(self, transform: Callable[[T], "GateResult[U]"]) -> "GateResult[U]":
        if not self.passed:
            return GateResult.fail(*self.errors)
        return transform(self.unwrap())

    def unwrap(self) -> T:
        if self.errors:
            raise ValueError("; ".join(self.errors))
        if self.value is None:
            raise ValueError("gate result has no value")
        return self.value


def is_dimensionless(dims: Sequence[Fraction]) -> bool:
    return all(_frac(x) == 0 for x in dims)


def same_rank_dims(quantities: Iterable[Quantity]) -> int:
    lengths = {len(q.dims) for q in quantities}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent dimension ranks: {sorted(lengths)}")
    return lengths.pop()


def require_dimensionless(quantity: Quantity, *, context: str = "") -> Quantity:
    """Raise if a quantity is not dimensionless."""

    if not quantity.dimensionless:
        where = f" for {context}" if context else ""
        raise ValueError(f"{quantity.name} must be dimensionless{where}; dims={quantity.dims}")
    return quantity


def check_dimensionless(quantity: Quantity, *, context: str = "") -> GateResult[Quantity]:
    """Return a composable gate result instead of raising on dimensional failure."""

    if quantity.dimensionless:
        return GateResult.ok(quantity)
    where = f" for {context}" if context else ""
    return GateResult.fail(f"{quantity.name} must be dimensionless{where}; dims={quantity.dims}")


def audit_dimensionless(
    quantities: Iterable[Quantity],
    *,
    context: str = "",
) -> GateResult[tuple[Quantity, ...]]:
    """Validate many quantities and accumulate every dimensional violation."""

    accepted: list[Quantity] = []
    errors: list[str] = []
    for quantity in quantities:
        check = check_dimensionless(quantity, context=context)
        if check.passed:
            accepted.append(check.unwrap())
        else:
            errors.extend(check.errors)
    if errors:
        return GateResult.fail(*errors)
    return GateResult.ok(tuple(accepted))


def _rref(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    rows = [row[:] for row in matrix]
    if not rows:
        return rows, []
    n_rows, n_cols = len(rows), len(rows[0])
    pivots: list[int] = []
    r = 0
    for c in range(n_cols):
        pivot = next((i for i in range(r, n_rows) if rows[i][c] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = Fraction(1, 1) / rows[r][c]
        rows[r] = [x * inv for x in rows[r]]
        for i in range(n_rows):
            if i != r and rows[i][c] != 0:
                factor = rows[i][c]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[r])]
        pivots.append(c)
        r += 1
        if r == n_rows:
            break
    return rows, pivots


def nullspace(matrix: Sequence[Sequence[int | float | Fraction]]) -> list[list[Fraction]]:
    """Exact rational basis for ``matrix @ x = 0``."""

    rows = [[_frac(x) for x in row] for row in matrix]
    if not rows:
        return []
    n_cols = len(rows[0])
    if any(len(row) != n_cols for row in rows):
        raise ValueError("ragged matrix")
    rref, pivots = _rref(rows)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n_cols) if c not in pivot_set]
    basis: list[list[Fraction]] = []
    for free in free_cols:
        vec = [Fraction(0, 1) for _ in range(n_cols)]
        vec[free] = Fraction(1, 1)
        for row_idx, pivot_col in enumerate(pivots):
            vec[pivot_col] = -rref[row_idx][free]
        basis.append(vec)
    return basis


def buckingham_pi_groups(quantities: Sequence[Quantity]) -> list[dict[str, Fraction]]:
    """Return exponent maps for independent dimensionless Pi groups."""

    if not quantities:
        return []
    rank = same_rank_dims(quantities)
    matrix = [[q.dims[row] for q in quantities] for row in range(rank)]
    groups = []
    for vec in nullspace(matrix):
        groups.append({q.name: exponent for q, exponent in zip(quantities, vec) if exponent != 0})
    return groups


def evaluate_group(quantities: Sequence[Quantity], exponents: dict[str, Fraction]) -> float:
    by_name = {q.name: q for q in quantities}
    missing = sorted(set(exponents) - set(by_name))
    if missing:
        raise KeyError(f"unknown quantities in group: {missing}")
    return prod(by_name[name].value ** float(power) for name, power in exponents.items())


def group_dimension(quantities: Sequence[Quantity], exponents: dict[str, Fraction]) -> DimVector:
    by_name = {q.name: q for q in quantities}
    rank = same_rank_dims(quantities)
    out = [Fraction(0, 1) for _ in range(rank)]
    for name, power in exponents.items():
        q = by_name[name]
        for i, exponent in enumerate(q.dims):
            out[i] += exponent * power
    return tuple(out)


def nondimensionalize(quantity: Quantity, scales: Sequence[Quantity]) -> Quantity:
    """Divide ``quantity`` by dimensional scales that span its dimension vector."""

    all_q = [*scales, quantity]
    rank = same_rank_dims(all_q)
    matrix = [[scale.dims[row] for scale in scales] for row in range(rank)]
    target = [-quantity.dims[row] for row in range(rank)]
    augmented = [row + [rhs] for row, rhs in zip(matrix, target)]
    rref, pivots = _rref(augmented)
    if len(scales) in pivots:
        raise ValueError(f"{quantity.name} dimensions cannot be spanned by supplied scales")
    powers = [Fraction(0, 1) for _ in scales]
    for row_idx, pivot_col in enumerate(pivots):
        if pivot_col < len(scales):
            powers[pivot_col] = rref[row_idx][-1]
    value = quantity.value * prod(scale.value ** float(power) for scale, power in zip(scales, powers))
    scale_part = " ".join(f"{s.name}^{p}" for s, p in zip(scales, powers) if p)
    name = f"{quantity.name}*{scale_part}" if scale_part else quantity.name
    return Quantity(name=name, value=value, dims=DIMENSIONLESS)


def exp_argument(quantity: Quantity) -> float:
    """Return a value that is safe to place in exp/log-like CE kernels."""

    require_dimensionless(quantity, context="exponential/logarithmic kernel")
    return quantity.value


def exp_arguments(quantities: Iterable[Quantity]) -> GateResult[tuple[float, ...]]:
    """Validate a batch of exp/log arguments and return their raw values."""

    return audit_dimensionless(
        quantities,
        context="exponential/logarithmic kernel",
    ).map(lambda checked: tuple(quantity.value for quantity in checked))
