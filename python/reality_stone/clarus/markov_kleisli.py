"""Minimal probability/state Kleisli algebra for CloudCell contracts.

The monad is the probability-and-state context

    T_S(X) = S -> D(S x X),

where ``D`` is a finite probability distribution.  A stateful CloudCell is a
Kleisli arrow ``U -> T_S(O)``; the cell is not itself the monad.

This module intentionally implements only finite-support distributions.  A
runtime with deterministic pseudo-randomness embeds through a Dirac mass.  A
model with genuine continuous noise needs a measurable-space probability
kernel (for example a Giry-kernel implementation), not an empirical sample
masquerading as an exact distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Generic, Hashable, TypeVar


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
S = TypeVar("S")
K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True)
class ProbabilityAtom(Generic[T]):
    """One value and its nonnegative probability mass."""

    value: T
    probability: float

    def __post_init__(self) -> None:
        probability = float(self.probability)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("probability mass must be finite and nonnegative")
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True)
class FiniteDistribution(Generic[T]):
    """Finite probability distribution with explicit support multiplicity."""

    atoms: tuple[ProbabilityAtom[T], ...]

    def __post_init__(self) -> None:
        if not self.atoms:
            raise ValueError("a probability distribution needs nonempty support")
        total = math.fsum(atom.probability for atom in self.atoms)
        if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"probability mass must sum to one; got {total:.17g}")

    @classmethod
    def pure(cls, value: T) -> "FiniteDistribution[T]":
        """Probability-monad unit ``eta(value)``."""

        return cls((ProbabilityAtom(value, 1.0),))

    def map(self, transform: Callable[[T], U]) -> "FiniteDistribution[U]":
        return FiniteDistribution(
            tuple(ProbabilityAtom(transform(atom.value), atom.probability) for atom in self.atoms)
        )

    def bind(
        self,
        transition: Callable[[T], "FiniteDistribution[U]"],
    ) -> "FiniteDistribution[U]":
        """Probability-monad bind, multiplying mass along each branch."""

        composed: list[ProbabilityAtom[U]] = []
        for source in self.atoms:
            branch = transition(source.value)
            composed.extend(
                ProbabilityAtom(target.value, source.probability * target.probability)
                for target in branch.atoms
            )
        return FiniteDistribution(tuple(composed))

    def mass_by(self, key: Callable[[T], K]) -> dict[K, float]:
        """Aggregate duplicate support values under a caller-supplied key."""

        result: dict[K, float] = {}
        for atom in self.atoms:
            atom_key = key(atom.value)
            result[atom_key] = result.get(atom_key, 0.0) + atom.probability
        return result

    def equivalent(
        self,
        other: "FiniteDistribution[T]",
        *,
        key: Callable[[T], K],
        abs_tol: float = 1e-12,
    ) -> bool:
        """Compare probability measures after coalescing duplicate support."""

        left = self.mass_by(key)
        right = other.mass_by(key)
        if left.keys() != right.keys():
            return False
        return all(
            math.isclose(left[item], right[item], rel_tol=0.0, abs_tol=abs_tol)
            for item in left
        )


StateKernel = Callable[[S], FiniteDistribution[tuple[S, T]]]
KleisliArrow = Callable[[T], StateKernel[S, U]]


def state_pure(value: T) -> StateKernel[S, T]:
    """State-probability monad unit: leave state unchanged."""

    def run(state: S) -> FiniteDistribution[tuple[S, T]]:
        return FiniteDistribution.pure((state, value))

    return run


def state_bind(
    computation: StateKernel[S, T],
    transition: Callable[[T], StateKernel[S, U]],
) -> StateKernel[S, U]:
    """State-probability bind, threading every branch's resulting state."""

    def run(state: S) -> FiniteDistribution[tuple[S, U]]:
        return computation(state).bind(
            lambda state_and_value: transition(state_and_value[1])(state_and_value[0])
        )

    return run


def kleisli_identity(value: T) -> StateKernel[S, T]:
    """Identity arrow in the Kleisli category of ``T_S``."""

    return state_pure(value)


def kleisli_compose(
    first: KleisliArrow[T, S, U],
    second: KleisliArrow[U, S, V],
) -> KleisliArrow[T, S, V]:
    """Compose ``T -> T_S(U)`` and ``U -> T_S(V)``."""

    def composed(value: T) -> StateKernel[S, V]:
        return state_bind(first(value), second)

    return composed


def deterministic_kleisli_arrow(
    transition: Callable[[S, T], tuple[S, U]],
) -> KleisliArrow[T, S, U]:
    """Lift a pure state transition into a Dirac-valued Kleisli arrow."""

    def arrow(value: T) -> StateKernel[S, U]:
        def run(state: S) -> FiniteDistribution[tuple[S, U]]:
            return FiniteDistribution.pure(transition(state, value))

        return run

    return arrow


__all__ = [
    "FiniteDistribution",
    "KleisliArrow",
    "ProbabilityAtom",
    "StateKernel",
    "deterministic_kleisli_arrow",
    "kleisli_compose",
    "kleisli_identity",
    "state_bind",
    "state_pure",
]
