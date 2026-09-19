"""Finite orbit quotients for expanding causal recursion networks.

"Indra net" is used only as a mnemonic. The implemented object is an equitable
partition of a nonnegative row-source coupling matrix. Exact quotient closure
is a theorem of that finite object; infinite-network language requires the
separate local-finiteness and bounded-row-sum assumptions documented by CE.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reality_stone.clarus.multispace_bootstrap import (
    minimal_multispace_fixed_point,
    multispace_bootstrap_map,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OrbitQuotient:
    orbit_labels: tuple[Hashable, ...]
    node_orbits: tuple[int, ...]
    coupling: tuple[tuple[float, ...], ...]
    maximum_block_sum_error: float

    def as_array(self) -> FloatArray:
        return np.asarray(self.coupling, dtype=np.float64)


@dataclass(frozen=True)
class CausalCone:
    active_nodes: tuple[int, ...]
    generations_completed: int
    budget_exhausted: bool


def _coupling_matrix(coupling: ArrayLike) -> FloatArray:
    matrix = np.asarray(coupling, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("coupling must be a non-empty square matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("coupling must be finite and nonnegative")
    return matrix


def equitable_orbit_quotient(
    coupling: ArrayLike,
    orbit_labels: Sequence[Hashable],
    *,
    tolerance: float = 1e-12,
) -> OrbitQuotient:
    """Return the exact row-block-sum quotient of an equitable partition."""

    matrix = _coupling_matrix(coupling)
    if len(orbit_labels) != matrix.shape[0]:
        raise ValueError("orbit_labels must contain one label per node")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    labels: list[Hashable] = []
    label_to_index: dict[Hashable, int] = {}
    node_orbits: list[int] = []
    for label in orbit_labels:
        if label not in label_to_index:
            label_to_index[label] = len(labels)
            labels.append(label)
        node_orbits.append(label_to_index[label])
    orbit_count = len(labels)
    members = [
        np.flatnonzero(np.asarray(node_orbits, dtype=np.int64) == orbit)
        for orbit in range(orbit_count)
    ]
    quotient = np.zeros((orbit_count, orbit_count), dtype=np.float64)
    maximum_error = 0.0
    for source_orbit, source_members in enumerate(members):
        reference: np.ndarray | None = None
        for source in source_members:
            block_sums = np.asarray(
                [float(np.sum(matrix[int(source), target_members])) for target_members in members]
            )
            if reference is None:
                reference = block_sums
                quotient[source_orbit] = block_sums
            else:
                maximum_error = max(maximum_error, float(np.max(np.abs(block_sums - reference))))
    if maximum_error > tolerance:
        raise ValueError(
            "partition is not equitable: row block sums differ within an orbit"
        )
    return OrbitQuotient(
        orbit_labels=tuple(labels),
        node_orbits=tuple(node_orbits),
        coupling=tuple(tuple(float(value) for value in row) for row in quotient),
        maximum_block_sum_error=maximum_error,
    )


def lift_orbit_state(orbit_state: ArrayLike, quotient: OrbitQuotient) -> FloatArray:
    state = np.asarray(orbit_state, dtype=np.float64)
    if state.shape != (len(quotient.orbit_labels),):
        raise ValueError("orbit_state has the wrong dimension")
    if not np.all(np.isfinite(state)):
        raise ValueError("orbit_state must be finite")
    return state[np.asarray(quotient.node_orbits, dtype=np.int64)]


def quotient_closure_error(
    coupling: ArrayLike,
    quotient: OrbitQuotient,
    orbit_state: ArrayLike,
) -> float:
    """Numerically audit ``F_A(Lq) = L F_Abar(q)``."""

    full_state = lift_orbit_state(orbit_state, quotient)
    full_next = multispace_bootstrap_map(full_state, coupling)
    quotient_next = multispace_bootstrap_map(orbit_state, quotient.as_array())
    lifted_next = lift_orbit_state(quotient_next, quotient)
    return float(np.max(np.abs(full_next - lifted_next)))


def normalized_orbit_expansion(
    quotient_coupling: ArrayLike,
    multiplicities: Sequence[int],
) -> tuple[FloatArray, tuple[int, ...]]:
    """Expand a finite quotient while preserving every row-block sum exactly."""

    quotient = _coupling_matrix(quotient_coupling)
    if len(multiplicities) != quotient.shape[0] or any(value <= 0 for value in multiplicities):
        raise ValueError("multiplicities must be positive with one entry per orbit")
    labels = tuple(
        orbit for orbit, count in enumerate(multiplicities) for _ in range(int(count))
    )
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    members = [np.flatnonzero(np.asarray(labels) == orbit) for orbit in range(quotient.shape[0])]
    for source, source_orbit in enumerate(labels):
        for target_orbit, target_members in enumerate(members):
            matrix[source, target_members] = (
                quotient[source_orbit, target_orbit] / float(target_members.size)
            )
    return matrix, labels


def finite_causal_cone(
    adjacency: Mapping[int, Iterable[int]],
    seeds: Iterable[int],
    *,
    generations: int,
    active_budget: int | None = None,
) -> CausalCone:
    """Explore only the finite forward cone reachable within a time horizon."""

    if generations < 0:
        raise ValueError("generations must be nonnegative")
    if active_budget is not None and active_budget <= 0:
        raise ValueError("active_budget must be positive when supplied")
    active = set(int(node) for node in seeds)
    if not active:
        raise ValueError("at least one causal seed is required")
    if active_budget is not None and len(active) > active_budget:
        ordered = tuple(sorted(active)[:active_budget])
        return CausalCone(ordered, 0, True)
    frontier = set(active)
    completed = 0
    exhausted = False
    for generation in range(1, generations + 1):
        next_frontier = {
            int(target)
            for source in frontier
            for target in adjacency.get(source, ())
            if int(target) not in active
        }
        if active_budget is not None and len(active) + len(next_frontier) > active_budget:
            remaining = active_budget - len(active)
            active.update(sorted(next_frontier)[:remaining])
            exhausted = True
            break
        active.update(next_frontier)
        frontier = next_frontier
        completed = generation
        if not frontier:
            break
    return CausalCone(tuple(sorted(active)), completed, exhausted)


def evaluate_orbit_scaling() -> dict[str, object]:
    """Compare growing full networks with one fixed three-orbit quotient."""

    base = np.asarray(
        (
            (1.15, 0.35, 0.10),
            (0.20, 1.10, 0.35),
            (0.15, 0.25, 1.20),
        ),
        dtype=np.float64,
    )
    quotient_fixed = minimal_multispace_fixed_point(base)
    sizes = (1, 2, 4, 8, 16)
    rows: list[dict[str, float | int]] = []
    maximum_closure_error = 0.0
    maximum_fixed_point_error = 0.0
    for size in sizes:
        full, labels = normalized_orbit_expansion(base, (size, size + 1, size + 2))
        quotient = equitable_orbit_quotient(full, labels)
        closure_error = quotient_closure_error(
            full, quotient, np.asarray((0.2, 0.5, 0.8), dtype=np.float64)
        )
        full_fixed = minimal_multispace_fixed_point(full)
        lifted_fixed = lift_orbit_state(quotient_fixed.as_array(), quotient)
        fixed_error = float(np.max(np.abs(full_fixed.as_array() - lifted_fixed)))
        maximum_closure_error = max(maximum_closure_error, closure_error)
        maximum_fixed_point_error = max(maximum_fixed_point_error, fixed_error)
        rows.append(
            {
                "nodes": full.shape[0],
                "quotient_nodes": base.shape[0],
                "closure_error": closure_error,
                "fixed_point_error": fixed_error,
            }
        )

    broken, broken_labels = normalized_orbit_expansion(base, (4, 5, 6))
    broken[0, np.asarray(broken_labels) == 1] *= 1.1
    symmetry_break_detected = False
    try:
        equitable_orbit_quotient(broken, broken_labels)
    except ValueError:
        symmetry_break_detected = True

    radius = 200
    line = {
        node: tuple(target for target in (node - 1, node + 1) if -radius <= target <= radius)
        for node in range(-radius, radius + 1)
    }
    cone = finite_causal_cone(line, (0,), generations=12)
    budgeted = finite_causal_cone(line, (0,), generations=12, active_budget=10)

    # Non-commuting limits: every finite open forward chain is nilpotent and
    # becomes extinct, while the translation-orbit quotient of the countably
    # infinite bulk is the one-type Poisson process with mean ``chain_depth``.
    chain_depth = 2.0
    finite_open_extinctions: list[float] = []
    periodic_bulk_errors: list[float] = []
    bulk_fixed = minimal_multispace_fixed_point(((chain_depth,),)).survival[0]
    for size in (4, 8, 16, 32):
        open_chain = np.zeros((size, size), dtype=np.float64)
        for source in range(size - 1):
            open_chain[source, source + 1] = chain_depth
        finite_open_extinctions.append(
            min(minimal_multispace_fixed_point(open_chain).survival)
        )
        periodic = open_chain.copy()
        periodic[-1, 0] = chain_depth
        periodic_fixed = minimal_multispace_fixed_point(periodic).as_array()
        periodic_bulk_errors.append(
            float(np.max(np.abs(periodic_fixed - bulk_fixed)))
        )
    infinite_chain_counterexample = (
        all(value == 1.0 for value in finite_open_extinctions)
        and bulk_fixed < 1.0
        and max(periodic_bulk_errors) <= 1e-12
    )
    gates = {
        "exact_quotient_closure": maximum_closure_error <= 1e-13,
        "fixed_point_size_invariance": maximum_fixed_point_error <= 1e-11,
        "quotient_dimension_fixed": all(row["quotient_nodes"] == 3 for row in rows),
        "symmetry_break_detected": symmetry_break_detected,
        "finite_causal_cone": len(cone.active_nodes) == 25,
        "active_budget_enforced": len(budgeted.active_nodes) == 10 and budgeted.budget_exhausted,
        "infinite_chain_scc_counterexample_detected": infinite_chain_counterexample,
    }
    return {
        "schema": "clarus.indra-orbit-causal-quotient.validation.v1",
        "rows": rows,
        "maximum_closure_error": maximum_closure_error,
        "maximum_fixed_point_error": maximum_fixed_point_error,
        "causal_cone_nodes_at_12": len(cone.active_nodes),
        "budgeted_cone_nodes": len(budgeted.active_nodes),
        "finite_open_chain_min_extinction": finite_open_extinctions,
        "translation_quotient_extinction": bulk_fixed,
        "periodic_bulk_max_error": max(periodic_bulk_errors),
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "STOP",
    }


__all__ = [
    "CausalCone",
    "OrbitQuotient",
    "equitable_orbit_quotient",
    "evaluate_orbit_scaling",
    "finite_causal_cone",
    "lift_orbit_state",
    "normalized_orbit_expansion",
    "quotient_closure_error",
]
