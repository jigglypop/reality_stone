"""Finite pre-equality probability tools.

This module implements the closed mathematical core from
``docs/9_등호이전``.  It deliberately stays finite-dimensional: no CE path
space, no physical readout, and no AGI performance claim lives here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


ArrayLike = Iterable[float] | np.ndarray


def _as_float_array(value: ArrayLike, *, name: str, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def normalize_weights(weights: ArrayLike) -> np.ndarray:
    """Normalize non-negative finite weights to a probability vector."""
    w = _as_float_array(weights, name="weights", ndim=1)
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return w / total


def gibbs_reweight(prior: ArrayLike, energy: ArrayLike, beta: float) -> np.ndarray:
    """Apply finite Gibbs/PreEq reweighting to a prior probability vector."""
    p = normalize_weights(prior)
    e = _as_float_array(energy, name="energy", ndim=1)
    if p.shape != e.shape:
        raise ValueError("prior and energy must have the same shape")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if np.any(e < 0.0):
        raise ValueError("energy must be non-negative")

    support = p > 0.0
    finite_support = support & np.isfinite(e)
    if not bool(np.any(finite_support)):
        raise ValueError("at least one supported candidate must have finite energy")

    m = float(np.min(e[finite_support]))
    shifted = np.zeros_like(p)
    shifted[finite_support] = np.exp(-float(beta) * (e[finite_support] - m)) * p[finite_support]
    total = float(shifted.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("partition function is not positive and finite")
    return shifted / total


def manifest_indices(prior: ArrayLike, energy: ArrayLike, *, atol: float = 0.0) -> np.ndarray:
    """Return supported indices attaining the minimum finite energy."""
    p = normalize_weights(prior)
    e = _as_float_array(energy, name="energy", ndim=1)
    if p.shape != e.shape:
        raise ValueError("prior and energy must have the same shape")
    support = p > 0.0
    finite_support = support & np.isfinite(e)
    if not bool(np.any(finite_support)):
        raise ValueError("at least one supported candidate must have finite energy")
    m = float(np.min(e[finite_support]))
    return np.flatnonzero(finite_support & (e <= m + float(atol)))


@dataclass(frozen=True)
class ResidualMeasure:
    """Raw and conditional non-selected residual mass."""

    raw: np.ndarray
    mass: float

    @property
    def conditional(self) -> np.ndarray:
        if self.mass <= 0.0:
            return np.zeros_like(self.raw)
        return self.raw / self.mass


def nonselected_residual(
    posterior: ArrayLike,
    selected: Iterable[int] | np.ndarray,
) -> ResidualMeasure:
    """Restrict a posterior distribution to the non-selected complement."""
    p = normalize_weights(posterior)
    mask = np.ones_like(p, dtype=bool)
    selected_idx = np.asarray(list(selected), dtype=int)
    if selected_idx.size:
        if np.any((selected_idx < 0) | (selected_idx >= p.size)):
            raise ValueError("selected index out of range")
        mask[selected_idx] = False
    raw = np.where(mask, p, 0.0)
    return ResidualMeasure(raw=raw, mass=float(raw.sum()))


def compose_weighted_kernels(first: ArrayLike, second: ArrayLike) -> np.ndarray:
    """Compose non-negative kernels using matrix multiplication.

    ``first`` has shape ``(A, B)`` and ``second`` has shape ``(B, C)``.
    """
    k = _as_float_array(first, name="first", ndim=2)
    l = _as_float_array(second, name="second", ndim=2)
    if k.shape[1] != l.shape[0]:
        raise ValueError("kernel dimensions do not compose")
    for name, arr in (("first", k), ("second", l)):
        if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} kernel must be finite and non-negative")
        if np.any(arr.sum(axis=1) <= 0.0):
            raise ValueError(f"{name} kernel must have non-zero rows")
    out = k @ l
    if np.any(out.sum(axis=1) <= 0.0):
        raise ValueError("composed kernel has a zero row")
    return out


def kernel_pushforward(prior: ArrayLike, kernel: ArrayLike, *, normalize: bool = True) -> np.ndarray:
    """Push a row-vector state through a finite weighted kernel."""
    p = normalize_weights(prior)
    k = _as_float_array(kernel, name="kernel", ndim=2)
    if p.size != k.shape[0]:
        raise ValueError("prior length must match kernel row count")
    if np.any(k < 0.0) or not np.all(np.isfinite(k)):
        raise ValueError("kernel must be finite and non-negative")
    if np.any(k.sum(axis=1) <= 0.0):
        raise ValueError("kernel must have non-zero rows")
    pushed = p @ k
    return normalize_weights(pushed) if normalize else pushed


def tropical_compose(first_energy: ArrayLike, second_energy: ArrayLike) -> np.ndarray:
    """Compose finite/extended energy kernels by min-plus convolution."""
    e = _as_float_array(first_energy, name="first_energy", ndim=2)
    f = _as_float_array(second_energy, name="second_energy", ndim=2)
    if e.shape[1] != f.shape[0]:
        raise ValueError("energy kernel dimensions do not compose")
    if np.any(e < 0.0) or np.any(f < 0.0):
        raise ValueError("energy kernels must be non-negative")
    return np.min(e[:, :, None] + f[None, :, :], axis=1)


def gibbs_kernel(energy: ArrayLike, beta: float) -> np.ndarray:
    """Convert an extended non-negative energy kernel into a Gibbs kernel."""
    e = _as_float_array(energy, name="energy", ndim=2)
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if np.any(e < 0.0):
        raise ValueError("energy must be non-negative")
    return np.exp(-float(beta) * e, where=np.isfinite(e), out=np.zeros_like(e))


def tropicalize(kernel: ArrayLike, beta: float) -> np.ndarray:
    """Map a non-negative kernel back to beta-scaled energy."""
    k = _as_float_array(kernel, name="kernel", ndim=2)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if np.any(k < 0.0) or not np.all(np.isfinite(k)):
        raise ValueError("kernel must be finite and non-negative")
    out = np.full_like(k, np.inf)
    positive = k > 0.0
    out[positive] = -np.log(k[positive]) / float(beta)
    return out


def joint_gibbs(prior: ArrayLike, energy: ArrayLike, beta: float) -> np.ndarray:
    """Apply Gibbs reweighting to a finite condition-value joint state."""
    p = _as_float_array(prior, name="prior", ndim=2)
    e = _as_float_array(energy, name="energy", ndim=2)
    if p.shape != e.shape:
        raise ValueError("prior and energy must have the same shape")
    flat = gibbs_reweight(p.reshape(-1), e.reshape(-1), beta)
    return flat.reshape(p.shape)


def marginals(joint: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return condition and value marginals for a joint state."""
    rho = _as_float_array(joint, name="joint", ndim=2)
    rho = normalize_weights(rho.reshape(-1)).reshape(rho.shape)
    return rho.sum(axis=1), rho.sum(axis=0)


def conditional_values(joint: ArrayLike, condition_index: int) -> np.ndarray:
    """Return P(value | condition) for a finite joint state."""
    rho = _as_float_array(joint, name="joint", ndim=2)
    rho = normalize_weights(rho.reshape(-1)).reshape(rho.shape)
    k = int(condition_index)
    if k < 0 or k >= rho.shape[0]:
        raise ValueError("condition_index out of range")
    row_mass = float(rho[k].sum())
    if row_mass <= 0.0:
        raise ValueError("condition has zero marginal mass")
    return rho[k] / row_mass


def free_energy_for_conditions(joint_prior: ArrayLike, energy: ArrayLike, beta: float) -> np.ndarray:
    """Finite condition free energies after summing over values."""
    rho = _as_float_array(joint_prior, name="joint_prior", ndim=2)
    e = _as_float_array(energy, name="energy", ndim=2)
    if rho.shape != e.shape:
        raise ValueError("joint_prior and energy must have the same shape")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    cond_mass = rho.sum(axis=1)
    out = np.full(rho.shape[0], np.inf)
    for k, mass in enumerate(cond_mass):
        if mass <= 0.0:
            continue
        cond_prior = rho[k] / mass
        finite = (cond_prior > 0.0) & np.isfinite(e[k])
        if not bool(np.any(finite)):
            continue
        m = float(np.min(e[k, finite]))
        z = float(np.sum(np.exp(-float(beta) * (e[k, finite] - m)) * cond_prior[finite]))
        out[k] = m - math.log(z) / float(beta)
    return out


def projected_minimizers(prior: ArrayLike, energy: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return joint minimizer coordinates and their condition/value projections."""
    rho = _as_float_array(prior, name="prior", ndim=2)
    e = _as_float_array(energy, name="energy", ndim=2)
    if rho.shape != e.shape:
        raise ValueError("prior and energy must have the same shape")
    supported = rho > 0.0
    finite = supported & np.isfinite(e)
    if not bool(np.any(finite)):
        raise ValueError("at least one supported joint candidate must have finite energy")
    m = float(np.min(e[finite]))
    coords = np.argwhere(finite & (e == m))
    return coords, np.unique(coords[:, 0]), np.unique(coords[:, 1])


def born_prior(amplitudes: Iterable[complex] | np.ndarray) -> np.ndarray:
    """Return the finite Born prior ``|c_i|^2 / sum_j |c_j|^2``."""
    amps = np.asarray(amplitudes, dtype=complex)
    if amps.ndim != 1 or amps.size == 0:
        raise ValueError("amplitudes must be a non-empty vector")
    weights = np.abs(amps) ** 2
    return normalize_weights(weights)


def refined_branch_prior(counts: Iterable[int]) -> np.ndarray:
    """Prior induced by grouping equal-amplitude microbranches."""
    ns = np.asarray(list(counts), dtype=int)
    if ns.ndim != 1 or ns.size == 0:
        raise ValueError("counts must be a non-empty vector")
    if np.any(ns < 0) or int(ns.sum()) <= 0:
        raise ValueError("counts must be non-negative with positive total")
    return ns.astype(float) / float(ns.sum())


def _checked_cost(prior: ArrayLike, cost: ArrayLike, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    p = normalize_weights(prior)
    c = _as_float_array(cost, name=name, ndim=1)
    if p.shape != c.shape:
        raise ValueError(f"prior and {name} must have the same shape")
    if np.any(np.isnan(c)) or np.any(c < 0.0):
        raise ValueError(f"{name} must be non-negative and not NaN")
    return p, c


def survival_fraction(prior: ArrayLike, energy: ArrayLike, threshold: float) -> float:
    """Finite hard-constraint survival fraction ``mu(E < threshold)``.

    Finite analogue of the threshold reading in ``docs/9_등호이전/05k``.
    """
    p, e = _checked_cost(prior, energy, name="energy")
    return float(p[e < float(threshold)].sum())


def conditioned_prior(prior: ArrayLike, energy: ArrayLike, threshold: float) -> np.ndarray:
    """Condition a finite prior on the hard-constraint set ``{E < threshold}``.

    Raises when the constraint set carries zero prior mass, the finite shadow
    of the continuum obstruction in ``05k`` section 2.
    """
    p, e = _checked_cost(prior, energy, name="energy")
    mask = (p > 0.0) & (e < float(threshold))
    mass = float(p[mask].sum())
    if mass <= 0.0:
        raise ValueError("hard constraint set has zero prior mass")
    return np.where(mask, p, 0.0) / mass


def tilt_survival(prior: ArrayLike, phi: ArrayLike) -> float:
    """Smooth-tilt survival probability ``<e^{-Phi}>`` on a finite space."""
    p, f = _checked_cost(prior, phi, name="phi")
    decay = np.exp(-f, where=np.isfinite(f), out=np.zeros_like(f))
    return float(np.sum(p * decay))


def layer_cake_survival(prior: ArrayLike, phi: ArrayLike) -> float:
    """Survival via the layer-cake integral of threshold fractions.

    Computes ``int_0^inf e^{-t} mu(Phi <= t) dt`` exactly as a step-function
    integral.  By ``05k`` theorem 5.1 this equals :func:`tilt_survival`.
    """
    p, f = _checked_cost(prior, phi, name="phi")
    order = np.argsort(f)
    cumulative = np.cumsum(p[order])
    decay = np.exp(-f[order], where=np.isfinite(f[order]), out=np.zeros_like(f))
    next_decay = np.append(decay[1:], 0.0)
    return float(np.sum(cumulative * (decay - next_decay)))


def mean_field_bounds(prior: ArrayLike, phi: ArrayLike) -> tuple[float, float]:
    """Jensen lower and second-order upper bound for ``<e^{-Phi}>``.

    Implements ``05k`` theorem 5.3: the mean-field value ``e^{-<Phi>}`` is a
    lower bound, and for bounded ``Phi`` the error is controlled by the
    variance.  Requires finite ``phi`` on the support.
    """
    p, f = _checked_cost(prior, phi, name="phi")
    support = p > 0.0
    if not np.all(np.isfinite(f[support])):
        raise ValueError("phi must be finite on the support for mean-field bounds")
    mean = float(np.sum(p * np.where(support, f, 0.0)))
    var = float(np.sum(p * np.where(support, (f - mean) ** 2, 0.0)))
    bound = float(np.max(f[support]))
    lower = math.exp(-mean)
    upper = math.exp(-mean + 0.5 * math.exp(bound) * var)
    return lower, upper

