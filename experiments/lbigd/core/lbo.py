"""
Laplace-Beltrami Operator (LBO) for Inverse Game Design

Implements the core algorithms from docs/08_lb_igd/:
- Theorem 4.2: Bellman <-> LBO equivalence: (rho - nu * Delta_g) V = r
- Theorem 4.3: Torsion/drift extension: (rho - nu * Delta_g - b.grad) V = r
- Graph discretization: (rho*I + nu*L + kappa*A) v = r
- STDP-based synaptic plasticity (from 07_synapse.md)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


def laplacian_mul(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Graph Laplacian multiply: (D - W) x

    - w: (n, n) non-negative weights (prefer symmetric)
    - x: (n,) or (n, k)
    """
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be square (n, n)")

    if x.ndim == 1:
        if x.shape[0] != w.shape[0]:
            raise ValueError("x must have shape (n,)")
        deg = w.sum(axis=1)
        return deg * x - (w @ x)

    if x.ndim == 2:
        if x.shape[0] != w.shape[0]:
            raise ValueError("x must have shape (n, k)")
        deg = w.sum(axis=1)
        return deg[:, None] * x - (w @ x)

    raise ValueError("x must be 1D or 2D")


def dirichlet_energy(w: np.ndarray, u: np.ndarray) -> float:
    """
    Dirichlet energy: u^T (D - W) u  (>= 0 if w is symmetric and non-negative)
    """
    lu = laplacian_mul(w, u)
    return float(np.dot(u, lu))


def decompose_symmetric_antisymmetric(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose weight matrix into symmetric (S) and antisymmetric (A) parts.

    W = S + A, where S = (W + W^T) / 2, A = (W - W^T) / 2

    - S: symmetric component (diffusion / Laplacian)
    - A: antisymmetric component (drift / torsion)

    This corresponds to the decomposition in Theorem 4.3 (07_synapse.md, Section 10.6).
    """
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be square (n, n)")
    w = w.astype(np.float64, copy=False)
    s = 0.5 * (w + w.T)
    a = 0.5 * (w - w.T)
    return s.astype(np.float32), a.astype(np.float32)


def build_laplacian_matrix(w: np.ndarray) -> np.ndarray:
    """
    Build the graph Laplacian matrix L = D - W.

    - w: (n, n) non-negative symmetric weights
    - Returns: (n, n) Laplacian matrix
    """
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be square (n, n)")
    w = w.astype(np.float64, copy=False)
    deg = w.sum(axis=1)
    return (np.diag(deg) - w).astype(np.float32)


def build_resolvent_operator(
    w: np.ndarray,
    rho: float = 1.0,
    nu: float = 1.0,
    kappa: float = 0.0,
) -> np.ndarray:
    """
    Build the resolvent operator matrix: (rho*I + nu*L + kappa*A)

    This is the discrete version of (rho - nu*Delta_g - b.grad) from Theorem 4.3.

    Args:
        w: Weight matrix (n, n). Can be asymmetric.
        rho: Discount rate (> 0). Corresponds to 1/tau_m in neural membrane.
        nu: Diffusion coefficient (>= 0). Corresponds to lambda^2/tau_m.
        kappa: Drift/torsion coefficient (>= 0). Controls antisymmetric contribution.

    Returns:
        (n, n) resolvent operator matrix
    """
    if rho <= 0:
        raise ValueError("rho must be > 0")
    if nu < 0:
        raise ValueError("nu must be >= 0")
    if kappa < 0:
        raise ValueError("kappa must be >= 0")

    n = w.shape[0]
    s, a = decompose_symmetric_antisymmetric(w)
    lap = build_laplacian_matrix(s)
    eye = np.eye(n, dtype=np.float64)
    op = rho * eye + nu * lap.astype(np.float64) + kappa * a.astype(np.float64)
    return op.astype(np.float32)


def solve_resolvent(
    w: np.ndarray,
    r: np.ndarray,
    rho: float = 1.0,
    nu: float = 1.0,
    kappa: float = 0.0,
) -> np.ndarray:
    """
    Solve the Bellman-LBO equation: (rho*I + nu*L + kappa*A) v = r

    This computes v = (rho*I + nu*L + kappa*A)^{-1} r, which is the
    smoothed value function from Theorem 4.2/4.3.

    Args:
        w: Weight matrix (n, n).
        r: Reward/source vector (n,) or (n, k).
        rho: Discount rate.
        nu: Diffusion coefficient.
        kappa: Drift/torsion coefficient.

    Returns:
        Solution v with same shape as r.
    """
    op = build_resolvent_operator(w, rho=rho, nu=nu, kappa=kappa)
    r = r.astype(np.float64, copy=False)
    if r.ndim == 1:
        v = np.linalg.solve(op.astype(np.float64), r)
        return v.astype(np.float32)
    if r.ndim == 2:
        v = np.linalg.solve(op.astype(np.float64), r)
        return v.astype(np.float32)
    raise ValueError("r must be 1D or 2D")


def solve_resolvent_cg(
    w: np.ndarray,
    r: np.ndarray,
    rho: float = 1.0,
    nu: float = 1.0,
    kappa: float = 0.0,
    tol: float = 1e-6,
    maxiter: int = 500,
) -> Tuple[np.ndarray, int]:
    """
    Solve the Bellman-LBO equation using Conjugate Gradient.

    For large sparse systems, CG is more efficient than direct solve.
    Requires the operator to be symmetric positive definite (kappa=0 or small).

    Args:
        w: Weight matrix (n, n).
        r: Reward/source vector (n,).
        rho, nu, kappa: Resolvent parameters.
        tol: Convergence tolerance.
        maxiter: Maximum iterations.

    Returns:
        (solution, iterations)
    """
    from scipy.sparse.linalg import cg

    op = build_resolvent_operator(w, rho=rho, nu=nu, kappa=kappa)
    r64 = r.astype(np.float64, copy=False)
    v, info = cg(op.astype(np.float64), r64, tol=float(tol), maxiter=int(maxiter))
    if info != 0:
        pass
    return v.astype(np.float32), int(info)


def smooth_winrate(
    w: np.ndarray,
    p: np.ndarray,
    rho: float = 0.5,
    nu: float = 1.0,
    kappa: float = 0.0,
) -> np.ndarray:
    """
    Smooth a win-rate surface using LBO resolvent.

    From docs/08_lb_igd/07_synapse.md Section 12.3:
        r(x) = P(x)  (win rate)
        V(x) = (rho*I + nu*L)^{-1} P = "noise-reduced win rate landscape"

    This makes ES updates more stable by smoothing noisy win-rate estimates.

    Args:
        w: Design-space adjacency/metric graph (n, n).
        p: Win-rate estimates at each design point (n,).
        rho, nu, kappa: Resolvent parameters.

    Returns:
        Smoothed win-rate estimates (n,).
    """
    return solve_resolvent(w, p, rho=rho, nu=nu, kappa=kappa)


def compute_smoothed_gradient(
    w: np.ndarray,
    p: np.ndarray,
    rho: float = 0.5,
    nu: float = 1.0,
    kappa: float = 0.0,
) -> np.ndarray:
    """
    Compute the gradient of smoothed win-rate landscape.

    From docs/08_lb_igd/07_synapse.md Section 12.3:
        Use grad_g V instead of noisy grad P for faster convergence.

    For discrete graph, gradient is approximated as weighted differences.

    Args:
        w: Design-space adjacency graph (n, n).
        p: Win-rate estimates (n,).
        rho, nu, kappa: Resolvent parameters.

    Returns:
        Gradient-like vector (n,) indicating direction of improvement.
    """
    v = smooth_winrate(w, p, rho=rho, nu=nu, kappa=kappa)
    grad = laplacian_mul(w, v)
    return -grad.astype(np.float32)


def solve_fourier_mode(
    w: np.ndarray,
    r_k: np.ndarray,
    omega_k: float,
    rho: float = 1.0,
    nu: float = 1.0,
    kappa: float = 0.0,
) -> np.ndarray:
    """
    Solve the Bellman-LBO equation for a single Fourier mode.

    From Theorem 4.3 extension (time-dependent reward):
        (rho + i*omega_k - L) V_k = r_k

    This handles periodic/oscillating inputs (e.g., seasonal user behavior).

    Args:
        w: Weight matrix (n, n).
        r_k: Fourier coefficient of reward at frequency omega_k (n,), complex.
        omega_k: Angular frequency of this mode.
        rho, nu, kappa: Resolvent parameters.

    Returns:
        V_k: Response at this frequency (n,), complex.
    """
    n = w.shape[0]
    s, a = decompose_symmetric_antisymmetric(w)
    lap = build_laplacian_matrix(s)
    eye = np.eye(n, dtype=np.complex128)
    op = (rho + 1j * omega_k) * eye + nu * lap.astype(np.complex128) + kappa * a.astype(np.complex128)
    r_k = np.asarray(r_k, dtype=np.complex128)
    v_k = np.linalg.solve(op, r_k)
    return v_k


def solve_fourier_parallel(
    w: np.ndarray,
    r_modes: np.ndarray,
    omegas: np.ndarray,
    rho: float = 1.0,
    nu: float = 1.0,
    kappa: float = 0.0,
) -> np.ndarray:
    """
    Solve the Bellman-LBO equation for multiple Fourier modes in parallel.

    From docs/08_lb_igd/07_synapse.md Section 10.4:
        Brain oscillations (alpha/beta/gamma) as Fourier modes.
        Linear system allows mode-by-mode parallelization.

    Args:
        w: Weight matrix (n, n).
        r_modes: Fourier coefficients (n_modes, n), complex.
        omegas: Angular frequencies (n_modes,).
        rho, nu, kappa: Resolvent parameters.

    Returns:
        V_modes: Responses at each frequency (n_modes, n), complex.
    """
    n_modes = r_modes.shape[0]
    n = w.shape[1]
    v_modes = np.zeros((n_modes, n), dtype=np.complex128)
    for k in range(n_modes):
        v_modes[k] = solve_fourier_mode(w, r_modes[k], float(omegas[k]), rho=rho, nu=nu, kappa=kappa)
    return v_modes


def reconstruct_from_fourier(
    v_modes: np.ndarray,
    omegas: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Reconstruct time-domain signal from Fourier modes.

    V(x, t) = sum_k V_k(x) * exp(i * omega_k * t)

    Args:
        v_modes: Fourier responses (n_modes, n), complex.
        omegas: Angular frequencies (n_modes,).
        t: Time point.

    Returns:
        Real-valued signal at time t (n,).
    """
    phases = np.exp(1j * omegas * t)
    signal = np.real(np.sum(v_modes * phases[:, None], axis=0))
    return signal.astype(np.float32)


@dataclass
class STDPState:
    """
    Spike-Timing Dependent Plasticity state.

    From docs/08_lb_igd/07_synapse.md Section 3.2:
        - pre_trace p_i[t]: Recent pre-synaptic activity
        - post_trace q_i[t]: Recent post-synaptic activity
        - eligibility e_ij[t]: Accumulated STDP signal (for 3-factor learning)

    Update rules:
        p_i[t+1] = lambda_+ * p_i[t] + s_i[t]
        q_i[t+1] = lambda_- * q_i[t] + s_i[t]
        Delta w_ij = eta * (A+ * p_i * s_j - A- * s_i * q_j)
    """
    n: int
    lambda_pos: float = 0.9
    lambda_neg: float = 0.9
    a_pos: float = 0.1
    a_neg: float = 0.1
    eta: float = 0.01
    lambda_e: float = 0.95
    w_max: float = 1.0
    w_min: float = 0.0
    pre_trace: np.ndarray = field(default=None, repr=False)
    post_trace: np.ndarray = field(default=None, repr=False)
    eligibility: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.pre_trace is None:
            self.pre_trace = np.zeros(self.n, dtype=np.float32)
        if self.post_trace is None:
            self.post_trace = np.zeros(self.n, dtype=np.float32)
        if self.eligibility is None:
            self.eligibility = np.zeros((self.n, self.n), dtype=np.float32)

    def reset(self) -> None:
        """Reset all traces to zero."""
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.eligibility.fill(0.0)


def stdp_update_traces(
    state: STDPState,
    spikes: np.ndarray,
) -> None:
    """
    Update STDP traces given current spike pattern.

    From docs/08_lb_igd/07_synapse.md Section 3.2:
        p_i[t+1] = lambda_+ * p_i[t] + s_i[t]
        q_i[t+1] = lambda_- * q_i[t] + s_i[t]

    Args:
        state: STDP state object.
        spikes: Binary spike indicators (n,), 0 or 1.
    """
    s = np.asarray(spikes, dtype=np.float32)
    state.pre_trace = state.lambda_pos * state.pre_trace + s
    state.post_trace = state.lambda_neg * state.post_trace + s


def stdp_compute_delta_w(
    state: STDPState,
    spikes: np.ndarray,
) -> np.ndarray:
    """
    Compute weight change from STDP rule (2-factor).

    From docs/08_lb_igd/07_synapse.md Section 3.2:
        Delta w_ij = eta * (A+ * p_i[t] * s_j[t] - A- * s_i[t] * q_j[t])

    Args:
        state: STDP state (with updated traces).
        spikes: Current spike pattern (n,).

    Returns:
        Weight change matrix (n, n).
    """
    s = np.asarray(spikes, dtype=np.float32)
    p = state.pre_trace
    q = state.post_trace
    ltp = state.a_pos * np.outer(p, s)
    ltd = state.a_neg * np.outer(s, q)
    delta_w = state.eta * (ltp - ltd)
    return delta_w.astype(np.float32)


def stdp_update_eligibility(
    state: STDPState,
    spikes: np.ndarray,
) -> None:
    """
    Update eligibility trace (for 3-factor learning).

    From docs/08_lb_igd/07_synapse.md Section 4.1:
        e_ij[t+1] = lambda_e * e_ij[t] + (A+ * p_i * s_j - A- * s_i * q_j)

    The eligibility trace accumulates STDP signals; actual weight change
    only occurs when dopamine signal delta[t] arrives.

    Args:
        state: STDP state object.
        spikes: Current spike pattern (n,).
    """
    s = np.asarray(spikes, dtype=np.float32)
    p = state.pre_trace
    q = state.post_trace
    ltp = state.a_pos * np.outer(p, s)
    ltd = state.a_neg * np.outer(s, q)
    state.eligibility = state.lambda_e * state.eligibility + (ltp - ltd)


def stdp_apply_dopamine(
    state: STDPState,
    w: np.ndarray,
    delta: float,
) -> np.ndarray:
    """
    Apply 3-factor learning: weight change gated by dopamine signal.

    From docs/08_lb_igd/07_synapse.md Section 4.1:
        Delta w_ij[t] = eta * delta[t] * e_ij[t]

    Args:
        state: STDP state with eligibility traces.
        w: Current weight matrix (n, n).
        delta: Dopamine signal (0 = no learning, 1 = full learning).

    Returns:
        Updated weight matrix (n, n).
    """
    delta_w = state.eta * float(delta) * state.eligibility
    w_new = w + delta_w
    w_new = np.clip(w_new, state.w_min, state.w_max)
    np.fill_diagonal(w_new, 0.0)
    return w_new.astype(np.float32)


def stdp_step(
    state: STDPState,
    w: np.ndarray,
    spikes: np.ndarray,
    delta: float = 1.0,
    use_3factor: bool = False,
) -> np.ndarray:
    """
    Complete STDP step: update traces and apply weight changes.

    Args:
        state: STDP state object.
        w: Current weight matrix (n, n).
        spikes: Current spike pattern (n,).
        delta: Dopamine signal (only used if use_3factor=True).
        use_3factor: If True, use eligibility-based 3-factor learning.

    Returns:
        Updated weight matrix (n, n).
    """
    spikes = np.asarray(spikes, dtype=np.float32)
    if use_3factor:
        stdp_update_eligibility(state, spikes)
        stdp_update_traces(state, spikes)
        return stdp_apply_dopamine(state, w, delta)
    else:
        delta_w = stdp_compute_delta_w(state, spikes)
        stdp_update_traces(state, spikes)
        w_new = w + delta_w
        w_new = np.clip(w_new, state.w_min, state.w_max)
        np.fill_diagonal(w_new, 0.0)
        return w_new.astype(np.float32)


def generate_spikes(
    u: np.ndarray,
    threshold: float = 0.5,
    stochastic: bool = False,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate spike pattern from continuous activity.

    From docs/08_lb_igd/07_synapse.md Section 2.2:
        - Threshold: s_i[t] = 1 if u_i[t] > theta else 0
        - Stochastic: s_i[t] ~ Bernoulli(sigmoid(u_i[t]))

    Args:
        u: Continuous activity (n,).
        threshold: Firing threshold.
        stochastic: If True, use probabilistic firing.
        rng: Random generator for stochastic mode.

    Returns:
        Binary spike pattern (n,).
    """
    if stochastic:
        if rng is None:
            rng = np.random.default_rng()
        prob = 1.0 / (1.0 + np.exp(-u))
        spikes = (rng.random(u.shape) < prob).astype(np.float32)
    else:
        spikes = (u > threshold).astype(np.float32)
    return spikes


def homeostatic_scaling(
    w: np.ndarray,
    target_degree: float = 1.0,
) -> np.ndarray:
    """
    Apply homeostatic synaptic scaling.

    From docs/08_lb_igd/07_synapse.md Section 5:
        Normalize rows/columns to preserve total input/output.
        Prevents runaway potentiation/depression.

    Args:
        w: Weight matrix (n, n).
        target_degree: Target sum per row.

    Returns:
        Scaled weight matrix (n, n).
    """
    w = w.astype(np.float64, copy=True)
    np.fill_diagonal(w, 0.0)
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    w = w * (target_degree / row_sums)
    return w.astype(np.float32)


def structural_plasticity(
    w: np.ndarray,
    w_on: float = 0.1,
    w_off: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply structural plasticity: connection formation/removal.

    From docs/08_lb_igd/07_synapse.md Section 5:
        - w_ij >= w_on: connection exists
        - w_ij <= w_off: connection removed

    Uses hysteresis (w_off < w_on) for stability.

    Args:
        w: Weight matrix (n, n).
        w_on: Threshold for connection formation.
        w_off: Threshold for connection removal.

    Returns:
        (pruned weights, connection mask)
    """
    if w_off >= w_on:
        raise ValueError("w_off must be < w_on for hysteresis")

    mask = (w >= w_off).astype(np.float32)
    mask[w < w_off] = 0.0
    w_pruned = w * mask
    np.fill_diagonal(w_pruned, 0.0)
    np.fill_diagonal(mask, 0.0)
    return w_pruned.astype(np.float32), mask.astype(np.float32)
