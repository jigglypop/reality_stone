"""Metric-aware CE ops: auto-dispatch to CUDA / Rust / PyTorch.

Phase 1 is inference-only. Public API keeps standard torch tensors while
moving the hot path (sparse relax loop) into native code when available.

This module is the canonical Python backend-dispatch layer. Higher-level
runtime policy should stay in Python modules such as `reality_stone.clarus.engine`
and `reality_stone.clarus.runtime`, while pure numerics route through here.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .quantum import ALPHA_B_DEFAULT, estimate_mu, iss_ball_radius

try:
    from .constants import PORTAL as DEFAULT_CB_W
except ImportError:
    from reality_stone.clarus.constants import PORTAL as DEFAULT_CB_W

_RUST = False
_CUDA = False
_cuda_mod = None

try:
    from ._rust import (
        nn_ce_pack_sparse as _rust_ce_pack_sparse,
        nn_ce_metric_basis_fwd as _rust_ce_metric_basis_fwd,
        nn_ce_codebook_pull as _rust_ce_codebook_pull,
        nn_ce_relax_fwd as _rust_ce_relax_fwd,
    )

    _RUST = True
except ImportError:
    _RUST = False

# No CUDA CE kernel module ships in this tree, so ``ce_backend("cuda")`` fails closed.
_CUDA = False
_cuda_mod = None


def has_rust() -> bool:
    return bool(_RUST)


def has_cuda() -> bool:
    return bool(_CUDA)


def ce_backend(device: torch.device, requested: str = "auto") -> str:
    requested = requested.lower()
    if requested == "auto":
        if device.type == "cuda" and _CUDA:
            return "cuda"
        if device.type == "cpu" and _RUST:
            return "rust"
        return "torch"
    if requested == "cuda":
        if device.type != "cuda":
            raise RuntimeError("CUDA CE backend requested for a non-CUDA tensor/device")
        if not _CUDA:
            raise RuntimeError("CUDA CE backend requested but CUDA CE kernels are unavailable")
        return "cuda"
    if requested == "rust":
        if device.type != "cpu":
            raise RuntimeError("Rust CE backend requested for a non-CPU tensor/device")
        if not _RUST:
            raise RuntimeError("Rust CE backend requested but reality_stone.clarus._rust is unavailable")
        return "rust"
    if requested == "torch":
        return "torch"
    raise ValueError(f"unknown CE backend: {requested}")


def checked_sparse_csr_tensor(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    *,
    size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a CSR tensor with structural invariant checks enabled.

    Recent PyTorch releases require the sparse-invariant context manager to
    opt in without emitting the unsafe-default warning.  Keep the keyword
    fallback for older supported releases that do not expose that context.
    """
    checker = getattr(torch.sparse, "check_sparse_tensor_invariants", None)
    kwargs = {
        "size": size,
        "device": device,
        "dtype": dtype,
    }
    if checker is None:  # pragma: no cover - compatibility with older torch
        return torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            check_invariants=True,
            **kwargs,
        )
    with checker(enable=True):
        return torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            **kwargs,
        )


def _as_cpu_numpy_flat(x: torch.Tensor):
    return x.detach().contiguous().view(-1).cpu().numpy()


def _hist_from_tensors(
    energy: torch.Tensor,
    delta: torch.Tensor,
    e_hop: torch.Tensor,
    e_bias: torch.Tensor,
    e_portal: torch.Tensor,
    e_cb: torch.Tensor,
    bypass_hist: torch.Tensor,
) -> Dict[str, list[float]]:
    return {
        "E": energy.detach().cpu().tolist(),
        "delta": delta.detach().cpu().tolist(),
        "E_hop": e_hop.detach().cpu().tolist(),
        "E_bias": e_bias.detach().cpu().tolist(),
        "E_portal": e_portal.detach().cpu().tolist(),
        "E_cb": e_cb.detach().cpu().tolist(),
        "bypass_C": bypass_hist.detach().cpu().tolist(),
    }


def pack_sparse(
    w: torch.Tensor,
    zero_tol: float = 0.0,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = w.shape[0]
    chosen = ce_backend(w.device if w.is_cuda else torch.device("cpu"), backend)

    if chosen == "rust" and not w.is_cuda:
        values_np, col_np, row_np = _rust_ce_pack_sparse(_as_cpu_numpy_flat(w), dim, float(zero_tol))
        return (
            torch.from_numpy(values_np),
            torch.from_numpy(col_np),
            torch.from_numpy(row_np),
        )

    mask = w.abs() > zero_tol
    rows, cols = mask.nonzero(as_tuple=True)
    values = w[rows, cols].to(dtype=torch.float32)
    col_idx = cols.to(dtype=torch.int32)
    row_counts = torch.bincount(rows, minlength=dim).to(dtype=torch.int32)
    row_ptr = torch.zeros(dim + 1, dtype=torch.int32, device=w.device)
    row_ptr[1:] = torch.cumsum(row_counts, dim=0)
    return values, col_idx, row_ptr


def build_metric_basis(
    codebook: torch.Tensor,
    m_ref: torch.Tensor,
    rank: int,
    w_eigvecs: Optional[torch.Tensor] = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Build orthonormal metric basis from codebook directions + optional Hessian eigenvectors.

    When w_eigvecs is provided (top-k eigenvectors of W), they are placed first
    in the basis to capture the principal curvature directions of E_hop.
    """
    n_code, dim = codebook.shape
    if rank <= 0:
        return codebook.new_empty((0, dim))

    chosen = ce_backend(codebook.device if codebook.is_cuda else torch.device("cpu"), backend)

    if chosen == "rust" and not codebook.is_cuda and w_eigvecs is None:
        basis_np = _rust_ce_metric_basis_fwd(
            _as_cpu_numpy_flat(codebook),
            _as_cpu_numpy_flat(m_ref),
            n_code,
            dim,
            rank,
        )
        basis_t = torch.from_numpy(basis_np)
        rows = 0 if basis_t.numel() == 0 else basis_t.numel() // dim
        return basis_t.reshape(rows, dim)

    basis_rows: list[torch.Tensor] = []

    if w_eigvecs is not None and w_eigvecs.numel() > 0:
        for j in range(w_eigvecs.shape[0]):
            v = w_eigvecs[j].clone()
            for b in basis_rows:
                v = v - torch.dot(v, b) * b
            n = v.norm()
            if n > 1e-6:
                basis_rows.append(v / n)
            if len(basis_rows) >= rank:
                break

    if n_code > 0 and len(basis_rows) < rank:
        remain = rank - len(basis_rows)
        logits = codebook @ m_ref
        probs = F.softmax(logits, dim=0)
        mean = (probs.unsqueeze(1) * codebook).sum(dim=0)
        idx = probs.topk(min(remain * 4, n_code)).indices
        for i in idx.tolist():
            v = (codebook[i] - mean) * probs[i].sqrt()
            for b in basis_rows:
                v = v - torch.dot(v, b) * b
            n = v.norm()
            if n > 1e-6:
                basis_rows.append(v / n)
            if len(basis_rows) >= rank:
                break

    if not basis_rows:
        return codebook.new_empty((0, dim))
    return torch.stack(basis_rows, dim=0)


def codebook_pull(
    m: torch.Tensor,
    codebook: torch.Tensor,
    beta: float,
    cb_w: float,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if codebook.numel() == 0:
        zero = torch.zeros_like(m)
        return zero, m.new_tensor(0.0)

    chosen = ce_backend(m.device, backend)
    n_code, dim = codebook.shape

    if chosen == "cuda" and m.is_cuda:
        grad, energy = _cuda_mod.ce_codebook_pull_fwd(
            m.contiguous(),
            codebook.contiguous(),
            float(beta),
            float(cb_w),
        )
        return grad, energy

    if chosen == "rust" and not m.is_cuda:
        grad_np, energy = _rust_ce_codebook_pull(
            _as_cpu_numpy_flat(m),
            _as_cpu_numpy_flat(codebook),
            n_code,
            dim,
            float(beta),
            float(cb_w),
        )
        return torch.from_numpy(grad_np), torch.tensor(energy, dtype=m.dtype)

    logits = beta * (codebook @ m)
    w = F.softmax(logits, dim=0)
    grad = -cb_w * (w @ codebook)
    energy = -(cb_w / max(beta, 1e-6)) * torch.logsumexp(logits, dim=0)
    return grad, energy


def _spmv_torch(
    values: torch.Tensor,
    col_idx: torch.Tensor,
    row_ptr: torch.Tensor,
    x: torch.Tensor,
    *,
    sparse_mat: torch.Tensor | None = None,
    dense_w: torch.Tensor | None = None,
) -> torch.Tensor:
    if dense_w is not None:
        return dense_w @ x
    dim = x.numel()
    sparse = sparse_mat
    if sparse is None:
        sparse = checked_sparse_csr_tensor(
            row_ptr.to(torch.int64),
            col_idx.to(torch.int64),
            values,
            size=(dim, dim),
            device=x.device,
            dtype=x.dtype,
        )
    return torch.sparse.mm(sparse, x.unsqueeze(1)).squeeze(1)


def _natural_direction_torch(
    grad: torch.Tensor,
    phi: torch.Tensor,
    recent_var: torch.Tensor,
    metric_basis: torch.Tensor,
    lambda0: float,
    lambda_phi: float,
    lambda_var: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    diag = lambda0 + lambda_phi * phi.square() + lambda_var * recent_var
    diag = diag.clamp_min(1e-4)
    inv_diag = diag.reciprocal()
    inv_diag_grad = grad * inv_diag

    if metric_basis.numel() == 0:
        return inv_diag_grad, diag

    basis = metric_basis
    weighted_basis = basis * inv_diag.unsqueeze(0)
    small = torch.eye(
        basis.shape[0],
        device=grad.device,
        dtype=grad.dtype,
    ) + basis @ weighted_basis.transpose(0, 1)
    rhs = basis @ inv_diag_grad
    tmp = torch.linalg.solve(small, rhs.unsqueeze(-1)).squeeze(-1)
    correction = basis.transpose(0, 1) @ tmp
    return inv_diag_grad - correction * inv_diag, diag


def _fdt_noise_torch(
    z: torch.Tensor,
    phi: torch.Tensor,
    recent_var: torch.Tensor,
    metric_basis: torch.Tensor,
    lambda0: float,
    lambda_phi: float,
    lambda_var: float,
) -> torch.Tensor:
    """Compute G^{-1/2} z for FDT-consistent Langevin noise.

    G = D + U U^T  (Woodbury SPD metric)
    G^{-1/2} = D^{-1/2} (I + Q Q^T)^{-1/2} where Q = D^{-1/2} U
    (I + Q Q^T)^{-1/2} computed via SVD of Q.
    """
    diag = lambda0 + lambda_phi * phi.square() + lambda_var * recent_var
    diag = diag.clamp_min(1e-4)
    inv_sqrt_diag = diag.rsqrt()

    if metric_basis.numel() == 0:
        return z * inv_sqrt_diag

    Q = metric_basis * inv_sqrt_diag.unsqueeze(0)
    if not torch.isfinite(Q).all():
        Q = torch.where(torch.isfinite(Q), Q, torch.zeros_like(Q))
    _, s_q, Vh_q = torch.linalg.svd(Q, full_matrices=False)

    factors = 1.0 - 1.0 / torch.sqrt(1.0 + s_q.square())
    proj = Vh_q @ z
    corrected = z - (Vh_q.T @ (factors * proj))
    return inv_sqrt_diag * corrected


def _energy_parts_torch(
    m: torch.Tensor,
    w_m: torch.Tensor,
    b: torch.Tensor,
    phi: torch.Tensor,
    codebook: torch.Tensor,
    portal: float,
    beta: float,
    cb_w: float,
    bypass_c: float = 0.0,
    bypass_coeff: float = 0.0,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    e_hop = -0.5 * torch.dot(m, w_m)
    e_bias = -torch.dot(m, b)
    e_portal = -portal * torch.dot(m, phi)
    e_bypass = -bypass_coeff * bypass_c * torch.dot(m, phi)
    if codebook.numel() == 0:
        e_cb = m.new_tensor(0.0)
    else:
        logits = beta * (codebook @ m)
        e_cb = -(cb_w / max(beta, 1e-6)) * torch.logsumexp(logits, dim=0)
    total = e_hop + e_bias + e_portal + e_cb + e_bypass
    return total, (e_hop, e_bias, e_portal, e_cb)


@torch.no_grad()
def _relax_packed_torch(
    values: torch.Tensor,
    col_idx: torch.Tensor,
    row_ptr: torch.Tensor,
    b: torch.Tensor,
    phi: torch.Tensor,
    m0: torch.Tensor,
    codebook: torch.Tensor,
    metric_basis: torch.Tensor,
    portal: float,
    bypass: float,
    t_wake: float,
    beta: float,
    cb_w: float,
    lambda0: float,
    lambda_phi: float,
    lambda_var: float,
    tau: float,
    dt: float,
    max_steps: int,
    tol: float,
    anneal_ratio: float,
    noise_scale: float,
    seed: int,
    dense_w: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, list[float]], int]:
    scale = float(m0.norm().item() or 1.0)
    m = m0 / scale
    b_n = b / scale
    phi_n = F.normalize(phi, dim=0)
    codebook_n = codebook / scale if codebook.numel() else codebook
    metric_basis_n = metric_basis

    m1 = m.clone()
    m2 = m.clone()

    tau = max(float(tau), 1e-6)
    dt_eff = min(float(dt), 0.9 * tau)
    anneal_end = max(1, int(round(anneal_ratio * max_steps)))
    t_eff = float(t_wake) / max(1, m.numel())

    sparse_mat = None
    if dense_w is None:
        sparse_mat = checked_sparse_csr_tensor(
            row_ptr.to(torch.int64),
            col_idx.to(torch.int64),
            values,
            size=(m.numel(), m.numel()),
            device=m.device,
            dtype=m.dtype,
        )

    w_m_probe = _spmv_torch(values, col_idx, row_ptr, m, sparse_mat=sparse_mat, dense_w=dense_w)
    spectral_est = w_m_probe.norm().item() / max(m.norm().item(), 1e-8)
    cfl_lambda0 = 2.0 * spectral_est * dt_eff / tau
    lambda0 = max(lambda0, cfl_lambda0)

    gen = None
    if noise_scale > 0.0:
        gen = torch.Generator(device=m.device)
        gen.manual_seed(int(seed))

    hist_e: list[float] = []
    hist_delta: list[float] = []
    hist_e_hop: list[float] = []
    hist_e_bias: list[float] = []
    hist_e_portal: list[float] = []
    hist_e_cb: list[float] = []
    hist_bypass: list[float] = []

    best_m = m.clone()
    best_e = float("inf")
    tail_states: deque[torch.Tensor] = deque(maxlen=min(16, max_steps))

    for k in range(max_steps):
        c_k = torch.norm(m - 2 * m1 + m2).item()
        w_m = _spmv_torch(values, col_idx, row_ptr, m, sparse_mat=sparse_mat, dense_w=dense_w)
        grad = w_m + b_n + float(portal) * phi_n + (c_k * float(bypass)) * phi_n

        if codebook_n.numel():
            cb_grad, _ = codebook_pull(m, codebook_n, beta=beta, cb_w=cb_w, backend="torch")
            grad = grad + cb_grad

        recent_var = 0.5 * ((m - m1).square() + (m1 - m2).square())
        nat_grad, _metric_diag = _natural_direction_torch(
            grad,
            phi_n,
            recent_var,
            metric_basis_n,
            lambda0,
            lambda_phi,
            lambda_var,
        )

        t_k = t_eff * max(0.0, 1.0 - k / anneal_end)
        noise_std = math.sqrt(max(0.0, 2.0 * t_k * dt_eff / tau)) * max(0.0, noise_scale)
        if noise_std > 0.0:
            z_raw = torch.randn(m.shape, dtype=m.dtype, device=m.device, generator=gen)
            noise = noise_std * _fdt_noise_torch(
                z_raw, phi_n, recent_var, metric_basis_n,
                lambda0, lambda_phi, lambda_var,
            )
        else:
            noise = torch.zeros_like(m)

        m2 = m1.clone()
        m1 = m.clone()
        dm = (dt_eff / tau) * nat_grad + noise
        if not torch.isfinite(dm).all():
            dm = torch.where(torch.isfinite(dm), dm, torch.zeros_like(dm))
        m = m + dm
        tail_states.append(m.detach().clone())

        w_m_new = _spmv_torch(values, col_idx, row_ptr, m, sparse_mat=sparse_mat, dense_w=dense_w)
        e_total, (e_hop, e_bias, e_portal, e_cb) = _energy_parts_torch(
            m, w_m_new, b_n, phi_n, codebook_n,
            portal, beta, cb_w,
            bypass_c=c_k, bypass_coeff=bypass,
        )
        e_item = float(e_total.item())
        delta = float(dm.norm().item())

        hist_e.append(e_item)
        hist_delta.append(delta)
        hist_e_hop.append(float(e_hop.item()))
        hist_e_bias.append(float(e_bias.item()))
        hist_e_portal.append(float(e_portal.item()))
        hist_e_cb.append(float(e_cb.item()))
        hist_bypass.append(float(c_k))

        if e_item < best_e:
            best_e = e_item
            best_m = m.clone()

        if k > 30 and delta < tol:
            break

    best_m = best_m * scale
    if tail_states:
        tail = torch.stack(list(tail_states), dim=0) * scale
        phi_var = (tail - best_m.unsqueeze(0)).square().mean(dim=0)
        hist_phi_var = phi_var.detach().cpu().tolist()
    else:
        hist_phi_var = []

    iss_report = _iss_from_tail(
        tail_states=tail_states,
        scale=scale,
        best_m=best_m,
        c_k_history=hist_bypass,
        delta_history=hist_delta,
        phi=phi,
        dt=dt_eff,
        tau=tau,
    )

    hist = {
        "E": hist_e,
        "delta": hist_delta,
        "E_hop": hist_e_hop,
        "E_bias": hist_e_bias,
        "E_portal": hist_e_portal,
        "E_cb": hist_e_cb,
        "bypass_C": hist_bypass,
        "phi_var": hist_phi_var,
        "iss": iss_report,
    }
    return best_m, hist, len(hist_e)


def _iss_from_tail(
    *,
    tail_states: deque,
    scale: float,
    best_m: torch.Tensor,
    c_k_history: list[float],
    delta_history: list[float],
    phi: torch.Tensor,
    dt: float,
    tau: float,
) -> Dict[str, float]:
    """Compute gate F2 ISS report from a relaxation trajectory (12_Equation appendix A.1).

    mu is estimated from the global ||dm_k|| contraction curve (full trajectory),
    not from `tail_states`, since the tail is post-convergence noise plateau.
    """
    if not delta_history:
        return {
            "samples": 0,
            "c_k_max": 0.0,
            "phi_inf_norm": 0.0,
            "mu": 0.0,
            "iss_ball_radius": float("inf"),
        }
    c_k_max = float(max(c_k_history) if c_k_history else 0.0)
    phi_inf_norm = float(phi.detach().abs().max().item())
    dt_over_tau = float(dt) / float(tau) if tau > 0.0 else 0.0
    mu = (
        estimate_mu(delta_history, dt_over_tau=dt_over_tau, skip=1)
        if dt_over_tau > 0.0
        else 0.0
    )
    radius = iss_ball_radius(
        c_k_max=c_k_max,
        phi_inf_norm=phi_inf_norm,
        mu=mu,
        alpha_b=ALPHA_B_DEFAULT,
    )
    return {
        "samples": len(delta_history),
        "c_k_max": c_k_max,
        "phi_inf_norm": phi_inf_norm,
        "mu": mu,
        "iss_ball_radius": radius,
    }


@torch.no_grad()
def relax_packed(
    values: torch.Tensor,
    col_idx: torch.Tensor,
    row_ptr: torch.Tensor,
    b: torch.Tensor,
    phi: torch.Tensor,
    m0: torch.Tensor,
    codebook: Optional[torch.Tensor] = None,
    metric_basis: Optional[torch.Tensor] = None,
    *,
    portal: float,
    bypass: float,
    t_wake: float,
    beta: float = 1.0,
    cb_w: float = DEFAULT_CB_W,
    lambda0: float = 1.0,
    lambda_phi: float = 0.5,
    lambda_var: float = 0.25,
    tau: float = 1.0,
    dt: float = 0.01,
    max_steps: int = 500,
    tol: float = 1e-4,
    anneal_ratio: float = 0.6,
    noise_scale: float = 1.0,
    metric_rank: int = 8,
    backend: str = "auto",
    seed: int = 0,
    dense_w: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, list[float]], int]:
    codebook = codebook if codebook is not None else m0.new_empty((0, m0.numel()))
    metric_basis = metric_basis if metric_basis is not None else build_metric_basis(
        codebook, m0, metric_rank, backend=backend
    )
    chosen = ce_backend(m0.device, backend)
    dim = m0.numel()
    n_code = int(codebook.shape[0]) if codebook.ndim == 2 else 0
    rank = int(metric_basis.shape[0]) if metric_basis.ndim == 2 else 0

    if chosen == "cuda" and m0.is_cuda:
        out = _cuda_mod.ce_relax_fwd(
            values.contiguous(),
            col_idx.contiguous(),
            row_ptr.contiguous(),
            b.contiguous(),
            phi.contiguous(),
            m0.contiguous(),
            codebook.contiguous(),
            metric_basis.contiguous(),
            float(portal),
            float(bypass),
            float(t_wake),
            float(beta),
            float(cb_w),
            float(lambda0),
            float(lambda_phi),
            float(lambda_var),
            float(tau),
            float(dt),
            int(max_steps),
            float(tol),
            float(anneal_ratio),
            float(noise_scale),
            int(seed),
        )
        best_m, energy, delta, e_hop, e_bias, e_portal, e_cb, bypass_hist = out
        return best_m, _hist_from_tensors(energy, delta, e_hop, e_bias, e_portal, e_cb, bypass_hist), int(energy.numel())

    if chosen == "rust" and not m0.is_cuda:
        out = _rust_ce_relax_fwd(
            _as_cpu_numpy_flat(values),
            _as_cpu_numpy_flat(col_idx.to(torch.int32)),
            _as_cpu_numpy_flat(row_ptr.to(torch.int32)),
            _as_cpu_numpy_flat(b),
            _as_cpu_numpy_flat(phi),
            _as_cpu_numpy_flat(m0),
            _as_cpu_numpy_flat(codebook),
            _as_cpu_numpy_flat(metric_basis),
            dim,
            n_code,
            rank,
            float(portal),
            float(bypass),
            float(t_wake),
            float(beta),
            float(cb_w),
            float(lambda0),
            float(lambda_phi),
            float(lambda_var),
            float(tau),
            float(dt),
            int(max_steps),
            float(tol),
            float(anneal_ratio),
            float(noise_scale),
            int(seed),
        )
        best_m_np, energy_np, delta_np, e_hop_np, e_bias_np, e_portal_np, e_cb_np, bypass_np, steps = out
        best_m = torch.from_numpy(best_m_np)
        hist = {
            "E": energy_np.tolist(),
            "delta": delta_np.tolist(),
            "E_hop": e_hop_np.tolist(),
            "E_bias": e_bias_np.tolist(),
            "E_portal": e_portal_np.tolist(),
            "E_cb": e_cb_np.tolist(),
            "bypass_C": bypass_np.tolist(),
        }
        return best_m, hist, int(steps)

    return _relax_packed_torch(
        values,
        col_idx,
        row_ptr,
        b,
        phi,
        m0,
        codebook,
        metric_basis,
        portal,
        bypass,
        t_wake,
        beta,
        cb_w,
        lambda0,
        lambda_phi,
        lambda_var,
        tau,
        dt,
        max_steps,
        tol,
        anneal_ratio,
        noise_scale,
        seed,
        dense_w=dense_w,
    )


def relax(
    w: torch.Tensor,
    b: torch.Tensor,
    phi: torch.Tensor,
    m0: torch.Tensor,
    codebook: Optional[torch.Tensor] = None,
    metric_basis: Optional[torch.Tensor] = None,
    *,
    portal: float,
    bypass: float,
    t_wake: float,
    zero_tol: float = 0.0,
    backend: str = "auto",
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, list[float]], int]:
    values, col_idx, row_ptr = pack_sparse(w, zero_tol=zero_tol, backend=backend)
    dense_w = None
    if backend != "rust" and values.numel() == w.numel():
        dense_w = w
    return relax_packed(
        values,
        col_idx,
        row_ptr,
        b,
        phi,
        m0,
        codebook,
        metric_basis,
        portal=portal,
        bypass=bypass,
        t_wake=t_wake,
        backend=backend,
        dense_w=dense_w,
        **kwargs,
    )


def pq_build_codebook(
    emb: torch.Tensor,
    *,
    subdim: int = 64,
    bits: int = 8,
    iters: int = 16,
    batch_size: int = 4096,
    sample_size: int = 16384,
    seed: int = 0,
) -> Dict[str, torch.Tensor | int]:
    emb_cpu = emb.detach().float().cpu().contiguous()
    n_token, dim = emb_cpu.shape
    if subdim <= 0 or dim % subdim != 0:
        raise ValueError(f"subdim must divide dim exactly: dim={dim}, subdim={subdim}")
    if bits <= 0 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    n_sub = dim // subdim
    n_centroid = 1 << bits
    if n_centroid > n_token:
        raise ValueError("number of PQ centroids cannot exceed number of tokens")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    centroids_out: list[torch.Tensor] = []
    codes = torch.empty((n_token, n_sub), dtype=torch.uint8)

    for sub_idx in range(n_sub):
        start = sub_idx * subdim
        stop = start + subdim
        sub = emb_cpu[:, start:stop]

        pool_n = min(sample_size, n_token)
        pool_idx = torch.randperm(n_token, generator=gen)[:pool_n]
        pool = sub.index_select(0, pool_idx)
        init_idx = torch.randperm(pool.shape[0], generator=gen)[:n_centroid]
        centers = pool.index_select(0, init_idx).clone()

        for _ in range(max(1, iters)):
            cur_batch = min(batch_size, n_token)
            batch_idx = torch.randperm(n_token, generator=gen)[:cur_batch]
            batch = sub.index_select(0, batch_idx)
            dist = torch.cdist(batch, centers)
            assign = dist.argmin(dim=1)

            new_centers = centers.clone()
            for cid in range(n_centroid):
                mask = assign == cid
                if mask.any():
                    new_centers[cid] = batch[mask].mean(dim=0)
                else:
                    refill = torch.randint(pool.shape[0], (1,), generator=gen).item()
                    new_centers[cid] = pool[refill]
            centers = new_centers

        all_assign: list[torch.Tensor] = []
        for start_idx in range(0, n_token, batch_size):
            stop_idx = min(start_idx + batch_size, n_token)
            batch = sub[start_idx:stop_idx]
            dist = torch.cdist(batch, centers)
            all_assign.append(dist.argmin(dim=1).to(torch.uint8))
        codes[:, sub_idx] = torch.cat(all_assign, dim=0)
        centroids_out.append(centers.to(dtype=torch.float16))

    return {
        "centroids": torch.stack(centroids_out, dim=0),
        "codes": codes,
        "subdim": subdim,
        "bits": bits,
    }


def pq_reconstruct_tokens(
    centroids: torch.Tensor,
    codes: torch.Tensor,
    token_ids: torch.Tensor | list[int] | None = None,
) -> torch.Tensor:
    if token_ids is None:
        selected_codes = codes
    else:
        if not torch.is_tensor(token_ids):
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=codes.device)
        token_ids = token_ids.to(device=codes.device, dtype=torch.long).view(-1)
        selected_codes = codes.index_select(0, token_ids)

    parts: list[torch.Tensor] = []
    for sub_idx in range(selected_codes.shape[1]):
        parts.append(
            centroids[sub_idx].index_select(0, selected_codes[:, sub_idx].long())
        )
    return torch.cat(parts, dim=1).to(dtype=torch.float32)


def pq_scores(
    query: torch.Tensor,
    centroids: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    query = query.to(dtype=torch.float32)
    n_sub, _, subdim = centroids.shape
    query_parts = query.view(n_sub, subdim)
    lut = torch.einsum("md,mkd->mk", query_parts, centroids.to(dtype=torch.float32))
    scores = torch.zeros(codes.shape[0], device=lut.device, dtype=lut.dtype)
    for sub_idx in range(n_sub):
        scores = scores + lut[sub_idx].index_select(0, codes[:, sub_idx].long())
    return scores
