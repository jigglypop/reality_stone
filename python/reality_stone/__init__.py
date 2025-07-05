import torch
import numpy as np
from torch.autograd import Function

_has_rust_ext = False
_has_cuda = False

try:
    # The module name is defined in pyproject.toml and Cargo.toml
    from ._rust import (
        mobius_add_cpu,
        mobius_scalar_cpu,
        poincare_distance_cpu,
        poincare_ball_layer_cpu,
        # TODO: Other CPU functions will be added here
    )
    _has_rust_ext = True

    # Conditionally import CUDA functions only if Rust extension loaded and CUDA is available
    if torch.cuda.is_available():
        from ._rust import (
            mobius_add_cuda, 
            mobius_scalar_cuda, 
            poincare_distance_cuda,
            poincare_ball_layer_cuda
        )
        _has_cuda = True

except ImportError:
    # This is a genuine import error, meaning the .so file is missing.
    print("⚠️ Reality Stone: Rust extension (.so file) not found.")
    print("   Please build the project first (e.g., `maturin develop`).")
except Exception as e:
    # Any other exception during import (e.g., linking error, CUDA error)
    import traceback
    print("🔥 Reality Stone: An unexpected error occurred while importing the Rust extension.")
    print(f"   Error Type: {type(e).__name__}")
    print(f"   Error Details: {e}")
    print("--- Traceback ---")
    traceback.print_exc()
    print("-----------------")
    print("   This might be due to a missing dependency (like CUDA toolkit) or a build mismatch.")


# --- Public API ---

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    if not _has_rust_ext:
        raise RuntimeError("Reality Stone's Rust extension is not installed. Please build it first.")

    if x.device.type == 'cuda':
        if not _has_cuda:
            raise RuntimeError("Reality Stone was not built with CUDA support, but the input tensor is on a CUDA device.")
        if not x.is_contiguous() or not y.is_contiguous():
            raise ValueError("Input tensors must be contiguous for CUDA operations.")
        
        out = torch.empty_like(x)
        batch_size, dim = x.shape
        
        mobius_add_cuda(
            x.data_ptr(),
            y.data_ptr(),
            out.data_ptr(),
            batch_size,
            dim,
            c
        )
        return out
    else:
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        result_np = mobius_add_cpu(x_np, y_np, c)
        return torch.from_numpy(result_np).to(x.device)


def mobius_scalar(x: torch.Tensor, r: float, c: float) -> torch.Tensor:
    if not _has_rust_ext:
        raise RuntimeError("Reality Stone's Rust extension is not installed. Please build it first.")
    if x.device.type == 'cuda':
        if not _has_cuda:
            raise RuntimeError("Reality Stone was not built with CUDA support, but the input tensor is on a CUDA device.")
        if not x.is_contiguous():
            raise ValueError("Input tensor must be contiguous for CUDA operations.")
        out = torch.empty_like(x)
        batch_size, dim = x.shape
        mobius_scalar_cuda(
            x.data_ptr(),
            out.data_ptr(),
            batch_size,
            dim,
            r,
            c
        )
        return out
    else:
        x_np = x.detach().cpu().numpy()
        result_np = mobius_scalar_cpu(x_np, r, c)
        return torch.from_numpy(result_np).to(x.device)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    if not _has_rust_ext:
        raise RuntimeError("Reality Stone's Rust extension is not installed. Please build it first.")

    if x.device.type == 'cuda':
        if not _has_cuda:
            raise RuntimeError("Reality Stone was not built with CUDA support, but the input tensor is on a CUDA device.")
        if not x.is_contiguous() or not y.is_contiguous():
            raise ValueError("Input tensors must be contiguous for CUDA operations.")
        
        out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
        batch_size, dim = x.shape
        
        poincare_distance_cuda(
            x.data_ptr(),
            y.data_ptr(),
            out.data_ptr(),
            batch_size,
            dim,
            c
        )
        return out
    else:
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        result_np = poincare_distance_cpu(x_np, y_np, c)
        return torch.from_numpy(result_np).to(x.device)


def poincare_ball_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    if not _has_rust_ext:
        raise RuntimeError("Reality Stone's Rust extension is not installed. Please build it first.")

    if u.device.type == 'cuda':
        if not _has_cuda:
            raise RuntimeError("Reality Stone was not built with CUDA support, but the input tensor is on a CUDA device.")
        if not u.is_contiguous() or not v.is_contiguous():
            raise ValueError("Input tensors must be contiguous for CUDA operations.")
        
        out = torch.empty_like(u)
        batch_size, dim = u.shape
        
        poincare_ball_layer_cuda(
            u.data_ptr(),
            v.data_ptr(),
            out.data_ptr(),
            batch_size,
            dim,
            c,
            t
        )
        return out
    else:
        u_np = u.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        result_np = poincare_ball_layer_cpu(u_np, v_np, c, t)
        return torch.from_numpy(result_np).to(u.device)

def _not_implemented(*args, **kwargs):
    raise NotImplementedError("This function is part of the legacy API and has not been ported to the new Rust backend yet.")

class PoincareBall(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        return poincare_ball_layer(u, v, c, t)

    @staticmethod
    def backward(ctx, grad_output):
        _not_implemented()

poincare_ball_forward_cpu = _not_implemented
poincare_ball_backward_cpu = _not_implemented
lorentz_forward_cpu = _not_implemented
lorentz_backward_cpu = _not_implemented
klein_forward_cpu = _not_implemented
klein_backward_cpu = _not_implemented
poincare_to_lorentz_cpu = _not_implemented
lorentz_to_poincare_cpu = _not_implemented
poincare_to_klein_cpu = _not_implemented
klein_to_poincare_cpu = _not_implemented
lorentz_to_klein_cpu = _not_implemented
klein_to_lorentz_cpu = _not_implemented

if _has_cuda:
    poincare_ball_forward_cuda = _not_implemented

# === Differentiable Torch Fallbacks ===
# Autograd-friendly implementations are used when gradients are required.

def _mobius_add_torch(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1e-7) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True)
    xy = (x * y).sum(dim=1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / denom.clamp_min(eps)

def _mobius_scalar_torch(x: torch.Tensor, r: float, c: float, eps: float = 1e-7) -> torch.Tensor:
    sqrtc = c ** 0.5
    x_norm = x.norm(dim=1, keepdim=True).clamp_min(eps)
    scale = torch.tanh(r * torch.atanh(sqrtc * x_norm)) / (sqrtc * x_norm)
    return scale * x

def _poincare_ball_layer_torch(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    u_prime = _mobius_scalar_torch(u, 1.0 - t, c)
    v_prime = _mobius_scalar_torch(v, t, c)
    return _mobius_add_torch(u_prime, v_prime, c)

# Preserve original fast (Rust/CUDA) functions
_mobius_add_fast = mobius_add  # type: ignore
_mobius_scalar_fast = mobius_scalar  # type: ignore
_poincare_ball_layer_fast = poincare_ball_layer  # type: ignore

# Override public API with gradient-aware wrapper

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:  # type: ignore
    if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad):
        return _mobius_add_torch(x, y, c)
    return _mobius_add_fast(x, y, c)

def mobius_scalar(x: torch.Tensor, r: float, c: float) -> torch.Tensor:  # type: ignore
    if torch.is_grad_enabled() and x.requires_grad:
        return _mobius_scalar_torch(x, r, c)
    return _mobius_scalar_fast(x, r, c)

def poincare_ball_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:  # type: ignore
    if torch.is_grad_enabled() and (u.requires_grad or v.requires_grad):
        return _poincare_ball_layer_torch(u, v, c, t)
    return _poincare_ball_layer_fast(u, v, c, t)

# Re-export
__all__ = ['mobius_add', 'mobius_scalar', '_has_rust_ext', '_has_cuda']
__all__.extend([
    'poincare_ball_layer',
    'mobius_add',
    'mobius_scalar',
])
