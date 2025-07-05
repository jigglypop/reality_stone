import torch
import numpy as np

_has_rust_ext = False
_has_cuda = False

try:
    from .._rust import (
        mobius_add_cpu,
        mobius_scalar_cpu,
        poincare_distance_cpu,
        poincare_ball_layer_cpu,
    )
    _has_rust_ext = True

    if torch.cuda.is_available():
        from .._rust import (
            mobius_add_cuda, 
            mobius_scalar_cuda, 
            poincare_distance_cuda,
            poincare_ball_layer_cuda
        )
        _has_cuda = True

except ImportError:
    print("⚠️ Reality Stone: Rust extension (.so file) not found.")
    print("   Please build the project first (e.g., `maturin develop`).")
except Exception as e:
    import traceback
    print("🔥 Reality Stone: An unexpected error occurred while importing the Rust extension.")
    print(f"   Error Type: {type(e).__name__}")
    print(f"   Error Details: {e}")
    print("--- Traceback ---")
    traceback.print_exc()
    print("-----------------")

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Möbius addition in hyperbolic space
    
    Args:
        x: First tensor [B, D]
        y: Second tensor [B, D]
        c: Curvature
        
    Returns:
        x ⊕_c y
    """
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
    """Möbius scalar multiplication
    
    Args:
        x: Input tensor [B, D]
        r: Scalar
        c: Curvature
        
    Returns:
        r ⊗_c x
    """
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
    """Poincaré distance between points
    
    Args:
        x: First tensor [B, D]
        y: Second tensor [B, D]
        c: Curvature
        
    Returns:
        Distance tensor [B]
    """
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
    """Poincaré ball layer operation
    
    Args:
        u: First tensor [B, D]
        v: Second tensor [B, D]
        c: Curvature
        t: Interpolation parameter
        
    Returns:
        Output tensor [B, D]
    """
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