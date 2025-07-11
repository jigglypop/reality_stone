import torch
from .core import (
    mobius_add as _mobius_add_fast,
    mobius_scalar as _mobius_scalar_fast,
    poincare_distance,
    poincare_ball_layer as _poincare_ball_layer_fast
)
from .core.tensors import (
    _mobius_add_torch,
    _mobius_scalar_torch,
    _poincare_ball_layer_torch
)

from .models import *
from .optimizations import *

# Gradient-aware wrappers
def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad):
        return _mobius_add_torch(x, y, c)
    return _mobius_add_fast(x, y, c)

def mobius_scalar(x: torch.Tensor, r: float, c: float) -> torch.Tensor:
    if torch.is_grad_enabled() and x.requires_grad:
        return _mobius_scalar_torch(x, r, c)
    return _mobius_scalar_fast(x, r, c)

def poincare_ball_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    if torch.is_grad_enabled() and (u.requires_grad or v.requires_grad):
        return _poincare_ball_layer_torch(u, v, c, t)
    return _poincare_ball_layer_fast(u, v, c, t)

# Check for Rust extension
try:
    from ._rust import __version__
    _has_rust_ext = True
    _has_cuda = torch.cuda.is_available()
except ImportError:
    _has_rust_ext = False
    _has_cuda = False

# Re-export
__all__ = [
    # Core operations
    'mobius_add', 'mobius_scalar', 'poincare_distance', 'poincare_ball_layer',
    # Models
    'LorentzMLP', 'KleinMLP',
    # Optimizations
    'OptimizationConfig', 'PerformanceProfiler', 'OptimizedModel',
    'AdaptiveBatchSize', 'MemoryOptimizer',
    # Status
    '_has_rust_ext', '_has_cuda'
]
