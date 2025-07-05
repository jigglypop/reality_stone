import torch
import torch.nn as nn
from torch.autograd import Function

# Import core operations
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

# Import layers
from .layers import *

# Import models  
from .models import *

# Import optimizations
from .optimizations import *

# Import advanced features from advanced.py directly
from .advanced import AdvancedConfig

# Gradient-aware wrappers
def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Möbius addition with automatic differentiation support"""
    if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad):
        return _mobius_add_torch(x, y, c)
    return _mobius_add_fast(x, y, c)

def mobius_scalar(x: torch.Tensor, r: float, c: float) -> torch.Tensor:
    """Möbius scalar multiplication with automatic differentiation support"""
    if torch.is_grad_enabled() and x.requires_grad:
        return _mobius_scalar_torch(x, r, c)
    return _mobius_scalar_fast(x, r, c)

def poincare_ball_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    """Poincaré ball layer with automatic differentiation support"""
    if torch.is_grad_enabled() and (u.requires_grad or v.requires_grad):
        return _poincare_ball_layer_torch(u, v, c, t)
    return _poincare_ball_layer_fast(u, v, c, t)

# Legacy API compatibility
def _not_implemented(*args, **kwargs):
    raise NotImplementedError("This function is part of the legacy API and has not been ported to the new Rust backend yet.")

class PoincareBall(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        return poincare_ball_layer(u, v, c, t)

    @staticmethod
    def backward(ctx, grad_output):
        _not_implemented()

# Legacy function stubs
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
poincare_ball_forward_cuda = _not_implemented

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
    # Layers
    'DynamicCurvatureLayer', 'HyperbolicLinearAdvanced', 'GeodesicActivationLayer',
    'RegularizedHyperbolicLayer', 'AdvancedHyperbolicMLP', 'DynamicCurvatureMLP',
    'FusedHyperbolicLayer',
    # Models
    'LorentzMLP', 'KleinMLP',
    # Optimizations
    'OptimizationConfig', 'PerformanceProfiler', 'OptimizedModel',
    'AdaptiveBatchSize', 'MemoryOptimizer',
    # Advanced
    'AdvancedConfig',
    # Status
    '_has_rust_ext', '_has_cuda'
]
