import torch
import os
import sys
from pathlib import Path

# 현재 파일의 디렉토리를 기준으로 .so 파일 경로를 명시적으로 지정
_lib_path = Path(__file__).parent.resolve()
_so_file = list(_lib_path.glob('_rust*.so'))

_has_rust_ext = False
_has_cuda = False

if _so_file:
    try:
        # sys.path에 .so 파일이 있는 디렉토리를 추가하여 임포트 보장
        if str(_lib_path) not in sys.path:
            sys.path.insert(0, str(_lib_path))
        from . import _rust
        _has_rust_ext = True
        _has_cuda = torch.cuda.is_available()
    except ImportError as e:
        print(f"🔥 Reality Stone: Found .so file, but failed to import: {_so_file[0]}")
        print(f"   Error: {e}")
else:
    print("⚠️ Reality Stone: Rust extension (.so file) not found in package directory.")
    print("   Please build the project first (e.g., `maturin develop`).")


from .core.ops import PoincareBallLayer, mobius_add, mobius_scalar, poincare_distance
from .models import *
from .optimizations import *

def poincare_ball_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    return PoincareBallLayer.apply(u, v, c, t)

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
