import torch
import sys
from pathlib import Path

_has_rust_ext = False
_has_cuda = False

# 1) 우선 site-packages에 설치된 확장을 우선 시도 (maturin develop로 설치된 최신 빌드)
try:
    from . import _rust  # type: ignore
    _has_rust_ext = True
except Exception:
    # 2) 실패 시에만 로컬 번들된 바이너리(_rust*.pyd/.so)를 fallback으로 시도
    _rust = None  # type: ignore
    try:
        lib_path = Path(__file__).parent.resolve()
        local_ext = list(lib_path.glob('_rust*.so')) or list(lib_path.glob('_rust*.pyd'))
        if local_ext:
            if str(lib_path) not in sys.path:
                sys.path.insert(0, str(lib_path))
            from . import _rust as _rust_local  # type: ignore
            _rust = _rust_local  # type: ignore
            _has_rust_ext = True
    except Exception:
        _rust = None  # type: ignore

# CUDA 가용 여부는 PyTorch CUDA 가능 + Rust 확장에 필요한 CUDA 심볼 존재 여부를 함께 확인
if _has_rust_ext and torch.cuda.is_available():
    required_cuda_symbols = [
        'mobius_add_cuda',
        'mobius_scalar_cuda',
        'poincare_ball_layer_cuda',
        'poincare_ball_layer_backward_cuda',
        'poincare_distance_cuda',
        'lorentz_layer_forward_cuda',
        'lorentz_ball_layer_backward_cuda',
        'lorentz_distance_cuda',
        'klein_layer_forward_cuda',
        'klein_ball_layer_backward_cuda',
        'klein_distance_cuda',
    ]
    _has_cuda = all(hasattr(_rust, name) for name in required_cuda_symbols)  # type: ignore
else:
    _has_cuda = False


from .core.mobius import MobiusAdd, MobiusScalarMul
# Explicit re-exports from layers to avoid wildcard imports
from .layers.poincare import (
    PoincareBallLayer,
    poincare_add,
    poincare_scalar_mul,
    poincare_distance,
    poincare_to_lorentz,
    poincare_to_klein,
)
from .layers.lorentz import (
    LorentzLayer,
    lorentz_add,
    lorentz_scalar_mul,
    lorentz_distance,
    lorentz_inner,
    lorentz_to_poincare,
    lorentz_to_klein,
)
from .layers.klein import (
    KleinLayer,
    klein_add,
    klein_scalar_mul,
    klein_distance,
    klein_to_poincare,
    klein_to_lorentz,
)
from .layers.spline import SplineLinear
# Optional MetriKey (Rust) binding
try:
    if _has_rust_ext:
        from ._rust import metrikey  # type: ignore
    else:
        metrikey = None  # type: ignore
except Exception:
    metrikey = None  # type: ignore

# 모델 변환 유틸리티 추가
from .conversion import convert_to_hyperbolic

def poincare_ball_layer(u: torch.Tensor, v: torch.Tensor, c: float = None, t: float = 0.5, kappas: torch.Tensor = None, layer_idx: int = None, c_min: float = -2.0, c_max: float = -0.1) -> torch.Tensor:
    return PoincareBallLayer.apply(u, v, c, t, kappas, layer_idx, c_min, c_max)

def klein_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    return KleinLayer.apply(u, v, c, t)

def lorentz_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    return LorentzLayer.apply(u, v, c, t)

# Re-export
__all__ = [
    # Core
    'MobiusAdd',
    'MobiusScalarMul',
    # Poincare
    'poincare_add', 
    'poincare_scalar_mul', 
    'poincare_distance', 
    'poincare_ball_layer',
    'PoincareBallLayer',
    'poincare_to_lorentz',
    'poincare_to_klein',
    # Lorentz
    'lorentz_add',
    'lorentz_scalar_mul',
    'lorentz_distance',
    'lorentz_inner',
    'lorentz_to_poincare',
    'lorentz_to_klein',
    'lorentz_layer',
    'LorentzLayer',
    # Klein
    'klein_add',
    'klein_scalar_mul',
    'klein_distance',
    'klein_to_poincare',
    'klein_to_lorentz',
    'klein_layer',
    'KleinLayer',
    # Status
    '_has_rust_ext', '_has_cuda',

    'SplineLinear',
    'convert_to_hyperbolic',
    # MetriKey (Rust bindings)
    'metrikey',
]
