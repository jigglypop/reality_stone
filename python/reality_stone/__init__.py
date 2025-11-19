import torch
import sys
from pathlib import Path

# 현재 파일의 디렉토리를 기준으로 .so 파일 경로를 명시적으로 지정
_lib_path = Path(__file__).parent.resolve()
_so_file = list(_lib_path.glob('_rust*.so')) or list(_lib_path.glob('_rust*.pyd'))

_has_rust_ext = False
_has_cuda = False

if _so_file:
    try:
        if str(_lib_path) not in sys.path:
            sys.path.insert(0, str(_lib_path))
        from . import _rust
        _has_rust_ext = True
        # CUDA 가용 여부는 PyTorch CUDA 가능 + Rust 확장에 필요한 CUDA 심볼 존재 여부를 함께 확인
        if torch.cuda.is_available():
            # 현재 Python 레이어에서 직접 사용하는 CUDA 바인딩 검사
            required_cuda_symbols = [
                # Möbius
                'mobius_add_cuda',
                'mobius_scalar_cuda',
                # Poincaré
                'poincare_ball_layer_cuda',
                'poincare_ball_layer_backward_cuda',
                'poincare_distance_cuda',
                # Lorentz
                'lorentz_layer_forward_cuda',
                'lorentz_ball_layer_backward_cuda',
                'lorentz_distance_cuda',
                # Klein
                'klein_layer_forward_cuda',
                'klein_ball_layer_backward_cuda',
                'klein_distance_cuda',
            ]
            _has_cuda = all(hasattr(_rust, name) for name in required_cuda_symbols)
        else:
            _has_cuda = False
    except ImportError as e:
        _rust = None  # type: ignore
else:
    _rust = None  # type: ignore


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
