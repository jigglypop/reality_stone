import torch
import sys
from pathlib import Path

# 현재 파일의 디렉토리를 기준으로 .so 파일 경로를 명시적으로 지정
_lib_path = Path(__file__).parent.resolve()
_so_file = list(_lib_path.glob('_rust*.so'))

_has_rust_ext = False
_has_cuda = False

if _so_file:
    try:
        if str(_lib_path) not in sys.path:
            sys.path.insert(0, str(_lib_path))
        from . import _rust
        _has_rust_ext = True
        _has_cuda = torch.cuda.is_available()
    except ImportError as e:
        print(f" Reality Stone: Found .so file, but failed to import: {_so_file[0]}")
        print(f" Error: {e}")
else:
    print(" Reality Stone: Rust extension (.so file) not found in package directory.")
    print(" Please build the project first (e.g., `maturin develop`).")


from .core.mobius import MobiusAdd, MobiusScalarMul
from .layers import *

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
    '_has_rust_ext', '_has_cuda'
]
