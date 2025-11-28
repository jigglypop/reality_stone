__version__ = "0.2.0"

import torch
import sys
from pathlib import Path

_has_rust_ext = False
_has_cuda = False

try:
    from . import _rust  # type: ignore
    _has_rust_ext = True
except Exception:
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

from .layers.poincare import (
    PoincareBallLayer,
    poincare_add,
    poincare_scalar_mul,
    poincare_distance,
    poincare_to_lorentz,
    poincare_to_klein,
    project_to_ball,
    HyperbolicLinear,
    GeodesicLinear,
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
from .layers.metric_attention import MetricAttention, SPDMetric

try:
    if _has_rust_ext:
        from ._rust import metrikey  # type: ignore
    else:
        metrikey = None  # type: ignore
except Exception:
    metrikey = None  # type: ignore

try:
    if _has_rust_ext:
        from ._rust import geodesic as _geodesic  # type: ignore
        geodesic_topk_attention = _geodesic.geodesic_topk_attention
        batched_cholesky = _geodesic.batched_cholesky_cuda
    else:
        geodesic_topk_attention = None  # type: ignore
        batched_cholesky = None  # type: ignore
except Exception:
    geodesic_topk_attention = None  # type: ignore
    batched_cholesky = None  # type: ignore

from .conversion import convert_to_full_riemannian, convert_to_hyperbolic
from .losses import HyperbolicSupConLoss, BellmanConsistencyLoss, laplacian_same_label, poincare_kinetic_energy

from . import optim
from . import layers

try:
    from . import data
except ImportError:
    data = None  # type: ignore

try:
    from . import models
except ImportError:
    models = None  # type: ignore

try:
    if _has_rust_ext:
        from ._rust import PyUnifiedRiemannianLayer as UnifiedRiemannianLayer  # type: ignore
        from ._rust import compute_metric, geodesic_distance, geodesic_interpolate  # type: ignore
    else:
        UnifiedRiemannianLayer = None  # type: ignore
        compute_metric = None  # type: ignore
        geodesic_distance = None  # type: ignore
        geodesic_interpolate = None  # type: ignore
except Exception:
    UnifiedRiemannianLayer = None  # type: ignore
    compute_metric = None  # type: ignore
    geodesic_distance = None  # type: ignore
    geodesic_interpolate = None  # type: ignore

try:
    if _has_rust_ext:
        from ._rust import PyRiemannianDiffusion  # type: ignore
    else:
        PyRiemannianDiffusion = None  # type: ignore
except Exception:
    PyRiemannianDiffusion = None  # type: ignore

try:
    if _has_rust_ext:
        from ._rust import PyRSULFLayer as RSULFLayer  # type: ignore
        from ._rust import fold_metric_svd, fold_ffn, build_causal_laplacian  # type: ignore
        from ._rust import verify_metric_consistency, fold_metric_optimized, nystrom_metric  # type: ignore
        from ._rust import bellman_geodesic_forward, bellman_geodesic_backward  # type: ignore
        from ._rust import extract_metric_cuda  # type: ignore
    else:
        RSULFLayer = None  # type: ignore
        fold_metric_svd = None  # type: ignore
        fold_ffn = None  # type: ignore
        build_causal_laplacian = None  # type: ignore
        verify_metric_consistency = None  # type: ignore
        fold_metric_optimized = None  # type: ignore
        nystrom_metric = None  # type: ignore
        bellman_geodesic_forward = None  # type: ignore
        bellman_geodesic_backward = None  # type: ignore
        extract_metric_cuda = None  # type: ignore
except Exception:
    RSULFLayer = None  # type: ignore
    fold_metric_svd = None  # type: ignore
    fold_ffn = None  # type: ignore
    build_causal_laplacian = None  # type: ignore
    verify_metric_consistency = None  # type: ignore
    fold_metric_optimized = None  # type: ignore
    nystrom_metric = None  # type: ignore
    bellman_geodesic_forward = None  # type: ignore
    bellman_geodesic_backward = None  # type: ignore
    extract_metric_cuda = None  # type: ignore


def poincare_ball_layer(u: torch.Tensor, v: torch.Tensor, c: float = None, t: float = 0.5, kappas: torch.Tensor = None, layer_idx: int = None, c_min: float = -2.0, c_max: float = -0.1) -> torch.Tensor:
    return PoincareBallLayer.apply(u, v, c, t, kappas, layer_idx, c_min, c_max)


def klein_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    return KleinLayer.apply(u, v, c, t)


def lorentz_layer(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    return LorentzLayer.apply(u, v, c, t)


__all__ = [
    '__version__',
    '_has_rust_ext',
    '_has_cuda',
    'MobiusAdd',
    'MobiusScalarMul',
    'poincare_add',
    'poincare_scalar_mul',
    'poincare_distance',
    'poincare_ball_layer',
    'PoincareBallLayer',
    'poincare_to_lorentz',
    'poincare_to_klein',
    'project_to_ball',
    'HyperbolicLinear',
    'GeodesicLinear',
    'lorentz_add',
    'lorentz_scalar_mul',
    'lorentz_distance',
    'lorentz_inner',
    'lorentz_to_poincare',
    'lorentz_to_klein',
    'lorentz_layer',
    'LorentzLayer',
    'klein_add',
    'klein_scalar_mul',
    'klein_distance',
    'klein_to_poincare',
    'klein_to_lorentz',
    'klein_layer',
    'KleinLayer',
    'SplineLinear',
    'MetricAttention',
    'SPDMetric',
    'convert_to_full_riemannian',
    'convert_to_hyperbolic',
    'HyperbolicSupConLoss',
    'BellmanConsistencyLoss',
    'laplacian_same_label',
    'poincare_kinetic_energy',
    'optim',
    'layers',
    'data',
    'models',
    'metrikey',
    'geodesic_topk_attention',
    'batched_cholesky',
    'UnifiedRiemannianLayer',
    'compute_metric',
    'geodesic_distance',
    'geodesic_interpolate',
    'PyRiemannianDiffusion',
    'RSULFLayer',
    'fold_metric_svd',
    'fold_ffn',
    'build_causal_laplacian',
    'verify_metric_consistency',
    'fold_metric_optimized',
    'nystrom_metric',
    'bellman_geodesic_forward',
    'bellman_geodesic_backward',
    'extract_metric_cuda',
]
