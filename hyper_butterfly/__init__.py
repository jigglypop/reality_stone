import torch
from torch.autograd import Function
import math

from ._C import (
    log_map_cpu,
    exp_map_cpu,
    poincare_forward_cpu,
    geodesic_cpu,
)
_has_cuda = False
if torch.cuda.is_available():
    try:
        from ._C import (
            log_map_cuda,
            exp_map_cuda,
            log_map_forward_cuda,
            exp_map_forward_cuda,
            poincare_forward_cuda,
            poincare_backward_cuda,
            geodesic_cuda,
        )
        _has_cuda = True
    except ImportError:
        _has_cuda = False

from .python.maps import log_map, exp_map, geodesic
from .python.layers import HyperButterflyFunction, GeodesicButterflyLayer

def hyper_butterfly(x: torch.Tensor, params: torch.Tensor, c: float, L: int):
    return HyperButterflyFunction.apply(x, params, c, L)

def geodesic_butterfly(x: torch.Tensor,params: torch.Tensor,c: float,L: int,t: float) -> torch.Tensor:
    layer = GeodesicButterflyLayer(x.size(1), c, L, t)
    return layer(x) 