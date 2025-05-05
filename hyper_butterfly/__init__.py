import torch
from torch.autograd import Function
import math

from ._C import (
    log_map_cpu,
    exp_map_cpu,
    poincare_forward_cpu,
    geodesic_cpu,
    geodesic_forward_cpu,
    geodesic_backward_cpu,
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
            geodesic_forward_cuda,
            geodesic_backward_cuda,
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


class GeodesicFunction(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        if u.is_cuda and _has_cuda:
            result = geodesic_forward_cuda(u, v, c, t)
        else:
            result = geodesic_forward_cpu(u, v, c, t)
        ctx.save_for_backward(u, v)
        ctx.c = c
        ctx.t = t
        return result

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        if u.is_cuda and _has_cuda:
            grad_u, grad_v = geodesic_backward_cuda(grad_output, u, v, c, t)
        else:
            grad_u, grad_v = geodesic_backward_cpu(grad_output, u, v, c, t)
    
        return grad_u, grad_v, None, None

def geodesic_layer(u, v, c, t):
    return GeodesicFunction.apply(u, v, c, t)