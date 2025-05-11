import torch
from torch.autograd import Function
import math

from ._C import (
    log_map_cpu,
    exp_map_cpu,
    poincare_forward_cpu,
    geodesic_cpu,
    geodesic_forward_cpu,
)
_has_cuda = False
if torch.cuda.is_available():
    print("CUDA is available")
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
        )
        _has_cuda = True
    except ImportError:
        _has_cuda = False

from .maps import log_map, exp_map, geodesic
from .layers import HyperButterflyFunction, GeodesicButterflyLayer

def hyper_butterfly(x: torch.Tensor, params: torch.Tensor, c: float, L: int):
    return HyperButterflyFunction.apply(x, params, c, L)

def geodesic_butterfly(x: torch.Tensor,params: torch.Tensor,c: float,L: int,t: float) -> torch.Tensor:
    layer = GeodesicButterflyLayer(x.size(1), c, L, t)
    return layer(x) 
# autograd를 사용하는 버전으로 변경
class GeodesicFunction(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c = c
        ctx.t = t
        
        # forward만 수행하고 requires_grad 처리
        with torch.enable_grad():
            if u.is_cuda and _has_cuda:
                result = geodesic_forward_cuda(u, v, c, t)
            else:
                result = geodesic_forward_cpu(u, v, c, t)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        
        # autograd를 사용하여 backprop 계산
        with torch.enable_grad():
            u_new = u.detach().requires_grad_(True)
            v_new = v.detach().requires_grad_(True)
            
            if u_new.is_cuda and _has_cuda:
                output = geodesic_forward_cuda(u_new, v_new, c, t)
            else:
                output = geodesic_forward_cpu(u_new, v_new, c, t)
            
            grad_u, grad_v = torch.autograd.grad(
                output, 
                [u_new, v_new], 
                grad_outputs=grad_output,
                retain_graph=True
            )
        
        return grad_u, grad_v, None, None

def geodesic_layer(u, v, c, t):
    return GeodesicFunction.apply(u, v, c, t)

# class GeodesicFunction(Function):
#     @staticmethod
#     def forward(ctx, u, v, c, t):
#         if u.is_cuda and _has_cuda:
#             result = geodesic_forward_cuda(u, v, c, t)
#         else:
#             result = geodesic_forward_cpu(u, v, c, t)
#         ctx.save_for_backward(u, v)
#         ctx.c = c
#         ctx.t = t
#         return result
# 
#     @staticmethod
#     def backward(ctx, grad_output):
#         u, v = ctx.saved_tensors
#         c, t = ctx.c, ctx.t
#         print(_has_cuda)
#         grad_u, grad_v = geodesic_backward_cuda(grad_output, u, v, c, t)
#         # else:
#         #     # CPU 버전이 문제이므로 대안 사용
#         #     grad_u = torch.zeros_like(u)
#         #     grad_v = torch.zeros_like(v)
#         #     
#         #     # 단순한 그래디언트 근사
#         #     grad_u = grad_output * (1.0 - t)
#         #     grad_v = grad_output * t
#         return grad_u, grad_v, None, None
# 
# def geodesic_layer(u, v, c, t):
#     return GeodesicFunction.apply(u, v, c, t)