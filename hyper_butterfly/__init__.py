# import torch
# from torch.autograd import Function
# import sys
# sys.modules['hb'] = sys.modules[__name__]
# 
# from ._C import (
#     log_map_origin_cpu,
#     exp_map_origin_cpu,
#     hyper_butterfly_cpu,
# )
# _has_cuda = False
# if torch.cuda.is_available():
#     try:
#         from ._C import (
#             log_map_origin_cuda,
#             exp_map_origin_cuda,
#             hyper_butterfly_cuda,
#             hyper_butterfly_backward_cuda,   
#         )
#         _has_cuda = True
#     except ImportError:
#         _has_cuda = False
# 
# from .hyper_butterfly_py import hyper_butterfly_py
# 
# class HyperButterflyFunction(Function):
#     @staticmethod
#     def forward(ctx, x, params, c, L):
#         ctx.save_for_backward(x, params)
#         ctx.c, ctx.L = c, L
#         if x.is_cuda and _has_cuda:
#             y, u, v = hyper_butterfly_cuda(x, params, torch.empty(0,device=x.device), c, L)
#         else:
#             if not x.is_cuda and 'hyper_butterfly_cpu' in globals():
#                 y, u, v = hyper_butterfly_cpu(x, params, c, L)
#             else:
#                 y = hyper_butterfly_py(x, params, c, L)
#         return y
# 
#     @staticmethod
#     def backward(ctx, grad_out):
#         x, params = ctx.saved_Tensors
#         c, L = ctx.c, ctx.L
#         if x.is_cuda and _has_cuda:
#             grad_x, grad_p = hyper_butterfly_backward_cuda(
#                 grad_out.contiguous(), x, params, c, L
#             )
#             return grad_x, grad_p, None, None
#         with torch.enable_grad():
#             x_req = x.detach().requires_grad_()
#             p_req = params.detach().requires_grad_()
#             y = hyper_butterfly_py(x_req, p_req, c, L)
#             gx, gp = torch.autograd.grad(y, (x_req, p_req), grad_out)
#         return gx, gp, None, None
# 
# def hyper_butterfly(x: torch.Tensor, params: torch.Tensor, c: float, L: int):
#     return HyperButterflyFunction.apply(x, params, c, L)
# core/__init__.py
import torch
from torch.autograd import Function
import sys

# hb로 접근 가능하도록 설정
sys.modules['hb'] = sys.modules[__name__]

# 새로운 명명법의 C++ 함수들 임포트
from ._C import (
    log_map_cpu,
    exp_map_cpu,
    butterfly_forward_cpu,
)

_has_cuda = False
if torch.cuda.is_available():
    try:
        from ._C import (
            log_map_cuda,
            exp_map_cuda,
            butterfly_forward_cuda,
            butterfly_backward_cuda,
            log_map_backward_cuda,
            exp_map_backward_cuda,
        )
        _has_cuda = True
    except ImportError:
        _has_cuda = False

# Hyper-Butterfly 연산을 위한 autograd Function
class HyperButterflyFunction(Function):
    @staticmethod
    def forward(ctx, x, params, c, L):
        ctx.save_for_backward(x, params)
        ctx.c, ctx.L = c, L
        # 로그 맵 적용
        if x.is_cuda and _has_cuda:
            u = log_map_cuda(x, c)
        else:
            u = log_map_cpu(x, c)
        # 버터플라이 변환 적용
        v = u.clone()
        batch_size, dim = x.shape
        log2_dim = int(torch.log2(torch.tensor(float(dim))).item())
        for l in range(L):
            layer_idx = l % log2_dim
            if v.is_cuda and _has_cuda:
                v = butterfly_forward_cuda(v, params, layer_idx, batch_size, dim)
            else:
                v = butterfly_forward_cpu(v, params, layer_idx, batch_size, dim)
        
        # 지수 맵 적용
        if v.is_cuda and _has_cuda:
            y = exp_map_cuda(v, c)
        else:
            y = exp_map_cpu(v, c)
        # 중간 결과 저장 (역전파용)
        ctx.save_for_backward(x, params)
        ctx.intermediate = (u, v)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, params = ctx.saved_tensors
        c, L = ctx.c, ctx.L
        u, v = ctx.intermediate
        batch_size, dim = x.shape
        log2_dim = int(torch.log2(torch.tensor(float(dim))).item())
        if x.is_cuda and _has_cuda:
            # 지수 맵 역전파 (C++ 구현 사용)
            grad_v = exp_map_backward_cuda(grad_out.contiguous(), v, c)[0]
            # 버터플라이 네트워크 역전파 (레이어별로 역순으로)
            grad_params = torch.zeros_like(params)
            current_grad = grad_v
            for l in range(L-1, -1, -1):
                layer_idx = l % log2_dim
                # 단일 레이어 역전파 (C++ 구현 사용)
                layer_grads = butterfly_backward_cuda(
                    current_grad, 
                    v if l == L-1 else u,  # 입력 텐서 (첫 레이어면 u, 나머지는 중간값이 필요)
                    params, 
                    layer_idx
                )
                current_grad = layer_grads[0]  # 입력에 대한 그래디언트
                grad_params += layer_grads[1]  # 파라미터 그래디언트 누적
            # 로그 맵 역전파 (C++ 구현 사용)
            grad_x = log_map_backward_cuda(current_grad, x, c)[0]
        else:
            # CPU 구현 또는 자동 미분
            with torch.enable_grad():
                x_req = x.detach().requires_grad_()
                p_req = params.detach().requires_grad_()
                # forward 과정 수행
                u_req = log_map_cpu(x_req, c)
                v_req = u_req.clone()
                for l in range(L):
                    layer_idx = l % log2_dim
                    v_req = butterfly_forward_cpu(v_req, p_req, layer_idx, batch_size, dim)
                y_req = exp_map_cpu(v_req, c)
                # 그래디언트 계산
                grad_x, grad_params = torch.autograd.grad(y_req, (x_req, p_req), grad_outputs=grad_out)
        return grad_x, grad_params, None, None
# 메인 API 함수
def hyper_butterfly(x: torch.Tensor, params: torch.Tensor, c: float, L: int):
    return HyperButterflyFunction.apply(x, params, c, L)

# 기본 연산들도 직접 노출
def log_map(x: torch.Tensor, c: float) -> torch.Tensor:
    """포앵카레 볼에서의 로그 맵 (접공간으로 매핑)"""
    if x.is_cuda and _has_cuda:
        return log_map_cuda(x, c)
    else:
        return log_map_cpu(x, c)

def exp_map(x: torch.Tensor, c: float) -> torch.Tensor:
    """포앵카레 볼에서의 지수 맵 (다양체로 매핑)"""
    if x.is_cuda and _has_cuda:
        return exp_map_cuda(x, c)
    else:
        return exp_map_cpu(x, c)

def butterfly_forward(x: torch.Tensor, params: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """단일 버터플라이 레이어 적용"""
    batch_size, dim = x.shape
    if x.is_cuda and _has_cuda:
        return butterfly_forward_cuda(x, params, layer_idx, batch_size, dim)
    else:
        return butterfly_forward_cpu(x, params, layer_idx, batch_size, dim)

# 모듈 내보내기
__all__ = [
    'hyper_butterfly',     # 주요 API 함수
    'log_map', 'exp_map',  # 기본 맵핑 함수
    'butterfly_forward',   # 단일 버터플라이 레이어
]