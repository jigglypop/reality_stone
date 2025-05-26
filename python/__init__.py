import torch
import numpy as np
from torch.autograd import Function

from ._C import (
    poincare_ball_forward_cpu, poincare_ball_backward_cpu,
    lorentz_forward_cpu,     lorentz_backward_cpu,
    klein_forward_cpu,       klein_backward_cpu,
    mobius_add_cpu,          mobius_scalar_cpu,
)

_has_cuda = False
if torch.cuda.is_available():
    try:
        from ._C import (
            poincare_ball_forward_cuda, poincare_ball_backward_cuda,
            lorentz_forward_cuda,     lorentz_backward_cuda,
            klein_forward_cuda,       klein_backward_cuda,
            mobius_add_cuda,          mobius_scalar_cuda,
        )
        _has_cuda = True
        print("CUDA is available")
    except ImportError:
        _has_cuda = False


class PoincareBall(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c, ctx.t = c, t
        if u.is_cuda and _has_cuda:
            return poincare_ball_forward_cuda(u, v, c, t)
        else:
            return poincare_ball_forward_cpu(u, v, c, t)

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        if u.is_cuda and _has_cuda:
            grad_u, grad_v = poincare_ball_backward_cuda(grad_output, u, v, c, t)
        else:
            grad_u, grad_v = poincare_ball_backward_cpu(grad_output, u, v, c, t)
        return grad_u, grad_v, None, None


class LorentzModel(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c, ctx.t = c, t
        if u.is_cuda and _has_cuda:
            return lorentz_forward_cuda(u, v, c, t)
        else:
            return lorentz_forward_cpu(u, v, c, t)

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        if u.is_cuda and _has_cuda:
            grad_u, grad_v = lorentz_backward_cuda(grad_output, u, v, c, t)
        else:
            grad_u, grad_v = lorentz_backward_cpu(grad_output, u, v, c, t)
        return grad_u, grad_v, None, None


class KleinModel(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c, ctx.t = c, t
        if u.is_cuda and _has_cuda:
            return klein_forward_cuda(u, v, c, t)
        else:
            return klein_forward_cpu(u, v, c, t)

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        if u.is_cuda and _has_cuda:
            grad_u, grad_v = klein_backward_cuda(grad_output, u, v, c, t)
        else:
            grad_u, grad_v = klein_backward_cpu(grad_output, u, v, c, t)
        return grad_u, grad_v, None, None


# Python API
def poincare_ball_layer(u, v, c, t):
    return PoincareBall.apply(u, v, c, t)

def lorentz_layer(u, v, c, t):
    return LorentzModel.apply(u, v, c, t)

def klein_layer(u, v, c, t):
    return KleinModel.apply(u, v, c, t)


# conversions
from ._C import (
    poincare_to_lorentz_cpu, lorentz_to_poincare_cpu,
    poincare_to_klein_cpu,   klein_to_poincare_cpu,
    lorentz_to_klein_cpu,    klein_to_lorentz_cpu,
)
if _has_cuda:
    from ._C import (
        poincare_to_lorentz_cuda, lorentz_to_poincare_cuda,
        poincare_to_klein_cuda,   klein_to_poincare_cuda,
        lorentz_to_klein_cuda,    klein_to_lorentz_cuda,
    )

def poincare_to_lorentz(x, c):
    fn = poincare_to_lorentz_cuda if (x.is_cuda and _has_cuda) else poincare_to_lorentz_cpu
    return fn(x, c)

def lorentz_to_poincare(x, c):
    fn = lorentz_to_poincare_cuda if (x.is_cuda and _has_cuda) else lorentz_to_poincare_cpu
    return fn(x, c)

def poincare_to_klein(x, c):
    fn = poincare_to_klein_cuda if (x.is_cuda and _has_cuda) else poincare_to_klein_cpu
    return fn(x, c)

def klein_to_poincare(x, c):
    fn = klein_to_poincare_cuda if (x.is_cuda and _has_cuda) else klein_to_poincare_cpu
    return fn(x, c)

def lorentz_to_klein(x, c):
    fn = lorentz_to_klein_cuda if (x.is_cuda and _has_cuda) else lorentz_to_klein_cpu
    return fn(x, c)

def klein_to_lorentz(x, c):
    fn = klein_to_lorentz_cuda if (x.is_cuda and _has_cuda) else klein_to_lorentz_cpu
    return fn(x, c)


def mobius_add(x, y, c):
    fn = mobius_add_cuda if (x.is_cuda and _has_cuda) else mobius_add_cpu
    return fn(x, y, c)

def mobius_scalar(x, r, c):
    fn = mobius_scalar_cuda if (x.is_cuda and _has_cuda) else mobius_scalar_cpu
    return fn(x, r, c)



def chebyshev_approximation(x, order=10, curvature=1.0):
    """체비셰프 다항식을 이용한 하이퍼볼릭 함수 근사"""
    # PyTorch fallback 구현
    x_clamped = torch.clamp(x, -0.99, 0.99)
    result = torch.zeros_like(x)
    # tanh(√c * x)의 체비셰프 급수 전개 (홀수 항만)
    for n in range(1, order + 1, 2):
        T_n = torch.cos(n * torch.acos(x_clamped))
        coeff = pow(-1, (n-1)//2) * 4.0 / (np.pi * (n*n - 0.25))
        result += coeff * T_n
    
    return torch.clamp(result, -10.0, 10.0)

def predict_dynamic_curvature(features, weight, bias, base_curvature):
    """동적 곡률 예측"""
    logits = torch.mm(features, weight.t()) + bias
    normalized = torch.sigmoid(logits)
    return base_curvature * normalized.squeeze()

def dynamic_mobius_add(u, v, curvatures):
    """동적 곡률을 사용한 Möbius 덧셈"""
    batch_size = u.size(0)
    result = torch.zeros_like(u)
    
    for b in range(batch_size):
        u_b = u[b]
        v_b = v[b] 
        c = curvatures[b].item()
        
        u2 = torch.sum(u_b * u_b)
        v2 = torch.sum(v_b * v_b)
        uv = torch.sum(u_b * v_b)
        
        c2 = c * c
        denom = 1.0 + 2.0 * c * uv + c2 * u2 * v2
        denom = max(denom, 1e-6)
        
        num_u = (1.0 + 2.0 * c * uv + c * v2) * u_b
        num_v = (1.0 - c * u2) * v_b
        
        result[b] = (num_u + num_v) / denom
    
    return result