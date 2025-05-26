import torch
import numpy as np
from torch.autograd import Function

from ._C import (
    poincare_ball_forward_cpu, poincare_ball_backward_cpu,
    lorentz_forward_cpu, lorentz_backward_cpu,
    klein_forward_cpu, klein_backward_cpu,
    mobius_add_cpu, mobius_scalar_cpu,
    poincare_to_lorentz_cpu, lorentz_to_poincare_cpu,
    poincare_to_klein_cpu, klein_to_poincare_cpu,
    lorentz_to_klein_cpu, klein_to_lorentz_cpu,
    chebyshev_approximation_cpu,
    chebyshev_distance_cpu,
    hyperbolic_laplacian_cpu,
    heat_kernel_cpu,
    hyperbolic_fft_cpu,
    spherical_harmonics_cpu,
)

_has_cuda = False
if torch.cuda.is_available():
    try:
        from ._C import (
            poincare_ball_forward_cuda, poincare_ball_backward_cuda,
            lorentz_forward_cuda, lorentz_backward_cuda,
            klein_forward_cuda, klein_backward_cuda,
            mobius_add_cuda, mobius_scalar_cuda,
            poincare_to_lorentz_cuda, lorentz_to_poincare_cuda,
            poincare_to_klein_cuda, klein_to_poincare_cuda,
            lorentz_to_klein_cuda, klein_to_lorentz_cuda,
            chebyshev_approximation_cuda,
            chebyshev_distance_cuda,
            hyperbolic_laplacian_cuda,
            heat_kernel_cuda,
            hyperbolic_fft_cuda,
            spherical_harmonics_cuda,
        )
        _has_cuda = True
        print("🚀 Reality Stone: CUDA Advanced Features Available")
    except ImportError as e:
        _has_cuda = False
        print(f"⚠️ Reality Stone: CUDA features not available: {e}")

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

def poincare_ball_layer(u, v, c, t):
    return PoincareBall.apply(u, v, c, t)

def lorentz_layer(u, v, c, t):
    return LorentzModel.apply(u, v, c, t)

def klein_layer(u, v, c, t):
    return KleinModel.apply(u, v, c, t)

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

def chebyshev_approximation(x, order=15, curvature=1.0):
    fn = chebyshev_approximation_cuda if (x.is_cuda and _has_cuda) else chebyshev_approximation_cpu
    return fn(x, order, curvature)

def chebyshev_distance(x, y, curvature=1.0):
    fn = chebyshev_distance_cuda if (x.is_cuda and _has_cuda) else chebyshev_distance_cpu
    return fn(x, y, curvature)

def hyperbolic_laplacian(f, curvature=1.0):
    fn = hyperbolic_laplacian_cuda if (f.is_cuda and _has_cuda) else hyperbolic_laplacian_cpu
    return fn(f, curvature)

def heat_kernel(x, t, curvature=1.0):
    fn = heat_kernel_cuda if (x.is_cuda and _has_cuda) else heat_kernel_cpu
    return fn(x, t, curvature)

def hyperbolic_fft(x, curvature=1.0):
    fn = hyperbolic_fft_cuda if (x.is_cuda and _has_cuda) else hyperbolic_fft_cpu
    return fn(x, curvature)

def spherical_harmonics(theta_phi, l_max=10):
    fn = spherical_harmonics_cuda if (theta_phi.is_cuda and _has_cuda) else spherical_harmonics_cpu
    return fn(theta_phi, l_max)

def predict_dynamic_curvature(features, weight, bias, base_curvature=1.0, 
                             min_curvature=1e-6, max_curvature=1e6):
    try:
        if _has_cuda and hasattr(_C, 'dynamic_curvature_prediction_cuda') and features.is_cuda:
            from ._C import dynamic_curvature_prediction_cuda
            return dynamic_curvature_prediction_cuda(features, weight, bias, base_curvature, 
                                                   min_curvature, max_curvature)
        else:
            logits = torch.mm(features, weight.t()) + bias
            curvatures = base_curvature * torch.exp(torch.clamp(logits, -20.0, 20.0))
            return torch.clamp(curvatures, min=min_curvature, max=max_curvature).squeeze()
    except Exception:
        logits = torch.mm(features, weight.t()) + bias
        curvatures = base_curvature * torch.exp(torch.clamp(logits, -20.0, 20.0))
        return torch.clamp(curvatures, min=min_curvature, max=max_curvature).squeeze()

def dynamic_curvature_pred(features, weight, bias, base_curvature=1.0):
    return predict_dynamic_curvature(features, weight, bias, base_curvature)

def dynamic_mobius_add(u, v, curvatures):
    try:
        if _has_cuda and hasattr(_C, 'dynamic_mobius_add_cuda') and u.is_cuda:
            from ._C import dynamic_mobius_add_cuda
            return dynamic_mobius_add_cuda(u, v, curvatures)
        else:
            batch_size = u.size(0)
            result = torch.zeros_like(u)
            for b in range(batch_size):
                c_b = max(1e-6, min(curvatures[b].item(), 1e6))
                result[b] = mobius_add(u[b:b+1], v[b:b+1], c_b)[0]
            return result
    except Exception:
        batch_size = u.size(0)
        result = torch.zeros_like(u)
        for b in range(batch_size):
            c_b = max(1e-6, min(curvatures[b].item(), 1e6))
            result[b] = mobius_add(u[b:b+1], v[b:b+1], c_b)[0]
        return result

def dynamic_poincare_layer(u, v, curvatures, t=0.5):
    try:
        if _has_cuda and hasattr(_C, 'dynamic_poincare_layer_cuda') and u.is_cuda:
            from ._C import dynamic_poincare_layer_cuda
            return dynamic_poincare_layer_cuda(u, v, curvatures, t)
        else:
            tv = t * v
            one_minus_t_u = (1.0 - t) * u
            return dynamic_mobius_add(one_minus_t_u, tv, curvatures)
    except Exception:
        tv = t * v
        one_minus_t_u = (1.0 - t) * u
        return dynamic_mobius_add(one_minus_t_u, tv, curvatures)

def boundary_penalty(x, curvature, epsilon=0.01):
    norm = torch.norm(x, 2, dim=-1)
    max_norm = 1.0 / torch.sqrt(torch.tensor(curvature)) - epsilon
    violation = torch.relu(norm - max_norm)
    return torch.mean(violation * violation)