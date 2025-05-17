import torch
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

# import torch
# from torch.autograd import Function
# 
# from ._C import (
#     poincare_ball_forward_cpu,
#     lorentz_forward_cpu,
#     klein_forward_cpu,
#     mobius_add_cpu,
#     mobius_scalar_cpu,
# )
# 
# _has_cuda = False
# if torch.cuda.is_available():
#     print("CUDA is available")
#     try:
#         from ._C import (
#             poincare_ball_forward_cuda,
#             lorentz_forward_cuda,
#             klein_forward_cuda,
#             mobius_add_cuda,
#             mobius_scalar_cuda,
#         )
#         _has_cuda = True
#     except ImportError:
#         _has_cuda = False
# 
# class PoincareBall(Function):
#     @staticmethod
#     def forward(ctx, u, v, c, t):
#         ctx.save_for_backward(u, v)
#         ctx.c = c
#         ctx.t = t
#         
#         # forward만 수행하고 requires_grad 처리
#         with torch.enable_grad():
#             if u.is_cuda and _has_cuda:
#                 result = poincare_ball_forward_cuda(u, v, c, t)
#             else:
#                 result = poincare_ball_forward_cpu(u, v, c, t)
#         return result
# 
#     @staticmethod
#     def backward(ctx, grad_output):
#         u, v = ctx.saved_tensors
#         c, t = ctx.c, ctx.t
#         
#         # autograd를 사용하여 backprop 계산
#         with torch.enable_grad():
#             u_new = u.detach().requires_grad_(True)
#             v_new = v.detach().requires_grad_(True)
#             
#             if u_new.is_cuda and _has_cuda:
#                 output = poincare_ball_forward_cuda(u_new, v_new, c, t)
#             else:
#                 output = poincare_ball_forward_cpu(u_new, v_new, c, t)
#             
#             grad_u, grad_v = torch.autograd.grad(
#                 output, 
#                 [u_new, v_new], 
#                 grad_outputs=grad_output,
#                 retain_graph=True
#             )
#         return grad_u, grad_v, None, None
# 
# class LorentzModel(Function):
#     @staticmethod
#     def forward(ctx, u, v, c, t):
#         ctx.save_for_backward(u, v)
#         ctx.c = c
#         ctx.t = t
#         
#         # forward만 수행하고 requires_grad 처리
#         with torch.enable_grad():
#             if u.is_cuda and _has_cuda:
#                 result = lorentz_forward_cuda(u, v, c, t)
#             else:
#                 result = lorentz_forward_cpu(u, v, c, t)
#         return result
# 
#     @staticmethod
#     def backward(ctx, grad_output):
#         u, v = ctx.saved_tensors
#         c, t = ctx.c, ctx.t
#         
#         # autograd를 사용하여 backprop 계산
#         with torch.enable_grad():
#             u_new = u.detach().requires_grad_(True)
#             v_new = v.detach().requires_grad_(True)
#             
#             if u_new.is_cuda and _has_cuda:
#                 output = lorentz_forward_cuda(u_new, v_new, c, t)
#             else:
#                 output = lorentz_forward_cpu(u_new, v_new, c, t)
#             
#             grad_u, grad_v = torch.autograd.grad(
#                 output, 
#                 [u_new, v_new], 
#                 grad_outputs=grad_output,
#                 retain_graph=True
#             )
#         return grad_u, grad_v, None, None
# 
# class KleinModel(Function):
#     @staticmethod
#     def forward(ctx, u, v, c, t):
#         ctx.save_for_backward(u, v)
#         ctx.c = c
#         ctx.t = t
#         
#         # forward만 수행하고 requires_grad 처리
#         with torch.enable_grad():
#             if u.is_cuda and _has_cuda:
#                 result = klein_forward_cuda(u, v, c, t)
#             else:
#                 result = klein_forward_cpu(u, v, c, t)
#         return result
# 
#     @staticmethod
#     def backward(ctx, grad_output):
#         u, v = ctx.saved_tensors
#         c, t = ctx.c, ctx.t
#         
#         # autograd를 사용하여 backprop 계산
#         with torch.enable_grad():
#             u_new = u.detach().requires_grad_(True)
#             v_new = v.detach().requires_grad_(True)
#             
#             if u_new.is_cuda and _has_cuda:
#                 output = klein_forward_cuda(u_new, v_new, c, t)
#             else:
#                 output = klein_forward_cpu(u_new, v_new, c, t)
#             
#             grad_u, grad_v = torch.autograd.grad(
#                 output, 
#                 [u_new, v_new], 
#                 grad_outputs=grad_output,
#                 retain_graph=True
#             )
#         return grad_u, grad_v, None, None
# 
# # Python API 함수들
# def poincare_ball_layer(u, v, c, t):
#     return PoincareBall.apply(u, v, c, t)
# 
# def lorentz_layer(u, v, c, t):
#     return LorentzModel.apply(u, v, c, t)
# 
# def klein_layer(u, v, c, t):
#     return KleinModel.apply(u, v, c, t)
# 
# # 모델 변환 API 추가
# from ._C import (
#     poincare_to_lorentz_cpu, lorentz_to_poincare_cpu,
#     poincare_to_klein_cpu, klein_to_poincare_cpu,
#     lorentz_to_klein_cpu, klein_to_lorentz_cpu
# )
# 
# if _has_cuda:
#     from ._C import (
#         poincare_to_lorentz_cuda, lorentz_to_poincare_cuda,
#         poincare_to_klein_cuda, klein_to_poincare_cuda,
#         lorentz_to_klein_cuda, klein_to_lorentz_cuda
#     )
# 
# def poincare_to_lorentz(x, c):
#     if x.is_cuda and _has_cuda:
#         return poincare_to_lorentz_cuda(x, c)
#     else:
#         return poincare_to_lorentz_cpu(x, c)
# 
# def lorentz_to_poincare(x, c):
#     if x.is_cuda and _has_cuda:
#         return lorentz_to_poincare_cuda(x, c)
#     else:
#         return lorentz_to_poincare_cpu(x, c)
# 
# def poincare_to_klein(x, c):
#     if x.is_cuda and _has_cuda:
#         return poincare_to_klein_cuda(x, c)
#     else:
#         return poincare_to_klein_cpu(x, c)
# 
# def klein_to_poincare(x, c):
#     if x.is_cuda and _has_cuda:
#         return klein_to_poincare_cuda(x, c)
#     else:
#         return klein_to_poincare_cpu(x, c)
# 
# def lorentz_to_klein(x, c):
#     if x.is_cuda and _has_cuda:
#         return lorentz_to_klein_cuda(x, c)
#     else:
#         return lorentz_to_klein_cpu(x, c)
# 
# def klein_to_lorentz(x, c):
#     if x.is_cuda and _has_cuda:
#         return klein_to_lorentz_cuda(x, c)
#     else:
#         return klein_to_lorentz_cpu(x, c)
# 
# 
# def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
#     if x.is_cuda and _has_cuda:
#         return mobius_add_cuda(x, y, c)
#     else:
#         return mobius_add_cpu(x, y, c)
# 
# def mobius_scalar(x: torch.Tensor, r: float, c: float) -> torch.Tensor:
#     if x.is_cuda and _has_cuda:
#         return mobius_scalar_cuda(x, r, c)
#     else:
#         return mobius_scalar_cpu(x, r, c)