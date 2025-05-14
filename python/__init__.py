import torch
from torch.autograd import Function

from ._C import (
    poincare_ball_forward_cpu,
    lorentz_forward_cpu,
    klein_forward_cpu,
)

_has_cuda = False
if torch.cuda.is_available():
    print("CUDA is available")
    try:
        from ._C import (
            poincare_ball_forward_cuda,
            lorentz_forward_cuda,
            klein_forward_cuda,
        )
        _has_cuda = True
    except ImportError:
        _has_cuda = False

class PoincareBall(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c = c
        ctx.t = t
        
        # forward만 수행하고 requires_grad 처리
        with torch.enable_grad():
            if u.is_cuda and _has_cuda:
                result = poincare_ball_forward_cuda(u, v, c, t)
            else:
                result = poincare_ball_forward_cpu(u, v, c, t)
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
                output = poincare_ball_forward_cuda(u_new, v_new, c, t)
            else:
                output = poincare_ball_forward_cpu(u_new, v_new, c, t)
            
            grad_u, grad_v = torch.autograd.grad(
                output, 
                [u_new, v_new], 
                grad_outputs=grad_output,
                retain_graph=True
            )
        return grad_u, grad_v, None, None

class LorentzModel(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c = c
        ctx.t = t
        
        # forward만 수행하고 requires_grad 처리
        with torch.enable_grad():
            if u.is_cuda and _has_cuda:
                result = lorentz_forward_cuda(u, v, c, t)
            else:
                result = lorentz_forward_cpu(u, v, c, t)
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
                output = lorentz_forward_cuda(u_new, v_new, c, t)
            else:
                output = lorentz_forward_cpu(u_new, v_new, c, t)
            
            grad_u, grad_v = torch.autograd.grad(
                output, 
                [u_new, v_new], 
                grad_outputs=grad_output,
                retain_graph=True
            )
        return grad_u, grad_v, None, None

class KleinModel(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.save_for_backward(u, v)
        ctx.c = c
        ctx.t = t
        
        # forward만 수행하고 requires_grad 처리
        with torch.enable_grad():
            if u.is_cuda and _has_cuda:
                result = klein_forward_cuda(u, v, c, t)
            else:
                result = klein_forward_cpu(u, v, c, t)
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
                output = klein_forward_cuda(u_new, v_new, c, t)
            else:
                output = klein_forward_cpu(u_new, v_new, c, t)
            
            grad_u, grad_v = torch.autograd.grad(
                output, 
                [u_new, v_new], 
                grad_outputs=grad_output,
                retain_graph=True
            )
        return grad_u, grad_v, None, None

# Python API 함수들
def poincare_ball_layer(u, v, c, t):
    """푸앵카레 볼 모델 레이어

    Args:
        u (torch.Tensor): 입력 텐서 (B, D)
        v (torch.Tensor): 입력 텐서 (B, D)
        c (float): 곡률 파라미터 (보통 양수 작은 값)
        t (float): 측지선 파라미터 (0 <= t <= 1)

    Returns:
        torch.Tensor: 결과 텐서 (B, D)
    """
    return PoincareBall.apply(u, v, c, t)

def lorentz_layer(u, v, c, t):
    """로렌츠 모델 레이어

    Args:
        u (torch.Tensor): 입력 텐서 (B, D+1)
        v (torch.Tensor): 입력 텐서 (B, D+1)
        c (float): 곡률 파라미터 (보통 양수 작은 값)
        t (float): 측지선 파라미터 (0 <= t <= 1)

    Returns:
        torch.Tensor: 결과 텐서 (B, D+1)
    """
    return LorentzModel.apply(u, v, c, t)

def klein_layer(u, v, c, t):
    """클라인 모델 레이어

    Args:
        u (torch.Tensor): 입력 텐서 (B, D)
        v (torch.Tensor): 입력 텐서 (B, D)
        c (float): 곡률 파라미터 (보통 양수 작은 값)
        t (float): 측지선 파라미터 (0 <= t <= 1)

    Returns:
        torch.Tensor: 결과 텐서 (B, D)
    """
    return KleinModel.apply(u, v, c, t)

# 모델 변환 API 추가
from ._C import (
    poincare_to_lorentz_cpu, lorentz_to_poincare_cpu,
    poincare_to_klein_cpu, klein_to_poincare_cpu,
    lorentz_to_klein_cpu, klein_to_lorentz_cpu
)

if _has_cuda:
    from ._C import (
        poincare_to_lorentz_cuda, lorentz_to_poincare_cuda,
        poincare_to_klein_cuda, klein_to_poincare_cuda,
        lorentz_to_klein_cuda, klein_to_lorentz_cuda
    )

def poincare_to_lorentz(x, c):
    """푸앵카레 볼 모델에서 로렌츠 모델로 변환

    Args:
        x (torch.Tensor): 푸앵카레 좌표 (B, D)
        c (float): 곡률 파라미터

    Returns:
        torch.Tensor: 로렌츠 좌표 (B, D+1)
    """
    if x.is_cuda and _has_cuda:
        return poincare_to_lorentz_cuda(x, c)
    else:
        return poincare_to_lorentz_cpu(x, c)

def lorentz_to_poincare(x, c):
    """로렌츠 모델에서 푸앵카레 볼 모델로 변환

    Args:
        x (torch.Tensor): 로렌츠 좌표 (B, D+1)
        c (float): 곡률 파라미터

    Returns:
        torch.Tensor: 푸앵카레 좌표 (B, D)
    """
    if x.is_cuda and _has_cuda:
        return lorentz_to_poincare_cuda(x, c)
    else:
        return lorentz_to_poincare_cpu(x, c)

def poincare_to_klein(x, c):
    """푸앵카레 볼 모델에서 클라인 모델로 변환

    Args:
        x (torch.Tensor): 푸앵카레 좌표 (B, D)
        c (float): 곡률 파라미터

    Returns:
        torch.Tensor: 클라인 좌표 (B, D)
    """
    if x.is_cuda and _has_cuda:
        return poincare_to_klein_cuda(x, c)
    else:
        return poincare_to_klein_cpu(x, c)

def klein_to_poincare(x, c):
    """클라인 모델에서 푸앵카레 볼 모델로 변환

    Args:
        x (torch.Tensor): 클라인 좌표 (B, D)
        c (float): 곡률 파라미터

    Returns:
        torch.Tensor: 푸앵카레 좌표 (B, D)
    """
    if x.is_cuda and _has_cuda:
        return klein_to_poincare_cuda(x, c)
    else:
        return klein_to_poincare_cpu(x, c)

def lorentz_to_klein(x, c):
    """로렌츠 모델에서 클라인 모델로 변환

    Args:
        x (torch.Tensor): 로렌츠 좌표 (B, D+1)
        c (float): 곡률 파라미터

    Returns:
        torch.Tensor: 클라인 좌표 (B, D)
    """
    if x.is_cuda and _has_cuda:
        return lorentz_to_klein_cuda(x, c)
    else:
        return lorentz_to_klein_cpu(x, c)

def klein_to_lorentz(x, c):
    """클라인 모델에서 로렌츠 모델로 변환

    Args:
        x (torch.Tensor): 클라인 좌표 (B, D)
        c (float): 곡률 파라미터

    Returns:
        torch.Tensor: 로렌츠 좌표 (B, D+1)
    """
    if x.is_cuda and _has_cuda:
        return klein_to_lorentz_cuda(x, c)
    else:
        return klein_to_lorentz_cpu(x, c)
# # import torch
# # from torch.autograd import Function
# # 
# # from ._C import (
# #     poincare_ball_forward_cpu,
# # )
# # _has_cuda = False
# # if torch.cuda.is_available():
# #     print("CUDA is available")
# #     try:
# #         from ._C import (
# #             poincare_ball_forward_cuda,
# #         )
# #         _has_cuda = True
# #     except ImportError:
# #         _has_cuda = False
# # 
# # class PoincareBall(Function):
# #     @staticmethod
# #     def forward(ctx, u, v, c, t):
# #         ctx.save_for_backward(u, v)
# #         ctx.c = c
# #         ctx.t = t
# #         
# #         # forward만 수행하고 requires_grad 처리
# #         with torch.enable_grad():
# #             if u.is_cuda and _has_cuda:
# #                 result = poincare_ball_forward_cuda(u, v, c, t)
# #             else:
# #                 result = poincare_ball_forward_cpu(u, v, c, t)
# #         return result
# # 
# #     @staticmethod
# #     def backward(ctx, grad_output):
# #         u, v = ctx.saved_tensors
# #         c, t = ctx.c, ctx.t
# #         
# #         # autograd를 사용하여 backprop 계산
# #         with torch.enable_grad():
# #             u_new = u.detach().requires_grad_(True)
# #             v_new = v.detach().requires_grad_(True)
# #             
# #             if u_new.is_cuda and _has_cuda:
# #                 output = poincare_ball_forward_cuda(u_new, v_new, c, t)
# #             else:
# #                 output = poincare_ball_forward_cpu(u_new, v_new, c, t)
# #             
# #             grad_u, grad_v = torch.autograd.grad(
# #                 output, 
# #                 [u_new, v_new], 
# #                 grad_outputs=grad_output,
# #                 retain_graph=True
# #             )
# #         return grad_u, grad_v, None, None
# # 
# # def poincare_ball_layer(u, v, c, t):
# #     return PoincareBall.apply(u, v, c, t)
# import torch
# from torch.autograd import Function
# 
# from ._C import (
#     poincare_ball_forward_cpu,
#     lorentz_forward_cpu,
# )
# 
# _has_cuda = False
# if torch.cuda.is_available():
#     print("CUDA is available")
#     try:
#         from ._C import (
#             poincare_ball_forward_cuda,
#             lorentz_forward_cuda,
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
#         with torch.enable_grad():
#             u_new = u.detach().requires_grad_(True)
#             v_new = v.detach().requires_grad_(True)
#             
#             if u_new.is_cuda and _has_cuda:
#                 output = poincare_ball_forward_cuda(u_new, v_new, c, t)
#             else:
#                 output = poincare_ball_forward_cpu(u_new, v_new, c, t)
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
# # class KleinModel(Function):
# #     @staticmethod
# #     def forward(ctx, u, v, c, t):
# #         ctx.save_for_backward(u, v)
# #         ctx.c = c
# #         ctx.t = t
# #         with torch.enable_grad():
# #             if u.is_cuda and _has_cuda:
# #                 result = klein_forward_cuda(u, v, c, t)
# #             else:
# #                 result = klein_forward_cpu(u, v, c, t)
# #         return result
# # 
# #     @staticmethod
# #     def backward(ctx, grad_output):
# #         u, v = ctx.saved_tensors
# #         c, t = ctx.c, ctx.t
# #         with torch.enable_grad():
# #             u_new = u.detach().requires_grad_(True)
# #             v_new = v.detach().requires_grad_(True)
# #             if u_new.is_cuda and _has_cuda:
# #                 output = klein_forward_cuda(u_new, v_new, c, t)
# #             else:
# #                 output = klein_forward_cpu(u_new, v_new, c, t)
# #             grad_u, grad_v = torch.autograd.grad(
# #                 output, 
# #                 [u_new, v_new], 
# #                 grad_outputs=grad_output,
# #                 retain_graph=True
# #             )
# #         return grad_u, grad_v, None, None
# 
# def poincare_ball_layer(u, v, c, t):
#     return PoincareBall.apply(u, v, c, t)
# 
# def lorentz_layer(u, v, c, t):
#     return LorentzModel.apply(u, v, c, t)
# 
# # 모델 변환 API 추가
# from ._C import (
#     poincare_to_lorentz_cpu, lorentz_to_poincare_cpu,
# )
# 
# if _has_cuda:
#     from ._C import (
#         poincare_to_lorentz_cuda, lorentz_to_poincare_cuda,
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