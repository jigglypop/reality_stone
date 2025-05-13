import torch
from torch.autograd import Function

from ._C import (
    poincare_ball_forward_cpu,
)
_has_cuda = False
if torch.cuda.is_available():
    print("CUDA is available")
    try:
        from ._C import (
            poincare_ball_forward_cuda,
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

def poincare_ball_layer(u, v, c, t):
    return PoincareBall.apply(u, v, c, t)
