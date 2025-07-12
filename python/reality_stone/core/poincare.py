import torch
from torch import Tensor
from torch.autograd import Function
from .. import _rust, _has_cuda

class PoincareBallLayer(Function):
    """
    Custom autograd Function for a layer operating in the Poincare ball model.
    This layer computes a hyperbolic weighted combination: t * u + (1-t) * v,
    where '+' and '*' are Mobius addition and scalar multiplication.
    """
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float, t: float) -> Tensor:
        ctx.c = c
        ctx.t = t
        
        u_prime = poincare_scalar_mul(u, 1.0 - t, c)
        v_prime = poincare_scalar_mul(v, t, c)
        output = poincare_add(u_prime, v_prime, c)
        
        ctx.save_for_backward(u.clone(), v.clone(), u_prime.clone(), v_prime.clone())
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, Tensor | None, None, None]:
        # This backward implementation is now partially redundant if MobiusAdd and 
        # MobiusScalarMul have their own correct backward passes.
        # However, we keep it for now as they are not yet implemented.
        u, v, u_prime, v_prime = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        grad_u = grad_v = None
        if grad_output.is_cuda and _has_cuda:
            grad_u = torch.empty_like(u)
            grad_v = torch.empty_like(v)
            _rust.poincare_ball_layer_backward_cuda(
                grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                grad_u.data_ptr(), grad_v.data_ptr(),
                c, t, u.shape[0], u.shape[1]
            )
        else:
            grad_u_np, grad_v_np = _rust.poincare_ball_layer_backward_cpu(
                grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t
            )
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
        return grad_u, grad_v, None, None

class MobiusAdd(Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, c: float) -> Tensor:
        ctx.c = c
        ctx.save_for_backward(x, y)
        if x.is_cuda and _has_cuda:
            output = torch.empty_like(x)
            _rust.mobius_add_cuda(x.data_ptr(), y.data_ptr(), output.data_ptr(), x.shape[0], x.shape[1], c)
            return output
        return torch.from_numpy(_rust.mobius_add_cpu(x.cpu().numpy(), y.cpu().numpy(), c))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, y = ctx.saved_tensors
        c = ctx.c
        grad_x_np, grad_y_np = _rust.mobius_add_vjp_cpu(
            grad_output.cpu().numpy(), x.cpu().numpy(), y.cpu().numpy(), c
        )
        grad_x = torch.from_numpy(grad_x_np).to(grad_output.device)
        grad_y = torch.from_numpy(grad_y_np).to(grad_output.device)
        return grad_x, grad_y, None

class MobiusScalarMul(Function):
    @staticmethod
    def forward(ctx, x: Tensor, r: float, c: float) -> Tensor:
        ctx.r = r
        ctx.c = c
        ctx.save_for_backward(x)
        if x.is_cuda and _has_cuda:
            output = torch.empty_like(x)
            _rust.mobius_scalar_cuda(x.data_ptr(), output.data_ptr(), x.shape[0], x.shape[1], r, c)
            return output
        return torch.from_numpy(_rust.mobius_scalar_cpu(x.cpu().numpy(), r, c))
        
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, = ctx.saved_tensors
        r, c = ctx.r, ctx.c
        grad_x_np = _rust.mobius_scalar_vjp_cpu(
            grad_output.cpu().numpy(), x.cpu().numpy(), c, r
        )
        grad_x = torch.from_numpy(grad_x_np).to(grad_output.device)
        return grad_x, None, None


def poincare_add(x: Tensor, y: Tensor, c: float) -> Tensor:
    """Performs Mobius addition of two tensors in the Poincare ball."""
    return MobiusAdd.apply(x, y, c)

def poincare_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:
    """Performs Mobius scalar multiplication in the Poincare ball."""
    return MobiusScalarMul.apply(x, r, c)

def poincare_distance(x: Tensor, y: Tensor, c: float) -> Tensor:
    """Computes the Poincare distance between two tensors."""
    if x.is_cuda and _has_cuda:
        output = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
        _rust.poincare_distance_cuda(x.data_ptr(), y.data_ptr(), output.data_ptr(), x.shape[0], x.shape[1], c)
        return output
    return torch.from_numpy(_rust.poincare_distance_cpu(x.cpu().numpy(), y.cpu().numpy(), c)) 