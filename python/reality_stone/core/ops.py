import torch
from torch.autograd import Function
from .. import _rust, _has_rust_ext, _has_cuda

class PoincareBallLayer(Function):
    @staticmethod
    def forward(ctx, u, v, c, t):
        ctx.c = c
        ctx.t = t
        if u.is_cuda and _has_cuda:
            output = torch.empty_like(u)
            _rust.poincare_ball_layer_cuda(
                u.data_ptr(),
                v.data_ptr(),
                output.data_ptr(),
                u.shape[0],
                u.shape[1],
                c,
                t
            )
            ctx.save_for_backward(u.clone(), v.clone())
        else:
            output = torch.from_numpy(_rust.poincare_ball_layer_cpu(u.cpu().numpy(), v.cpu().numpy(), c, t))
            ctx.save_for_backward(u.clone(), v.clone())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        grad_u = grad_v = None
        if grad_output.is_cuda and _has_cuda:
            grad_u = torch.empty_like(u)
            grad_v = torch.empty_like(v)
            _rust.poincare_ball_layer_backward_cuda(
                grad_output.data_ptr(),
                u.data_ptr(),
                v.data_ptr(),
                grad_u.data_ptr(),
                grad_v.data_ptr(),
                c,
                t,
                u.shape[0],
                u.shape[1]
            )
        else:
            grad_u_np, grad_v_np = _rust.poincare_ball_layer_backward_cpu(grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t)
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
        return grad_u, grad_v, None, None

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    if x.is_cuda and _has_cuda:
        output = torch.empty_like(x)
        _rust.mobius_add_cuda(x.data_ptr(), y.data_ptr(), output.data_ptr(), x.shape[0], x.shape[1], c)
        return output
    return torch.from_numpy(_rust.mobius_add_cpu(x.cpu().numpy(), y.cpu().numpy(), c))

def mobius_scalar(x: torch.Tensor, r: float, c: float) -> torch.Tensor:
    if x.is_cuda and _has_cuda:
        output = torch.empty_like(x)
        _rust.mobius_scalar_cuda(x.data_ptr(), output.data_ptr(), x.shape[0], x.shape[1], r, c)
        return output
    return torch.from_numpy(_rust.mobius_scalar_cpu(x.cpu().numpy(), r, c))

def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    if x.is_cuda and _has_cuda:
        output = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
        _rust.poincare_distance_cuda(x.data_ptr(), y.data_ptr(), output.data_ptr(), x.shape[0], x.shape[1], c)
        return output
    return torch.from_numpy(_rust.poincare_distance_cpu(x.cpu().numpy(), y.cpu().numpy(), c)) 