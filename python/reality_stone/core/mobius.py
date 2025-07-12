import torch
from torch import Tensor
from torch.autograd import Function
from .. import _rust, _has_cuda

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

