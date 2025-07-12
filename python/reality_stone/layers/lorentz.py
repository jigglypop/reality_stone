import torch
from torch import Tensor
from torch.autograd import Function
from .. import _rust, _has_cuda

class LorentzLayer(Function):
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float, t: float) -> Tensor:
        ctx.c = c
        ctx.t = t
        ctx.save_for_backward(u.clone(), v.clone())
        
        if u.is_cuda and _has_cuda:
            output = torch.empty_like(u)
            _rust.lorentz_layer_forward_cuda(
                output.data_ptr(), u.data_ptr(), v.data_ptr(),
                c, t, u.shape[0], u.shape[1]
            )
            return output
        else:
            output_np = _rust.lorentz_layer_forward(u.cpu().numpy(), v.cpu().numpy(), c, t)
            return torch.from_numpy(output_np).to(u.device)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, Tensor | None, None, None]:
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        grad_u = grad_v = None
        if grad_output.is_cuda and _has_cuda:
            grad_u = torch.empty_like(u)
            grad_v = torch.empty_like(v)
            _rust.lorentz_ball_layer_backward_cuda(
                grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                grad_u.data_ptr(), grad_v.data_ptr(),
                c, t, u.shape[0], u.shape[1]
            )
        else:
            grad_u_np, grad_v_np = _rust.lorentz_ball_layer_backward_cpu(
                grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t
            )
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
        return grad_u, grad_v, None, None

def lorentz_add(u: Tensor, v: Tensor, c: float) -> Tensor:
    result_np = _rust.lorentz_add(u.cpu().numpy(), v.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(u.device)

def lorentz_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:
    result_np = _rust.lorentz_scalar(x.cpu().numpy(), r, c)
    return torch.from_numpy(result_np).to(x.device)

def lorentz_distance(x: Tensor, y: Tensor, c: float) -> Tensor:
    if x.is_cuda and _has_cuda:
        output = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
        _rust.lorentz_distance_cuda(output.data_ptr(), x.data_ptr(), y.data_ptr(), c, x.shape[0], x.shape[1])
        return output
    result_np = _rust.lorentz_distance(x.cpu().numpy(), y.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device)

def lorentz_inner(u: Tensor, v: Tensor) -> Tensor:
    result_np = _rust.lorentz_inner(u.cpu().numpy(), v.cpu().numpy())
    return torch.from_numpy(result_np).to(u.device)

def lorentz_to_poincare(x: Tensor, c: float) -> Tensor:
    result_np = _rust.lorentz_to_poincare(x.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device)

def lorentz_to_klein(x: Tensor, c: float) -> Tensor:
    result_np = _rust.lorentz_to_klein(x.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device) 