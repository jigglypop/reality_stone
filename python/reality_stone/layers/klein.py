import torch
from torch import Tensor
from torch.autograd import Function
from .. import _rust, _has_cuda
import math
from .poincare import poincare_to_klein

class KleinLayer(Function):
    @staticmethod
    def forward(
        ctx,
        u: Tensor,
        v: Tensor,
        c: float = None,
        t: float = 0.5,
        kappas: Tensor = None,
        layer_idx: int = None,
        c_min: float = 0.1,
        c_max: float = 5.0,
    ) -> Tensor:
        ctx.t = t
        if kappas is not None and layer_idx is not None:
            ctx.use_dynamic = True
            ctx.layer_idx = layer_idx
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(u, v, kappas)
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
            if hasattr(_rust, "klein_layer_layerwise_cpu"):
                out_np, c_val = _rust.klein_layer_layerwise_cpu(
                    u.cpu().numpy(), v.cpu().numpy(), kappa_val, layer_idx, c_min, c_max, t
                )
                ctx.c_val = c_val
                return torch.from_numpy(out_np).to(u.device)
            sig = 1.0 / (1.0 + torch.exp(torch.tensor(-kappa_val)))
            c_val = c_min + (c_max - c_min) * sig.item()
            ctx.c_val = c_val
            out_np = _rust.klein_layer_forward(u.cpu().numpy(), v.cpu().numpy(), float(c_val), t)
            return torch.from_numpy(out_np).to(u.device)
        ctx.use_dynamic = False
        ctx.c = c if c is not None else 1.0
        ctx.save_for_backward(u, v)
        out_np = _rust.klein_layer_forward(u.cpu().numpy(), v.cpu().numpy(), float(ctx.c), t)
        return torch.from_numpy(out_np).to(u.device)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        t = ctx.t
        if getattr(ctx, "use_dynamic", False):
            u, v, kappas = ctx.saved_tensors
            c_val = getattr(ctx, "c_val", None)
            if c_val is None:
                layer_idx = ctx.layer_idx
                c_min = ctx.c_min
                c_max = ctx.c_max
                if kappas.dim() == 0:
                    kappa_val = kappas.item()
                else:
                    kappa_val = kappas[layer_idx].item()
                sig = 1.0 / (1.0 + torch.exp(torch.tensor(-kappa_val)))
                c_val = c_min + (c_max - c_min) * sig.item()
                ctx.c_val = c_val
            c = float(c_val)
        else:
            u, v = ctx.saved_tensors
            c = float(ctx.c)
        grad_u = grad_v = None
        
        if grad_output.is_cuda and _has_cuda:
            grad_u = torch.empty_like(u)
            grad_v = torch.empty_like(v)
            _rust.klein_ball_layer_backward_cuda(
                grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                grad_u.data_ptr(), grad_v.data_ptr(),
                float(c), t, u.shape[0], u.shape[1]
            )
        else:
            grad_u_np, grad_v_np = _rust.klein_ball_layer_backward_cpu(
                grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), float(c), t
            )
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
        if getattr(ctx, "use_dynamic", False):
            if kappas.dim() == 0:
                grad_kappas = torch.zeros_like(kappas)
            else:
                grad_kappas = torch.zeros_like(kappas)
            return grad_u, grad_v, None, None, grad_kappas, None, None, None
        return grad_u, grad_v, None, None, None, None, None, None

def klein_add(u: Tensor, v: Tensor, c: float) -> Tensor:
    result_np = _rust.klein_add(u.cpu().numpy(), v.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(u.device)

def klein_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:
    result_np = _rust.klein_scalar(x.cpu().numpy(), r, c)
    return torch.from_numpy(result_np).to(x.device)

def klein_distance(x: Tensor, y: Tensor, c: float) -> Tensor:
    if isinstance(c, Tensor):
        eps = 1e-7
        c_t = c
        x2 = (x * x).sum(dim=-1)
        y2 = (y * y).sum(dim=-1)
        xy = (x * y).sum(dim=-1)
        den = (1.0 - c_t * x2) * (1.0 - c_t * y2)
        den = den.clamp_min(eps)
        arg = (1.0 - c_t * xy) / torch.sqrt(den)
        arg = arg.clamp_min(1.0 + eps)
        return torch.acosh(arg) / torch.sqrt(c_t)
    c_f = float(c)
    if x.is_cuda and _has_cuda:
        output = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
        _rust.klein_distance_cuda(output.data_ptr(), x.data_ptr(), y.data_ptr(), c_f, x.shape[0], x.shape[1])
        return output
    result_np = _rust.klein_distance(x.cpu().numpy(), y.cpu().numpy(), c_f)
    return torch.from_numpy(result_np).to(x.device)

def klein_to_poincare(x: Tensor, c: float) -> Tensor:
    result_np = _rust.klein_to_poincare(x.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device)

def klein_to_lorentz(x: Tensor, c: float) -> Tensor:
    result_np = _rust.klein_to_lorentz(x.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device) 

class KleinFromPoincare(Function):
    @staticmethod
    def forward(ctx, x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
        if kappas is not None:
            ctx.use_dynamic = True
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(x, kappas)
            
            output_np, c_val = _rust.from_poincare_dynamic_cpu(
                x.cpu().numpy(), kappas.item(), c_min, c_max
            )
            ctx.c_val = c_val
            return torch.from_numpy(output_np).to(x.device)
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            # Delegate to poincare_to_klein for non-dynamic path
            output = poincare_to_klein(x, ctx.c)
            ctx.save_for_backward(x)
            return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.use_dynamic:
            x, kappas = ctx.saved_tensors
            grad_x_np, grad_kappa_val = _rust.from_poincare_dynamic_backward_cpu(
                grad_output.cpu().numpy(), x.cpu().numpy(), kappas.item(), ctx.c_min, ctx.c_max
            )
            grad_x = torch.from_numpy(grad_x_np).to(grad_output.device)
            grad_kappas = torch.tensor(grad_kappa_val, device=kappas.device)
            return grad_x, None, grad_kappas, None, None
        else:
            # VJP for non-dynamic version is not implemented yet.
            x, = ctx.saved_tensors
            grad_x = torch.zeros_like(x)
            return grad_x, None, None, None, None

def from_poincare(x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
    return KleinFromPoincare.apply(x, c, kappas, c_min, c_max) 


def project_to_klein(x: Tensor, c: float | Tensor, epsilon: float = 1e-5) -> Tensor:
    if isinstance(c, Tensor):
        radius = torch.rsqrt(c).clamp(min=epsilon)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = radius - float(epsilon)
        scale = torch.where(norm > max_norm, max_norm / norm.clamp_min(epsilon), torch.ones_like(norm))
        return x * scale
    radius = (1.0 / math.sqrt(c)) if c > 0 else 1.0
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    max_norm = radius - epsilon
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale