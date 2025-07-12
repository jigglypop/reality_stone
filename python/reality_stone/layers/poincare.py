import torch
from torch import Tensor
from torch.autograd import Function
from .. import _rust, _has_cuda
from ..core.mobius import MobiusAdd, MobiusScalarMul

def project_to_ball(x: Tensor, epsilon: float = 1e-5) -> Tensor:
    output_np = _rust.poincare.project_to_ball_cpu(x.cpu().numpy(), epsilon)
    return torch.from_numpy(output_np).to(x.device)

class PoincareBallLayer(Function):

    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float = None, t: float = 0.5, kappas: Tensor = None, layer_idx: int = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
        ctx.t = t
        
        if kappas is not None and layer_idx is not None:
            ctx.use_dynamic = True
            ctx.layer_idx = layer_idx
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(u, v, kappas)
            
            # kappas가 0차원 텐서면 바로 item(), 1차원 이상이면 인덱싱
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
                
            output_np, c_val = _rust.poincare_ball_layer_layerwise_cpu(
                u.cpu().numpy(), v.cpu().numpy(), kappa_val, layer_idx, c_min, c_max, t
            )
            ctx.c_val = c_val
            return torch.from_numpy(output_np).to(u.device)
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            u_prime = poincare_scalar_mul(u, 1.0 - t, ctx.c)
            v_prime = poincare_scalar_mul(v, t, ctx.c)
            output = poincare_add(u_prime, v_prime, ctx.c)
            ctx.save_for_backward(u.clone(), v.clone(), u_prime.clone(), v_prime.clone())
            return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        t = ctx.t
        
        if ctx.use_dynamic:
            u, v, kappas = ctx.saved_tensors
            layer_idx = ctx.layer_idx
            c_min = ctx.c_min
            c_max = ctx.c_max
            
            # kappas가 0차원 텐서면 바로 item(), 1차원 이상이면 인덱싱
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
                
            grad_u_np, grad_v_np, grad_kappa_val = _rust.poincare_ball_layer_layerwise_backward_cpu(
                grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), 
                kappa_val, layer_idx, c_min, c_max, t
            )
            
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
            
            # kappas와 같은 차원의 gradient 생성
            if kappas.dim() == 0:
                grad_kappas = torch.tensor(grad_kappa_val, device=kappas.device)
            else:
                grad_kappas = torch.zeros_like(kappas)
                grad_kappas[layer_idx] = grad_kappa_val
            
            return grad_u, grad_v, None, None, grad_kappas, None, None, None
        else:
            u, v, u_prime, v_prime = ctx.saved_tensors
            c = ctx.c
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
            return grad_u, grad_v, None, None, None, None, None, None

def poincare_add(x: Tensor, y: Tensor, c: float = None, kappas: Tensor = None, layer_idx: int = None) -> Tensor:
    return MobiusAdd.apply(x, y, c, kappas, layer_idx)

def poincare_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:
    return MobiusScalarMul.apply(x, r, c)

def poincare_distance(x: Tensor, y: Tensor, c: float) -> Tensor:
    if x.is_cuda and _has_cuda:
        output = torch.empty(x.shape[0], device=x.device)
        _rust.poincare_distance_cuda(
            x.data_ptr(), y.data_ptr(), output.data_ptr(),
            x.shape[0], x.shape[1], c
        )
        return output
    else:
        output_np = _rust.poincare_distance_cpu(x.cpu().numpy(), y.cpu().numpy(), c)
        return torch.from_numpy(output_np).to(x.device)

def poincare_to_lorentz(x: Tensor, c: float) -> Tensor:
    output_np = _rust.poincare_to_lorentz(x.cpu().numpy(), c)
    return torch.from_numpy(output_np).to(x.device)

def poincare_to_klein(x: Tensor, c: float) -> Tensor:
    output_np = _rust.poincare_to_klein(x.cpu().numpy(), c)
    return torch.from_numpy(output_np).to(x.device) 