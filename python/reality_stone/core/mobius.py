import torch
from torch import Tensor
from torch.autograd import Function
from .. import _rust, _has_cuda

class MobiusAdd(Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, c: float = None, kappas: Tensor = None, layer_idx: int = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
        ctx.save_for_backward(x, y)
        
        if kappas is not None and layer_idx is not None:
            ctx.use_dynamic = True
            ctx.layer_idx = layer_idx
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(x, y, kappas)
            
            # kappas가 0차원이면 단일 값으로, 아니면 리스트로 변환
            if kappas.dim() == 0:
                kappas_list = [kappas.item()]
            else:
                kappas_list = kappas.cpu().tolist()
                
            output_np, c_val = _rust.mobius_add_layerwise_cpu(
                x.cpu().numpy(), y.cpu().numpy(), kappas_list, layer_idx, c_min, c_max
            )
            ctx.c_val = c_val
            return torch.from_numpy(output_np).to(x.device)
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            if x.is_cuda and _has_cuda:
                output = torch.empty_like(x)
                _rust.mobius_add_cuda(x.data_ptr(), y.data_ptr(), output.data_ptr(), x.shape[0], x.shape[1], ctx.c)
                return output
            return torch.from_numpy(_rust.mobius_add_cpu(x.cpu().numpy(), y.cpu().numpy(), ctx.c))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.use_dynamic:
            x, y, kappas = ctx.saved_tensors
            layer_idx = ctx.layer_idx
            c_min = ctx.c_min
            c_max = ctx.c_max
            
            # kappas가 0차원이면 단일 값으로, 아니면 리스트로 변환
            if kappas.dim() == 0:
                kappas_list = [kappas.item()]
            else:
                kappas_list = kappas.cpu().tolist()
                
            grad_x_np, grad_y_np, grad_kappa_val = _rust.mobius_add_layerwise_backward_cpu(
                grad_output.cpu().numpy(), x.cpu().numpy(), y.cpu().numpy(),
                kappas_list, layer_idx, c_min, c_max
            )
            
            grad_x = torch.from_numpy(grad_x_np).to(grad_output.device)
            grad_y = torch.from_numpy(grad_y_np).to(grad_output.device)
            
            # kappas와 같은 차원의 gradient 생성
            if kappas.dim() == 0:
                grad_kappas = torch.tensor(grad_kappa_val, device=kappas.device)
            else:
                grad_kappas = torch.zeros_like(kappas)
                grad_kappas[layer_idx] = grad_kappa_val
            
            return grad_x, grad_y, None, grad_kappas, None, None, None
        else:
            x, y = ctx.saved_tensors
            c = ctx.c
            grad_x_np, grad_y_np = _rust.mobius_add_vjp_cpu(
                grad_output.cpu().numpy(), x.cpu().numpy(), y.cpu().numpy(), c
            )
            grad_x = torch.from_numpy(grad_x_np).to(grad_output.device)
            grad_y = torch.from_numpy(grad_y_np).to(grad_output.device)
            return grad_x, grad_y, None, None, None, None, None

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

