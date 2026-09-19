from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import Function

from .. import _has_cuda, _rust
from .._fallback import (
    autograd_vjp,
    dynamic_curvature,
    dynamic_curvature_torch,
    mobius_add_torch,
    mobius_scalar_torch,
)

# ``_rust`` may be the pure-Python stub (IS_FALLBACK=True). Only a compiled module counts
# as native; every other path uses the differentiable torch formulas below.
_HAS_NATIVE = _rust is not None and not bool(getattr(_rust, "IS_FALLBACK", False))


def _kappa_leaf(kappas: Tensor, layer_idx: int) -> Tensor:
    return kappas if kappas.dim() == 0 else kappas[int(layer_idx)]


def _scatter_kappa_grad(kappas: Tensor, layer_idx: int, grad_k: Tensor) -> Tensor:
    if kappas.dim() == 0:
        return grad_k.reshape(()).to(dtype=kappas.dtype, device=kappas.device)
    grad_kappas = torch.zeros_like(kappas)
    grad_kappas[int(layer_idx)] = grad_k.to(dtype=kappas.dtype, device=kappas.device)
    return grad_kappas


class MobiusAdd(Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        y: Tensor,
        c: float = None,
        kappas: Tensor = None,
        layer_idx: int = None,
        c_min: float = -2.0,
        c_max: float = -0.1,
    ) -> Tensor:
        ctx.use_dynamic = kappas is not None and layer_idx is not None
        ctx.c = c if c is not None else 1.0
        ctx.layer_idx = layer_idx
        ctx.c_min = c_min
        ctx.c_max = c_max

        if ctx.use_dynamic:
            ctx.save_for_backward(x, y, kappas)
            if kappas.dim() == 0:
                kappas_list = [float(kappas.item())]
            else:
                kappas_list = [float(v) for v in kappas.detach().cpu().tolist()]
            if _HAS_NATIVE:
                output_np, c_val = _rust.mobius_add_layerwise_cpu(
                    x.detach().cpu().numpy(),
                    y.detach().cpu().numpy(),
                    kappas_list,
                    int(layer_idx),
                    float(c_min),
                    float(c_max),
                )
                ctx.c_val = float(c_val)
                return torch.from_numpy(output_np).to(device=x.device, dtype=x.dtype)
            ctx.c_val = dynamic_curvature(kappas_list[int(layer_idx)], c_min, c_max)
            return mobius_add_torch(x, y, ctx.c_val)

        ctx.save_for_backward(x, y)
        if x.is_cuda and _has_cuda and _HAS_NATIVE:
            output = torch.empty_like(x)
            _rust.mobius_add_cuda(
                x.data_ptr(),
                y.data_ptr(),
                output.data_ptr(),
                x.shape[0],
                x.shape[1],
                float(ctx.c),
            )
            return output
        if _HAS_NATIVE:
            out = _rust.mobius_add_cpu(
                x.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                float(ctx.c),
            )
            return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
        return mobius_add_torch(x, y, float(ctx.c))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.use_dynamic:
            x, y, kappas = ctx.saved_tensors
            layer_idx = int(ctx.layer_idx)
            c_min, c_max = float(ctx.c_min), float(ctx.c_max)
            if _HAS_NATIVE:
                if kappas.dim() == 0:
                    kappas_list = [float(kappas.item())]
                else:
                    kappas_list = [float(v) for v in kappas.detach().cpu().tolist()]
                gx_np, gy_np, gk = _rust.mobius_add_layerwise_backward_cpu(
                    grad_output.detach().cpu().numpy(),
                    x.detach().cpu().numpy(),
                    y.detach().cpu().numpy(),
                    kappas_list,
                    layer_idx,
                    c_min,
                    c_max,
                )
                gx = torch.from_numpy(gx_np).to(device=grad_output.device, dtype=grad_output.dtype)
                gy = torch.from_numpy(gy_np).to(device=grad_output.device, dtype=grad_output.dtype)
                gk_tensor = _scatter_kappa_grad(kappas, layer_idx, torch.as_tensor(float(gk)))
                return gx, gy, None, gk_tensor, None, None, None

            def _fn(x_: Tensor, y_: Tensor, k_: Tensor) -> Tensor:
                return mobius_add_torch(x_, y_, dynamic_curvature_torch(k_, c_min, c_max))

            gx, gy, gk = autograd_vjp(_fn, grad_output, x, y, _kappa_leaf(kappas, layer_idx))
            return gx, gy, None, _scatter_kappa_grad(kappas, layer_idx, gk), None, None, None

        x, y = ctx.saved_tensors
        if _HAS_NATIVE and hasattr(_rust, "poincare"):
            gx_np, gy_np = _rust.poincare.mobius_add_vjp_cpu(
                grad_output.detach().cpu().numpy(),
                x.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                float(ctx.c),
            )
            gx = torch.from_numpy(gx_np).to(device=grad_output.device, dtype=grad_output.dtype)
            gy = torch.from_numpy(gy_np).to(device=grad_output.device, dtype=grad_output.dtype)
            return gx, gy, None, None, None, None, None

        c_val = float(ctx.c)
        gx, gy = autograd_vjp(lambda x_, y_: mobius_add_torch(x_, y_, c_val), grad_output, x, y)
        return gx, gy, None, None, None, None, None


class MobiusScalarMul(Function):
    @staticmethod
    def forward(ctx, x: Tensor, r: float, c: float) -> Tensor:
        ctx.r = float(r)
        ctx.c = float(c)
        ctx.save_for_backward(x)
        if x.is_cuda and _has_cuda and _HAS_NATIVE:
            output = torch.empty_like(x)
            _rust.mobius_scalar_cuda(
                x.data_ptr(),
                output.data_ptr(),
                x.shape[0],
                x.shape[1],
                float(r),
                float(c),
            )
            return output
        if _HAS_NATIVE:
            out = _rust.mobius_scalar_cpu(x.detach().cpu().numpy(), float(r), float(c))
            return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
        return mobius_scalar_torch(x, float(r), float(c))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        if _HAS_NATIVE and hasattr(_rust, "poincare"):
            gx_np = _rust.poincare.mobius_scalar_vjp_cpu(
                grad_output.detach().cpu().numpy(),
                x.detach().cpu().numpy(),
                float(ctx.c),
                float(ctx.r),
            )
            gx = torch.from_numpy(gx_np).to(device=grad_output.device, dtype=grad_output.dtype)
            return gx, None, None
        r_val, c_val = float(ctx.r), float(ctx.c)
        (gx,) = autograd_vjp(lambda x_: mobius_scalar_torch(x_, r_val, c_val), grad_output, x)
        return gx, None, None
