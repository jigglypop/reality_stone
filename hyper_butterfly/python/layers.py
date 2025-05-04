# layers.py

import math
import torch
from torch import nn
from torch.autograd import Function
from .._C import (
    poincare_forward_cpu,
    poincare_forward_cuda,
    poincare_backward_cuda,
)
from .. import _has_cuda
from .maps import log_map, exp_map, mobius_sub, geodesic

def butterfly_transform(x: torch.Tensor, params: torch.Tensor, L: int) -> torch.Tensor:
    batch, dim = x.shape
    log2_dim = int(math.log2(dim))
    out = x
    offset = 0
    for l in range(L):
        layer = l % log2_dim
        bs = 1 << layer
        nb = dim // (2 * bs)
        p = params[offset:offset + nb * 2].view(nb, 2)
        offset += nb * 2
        out = out.view(batch, nb, 2, bs)
        a = p[:, 0].view(1, nb, 1)
        b = p[:, 1].view(1, nb, 1)
        x1 = out[:, :, 0, :]
        x2 = out[:, :, 1, :]
        y1 = a * x1 + b * x2
        y2 = -b * x1 + a * x2
        out = torch.stack([y1, y2], dim=2).reshape(batch, dim)
    return out

def hyper_butterfly_py(x: torch.Tensor, params: torch.Tensor, c: float, L: int) -> torch.Tensor:
    u = log_map(x, c)
    v = butterfly_transform(u, params, L)
    y = exp_map(v, c)
    return y

class HyperButterflyFunction(Function):
    @staticmethod
    def forward(ctx, x, params, c, L):
        ctx.save_for_backward(x, params)
        ctx.c, ctx.L = c, L
        if x.is_cuda and _has_cuda:
            y, _, __ = poincare_forward_cuda(x, params, torch.empty(0,device=x.device), c, L)
        else:
            if not x.is_cuda and 'poincare_forward_cpu,' in globals():
                y, _, __ = poincare_forward_cpu(x, params, c, L)
            else:
                y = hyper_butterfly_py(x, params, c, L)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, params = ctx.saved_tensors
        c, L = ctx.c, ctx.L
        if x.is_cuda and _has_cuda:
            grad_x, grad_p = poincare_backward_cuda(
                grad_out.contiguous(), x, params, c, L
            )
            return grad_x, grad_p, None, None
        with torch.enable_grad():
            x_req = x.detach().requires_grad_()
            p_req = params.detach().requires_grad_()
            y = hyper_butterfly_py(x_req, p_req, c, L)
            gx, gp = torch.autograd.grad(y, (x_req, p_req), grad_out)
        return gx, gp, None, None
    

class GeodesicButterflyLayer(nn.Module):
    def __init__(self, dim: int, c: float, L: int, t: float = 0.5) -> None:
        super().__init__()
        self.c = c
        self.L = L
        self.t = t
        log2_d = int(math.log2(dim))
        total = 0
        for l in range(L):
            bs = 1 << (l % log2_d)
            nb = dim // (2 * bs)
            total += nb * 2
        self.params = nn.Parameter(torch.randn(total) * 1e-3)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = HyperButterflyFunction.apply(h, self.params, self.c, self.L)
        z = geodesic(h, u, self.c, self.t)
        if torch.isnan(z).any():z = torch.relu(h)
        return z