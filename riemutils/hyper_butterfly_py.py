import torch
import math


def log_map(x: torch.Tensor, c: float) -> torch.Tensor:
    """
    Hyperbolic log map at the origin with clamping.
    """
    # 1) Compute L2 norm per example, clamp to avoid 0
    norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    # 2) Compute scaled norm, clamp between [EPS, 1 - EPS]
    scn = (math.sqrt(c) * norm).clamp(min=1e-6, max=1.0 - 1e-6)
    # 3) Compute factor safely
    factor = torch.atanh(scn) / (scn + 1e-6)
    # 4) Scale input
    return factor * x


def exp_map(x: torch.Tensor, c: float) -> torch.Tensor:
    """
    Hyperbolic exp map at the origin with clamping.
    """
    # 1) Compute L2 norm per example, clamp to avoid 0
    norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    # 2) Compute scaled norm, clamp between [EPS, MAX]
    scn = (math.sqrt(c) * norm).clamp(min=1e-6, max=10.0)
    # 3) Compute factor safely
    factor = torch.tanh(scn) / (scn + 1e-3)
    # 4) Scale input
    return factor * x


def butterfly_transform(x: torch.Tensor, params: torch.Tensor, L: int) -> torch.Tensor:
    """
    Butterfly network of L layers implemented in pure PyTorch.
    """
    batch, dim = x.shape
    # Determine log2 of dimension
    log2_dim = int(math.log2(dim))

    out = x
    offset = 0
    for l in range(L):
        layer = l % log2_dim
        bs = 1 << layer
        nb = dim // (2 * bs)
        # Extract parameters for this layer: shape [nb, 2]
        p = params[offset:offset + nb * 2].view(nb, 2)
        offset += nb * 2

        # Reshape for block computations: [batch, nb, 2, bs]
        out = out.view(batch, nb, 2, bs)
        a = p[:, 0].view(1, nb, 1)
        b = p[:, 1].view(1, nb, 1)
        x1 = out[:, :, 0, :]
        x2 = out[:, :, 1, :]
        y1 = a * x1 + b * x2
        y2 = -b * x1 + a * x2
        # Reassemble
        out = torch.stack([y1, y2], dim=2).reshape(batch, dim)
    return out


def hyper_butterfly_py(x: torch.Tensor, params: torch.Tensor, c: float, L: int) -> torch.Tensor:
    """
    Pure-PyTorch implementation of Hyper-Butterfly forward.
    Automatically supports autograd for backward.
    """
    # Log map
    u = log_map(x, c)
    # Butterfly transform
    v = butterfly_transform(u, params, L)
    # Exp map
    y = exp_map(v, c)
    return y
