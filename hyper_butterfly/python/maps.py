import math
import torch
from .._C import (
    geodesic_cuda,
    geodesic_cpu,
    log_map_forward_cuda,
    exp_map_forward_cuda,
)
from .. import _has_cuda

def log_map(x: torch.Tensor, c: float) -> torch.Tensor:
    if x.is_cuda and _has_cuda:
        return log_map_forward_cuda(x, c)
    norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    scn = (math.sqrt(c) * norm).clamp(min=1e-6, max=1.0 - 1e-6)
    factor = torch.atanh(scn) / (scn + 1e-6)
    return factor * x

def exp_map(x: torch.Tensor, c: float) -> torch.Tensor:
    if x.is_cuda and _has_cuda:
        return exp_map_forward_cuda(x, c)
    norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    scn = (math.sqrt(c) * norm).clamp(min=1e-6, max=10.0)
    factor = torch.tanh(scn) / (scn + 1e-3)
    return factor * x

def geodesic(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    fn = geodesic_cuda if u.is_cuda else geodesic_cpu
    return fn(u, v, c, t)