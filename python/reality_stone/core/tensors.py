import torch

def _mobius_add_torch(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1e-7) -> torch.Tensor:
    """PyTorch implementation of Möbius addition (for gradient computation)"""
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True)
    xy = (x * y).sum(dim=1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / denom.clamp_min(eps)

def _mobius_scalar_torch(x: torch.Tensor, r: float, c: float, eps: float = 1e-7) -> torch.Tensor:
    """PyTorch implementation of Möbius scalar multiplication"""
    sqrtc = c ** 0.5
    x_norm = x.norm(dim=1, keepdim=True).clamp_min(eps)
    scale = torch.tanh(r * torch.atanh(sqrtc * x_norm)) / (sqrtc * x_norm)
    return scale * x

def _poincare_ball_layer_torch(u: torch.Tensor, v: torch.Tensor, c: float, t: float) -> torch.Tensor:
    """PyTorch implementation of Poincaré ball layer"""
    u_prime = _mobius_scalar_torch(u, 1.0 - t, c)
    v_prime = _mobius_scalar_torch(v, t, c)
    return _mobius_add_torch(u_prime, v_prime, c) 