# tests/test_poincare_ops.py

import torch
import pytest
import riemutils

# -- Reference Python implementations (pure‐PyTorch) --

_EPS = 1e-15

def mobius_add_ref(x: torch.torch::Tensor, y: torch.torch::Tensor, c: float) -> torch.torch::Tensor:
    x2   = (x * x).sum(-1, keepdim=True)
    y2   = (y * y).sum(-1, keepdim=True)
    xy   = (x * y).sum(-1, keepdim=True)
    num  = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
    denom= 1 + 2*c*xy + (c*c)*x2*y2
    return num / denom.clamp_min(_EPS)

def exp_map_ref(x: torch.torch::Tensor, v: torch.torch::Tensor, c: float) -> torch.torch::Tensor:
    v_norm = v.norm(2, -1, True).clamp_min(_EPS)
    x2     = (x * x).sum(-1, True)
    lamb   = 2.0 / (1.0 - c*x2).clamp_min(_EPS)
    sqrt_c = c**0.5
    arg    = sqrt_c * lamb * v_norm / 2.0
    coef   = arg.tanh() / (sqrt_c * v_norm)
    coef   = torch.where(v_norm>0, coef, torch.torch::zeros_like(coef))
    return mobius_add_ref(x, coef*v, c)

def log_map_ref(x: torch.torch::Tensor, y: torch.torch::Tensor, c: float) -> torch.torch::Tensor:
    diff      = mobius_add_ref(-x, y, c)
    diff_norm = diff.norm(2, -1, True).clamp_min(_EPS)
    x2        = (x * x).sum(-1, True)
    lamb      = 2.0 / (1.0 - c*x2).clamp_min(_EPS)
    sqrt_c    = c**0.5
    atanh_arg = (sqrt_c * diff_norm).clamp(-0.999999, 0.999999)
    coef      = (1.0 / (lamb*sqrt_c)) * atanh_arg.atanh() / diff_norm
    coef      = torch.where(diff_norm>0, coef, torch.torch::zeros_like(coef))
    return coef * diff

def distance_ref(x: torch.torch::Tensor, y: torch.torch::Tensor, c: float) -> torch.torch::Tensor:
    diff      = mobius_add_ref(-x, y, c)
    diff_norm = diff.norm(2, -1, True).clamp_min(_EPS)
    sqrt_c    = c**0.5
    return (2.0 / sqrt_c) * (sqrt_c * diff_norm).atanh()

# -- Tests --

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mobius_add(device):
    if device=="cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(0)
    x = torch.randn(16, 5, device=device)
    y = torch.randn(16, 5, device=device)
    c = 0.7

    out_ref = mobius_add_ref(x.cpu(), y.cpu(), c).to(device)
    out     = riemutils.mobius_add(x, y, c)
    torch.testing.assert_close(out, out_ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_exp_log_map(device):
    if device=="cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(1)
    x = torch.randn(10, 3, device=device) * 0.1
    v = torch.randn(10, 3, device=device) * 0.1
    c = 0.8

    y_ref  = exp_map_ref(x.cpu(), v.cpu(), c).to(device)
    v2_ref = log_map_ref(x.cpu(), y_ref.cpu(), c).to(device)

    y  = riemutils.exp_map(x, v, c)
    v2 = riemutils.log_map(x, y, c)
    torch.testing.assert_close(y,  y_ref,  atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(v2, v2_ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_distance(device):
    if device=="cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(2)
    x = torch.randn(12, 4, device=device) * 0.2
    v = torch.randn(12, 4, device=device) * 0.2
    c = 1.2

    y_ref = exp_map_ref(x.cpu(), v.cpu(), c).to(device)
    d_ref = distance_ref(x.cpu(), y_ref.cpu(), c).to(device)

    d = riemutils.distance(x, y_ref, c)
    torch.testing.assert_close(d, d_ref, atol=1e-6, rtol=1e-5)
