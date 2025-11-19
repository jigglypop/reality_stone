import math
import torch
import pytest

import reality_stone as rs


def test_mobius_scalar_zero():
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float32)
    y = rs.poincare_scalar_mul(x, r=0.0, c=1.0)
    assert torch.allclose(y, torch.zeros_like(x), atol=1e-6)


def test_mobius_scalar_identity():
    x = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    y = rs.poincare_scalar_mul(x, r=1.0, c=0.7)
    assert torch.allclose(y, x, atol=1e-3)


def test_mobius_add_identity_zero():
    x = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    z = torch.zeros_like(x)
    y = rs.poincare_add(x, z, c=0.5)
    assert torch.allclose(y, x, atol=1e-6)


def test_mobius_add_euclidean_limit_small_c():
    c = 1e-6
    x = torch.tensor([[0.01, 0.02]], dtype=torch.float32)
    y = torch.tensor([[0.03, -0.01]], dtype=torch.float32)
    m = rs.poincare_add(x, y, c=c)
    e = x + y
    assert torch.allclose(m, e, atol=1e-4)


def test_mobius_dynamic_matches_fixed_when_kappa_zero():
    x = torch.tensor([[0.05, -0.03]], dtype=torch.float32)
    y = torch.tensor([[0.02, 0.01]], dtype=torch.float32)
    c_min, c_max = -2.0, -0.1
    kappa = torch.tensor([0.0], dtype=torch.float32)
    c_mid = c_min + (c_max - c_min) * 0.5
    dyn = rs.poincare_add(x, y, kappas=kappa, layer_idx=0, c_min=c_min, c_max=c_max)
    fix = rs.poincare_add(x, y, c=float(c_mid))
    assert torch.allclose(dyn, fix, atol=5e-4)

