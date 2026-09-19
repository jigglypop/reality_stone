"""Regression for the native Mobius second-input VJP and dynamic curvature."""
import pytest
import torch

import reality_stone as rs
from reality_stone._fallback import dynamic_curvature_torch, mobius_add_torch


@pytest.mark.parametrize("curvature", [0.0, 0.7, -1.1])
def test_native_mobius_gradients_match_torch(curvature):
    assert rs._has_rust_ext, "this release gate requires the installed native wheel"
    generator = torch.Generator().manual_seed(29)
    x = (torch.randn(5, 7, generator=generator) * 0.08).requires_grad_()
    y = (torch.randn(5, 7, generator=generator) * 0.08).requires_grad_()
    upstream = torch.randn(5, 7, generator=generator)
    actual = rs.poincare_add(x, y, c=curvature)
    (actual * upstream).sum().backward()
    xr, yr = x.detach().requires_grad_(), y.detach().requires_grad_()
    expected = mobius_add_torch(xr, yr, curvature)
    (expected * upstream).sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(x.grad, xr.grad, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(y.grad, yr.grad, atol=2e-5, rtol=2e-4)


def test_native_dynamic_curvature_gradient_matches_torch():
    assert rs._has_rust_ext
    x = torch.tensor([[0.1, -0.05]], requires_grad=True)
    y = torch.tensor([[0.03, 0.07]], requires_grad=True)
    kappa = torch.tensor([0.3], requires_grad=True)
    actual = rs.poincare_add(x, y, kappas=kappa, layer_idx=0, c_min=-2.0, c_max=-0.1)
    actual.square().sum().backward()
    xr, yr, kr = [v.detach().requires_grad_() for v in (x, y, kappa)]
    expected = mobius_add_torch(xr, yr, dynamic_curvature_torch(kr[0], -2.0, -0.1))
    expected.square().sum().backward()
    for got, wanted in [(actual, expected), (x.grad, xr.grad), (y.grad, yr.grad), (kappa.grad, kr.grad)]:
        torch.testing.assert_close(got, wanted, atol=2e-5, rtol=2e-4)
