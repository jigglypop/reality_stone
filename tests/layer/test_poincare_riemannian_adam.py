import numpy as np
import torch
import pytest

import reality_stone as rs


def _euclidean_adam_step(x, g, m, v, step, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)
    t = float(step)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    u = -lr * m_hat / (v_hat.sqrt() + eps)
    x_new = x + u
    return x_new, m, v


def test_poincare_riemannian_adam_cpu_euclidean_limit():
    if not rs._has_rust_ext:
        pytest.skip("Rust extension not available")

    x = torch.tensor([[0.5, -0.3]], dtype=torch.float32)
    g = torch.tensor([[0.5, -0.3]], dtype=torch.float32)
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)

    step = 1
    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    c = 0.0

    x_np = x.cpu().numpy().astype(np.float32)
    g_np = g.cpu().numpy().astype(np.float32)
    m_np = m.cpu().numpy().astype(np.float32)
    v_np = v.cpu().numpy().astype(np.float32)

    x_new_np, m_new_np, v_new_np = rs._rust.poincare.poincare_riemannian_adam_step_cpu(
        x_np,
        g_np,
        m_np,
        v_np,
        step,
        float(c),
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
    )

    x_new_ref, m_ref, v_ref = _euclidean_adam_step(
        x, g, m, v, step, lr=lr, beta1=beta1, beta2=beta2, eps=eps
    )

    x_new = torch.from_numpy(x_new_np).to(x.device)
    m_new = torch.from_numpy(m_new_np).to(x.device)
    v_new = torch.from_numpy(v_new_np).to(x.device)

    assert torch.allclose(x_new, x_new_ref, atol=1e-6)
    assert torch.allclose(m_new, m_ref, atol=1e-6)
    assert torch.allclose(v_new, v_ref, atol=1e-6)


def test_poincare_riemannian_adam_cpu_stays_inside_ball():
    if not rs._has_rust_ext:
        pytest.skip("Rust extension not available")

    x = torch.tensor([[0.5, 0.4]], dtype=torch.float32)
    g = torch.tensor([[0.5, 0.4]], dtype=torch.float32)
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)

    step = 1
    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    c = 1.0

    x_np = x.cpu().numpy().astype(np.float32)
    g_np = g.cpu().numpy().astype(np.float32)
    m_np = m.cpu().numpy().astype(np.float32)
    v_np = v.cpu().numpy().astype(np.float32)

    x_new_np, m_new_np, v_new_np = rs._rust.poincare.poincare_riemannian_adam_step_cpu(
        x_np,
        g_np,
        m_np,
        v_np,
        step,
        float(c),
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
    )

    x_new = torch.from_numpy(x_new_np)
    norm = x_new.norm(p=2).item()
    max_norm = 1.0 / np.sqrt(c) - 1e-3
    assert norm < max_norm


