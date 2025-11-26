import torch
import torch.nn.functional as F
import pytest
import numpy as np
from reality_stone.models.rsulf import RSULF, curvature_from_qk, extract_metric, stabilize_metric, fold_metric_layer

@pytest.fixture
def rs_layer():
    d_model = 64
    return RSULF(d_model=d_model)

def test_metric_pd_property(rs_layer):
    """Step 9: Check if metric is Positive Definite"""
    g = rs_layer.get_metric_g()
    
    # Check symmetry (if diagonal, it is symmetric)
    assert torch.allclose(g, g.t(), atol=1e-5)
    
    # Check eigenvalues > 0
    eigvals = torch.linalg.eigvalsh(g)
    assert torch.all(eigvals > 0)
    
    print(f"Min eigenvalue: {eigvals.min().item()}")

def test_inner_product_preservation(rs_layer):
    """Step 14-1: Inner Product Preservation"""
    # (q_i k_j) approx x_i^T g x_j
    # Here we check if x^T g x relates to QK attention structure.
    # Actually, g = WQ^T WK.
    # So x^T g x = x^T WQ^T WK x = (WQ x)^T (WK x) = q^T k.
    
    x = torch.randn(1, 10, rs_layer.d_model) # B, L, D
    
    # Compute q, k
    q = F.linear(x, rs_layer.WQ)
    k = F.linear(x, rs_layer.WK)
    qk_dot = (q * k).sum(dim=-1) # (B, L) - diagonal of QK^T for each token
    
    # Compute x^T g x
    g = rs_layer.get_metric_g()
    # x (B, L, D) @ g (D, D) @ x^T (B, D, L) -> (B, L, L)
    # We want token-wise: x_i g x_i
    xg = torch.matmul(x, g) # (B, L, D)
    xgx = (xg * x).sum(dim=-1) # (B, L)
    
    # If g is stabilized (diagonal), it might differ slightly from WQ^T WK
    # Let's check with raw g first
    raw_g = extract_metric(rs_layer.WQ, rs_layer.WK)
    xg_raw = torch.matmul(x, raw_g)
    xgx_raw = (xg_raw * x).sum(dim=-1)
    
    assert torch.allclose(qk_dot, xgx_raw, atol=1e-4)
    print("Inner product preserved exactly with raw metric.")

def test_potential_gradient_preservation(rs_layer):
    """Step 14-2: Potential Gradient Preservation"""
    # FFN(x) approx grad Phi(x)
    # In our implementation, grad_Phi is exactly computed from Phi.
    # So this tests the consistency of get_potential_gradient.
    
    x = torch.randn(1, 10, rs_layer.d_model)
    grad = rs_layer.get_potential_gradient(x)
    
    # Check shape
    assert grad.shape == x.shape
    
    # Check numerical gradient
    epsilon = 1e-4
    x_pert = x.clone()
    x_pert[0, 0, 0] += epsilon
    
    phi_1 = rs_layer.get_potential_phi(x).sum()
    phi_2 = rs_layer.get_potential_phi(x_pert).sum()
    
    num_grad = (phi_2 - phi_1) / epsilon
    ana_grad = grad[0, 0, 0]
    
    assert torch.abs(num_grad - ana_grad) < 1e-2
    print(f"Gradient check passed: {num_grad:.5f} vs {ana_grad:.5f}")

def test_folding_reconstruction():
    """Step 11: Folding Reconstruction Test"""
    d = 64
    WQ = torch.randn(d, d)
    
    # Fold
    WQ_folded, V = fold_metric_layer(WQ, reduction_ratio=0.5)
    
    assert WQ_folded.shape[1] == d // 2
    
    # Reconstruct (approximate)
    # WQ approx WQ_folded @ V
    # Since V is V_top from SVD, V @ V.T is identity on subspace.
    # WQ_approx = U S V.T
    WQ_approx = WQ_folded @ V
    
    # Check error is not too massive (depends on rank)
    # Just check shapes and basic properties for now
    assert WQ_approx.shape == WQ.shape

def test_rsulf_forward(rs_layer):
    """Step 12: Functional Test"""
    x = torch.randn(2, 10, rs_layer.d_model)
    V = torch.zeros(2, 10, rs_layer.d_model)
    
    x_next, V_next = rs_layer(x, V)
    
    assert x_next.shape == x.shape
    assert V_next.shape == x.shape
    assert not torch.isnan(x_next).any()

def test_rs_lagrangian_potential_decrease(rs_layer):
    x = torch.randn(4, 16, rs_layer.d_model)
    rs_layer.alpha = 0.0
    rs_layer.beta = 0.0
    rs_layer.gamma = 0.0
    phi_before = rs_layer.get_potential_phi(x).mean().item()
    x_next, _ = rs_layer(x, None)
    phi_after = rs_layer.get_potential_phi(x_next).mean().item()
    assert phi_after < phi_before

def test_diffusion_stability():
    d_model = 16
    L = torch.randn(d_model, d_model) * 0.01
    layer = RSULF(d_model=d_model, L_matrix=L)
    layer.lr = 0.0
    layer.gamma = 0.0
    layer.alpha = 0.01
    layer.beta = 0.01
    x = torch.randn(2, 32, d_model)
    var_before = x.var(dim=1).mean().item()
    for _ in range(5):
        x, _ = layer(x, None)
    var_after = x.var(dim=1).mean().item()
    assert not torch.isnan(x).any()
    assert var_after < var_before * 10.0

def test_dp_memory_stability():
    """Step 8.4: Check if DP/Bellman Memory is stable over long sequences"""
    d_model = 16
    layer = RSULF(d_model=d_model)
    layer.lr = 0.0
    layer.alpha = 0.0
    layer.beta = 0.0
    layer.gamma = 0.9  # Discount factor
    
    x = torch.randn(1, 10, d_model)
    # Initialize V with shape (B, L) to trigger DP memory logic
    # NOT (B, L, D) which triggers momentum logic
    V = torch.zeros(1, 10)
    
    # Run for 100 steps
    for _ in range(100):
        x, V = layer(x, V)
        
    # Check if V exploded
    assert not torch.isnan(V).any()
    assert not torch.isinf(V).any()
    
    # With constant x (lr=0), phi is constant.
    # V_t = gamma * V_{t-1} + phi
    # It should converge to phi / (1 - gamma)
    phi = layer.get_potential_phi(x) # (1, 10)
    expected_V = phi / (1.0 - layer.gamma)
    
    assert torch.allclose(V, expected_V, rtol=1e-3, atol=1e-3)
    print("DP Memory converged to theoretical value correctly.")
