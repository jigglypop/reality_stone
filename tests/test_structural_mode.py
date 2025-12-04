import torch
import torch.nn as nn
import torch.nn.functional as F

from reality_stone.models.transformer_converter import FFNPotential


def test_ffn_potential_matches_toy_ffn():
    torch.manual_seed(0)
    d_model = 8
    hidden = 16
    ffn = nn.Sequential(
        nn.Linear(d_model, hidden),
        nn.ReLU(),
        nn.Linear(hidden, d_model),
    )
    potential = FFNPotential(d_model, hidden_dim=16)
    optimizer = torch.optim.Adam(potential.parameters(), lr=1e-2)
    for _ in range(50):
        x = torch.randn(4, 3, d_model)
        with torch.no_grad():
            f_out = ffn(x)
        grad_phi = potential.gradient(x)
        loss = F.mse_loss(grad_phi, -f_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    x_test = torch.randn(2, 3, d_model)
    with torch.no_grad():
        f_out = ffn(x_test)
        g_out = -potential.gradient(x_test)
        f_flat = f_out.view(-1, d_model)
        g_flat = g_out.view(-1, d_model)
        cos = F.cosine_similarity(f_flat, g_flat, dim=-1).mean().item()
        rel = (f_flat - g_flat).norm() / (f_flat.norm() + 1e-8)
    assert cos > 0.9
    assert rel < 0.5


