import torch
import reality_stone.losses as losses


def test_laplacian_same_label_basic():
    x = torch.tensor([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0]], dtype=torch.float32)
    dists_sq = torch.cdist(x, x).pow(2)
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    val = losses.laplacian_same_label(dists_sq, labels, tau=0.5)
    assert val.ndim == 0
    assert torch.isfinite(val)
    assert val >= 0


def test_poincare_kinetic_energy_positive():
    x = torch.randn(16, 4) * 0.1
    val = losses.poincare_kinetic_energy(x, curvature=1.0)
    assert val.ndim == 0
    assert torch.isfinite(val)
    assert val >= 0


def test_bellman_consistency_loss_shapes():
    crit = losses.BellmanConsistencyLoss(lambda_bellman=0.1, gamma=0.9)
    logits = torch.randn(8, 5)
    labels = torch.randint(0, 5, (8,))
    out = crit(logits, labels, apply_bellman=True)
    assert set(out.keys()) == {"total", "classification", "bellman"}
    for v in out.values():
        assert v.ndim == 0
        assert torch.isfinite(v)


