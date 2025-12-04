import numpy as np
import torch
import torch.nn as nn

from reality_stone.models.manifold_learner import GlobalManifoldLearner


class ToyBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ToyModel(nn.Module):
    def __init__(self, d_model: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(d_model) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _init_toy_qk_weights(model: ToyModel, d_model: int) -> None:
    with torch.no_grad():
        for i, block in enumerate(model.layers):
            eye = torch.eye(d_model)
            block.q_proj.weight.copy_(eye)
            scale = 1.0 + float(i)
            diag = torch.arange(1, d_model + 1, dtype=torch.float32) * scale
            block.k_proj.weight.copy_(torch.diag(diag))


def test_global_manifold_learner_creates_hypermetric_toy():
    torch.manual_seed(0)
    np.random.seed(0)

    d_model = 4
    num_layers = 2
    r = 2

    model = ToyModel(d_model=d_model, num_layers=num_layers)
    _init_toy_qk_weights(model, d_model)

    learner = GlobalManifoldLearner(
        model=model,
        d_model=d_model,
        r=r,
        hyper_hidden_dim=8,
        layer_emb_dim=4,
    )

    learner.collect_weights()
    learner.extract_global_basis()

    learner.train_hypernet(epochs=5, batch_size=num_layers, lr=1e-2)

    rust_hm = learner.create_rust_hyper_metric()

    for idx in range(num_layers):
        emb = learner.get_layer_embedding(idx)
        core = rust_hm.generate_core(emb)
        assert core.shape == (r, r)
        assert np.isfinite(core).all()


