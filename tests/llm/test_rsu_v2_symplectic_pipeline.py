import numpy as np
import torch
import torch.nn as nn

from reality_stone.models.manifold_learner import GlobalManifoldLearner, TinyMLP


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


def test_rsu_v2_symplectic_end_to_end(tmp_path):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner.layer_embeddings = nn.Embedding(num_layers, 4).to(device)
    with torch.no_grad():
        learner.layer_embeddings.weight.zero_()
    hypernet = TinyMLP(input_dim=4, hidden_dim=8, output_dim=r * r).to(device)
    with torch.no_grad():
        hypernet.l1.weight.zero_()
        hypernet.l1.bias.zero_()
        hypernet.l2.weight.zero_()
        hypernet.l2.bias.zero_()
    learner.hypernet = hypernet

    rsu_path = tmp_path / "toy_hypermetric.rsu2.npz"
    learner.save_rsu_v2(rsu_path)

    learner_loaded = GlobalManifoldLearner.from_rsu_v2(
        model=model,
        path=rsu_path,
    )

    wrapped = learner_loaded.replace_layers()

    x = torch.randn(3, d_model)
    out = wrapped(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

