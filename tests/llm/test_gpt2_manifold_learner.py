import numpy as np
import torch
from transformers import GPT2LMHeadModel
from reality_stone.models.manifold_learner import GlobalManifoldLearner


def test_gpt2_manifold_learner_collect_weights():
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    d_model = model.config.n_embd
    print(f"d_model: {d_model}")

    learner = GlobalManifoldLearner(
        model=model,
        d_model=d_model,
        r=64,
        hyper_hidden_dim=32,
        layer_emb_dim=16,
    )

    learner.collect_weights()
    print(f"Collected layers: {len(learner.layers_wq)}")

    assert len(learner.layers_wq) == 12, f"Expected 12 layers, got {len(learner.layers_wq)}"
    assert learner.layers_wq[0].shape == (d_model, d_model), f"WQ shape mismatch: {learner.layers_wq[0].shape}"
    assert learner.layers_wk[0].shape == (d_model, d_model), f"WK shape mismatch: {learner.layers_wk[0].shape}"

    print(f"WQ[0] shape: {learner.layers_wq[0].shape}")
    print(f"WK[0] shape: {learner.layers_wk[0].shape}")

    learner.extract_global_basis()
    print("Global basis extracted successfully")

    assert learner.u_global is not None
    assert learner.v_global is not None
    print(f"U global shape: {learner.u_global.shape}")
    print(f"V global shape: {learner.v_global.shape}")

    print("TEST PASSED")


def test_gpt2_manifold_learner_full_pipeline():
    print("\n=== Full Pipeline Test ===")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    d_model = model.config.n_embd
    r = 64

    learner = GlobalManifoldLearner(
        model=model,
        d_model=d_model,
        r=r,
        hyper_hidden_dim=32,
        layer_emb_dim=16,
    )

    learner.collect_weights()
    learner.extract_global_basis()
    learner.train_hypernet(epochs=10, batch_size=12, lr=1e-2)

    rust_hm = learner.create_rust_hyper_metric()

    for idx in range(len(learner.layers_wq)):
        emb = learner.get_layer_embedding(idx)
        core = rust_hm.generate_core(emb)
        assert core.shape == (r, r), f"Core shape mismatch at layer {idx}"
        assert np.isfinite(core).all(), f"Core has non-finite values at layer {idx}"

    print("HyperMetric generation: OK")

    wrapped = learner.replace_layers()
    x = torch.randn(2, 16, d_model)
    out = wrapped(x)

    assert out.shape == (2, 16, d_model), f"Output shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "Output has non-finite values"

    print(f"Wrapped forward: OK, output shape {out.shape}")
    print("FULL PIPELINE TEST PASSED")


if __name__ == "__main__":
    test_gpt2_manifold_learner_collect_weights()
    test_gpt2_manifold_learner_full_pipeline()

