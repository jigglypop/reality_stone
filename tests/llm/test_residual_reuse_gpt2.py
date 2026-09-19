import torch
from transformers import GPT2Config, GPT2LMHeadModel

from reality_stone.models.residual_reuse import (
    ResidualReuseMLP,
    install_gpt2_residual_reuse,
    reset_residual_reuse,
    residual_reuse_report,
)


def tiny_gpt2() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=64,
        n_positions=16,
        n_ctx=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
        use_cache=True,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def test_install_wraps_gpt2_mlp_layers():
    model = tiny_gpt2()
    wrappers = install_gpt2_residual_reuse(model, similarity_threshold=0.999)

    assert len(wrappers) == model.config.n_layer
    assert all(isinstance(block.mlp, ResidualReuseMLP) for block in model.transformer.h)


def test_reuses_identical_decode_residual_without_changing_logits():
    torch.manual_seed(7)
    model = tiny_gpt2()
    wrappers = install_gpt2_residual_reuse(
        model,
        similarity_threshold=0.999,
        max_entries=8,
        signature_dim=None,
    )
    input_ids = torch.tensor([[3]])

    with torch.no_grad():
        first = model(input_ids, use_cache=True).logits
        second = model(input_ids, use_cache=True).logits

    report = residual_reuse_report(wrappers)
    assert report["hits"] >= model.config.n_layer
    assert report["hit_rate"] > 0
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_prefill_sequence_does_not_use_residual_reuse():
    model = tiny_gpt2()
    wrappers = install_gpt2_residual_reuse(model, similarity_threshold=0.0)
    input_ids = torch.tensor([[3, 4, 5]])

    with torch.no_grad():
        _ = model(input_ids, use_cache=True)

    report = residual_reuse_report(wrappers)
    assert report["hits"] == 0
    assert report["misses"] == 0
    assert report["disabled"] >= model.config.n_layer


def test_reset_clears_cache_and_stats():
    model = tiny_gpt2()
    wrappers = install_gpt2_residual_reuse(model, similarity_threshold=0.999)
    input_ids = torch.tensor([[3]])

    with torch.no_grad():
        _ = model(input_ids, use_cache=True)
        _ = model(input_ids, use_cache=True)

    assert residual_reuse_report(wrappers)["hits"] > 0
    reset_residual_reuse(wrappers)
    report = residual_reuse_report(wrappers)
    assert report["hits"] == 0
    assert report["misses"] == 0


def test_relative_l2_match_reuses_identical_decode_residual():
    model = tiny_gpt2()
    wrappers = install_gpt2_residual_reuse(
        model,
        match_metric="relative_l2",
        distance_tolerance=1e-6,
        signature_dim=None,
    )
    input_ids = torch.tensor([[3]])

    with torch.no_grad():
        first = model(input_ids, use_cache=True).logits
        second = model(input_ids, use_cache=True).logits

    report = residual_reuse_report(wrappers)
    assert report["hits"] >= model.config.n_layer
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_audit_records_hit_residual_error():
    model = tiny_gpt2()
    wrappers = install_gpt2_residual_reuse(
        model,
        match_metric="relative_l2",
        distance_tolerance=1e-6,
        signature_dim=None,
        audit=True,
        audit_return_cached=False,
    )
    input_ids = torch.tensor([[3]])

    with torch.no_grad():
        _ = model(input_ids, use_cache=True)
        _ = model(input_ids, use_cache=True)

    report = residual_reuse_report(wrappers)
    assert report["audit_hits"] >= model.config.n_layer
    assert report["audit_rel_error_max"] <= 1e-5
