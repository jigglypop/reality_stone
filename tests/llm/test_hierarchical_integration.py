import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
    infer_hierarchical_llm_on_text,
)
from reality_stone.utils.pre_segmenter import PreSegmenter


@pytest.fixture
def sample_config():
    return HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        enable_structural_edit=False,
        lambda_consistency=0.1,
        lambda_diversity=0.05,
        c_poincare=0.1,
    )


@pytest.fixture
def sample_model(sample_config):
    return HierarchicalSentenceTopicLLM(sample_config)


def test_hierarchical_llm_forward_basic(sample_model):
    B, T, L, K = 2, 3, 10, 2
    
    batch = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    logits, info = sample_model(batch, compute_loss=True, use_tree_processing=False)
    
    assert logits.shape[0] == B
    assert "loss" in info
    assert "P_topic" in info
    assert "metric_ctx" in info
    
    loss = info["loss"]
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss).any()
    assert loss.item() >= 0


def test_hierarchical_llm_forward_with_tree(sample_model):
    segmenter = PreSegmenter(max_length=32, k_neighbors=2)
    text = "This is test sentence one. This is test sentence two."
    
    output = segmenter(text)
    
    B = 1
    batch = {
        "tokens": output["tokens"].unsqueeze(0),
        "topo_idx": output["topo_idx"].unsqueeze(0),
        "tree": [output["tree"]],
    }
    
    logits, info = sample_model(batch, compute_loss=True, use_tree_processing=True)
    
    assert logits.shape[0] == B
    assert "loss" in info
    assert not torch.isnan(info["loss"]).any()


def test_hierarchical_llm_forward_loss_components(sample_model):
    B, T, L, K = 2, 3, 10, 2
    
    batch = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    logits, info = sample_model(batch, compute_loss=True)
    
    assert "loss_lm" in info
    assert "loss_consistency" in info
    assert "loss_diversity" in info
    assert "loss_length" in info
    
    for loss_key in ["loss_lm", "loss_consistency", "loss_diversity", "loss_length"]:
        loss_val = info[loss_key]
        assert isinstance(loss_val, torch.Tensor)
        assert not torch.isnan(loss_val).any()


def test_hierarchical_llm_encode_decode_cycle(sample_model):
    B, T, L = 1, 2, 10
    tokens = torch.randint(1, 100, (B, T, L))
    
    sentence_embeddings = sample_model.encode_tokens_to_sentences(tokens)
    
    assert sentence_embeddings.shape == (B, T, sample_model.config.d_model)
    assert not torch.isnan(sentence_embeddings).any()
    
    paragraph_embedding = sample_model.encode_sentences_to_paragraph(sentence_embeddings)
    
    assert paragraph_embedding.shape == (B, sample_model.config.d_model)
    assert not torch.isnan(paragraph_embedding).any()


def test_hierarchical_llm_metric_context_generation(sample_model):
    B, T, L, K = 2, 3, 10, 2
    
    batch = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    logits, info = sample_model(batch, compute_loss=False)
    
    metric_ctx = info["metric_ctx"]
    assert metric_ctx.shape == (B, T, sample_model.config.d_head, sample_model.config.d_head)
    assert not torch.isnan(metric_ctx).any()
    
    P_topic = info["P_topic"]
    assert P_topic.shape == (B, T, sample_model.config.num_topics)
    assert torch.allclose(P_topic.sum(dim=-1), torch.ones(B, T), atol=1e-5)


def test_hierarchical_llm_backward_pass(sample_model):
    B, T, L, K = 2, 3, 10, 2
    
    batch = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    logits, info = sample_model(batch, compute_loss=True)
    loss = info["loss"]
    
    loss.backward()
    
    for name, param in sample_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient not computed for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_infer_hierarchical_llm_basic():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        enable_structural_edit=False,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    text = "This is a test sentence. Another test sentence here."
    
    result = infer_hierarchical_llm_on_text(
        model=model,
        text=text,
        max_length=32,
        k_neighbors=2,
        use_top_down=False,
    )
    
    assert "original_text" in result
    assert "generated_text" in result
    assert "topics" in result
    assert result["original_text"] == text
    assert isinstance(result["generated_text"], str)
    assert isinstance(result["topics"], list)


def test_infer_hierarchical_llm_with_top_down():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        enable_structural_edit=False,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    text = "First sentence. Second sentence. Third sentence."
    
    result = infer_hierarchical_llm_on_text(
        model=model,
        text=text,
        max_length=32,
        k_neighbors=2,
        use_top_down=True,
    )
    
    assert "original_text" in result
    assert "generated_text" in result
    assert isinstance(result["generated_text"], str)


def test_hierarchical_llm_structural_edit():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        enable_structural_edit=True,
        lambda_edit=0.1,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    text = "Test sentence one. Test sentence two."
    
    result = infer_hierarchical_llm_on_text(
        model=model,
        text=text,
        max_length=32,
        k_neighbors=2,
        use_top_down=False,
    )
    
    assert "generated_text" in result
    assert isinstance(result["generated_text"], str)


def test_hierarchical_llm_dynamic_manifold():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
    )
    config.enable_dynamic_manifold = True
    
    model = HierarchicalSentenceTopicLLM(config)
    
    B, T, L, K = 2, 3, 10, 2
    batch = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    logits, info = model(batch, compute_loss=True)
    
    assert "loss" in info
    assert not torch.isnan(info["loss"]).any()


def test_hierarchical_llm_empty_input():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    text = ""
    
    result = infer_hierarchical_llm_on_text(
        model=model,
        text=text,
        max_length=32,
        k_neighbors=2,
    )
    
    assert result["original_text"] == text
    assert result["sentences"] == []


def test_hierarchical_llm_long_input():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        max_lm_seq_len=64,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    text = " ".join([f"Sentence number {i}." for i in range(20)])
    
    result = infer_hierarchical_llm_on_text(
        model=model,
        text=text,
        max_length=16,
        k_neighbors=2,
    )
    
    assert "generated_text" in result
    assert isinstance(result["generated_text"], str)


def test_hierarchical_llm_gradient_accumulation(sample_model):
    B, T, L, K = 2, 3, 10, 2
    
    batch1 = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    batch2 = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    sample_model.zero_grad()
    
    logits1, info1 = sample_model(batch1, compute_loss=True)
    loss1 = info1["loss"]
    loss1.backward()
    
    grads1 = {name: param.grad.clone() for name, param in sample_model.named_parameters() if param.grad is not None}
    
    logits2, info2 = sample_model(batch2, compute_loss=True)
    loss2 = info2["loss"]
    loss2.backward()
    
    for name, param in sample_model.named_parameters():
        if param.grad is not None and name in grads1:
            assert not torch.equal(param.grad, grads1[name]), f"Gradient not accumulated for {name}"

