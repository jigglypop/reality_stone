import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
    _apply_top_down_decoding,
)
from reality_stone.utils.pre_segmenter import DocumentTree, TreeNode


@pytest.fixture
def sample_model():
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
    return HierarchicalSentenceTopicLLM(config)


@pytest.fixture
def sample_tree():
    nodes = [
        TreeNode(id=0, type="document", parent=None, text="test document"),
        TreeNode(id=1, type="sentence", parent=0, text="sentence 1"),
        TreeNode(id=2, type="sentence", parent=0, text="sentence 2"),
    ]
    return DocumentTree(nodes=nodes, root_id=0)


def test_top_down_decoding_basic(sample_model, sample_tree):
    B, T, L = 1, 2, 10
    device = next(sample_model.parameters()).device
    
    tokens = torch.randint(1, 100, (B, T, L))
    replacement_mask = torch.ones(T, L)
    
    hidden = torch.randn(B, T * L, sample_model.config.d_model)
    info = {"hidden": hidden}
    
    result = _apply_top_down_decoding(
        model=sample_model,
        tree=sample_tree,
        info=info,
        tokens=tokens,
        replacement_mask=replacement_mask,
        device=device,
    )
    
    assert result.shape[0] == B
    assert result.shape[1] <= T * L
    assert not torch.isnan(result).any()


def test_top_down_decoding_with_tree_processor(sample_model, sample_tree):
    B, T, L = 1, 2, 10
    device = next(sample_model.parameters()).device
    
    tokens = torch.randint(1, 100, (B, T, L))
    replacement_mask = torch.ones(T, L)
    
    hidden = torch.randn(B, T * L, sample_model.config.d_model)
    info = {"hidden": hidden}
    
    result = _apply_top_down_decoding(
        model=sample_model,
        tree=sample_tree,
        info=info,
        tokens=tokens,
        replacement_mask=replacement_mask,
        device=device,
    )
    
    assert result.shape[0] == B
    assert result.dtype == torch.long
    assert (result >= 0).all()
    assert (result < sample_model.config.vocab_size).all()


def test_top_down_decoding_preserves_structure(sample_model, sample_tree):
    B, T, L = 1, 2, 10
    device = next(sample_model.parameters()).device
    
    tokens = torch.randint(1, 100, (B, T, L))
    replacement_mask = torch.zeros(T, L)
    replacement_mask[:, :3] = 1
    
    hidden = torch.randn(B, T * L, sample_model.config.d_model)
    info = {"hidden": hidden}
    
    result = _apply_top_down_decoding(
        model=sample_model,
        tree=sample_tree,
        info=info,
        tokens=tokens,
        replacement_mask=replacement_mask,
        device=device,
    )
    
    tokens_flat = tokens.view(B, T * L)
    for b in range(B):
        for i in range(min(T * L, result.shape[1])):
            sent_idx = i // L
            tok_idx = i % L
            if sent_idx < T and tok_idx < L:
                if replacement_mask[sent_idx, tok_idx] == 0:
                    assert result[b, i] == tokens_flat[b, i] or tokens_flat[b, i] == 0

