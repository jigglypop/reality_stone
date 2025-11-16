import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    TreeNodeOperator,
    LevelInvariantTreeProcessor,
)


def test_dynamic_manifold_basic():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    assert hasattr(operator, "manifold_selector")
    assert hasattr(operator, "aggregator_poincare")
    assert hasattr(operator, "aggregator_lorentz")
    assert hasattr(operator, "aggregator_klein")


def test_dynamic_manifold_up_operator():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 3
    children_embs = torch.randn(B, N, d_model)
    
    result = operator.up_operator(children_embs)
    
    assert result.shape == (B, d_model)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_dynamic_manifold_down_operator():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B = 2
    num_children = 3
    parent_emb = torch.randn(B, d_model)
    
    result = operator.down_operator(parent_emb, num_children)
    
    assert result.shape == (B, num_children, d_model)
    assert not torch.isnan(result).any()


def test_dynamic_manifold_selection():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 5
    children_embs = torch.randn(B, N, d_model)
    
    mean_emb = children_embs.mean(dim=1)
    manifold_logits = operator.manifold_selector(mean_emb)
    
    assert manifold_logits.shape == (B, 3)
    
    manifold_probs = torch.softmax(manifold_logits, dim=-1)
    assert torch.allclose(manifold_probs.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_dynamic_manifold_gradient_flow():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 3
    children_embs = torch.randn(B, N, d_model, requires_grad=True)
    
    result = operator.up_operator(children_embs)
    loss = result.sum()
    loss.backward()
    
    assert children_embs.grad is not None
    assert not torch.isnan(children_embs.grad).any()


def test_tree_processor_dynamic_manifold():
    d_model = 64
    processor = LevelInvariantTreeProcessor(d_model, enable_dynamic_manifold=True)
    
    for node_type, operator in processor.node_operators.items():
        assert operator.enable_dynamic_manifold
        assert hasattr(operator, "manifold_selector")


def test_dynamic_manifold_different_manifolds():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 3
    children_embs = torch.randn(B, N, d_model)
    
    result_poincare = operator.aggregator_poincare(children_embs)
    result_lorentz = operator.aggregator_lorentz(children_embs)
    result_klein = operator.aggregator_klein(children_embs)
    
    assert result_poincare.shape == (B, d_model)
    assert result_lorentz.shape == (B, d_model)
    assert result_klein.shape == (B, d_model)
    
    assert not torch.equal(result_poincare, result_lorentz)
    assert not torch.equal(result_poincare, result_klein)


def test_dynamic_manifold_weighted_combination():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 1, 3
    children_embs = torch.randn(B, N, d_model)
    
    result_full = operator.up_operator(children_embs)
    
    mean_emb = children_embs.mean(dim=1)
    manifold_logits = operator.manifold_selector(mean_emb)
    manifold_probs = torch.softmax(manifold_logits, dim=-1)
    
    result_poincare = operator.aggregator_poincare(children_embs)
    result_lorentz = operator.aggregator_lorentz(children_embs)
    result_klein = operator.aggregator_klein(children_embs)
    
    expected = (
        manifold_probs[0, 0] * result_poincare[0] +
        manifold_probs[0, 1] * result_lorentz[0] +
        manifold_probs[0, 2] * result_klein[0]
    )
    
    assert torch.allclose(result_full[0], expected, atol=1e-5)


def test_dynamic_manifold_batch_consistency():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 5, 3
    children_embs = torch.randn(B, N, d_model)
    
    result_batched = operator.up_operator(children_embs)
    
    results_individual = []
    for b in range(B):
        result_b = operator.up_operator(children_embs[b:b+1])
        results_individual.append(result_b[0])
    
    results_stacked = torch.stack(results_individual)
    
    assert torch.allclose(result_batched, results_stacked, atol=1e-5)


def test_dynamic_manifold_deterministic():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 3
    children_embs = torch.randn(B, N, d_model)
    
    torch.manual_seed(42)
    result1 = operator.up_operator(children_embs)
    
    torch.manual_seed(42)
    result2 = operator.up_operator(children_embs)
    
    assert torch.equal(result1, result2)

