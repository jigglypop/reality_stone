import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    LevelInvariantTreeProcessor,
    TreeNodeOperator,
)
from reality_stone.utils.pre_segmenter import PreSegmenter, DocumentTree, TreeNode


@pytest.fixture
def sample_tree():
    nodes = [
        TreeNode(id=0, type="document", parent=None, text="test document"),
        TreeNode(id=1, type="sentence", parent=0, text="sentence 1"),
        TreeNode(id=2, type="sentence", parent=0, text="sentence 2"),
        TreeNode(id=3, type="token", parent=1, text="test"),
        TreeNode(id=4, type="token", parent=1, text="token"),
        TreeNode(id=5, type="token", parent=2, text="another"),
    ]
    return DocumentTree(nodes=nodes, root_id=0)


def test_tree_node_operator_up():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3)
    
    B, N = 2, 3
    children_embs = torch.randn(B, N, d_model)
    
    result = operator.up_operator(children_embs)
    
    assert result.shape == (B, d_model)
    assert not torch.isnan(result).any()


def test_tree_node_operator_down():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3)
    
    B = 2
    num_children = 3
    parent_emb = torch.randn(B, d_model)
    
    result = operator.down_operator(parent_emb, num_children)
    
    assert result.shape == (B, num_children, d_model)
    assert not torch.isnan(result).any()


def test_tree_node_operator_dynamic_manifold():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 3
    children_embs = torch.randn(B, N, d_model)
    
    result = operator.up_operator(children_embs)
    
    assert result.shape == (B, d_model)
    assert not torch.isnan(result).any()


def test_tree_processor_up(sample_tree):
    d_model = 64
    processor = LevelInvariantTreeProcessor(d_model)
    
    node_embeddings = {
        1: torch.randn(d_model),
        2: torch.randn(d_model),
        3: torch.randn(d_model),
        4: torch.randn(d_model),
        5: torch.randn(d_model),
    }
    
    result = processor.process_tree(sample_tree, node_embeddings, direction="up")
    
    assert 0 in result
    assert 1 in result
    assert 2 in result
    assert not torch.isnan(result[0]).any()


def test_tree_processor_down(sample_tree):
    d_model = 64
    processor = LevelInvariantTreeProcessor(d_model)
    
    node_embeddings = {
        0: torch.randn(d_model),
    }
    
    result = processor.process_tree(sample_tree, node_embeddings, direction="down")
    
    assert 1 in result
    assert 2 in result
    assert not torch.isnan(result[1]).any()


def test_tree_processor_dynamic_manifold(sample_tree):
    d_model = 64
    processor = LevelInvariantTreeProcessor(d_model, enable_dynamic_manifold=True)
    
    node_embeddings = {
        1: torch.randn(d_model),
        2: torch.randn(d_model),
        3: torch.randn(d_model),
        4: torch.randn(d_model),
        5: torch.randn(d_model),
    }
    
    result = processor.process_tree(sample_tree, node_embeddings, direction="up")
    
    assert 0 in result
    assert not torch.isnan(result[0]).any()


def test_pre_segmenter_tree_output():
    segmenter = PreSegmenter(max_length=32, k_neighbors=2)
    text = "This is a test. Another sentence."
    
    output = segmenter(text)
    
    assert "tree" in output
    tree = output["tree"]
    
    assert isinstance(tree, DocumentTree)
    assert len(tree.nodes) > 0
    
    doc_nodes = [n for n in tree.nodes if n.type == "document"]
    sent_nodes = [n for n in tree.nodes if n.type == "sentence"]
    
    assert len(doc_nodes) > 0
    assert len(sent_nodes) > 0

