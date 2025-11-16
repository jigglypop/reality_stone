import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import EditOperationHead


@pytest.fixture
def edit_head():
    return EditOperationHead(d_model=128, num_ops=5, edit_budget=0.25)


def test_edit_head_forward(edit_head):
    B, S, d = 2, 10, 128
    hidden = torch.randn(B, S, d)
    
    edit_logits = edit_head(hidden)
    
    assert edit_logits.shape == (B, S, 5)
    assert not torch.isnan(edit_logits).any()


def test_edit_head_apply_edits_disabled(edit_head):
    B, S = 2, 10
    tokens = torch.randint(1, 100, (B, S))
    edit_logits = torch.randn(B, S, 5)
    pred_tokens = torch.randint(1, 100, (B, S))
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=False,
    )
    
    assert result.shape == tokens.shape
    assert torch.equal(result, tokens)


def test_edit_head_apply_edits_enabled(edit_head):
    B, S = 2, 10
    tokens = torch.randint(1, 100, (B, S))
    edit_logits = torch.randn(B, S, 5)
    pred_tokens = torch.randint(1, 100, (B, S))
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
    )
    
    assert result.shape[0] == B
    assert result.shape[1] >= S * 0.75
    assert result.shape[1] <= S * 1.25
    assert not torch.isnan(result).any()


def test_edit_head_apply_edits_with_replacement_mask(edit_head):
    B, S = 2, 10
    tokens = torch.randint(1, 100, (B, S))
    edit_logits = torch.randn(B, S, 5)
    pred_tokens = torch.randint(1, 100, (B, S))
    replacement_mask = torch.zeros(B, S)
    replacement_mask[:, :3] = 1
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
        replacement_mask=replacement_mask,
    )
    
    assert result.shape[0] == B
    assert not torch.isnan(result).any()


def test_edit_head_budget_constraint(edit_head):
    B, S = 1, 20
    tokens = torch.arange(1, S + 1).unsqueeze(0)
    
    edit_logits = torch.zeros(B, S, 5)
    edit_logits[:, :, 2] = 10.0
    
    pred_tokens = torch.full((B, S), 999)
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
    )
    
    max_inserts = int(S * edit_head.edit_budget)
    max_expected_length = S + max_inserts
    
    assert result.shape[1] <= max_expected_length + 1


def test_edit_head_keep_operation(edit_head):
    B, S = 1, 5
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    
    edit_logits = torch.zeros(B, S, 5)
    edit_logits[:, :, 0] = 10.0
    
    pred_tokens = torch.full((B, S), 999)
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
    )
    
    assert result.shape == tokens.shape
    assert torch.equal(result, tokens)


def test_edit_head_replace_operation(edit_head):
    B, S = 1, 5
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    
    edit_logits = torch.zeros(B, S, 5)
    edit_logits[:, :, 1] = 10.0
    
    pred_tokens = torch.tensor([[10, 20, 30, 40, 50]])
    
    replacement_mask = torch.ones(B, S)
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
        replacement_mask=replacement_mask,
    )
    
    max_replacements = int(S * edit_head.edit_budget)
    num_replaced = (result[0] != tokens[0]).sum().item()
    
    assert num_replaced <= max_replacements
    assert result.shape[1] == S


def test_edit_head_delete_operation(edit_head):
    B, S = 1, 10
    tokens = torch.arange(1, S + 1).unsqueeze(0)
    
    edit_logits = torch.zeros(B, S, 5)
    edit_logits[:, :3, 4] = 10.0
    
    pred_tokens = torch.full((B, S), 999)
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
    )
    
    assert result.shape[1] < S
    assert result.shape[0] == B

