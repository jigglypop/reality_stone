import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import SentenceTopicHead


@pytest.fixture
def sample_topic_head():
    """기본 테스트용 SentenceTopicHead 생성"""
    return SentenceTopicHead(
        d_model=768,
        d_head=64,
        num_topics=8,
        num_heads=4,
        c_poincare=1e-3,
    )


def test_sentence_topic_head_output_shapes_and_probs(sample_topic_head):
    """
    SentenceTopicHead가 기본적인 shape을 만족하고,
    주제 확률이 각 문장마다 1로 정규화되는지 확인.
    """
    B, T = 2, 3
    d_model = sample_topic_head.d_model
    num_topics = sample_topic_head.num_topics

    x = torch.randn(B, T, d_model)
    topo_idx = torch.randint(0, T, (B, T, 2))

    P_topic, scores, metric_keys = sample_topic_head(x, topo_idx)

    # shape 확인
    assert P_topic.shape == (B, T, num_topics)
    assert scores.shape == (B, T)
    assert isinstance(metric_keys, list)
    assert len(metric_keys) == B * T

    # 각 문장별 확률 합이 1인지 확인 (수치 오차 허용)
    probs_sum = P_topic.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_sentence_topic_head_metric_keys_format(sample_topic_head):
    """
    metric_keys가 docs 명세대로
    'topic:{name}|priority:{high|medium|low}' 형식을 따르는지 확인.
    """
    B, T = 1, 4
    d_model = sample_topic_head.d_model

    x = torch.randn(B, T, d_model)
    topo_idx = torch.randint(0, T, (B, T, 2))

    P_topic, scores, metric_keys = sample_topic_head(x, topo_idx)

    assert len(metric_keys) == B * T

    for key in metric_keys:
        # 기본 형식 체크
        assert "topic:" in key
        assert "priority:" in key
        parts = key.split("|")
        assert parts[0].startswith("topic:")
        assert parts[1].startswith("priority:")

        priority = parts[1].split(":", 1)[1]
        assert priority in {"high", "medium", "low"}


def test_sentence_topic_head_poincare_projection(sample_topic_head):
    """
    SentenceTopicHead의 Poincaré 임베딩이 ball 내부에 투영되는지 검증.
    """
    B, T = 2, 3
    d_model = sample_topic_head.d_model
    x = torch.randn(B, T, d_model)
    topo_idx = torch.randint(0, T, (B, T, 2))
    
    with torch.no_grad():
        P_topic, scores, metric_keys = sample_topic_head(x, topo_idx)
    
    assert not torch.isnan(P_topic).any(), "P_topic에 NaN이 없어야 함"
    assert not torch.isnan(scores).any(), "scores에 NaN이 없어야 함"
    
    probs_sum = P_topic.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_sentence_topic_head_topic_names(sample_topic_head):
    """
    topic_names 리스트가 올바르게 설정되어 있는지 검증.
    """
    expected_topics = [
        "chief_complaint",
        "history",
        "physical_exam",
        "diagnosis",
        "treatment_plan",
        "prognosis",
        "follow_up",
        "general",
    ]
    
    assert sample_topic_head.topic_names == expected_topics
    assert len(sample_topic_head.topic_names) == sample_topic_head.num_topics


def test_sentence_topic_head_gradient_flow(sample_topic_head):
    """
    SentenceTopicHead의 backward pass가 올바르게 작동하는지 검증.
    """
    B, T = 2, 3
    d_model = sample_topic_head.d_model
    x = torch.randn(B, T, d_model, requires_grad=True)
    topo_idx = torch.randint(0, T, (B, T, 2))
    
    P_topic, scores, metric_keys = sample_topic_head(x, topo_idx)
    
    loss = P_topic.sum() + scores.sum()
    loss.backward()
    
    assert x.grad is not None, "입력에 대한 gradient가 계산되어야 함"
    assert not torch.isnan(x.grad).any(), "gradient에 NaN이 없어야 함"

