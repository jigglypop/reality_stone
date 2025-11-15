import torch

from python.reality_stone.models.sentence_topic_head import SentenceTopicHead


def test_sentence_topic_head_output_shapes_and_probs():
    """
    SentenceTopicHead가 기본적인 shape을 만족하고,
    주제 확률이 각 문장마다 1로 정규화되는지 확인.
    """
    B, T, d_model = 2, 3, 768
    d_head = 64
    num_topics = 8

    model = SentenceTopicHead(
        d_model=d_model,
        d_head=d_head,
        num_topics=num_topics,
        num_heads=4,
    )

    x = torch.randn(B, T, d_model)
    topo_idx = torch.randint(0, T, (B, T, 2))

    P_topic, scores, metric_keys = model(x, topo_idx)

    # shape 확인
    assert P_topic.shape == (B, T, num_topics)
    assert scores.shape == (B, T)
    assert isinstance(metric_keys, list)
    assert len(metric_keys) == B * T

    # 각 문장별 확률 합이 1인지 확인 (수치 오차 허용)
    probs_sum = P_topic.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_sentence_topic_head_metric_keys_format():
    """
    metric_keys가 docs 명세대로
    'topic:{name}|priority:{high|medium|low}' 형식을 따르는지 확인.
    """
    B, T, d_model = 1, 4, 768

    model = SentenceTopicHead(
        d_model=d_model,
        d_head=64,
        num_topics=8,
        num_heads=4,
    )

    x = torch.randn(B, T, d_model)
    topo_idx = torch.randint(0, T, (B, T, 2))

    P_topic, scores, metric_keys = model(x, topo_idx)

    assert len(metric_keys) == B * T

    for key in metric_keys:
        # 기본 형식 체크
        assert "topic:" in key
        assert "priority:" in key
        parts = key.split("|")
        assert parts[0].startswith("topic:")
        assert parts[1].startswith("priority:")

        # priority 값이 high/medium/low 중 하나인지 확인
        priority = parts[1].split(":", 1)[1]
        assert priority in {"high", "medium", "low"}


