import torch
from reality_stone.layers.metric_attention import MetricAttention


def test_metric_attention_dot_product_basic():
    """
    Dot-product 모드에서 MetricAttention이 기본적인 shape을 만족하고
    NaN 없이 동작하는지 확인한다.
    """
    B, H, T, S, d_h, d_v = 2, 4, 5, 7, 16, 32
    q = torch.randn(B, H, T, d_h)
    k = torch.randn(B, H, S, d_h)
    v = torch.randn(B, H, S, d_v)
    attn = MetricAttention(
        hidden_size=d_h,
        normalizer="softmax",
        rank=0,
        mode="dot",
        manifold="poincare",
        c=1e-3,
    )
    y = attn(q, k, v)
    # 출력 shape: (B, H, T, d_v)
    assert y.shape == (B, H, T, d_v)
    # NaN 또는 Inf 가 없어야 한다
    assert torch.isfinite(y).all()


def test_metric_attention_geodesic_with_topology():
    """
    Geodesic 모드 + topology index 사용 시에도
    출력 shape과 수치 안정성이 보장되는지 확인한다.
    """
    B, H, T, d_h, d_v = 1, 2, 4, 8, 16

    # geodesic 모드는 T==S 인 self-attention 케이스가 자연스럽다
    q = torch.randn(B, H, T, d_h)
    k = torch.randn(B, H, T, d_h)
    v = torch.randn(B, H, T, d_v)

    # 각 토큰의 이웃을 간단히 "자기 자신 + 다음 토큰"으로 설정
    idx = torch.empty(B, T, 2, dtype=torch.long)
    for t in range(T):
        idx[0, t, 0] = t
        idx[0, t, 1] = min(T - 1, t + 1)

    topo_idx = {"neighbor": idx}
    topk_cfg = {"neighbor": 2}

    attn = MetricAttention(
        hidden_size=d_h,
        normalizer="softmax",
        rank=0,
        mode="geodesic",
        manifold="poincare",
        c=1e-3,
    )

    y = attn(
        q,
        k,
        v,
        topo_idx=topo_idx,
        topk_cfg=topk_cfg,
    )

    # 출력 shape: (B, H, T, d_v)
    assert y.shape == (B, H, T, d_v)
    # NaN 또는 Inf 가 없어야 한다
    assert torch.isfinite(y).all()


