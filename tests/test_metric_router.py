import torch
import pytest

from python.reality_stone.models.metric_router import MetricContextRouter, HAS_METRIKEY


def test_metric_router_shape_and_spd():
    """
    MetricContextRouter가 기본적인 shape을 만족하고,
    생성된 메트릭이 SPD(고유값 > 0)를 가지는지 확인한다.
    """
    router = MetricContextRouter(d_head=16)

    keys = ["topic:diagnosis|priority:high", "topic:general|priority:low"]
    scores = torch.tensor([[0.8, 0.2]])  # [B=1, T=2]

    L = router(keys, scores)

    # shape: [B, T, d_head, d_head]
    assert L.shape == (1, 2, 16, 16)

    # SPD 확인: 각 (B,T) 위치에서 L L^T 의 고유값이 모두 양수
    for b in range(L.shape[0]):
        for t in range(L.shape[1]):
            G = L[b, t] @ L[b, t].T
            # 대칭 여부
            assert torch.allclose(G, G.T, atol=1e-5)
            eigvals = torch.linalg.eigvalsh(G)
            assert torch.all(eigvals > 0)


@pytest.mark.skipif(HAS_METRIKEY, reason="Fallback 경로는 MetriKey가 없는 환경에서만 의미 있음")
def test_metric_router_identity_fallback_without_metrikey():
    """
    MetriKey 확장이 없는 환경에서는 MetricContextRouter가
    사실상 identity에 가까운 SPD 메트릭을 생성하는지 확인한다.
    """
    d_head = 8
    router = MetricContextRouter(d_head=d_head)

    keys = ["topic:diagnosis|priority:high"]
    scores = torch.tensor([[0.5]])

    L = router(keys, scores)  # [1,1,d_head,d_head]
    G = L[0, 0] @ L[0, 0].T

    # eigenvalue 범위가 설정된 [lambda_min, lambda_max] 안에 있는지만 확인
    eigvals = torch.linalg.eigvalsh(G)
    assert torch.all(eigvals >= router.lambda_min - 1e-5)
    assert torch.all(eigvals <= router.lambda_max + 1e-5)


