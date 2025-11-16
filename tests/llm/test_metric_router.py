import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import MetricContextRouter

try:
    import reality_stone.metrikey as _metrikey
    HAS_METRIKEY = True
except Exception:
    HAS_METRIKEY = False


@pytest.fixture
def sample_router():
    """기본 테스트용 MetricContextRouter 생성"""
    return MetricContextRouter(d_head=16, lambda_min=0.5, lambda_max=2.0)


def test_metric_router_shape_and_spd(sample_router):
    """
    MetricContextRouter가 기본적인 shape을 만족하고,
    생성된 메트릭이 SPD(고유값 > 0)를 가지는지 확인한다.
    """
    keys = ["topic:diagnosis|priority:high", "topic:general|priority:low"]
    scores = torch.tensor([[0.8, 0.2]])  # [B=1, T=2]

    L = sample_router(keys, scores)

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

    eigvals = torch.linalg.eigvalsh(G)
    assert torch.all(eigvals >= router.lambda_min - 1e-5)
    assert torch.all(eigvals <= router.lambda_max + 1e-5)


def test_metric_router_cache_functionality(sample_router):
    """
    MetricContextRouter의 LRU 캐시가 올바르게 동작하는지 검증.
    동일한 key/score 조합은 캐시에서 반환되어야 함.
    """
    keys = ["topic:diagnosis|priority:high"]
    scores = torch.tensor([[1.0]])
    
    L1 = sample_router(keys, scores)
    
    cache_size_before = len(sample_router._cache)
    
    L2 = sample_router(keys, scores)
    
    cache_size_after = len(sample_router._cache)
    
    assert cache_size_before == cache_size_after, "동일 키는 캐시에서 가져와야 함"
    assert torch.allclose(L1, L2), "캐시된 값은 동일해야 함"


def test_metric_router_score_quantization(sample_router):
    """
    score 값이 quantize되어 캐시 효율성이 높아지는지 검증.
    """
    keys = ["topic:treatment|priority:medium"]
    
    scores1 = torch.tensor([[0.501]])
    scores2 = torch.tensor([[0.499]])
    
    L1 = sample_router(keys, scores1)
    L2 = sample_router(keys, scores2)
    
    assert torch.allclose(L1, L2), "근접한 score는 quantize되어 같은 메트릭 반환"

