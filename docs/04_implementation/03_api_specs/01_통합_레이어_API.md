# 통합 리만 레이어 API 레퍼런스 (Unified Riemannian Layer API Reference)

## Python API

### UnifiedRiemannianLayer

통합 리만 레이어 - 푸앵카레/로렌츠/클라인/대각 메트릭을 하나의 인터페이스로 제공

#### 생성자

```python
import reality_stone as rs

layer = rs.UnifiedRiemannianLayer(
    metric_type: str,
    curvature: float = 1.0,
    input_dim: int = 64,
    enable_bellman: bool = False,
    gamma: float = 0.99
)
```