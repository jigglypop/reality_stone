# Klein 연산 (Klein Disk Model)

Klein 디스크 모델에서의 핵심 연산들을 구현합니다. Klein 모델은 직선 측지선을 특징으로 하는 하이퍼볼릭 기하학 모델입니다.

## 📐 수학적 배경

### Klein 디스크 모델 (Klein Disk Model)
Klein 디스크 $\mathbb{K}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$는 측지선이 유클리드 직선으로 표현되는 하이퍼볼릭 공간 모델입니다.

**계량 텐서**:
$$g_{ij} = \frac{\delta_{ij}}{1-\|x\|^2} + \frac{x_ix_j}{(1-\|x\|^2)^2}$$

**특징**:
- 측지선이 직선으로 표현됨 (각도 보존은 안됨)
- 경계에서의 계산이 비교적 안정적
- 포인카레 모델보다 일부 계산이 단순함

### Klein 덧셈 (Klein Addition)

Klein 모델에서의 "덧셈"은 Poincare 모델로 변환 후 Möbius 덧셈을 수행:

$$u \oplus_K v = \text{P2K}(\text{K2P}(u) \oplus_P \text{K2P}(v))$$

**직접 공식**:
$$u \oplus_K v = \frac{(1+\sqrt{1-\|v\|^2})u + (1+\sqrt{1-\|u\|^2})v}{1 + \sqrt{(1-\|u\|^2)(1-\|v\|^2)} + \langle u,v \rangle}$$

### Klein 거리 (Klein Distance)

두 점 $u, v \in \mathbb{K}^n$ 사이의 거리:

$$d_K(u,v) = \frac{1}{2} \ln\left(\frac{(1-\langle u,v \rangle)^2 - (\|u\|^2-1)(\|v\|^2-1)}{(\langle u,v \rangle - 1)^2 - (\|u\|^2-1)(\|v\|^2-1)}\right)$$

**단순화된 형태**:
$$d_K(u,v) = \text{arccosh}\left(\frac{1-\langle u,v \rangle}{\sqrt{(1-\|u\|^2)(1-\|v\|^2)}}\right)$$

## 🔄 모델 간 변환

### Klein ↔ Poincare 변환

**Klein → Poincare**:
$$\text{K2P}(x) = \frac{x}{1 + \sqrt{1-\|x\|^2}}$$

**Poincare → Klein**:
$$\text{P2K}(x) = \frac{2x}{1 + \|x\|^2}$$

### Klein ↔ Lorentz 변환

**Klein → Lorentz**:
$$\text{K2L}(x) = \frac{1}{\sqrt{1-\|x\|^2}}\left(1, x\right)$$

**Lorentz → Klein**:
$$\text{L2K}(x) = \frac{(x_1, x_2, \ldots, x_n)}{x_0}$$

## 🔧 구현 세부사항

### 파일 구조
```
src/core/ops/
├── klein_cpu.cpp      # CPU 구현
└── klein_cuda.cu      # CUDA 구현

src/include/ops/
└── klein.h            # 함수 선언
```

### CPU 구현 (`klein_cpu.cpp`)

```cpp
torch::Tensor klein_add_cpu(torch::Tensor u, torch::Tensor v, float c) {
    // Klein → Poincare → Klein 변환을 통한 덧셈
    auto u_poincare = klein_to_poincare_cpu(u, c);
    auto v_poincare = klein_to_poincare_cpu(v, c);
    
    // Poincare 공간에서 Möbius 덧셈
    auto result_poincare = mobius_add_cpu(u_poincare, v_poincare, c);
    
    // 다시 Klein 모델로 변환
    return poincare_to_klein_cpu(result_poincare, c);
}

torch::Tensor klein_distance_cpu(torch::Tensor u, torch::Tensor v, float c) {
    auto u_dot_v = torch::sum(u * v, -1);
    auto u_norm_sq = torch::sum(u * u, -1);
    auto v_norm_sq = torch::sum(v * v, -1);
    
    // 수치적 안정성을 위한 클리핑
    auto numerator = 1 - u_dot_v;
    auto denominator_sq = (1 - u_norm_sq) * (1 - v_norm_sq);
    auto denominator = torch::sqrt(torch::clamp(denominator_sq, 1e-8f));
    
    auto ratio = torch::clamp(numerator / denominator, 1.0f + 1e-6f);
    return torch::acosh(ratio) / std::sqrt(c);
}

torch::Tensor klein_to_poincare_cpu(torch::Tensor x, float c) {
    auto x_norm_sq = torch::sum(x * x, -1, true);
    auto denominator = 1 + torch::sqrt(1 - c * x_norm_sq);
    return x / denominator;
}

torch::Tensor poincare_to_klein_cpu(torch::Tensor x, float c) {
    auto x_norm_sq = torch::sum(x * x, -1, true);
    auto factor = 2 / (1 + c * x_norm_sq);
    return factor * x;
}
```

### CUDA 구현 (`klein_cuda.cu`)

```cuda
__global__ void klein_add_kernel(
    const float* u, const float* v, float* result,
    float c, int batch_size, int dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    const float* u_batch = u + tid * dim;
    const float* v_batch = v + tid * dim;
    float* result_batch = result + tid * dim;
    
    // Klein → Poincare 변환
    float u_norm_sq = 0.0f, v_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        u_norm_sq += u_batch[i] * u_batch[i];
        v_norm_sq += v_batch[i] * v_batch[i];
    }
    
    float u_denom = 1.0f + sqrtf(fmaxf(1.0f - c * u_norm_sq, 1e-8f));
    float v_denom = 1.0f + sqrtf(fmaxf(1.0f - c * v_norm_sq, 1e-8f));
    
    // Poincare 공간에서 임시 벡터
    float u_poincare[MAX_DIM], v_poincare[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        u_poincare[i] = u_batch[i] / u_denom;
        v_poincare[i] = v_batch[i] / v_denom;
    }
    
    // Möbius 덧셈 수행
    float uv_dot = 0.0f;
    float up_norm_sq = 0.0f, vp_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        uv_dot += u_poincare[i] * v_poincare[i];
        up_norm_sq += u_poincare[i] * u_poincare[i];
        vp_norm_sq += v_poincare[i] * v_poincare[i];
    }
    
    float factor1 = 1.0f + 2.0f * c * uv_dot + c * vp_norm_sq;
    float factor2 = 1.0f - c * up_norm_sq;
    float denom = 1.0f + 2.0f * c * uv_dot + c * c * up_norm_sq * vp_norm_sq;
    denom = fmaxf(denom, 1e-8f);
    
    float result_poincare[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        result_poincare[i] = (factor1 * u_poincare[i] + factor2 * v_poincare[i]) / denom;
    }
    
    // Poincare → Klein 변환
    float rp_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        rp_norm_sq += result_poincare[i] * result_poincare[i];
    }
    
    float klein_factor = 2.0f / (1.0f + c * rp_norm_sq);
    for (int i = 0; i < dim; i++) {
        result_batch[i] = klein_factor * result_poincare[i];
    }
}

__global__ void klein_distance_kernel(
    const float* u, const float* v, float* distances,
    float c, int batch_size, int dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    const float* u_batch = u + tid * dim;
    const float* v_batch = v + tid * dim;
    
    float u_dot_v = 0.0f;
    float u_norm_sq = 0.0f, v_norm_sq = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        u_dot_v += u_batch[i] * v_batch[i];
        u_norm_sq += u_batch[i] * u_batch[i];
        v_norm_sq += v_batch[i] * v_batch[i];
    }
    
    float numerator = 1.0f - u_dot_v;
    float denominator_sq = (1.0f - u_norm_sq) * (1.0f - v_norm_sq);
    float denominator = sqrtf(fmaxf(denominator_sq, 1e-8f));
    
    float ratio = fmaxf(numerator / denominator, 1.0f + 1e-6f);
    distances[tid] = acoshf(ratio) / sqrtf(c);
}
```

## 📊 성능 비교

### 모델별 연산 복잡도

| 연산 | Klein | Poincare | Lorentz |
|------|-------|----------|---------|
| 덧셈 | O(d) | O(d) | O(d) |
| 거리 | O(d) | O(d) | O(d) |
| 좌표변환 | O(d) | O(d) | O(d) |
| 수치 안정성 | ★★★ | ★★ | ★★★ |

### 장단점 분석

**Klein 모델 장점**:
- 측지선이 직선으로 표현되어 기하학적 직관성이 높음
- 경계 근처에서 비교적 안정적
- 일부 거리 계산이 단순함

**Klein 모델 단점**:
- 각도 보존 안됨 (conformal하지 않음)
- 덧셈 연산이 다른 모델 변환을 거쳐야 함
- 일부 복잡한 연산에서 효율성 떨어짐

## 🧪 테스트 케이스

### 수학적 성질 검증

```cpp
void test_klein_properties() {
    auto x = torch::randn({100, 64}) * 0.8;  // Klein 내부 점들
    auto y = torch::randn({100, 64}) * 0.8;
    
    // 1. 좌표 변환 가역성
    auto x_poincare = klein_to_poincare_cpu(x, 1.0);
    auto x_back = poincare_to_klein_cpu(x_poincare, 1.0);
    auto diff1 = torch::max(torch::abs(x - x_back));
    assert(diff1.item<float>() < 1e-5);
    
    // 2. 거리 보존성 (변환 후에도 거리 동일)
    auto d_klein = klein_distance_cpu(x, y, 1.0);
    auto d_poincare = poincare_distance(x_poincare, 
                                       klein_to_poincare_cpu(y, 1.0), 1.0);
    auto diff2 = torch::max(torch::abs(d_klein - d_poincare));
    assert(diff2.item<float>() < 1e-4);
    
    // 3. 경계 조건 (모든 점이 단위원 내부)
    auto norms = torch::norm(x, 2, -1);
    assert(torch::all(norms < 0.99).item<bool>());
}
```

### 수치적 안정성 테스트

```cpp
void test_numerical_stability() {
    // 경계 근처 점들로 테스트
    auto x_boundary = torch::ones({10, 64}) * 0.99;
    auto y_boundary = torch::ones({10, 64}) * 0.98;
    
    // NaN, Inf 발생하지 않는지 확인
    auto result = klein_add_cpu(x_boundary, y_boundary, 1.0);
    assert(!torch::any(torch::isnan(result)).item<bool>());
    assert(!torch::any(torch::isinf(result)).item<bool>());
    
    // 결과가 여전히 Klein 디스크 내부에 있는지 확인
    auto result_norms = torch::norm(result, 2, -1);
    assert(torch::all(result_norms < 1.0).item<bool>());
}
```

## 🎯 응용 분야

### 1. 하이퍼볼릭 임베딩
```python
# 계층적 데이터 임베딩
def hierarchical_embedding(data, dim=64):
    # Klein 모델의 직선 측지선 활용
    embeddings = klein_embedding_layer(data, dim)
    return embeddings
```

### 2. 그래프 분석
```python
# 트리 구조 분석에서 Klein 모델 활용
def tree_distance_analysis(tree_nodes):
    klein_coords = embed_tree_to_klein(tree_nodes)
    distances = klein_distance_cuda(klein_coords, klein_coords, c=1.0)
    return analyze_tree_structure(distances)
```

### 3. 최적화 알고리즘
```python
# Klein 공간에서의 gradient descent
def klein_gradient_descent(x, grad, lr, c=1.0):
    # Klein → Poincare → 최적화 → Klein
    x_poincare = klein_to_poincare_cuda(x, c)
    x_updated = riemannian_sgd_step_cuda(x_poincare, grad, lr, c)
    return poincare_to_klein_cuda(x_updated, c)
```

## 🔗 관련 함수들

- `klein_scalar_cpu/cuda`: Klein 스칼라 곱셈
- `klein_exp_map`: Klein 지수 맵핑
- `klein_log_map`: Klein 로그 맵핑  
- `klein_midpoint`: Klein 중점 계산
- `klein_reflection`: Klein 반사 변환

## 📚 참고 문헌

1. **Klein Disk Model** - Wikipedia Mathematics
2. **Hyperbolic Geometry** - Cannon et al. (1997)
3. **Models of Hyperbolic Geometry** - Stillwell (1996)
4. **Riemannian Computing in Computer Vision** - Turaga et al. (2011)
5. **Hyperbolic Neural Networks** - Ganea et al. (2018) 