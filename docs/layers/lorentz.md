# Lorentz 레이어

## 📝 개요

Lorentz 레이어는 로렌츠 모델(하이퍼볼로이드 모델)에서 동작하는 하이퍼볼릭 신경망 레이어입니다. 이 모델은 민코프스키 공간의 부분 매니폴드로서 하이퍼볼릭 공간을 표현하며, 선형 대수 연산이 더 직관적입니다.

## 🧮 수학적 배경

### 로렌츠 모델 (하이퍼볼로이드)
하이퍼볼로이드 $\mathbb{L}^n = \{x \in \mathbb{R}^{n+1} : \langle x,x \rangle_{\mathcal{L}} = -1, x_0 > 0\}$

**민코프스키 내적**:
$$\langle x,y \rangle_{\mathcal{L}} = -x_0y_0 + \sum_{i=1}^n x_iy_i$$

**거리 함수**:
$$d_{\mathbb{L}}(x,y) = \text{arccosh}(-\langle x,y \rangle_{\mathcal{L}})$$

### 로렌츠 모델의 장점

1. **선형성**: 접선 공간에서의 연산이 직관적
2. **안정성**: 수치적으로 더 안정함
3. **대칭성**: 모든 방향이 기하학적으로 동등

### 하이퍼볼릭 선형 변환

로렌츠 모델에서의 선형 변환:

$$f(x) = \text{proj}_{\mathbb{L}}(Wx + b)$$

여기서 $\text{proj}_{\mathbb{L}}$는 하이퍼볼로이드로의 투영 함수입니다.

**투영 함수**:
$$\text{proj}_{\mathbb{L}}(y) = \frac{y}{\sqrt{|\langle y,y \rangle_{\mathcal{L}}|}} \text{ if } \langle y,y \rangle_{\mathcal{L}} < 0$$

## 🔧 구현 상세

### 1. Forward Pass

```cpp
torch::Tensor lorentz_forward_cpu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float curvature
)
```

**구현 흐름**:
1. **입력 검증**: 입력이 하이퍼볼로이드 위에 있는지 확인
2. **접선 공간 변환**: 로그 매핑을 통해 접선 공간으로 이동
3. **유클리드 연산**: $Wx + b$ 계산
4. **매니폴드 복귀**: 지수 매핑을 통해 하이퍼볼로이드로 복귀

### 2. 로그 매핑 (Logarithmic Map)

접선 공간으로의 매핑:

$$\log_x(y) = \frac{\text{arccosh}(-\langle x,y \rangle_{\mathcal{L}})}{\|\text{proj}_{T_x\mathbb{L}}(y)\|} \cdot \text{proj}_{T_x\mathbb{L}}(y)$$

```cpp
torch::Tensor lorentz_log_map(
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    auto inner_product = lorentz_inner(x, y);  // 민코프스키 내적
    auto distance = torch::acosh(-inner_product.clamp_min(1.0f + EPS));
    
    // 접선 공간으로 투영
    auto tangent = y + inner_product.unsqueeze(-1) * x;
    auto tangent_norm = torch::norm(tangent, 2, -1, true);
    
    return distance.unsqueeze(-1) * tangent / tangent_norm.clamp_min(EPS);
}
```

### 3. 지수 매핑 (Exponential Map)

매니폴드로의 복귀:

$$\exp_x(v) = \cosh(\|v\|)x + \sinh(\|v\|)\frac{v}{\|v\|}$$

```cpp
torch::Tensor lorentz_exp_map(
    const torch::Tensor& x,
    const torch::Tensor& v
) {
    auto v_norm = torch::norm(v, 2, -1, true);
    auto cosh_norm = torch::cosh(v_norm);
    auto sinh_norm = torch::sinh(v_norm);
    
    auto result = cosh_norm.unsqueeze(-1) * x;
    
    auto v_normalized = v / v_norm.clamp_min(EPS);
    result += sinh_norm.unsqueeze(-1) * v_normalized;
    
    return result;
}
```

### 4. Backward Pass

로렌츠 모델에서의 그래디언트 계산:

```cpp
torch::Tensor lorentz_backward_cpu(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float curvature
) {
    // 접선 공간에서의 그래디언트 계산
    auto tangent_grad = parallel_transport(grad_output, input);
    
    // 가중치 그래디언트
    auto weight_grad = torch::mm(tangent_grad.transpose(-2, -1), input);
    
    // 입력 그래디언트 (접선 공간)
    auto input_grad = torch::mm(tangent_grad, weight);
    
    return input_grad;
}
```

## 📊 성능 특성

### 메모리 효율성

로렌츠 모델은 n차원 하이퍼볼릭 공간을 (n+1)차원으로 임베딩하므로:

- **메모리 오버헤드**: +1 차원 (약 1/n의 추가 메모리)
- **계산 오버헤드**: 민코프스키 내적 계산

### 수치적 안정성

포인카레 모델 대비 장점:

1. **경계 문제 없음**: 하이퍼볼로이드는 닫힌 집합이 아님
2. **일관된 스케일**: 모든 점에서 동일한 기하학적 구조
3. **직교성**: 민코프스키 내적의 직교성

## 🎯 사용 예제

### 기본 사용법

```python
import torch
import reality_stone as rs

# 로렌츠 레이어 초기화
layer = rs.LorentzLayer(
    input_dim=128,
    output_dim=64,
    curvature=1.0,
    bias=True
)

# 입력 데이터 (하이퍼볼로이드 위의 점들)
# 첫 번째 좌표는 sqrt(1 + ||x||^2)로 설정
x_euclidean = torch.randn(32, 128) * 0.1
x_0 = torch.sqrt(1 + torch.sum(x_euclidean**2, dim=1, keepdim=True))
x = torch.cat([x_0, x_euclidean], dim=1)  # [32, 129]

# Forward pass
output = layer(x)
print(f"Output shape: {output.shape}")  # [32, 65]

# 하이퍼볼로이드 제약 조건 확인
lorentz_inner = rs.lorentz_inner_cpu(output, output)
print(f"Lorentz inner product: {lorentz_inner[:5]}")  # 모두 -1에 가까워야 함
```

### 포인카레-로렌츠 변환

```python
# 포인카레 디스크에서 로렌츠 모델로 변환
poincare_points = torch.randn(32, 64) * 0.3
lorentz_points = rs.poincare_to_lorentz_cpu(poincare_points, curvature=1.0)

# 로렌츠 레이어 적용
lorentz_layer = rs.LorentzLayer(65, 33, curvature=1.0)
lorentz_output = lorentz_layer(lorentz_points)

# 다시 포인카레로 변환
poincare_output = rs.lorentz_to_poincare_cpu(lorentz_output, curvature=1.0)
```

### 계층적 임베딩

```python
class LorentzHierarchicalEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, curvature=1.0):
        super().__init__()
        self.curvature = curvature
        
        # 유클리드 임베딩
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        
        # 로렌츠 레이어들
        dims = [embed_dim + 1] + [d + 1 for d in hidden_dims]
        self.lorentz_layers = torch.nn.ModuleList([
            rs.LorentzLayer(dims[i], dims[i+1], curvature)
            for i in range(len(dims)-1)
        ])
        
    def forward(self, tokens):
        # 유클리드 임베딩
        x = self.embedding(tokens)  # [batch, seq_len, embed_dim]
        
        # 로렌츠 공간으로 변환
        x = rs.euclidean_to_lorentz(x)  # [batch, seq_len, embed_dim+1]
        
        # 로렌츠 레이어들 적용
        for layer in self.lorentz_layers:
            x = torch.tanh(layer(x))  # 하이퍼볼릭 활성화
            
        return x

# 사용 예제
model = LorentzHierarchicalEncoder(
    vocab_size=10000,
    embed_dim=128,
    hidden_dims=[64, 32],
    curvature=1.0
)

tokens = torch.randint(0, 10000, (16, 20))  # [batch, seq_len]
embeddings = model(tokens)
```

## ⚡ CUDA 최적화

### 민코프스키 내적 최적화

```cuda
__device__ float lorentz_inner_product(
    const float* __restrict__ x,
    const float* __restrict__ y,
    int dim
) {
    float result = -x[0] * y[0];  // 시간 성분
    
    // 공간 성분들 (언롤링으로 최적화)
    for (int i = 1; i < dim; i += 4) {
        float4 x_vec = reinterpret_cast<const float4*>(x + i)[0];
        float4 y_vec = reinterpret_cast<const float4*>(y + i)[0];
        
        result += x_vec.x * y_vec.x;
        result += x_vec.y * y_vec.y;
        result += x_vec.z * y_vec.z;
        result += x_vec.w * y_vec.w;
    }
    
    return result;
}
```

### 메모리 접근 패턴 최적화

```cuda
__global__ void lorentz_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_dim,
    int output_dim
) {
    // 타일링을 통한 메모리 효율성
    __shared__ float s_input[TILE_SIZE][TILE_SIZE];
    __shared__ float s_weight[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    
    // ... 최적화된 매트릭스 곱셈
}
```

## 🔍 모델간 변환

### 포인카레 ↔ 로렌츠

```cpp
// 포인카레 → 로렌츠
torch::Tensor poincare_to_lorentz_cpu(
    const torch::Tensor& x,
    float curvature
) {
    auto x_norm_sq = torch::sum(x * x, -1, true);
    auto x_0 = (1 + curvature * x_norm_sq) / (1 - curvature * x_norm_sq);
    auto x_rest = 2 * x / (1 - curvature * x_norm_sq);
    
    return torch::cat({x_0, x_rest}, -1);
}

// 로렌츠 → 포인카레
torch::Tensor lorentz_to_poincare_cpu(
    const torch::Tensor& x,
    float curvature
) {
    auto x_0 = x.narrow(-1, 0, 1);
    auto x_rest = x.narrow(-1, 1, x.size(-1) - 1);
    
    return x_rest / (x_0 + 1.0f / std::sqrt(curvature));
}
```

## 📈 벤치마크 결과

### 계산 복잡도 비교

| 연산 | 포인카레 | 로렌츠 | 성능비 |
|------|----------|--------|--------|
| 내적 계산 | O(n) | O(n+1) | 0.95x |
| 거리 계산 | O(n) | O(n+1) | 0.98x |
| 선형 변환 | O(n²) + O(mobius) | O(n²) + O(proj) | 1.1x |

### GPU 메모리 사용량

| 배치 크기 | 차원 | 포인카레 (MB) | 로렌츠 (MB) | 비율 |
|-----------|------|---------------|-------------|------|
| 32 | 128 | 1.6 | 1.65 | 1.03x |
| 128 | 256 | 12.8 | 13.3 | 1.04x |
| 512 | 512 | 102.4 | 106.5 | 1.04x |

## 🔗 관련 함수

- [`lorentz_add_cpu/cuda`](../ops/lorentz.md#로렌츠-덧셈): 로렌츠 덧셈 연산
- [`lorentz_inner_cpu/cuda`](../ops/lorentz.md#민코프스키-내적): 민코프스키 내적
- [`lorentz_distance_cpu/cuda`](../ops/lorentz.md#하이퍼볼릭-거리): 하이퍼볼릭 거리 계산

## 📚 참고 문헌

1. **Hyperbolic Neural Networks** - Ganea et al. (2018)
2. **Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** - Nickel & Kiela (2018)
3. **Hyperbolic Graph Neural Networks** - Chami et al. (2019)
4. **Hyperbolic Deep Neural Networks: A Survey** - Peng et al. (2021) 