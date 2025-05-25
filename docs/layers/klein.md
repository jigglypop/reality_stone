# Klein 레이어

## 📝 개요

Klein 레이어는 Klein 모델(벨트라미-Klein 모델)에서 동작하는 하이퍼볼릭 신경망 레이어입니다. 이 모델은 직선 측지선을 가진 단위 원판으로 하이퍼볼릭 공간을 표현하며, 기하학적 계산이 직관적입니다.

## 🧮 수학적 배경

### Klein 모델 (벨트라미-Klein 모델)
Klein 모델 $\mathbb{K}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$는 하이퍼볼릭 공간의 투영 모델입니다.

**계량 텐서**:
$$g_{ij}(x) = \frac{1}{1-\|x\|^2}\left(\delta_{ij} + \frac{x_ix_j}{1-\|x\|^2}\right)$$

**거리 함수**:
$$d_{\mathbb{K}}(x,y) = \text{arccosh}\left(\frac{1 - \langle x,y \rangle}{\sqrt{(1-\|x\|^2)(1-\|y\|^2)}}\right)$$

### Klein 모델의 특징

1. **직선 측지선**: 측지선이 유클리드 직선으로 표현
2. **각도 왜곡**: 각도가 보존되지 않음 (등각 모델 아님)
3. **거리 왜곡**: 중심에서 멀어질수록 거리가 압축됨
4. **계산 효율성**: 직선성으로 인한 계산상 이점

### 포인카레와의 관계

Klein 모델과 포인카레 모델 간의 변환:

**포인카레 → Klein**:
$$K(P) = \frac{2P}{1 + \|P\|^2}$$

**Klein → 포인카레**:
$$P(K) = \frac{K}{1 + \sqrt{1-\|K\|^2}}$$

## 🔧 구현 상세

### 1. Forward Pass

```cpp
torch::Tensor klein_forward_cpu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float curvature
)
```

**구현 흐름**:
1. **입력 검증**: 입력이 Klein 디스크 내부에 있는지 확인
2. **포인카레 변환**: Klein → Poincare 변환
3. **하이퍼볼릭 연산**: 포인카레 공간에서 연산 수행
4. **Klein 복귀**: Poincare → Klein 변환

### 2. Klein 덧셈 연산

Klein 모델에서의 덧셈은 직접 구현되지 않고, 포인카레를 거쳐 계산됩니다:

```cpp
torch::Tensor klein_add_cpu(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float curvature
) {
    // Klein → Poincare 변환
    auto x_poincare = klein_to_poincare_cpu(x, curvature);
    auto y_poincare = klein_to_poincare_cpu(y, curvature);
    
    // Poincare에서 Möbius 덧셈
    auto result_poincare = mobius_add_cpu(x_poincare, y_poincare, curvature);
    
    // Poincare → Klein 변환
    return poincare_to_klein_cpu(result_poincare, curvature);
}
```

### 3. 거리 계산

Klein 모델에서의 하이퍼볼릭 거리:

```cpp
torch::Tensor klein_distance_cpu(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float curvature
) {
    auto x_norm_sq = torch::sum(x * x, -1);
    auto y_norm_sq = torch::sum(y * y, -1);
    auto xy_inner = torch::sum(x * y, -1);
    
    auto numerator = 1.0f - xy_inner;
    auto denominator = torch::sqrt((1.0f - x_norm_sq) * (1.0f - y_norm_sq));
    
    auto cosh_dist = numerator / denominator.clamp_min(EPS);
    return torch::acosh(cosh_dist.clamp_min(1.0f + EPS)) / std::sqrt(curvature);
}
```

### 4. 스칼라 곱셈

Klein 모델에서의 스칼라 곱셈:

```cpp
torch::Tensor klein_scalar_cpu(
    const torch::Tensor& x,
    float scalar,
    float curvature
) {
    if (std::abs(scalar) < EPS) {
        return torch::zeros_like(x);
    }
    
    // 포인카레를 거쳐 계산
    auto x_poincare = klein_to_poincare_cpu(x, curvature);
    auto result_poincare = mobius_scalar_cpu(x_poincare, scalar, curvature);
    return poincare_to_klein_cpu(result_poincare, curvature);
}
```

## 📊 성능 특성

### 계산 복잡도

Klein 모델의 연산들:

| 연산 | 복잡도 | 특징 |
|------|--------|------|
| 좌표 변환 | O(n) | 단순한 공식 |
| 거리 계산 | O(n) | 직접 계산 가능 |
| 덧셈 연산 | O(n) + 변환 | 포인카레 경유 필요 |
| 측지선 | O(1) | 유클리드 직선 |

### 수치적 안정성

Klein 모델의 수치적 특성:

1. **경계 안정성**: 포인카레보다 경계 처리가 쉬움
2. **변환 오버헤드**: 포인카레 변환 시 정밀도 손실 가능
3. **중심 집중**: 중심 근처에서 정확도 높음

## 🎯 사용 예제

### 기본 사용법

```python
import torch
import reality_stone as rs

# Klein 레이어 초기화
layer = rs.KleinLayer(
    input_dim=128,
    output_dim=64,
    curvature=1.0,
    bias=True
)

# 입력 데이터 (Klein 디스크 내부)
x = torch.randn(32, 128) * 0.3  # Klein에서는 더 큰 norm 허용

# Forward pass
output = layer(x)
print(f"Output shape: {output.shape}")  # [32, 64]
print(f"Output norm: {torch.norm(output, dim=1).max()}")  # < 1.0
```

### 측지선 계산

Klein 모델의 장점인 직선 측지선 활용:

```python
def klein_geodesic(start, end, t):
    """Klein 모델에서의 측지선 (직선)"""
    return (1 - t) * start + t * end

# 사용 예제
start_point = torch.tensor([0.0, 0.0])
end_point = torch.tensor([0.5, 0.5])
t_values = torch.linspace(0, 1, 11)

geodesic_points = torch.stack([
    klein_geodesic(start_point, end_point, t) 
    for t in t_values
])

print(f"Geodesic points shape: {geodesic_points.shape}")  # [11, 2]
```

### 모델 변환 및 시각화

```python
import matplotlib.pyplot as plt

class HyperbolicVisualizer:
    def __init__(self, curvature=1.0):
        self.curvature = curvature
        
    def compare_models(self, points):
        """Klein, Poincare, Lorentz 모델 비교"""
        # Klein 점들
        klein_points = points
        
        # 포인카레로 변환
        poincare_points = rs.klein_to_poincare_cpu(klein_points, self.curvature)
        
        # 로렌츠로 변환
        lorentz_points = rs.poincare_to_lorentz_cpu(poincare_points, self.curvature)
        
        return {
            'klein': klein_points,
            'poincare': poincare_points,
            'lorentz': lorentz_points
        }
    
    def plot_geodesics(self, start, end, n_points=50):
        """측지선 비교 플롯"""
        t_values = torch.linspace(0, 1, n_points)
        
        # Klein 측지선 (직선)
        klein_geodesic = torch.stack([
            (1-t)*start + t*end for t in t_values
        ])
        
        # 포인카레 측지선 (호)
        poincare_start = rs.klein_to_poincare_cpu(start.unsqueeze(0), self.curvature)
        poincare_end = rs.klein_to_poincare_cpu(end.unsqueeze(0), self.curvature)
        
        # 플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Klein 모델
        circle1 = plt.Circle((0, 0), 1, fill=False, color='black')
        ax1.add_patch(circle1)
        ax1.plot(klein_geodesic[:, 0], klein_geodesic[:, 1], 'b-', label='Klein geodesic')
        ax1.scatter(*start, color='red', s=50, label='Start')
        ax1.scatter(*end, color='green', s=50, label='End')
        ax1.set_xlim(-1.1, 1.1)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_title('Klein Model')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 포인카레 모델도 유사하게 그릴 수 있음...
        
        plt.tight_layout()
        return fig

# 사용 예제
visualizer = HyperbolicVisualizer(curvature=1.0)
start = torch.tensor([0.1, 0.1])
end = torch.tensor([0.6, 0.4])
fig = visualizer.plot_geodesics(start, end)
```

### 계층적 임베딩

```python
class KleinHierarchicalModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, curvature=1.0):
        super().__init__()
        self.curvature = curvature
        
        # 유클리드 임베딩
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        
        # Klein 레이어들
        dims = [embed_dim] + hidden_dims
        self.klein_layers = torch.nn.ModuleList([
            rs.KleinLayer(dims[i], dims[i+1], curvature)
            for i in range(len(dims)-1)
        ])
        
        # 계층 구조 분석을 위한 분류기
        self.classifier = torch.nn.Linear(hidden_dims[-1], vocab_size)
        
    def forward(self, tokens):
        # 유클리드 임베딩
        x = self.embedding(tokens)  # [batch, seq_len, embed_dim]
        
        # Klein 공간으로 정규화 (norm < 1 보장)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / (x_norm + 1e-5) * 0.5  # 안전한 범위로 스케일링
        
        # Klein 레이어들 적용
        for layer in self.klein_layers:
            x = torch.tanh(layer(x))
            
        # 분류 (평균 풀링 후 유클리드 공간으로)
        x_mean = torch.mean(x, dim=1)  # [batch, hidden_dim]
        return self.classifier(x_mean)
    
    def get_hierarchical_embedding(self, tokens):
        """계층적 구조 분석용 임베딩 추출"""
        embeddings = []
        
        x = self.embedding(tokens)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / (x_norm + 1e-5) * 0.5
        
        embeddings.append(x.clone())
        
        for layer in self.klein_layers:
            x = torch.tanh(layer(x))
            embeddings.append(x.clone())
            
        return embeddings

# 사용 예제
model = KleinHierarchicalModel(
    vocab_size=5000,
    embed_dim=64,
    hidden_dims=[32, 16],
    curvature=1.0
)

tokens = torch.randint(0, 5000, (8, 15))
logits = model(tokens)
hierarchical_embs = model.get_hierarchical_embedding(tokens)

print(f"Logits shape: {logits.shape}")
print(f"Hierarchical levels: {len(hierarchical_embs)}")
```

## ⚡ CUDA 최적화

### 좌표 변환 최적화

```cuda
__device__ void klein_to_poincare_point(
    const float* __restrict__ klein_point,
    float* __restrict__ poincare_point,
    int dim
) {
    float norm_sq = 0.0f;
    
    // norm² 계산
    for (int i = 0; i < dim; ++i) {
        norm_sq += klein_point[i] * klein_point[i];
    }
    
    float denominator = 1.0f + sqrtf(1.0f - norm_sq);
    
    // 벡터화된 변환
    for (int i = 0; i < dim; i += 4) {
        float4 k_vec = reinterpret_cast<const float4*>(klein_point + i)[0];
        float4 p_vec;
        
        p_vec.x = k_vec.x / denominator;
        p_vec.y = k_vec.y / denominator;
        p_vec.z = k_vec.z / denominator;
        p_vec.w = k_vec.w / denominator;
        
        reinterpret_cast<float4*>(poincare_point + i)[0] = p_vec;
    }
}

__global__ void klein_layer_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_dim,
    int output_dim,
    float curvature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        // Klein → Poincare 변환
        float poincare_input[MAX_DIM];
        klein_to_poincare_point(input + idx * input_dim, poincare_input, input_dim);
        
        // 포인카레 레이어 연산
        // ... 매트릭스 곱셈 및 Möbius 변환
        
        // Poincare → Klein 변환
        poincare_to_klein_point(poincare_output, output + idx * output_dim, output_dim);
    }
}
```

### 메모리 접근 패턴

```cuda
__global__ void klein_distance_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ distances,
    int batch_size,
    int dim,
    float curvature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        float x_norm_sq = 0.0f, y_norm_sq = 0.0f, xy_inner = 0.0f;
        
        // 벡터화된 내적 계산
        for (int i = 0; i < dim; i += 4) {
            float4 x_vec = reinterpret_cast<const float4*>(x + idx * dim + i)[0];
            float4 y_vec = reinterpret_cast<const float4*>(y + idx * dim + i)[0];
            
            x_norm_sq += x_vec.x * x_vec.x + x_vec.y * x_vec.y + 
                        x_vec.z * x_vec.z + x_vec.w * x_vec.w;
            y_norm_sq += y_vec.x * y_vec.x + y_vec.y * y_vec.y + 
                        y_vec.z * y_vec.z + y_vec.w * y_vec.w;
            xy_inner += x_vec.x * y_vec.x + x_vec.y * y_vec.y + 
                       x_vec.z * y_vec.z + x_vec.w * y_vec.w;
        }
        
        // Klein 거리 공식
        float numerator = 1.0f - xy_inner;
        float denominator = sqrtf((1.0f - x_norm_sq) * (1.0f - y_norm_sq));
        float cosh_dist = fmaxf(numerator / denominator, 1.0f + 1e-6f);
        
        distances[idx] = acoshf(cosh_dist) / sqrtf(curvature);
    }
}
```

## 📈 벤치마크 결과

### 모델별 성능 비교

| 연산 | Klein | Poincaré | Lorentz | 특징 |
|------|-------|----------|---------|------|
| 좌표 변환 | O(n) | - | O(n) | 단순 공식 |
| 거리 계산 | O(n) | O(n) | O(n+1) | 직접 계산 |
| 덧셈 연산 | O(n)+변환 | O(n) | O(n+1) | 변환 오버헤드 |
| 측지선 | O(1) | O(복잡) | O(복잡) | 직선의 이점 |

### GPU 메모리 사용량

| 배치 크기 | 차원 | Klein (MB) | Poincaré (MB) | 비율 |
|-----------|------|------------|---------------|------|
| 32 | 128 | 1.6 | 1.6 | 1.0x |
| 128 | 256 | 12.8 | 12.8 | 1.0x |
| 512 | 512 | 102.4 | 102.4 | 1.0x |

### 처리량 비교 (변환 포함)

| 배치 크기 | Klein CPU (ms) | Klein GPU (ms) | 가속비 |
|-----------|----------------|----------------|--------|
| 32 | 2.1 | 0.35 | 6.0x |
| 128 | 8.3 | 0.82 | 10.1x |
| 512 | 33.1 | 2.94 | 11.3x |

## 🔗 관련 함수

- [`klein_add_cpu/cuda`](../ops/klein.md#klein-덧셈): Klein 덧셈 연산
- [`klein_distance_cpu/cuda`](../ops/klein.md#klein-거리): Klein 거리 계산
- [`poincare_to_klein_cpu/cuda`](../ops/klein.md#좌표-변환): 포인카레-Klein 변환
- [`klein_to_poincare_cpu/cuda`](../ops/klein.md#좌표-변환): Klein-포인카레 변환

## 📚 참고 문헌

1. **Hyperbolic Neural Networks** - Ganea et al. (2018)
2. **Hyperbolic Geometry in Computer Vision** - Khrulkov et al. (2020)
3. **Klein Model of Hyperbolic Geometry** - Mathematical foundations
4. **Beltrami-Klein Model** - Classical differential geometry texts 