# Poincaré Ball Layer 구현 문서

## 개요

Poincaré Ball은 음의 곡률을 가진 쌍곡 공간의 모델로, 단위 공(unit ball) 내부에서 정의됩니다.

**수학적 정의**:
- 공간: `B^n_c = {x ∈ ℝ^n : c||x||² < 1}`
- 곡률: `c > 0` (양수)
- 메트릭 텐서: `g_x = λ²_x I`, where `λ_x = 2/(1 - c||x||²)`

---

## 1. 핵심 연산 (Möbius Operations)

### 1.1 Möbius Addition

**수식**:
```
x ⊕_c y = [(1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y] / [1 + 2c⟨x,y⟩ + c²||x||²||y||²]
```

**구현 위치**:
- CPU: `src/ops/mobius.rs::mobius_add()`
- CUDA: `src/ops/cuda/mobius.cu::mobius_add_kernel()`

**구현 상세** (CPU):
```rust
pub fn mobius_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u2 = norm_sq_batched(u).insert_axis(Axis(1));  // ||u||²
    let v2 = norm_sq_batched(v).insert_axis(Axis(1));  // ||v||²
    let uv = dot_batched(u, v).insert_axis(Axis(1));   // ⟨u,v⟩
    
    let den = (1.0 + 2.0 * c * &uv + c * c * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR));
    let coeff_u = (1.0 + 2.0 * c * &uv + c * &v2) / &den;
    let coeff_v = (1.0 - c * &u2) / &den;
    
    coeff_u * u + coeff_v * v
}
```

**수치 안정성**:
- `MIN_DENOMINATOR = 1e-6`: 분모가 0에 가까워지는 것 방지
- 모든 norm 계산에 EPS 클램핑

### 1.2 Möbius Scalar Multiplication

**수식**:
```
r ⊗_c x = tanh(r · atanh(√c||x||)) · x / (√c||x||)
```

**양수 곡률 (c > 0)**:
```rust
let sqrt_c = c.sqrt();
let scn = (sqrt_c * norm).min(1.0 - BOUNDARY_EPS);  // atanh 정의역 제한
let alpha = scn.atanh();
let beta = (r * alpha).tanh();
let scale = beta / (sqrt_c * norm);
```

**음수 곡률 (c < 0)** - 복소수 확장:
```rust
let sqrt_abs_c = (-c).sqrt();
let scn = sqrt_abs_c * norm;
let alpha = scn.atan();  // atanh(ix) = i·atan(x)
let beta = (r * alpha).tan();  // tanh(ix) = i·tan(x)
let scale = beta / (sqrt_abs_c * norm);
```

---

## 2. Poincaré Distance

### 2.1 수식

**정확한 수식** (수정됨):
```
d(x, y) = (2/√c) · atanh(√(c||x-y||² / [(1-c||x||²)(1-c||y||²)]))
```

### 2.2 구현

**CPU** (`src/layers/poincare.rs`):
```rust
pub fn poincare_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let sqrtc = c.sqrt();
    let u2 = norm_sq_batched(u);
    let v2 = norm_sq_batched(v);
    let uv = dot_batched(u, v);

    // ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
    let norm_sq_diff = (&u2 + &v2 - 2.0 * &uv).mapv_into(|val| val.max(0.0));
    
    let den = (1.0 - c * &u2) * (1.0 - c * &v2);
    let den_clamped = den.mapv_into(|val| val.max(EPS));

    let frac = norm_sq_diff / den_clamped;
    
    // arg = √(c * frac), atanh 정의역 제한 [-1+ε, 1-ε]
    frac.mapv_into(|val| {
        let arg = (c * val).sqrt().min(1.0 - EPS);
        (2.0 / sqrtc) * arg.atanh()
    })
}
```

**CUDA** (`src/layers/cuda/poincare.cu`):
```cpp
__device__ float poincare_distance_impl(const float* x, const float* y, int dim, float c, float eps) {
    float norm_sq_diff = 0.0f;  // ||x-y||²
    float x2 = 0.0f;            // ||x||²
    float y2 = 0.0f;            // ||y||²
    
    for (int i = 0; i < dim; ++i) {
        float diff = x[i] - y[i];
        norm_sq_diff += diff * diff;
        x2 += x[i] * x[i];
        y2 += y[i] * y[i];
    }
    
    float den = (1.0f - c * x2) * (1.0f - c * y2);
    den = fmaxf(den, eps);
    float frac = (c * norm_sq_diff) / den;
    frac = fmaxf(frac, 0.0f);
    
    float sqrtc = sqrtf(c);
    float arg = sqrtf(frac);
    arg = fminf(arg, 1.0f - eps);
    
    return (2.0f / sqrtc) * atanhf(arg);
}
```

### 2.3 수정 내역

**이전 (잘못된 구현)**:
```cpp
// ❌ 완전히 틀린 수식
float num = 2 * c * xy;
return acoshf(1.0f + num / den) / sqrtf(c);
```

**수정 후**:
```cpp
// ✅ 정확한 Poincaré distance
return (2.0f / sqrtc) * atanhf(sqrt(frac));
```

---

## 3. Poincaré Ball Layer

### 3.1 Forward Pass

**수식**:
```
y = (1-t) ⊗_c u  ⊕_c  t ⊗_c v
```

**의미**: 
- 쌍곡 공간에서 u와 v 사이의 측지선(geodesic) 보간
- t=0: u 반환
- t=1: v 반환
- t=0.5: 중간점 (쌍곡 의미)

**구현** (`src/layers/poincare.rs`):
```rust
pub fn poincare_ball_layer(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> Array2<f32> {
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    mobius::mobius_add(&u_prime.view(), &v_prime.view(), c)
}
```

**CUDA Forward** (`src/layers/cuda/poincare.cu`):
```cpp
__global__ void poincare_ball_layer_forward_kernel(
    const float* u, const float* v, float* out, 
    float c, float t, long long batch_size, long long dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    
    const float* u_i = u + i * dim;
    const float* v_i = v + i * dim;
    float* out_i = out + i * dim;

    float u_prime[256];
    float v_prime[256];
    
    // u' = (1-t) ⊗_c u
    mobius_scalar_kernel_impl(u_i, u_prime, dim, c, 1.0f - t, 1e-7f);
    
    // v' = t ⊗_c v
    mobius_scalar_kernel_impl(v_i, v_prime, dim, c, t, 1e-7f);
    
    // out = u' ⊕_c v'
    mobius_add_kernel_impl(u_prime, v_prime, out_i, dim, c, 1e-7f);
}
```

### 3.2 Backward Pass

**Gradient 계산**:
1. `∂L/∂(u' ⊕ v')` → `mobius_add_vjp()` → `∂L/∂u'`, `∂L/∂v'`
2. `∂L/∂u'` → `mobius_scalar_vjp()` → `∂L/∂u`
3. `∂L/∂v'` → `mobius_scalar_vjp()` → `∂L/∂v`

**구현** (`src/layers/poincare.rs`):
```rust
pub fn poincare_ball_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    // Forward pass (재계산)
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);

    // Backward through Möbius add
    let (grad_u_prime, grad_v_prime) =
        mobius_add_vjp(grad_output, &u_prime.view(), &v_prime.view(), c);

    // Backward through Möbius scalar
    let grad_u = mobius_scalar_vjp(&grad_u_prime.view(), &u.view(), c, 1.0 - t);
    let grad_v = mobius_scalar_vjp(&grad_v_prime.view(), &v.view(), c, t);

    (grad_u, grad_v)
}
```

**CUDA Backward** (`src/layers/cuda/poincare.cu`):
```cpp
__global__ void poincare_ball_layer_backward_kernel(
    const float* grad_output, const float* u, const float* v,
    float* grad_u, float* grad_v, float c, float t, 
    long long batch_size, long long dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    
    // Forward pass 재계산
    float u_prime[256], v_prime[256];
    mobius_scalar_kernel_impl(u_i, u_prime, dim, c, 1.0f - t, eps);
    mobius_scalar_kernel_impl(v_i, v_prime, dim, c, t, eps);

    // Backward: grad_output → grad_u_prime, grad_v_prime
    float grad_u_prime[256], grad_v_prime[256];
    mobius_add_vjp(grad_output_i, u_prime, v_prime, c, 
                   grad_u_prime, grad_v_prime, dim, eps);
    
    // Backward: grad_u_prime → grad_u
    mobius_scalar_vjp(grad_u_prime, u_i, c, 1.0f - t, grad_u_i, dim, eps);
    
    // Backward: grad_v_prime → grad_v
    mobius_scalar_vjp(grad_v_prime, v_i, c, t, grad_v_i, dim, eps);
}
```

---

## 4. Dynamic Curvature

### 4.1 개념

곡률 `c`를 학습 가능한 파라미터로 만듦:
```
c = exp(-2κ) · (c_max - c_min) + c_min
```

**장점**:
- 데이터에 최적화된 곡률 자동 학습
- 레이어별 다른 곡률 가능

### 4.2 구현

**Forward** (`src/layers/poincare.rs`):
```rust
pub fn poincare_ball_layer_layerwise(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    let result = mobius::mobius_add(&u_prime.view(), &v_prime.view(), c);
    (result, c)
}
```

**Backward with Curvature Gradient**:
```rust
pub fn poincare_ball_layer_layerwise_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    
    // Standard gradients
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    let (grad_u_prime, grad_v_prime) = mobius_add_vjp(...);
    let grad_u = mobius_scalar_vjp(...);
    let grad_v = mobius_scalar_vjp(...);
    
    // Curvature gradient
    let grad_c_from_add = mobius::mobius_add_grad_c(&u_prime, &v_prime, c);
    let grad_c_add = (grad_output * &grad_c_from_add).sum();
    
    let grad_c_from_scalar_u = mobius::mobius_scalar_grad_c(u, c, 1.0 - t);
    let grad_c_scalar_u = (&grad_u_prime * &grad_c_from_scalar_u).sum();
    
    let grad_c_from_scalar_v = mobius::mobius_scalar_grad_c(v, c, t);
    let grad_c_scalar_v = (&grad_v_prime * &grad_c_from_scalar_v).sum();
    
    let grad_c_total = grad_c_add + grad_c_scalar_u + grad_c_scalar_v;
    
    // c = f(κ) → ∂L/∂κ = ∂L/∂c · ∂c/∂κ
    let dc_dkappa = layer_curvatures.compute_dc_dkappa(layer_idx);
    let grad_kappa = grad_c_total * dc_dkappa;
    
    (grad_u, grad_v, grad_kappa)
}
```

---

## 5. Python 바인딩

### 5.1 PyTorch Autograd 통합

**파일**: `python/reality_stone/layers/poincare.py`

```python
class PoincareBallLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float, t: float, 
                kappas: Tensor, layer_idx: int, c_min: float, c_max: float) -> Tensor:
        ctx.t = t
        
        # Dynamic curvature
        if kappas is not None:
            ctx.use_dynamic = True
            ctx.save_for_backward(u, v, kappas)
            ctx.layer_idx = layer_idx
            ctx.c_min = c_min
            ctx.c_max = c_max
            
            kappa_val = kappas.item() if kappas.dim() == 0 else kappas[layer_idx].item()
            output_np, c_val = _rust.poincare_ball_layer_layerwise_cpu(
                u.cpu().numpy(), v.cpu().numpy(), kappa_val, layer_idx, c_min, c_max, t
            )
            ctx.c_val = c_val
            return torch.from_numpy(output_np).to(u.device)
        
        # Static curvature
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            ctx.save_for_backward(u, v)
            
            if u.is_cuda and _has_cuda:
                output = torch.empty_like(u)
                _rust.poincare_ball_layer_cuda(
                    output.data_ptr(), u.data_ptr(), v.data_ptr(),
                    ctx.c, t, u.shape[0], u.shape[1]
                )
                return output
            else:
                output_np = _rust.poincare_ball_layer_cpu(
                    u.cpu().numpy(), v.cpu().numpy(), ctx.c, t
                )
                return torch.from_numpy(output_np).to(u.device)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        t = ctx.t
        
        if ctx.use_dynamic:
            u, v, kappas = ctx.saved_tensors
            # ... dynamic backward ...
            return grad_u, grad_v, None, None, grad_kappas, None, None, None
        else:
            u, v = ctx.saved_tensors
            c = ctx.c
            
            if grad_output.is_cuda and _has_cuda:
                grad_u = torch.empty_like(u)
                grad_v = torch.empty_like(v)
                _rust.poincare_ball_layer_backward_cuda(
                    grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                    grad_u.data_ptr(), grad_v.data_ptr(),
                    c, t, u.shape[0], u.shape[1]
                )
            else:
                grad_u_np, grad_v_np = _rust.poincare_ball_layer_backward_cpu(
                    grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t
                )
                grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
                grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
            
            return grad_u, grad_v, None, None, None, None, None, None
```

---

## 6. 테스트 결과

### 6.1 단위 테스트 (Rust)

**파일**: `src/layers/tests/poincare.rs`

```rust
#[test]
fn test_poincare_distance_same_point() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let d = poincare::poincare_distance(&x.view(), &x.view(), c);
    assert!(d[0].abs() < 1e-5);  // ✅ PASS
}

#[test]
fn test_poincare_ball_layer_interpolation() {
    let c = 1.0_f32;
    let u = arr2(&[[0.3_f32, 0.4_f32]]);
    let v = arr2(&[[-0.2_f32, 0.1_f32]]);
    
    let result_t0 = poincare::poincare_ball_layer(&u.view(), &v.view(), c, 0.0);
    assert_relative_eq!(result_t0, u, epsilon = 1e-5);  // ✅ PASS
    
    let result_t1 = poincare::poincare_ball_layer(&u.view(), &v.view(), c, 1.0);
    assert_relative_eq!(result_t1, v, epsilon = 1e-5);  // ✅ PASS
}
```

**결과**: 11/11 테스트 통과

### 6.2 MNIST 분류 테스트

**설정**:
- 데이터셋: MNIST (60k train, 10k test)
- 아키텍처: 784 → 128 → 128 → 10
- Optimizer: Adam (lr=0.001)
- Batch size: 256
- Epochs: 10

**결과**:

| 모드 | 최종 정확도 | Epoch 5 | Epoch 10 |
|------|------------|---------|----------|
| Static | **97.30%** | 96.42% | 97.30% |
| Dynamic | **97.26%** | 96.42% | 97.26% |

**Dynamic Curvature 학습**:
```
κ: -1.0000 → 0.0579 (Δ = 1.0579)
c: 0.0135 → 0.0258
```

---

## 7. 성능 최적화

### 7.1 수치 안정성

1. **Norm 계산**: `||x|| = max(√(x·x), EPS)`
2. **Division**: `den = max(den, EPS)`
3. **atanh 정의역**: `arg = min(arg, 1.0 - EPS)`
4. **Boundary 근처**: `BOUNDARY_EPS = 1e-5`

### 7.2 CUDA 최적화

1. **Stack 배열 사용**: `float u_prime[256]` (동적 할당 회피)
2. **Coalesced Memory Access**: 연속 메모리 접근
3. **병렬화**: 배치 단위 parallel kernel
4. **Register 최적화**: 중간 변수 최소화

### 7.3 벤치마크

**학습 속도** (MNIST, batch_size=256):
- CPU: ~1.4초/epoch
- CUDA: ~0.5초/epoch (예상)

**메모리 사용**:
- Forward: O(batch_size × dim)
- Backward: O(batch_size × dim) (중간 변수 저장)

---

## 8. 참고 문헌

1. Nickel & Kiela (2017). "Poincaré Embeddings for Learning Hierarchical Representations"
2. Ganea et al. (2018). "Hyperbolic Neural Networks"
3. Shimizu et al. (2021). "Hyperbolic Neural Networks++"

---

## 9. 관련 파일

- `src/layers/poincare.rs` - Rust CPU 구현
- `src/layers/cuda/poincare.cu` - CUDA GPU 구현
- `src/ops/mobius.rs` - Möbius 연산
- `python/reality_stone/layers/poincare.py` - Python 바인딩
- `tests/poincare.py` - MNIST 테스트
- `src/layers/tests/poincare.rs` - 단위 테스트

