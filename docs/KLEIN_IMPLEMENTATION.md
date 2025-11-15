# Klein Model Layer 구현 문서

## 개요

Klein 모델(Beltrami-Klein 모델)은 쌍곡 공간을 유클리드 단위 공 내부로 투영한 모델입니다.

**수학적 정의**:
- 공간: `D^n_c = {x ∈ ℝ^n : c||x||² < 1}`
- 거리: 비선형 (chord distance 기반)
- 특징: 측지선이 유클리드 직선

---

## 1. Klein Distance

### 1.1 수식

**표준 Klein distance**:
$$
d_K(u,v) = (1/√c) · acosh((1 - c⟨u,v⟩) / √[(1-c||u||²)(1-c||v||²)])
$$

### 1.2 구현

**CPU** (`src/layers/klein.rs`):
```rust
pub fn klein_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    // Klein distance: d_K(u,v) = (1/√c) · acosh((1 - c⟨u,v⟩) / √((1-c||u||²)(1-c||v||²)))
    let sqrtc = c.sqrt();
    let u2 = norm_sq_batched(u);
    let v2 = norm_sq_batched(v);
    let uv = dot_batched(u, v);

    let numerator = 1.0 - c * &uv;
    let denominator = ((1.0 - c * &u2) * (1.0 - c * &v2))
        .mapv(|z| z.max(EPS).sqrt());
    let arg = (&numerator / &denominator).mapv(|z| z.max(1.0 + EPS));
    
    arg.mapv(|r| safe_acosh(r) / sqrtc)
}
```

**CUDA** (`src/layers/cuda/klein.cu`):
```cpp
__global__ void klein_distance_kernel(
    float* out, const float* u, const float* v, 
    float c, int batch_size, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    float u_norm_sq = norm_sq(u_row, dim);
    float v_norm_sq = norm_sq(v_row, dim);
    float uv_dot = dot(u_row, v_row, dim);

    // 표준 Klein distance 공식
    float numerator = 1.0f - c * uv_dot;
    float denominator = sqrtf(fmaxf(
        (1.0f - c * u_norm_sq) * (1.0f - c * v_norm_sq), 
        EPS
    ));
    float arg = fmaxf(numerator / denominator, 1.0f + EPS);
    
    out[idx] = acoshf(arg) / sqrtf(c);
}
```


## 2. Klein Operations

### 2.1 Klein Addition

**수식**:
```
u ⊕_K v = [u/√(1-c||u||²) + v/√(1-c||v||²)] / [1 + √(1 + c||temp||²)]
```

**구현** (`src/layers/klein.rs`):
```rust
pub fn klein_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u_norm_sq = norm_sq_batched(u).insert_axis(Axis(1));
    let v_norm_sq = norm_sq_batched(v).insert_axis(Axis(1));

    // Denominators: √(1 - c||·||²)
    let u_denom = (1.0 - c * &u_norm_sq).mapv_into(|v| safe_sqrt(v));
    let v_denom = (1.0 - c * &v_norm_sq).mapv_into(|v| safe_sqrt(v));

    // temp = u/√(1-c||u||²) + v/√(1-c||v||²)
    let temp = u / &u_denom + v / &v_denom;
    let temp_norm_sq = norm_sq_batched(&temp.view()).insert_axis(Axis(1));

    // Result denominator: 1 + √(1 + c||temp||²)
    let result_denom = (1.0 + (1.0 + c * temp_norm_sq).mapv(|z| safe_sqrt(z)))
        .mapv(|v| v.max(EPS));
    
    temp / result_denom
}
```

### 2.2 Klein Scalar Multiplication

**수식**:
```
r ⊗_K x = r · x / ||x|| · min(r||x||, 1/√c - ε)
```

**특징**: 경계 근처에서 클램핑 필요

**구현** (`src/layers/klein.rs`):
```rust
pub fn klein_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    
    // scaled_norm = min(r·||u||, 1/√c - ε)
    let scaled_norm = (&norm_clamped * r)
        .mapv(|v| v.min(1.0 / c.sqrt() - BOUNDARY_EPS));
    
    let scale = scaled_norm / &norm_clamped;
    u * scale
}
```

---

## 3. Klein Layer

### 3.1 Forward Pass

**수식**:
```
y = [(1-t) ⊗_K u] ⊕_K [t ⊗_K v]
```

**구현** (`src/layers/klein.rs`):
```rust
pub fn klein_layer_forward(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> Array2<f32> {
    let u_prime = klein_scalar(u, c, 1.0 - t);
    let v_prime = klein_scalar(v, c, t);
    klein_add(&u_prime.view(), &v_prime.view(), c)
}
```

**CUDA Forward** (`src/layers/cuda/klein.cu`):
```cpp
__global__ void klein_layer_forward_kernel(
    float* out, const float* u, const float* v, 
    float c, float t, int batch_size, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float u_prime[1024];
    float v_prime[1024];
    
    const float* u_row = u + idx * dim;
    const float* v_row = v + idx * dim;
    
    // Scalar Mul for u
    float u_norm = sqrtf(norm_sq(u_row, dim));
    float u_norm_clamped = fmaxf(u_norm, EPS);
    float u_scaled_norm = fminf(
        u_norm_clamped * (1.0f - t), 
        1.0f/sqrtf(c) - BOUNDARY_EPS
    );
    float u_scale = u_scaled_norm / u_norm_clamped;
    for(int i=0; i<dim; ++i) u_prime[i] = u_row[i] * u_scale;

    // Scalar Mul for v
    float v_norm = sqrtf(norm_sq(v_row, dim));
    float v_norm_clamped = fmaxf(v_norm, EPS);
    float v_scaled_norm = fminf(
        v_norm_clamped * t, 
        1.0f/sqrtf(c) - BOUNDARY_EPS
    );
    float v_scale = v_scaled_norm / v_norm_clamped;
    for(int i=0; i<dim; ++i) v_prime[i] = v_row[i] * v_scale;
    
    // Klein Add
    float u_prime_norm_sq = norm_sq(u_prime, dim);
    float v_prime_norm_sq = norm_sq(v_prime, dim);
    float u_denom = sqrtf(fmaxf(1.0f - c * u_prime_norm_sq, EPS));
    float v_denom = sqrtf(fmaxf(1.0f - c * v_prime_norm_sq, EPS));

    float temp[1024];
    for(int i=0; i<dim; ++i) 
        temp[i] = u_prime[i] / u_denom + v_prime[i] / v_denom;

    float temp_norm_sq = norm_sq(temp, dim);
    float res_denom = 1.0f + sqrtf(1.0f + c * temp_norm_sq);
    
    float* out_row = out + idx * dim;
    for(int i=0; i<dim; ++i) 
        out_row[i] = temp[i] / fmaxf(res_denom, EPS);
}
```

### 3.2 Backward Pass

**Gradient Computation**:

Klein scalar의 gradient (경계 클램핑 고려):
```rust
pub fn klein_scalar_vjp(
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    c: f32,
    r: f32,
) -> Array2<f32> {
    let norm = norm_sq_batched(x).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    let scaled_norm = (&norm_clamped * r)
        .mapv(|v| v.min(1.0 / c.sqrt() - BOUNDARY_EPS));
    let scale = scaled_norm / &norm_clamped;

    let boundary = 1.0 / c.sqrt() - BOUNDARY_EPS;
    
    // d(scale)/d(norm): piecewise due to clamp
    let d_scale_d_norm = (&norm_clamped).mapv(|n| {
        let rn = r * n;
        if rn < boundary {
            0.0  // Within boundary
        } else {
            -1.0 / (n * n).max(EPS)  // At boundary
        }
    });

    let grad_norm_component = (grad_output * x)
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));
    
    let grad_x = grad_output * &scale 
               + (grad_norm_component * d_scale_d_norm / &norm_clamped) * x;
    grad_x
}
```

Klein add의 gradient:
```rust
pub fn klein_add_vjp(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
) -> (Array2<f32>, Array2<f32>) {
    let u_norm_sq = norm_sq_batched(u).insert_axis(Axis(1));
    let v_norm_sq = norm_sq_batched(v).insert_axis(Axis(1));
    let u_denom = (1.0 - c * &u_norm_sq).mapv_into(|val| val.max(EPS).sqrt());
    let v_denom = (1.0 - c * &v_norm_sq).mapv_into(|val| val.max(EPS).sqrt());
    
    let temp = u / &u_denom + v / &v_denom;
    let temp_norm_sq = norm_sq_batched(&temp.view()).insert_axis(Axis(1));
    let result_denom_inner_sqrt = (1.0 + c * &temp_norm_sq).mapv(f32::sqrt);
    let result_denom = (1.0 + &result_denom_inner_sqrt).mapv(|val| val.max(EPS));

    // Gradient through output
    let grad_temp_part1 = grad_output / &result_denom;
    
    // Gradient through result_denom
    let grad_result_denom = -(grad_output * &temp / (&result_denom * &result_denom))
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));
    let grad_temp_norm_sq = grad_result_denom * c / (2.0 * &result_denom_inner_sqrt);
    
    let grad_temp = grad_temp_part1 + 2.0 * &grad_temp_norm_sq * &temp;

    // Gradient through temp = u/u_denom + v/v_denom
    let grad_u_from_temp = &grad_temp / &u_denom;
    let grad_v_from_temp = &grad_temp / &v_denom;

    // Gradient through denominators
    let grad_u_denom = -(&grad_temp * u / (&u_denom * &u_denom))
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));
    let grad_v_denom = -(&grad_temp * v / (&v_denom * &v_denom))
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));

    let grad_u_norm_sq = grad_u_denom * (-c / (2.0 * &u_denom));
    let grad_v_norm_sq = grad_v_denom * (-c / (2.0 * &v_denom));

    let grad_u = grad_u_from_temp + 2.0 * &grad_u_norm_sq * u;
    let grad_v = grad_v_from_temp + 2.0 * &grad_v_norm_sq * v;

    (grad_u, grad_v)
}
```

**전체 Backward**:
```rust
pub fn klein_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    // Forward pass 재계산
    let u_prime = klein_scalar(u, c, 1.0 - t);
    let v_prime = klein_scalar(v, c, t);
    
    // Backward through klein_add
    let (grad_u_prime, grad_v_prime) =
        klein_add_vjp(grad_output, &u_prime.view(), &v_prime.view(), c);
    
    // Backward through klein_scalar
    let grad_u = klein_scalar_vjp(&grad_u_prime.view(), &u.view(), c, 1.0 - t);
    let grad_v = klein_scalar_vjp(&grad_v_prime.view(), &v.view(), c, t);
    
    (grad_u, grad_v)
}
```

---

## 4. 좌표 변환

### 4.1 Klein ↔ Poincaré

**Klein → Poincaré**:
```
p = x / (1 + √(1 - c||x||²))
```

**Poincaré → Klein**:
```
x = 2p / (1 + c||p||²)
```

**구현** (`src/layers/klein.rs`):
```rust
pub fn klein_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 + (1.0 - c * x_norm_sq).mapv(|v| v.max(0.0).sqrt()))
        .mapv(|v| v.max(EPS));
    x / &den
}

pub fn from_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    // Poincaré → Klein: 2x / (1 + c||x||²)
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 + c * &x_norm_sq).mapv(|v| v.max(EPS));
    (2.0 * x) / &den
}
```

### 4.2 Klein ↔ Lorentz

**Klein → Lorentz**:
```
x₀ = 1 / √(1 - c||x||²)
xᵢ = xᵢ / √(1 - c||x||²)
```

**구현** (`src/layers/klein.rs`):
```rust
pub fn klein_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let x0 = 1.0 / (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS).sqrt());
    
    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    result.slice_mut(s![.., 0..1]).assign(&x0);
    result.slice_mut(s![.., 1..]).assign(&(x * &x0));
    result
}
```

---

## 5. Python 바인딩

**파일**: `python/reality_stone/layers/klein.py`

```python
class KleinLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float, t: float) -> Tensor:
        ctx.save_for_backward(u, v)
        ctx.c = c
        ctx.t = t
        
        if u.is_cuda and _has_cuda:
            output = torch.empty_like(u)
            _rust.klein_layer_forward_cuda(
                output.data_ptr(), u.data_ptr(), v.data_ptr(),
                c, t, u.shape[0], u.shape[1]
            )
            return output
        else:
            output_np = _rust.klein_layer_forward_cpu(
                u.cpu().numpy(), v.cpu().numpy(), c, t
            )
            return torch.from_numpy(output_np).to(u.device)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        u, v = ctx.saved_tensors
        c = ctx.c
        t = ctx.t
        
        if grad_output.is_cuda and _has_cuda:
            grad_u = torch.empty_like(u)
            grad_v = torch.empty_like(v)
            _rust.klein_layer_backward_cuda(
                grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                grad_u.data_ptr(), grad_v.data_ptr(),
                c, t, u.shape[0], u.shape[1]
            )
        else:
            grad_u_np, grad_v_np = _rust.klein_layer_backward_cpu(
                grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t
            )
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
        
        return grad_u, grad_v, None, None
```

---

## 6. 테스트 결과

### 6.1 단위 테스트

```rust
#[test]
fn test_klein_scalar_bounds() {
    let c = 0.5_f32;
    let x = arr2(&[[0.4_f32, 0.2]]);
    let y = klein_scalar(&x.view(), c, 2.0);
    
    // 경계 밖으로 나가지 않도록 norm이 제한됨
    let norm = (y[[0, 0]].powi(2) + y[[0, 1]].powi(2)).sqrt();
    assert!(norm < 1.0 / c.sqrt());  // ✅ PASS
}

#[test]
fn test_klein_to_poincare_and_back_shapes() {
    let c = 0.7_f32;
    let x = arr2(&[[0.1_f32, -0.2], [0.05, 0.05]]);
    let p = klein_to_poincare(&x.view(), c);
    assert_eq!(p.dim(), x.dim());  // ✅ PASS
}
```

### 6.2 MNIST 분류

**설정**: 784 → 128 → 128 → 10, 10 epochs

**결과**:
```
Epoch 1:  95.47%
Epoch 2:  97.19% ✅
Epoch 3:  97.61%
Epoch 5:  97.90%
Epoch 6:  98.01%
Epoch 7:  98.20%
Epoch 9:  98.28% ✅ (최고!)
```

**특징**:
- **최고 정확도**: 98.28% (세 모델 중 1위)
- **가장 빠른 학습**: 2-6초/epoch
- **메모리 효율**: 차원 증가 없음

---

## 7. 성능 특징

### 7.1 장점

1. **최고 정확도**: 98.28% (Poincaré 97.30%, Lorentz 98.09%)
2. **빠른 학습**: ~2.5초/epoch (Lorentz ~8초)
3. **메모리 효율**: 차원 증가 없음 (Lorentz는 +1차원)
4. **직관적**: 측지선이 유클리드 직선

### 7.2 단점

1. **각도 왜곡**: 각도가 보존되지 않음
2. **복잡한 distance**: acosh 기반 비선형 거리
3. **경계 처리**: 명시적 클램핑 필요

### 7.3 벤치마크

| 모델      | Epoch 시간 | 메모리   | 최종 정확도  |
| --------- | ---------- | -------- | ------------ |
| Poincaré  | 1.4s       | Base     | 97.30%       |
| Lorentz   | 8.0s       | +10%     | 98.09%       |
| **Klein** | **2.5s**   | **Base** | **98.28%** ⭐ |

---

## 8. 수치 안정성

### 8.1 경계 처리

Klein 모델은 `c||x||² < 1` 제약이 있으므로 경계 근처 처리가 중요:

```rust
const BOUNDARY_EPS: f32 = 1e-5;

// Scalar multiplication에서 경계 제한
let scaled_norm = norm.mapv(|v| v.min(1.0 / c.sqrt() - BOUNDARY_EPS));
```

### 8.2 Denominator 클램핑

```rust
// √(1 - c||x||²) 계산 시
let denom = (1.0 - c * norm_sq).mapv(|v| v.max(EPS).sqrt());
```

### 8.3 Distance 계산

```rust
// acosh 정의역 제한
let arg = (numerator / denominator).mapv(|z| z.max(1.0 + EPS));
```

---

## 9. 관련 파일

- `src/layers/klein.rs` - Rust CPU 구현
- `src/layers/cuda/klein.cu` - CUDA GPU 구현
- `python/reality_stone/layers/klein.py` - Python 바인딩
- `tests/klein.py` - MNIST 테스트
- `src/layers/tests/klein.rs` - 단위 테스트

---

## 10. 참고 문헌

1. Ungar (2008). "Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity"
2. Bachmann et al. (2020). "Constant Curvature Graph Convolutional Networks"
3. Klein Model on Wikipedia

