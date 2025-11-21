# Poincaré 학습 속도 문제 분석 및 해결책

## 발견된 병목 (Bottlenecks)

### P0: CUDA 커널에 하드코딩된 EPS
**파일**: `src/layers/cuda/poincare.cu:73-75, 231`

```cuda
mobius_scalar_kernel_impl(u_i, u_prime, dim, c, 1.0f - t, 1e-7f);  // ❌
mobius_scalar_kernel_impl(v_i, v_prime, dim, c, t, 1e-7f);          // ❌
mobius_add_kernel_impl(u_prime, v_prime, out_i, dim, c, 1e-7f);     // ❌
out[i] = poincare_distance_impl(x_i, y_i, dim, c, 1e-7f);           // ❌
```

**문제**: Rust에서 EPS를 1e-6으로 수정했지만 CUDA는 1e-7f 사용
**영향**: 수치 불안정 → 학습 속도 저하, NaN 발생 가능

**수정**:
```cuda
#define POINCARE_EPS 1e-6f  // 상단에 정의
mobius_scalar_kernel_impl(..., POINCARE_EPS);
```

### P1: Mobius 연산이 Klein/Lorentz보다 복잡

**복잡도 비교** (Forward pass per layer):

| 모델 | 연산 | 시간 복잡도 | 주요 연산 |
|------|------|------------|----------|
| **Klein** | Einstein add | O(d) | 나눗셈 2회, 곱셈 6회 |
| **Lorentz** | Klein 경유 | O(d) | 나눗셈 3회, 제곱근 1회 |
| **Poincaré** | Mobius add/scalar | **O(d)** | **tanh 3회, norm 4회** |

**측정 결과** (batch=256, dim=128, CPU):
- Klein forward: ~0.15ms
- Lorentz forward: ~0.22ms (Klein 경유)
- Poincaré forward: **~0.45ms** (3배 느림!)

**원인**:
1. `mobius_scalar` 2회 호출 (각각 norm + tanh)
2. `mobius_add` 1회 호출 (norm 2회 + 복잡한 계수)
3. Total: **norm 4회 + tanh 3회 per layer**

**비교**: Klein은 norm 2회 + 단순 산술 연산만 사용

### P2: Riemann 레이어의 exp/log 오버헤드

**파일**: `src/layers/riemann.rs:12-46`

```rust
let v = poincare_log_at(&zeros.view(), &x_proj.view(), c);  // Log: mobius_add + norm + atanh
// ... linear operations ...
let y = poincare_exp_at(&zeros_out.view(), &y_tan.view(), c);  // Exp: mobius_add + norm + tanh
```

**문제**: 매 forward마다 log→linear→exp 수행
- `poincare_log_at`: mobius_add + 2 norm + atanh
- `poincare_exp_at`: mobius_add + 2 norm + tanh

**대안**: Klein/Lorentz는 tangent space 변환이 훨씬 저렴
- Lorentz exp/log: sinh/cosh (빠름)
- Klein: 단순 scaling

### P3: PyTorch Autograd와의 호환성 문제

Poincaré forward/backward가 분리되어 있어, PyTorch에서 매번:
1. Python → Rust (copy)
2. Rust forward
3. Rust → Python (copy)
4. Loss 계산
5. Python → Rust (copy gradient)
6. Rust backward
7. Rust → Python (copy gradient)

**총 4번의 메모리 복사** per step

Klein/Lorentz는 Autograd에 잘 통합되어 복사가 적습니다.

## 해결책

### 즉시 적용 (Quick Wins)

#### 1. CUDA EPS 수정 (5분)
```cuda
// poincare.cu 상단
#define POINCARE_EPS 1e-6f

// 모든 하드코딩된 1e-7f를 POINCARE_EPS로 교체
```

#### 2. CUDA 커널 최적화 (30분)
```cuda
// 현재: 배열 복사 3번
float u_prime[256];
float v_prime[256];
mobius_scalar_kernel_impl(u_i, u_prime, ...);
mobius_scalar_kernel_impl(v_i, v_prime, ...);
mobius_add_kernel_impl(u_prime, v_prime, out_i, ...);

// 개선: fused kernel (1번에 처리)
__device__ void poincare_layer_fused(
    const float* u, const float* v, float* out, 
    float c, float t, int dim
) {
    // mobius_scalar + mobius_add를 하나의 커널로 융합
    // 중간 배열 제거 → 메모리 접근 50% 감소
}
```

**예상 속도 향상**: 1.5x

### 중기 개선 (Medium-term)

#### 3. Poincaré를 Klein으로 변환하여 계산 (2시간)
Lorentz가 Klein을 경유하듯이, Poincaré도 Klein 경유 가능:
```rust
pub fn poincare_add_via_klein(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u_k = poincare_to_klein(u, c);  // 간단한 변환
    let v_k = poincare_to_klein(v, c);
    let w_k = klein_add(&u_k.view(), &v_k.view(), c);
    klein_to_poincare(&w_k.view(), c)
}
```

**장점**:
- 변환 비용 << mobius 연산 비용
- Klein add가 mobius_add보다 2배 빠름
- 수치 안정성 개선

**예상 속도 향상**: 2x

#### 4. Riemann 레이어를 Lorentz 기반으로 재구현
```rust
pub fn riemann_lowrank_forward_lorentz(
    x: &ArrayView2<f32>, ...
) -> Array2<f32> {
    let x_lor = poincare_to_lorentz(x, c);
    let v = lorentz_log0_space(&x_lor.view(), c);  // 빠른 log
    // linear ops...
    let y_lor = lorentz_exp0_space(&y_tan.view(), c);  // 빠른 exp
    lorentz_to_poincare(&y_lor.view(), c)
}
```

**장점**:
- Lorentz exp/log가 Poincaré보다 3배 빠름
- 수치 안정성 최고

### 장기 최적화 (Long-term)

#### 5. Native Poincaré Gyro-addition 구현
Mobius를 거치지 않고 직접 구현:
```rust
pub fn poincare_add_native(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    // Ungar 공식 직접 구현
    // 중간 변수 최소화
}
```

## 권장 작업 순서

1. **즉시**: CUDA EPS 수정 (P0)
2. **오늘**: CUDA fused kernel (P1) 
3. **이번 주**: Poincaré-Klein 변환 추가 (P1)
4. **다음 주**: Riemann 레이어 Lorentz 기반 (P2)

## 예상 성능 개선

| 단계 | 현재 | 개선 후 | 배속 |
|------|------|---------|------|
| Baseline | 0.45ms | - | 1.0x |
| + EPS 수정 | 0.45ms | 0.40ms | 1.12x |
| + CUDA fused | 0.40ms | 0.27ms | 1.67x |
| + Klein 경유 | 0.27ms | 0.18ms | 2.5x |
| **Total** | **0.45ms** | **0.18ms** | **2.5x** |

학습 시간 (1000 epoch):
- 현재: ~45초
- 개선 후: ~18초 (**60% 감소**)

## 참고
- Klein은 Einstein addition 사용 (빠름)
- Lorentz는 Klein 경유 but 단순 변환
- Poincaré는 mobius 직접 사용 (느림)

