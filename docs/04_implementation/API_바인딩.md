# Reality Stone Python API Bindings

이 문서는 Rust에서 Python으로 바인딩된 모든 함수와 클래스를 정리합니다.

## 바인딩 구조

Reality Stone은 `_rust` 확장 모듈을 통해 Rust 구현을 Python에서 사용할 수 있습니다.

```python
from reality_stone import _rust
```

## 1. Mobius Operations (`_rust`)

### CPU 함수

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `mobius_add_cpu` | Mobius 덧셈 (CPU) | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `mobius_scalar_cpu` | Mobius 스칼라 곱 (CPU) | `u: Array2<f32>`, `r: f32`, `c: f32` | `Array2<f32>` |
| `mobius_add_dynamic_cpu` | 동적 곡률 Mobius 덧셈 | `u: Array2<f32>`, `v: Array2<f32>`, `kappa: f32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, f32)` |
| `mobius_add_dynamic_backward_cpu` | 동적 곡률 Mobius 덧셈 backward | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `kappa: f32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, Array2<f32>, f32)` |
| `mobius_add_layerwise_cpu` | Layer-wise 동적 곡률 Mobius 덧셈 | `u: Array2<f32>`, `v: Array2<f32>`, `kappas: Array1<f32>`, `layer_idx: i32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, f32)` |
| `mobius_add_layerwise_backward_cpu` | Layer-wise 동적 곡률 Mobius 덧셈 backward | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `kappas: Array1<f32>`, `layer_idx: i32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, Array2<f32>, Array1<f32>)` |

### CUDA 함수 (feature = "cuda")

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `mobius_add_cuda` | Mobius 덧셈 (CUDA) | `u_ptr: usize`, `v_ptr: usize`, `out_ptr: usize`, `batch_size: i64`, `dim: i64`, `c: f32` | `None` |
| `mobius_scalar_cuda` | Mobius 스칼라 곱 (CUDA) | `u_ptr: usize`, `out_ptr: usize`, `batch_size: i64`, `dim: i64`, `r: f32`, `c: f32` | `None` |

## 2. Poincaré Ball Operations (`_rust.poincare`)

### CPU 함수

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `poincare_distance_cpu` | Poincaré 거리 계산 | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array1<f32>` |
| `poincare_to_lorentz_cpu` | Poincaré → Lorentz 변환 | `x: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `poincare_to_klein_cpu` | Poincaré → Klein 변환 | `x: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `poincare_ball_layer_cpu` | Poincaré ball layer forward | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `Array2<f32>` |
| `poincare_exp_at_cpu` | Exponential map at point | `x: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `poincare_log_at_cpu` | Logarithmic map at point | `x: Array2<f32>`, `y: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `poincare_ball_layer_backward_cpu` | Poincaré ball layer backward | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `(Array2<f32>, Array2<f32>)` |
| `mobius_add_vjp_cpu` | Mobius 덧셈 VJP | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `(Array2<f32>, Array2<f32>)` |
| `mobius_scalar_vjp_cpu` | Mobius 스칼라 곱 VJP | `grad_output: Array2<f32>`, `u: Array2<f32>`, `r: f32`, `c: f32` | `(Array2<f32>, f32)` |
| `project_to_ball_cpu` | Ball로 projection | `x: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `poincare_ball_layer_dynamic_cpu` | 동적 곡률 Poincaré layer | `u: Array2<f32>`, `v: Array2<f32>`, `t: f32`, `kappa: f32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, f32)` |
| `poincare_ball_layer_dynamic_backward_cpu` | 동적 곡률 Poincaré layer backward | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `t: f32`, `kappa: f32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, Array2<f32>, f32)` |
| `poincare_ball_layer_layerwise_cpu` | Layer-wise 동적 곡률 Poincaré layer | `u: Array2<f32>`, `v: Array2<f32>`, `t: f32`, `kappas: Array1<f32>`, `layer_idx: i32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, f32)` |
| `poincare_ball_layer_layerwise_backward_cpu` | Layer-wise 동적 곡률 Poincaré layer backward | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `t: f32`, `kappas: Array1<f32>`, `layer_idx: i32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, Array2<f32>, Array1<f32>)` |

### CUDA 함수 (feature = "cuda")

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `poincare_distance_cuda` | Poincaré 거리 계산 (CUDA) | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array1<f32>` |
| `poincare_ball_layer_cuda` | Poincaré ball layer forward (CUDA) | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `Array2<f32>` |
| `poincare_ball_layer_backward_cuda` | Poincaré ball layer backward (CUDA) | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `(Array2<f32>, Array2<f32>)` |

**참고**: CUDA 함수는 루트 모듈(`_rust`)에도 재노출됩니다.

## 3. Lorentz Model Operations (`_rust`)

### CPU 함수

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `lorentz_add` | Lorentz 덧셈 | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `lorentz_scalar` | Lorentz 스칼라 곱 | `u: Array2<f32>`, `r: f32`, `c: f32` | `Array2<f32>` |
| `lorentz_distance` | Lorentz 거리 계산 | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array1<f32>` |
| `lorentz_inner` | Lorentz 내적 | `u: Array2<f32>`, `v: Array2<f32>` | `Array1<f32>` |
| `lorentz_to_poincare` | Lorentz → Poincaré 변환 | `x: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `lorentz_to_klein` | Lorentz → Klein 변환 | `x: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `lorentz_layer_forward` | Lorentz layer forward | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `Array2<f32>` |
| `lorentz_ball_layer_backward_cpu` | Lorentz layer backward | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `(Array2<f32>, Array2<f32>)` |
| `lorentz_layer_dynamic_cpu` | 동적 곡률 Lorentz layer | `u: Array2<f32>`, `v: Array2<f32>`, `t: f32`, `kappa: f32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, f32)` |
| `lorentz_layer_dynamic_backward_cpu` | 동적 곡률 Lorentz layer backward | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `t: f32`, `kappa: f32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, Array2<f32>, f32)` |
| `lorentz_layer_layerwise_cpu` | Layer-wise 동적 곡률 Lorentz layer | `u: Array2<f32>`, `v: Array2<f32>`, `t: f32`, `kappas: Array1<f32>`, `layer_idx: i32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, f32)` |
| `lorentz_layer_layerwise_backward_cpu` | Layer-wise 동적 곡률 Lorentz layer backward | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `t: f32`, `kappas: Array1<f32>`, `layer_idx: i32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, Array2<f32>, Array1<f32>)` |
| `from_poincare_dynamic_cpu` | Poincaré → Lorentz (동적 곡률) | `x: Array2<f32>`, `kappa: f32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, f32)` |
| `from_poincare_dynamic_backward_cpu` | Poincaré → Lorentz backward (동적 곡률) | `grad_output: Array2<f32>`, `x: Array2<f32>`, `kappa: f32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, f32)` |

### CUDA 함수 (feature = "cuda")

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `lorentz_distance_cuda` | Lorentz 거리 계산 (CUDA) | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array1<f32>` |
| `lorentz_layer_forward_cuda` | Lorentz layer forward (CUDA) | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `Array2<f32>` |
| `lorentz_ball_layer_backward_cuda` | Lorentz layer backward (CUDA) | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `(Array2<f32>, Array2<f32>)` |

## 4. Klein Model Operations (`_rust`)

### CPU 함수

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `klein_add` | Klein 덧셈 | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `klein_scalar` | Klein 스칼라 곱 | `u: Array2<f32>`, `r: f32`, `c: f32` | `Array2<f32>` |
| `klein_distance` | Klein 거리 계산 | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array1<f32>` |
| `klein_to_poincare` | Klein → Poincaré 변환 | `x: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `klein_to_lorentz` | Klein → Lorentz 변환 | `x: Array2<f32>`, `c: f32` | `Array2<f32>` |
| `klein_layer_forward` | Klein layer forward | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `Array2<f32>` |
| `klein_ball_layer_backward_cpu` | Klein layer backward | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `(Array2<f32>, Array2<f32>)` |
| `from_poincare_dynamic_cpu` | Poincaré → Klein (동적 곡률) | `x: Array2<f32>`, `kappa: f32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, f32)` |
| `from_poincare_dynamic_backward_cpu` | Poincaré → Klein backward (동적 곡률) | `grad_output: Array2<f32>`, `x: Array2<f32>`, `kappa: f32`, `c_min: f32`, `c_max: f32` | `(Array2<f32>, f32)` |

### CUDA 함수 (feature = "cuda")

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `klein_distance_cuda` | Klein 거리 계산 (CUDA) | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32` | `Array1<f32>` |
| `klein_layer_forward_cuda` | Klein layer forward (CUDA) | `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `Array2<f32>` |
| `klein_ball_layer_backward_cuda` | Klein layer backward (CUDA) | `grad_output: Array2<f32>`, `u: Array2<f32>`, `v: Array2<f32>`, `c: f32`, `t: f32` | `(Array2<f32>, Array2<f32>)` |

## 5. Riemann Low-Rank Operations (`_rust`)

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `riemann_lowrank_forward_cpu` | Riemann low-rank forward | `x: Array2<f32>`, `p: Array2<f32>`, `sigma: Array2<f32>`, `q: Array2<f32>`, `b_tan: Array1<f32>`, `c: f32`, `epsilon: f32` | `Array2<f32>` |

## 6. Spline Layer (`_rust.spline`)

### 클래스

| 클래스명 | 설명 | 메서드 |
|----------|------|--------|
| `SplineLayer` | Catmull-Rom spline interpolation layer | `__new__(num_points, dim)`, `from_weight(weight)`, `interpolate(x)`, `forward(x)` |

### CUDA 함수 (feature = "cuda")

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `spline_interpolate_cuda` | Spline interpolation (CUDA) | `x: Array2<f32>`, `control_points: Array2<f32>` | `Array2<f32>` |
| `spline_forward_cuda` | Spline forward (CUDA) | `x: Array2<f32>`, `control_points: Array2<f32>` | `Array2<f32>` |
| `spline_backward_cuda` | Spline backward (CUDA) | `grad_output: Array2<f32>`, `x: Array2<f32>`, `control_points: Array2<f32>` | `(Array2<f32>, Array2<f32>)` |

## 7. Suppression Field (`_rust`)

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `compute_suppression_field` | 동적 억제 필드 계산 | `x: Array2<f32>`, `base: f32`, `linear: f32`, `hyp: f32`, `scale: f32` | `Array2<f32>` |

## 8. Geodesic Attention (`_rust.geodesic`)

### CUDA 함수 (feature = "cuda")

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `geodesic_topk_attention` | Fused geodesic top-k attention (CUDA) | `q: Array4<f32>`, `k: Array4<f32>`, `v: Array4<f32>`, `idx: Array3<i64>`, `l_factor: Array2<f32>`, `c: f32=1.0`, `tau: f32=1.0` | `Array4<f32>` |
| `batched_cholesky_cuda` | Batched Cholesky decomposition (CUDA) | `g: Array4<f32>` | `Array4<f32>` |

## 9. MetriKey Module (`_rust.metrikey`)

### 함수

| 함수명 | 설명 | 인자 | 반환 |
|--------|------|------|------|
| `spd_metric_from_key` | SPD metric from key | `key: Array2<f32>` | `Array3<f32>` |
| `metric_factor_cholesky` | Metric factor Cholesky | `g: Array3<f32>` | `Array3<f32>` |
| `mahalanobis_distance_sq_g` | Mahalanobis distance (G) | `x: Array2<f32>`, `y: Array2<f32>`, `g: Array3<f32>` | `Array1<f32>` |
| `mahalanobis_distance_sq_l` | Mahalanobis distance (L) | `x: Array2<f32>`, `y: Array2<f32>`, `l: Array3<f32>` | `Array1<f32>` |
| `block_orthogonal_from_key` | Block orthogonal from key | `key: Array2<f32>`, `block_size: i32` | `Array3<f32>` |
| `spd_block_metric_from_key` | SPD block metric from key | `key: Array2<f32>`, `block_size: i32` | `Array3<f32>` |
| `spd_metric_from_key_weighted` | SPD metric from key (weighted) | `key: Array2<f32>`, `weight: Array1<f32>` | `Array3<f32>` |
| `compose_layers_order_preserving` | Compose layers (order preserving) | `g_list: Vec<Array3<f32>>` | `Array3<f32>` |
| `compose_layers_gravity` | Compose layers (gravity) | `g_list: Vec<Array3<f32>>` | `Array3<f32>` |
| `compose_layers_gravity_f64` | Compose layers (gravity, f64) | `g_list: Vec<Array3<f64>>` | `Array3<f64>` |
| `apply_linear` | Apply linear transformation | `x: Array2<f32>`, `w: Array2<f32>`, `b: Array1<f32>` | `Array2<f32>` |
| `apply_linear_f64` | Apply linear transformation (f64) | `x: Array2<f64>`, `w: Array2<f64>`, `b: Array1<f64>` | `Array2<f64>` |
| `layer_norm_forward_exact_f32_py` | Layer norm forward (exact, f32) | `x: Array2<f32>`, `gamma: Array1<f32>`, `beta: Array1<f32>`, `eps: f32` | `(Array2<f32>, Array1<f32>, Array1<f32>)` |
| `gelu_new_f32_py` | GELU activation (f32) | `x: Array2<f32>` | `Array2<f32>` |
| `softmax_lastdim_f32_py` | Softmax last dim (f32) | `x: Array2<f32>` | `Array2<f32>` |

### 클래스

| 클래스명 | 설명 | 메서드 |
|----------|------|--------|
| `CollapsedTransformF64` | Collapsed transform (f64) | `__new__(d_in, d_out, key_dim, num_layers)`, `forward(x)` |
| `CollapsedTransformF32` | Collapsed transform (f32) | `__new__(d_in, d_out, key_dim, num_layers)`, `forward(x)` |
| `CollapsedRunnerF64` | Collapsed runner (f64) | `__new__(d_in, d_out, key_dim, num_layers)`, `forward(x)` |
| `CollapsedRunnerF32` | Collapsed runner (f32) | `__new__(d_in, d_out, key_dim, num_layers)`, `forward(x)` |

## Python 레이어 래퍼 (`reality_stone`)

Reality Stone은 Rust 바인딩 위에 PyTorch 호환 Python 레이어를 제공합니다.

### Core Operations

```python
from reality_stone import (
    MobiusAdd,
    MobiusScalarMul,
)
```

### Poincaré Ball

```python
from reality_stone import (
    PoincareBallLayer,
    poincare_add,
    poincare_scalar_mul,
    poincare_distance,
    poincare_to_lorentz,
    poincare_to_klein,
    poincare_ball_layer,  # 함수형 API
)
```

### Lorentz Model

```python
from reality_stone import (
    LorentzLayer,
    lorentz_add,
    lorentz_scalar_mul,
    lorentz_distance,
    lorentz_inner,
    lorentz_to_poincare,
    lorentz_to_klein,
    lorentz_layer,  # 함수형 API
)
```

### Klein Model

```python
from reality_stone import (
    KleinLayer,
    klein_add,
    klein_scalar_mul,
    klein_distance,
    klein_to_poincare,
    klein_to_lorentz,
    klein_layer,  # 함수형 API
)
```

### Spline Layer

```python
from reality_stone import SplineLinear
```

### Utilities

```python
from reality_stone import convert_to_hyperbolic
```

### MetriKey (Rust bindings)

```python
from reality_stone import metrikey
```

## 바인딩 확인

### 1. Rust 확장 로드 확인

```python
import reality_stone as rs

print("Rust extension loaded:", rs._has_rust_ext)
print("CUDA support:", rs._has_cuda)
```

### 2. 사용 가능한 함수 확인

```python
from reality_stone import _rust

# 루트 레벨 함수
print("Root-level functions:")
for name in dir(_rust):
    if not name.startswith('_'):
        print(f"  - {name}")

# Poincaré 서브모듈
print("\nPoincaré functions:")
for name in dir(_rust.poincare):
    if not name.startswith('_'):
        print(f"  - {name}")

# MetriKey 서브모듈
print("\nMetriKey functions:")
for name in dir(_rust.metrikey):
    if not name.startswith('_'):
        print(f"  - {name}")
```

### 3. CUDA 심볼 확인

```python
import reality_stone as rs

if rs._has_cuda:
    required_cuda_symbols = [
        'mobius_add_cuda',
        'mobius_scalar_cuda',
        'poincare_ball_layer_cuda',
        'poincare_ball_layer_backward_cuda',
        'poincare_distance_cuda',
        'lorentz_layer_forward_cuda',
        'lorentz_ball_layer_backward_cuda',
        'lorentz_distance_cuda',
        'klein_layer_forward_cuda',
        'klein_ball_layer_backward_cuda',
        'klein_distance_cuda',
    ]
    
    print("CUDA symbols available:")
    for name in required_cuda_symbols:
        has_symbol = hasattr(rs._rust, name)
        print(f"  - {name}: {has_symbol}")
```

## 바인딩 추가 가이드

새로운 Rust 함수를 Python에 바인딩하려면:

1. **Rust 함수 작성** (`src/layers/` 또는 `src/ops/`)
2. **바인딩 함수 작성** (`src/bindings/` 해당 모듈)
   ```rust
   #[pyfunction]
   pub fn my_function<'py>(
       py: Python<'py>,
       input: PyReadonlyArray2<f32>,
   ) -> &'py PyArray2<f32> {
       let input_arr = input.as_array();
       let result = crate::ops::my_function(&input_arr);
       result.into_pyarray(py)
   }
   ```
3. **모듈에 등록** (`src/bindings/` 해당 모듈의 `register` 함수)
   ```rust
   pub fn register(m: &PyModule) -> PyResult<()> {
       m.add_function(wrap_pyfunction!(my_function, m)?)?;
       Ok(())
   }
   ```
4. **빌드 및 테스트**
   ```bash
   uv run maturin develop --features cuda
   uv run python -c "from reality_stone import _rust; print(hasattr(_rust, 'my_function'))"
   ```

## 참고

- 모든 배열 타입은 NumPy 호환입니다
- CUDA 함수는 `--features cuda`로 빌드 시에만 사용 가능합니다
- 동적 곡률 함수는 학습 가능한 곡률 파라미터를 지원합니다
- Layer-wise 함수는 레이어별로 다른 곡률을 사용할 수 있습니다

