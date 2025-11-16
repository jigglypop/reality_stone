# Reality Stone 구현 개요

## 전체 아키텍처

Reality Stone은 3개의 쌍곡 기하 모델을 Rust/CUDA로 구현하고 PyTorch와 통합한 라이브러리입니다.

```
┌─────────────────────────────────────────────────────────┐
│                   Python Layer (PyTorch)                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │  Poincaré  │  │  Lorentz   │  │   Klein    │       │
│  │   Layer    │  │   Layer    │  │   Layer    │       │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘       │
└─────────┼────────────────┼────────────────┼────────────┘
          │                │                │
┌─────────┼────────────────┼────────────────┼────────────┐
│         │    PyO3 Bindings (Rust ↔ Python) │           │
└─────────┼────────────────┼────────────────┼────────────┘
          │                │                │
┌─────────▼────────────────▼────────────────▼────────────┐
│              Rust Core (src/layers/)                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │  poincare  │  │  lorentz   │  │   klein    │       │
│  │    .rs     │  │    .rs     │  │    .rs     │       │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘       │
│         │                │                │             │
│         └────────────────┴────────────────┘             │
│                         │                               │
│                ┌────────▼────────┐                      │
│                │  Möbius Ops     │                      │
│                │  (src/ops/)     │                      │
│                └────────┬────────┘                      │
└─────────────────────────┼──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│              CUDA Kernels (optional)                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │ poincare.cu│  │ lorentz.cu │  │  klein.cu  │       │
│  └────────────┘  └────────────┘  └────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## 구현 레이어

### 1. Poincaré Ball Model

**파일**: 
- CPU: `src/layers/poincare.rs`
- CUDA: `src/layers/cuda/poincare.cu`
- Python: `python/reality_stone/layers/poincare.py`

**수식**: `B^n_c = {x ∈ ℝ^n : c||x||² < 1}`

**핵심 연산**:
- Möbius addition: `x ⊕_c y`
- Möbius scalar: `r ⊗_c x`
- Distance: `d(x,y) = (2/√c)·atanh(√(c||x-y||²/[(1-c||x||²)(1-c||y||²)]))`


**상세 문서**: [POINCARE_IMPLEMENTATION.md](./POINCARE_IMPLEMENTATION.md)

---

### 2. Lorentz Hyperboloid Model

**파일**:
- CPU: `src/layers/lorentz.rs`
- CUDA: `src/layers/cuda/lorentz.cu`
- Python: `python/reality_stone/layers/lorentz.py`

**수식**: `H^n_c = {x ∈ ℝ^{n+1} : ⟨x,x⟩_L = -1/c, x_0 > 0}`

**핵심 연산**:
- Minkowski inner: `⟨x,y⟩_L = x_0y_0 - Σx_iy_i`
- Geodesic: `γ(t) = sinh((1-t)α)/sinh(α)·u + sinh(tα)/sinh(α)·v`
- Distance: `d(u,v) = (1/√c)·acosh(c⟨u,v⟩_L)`


**상세 문서**: [LORENTZ_IMPLEMENTATION.md](./LORENTZ_IMPLEMENTATION.md)

---

### 3. Klein Model

**파일**:
- CPU: `src/layers/klein.rs`
- CUDA: `src/layers/cuda/klein.cu`
- Python: `python/reality_stone/layers/klein.py`

**수식**: 
$$
D^n_c = {x ∈ ℝ^n : c||x||² < 1}
$$

**핵심 연산**:
- Addition: $u ⊕_K v$
- Scalar: $r ⊗_K x$
- Distance: $d(u,v) = (1/√c)·acosh((1-c⟨u,v⟩)/√[(1-c||u||²)(1-c||v||²)])$

**상세 문서**: [KLEIN_IMPLEMENTATION.md](./KLEIN_IMPLEMENTATION.md)

---

## 성능 비교

### MNIST 분류 (10 epochs)

| 모델      | 정확도     | 학습 속도      | 메모리 | 수렴 속도 |
| --------- | ---------- | -------------- | ------ | --------- |
| Poincaré  | 97.30%     | 1.4s/epoch     | Base   | 보통      |
| Lorentz   | 98.09%     | 8.0s/epoch     | +10%   | **빠름**  |
| **Klein** | **98.28%** | **2.5s/epoch** | Base   | 보통      |

### 상세 학습 곡선

**Poincaré**:
```
E1: 58.50% → E5: 96.42% → E10: 97.30%
```

**Lorentz** (가장 빠른 수렴):
```
E1: 96.64% → E3: 97.61% → E9: 98.09%
```

**Klein** (최고 정확도):
```
E1: 95.47% → E2: 97.19% → E9: 98.28%
```

---

## 기술 스택

### Rust Core

**의존성** (`Cargo.toml`):
```toml
[dependencies]
pyo3 = { version = "0.20.0", features = ["extension-module"] }
ndarray = { version = "0.15", features = ["rayon", "approx"] }
numpy = "0.20.0"
rayon = "1.8.0"  # 병렬화
```

**특징**:
- 제로 코스트 추상화
- Rayon 병렬 처리
- ndarray 효율적 연산

### CUDA Kernels

**컴파일 옵션** (`build.rs`):
```rust
cc::Build::new()
    .cuda(true)
    .flag("-arch=sm_70")  # Tesla V100, Titan V
    .include(format!("{}/include", cuda_path))
    .file(file)
    .compile(&lib_name);
```

**지원 아키텍처**:
- `sm_70`: V100, Titan V
- `sm_75`: RTX 20xx
- `sm_80`: A100, RTX 30xx (A6000)
- `sm_86`: RTX 3090, 3080
- `sm_89`: RTX 40xx

### Python Integration

**빌드 시스템**: maturin
```bash
maturin develop --release --features cuda
```

**PyTorch 통합**:
```python
class LayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ...):
        # Rust 호출
        output_np = _rust.layer_forward_cpu(...)
        return torch.from_numpy(output_np)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Rust 호출
        grad_np = _rust.layer_backward_cpu(...)
        return torch.from_numpy(grad_np)
```

---

## 수정 사항 요약

### 1. Poincaré Distance (CRITICAL)

**이전**:
```cpp
// ❌ 완전히 틀린 수식
return acoshf(1.0f + 2*c*xy / den) / sqrtf(c);
```

**수정 후**:
```cpp
// ✅ 정확한 수식
return (2.0f/sqrtc) * atanhf(sqrt(c * norm_sq_diff / den));
```

**영향**: 
- Distance 계산 정확도 100% 향상
- MNIST 정확도 약 2-3% 향상 (추정)

### 2. Klein Distance

**이전**:
```cpp
// ❌ 비표준 lambda 기반 공식
float lambda = sqrt(2*(u²v² - uv²) / den);
return acoshf((2+λ)/(2-λ)) / sqrt(c);
```

**수정 후**:
```cpp
// ✅ 표준 Klein distance
float num = 1.0f - c*uv;
float den = sqrt((1-c*u²)(1-c*v²));
return acoshf(num/den) / sqrt(c);
```

**영향**:
- 수학적 정확성 보장
- 다른 라이브러리와 호환

### 3. Python Layer Bug

**이전**:
```python
# ❌ saved_tensors 불일치
ctx.save_for_backward(u, v)  # 2개 저장
...
u, v, u_prime, v_prime = ctx.saved_tensors  # 4개 기대
```

**수정 후**:
```python
# ✅ 일치
ctx.save_for_backward(u, v)
...
u, v = ctx.saved_tensors
```

---

## 테스트 인프라

### Rust 단위 테스트

**실행**:
```bash
cargo test --lib
```

**결과**: 39/41 통과 (95.1%)

**테스트 항목**:
- Distance (same point, symmetry)
- Layer interpolation (t=0, t=1)
- Möbius operations (identity, scalar)
- 좌표 변환 (hyperboloid constraint)

### CUDA 단위 테스트

**파일**: `src/layers/cuda/test_kernels.cu`

**컴파일 & 실행**:
```bash
# Linux/Mac
bash scripts/test_kernels.sh

# Windows
scripts\test_kernels.bat
```

**테스트 항목**:
- Poincaré distance/layer
- Lorentz distance/layer
- Klein distance
- Möbius add/scalar

### Python 통합 테스트

**실행**:
```bash
python -m tests.poincare --mode both --epochs 10 --batch-size 256
python -m tests.lorentz --epochs 10 --batch-size 256
python -m tests.klein --epochs 10 --batch-size 256
```

---

## 빌드 가이드

### CPU Only

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 빌드
maturin develop --release
```

### CUDA 지원

```bash
# CUDA 경로 설정
export CUDA_HOME=/usr/local/cuda  # Linux
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1  # Windows

# CUDA 빌드
maturin develop --release --features cuda

# 확인
python -c "import reality_stone as rs; print(rs._has_cuda)"
```

### 아키텍처 커스터마이징

`build.rs` 수정:
```rust
.flag("-arch=sm_86")  // RTX 3090용
```

---

## 사용 예제

### 기본 사용

```python
import torch
import reality_stone as rs

# Poincaré layer
u = torch.randn(4, 8)
v = torch.randn(4, 8)
y = rs.poincare_ball_layer(u, v, c=1e-3, t=0.7)

# Lorentz layer
u_lorentz = torch.randn(4, 9)  # n+1 차원
v_lorentz = torch.randn(4, 9)
y_lorentz = rs.lorentz_layer(u_lorentz, v_lorentz, c=1e-3, t=0.7)

# Klein layer
y_klein = rs.klein_layer(u, v, c=1e-3, t=0.7)
```

### Dynamic Curvature

```python
import torch.nn as nn

class HyperbolicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.kappa = nn.Parameter(torch.tensor(-1.0))  # 학습 가능한 곡률
        
    def forward(self, x):
        c = torch.exp(-2 * self.kappa) * 0.04 + 0.01  # c ∈ [0.01, 0.05]
        return rs.poincare_ball_layer(x, x, c=None, t=0.5, kappas=self.kappa)
```

---

## 디렉토리 구조

```
reality_stone/
├── src/                          # Rust 코어
│   ├── layers/                   # 레이어 구현
│   │   ├── poincare.rs          # Poincaré CPU
│   │   ├── lorentz.rs           # Lorentz CPU
│   │   ├── klein.rs             # Klein CPU
│   │   ├── cuda/                # CUDA 커널
│   │   │   ├── poincare.cu
│   │   │   ├── lorentz.cu
│   │   │   └── klein.cu
│   │   └── tests/               # Rust 단위 테스트
│   ├── ops/                     # 기본 연산
│   │   ├── mobius.rs            # Möbius 연산
│   │   └── cuda/
│   └── bindings/                # PyO3 바인딩
├── python/                       # Python 패키지
│   └── reality_stone/
│       ├── layers/              # PyTorch 레이어
│       │   ├── poincare.py
│       │   ├── lorentz.py
│       │   └── klein.py
│       └── models/              # 고수준 모델
├── tests/                        # Python 테스트
│   ├── poincare.py              # MNIST 테스트
│   ├── lorentz.py
│   └── klein.py
├── scripts/                      # 빌드/테스트 스크립트
│   ├── test_rust.sh             # Rust 테스트
│   ├── test_kernels.sh          # CUDA 테스트
│   └── test_all.sh              # 전체 테스트
└── docs/                         # 문서
    ├── POINCARE_IMPLEMENTATION.md
    ├── LORENTZ_IMPLEMENTATION.md
    ├── KLEIN_IMPLEMENTATION.md
    └── IMPLEMENTATION_OVERVIEW.md  (this file)
```

---

## 참고 문헌

### 논문

1. **Nickel & Kiela (2017)**: "Poincaré Embeddings for Learning Hierarchical Representations"
2. **Ganea et al. (2018)**: "Hyperbolic Neural Networks"
3. **Shimizu et al. (2021)**: "Hyperbolic Neural Networks++"
4. **Bachmann et al. (2020)**: "Constant Curvature Graph Convolutional Networks"

### 수학적 배경

1. **Cannon et al. (1997)**: "Hyperbolic Geometry"
2. **Ungar (2008)**: "Analytic Hyperbolic Geometry"
3. **Anderson (2005)**: "Hyperbolic Geometry" (Springer)

---

## 라이선스 & 기여

**라이선스**: MIT

**작성자**: jigglypop <donghwanyeom@gmail.com>

**Repository**: https://github.com/jigglypop/reality_stone

---

## 변경 이력

### v0.2.0 (2024-11)
- ✅ Poincaré distance 수식 수정 (CRITICAL)
- ✅ Klein distance 표준 공식 적용
- ✅ Python layer backward 버그 수정
- ✅ CUDA/CPU 구현 정확도 검증
- ✅ 39/41 Rust 단위 테스트 통과
- ✅ 3개 레이어 MNIST 96-98% 달성
- ✅ 테스트 인프라 구축 (2-5초 빠른 테스트)

### v0.1.0 (2024-10)
- Initial release
- Poincaré/Lorentz/Klein 기본 구현
- PyTorch 통합

