## Reality Stone Core / Kernel 문서 인덱스

### 1. 목적

- **역할**: `reality_stone`의 하이퍼볼릭 / 리만 기하 코어와 CUDA 커널 구현에 대한 문서를 한 곳에서 모아서 정리한다.
- **분리 원칙**:
  - 하이퍼볼릭 커널 / 수학 / 성능 관련 내용은 `docs/core/`에서 다룬다.
  - Sentence-Topic LLM 및 상위 모델 설계는 `docs/llm/` 하위 문서에 정리한다.

### 2. 핵심 문서 링크

- **전체 구현 개요**: `../IMPLEMENTATION_OVERVIEW.md`
- **Poincaré 모델 레이어/거리**: `../POINCARE_IMPLEMENTATION.md`
- **Lorentz 하이퍼볼로이드 모델**: `../LORENTZ_IMPLEMENTATION.md`
- **Klein 모델 레이어/거리**: `../KLEIN_IMPLEMENTATION.md`
- **리만 메트릭 / MetriKey / SPD 합성**: `../RIEMANNIAN_METRIC_LEARNING.md`

위 문서들은 Rust 레이어, CUDA 커널, Python Autograd 레이어 간의 연계를 상세히 설명하며,  
새로운 LLM 설계(`docs/llm/` 하위 문서)에서 “하위 커널로 재사용되는 계층”에 해당한다.

### 3. 커널 및 코어 파일 구조 요약

- **Rust 레이어 (쌍곡 모델)**
  - `src/layers/poincare.rs`
  - `src/layers/lorentz.rs`
  - `src/layers/klein.rs`
  - `src/layers/riemann.rs`
- **CUDA 커널**
  - `src/layers/cuda/poincare.cu`
  - `src/layers/cuda/lorentz.cu`
  - `src/layers/cuda/klein.cu`
  - `src/layers/cuda/mobius.cu`
- **리만 / SPD / MetriKey**
  - `src/ops/metrikey.rs`
  - `src/ops/curvature.rs`
  - `src/ops/mobius.rs`
  - `src/ops/project.rs`

각 커널의 수학적 정의와 구현 세부사항은 위에 링크된 구현 문서에서 다루며,  
이 인덱스 문서는 “어떤 파일이 어떤 문서와 대응되는지”를 빠르게 확인하기 위한 용도이다.

### 4. CUDA 빌드 및 사용 개요

- **빌드 플래그**
  - Cargo feature: `cuda`
  - 빌드 스크립트: `build.rs` 에서 `CUDA_HOME` 또는 `CUDA_PATH` 기반으로 CUDA Toolkit 경로를 검색.
  - 기본 아키텍처 플래그: `-arch=sm_70` (GPU에 맞게 조정 필요).

- **Python 빌드 예시**
  - CPU 전용:
    - `maturin develop --release`
  - CUDA 포함:
    - `maturin develop --release --features cuda`

- **런타임에서의 CUDA 감지**
  - Python 진입점: `python/reality_stone/__init__.py`
  - 플래그:
    - **`_has_rust_ext`**: Rust 확장 모듈 로딩 여부.
    - **`_has_cuda`**: PyTorch CUDA 가능 여부와 Rust 모듈 내 CUDA 심볼 존재 여부를 동시에 검사한 결과.
  - 현재 설계:
    - Möbius / Lorentz / Klein 에 대해서는 CUDA 커널이 제공되며,
    - Poincaré 레이어는 **안전한 CPU 경로가 기본**이고, CUDA 심볼이 없으면 자동으로 CPU로 폴백된다.

CUDA가 활성화된 경우에도, 개별 레이어는 다음 정책을 따른다.

- **조건**: 텐서가 `.is_cuda == True` 이고 `_has_cuda == True` 인 경우에만 CUDA 경로를 시도한다.
- **그 외 경우**: 항상 Rust CPU 또는 순수 PyTorch 구현으로 폴백한다.

이렇게 분리해 두면, 일부 모델(Poincaré)이 아직 CUDA 바인딩을 완전히 갖추지 못한 상태에서도  
나머지 커널(Möbius, Lorentz, Klein)은 안전하게 CUDA를 활용할 수 있다.

### 5. LLM 설계 문서와의 관계

- **코어 / 커널 레벨**: 이 문서와 상기 구현 문서들은
  - 거리 함수, Möbius 연산, 곡률 최적화, SPD 메트릭 합성 같은 **저수준 연산의 정확도와 성능**에 초점을 둔다.
- **LLM / 상위 레벨**:
  - Sentence-Topic Guided LLM, RCE-Transformer, 리만 메트릭 컨텍스트 스위칭과 같은 상위 설계는
  - `docs/llm/` 하위 문서에서 다루며, 여기서 정의된 커널과 메트릭 연산을 그대로 재사용하는 구조를 취한다.

코어 문서와 LLM 문서를 명시적으로 분리해 둠으로써,

- 커널/수학/성능 개선 작업과
- 상위 LLM 아키텍처/데이터 파이프라인 작업을 서로 독립적으로 리팩터링하기 쉽도록 한다.

### 6. 코어 커널 API 요약 (Rust / CUDA / Python)

#### 6.1 Poincare / Möbius 계열

- **Rust 레이어 (CPU)**  
  - `src/layers/poincare.rs`
    - `poincare_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32>`
    - `poincare_ball_layer(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32, t: f32) -> Array2<f32>`
    - `poincare_ball_layer_backward(grad_output, u, v, c, t) -> (Array2<f32>, Array2<f32>)`
  - `src/ops/mobius.rs`
    - `mobius_add(u, v, c)` / `mobius_scalar(u, c, r)` (정적 곡률)
    - `mobius_add_dynamic` / `mobius_add_layerwise` (+ 각 backward)

- **CUDA 커널 (C++)**  
  - `src/layers/cuda/poincare.cu`
    - `poincare_distance_kernel(...)` → FFI: `poincare_distance_cuda(...)`
    - `poincare_ball_layer_forward_kernel(...)` → FFI: `poincare_ball_layer_cuda(...)`
    - `poincare_ball_layer_backward_kernel(...)` → FFI: `poincare_ball_layer_backward_cuda(...)`
  - `src/ops/cuda/mobius.cu`
    - `mobius_add_kernel(...)` → FFI: `mobius_add_cuda(...)`
    - `mobius_scalar_kernel(...)` → FFI: `mobius_scalar_cuda(...)`

- **PyO3 바인딩 / Python 공개 API**  
  - `src/bindings/poincare.rs`
    - CPU: `poincare_ball_layer_cpu`, `poincare_ball_layer_backward_cpu`, `poincare_distance_cpu`
    - CUDA: `poincare_distance_cuda`, `poincare_ball_layer_cuda`, `poincare_ball_layer_backward_cuda`
  - `src/bindings/mobius.rs`
    - CPU: `mobius_add_cpu`, `mobius_scalar_cpu`, `mobius_add_dynamic_cpu`, ...
    - CUDA: `mobius_add_cuda`, `mobius_scalar_cuda`
  - `python/reality_stone/layers/poincare.py`
    - Autograd: `PoincareBallLayer(Function)` (`forward`/`backward` 내부에서 Rust/ CUDA 경로 선택)
    - Helpers: `poincare_add`, `poincare_scalar_mul`, `poincare_distance`, `poincare_to_lorentz`, `poincare_to_klein`
  - `python/reality_stone/__init__.py`
    - 최상위: `poincare_ball_layer`, `poincare_distance`, `MobiusAdd`, `MobiusScalarMul`

#### 6.2 Lorentz 하이퍼볼로이드

- **Rust 레이어 (CPU)**  
  - `src/layers/lorentz.rs`
    - 기본 연산: `lorentz_inner`, `lorentz_distance`, `lorentz_add`, `lorentz_exp0_space`, `lorentz_log0_space`
    - 레이어: `lorentz_layer_forward(u, v, c, t) -> Array2<f32>` / `lorentz_layer_backward(...)`
    - 동적 곡률: `lorentz_layer_dynamic`, `lorentz_layer_layerwise` (+ 각 backward)

- **CUDA 커널 (C++)**  
  - `src/layers/cuda/lorentz.cu`
    - `lorentz_distance_kernel(...)` → FFI: `lorentz_distance_cuda(...)`
    - `lorentz_layer_forward_kernel(...)` → FFI: `lorentz_layer_forward_cuda(...)`
    - `lorentz_layer_backward_kernel(...)` → FFI: `lorentz_layer_backward_cuda(...)`

- **PyO3 바인딩 / Python 공개 API**  
  - `src/bindings/lorentz.rs`
    - CPU: `lorentz_add`, `lorentz_distance`, `lorentz_layer_forward`, `lorentz_layer_backward`, 동적/레이어별 variant들
    - CUDA: `lorentz_distance_cuda`, `lorentz_layer_forward_cuda`, `lorentz_layer_backward_cuda`
  - `python/reality_stone/layers/lorentz.py`
    - Autograd:
      - `LorentzLayer(Function)` → 정적 곡률 geodesic 레이어
      - `LorentzBallLayer(Function)` → 동적/레이어별 곡률 포함 지오데식 레이어
      - `LorentzFromPoincare(Function)` → Poincare → Lorentz 변환 (동적 곡률 옵션)
    - Helpers: `lorentz_add`, `lorentz_scalar_mul`, `lorentz_distance`, `lorentz_inner`, `lorentz_to_poincare`, `lorentz_to_klein`
  - `python/reality_stone/__init__.py`
    - 최상위: `lorentz_layer`, `lorentz_distance`, `LorentzLayer`

#### 6.3 Klein 모델

- **Rust 레이어 (CPU)**  
  - `src/layers/klein.rs`
    - 거리: `klein_distance(u, v, c) -> Array1<f32>`
    - 연산: `klein_add`, `klein_scalar`, `klein_to_poincare`, `klein_to_lorentz`
    - 레이어: `klein_layer_forward(u, v, c, t)`, `klein_layer_backward(...)`

- **CUDA 커널 (C++)**  
  - `src/layers/cuda/klein.cu`
    - `klein_distance_kernel(...)` → FFI: `klein_distance_cuda(...)`
    - `klein_layer_forward_kernel(...)` → FFI: `klein_layer_forward_cuda(...)`
    - `klein_layer_backward_kernel(...)` → FFI: `klein_layer_backward_cuda(...)` (현재 Python 레이어에서는 CPU backward 사용)

- **PyO3 바인딩 / Python 공개 API**  
  - `src/bindings/klein.rs`
    - CPU: `klein_add`, `klein_scalar`, `klein_distance`, `klein_to_poincare`, `klein_to_lorentz`, `klein_layer_forward`, `klein_ball_layer_backward_cpu`
    - CUDA: `klein_distance_cuda`, `klein_layer_forward_cuda`, `klein_ball_layer_backward_cuda`
  - `python/reality_stone/layers/klein.py`
    - Autograd: `KleinLayer(Function)` (`forward` 에서 CPU/CUDA 분기, `backward` 는 항상 CPU 경로 사용)
    - Helpers: `klein_add`, `klein_scalar_mul`, `klein_distance`, `klein_to_poincare`, `klein_to_lorentz`, `project_to_klein`
  - `python/reality_stone/__init__.py`
    - 최상위: `klein_layer`, `klein_distance`, `KleinLayer`

#### 6.4 지오데식 Top-k Attention / SPD 메트릭

- **Geodesic Top-k Attention (CUDA 전용 커널)**  
  - `src/layers/cuda/geodesic_topk_attention.cu`
    - fused 커널: `geodesic_topk_attention_fused_kernel(...)`
  - `src/bindings/geodesic_attention.rs`
    - PyO3: `geodesic_topk_attention(q, k, v, idx, l_factor, c, tau) -> torch.Tensor[B, H, T, d_v]`
    - Python 모듈: `_rust_geodesic` 로 export
  - `python/reality_stone/layers/metric_attention.py`
    - `MetricAttention` 내부에서 `HAS_CUDA_KERNEL` 이 True 이고 `mode="geodesic"` 일 때 `geodesic_topk_attention` 사용

- **SPD / MetriKey / 곡률 유틸리티**  
  - `src/ops/metrikey.rs`
    - 메트릭 키 파싱, SPD 메트릭 슬롯, 캐시/조합 로직
  - `src/layers/riemann.rs`
    - SPD 기반 리만 메트릭 합성, 로우랭크 메트릭 등
  - Python 측:
    - `python/reality_stone/layers/metric_attention.py::SPDMetric` (diag + low-rank SPD 파라메터화)
    - `python/reality_stone/__init__.py` 에서 `metrikey` (Rust 모듈)를 선택적으로 re-export

---

이 섹션은 **“어떤 연산을 어디에서 호출해야 하는지”** 를 빠르게 찾기 위한 API 인덱스이고,  
각 연산의 수식/유도/수치 안정성은 `POINCARE_IMPLEMENTATION.md`, `LORENTZ_IMPLEMENTATION.md`, `KLEIN_IMPLEMENTATION.md`, `RIEMANNIAN_METRIC_LEARNING.md` 에서 자세히 다룬다.
