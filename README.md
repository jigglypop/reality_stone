
# Reality Stone: 완전한 AGI 아키텍처

Reality Stone은 **벨만-리만 통합 이론**, **하이퍼볼릭 기하학**, **계층적 LLM**을 결합한 차세대 AGI 아키텍처입니다.

## 핵심 철학

**단일 원리**: 최소 작용의 원리 (Principle of Least Action)

모든 사고와 학습 과정을 하나의 물리 법칙으로 통합:

```
δ∫L dt = 0
```

여기서 라그랑지안 L은:
```
L = (운동 에너지) - (잠재 에너지)
  = (1/2) g_μν ẋ^μ ẋ^ν - (-Q*(x) + V_reg(x,g))
```

## 핵심 혁신

1. **벨만 방정식을 좌표계로**: 강화학습의 가치 함수를 신경망의 기본 좌표계로 사용
2. **리만 메트릭을 공간 구조로**: 학습 가능한 기하학적 구조로 계층 관계 자연스럽게 표현
3. **라그랑지안을 최적화 원리로**: 물리적 최소작용원리로 측지선 경로 탐색
4. **3개 하이퍼볼릭 레이어 병렬**: Poincaré, Lorentz, Klein 모델 동시 활용
5. **시간축 창의성**: 시간 미분으로 창의성 정량화
6. **자연 그라디언트**: Fisher 정보 행렬 기반 최적화

## 성능 목표

- **압축률**: 2-3배 (860억 → 340억 파라미터)
- **학습 속도**: 2-3배 빠른 수렴
- **추론 능력**: 1.2-1.5배 향상 (동일 데이터 조건)
- **계층적 추론**: 1.4-2.0배 향상

현재 릴리스: 0.2.0, 라이선스: MIT, Python 3.8–3.12, PyTorch 2.0+ 지원.


## 아키텍처 계층

Reality Stone AGI는 7개 계층으로 구성됩니다:

```
Level 0: 물리적 기반 (최소 작용의 원리)
    ↓
Level 1: 벨만 좌표계 (강화학습 통합)
    ↓
Level 2: 리만 메트릭 (학습 가능한 기하학)
    ↓
Level 3: 3개 하이퍼볼릭 레이어 (Poincaré, Lorentz, Klein)
    ↓
Level 4: 계층적 LLM (Sentence-Topic 구조)
    ↓
Level 5: 라그랑지안 최적화 (에너지 최소화)
    ↓
Level 6: 시간축 창의성 (시간 미분)
    ↓
Level 7: 자연 그라디언트 (Fisher 정보)
```

## 주요 기능

### 1. 벨만-리만 통합
- **BellmanCoordinateSystem**: 가치 함수 기반 좌표계
- **RiemannianMetricTensor**: 상태 의존적 메트릭 학습
- **LagrangianEnergySystem**: 물리적 에너지 최소화
- **TemporalCreativityModule**: 시간 미분 창의성 측정

### 2. 하이퍼볼릭 기하학 (Rust + CUDA)
- **3개 모델 병렬**: Poincaré, Lorentz, Klein
- **고성능 커널**: CUDA 최적화 (10-100배 가속)
- **동적 곡률**: 레이어별 학습 가능한 곡률
- **모델 변환**: 3개 기하학 간 seamless 변환

### 3. 계층적 LLM
- **Tree Processor**: Bottom-up & Top-down 메시지 패싱
- **Sentence-Topic Head**: 문장-주제 계층 구조
- **Metric Attention**: SPD 메트릭 기반 어텐션
- **Structural Edit**: 문장 구조 편집 (Replace/Insert/Delete/Reorder)
- **Top-Down Decoder**: 계층적 생성

### 4. 최적화
- **Natural Gradient**: Fisher 정보 행렬 기반
- **배치 고유값 분해**: 100배 속도 향상
- **Fast SPD Mixing**: 10-100배 속도 향상
- **Mixed Precision**: FP16 학습 지원


## 설치 및 빌드

사전 준비물
- Python 3.8 이상, pip
- Rust toolchain (stable), Cargo
- PyTorch 2.0 이상
- CUDA 사용 시: NVIDIA CUDA Toolkit 설치, 환경 변수 `CUDA_HOME` 또는 `CUDA_PATH` 설정

가상환경 생성 및 필수 패키지 설치
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU 전용 빌드
```bash
maturin develop
python -c "import reality_stone as rs; print(rs._has_rust_ext, rs._has_cuda)"
```

CUDA 빌드
```bash
export CUDA_HOME=/usr/local/cuda   # Windows: set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x
maturin develop --features cuda --release
python -c "import reality_stone as rs; print(rs._has_rust_ext, rs._has_cuda)"
```

주의: 기본 CUDA 아키텍처 플래그는 `sm_70`입니다. 다른 GPU를 사용한다면 `build.rs`의 `-arch=sm_70`을 환경에 맞게 수정하세요.


## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/jigglypop/reality_stone.git
cd reality_stone
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

maturin develop --release

maturin develop --features cuda --release
```

### 2. 완전한 AGI 모델

```python
from reality_stone.models.agi import RealityStoneAGI, AGIConfig

config = AGIConfig(
    state_dim=128,
    action_dim=10,
    hidden_dim=256,
    num_layers=4,
    num_topics=8,
    use_encryption=True,
    enable_top_down=True,
    enable_structural_edit=True
)

model = RealityStoneAGI(config)

state = torch.randn(16, 128)
action = torch.randn(16, 10)
key = torch.randn(16, 32)

outputs = model(state, action, key, return_all=True)

print(f"Value: {outputs['value']}")
print(f"Policy: {outputs['policy']}")
print(f"Creativity: {outputs['creativity']}")
print(f"Lagrangian Loss: {outputs['lagrangian_loss']}")
```

### 3. 계층적 LLM

```python
from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalLLMConfig,
    train_hierarchical_llm_from_text
)

config = HierarchicalLLMConfig(
    d_model=128,
    num_topics=8,
    use_fast_spd_mixing=True,
    enable_top_down=True,
    enable_structural_edit=True
)

model, info = train_hierarchical_llm_from_text(
    data_path="data/text.txt",
    config=config,
    epochs=10,
    batch_size=4
)
```

### 4. 하이퍼볼릭 레이어

```python
import torch
import reality_stone as rs

u = torch.randn(4, 8)
v = torch.randn(4, 8)
y = rs.poincare_ball_layer(u, v, c=1e-3, t=0.7)

x = torch.randn(2, 4)
y = torch.randn(2, 4)
d = rs.poincare_distance(x, y, c=1e-3)
xL = rs.poincare_to_lorentz(x, c=1e-3)
```

### 5. 데모 실행

```bash
python examples/bellman_riemannian_demo.py

python examples/train_on_real_data.py

python -m tests.poincare --quick
python -m tests.lorentz --quick
python -m tests.klein --quick
```


## 프로젝트 구조

```
reality_stone/
├── docs/                               # 문서
│   ├── COMPLETE_AGI_ARCHITECTURE.md   # ⭐ 완전한 AGI 아키텍처
│   ├── AGI_IMPLEMENTATION_ROADMAP.md  # ⭐ 구현 로드맵
│   ├── CORE_EQUATIONS.md              # 핵심 수식
│   ├── IMPLEMENTATION_GUIDE.md        # 구현 가이드
│   ├── BELLMAN_RIEMANNIAN_SUMMARY.md  # 요약
│   ├── unified_geometric_agi_architecture.md  # 통합 조감도
│   └── ...
├── src/                                # Rust 코어
│   ├── layers/                         # 하이퍼볼릭 레이어
│   │   ├── poincare.rs
│   │   ├── lorentz.rs
│   │   ├── klein.rs
│   │   └── cuda/                       # CUDA 커널
│   ├── ops/                            # 연산자
│   └── bindings/                       # PyO3 바인딩
├── python/reality_stone/               # Python API
│   ├── models/                         # 모델
│   │   ├── agi.py                      # ⭐ 통합 AGI 모델
│   │   ├── hierarchical_sentence_topic_llm.py  # 계층적 LLM
│   │   ├── bellman.py                  # 벨만 좌표계
│   │   └── ...
│   ├── layers/                         # Autograd 레이어
│   └── optimizers/                     # 최적화기
├── examples/                           # 예제
│   ├── bellman_riemannian_demo.py
│   ├── train_on_real_data.py
│   └── ...
└── tests/                              # 테스트
```

## 데이터 플로우

```
Python Input
    ↓
벨만 좌표 인코딩
    ↓
리만 메트릭 계산
    ↓
[Poincaré] ← PyO3 → Rust/CUDA
[Lorentz]  ← PyO3 → Rust/CUDA  (병렬)
[Klein]    ← PyO3 → Rust/CUDA
    ↓
메트릭 가중 결합
    ↓
계층적 LLM 처리
    ↓
라그랑지안 최적화
    ↓
시간 미분 (창의성)
    ↓
자연 그라디언트 업데이트
    ↓
Output (Value + Policy + Generated Text)
```


## API 개요 (Python)

상위 함수
- `poincare_ball_layer(u, v, c, t)`
- `lorentz_layer(u, v, c, t)`
- `klein_layer(u, v, c, t)`

거리/변환
- `poincare_distance(x, y, c)`
- `poincare_to_lorentz(x, c)`, `poincare_to_klein(x, c)`
- `lorentz_to_poincare(x, c)`, `lorentz_to_klein(x, c)`
- `klein_to_poincare(x, c)`, `klein_to_lorentz(x, c)`

레이어
- `PoincareBallLayer`, `LorentzLayer`, `KleinLayer`
- 하이퍼볼릭 선형 변형: `HyperbolicLinear`, `GeodesicLinear`, `EquivalentHyperbolicLinear`
- 압축: `SplineLinear`

기타
- 투영: `project_to_ball(x, epsilon)`
- 메트릭 합성: `from reality_stone import metrikey` (SPD 메트릭/합성/암시적 변환 함수 제공)


## 테스트 실행

### 벨만-리만 데모
```bash
python examples/bellman_riemannian_demo.py
```

### 하이퍼볼릭 레이어 테스트
```bash
python -m tests.poincare --mode both --quick --epochs 2 --batch-size 256
python -m tests.lorentz  --quick --epochs 2 --batch-size 256
python -m tests.klein    --quick --epochs 2 --batch-size 256
```

공통 옵션: `--device {auto,cpu,cuda}`, `--data-dir tests/data`, `--epochs`, `--batch-size`, `--lr`, `--t`, `--c`, `--quick`, `--seed`


## 문제 해결

- Rust 확장 모듈을 찾지 못함: `maturin develop`로 빌드 후 다시 시도하세요.
- CUDA가 비활성으로 표시됨: `CUDA_HOME`/`CUDA_PATH` 확인, GPU 드라이버/Toolkit 설치 상태 점검, CUDA 피처로 빌드했는지 확인.
- Windows 빌드: Visual C++ Build Tools 설치 필요. PowerShell 대신 CMD/Developer Prompt 또는 Git Bash 사용 가능.
- CUDA 아키텍처 오류: `build.rs`의 `-arch=sm_70`을 환경에 맞게 수정.
- NumPy 2.x와의 호환성: 현재 `numpy>=1.21,<2.0`을 사용합니다.


## 변경 사항 요약 (v0.2.0)

### 새로운 기능 (벨만-리만 통합)
- **벨만-리만 통합 아키텍처**: 벨만 방정식, 리만 기하학, 라그랑지안 역학 통합
- **핵심 수식 문서화**: `docs/CORE_EQUATIONS.md` - 모든 핵심 수식과 이론적 근거
- **구현 가이드**: `docs/IMPLEMENTATION_GUIDE.md` - 모듈별 상세 구현 방법
- **데모 예제**: `examples/bellman_riemannian_demo.py` - 실행 가능한 완전한 데모
- **BellmanCoordinateSystem**: 벨만 방정식을 좌표계로 사용
- **RiemannianMetricTensor**: 상태 의존적 메트릭 학습 + 암호화
- **LagrangianEnergySystem**: 라그랑지안 기반 최적화
- **TemporalCreativityModule**: 시간 미분으로 창의성 측정
- **NaturalGradientOptimizer**: Fisher 정보 행렬 기반 최적화

### 기존 기능 개선
- Poincaré/Lorentz/Klein 레이어 및 연산의 Python Autograd 경로 정비
- 동적/레이어별 곡률 API 추가: `poincare_ball_layer_layerwise_cpu` 및 정확한 backward
- 스플라인 압축 레이어 `SplineLinear` 추가 및 `from_linear` 최적화 파이프라인 도입
- 하이퍼볼릭 선형 변형 레이어군 추가: `HyperbolicLinear`, `GeodesicLinear`, `EquivalentHyperbolicLinear`
- `metrikey` 서브모듈 공개: SPD 메트릭 합성/적용, 암시적 변환 체인

### 성능 예측
- 압축률: 2-3배 (860억 → 340억 파라미터)
- 학습 속도: 2-3배 (Natural Gradient + 라그랑지안)
- 추론 능력: 1.2-1.5배 (동일 데이터 조건)


## 라이선스

MIT License

