
# Reality Stone

Reality Stone은 **벨만-리만 통합 이론**, **하이퍼볼릭 기하학**, **계층적 LLM**을 결합한 차세대 아키텍처입니다.

## 핵심 철학

**단일 원리**: 최소 작용의 원리 (Principle of Least Action)

모든 사고와 학습 과정을 하나의 물리 법칙으로 통합:

```
δ∫L dt = 0

L = (1/2) g_μν ẋ^μ ẋ^ν - (-Q*(x) + V_reg(x,g))
```

## 핵심 혁신

1. **벨만 방정식을 좌표계로**: 강화학습의 가치 함수를 신경망의 기본 좌표계로 사용
2. **리만 메트릭을 공간 구조로**: 학습 가능한 기하학적 구조로 계층 관계 자연스럽게 표현
3. **라그랑지안을 최적화 원리로**: 물리적 최소작용원리로 측지선 경로 탐색
4. **3개 하이퍼볼릭 레이어 병렬**: Poincare, Lorentz, Klein 모델 동시 활용
5. **시간축 창의성**: 시간 미분으로 창의성 정량화
6. **자연 그라디언트**: Fisher 정보 행렬 기반 최적화
7. **Bellman-LBO 완전 정합**: 이산 벨만을 라플라스-벨트라미 PDE로 재매개화

## 아키텍처 계층

```
Level 0: 물리적 기반 (최소 작용의 원리)
    ↓
Level 1: 벨만 좌표계 (강화학습 통합)
    ↓
Level 2: 리만 메트릭 (학습 가능한 기하학)
    ↓
Level 3: 3개 하이퍼볼릭 레이어 (Poincare, Lorentz, Klein)
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

### 벨만-리만 통합
- `BellmanCoordinateSystem`: 가치 함수 기반 좌표계
- `RiemannianMetricTensor`: 상태 의존적 메트릭 학습
- `LagrangianEnergySystem`: 물리적 에너지 최소화
- `TemporalCreativityModule`: 시간 미분 창의성 측정

### 하이퍼볼릭 기하학 (Rust + CUDA)
- 3개 모델 병렬: Poincare, Lorentz, Klein
- 고성능 커널: CUDA 최적화 (10-100배 가속)
- 동적 곡률: 레이어별 학습 가능한 곡률

### 계층적 LLM
- Tree Processor: Bottom-up & Top-down 메시지 패싱
- Sentence-Topic Head: 문장-주제 계층 구조
- Metric Attention: SPD 메트릭 기반 어텐션

### LB-IGD (Laplace-Beltrami Inverse Game Design)

게임 밸런스 설계를 확산 PDE 기반 연속 최적화로 푸는 프레임워크.

- **정리 4.2**: 연속시간 MRP에서 Bellman과 LBO가 완전 동치
- **정리 4.3**: 시냅스 방향성(STDP)을 드리프트로 확장
- **뇌 물리 대응**: 케이블 방정식에서 동일한 수학이 유도됨

```
(rho - nu*Delta_g - b.grad_g) V = r
```

자세한 내용: [docs/08_lb_igd/01_guide.md](docs/08_lb_igd/01_guide.md)


## 설치

**사전 준비물**: Python 3.8+, Rust toolchain, PyTorch 2.0+, (선택) CUDA Toolkit

```bash
git clone https://github.com/jigglypop/reality_stone.git
cd reality_stone

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# CPU 빌드
maturin develop --release

# CUDA 빌드 (선택)
export CUDA_HOME=/usr/local/cuda  # Windows: set CUDA_PATH=...
maturin develop --features cuda --release
```

## 빠른 시작

### AGI 모델

```python
from reality_stone.models.agi import RealityStoneAGI, AGIConfig
import torch

config = AGIConfig(
    state_dim=128, action_dim=10, hidden_dim=256,
    num_layers=4, num_topics=8
)
model = RealityStoneAGI(config)

outputs = model(
    torch.randn(16, 128),  # state
    torch.randn(16, 10),   # action
    torch.randn(16, 32),   # key
    return_all=True
)
# outputs: value, policy, creativity, lagrangian_loss
```

### 하이퍼볼릭 레이어

```python
import reality_stone as rs

y = rs.poincare_ball_layer(u, v, c=1e-3, t=0.7)
d = rs.poincare_distance(x, y, c=1e-3)
xL = rs.poincare_to_lorentz(x, c=1e-3)
```

### 데모 및 테스트

```bash
python examples/bellman_riemannian_demo.py
python -m tests.poincare --quick
python -m tests.lorentz --quick
python -m tests.klein --quick
```

## 프로젝트 구조

```
reality_stone/
├── src/                     # Rust 코어 (하이퍼볼릭 레이어, CUDA 커널)
├── python/reality_stone/    # Python API (모델, 레이어, 최적화기)
├── experiments/lbigd/       # LB-IGD 실험 (ES+LBO 기반 게임 밸런스)
├── docs/                    # 문서
│   └── 08_lb_igd/           # LB-IGD 이론 (Bellman-LBO 증명)
├── examples/                # 예제
└── tests/                   # 테스트
```

## API 요약

| 분류 | 함수/클래스 |
|------|-------------|
| 레이어 | `poincare_ball_layer`, `lorentz_layer`, `klein_layer` |
| 거리 | `poincare_distance`, `lorentz_distance`, `klein_distance` |
| 변환 | `poincare_to_lorentz`, `lorentz_to_klein`, ... |
| 클래스 | `PoincareBallLayer`, `LorentzLayer`, `KleinLayer` |
| 선형 | `HyperbolicLinear`, `GeodesicLinear`, `SplineLinear` |
| 메트릭 | `metrikey` (SPD 메트릭 합성/변환) |

## 문제 해결

| 문제 | 해결 |
|------|------|
| Rust 모듈 없음 | `maturin develop` 실행 |
| CUDA 비활성 | `CUDA_HOME` 설정 후 `--features cuda`로 빌드 |
| Windows 빌드 | Visual C++ Build Tools 필요 |
| CUDA 아키텍처 오류 | `build.rs`의 `-arch=sm_70` 수정 |

## 문서

- [LB-IGD 가이드](docs/08_lb_igd/01_guide.md) - Bellman-LBO 완전 정합 이론
- [AGI 아키텍처](docs/COMPLETE_AGI_ARCHITECTURE.md) - 전체 설계
- [핵심 수식](docs/CORE_EQUATIONS.md) - 수학적 기초
- [구현 가이드](docs/IMPLEMENTATION_GUIDE.md) - 모듈별 구현

## 성능 목표

| 항목 | 목표 |
|------|------|
| 압축률 | 2-3배 (860억 → 340억) |
| 학습 속도 | 2-3배 |
| 추론 능력 | 1.2-1.5배 |

---

v0.2.0 | MIT License | Python 3.8-3.12 | PyTorch 2.0+
