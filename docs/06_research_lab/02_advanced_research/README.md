# Reality Stone 연구 문서 (Research Documentation)

이 디렉토리는 Reality Stone 프로젝트의 핵심 연구 내용을 담고 있습니다.

## 📚 문서 목록

### 01. [순수 다양체와 확산](01_PURE_MANIFOLD_DIFFUSION.md)
**주제**: 층 없는 딥러닝 - 기하학과 동역학만으로 AI 구현

**핵심 가설**: "학습 가능한 선형 계층을 완전히 제거하고, 기하학적 공간과 에너지 흐름만으로 고성능 AI를 구현할 수 있는가?"

**실험 결과**:
- 순수 다양체 모델 (Klein): 92.82% (MNIST, 학습 가능한 층 0개)
- 다양체 확산 모델: 97.87% (MNIST, 단일 재귀 구조)
- 리만 라그랑지안 디퓨전: 97.18% (MNIST, Rust+CUDA)

**의의**: 깊이(Depth)를 시간(Time)으로 치환 가능함을 입증

---

### 02. [Intent SOTA Diffusion](02_INTENT_SOTA_DIFFUSION.md)
**주제**: 의도 분류를 위한 최첨단 디퓨전 모델

**핵심 기술**:
- DeBERTa-v3 기반 인코더
- 리만 다양체 디퓨전
- Banking77 데이터셋 적용

---

### 03. [YDE - 역동역학 인코더](03_YDE.md)
**주제**: Yogācāra Dynamics Encoder - 유식(唯識) 철학 기반 표현 학습

**철학적 배경**:
- 아뢰야식(Ālayavijñāna): 모든 경험의 저장소
- 종자(Bīja): 잠재적 가능성
- 훈습(Vāsanā): 경험에 의한 변화

**구현**: 리만 다양체 위의 역학 시스템으로 표현

---

### 04. [리만 동역학 최소 사양](04_RIEMANNIAN_DYNAMICS_SPEC.md)
**주제**: Reality Stone 리만 역학 시스템의 기술 사양

**핵심 구성 요소**:
1. 상태 공간: Poincaré Ball (곡률 c > 0)
2. 에너지: 운동 에너지 + 퍼텐셜 에너지
3. 라그랑지안: L = T - V
4. 측지선 흐름: 지수/로그 맵

**구현 파일**:
- `src/layers/metric.rs`: 메트릭 텐서
- `src/layers/geodesic.rs`: 지수/로그 맵
- `src/layers/bellman_lagrangian.rs`: 에너지 계산

**성능**: MNIST 97.18% (5 에폭, ~9초/에폭)

---

### 05. [리만 라그랑지안 디퓨전](05_LAGRANGIAN_DIFFUSION.md) ⭐ **최신**
**주제**: 물리 기반 신경망 학습 - 라그랑지안 역학과 리만 기하학의 융합

**핵심 개념**:
1. **상태 공간 = 리만 다양체**: 신경망 은닉 상태가 Poincaré Ball에 존재
2. **학습 = 에너지 최소화**: 라그랑지안 $\mathcal{L} = T - V$ 최소화
3. **디퓨전 = 시간 진화**: 측지선을 따라 에너지 최소 상태로 흐름

**기술 스택**:
- **Rust Core**: 리만 메트릭, 에너지 계산, 측지선
- **CUDA Kernel**: GPU 직접 연산 (`src/layers/cuda/diffusion.cu`)
- **PyTorch Integration**: Zero-Copy Autograd 통합

**성능**:
- 정확도: **97.18%** (MNIST, 5 에폭)
- 속도: ~9초/에폭 (CUDA Zero-Copy)
- 개선: CPU 복사 대비 **10배 빠름**

**실험 코드**: `experiments/benchmark_mnist_diffusion.py`

**API**:
```python
# Rust 디퓨전 엔진 초기화
engine = rs.PyRiemannianDiffusion(dim=2048, alpha=0.9, dt=0.1)

# CUDA 스텝 (Zero-Copy)
engine.step_cuda(h.data_ptr(), flow.data_ptr(), out.data_ptr(), N, D)
```

**의의**:
1. 신경망 학습을 물리 법칙으로 재해석
2. 기하학적 귀납 편향 (계층 구조 자동 학습)
3. 고성능 구현 (실용적 속도)
4. 확장 가능 (뇌 모델링, 강화학습, 생성 모델)

---

## 🎯 연구 철학

Reality Stone 프로젝트는 단순한 "더 나은 신경망"을 만드는 것이 아니라, **새로운 AI 패러다임**을 제시합니다:

### 전통적 딥러닝
```
Input → Linear → ReLU → Linear → ReLU → ... → Output
        (층 층 쌓기)
```

### Reality Stone 접근
```
Input → [Fixed High-dim Basis] → [Riemannian Manifold] → [Lagrangian Dynamics] → Output
        (공간의 기하학)        (측지선 흐름)         (물리 법칙)
```

**핵심 차이**:
1. **Geometry over Architecture**: 구조보다 공간의 기하학이 중요
2. **Dynamics over Layers**: 층 대신 시간 진화
3. **Physics over Heuristics**: 경험적 기법 대신 물리 법칙

---

## 📊 주요 결과 요약

| 모델 | 학습 가능 층 | MNIST 정확도 | 특징 |
|------|-------------|-------------|------|
| 순수 다양체 (Klein) | 0개 | 92.82% | 프로토타입만 학습 |
| 유클리드 디퓨전 | 1개 (재귀) | 97.87% | 시간 진화 |
| 리만 디퓨전 (Rust+CUDA) | 1개 (재귀) | 97.18% | 물리 법칙 + 10배 빠름 |

---

## 🔬 향후 연구 방향

### 단기 (진행 중)
- [ ] ImageNet, CIFAR-100 확장
- [ ] Spiking Neural Network (SNN) 통합
- [ ] 뇌 신경 역학 모델링 (fMRI)

### 중기
- [ ] 변분 리만 디퓨전 (확률적 버전)
- [ ] 다중 다양체 융합 (Poincaré + Lorentz + Klein)
- [ ] 적응형 메트릭 학습 (데이터 기반 곡률)

### 장기
- [ ] Transformer 아키텍처 통합
- [ ] 대규모 언어 모델 적용
- [ ] 양자 컴퓨팅 시뮬레이션

---

## 📖 참고 문헌

1. **리만 기하학**: Do Carmo, "Riemannian Geometry"
2. **쌍곡 신경망**: Ganea et al., "Hyperbolic Neural Networks" (NeurIPS 2018)
3. **라그랑지안 역학**: Goldstein, "Classical Mechanics"
4. **벨만 방정식**: Sutton & Barto, "Reinforcement Learning"
5. **열 방정식 (다양체)**: Grigor'yan, "Heat Kernel and Analysis on Manifolds"

---

## 🛠️ 실험 재현

모든 실험은 `experiments/` 디렉토리에서 재현 가능합니다:

```bash
# 순수 다양체 모델
uv run python experiments/benchmark_mnist_pure.py

# 유클리드 디퓨전
uv run python experiments/benchmark_mnist_diffusion_euclidean.py

# 리만 라그랑지안 디퓨전 (Rust+CUDA)
uv run maturin develop --features cuda
uv run python experiments/benchmark_mnist_diffusion.py
```

---

*Reality Stone: Where Geometry Meets Dynamics, and AI Meets Physics.*

