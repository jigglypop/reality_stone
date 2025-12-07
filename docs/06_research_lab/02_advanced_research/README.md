# Reality Stone 연구 문서 (Research Documentation)

이 디렉토리는 Reality Stone 프로젝트의 핵심 연구 내용을 담고 있습니다.

## 📚 문서 목록

### 01. 리만 라그랑지안 디퓨전
**파일**: `05_라그랑주_디퓨전.md`

**주제**: 리만 라그랑지안 디퓨전을 이용한 기하학 기반 딥러닝

**핵심 내용**:
- 상태 공간을 리만 다양체로 두고, 라그랑지안 역학과 디퓨전을 결합한 학습 구조
- MNIST 등에서 층 없는 구조로도 높은 정확도를 달성한 실험 결과

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

