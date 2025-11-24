## 리만 동역학(Riemannian Dynamics) 최소 사양

> **업데이트**: 본 사양은 현재 Reality Stone의 **리만 라그랑지안 디퓨전** 시스템으로 완전히 구현되었습니다.  
> 상세 내용은 `05_LAGRANGIAN_DIFFUSION.md`를 참조하세요.

### 1) 상태, 다양체, 메트릭
- 상태 q ∈ M: Poincaré 볼(곡률 c > 0)
- 원점 접공간 매개변수 z ∈ R^d, 사상: q = exp_0^c(z)
- 메트릭 g^c: Poincaré 메트릭. Reality Stone는 exp/log, 거리 연산의 수치안정 처리를 제공

**구현 파일**:
- `src/layers/metric.rs`: DiagonalMetric, PoincareMetric, LorentzMetric, KleinMetric
- `src/layers/geodesic.rs`: exponential_map, logarithmic_map

### 2) 에너지와 라그랑지안
- 운동에너지: T(q̇) = 0.5 · ||q̇||^2_g
- 퍼텐셜: V(q; μ) = 0.5 · d_c(q, μ)^2 (μ는 클래스 프로토타입)
- 라그랑지안: L(q, q̇) = T(q̇) − V(q)

**구현 파일**:
- `src/layers/bellman_lagrangian.rs`: kinetic_energy, bellman_potential, lagrangian
- `src/layers/diffusion.rs`: RiemannianDiffusion (디퓨전 역학 엔진)

### 3) 이산 지오데식 그래디언트 흐름(heavy-ball)
- 원점 접공간에서 갱신하고 exp로 되감기:
  - q_t = exp_0^c(z_t)
  - Φ(z) = 0.5 · d_c(exp_0^c(z), μ)^2
  - 갱신(배치 벡터화):
    - g_t = ∇_z Φ(z_t)
    - v_{t+1} = β · v_t − η · g_t
    - z_{t+1} = z_t + v_{t+1}
    - q_{t+1} = exp_0^c(z_{t+1})

**구현 파일**:
- `src/layers/cuda/diffusion.cu`: CUDA 커널로 GPU 가속
- `src/bindings/diffusion.rs`: Python 바인딩 (Zero-Copy)
- `experiments/benchmark_mnist_diffusion.py`: MNIST 실험 (97.18% 정확도)

### 4) 학습 목표
- 조건(라벨) 수준 정렬: 뇌 RDM(코사인)과 모델 RDM(푸앵카레 거리)의 상관 최대화 (프로토타입만 학습)
- 붕괴 방지: 모델 거리의 분산을 키우고(표준편차↑), 접공간 노름을 [r_min, r_max]로 유도
- 샘플 생성: 무작위 z_0에서 위 흐름을 T스텝 돌려 q_T 생성, 샘플 RDM은 Poincaré 거리로 평가

### 5) Reality Stone 연동
- 지수/로그 사상: `rs.layers.poincare.exp_map_zero`, `rs.layers.poincare.log_map_zero`
- 거리: `rs.poincare_distance`
- **디퓨전 엔진**: `rs.PyRiemannianDiffusion(dim, alpha, dt)`
  - CUDA 가속: `engine.step_cuda(h_ptr, flow_ptr, out_ptr, N, D)`
  - CPU Fallback: `engine.step_cpu(h_numpy, flow_numpy)`

### 6) 성능 지표

**MNIST 벤치마크** (2048-dim, 5 steps, alpha=0.9):
- 정확도: **97.18%** (5 에폭)
- 속도: ~9초/에폭 (CUDA Zero-Copy)
- 개선: CPU 복사 버전 대비 **10배 빠름**

### 7) 참고 문서
- 상세 이론: `05_LAGRANGIAN_DIFFUSION.md`
- 철학: `../01_philosophy/01_WHY_RIEMANNIAN.md`
- API: `../API_BINDINGS.md`


