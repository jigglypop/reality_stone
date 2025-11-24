## 리만 동역학(Riemannian Dynamics) 최소 사양

### 1) 상태, 다양체, 메트릭
- 상태 q ∈ M: Poincaré 볼(곡률 c > 0)
- 원점 접공간 매개변수 z ∈ R^d, 사상: q = exp_0^c(z)
- 메트릭 g^c: Poincaré 메트릭. Reality Stone는 exp/log, 거리 연산의 수치안정 처리를 제공

### 2) 에너지와 라그랑지안
- 운동에너지: T(q̇) = 0.5 · ||q̇||^2_g
- 퍼텐셜: V(q; μ) = 0.5 · d_c(q, μ)^2 (μ는 클래스 프로토타입)
- 라그랑지안: L(q, q̇) = T(q̇) − V(q)

### 3) 이산 지오데식 그래디언트 흐름(heavy-ball)
- 원점 접공간에서 갱신하고 exp로 되감기:
  - q_t = exp_0^c(z_t)
  - Φ(z) = 0.5 · d_c(exp_0^c(z), μ)^2
  - 갱신(배치 벡터화):
    - g_t = ∇_z Φ(z_t)
    - v_{t+1} = β · v_t − η · g_t
    - z_{t+1} = z_t + v_{t+1}
    - q_{t+1} = exp_0^c(z_{t+1})

### 4) 학습 목표
- 조건(라벨) 수준 정렬: 뇌 RDM(코사인)과 모델 RDM(푸앵카레 거리)의 상관 최대화 (프로토타입만 학습)
- 붕괴 방지: 모델 거리의 분산을 키우고(표준편차↑), 접공간 노름을 [r_min, r_max]로 유도
- 샘플 생성: 무작위 z_0에서 위 흐름을 T스텝 돌려 q_T 생성, 샘플 RDM은 Poincaré 거리로 평가

### 5) Reality Stone 연동
- 지수/로그 사상: `rs.layers.poincare.exp_map_zero`, `rs.layers.poincare.log_map_zero`
- 거리: `rs.poincare_distance`


