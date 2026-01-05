# LB-IGD (Laplace-Beltrami Inverse Game Design) 문서 가이드

> **한 줄 요약**: 벨만 방정식을 라플라스-벨트라미로 재매개화하여, 게임 밸런스 설계를 **확산 PDE 기반 연속 최적화**로 푸는 프레임워크.

---

## 문서 구조 (읽기 순서)

```
[01. 이 문서 (01_guide.md)]
    |
    v
[02. 02_bellman.md] ────────────────────────────────────┐
    벨만 방정식의 전제와 설계 최적화의 계층적 정식화      |
    - 플레이 문제 vs 설계 문제 분리                      |
    - 왜 벨만/DP가 설계 레벨에 직접 적용되기 어려운가     |
    - 확률적 black-box 최적화로의 정식화                 |
    |                                                    |
    v                                                    |
[03. 03_lbo.md] ────────────────────────────────────────┤
    라플라스-벨트라미(LBO): 벨만 → HJB → Ito → Delta_g   |
    - 정리 4.2: Bellman <-> LBO 완전 정합 증명           |
    - 정리 4.3: 토션/드리프트 확장                       |
    - 승률로 치환, 그래프 라플라시안 이산화               |
    |                                                    |
    v                                                    |
[04. 04_blackbox.md] ───────────────────────────────────┤
    확률적 블랙박스 최적화: ES 유도와 교전 거리 분포 정합  |
    - Score Function Estimator (ES) 유도                 |
    - 분산 감소: 대칭 샘플링, CRN                        |
    - Wasserstein 거리 W_2 정합                         |
    |                                                    |
    v                                                    |
[05. 05_evaluation.md] ─────────────────────────────────┤
    평가 프로토콜: self-play 기반 성공 판정과 재현성      |
    - 고정해야 할 것들 (학습 예산, 시드, 매칭)            |
    - 승률 기반 밸런스 지표                              |
    - 퇴화 해 탐지                                       |
    |                                                    |
    v                                                    |
[06. 06_inverse.md] ────────────────────────────────────┤
    역설계: 목표 메타 기반 팩션/맵 생성                   |
    - 팩션 정체성을 분포로 정의                          |
    - 생성기 g(z; phi)와 설계 x 동시 최적화              |
    - 분포 정합 (Wasserstein)                           |
    |                                                    |
    v                                                    |
[07. 07_synapse.md] ────────────────────────────────────┘
    (부록) 뇌와의 연결: 시냅스 가소성, 케이블 방정식
    - STDP, 3-factor 학습
    - 케이블 방정식 → Delta_g + 드리프트 유도
    - 뇌 물리와 수학 프레임워크의 1:1 대응
    - "뭘 얻나" (이론적/계산적 이득)
```

---

## 핵심 정리 요약

### 정리 4.2 (03_lbo.md): Bellman <-> LBO 완전 정합

연속시간 Markov reward process에서 생성자가 \(\mathcal{L}=\nu\Delta_g\)이면:

$$
V(x) := \mathbb{E}_x\Big[\int_0^\infty e^{-\rho t} r(X_t) dt\Big]
$$

는 다음을 만족한다:

$$
\boxed{\rho V - \nu \Delta_g V = r}
$$

- 그래프 이산화: \((\rho I + \nu L) v = r\)
- 승률 치환: \(r(x) = P(x)\) 또는 \(r(x) = -J(x)\)

### 정리 4.3 (03_lbo.md): 토션/드리프트 확장

시냅스 방향성(STDP)을 넣으면 생성자가 \(\mathcal{L}=\nu\Delta_g + b\cdot\nabla_g\)가 되고:

$$
\boxed{(\rho - \nu\Delta_g - b\cdot\nabla_g) V = r}
$$

- 그래프 이산화: \((\rho I + \nu L_{\text{sym}} + \kappa A) v = r\)
- \(A = (W - W^\top)/2\): 반대칭 성분 (토션/방향성)

---

## 비개발자를 위한 요약

### 이게 뭔가요?

게임 밸런스를 맞추는 건 정말 힘든 일입니다.
- "이 유닛이 너무 센가?"
- "맵이 너무 좁아서 원거리 유닛이 불리한가?"
- "선공이 너무 유리한가?"

보통은 기획자가 감으로 수치를 조절하고, QA팀이 수백 번 플레이해보며 맞춥니다.
**LB-IGD**는 이 과정을 자동화했습니다.

1. **AI 기획자(Optimizer)**가 게임 규칙(유닛 스펙, 맵 크기 등)을 제안합니다.
2. **AI 플레이어(Agent)** 두 명이 그 규칙으로 수백 판을 싸웁니다.
3. 결과를 분석해서 "이 규칙은 A가 너무 유리하네, 기각!" 또는 "이건 50:50으로 박빙이네, 합격!" 하고 판단합니다.
4. 이 과정을 반복하며 점점 더 완벽한 밸런스를 찾아냅니다.

### 핵심 기술: "튼튼한 밸런스" (LBO)

어떤 게임 규칙이 승률 50:50이 나왔다고 칩시다. 그런데 유닛 체력을 딱 1만 줄여도 승률이 90:10으로 확 깨진다면?
그건 좋은 밸런스가 아닙니다. "살얼음판 밸런스"죠.

우리는 **"튼튼한 밸런스"**를 원합니다.
- 유닛 공격력이 조금 바뀌어도,
- 맵 배치가 살짝 달라져도,
- 플레이어가 실수를 좀 해도,

여전히 50:50을 유지하는 **안정적인 규칙**을 찾습니다.
이걸 수학적으로 계산하는 기술이 바로 **LBO(라플라스-벨트라미 연산자)**입니다. 
AI가 "너무 예민한 규칙"은 스스로 피하도록 만듭니다.

---

## 코드 대응

| 문서 | 주요 코드 |
|-----|----------|
| 02_bellman.md | `src/layers/bellman.rs`, `bellman_lagrangian.rs` |
| 03_lbo.md | `experiments/lbigd/core/lbo.py`, `laplace_beltrami_matrix` (Rust) |
| 04_blackbox.md | `experiments/lbigd/core/designer.py` (ES + CRN + LBO 가중) |
| 05_evaluation.md | `experiments/lbigd/core/simulation.py` |
| 06_inverse.md | (일부 개념은 아직 미구현) |
| 07_synapse.md | `experiments/lbigd/core/metric.py`, `dopamine.py` |

---

## 수학적 배경이 필요한 독자를 위한 추천 순서

1. **02_bellman.md 1-3절**: MDP/벨만 기초, 플레이 vs 설계 분리
2. **03_lbo.md 1-4절**: HJB, 이토, 라플라시안 출현, 정리 4.2/4.3 증명
3. **07_synapse.md 11절**: 뇌 물리(케이블 방정식)에서 동일 수학 유도
4. **04_blackbox.md**: ES 유도, 분산 감소
5. **05_evaluation.md**: 평가 프로토콜
6. **06_inverse.md**: 역설계 확장

---

## 빠른 참조: 핵심 수식

| 이름 | 수식 |
|-----|------|
| 이산 벨만 | \(V^*(s) = \max_a [r + \gamma \mathbb{E}[V^*(s')]]\) |
| 연속 HJB | \(\rho V = \sup_a [r + \mathcal{L}^a V]\) |
| 확산 생성자 | \(\mathcal{L} = \nu\Delta_g + b\cdot\nabla_g\) |
| Bellman <-> LBO | \((\rho - \nu\Delta_g) V = r\) |
| 그래프 이산화 | \((\rho I + \nu L + \kappa A) v = r\) |
| ES 그라디언트 | \(\nabla J \approx \mathbb{E}[(J(x+\sigma e) - J(x-\sigma e)) e / (2\sigma)]\) |
| 케이블 방정식 | \(\tau_m \partial_t V = \lambda^2 \partial_{xx} V - (V - V_{\text{rest}}) + R_m I_{\text{syn}}\) |
