# Reality Stone 핵심 수식 참조 (Reference)

> **역할:** `CORE_EQUATIONS.md` 에 실린 기본식을 보강하는 “참고용 레퍼런스”로, 구현·연구 단계에서 필요한 세부 정의와 확장 수식을 모아둔 문서다.

## 1. 벨만 방정식 계열

### 1.1 가치 함수
```text
V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
```

### 1.2 Q-함수
```text
Q(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q(s',a')
```

### 1.3 벨만 오차
```text
δ = Q(s,a) - [R(s,a) + γ V(s')]
```

## 2. 리만 기하학 계열

### 2.1 메트릭 텐서와 거리
```text
g_ij(s) = ⟨∂_i, ∂_j⟩
ds² = g_ij dx^i dx^j
```

### 2.2 크리스토펠 기호
```text
Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
```

### 2.3 공변 미분
```text
∇_i V^j = ∂_i V^j + Γ^j_ik V^k
```

### 2.4 측지선 방정식
```text
d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0
```

### 2.5 리만 곡률 텐서와 리치 곡률
```text
R^l_ijk = ∂_j Γ^l_ik - ∂_k Γ^l_ij + Γ^l_jm Γ^m_ik - Γ^l_km Γ^m_ij
R_ij = R^k_ikj = g^kl R_kilj
```

## 3. 라그랑지안 역학 계열

### 3.1 라그랑지안과 작용
```text
L(x, ẋ) = (1/2) g_ij ẋ^i ẋ^j - V(x)
S = ∫_t1^t2 L(x, ẋ) dt
```

### 3.2 오일러-라그랑주 방정식
```text
d/dt (∂L/∂ẋ^i) - ∂L/∂x^i = 0
```

### 3.3 해밀토니안과 해밀턴 방정식
```text
H = p_i ẋ^i - L = T + V
```
여기서 `p_i = ∂L/∂ẋ^i = g_ij ẋ^j`

```text
dq^i/dt = ∂H/∂p_i
dp_i/dt = -∂H/∂q^i
```

## 4. 하이퍼볼릭 기하학 계열

### 4.1 푸앵카레 메트릭/거리
```text
g_ij^P = (4/(1-||x||²)²) δ_ij
d_P(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
```

### 4.2 뫼비우스 연산
```text
x ⊕_c y = ((1+2c⟨x,y⟩+c||y||²)x + (1-c||x||²)y) / (1+2c⟨x,y⟩+c²||x||²||y||²)
r ⊗_c x = (1/√c) tanh(r tanh^(-1)(√c ||x||)) (x/||x||)
```

### 4.3 로렌츠/클라인 메트릭
```text
⟨x,y⟩_L = -x_0 y_0 + Σ_(i=1)^d x_i y_i
d_L(x,y) = arcosh(-⟨x,y⟩_L)
g_ij^K = δ_ij/(1-||x||²) + x_i x_j/(1-||x||²)²
```

## 5. 손실 함수 및 정규화 (확장용)

### 5.1 통합 손실 스케치
```text
L_bellman   = E_τ [(Q(s,a) - (R + γ V(s')))²]
L_lagrangian = ∫ [(1/2) g_ij ẋ^i ẋ^j - V(x)]² dt
L_metric    = ||g - SPD(g)||²_F + λ |det(g) - 1|
L_rl        = -E_τ [Σ_t γ^t R(s_t, a_t)]
L_creativity = -||∂_t V||_g = -√(g^ij ∂_t V_i ∂_t V_j)

L_total = L_bellman + λ_1 L_lagrangian + λ_2 L_metric + λ_3 L_rl + λ_4 L_creativity
```

### 5.2 권장 하이퍼파라미터 (초기 값)
```text
λ_1 = 0.1
λ_2 = 0.01
λ_3 = 1.0
λ_4 = 0.01
```

## 6. 자연 그라디언트 및 Fisher 정보

### 6.1 Fisher 정보 행렬
```text
F_ij = E[∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j]
```

### 6.2 자연 그라디언트와 업데이트
```text
∇̃_θ L = F^(-1) ∇_θ L
θ_(t+1) = θ_t - η F^(-1) ∇_θ L
```

## 7. 시간축과 창의성

### 7.1 시간 의존 가치와 시간 미분
```text
V(s,t) = V^(0)(s) + t V^(1)(s) + (t²/2) V^(2)(s) + ...
∂_t V = lim_(Δt→0) [V(s,t+Δt) - V(s,t)] / Δt
```

### 7.2 창의성 측정 및 주기 표현
```text
C(s) = ||∂_t V||_g = √(g^ij ∂_t V_i ∂_t V_j)
V(s,t) = V_0(s) + Σ_k [A_k(s) cos(ω_k t) + B_k(s) sin(ω_k t)]
```

여기서 예시 주파수:
```text
일주기: ω = 2π / (24시간)
월주기: ω = 2π / (30일)
연주기: ω = 2π / (365일)
```

## 8. 정책 그라디언트 (RL 통합용)

### 8.1 기본 정책 그라디언트
```text
∇_θ J(θ) = E_τ [Σ_t ∇_θ log π_θ(a_t|s_t) Q^π(s_t,a_t)]
```

### 8.2 메트릭 조건부 정책 / Actor-Critic
```text
∇_θ J(θ) = E_τ [Σ_t ∇_θ log π_θ(a_t|s_t,g_t) Q^π(s_t,a_t)]
∇_θ J(θ) = E_τ [Σ_t ∇_θ log π_θ(a_t|s_t) (Q(s_t,a_t) - V(s_t))]
```

### 8.3 PPO 목적 함수
```text
L^CLIP(θ) = E_t [min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

## 9. 벨만-리만 통합 식 (요약)

### 9.1 벨만-측지선 대응
```text
V(s) = max_a [R(s,a) + γ V(s')]
⟺
d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0
```

### 9.2 에너지 보존
```text
E = (1/2) g_ij ẋ^i ẋ^j + V(x) = constant
```

### 9.3 메트릭-Fisher 대응 / 자연 그라디언트 흐름
```text
g_ij(θ) = E[∂log p/∂θ_i · ∂log p/∂θ_j] = F_ij
dx^i/dt = g^ij ∂V/∂x^j
```

## 10. 구현 메모 (요약)

- **단위 시스템**
  - 시간: 초(s) 또는 스텝
  - 공간: 임베딩 차원(dim)
  - 에너지: 무차원(정규화)
  - 곡률: m^(-2) (하이퍼볼릭 공간)
  - 할인 인자: 무차원, `0 < γ < 1`

- **기호**
  - `s`: 상태 (state)
  - `a`: 행동 (action)
  - `V`: 가치 함수 (value)
  - `Q`: Q-함수
  - `g`: 메트릭 텐서
  - `Γ`: 크리스토펠 기호
  - `L`: 라그랑지안
  - `H`: 해밀토니안
  - `γ`: 할인 인자
  - `η`: 학습률
  - `ε`: 정규화 상수
  - `θ`: 파라미터
  - `⊕`: 뫼비우스 덧셈
  - `⊗`: 뫼비우스 스칼라 곱

- **합 기호 규약 (Einstein Summation)**
  - 반복되는 인덱스는 자동 합산한다.
  - 예: `g_ij x^i y^j = Σ_i Σ_j g_ij x^i y^j`