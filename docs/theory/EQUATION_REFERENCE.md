# Reality Stone 핵심 수식 참조

## 1. 벨만 방정식 계열

### 1.1 가치 함수
```
V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
```

### 1.2 Q-함수
```
Q(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q(s',a')
```

### 1.3 벨만 오차
```
δ = Q(s,a) - [R(s,a) + γ V(s')]
```

## 2. 리만 기하학 계열

### 2.1 메트릭 텐서
```
g_ij(s) = ⟨∂_i, ∂_j⟩
ds² = g_ij dx^i dx^j
```

### 2.2 크리스토펠 기호
```
Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
```

### 2.3 공변 미분
```
∇_i V^j = ∂_i V^j + Γ^j_ik V^k
```

### 2.4 측지선 방정식
```
d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0
```

### 2.5 리만 곡률 텐서
```
R^l_ijk = ∂_j Γ^l_ik - ∂_k Γ^l_ij + Γ^l_jm Γ^m_ik - Γ^l_km Γ^m_ij
```

### 2.6 리치 곡률
```
R_ij = R^k_ikj = g^kl R_kilj
```

## 3. 라그랑지안 역학 계열

### 3.1 라그랑지안
```
L(x, ẋ) = T - V = (1/2) g_ij ẋ^i ẋ^j - V(x)
```

### 3.2 작용
```
S = ∫_t1^t2 L(x, ẋ) dt
```

### 3.3 오일러-라그랑주 방정식
```
d/dt (∂L/∂ẋ^i) - ∂L/∂x^i = 0
```

### 3.4 해밀토니안
```
H = p_i ẋ^i - L = T + V
```
여기서 `p_i = ∂L/∂ẋ^i = g_ij ẋ^j`

### 3.5 해밀턴 방정식
```
dq^i/dt = ∂H/∂p_i
dp_i/dt = -∂H/∂q^i
```

## 4. 하이퍼볼릭 기하학 계열

### 4.1 푸앵카레 메트릭
```
g_ij^P = (4/(1-||x||²)²) δ_ij
```

### 4.2 푸앵카레 거리
```
d_P(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
```

### 4.3 뫼비우스 덧셈
```
x ⊕_c y = ((1+2c⟨x,y⟩+c||y||²)x + (1-c||x||²)y) / (1+2c⟨x,y⟩+c²||x||²||y||²)
```

### 4.4 뫼비우스 스칼라 곱
```
r ⊗_c x = (1/√c) tanh(r tanh^(-1)(√c ||x||)) (x/||x||)
```

### 4.5 로렌츠 메트릭
```
⟨x,y⟩_L = -x_0 y_0 + Σ_(i=1)^d x_i y_i
```

### 4.6 로렌츠 거리
```
d_L(x,y) = arcosh(-⟨x,y⟩_L)
```

### 4.7 클라인 메트릭
```
g_ij^K = δ_ij/(1-||x||²) + x_i x_j/(1-||x||²)²
```

## 5. 통합 손실 함수

### 5.1 벨만 손실
```
L_bellman = E_τ [(Q(s,a) - (R + γ V(s')))²]
```

### 5.2 라그랑지안 손실
```
L_lagrangian = ∫ [(1/2) g_ij ẋ^i ẋ^j - V(x)]² dt
```

### 5.3 메트릭 정규화
```
L_metric = ||g - SPD(g)||²_F + λ |det(g) - 1|
```
SPD(g)는 g의 최근접 양정치 행렬

### 5.4 강화학습 손실
```
L_rl = -E_τ [Σ_t γ^t R(s_t, a_t)]
```

### 5.5 창의성 정규화
```
L_creativity = -||∂_t V||_g = -√(g^ij ∂_t V_i ∂_t V_j)
```

### 5.6 전체 손실
```
L_total = L_bellman + λ_1 L_lagrangian + λ_2 L_metric + λ_3 L_rl + λ_4 L_creativity
```

권장 하이퍼파라미터:
- λ_1 = 0.1
- λ_2 = 0.01
- λ_3 = 1.0
- λ_4 = 0.01

## 6. 자연 그라디언트

### 6.1 Fisher 정보 행렬
```
F_ij = E[∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j]
```

### 6.2 자연 그라디언트
```
∇̃_θ L = F^(-1) ∇_θ L
```

### 6.3 업데이트 규칙
```
θ_(t+1) = θ_t - η F^(-1) ∇_θ L
```

## 7. 시간 미분 (창의성)

### 7.1 시간 의존 가치
```
V(s,t) = V^(0)(s) + t V^(1)(s) + (t²/2) V^(2)(s) + ...
```

### 7.2 시간 미분
```
∂_t V = lim_(Δt→0) [V(s,t+Δt) - V(s,t)] / Δt
```

### 7.3 창의성 측정
```
C(s) = ||∂_t V||_g = √(g^ij ∂_t V_i ∂_t V_j)
```

### 7.4 시간 패턴 (주기성)
```
V(s,t) = V_0(s) + Σ_k [A_k(s) cos(ω_k t) + B_k(s) sin(ω_k t)]
```

여기서 ω_k는 특성 주파수:
- 일주기: ω = 2π / (24시간)
- 월주기: ω = 2π / (30일)
- 연주기: ω = 2π / (365일)

## 8. 메트릭 암호화

### 8.1 키 기반 스케일링
```
g_encrypted(s, K) = exp(K · H(s)) · g_base(s)
```

### 8.2 복호화
```
g_base(s) = exp(-K · H(s)) · g_encrypted(s, K)
```

### 8.3 해시 함수
```
H(s) = SHA256(s) mod p
```

### 8.4 키 검증
```
verify(g, K) = [det(g) > 0] ∧ [all eigenvalues > 0]
```

## 9. 정책 그라디언트

### 9.1 정책 그라디언트 정리
```
∇_θ J(θ) = E_τ [Σ_t ∇_θ log π_θ(a_t|s_t) Q^π(s_t,a_t)]
```

### 9.2 메트릭 조건부 정책
```
∇_θ J(θ) = E_τ [Σ_t ∇_θ log π_θ(a_t|s_t,g_t) Q^π(s_t,a_t)]
```

### 9.3 Actor-Critic
```
∇_θ J(θ) = E_τ [Σ_t ∇_θ log π_θ(a_t|s_t) (Q(s_t,a_t) - V(s_t))]
```

### 9.4 PPO 목적 함수
```
L^CLIP(θ) = E_t [min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
```

여기서 `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`

## 10. 벨만-리만 통합 수식

### 10.1 벨만-측지선 동등성
```
V(s) = max_a [R(s,a) + γ V(s')]
⟺
d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0
```

증명: 벨만 최적 정책은 가치 함수 공간의 측지선

### 10.2 에너지 보존
```
E = (1/2) g_ij ẋ^i ẋ^j + V(x) = constant
```

이는 벨만 일관성 조건 `V(s) = R + γ V(s')`과 동등

### 10.3 메트릭-Fisher 대응
```
g_ij(θ) = E[∂log p/∂θ_i · ∂log p/∂θ_j] = F_ij
```

### 10.4 측지선 흐름
```
dx^i/dt = g^ij ∂V/∂x^j
```

이는 자연 그라디언트 흐름과 동일

### 10.5 최적 제어
```
u*(t) = argmin_u [L(x,u) + λ(ẋ - f(x,u))]
```

여기서 λ는 라그랑주 승수 (벨만 가치 함수와 대응)

## 11. 계산 복잡도

### 11.1 크리스토펠 기호
```
시간: O(d³)
공간: O(d³)
```

### 11.2 메트릭 역행렬
```
시간: O(d³) (Cholesky 분해)
공간: O(d²)
```

### 11.3 고유값 분해 (배치)
```
시간: O(B·d³) → O(d³) (병렬화)
공간: O(B·d²)
```

### 11.4 측지선 업데이트
```
시간: O(d² + d³) ≈ O(d³)
공간: O(d²)
```

### 11.5 벨만 업데이트
```
시간: O(|A|)
공간: O(|S|)
```

## 12. 수치 안정성

### 12.1 메트릭 정규화
```
g_regularized = g + ε·I
```
권장: ε = 0.1

### 12.2 로그 공간 연산
```
log det(g) = tr(log(g)) = Σ log(λ_i)
```

### 12.3 그라디언트 클리핑
```
g_clipped = g / max(1, ||g||/threshold)
```
권장: threshold = 1.0

### 12.4 지수 스케일링
```
exp_safe(x) = exp(clip(x, -10, 10))
```

## 13. 실용 공식

### 13.1 배치 메트릭 연산
```python
metric_inv = torch.linalg.inv(metric + eps * I)
christoffel = 0.5 * torch.einsum('...kl,...ilj->...kij', metric_inv, metric_grad)
```

### 13.2 자연 그라디언트 근사
```python
nat_grad = grad / (fisher + damping)
```

### 13.3 창의성 계산
```python
creativity = torch.sqrt(torch.sum(time_deriv * torch.solve(time_deriv, metric), dim=-1))
```

## 14. 참고 수식

### 14.1 소프트맥스
```
σ(x)_i = exp(x_i) / Σ_j exp(x_j)
```

### 14.2 크로스 엔트로피
```
H(p,q) = -Σ_i p_i log(q_i)
```

### 14.3 KL 발산
```
D_KL(p||q) = Σ_i p_i log(p_i/q_i)
```

### 14.4 Jensen-Shannon 발산
```
D_JS(p||q) = (1/2)[D_KL(p||m) + D_KL(q||m)]
```
여기서 m = (p+q)/2

## 15. 최적화 팁

### 15.1 학습률 스케줄
```
η(t) = η_0 / (1 + decay · t)
```
권장: η_0 = 0.01, decay = 0.0001

### 15.2 모멘텀
```
v_(t+1) = β·v_t + (1-β)·∇L
θ_(t+1) = θ_t - η·v_(t+1)
```
권장: β = 0.9

### 15.3 Adam
```
m_t = β_1·m_(t-1) + (1-β_1)·∇L
v_t = β_2·v_(t-1) + (1-β_2)·(∇L)²
θ_t = θ_(t-1) - η·m_t/√(v_t + ε)
```
권장: β_1 = 0.9, β_2 = 0.999, ε = 1e-8

## 16. 특수 함수

### 16.1 쌍곡 함수
```
sinh(x) = (exp(x) - exp(-x))/2
cosh(x) = (exp(x) + exp(-x))/2
tanh(x) = sinh(x)/cosh(x)
```

### 16.2 역쌍곡 함수
```
arcsinh(x) = log(x + √(x²+1))
arccosh(x) = log(x + √(x²-1))
arctanh(x) = (1/2)log((1+x)/(1-x))
```

### 16.3 로짓
```
logit(p) = log(p/(1-p))
```

### 16.4 시그모이드
```
sigmoid(x) = 1/(1+exp(-x))
```

## 사용 예제

```python
import torch

s = torch.randn(16, 64)
a = torch.randn(16, 8)
r = torch.randn(16)

metric = compute_metric(s)

gamma = christoffel_symbols(metric)

velocity = (s_next - s) / dt

L = 0.5 * torch.einsum('bi,bij,bj->b', velocity, metric, velocity) - V(s)

action = torch.einsum('bijk,bj,bk->bi', gamma, velocity, velocity)

loss = bellman_loss + 0.1 * L.mean()
```

## 단위 시스템

모든 수식에서:
- 시간: 초 (s) 또는 스텝
- 공간: 임베딩 차원 (dim)
- 에너지: 무차원 (정규화)
- 곡률: m^(-2) (하이퍼볼릭 공간)
- 할인 인자: 무차원 (0 < γ < 1)

## 기호 정리

- s: 상태 (state)
- a: 행동 (action)
- V: 가치 함수 (value)
- Q: Q-함수
- g: 메트릭 텐서
- Γ: 크리스토펠 기호
- L: 라그랑지안
- H: 해밀토니안
- γ: 할인 인자
- η: 학습률
- ε: 정규화 상수
- θ: 파라미터
- ∇: 그라디언트
- ∂: 편미분
- d: 전미분
- ⊕: 뫼비우스 덧셈
- ⊗: 뫼비우스 스칼라 곱
- ⟨·,·⟩: 내적
- ||·||: 노름

## 참고

모든 수식은 Einstein summation convention 사용:
반복되는 인덱스는 자동으로 합산됨

예: `g_ij x^i y^j = Σ_i Σ_j g_ij x^i y^j`

