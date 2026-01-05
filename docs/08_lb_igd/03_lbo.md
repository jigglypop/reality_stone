## 제2장 라플라스–벨트라미(LBO): 벨만 → HJB → 이토(Itô) → 라플라시안 → (다양체) LBO → (설계공간) 2차 차분 정규화

> 본 문서는 LB-IGD에서 **"왜 라플라시안/라플라스-벨트라미(LBO) 관점이 필요한가?"**를
> 강화학습의 기반인 **벨만 방정식**에서 출발해, 필요한 만큼만(최소한의 증명 스케치) 연결해서 설명합니다.
>
> - **읽기 순서**: 제1장 `02_bellman.md` → 제2장 (현재) → 제3장 `04_blackbox.md` → 제4장 `05_evaluation.md` → 제5장 `06_inverse.md`
> - **부록**: `07_synapse.md` (뇌와의 연결, 정리 4.2/4.3의 물리적 유도)
> - **코드 대응**: `experiments/lbigd/core/designer.py`, `lbo.py`, `laplace_beltrami_matrix` (Rust)

---

### 0. 표기(Notation)
이 문서에서 자주 쓰는 기호만 최소로 고정합니다.

- 설계 변수: \(x \in \mathcal{X}\) (맵/장애물/유닛 수/패턴/스탯 등)
- 설계가 만드는 게임: \(\mathcal{M}_x\) (플레이 레벨의 MDP)
- (플레이 레벨) 가치함수: \(V(s)\) 또는 \(V(x)\)
- (설계 레벨) 승률 지형: \(P(x)\in[0,1]\)
  - 2팩션이면 보통 \(P(x)=\mathrm{WinRate}_{p0}(x)\)
  - 다팩션이면 \(P(x)\)를 쌍대결 승률행렬 \(W_{f,g}(x)\)로 본다(`05_evaluation.md`)

---

### 1. 출발점: 이산 시간 벨만 방정식
강화학습의 표준 형태(이산 시간, 마르코프성 가정)에서, 최적 가치함수는

$$
V^*(s)
=
\max_{a\in\mathcal{A}}\Big(
r(s,a)+\gamma\,\mathbb{E}[V^*(s')\mid s,a]
\Big)
$$

를 만족합니다. 핵심은 “현재 가치 = 즉시 보상 + 미래 가치의 할인된 기대값”이라는 **재귀 구조**입니다.

---

### 2. 연속 시간으로 보내기: HJB(해밀턴–야코비–벨만)
연속 시간 버전은 “아주 짧은 시간 \(\Delta t\)”에 대해 동적계획 원리를 다시 쓰는 것으로 얻습니다.
직관적으로는 다음 형태를 가정합니다.

1) 상태는 연속 시간에서 확률미분방정식(SDE)을 따른다:

$$
dX_t = b(X_t,a_t)\,dt + \sigma(X_t,a_t)\,dW_t
$$

2) 할인율을 \(\rho>0\)로 두고, 짧은 시간 구간에서의 Bellman 원리를 쓴다:

$$
V(x)=
\max_a \mathbb{E}\Big[
\int_0^{\Delta t} e^{-\rho t} r(X_t,a)\,dt
+
e^{-\rho\Delta t} V(X_{\Delta t})
\Big].
$$

\(\Delta t\to 0\)에서의 1차 근사를 적용하면

- \(e^{-\rho \Delta t}\approx 1-\rho\Delta t\)
- \(\mathbb{E}[V(X_{\Delta t})]\approx V(x) + \Delta t\,\mathcal{L}^a V(x)\) (여기서 \(\mathcal{L}^a\)는 생성자(generator))

가 되고, 정리하면 HJB가 됩니다.

여기서 (유클리드 공간에서) 생성자 \(\mathcal{L}^a\)는 위 SDE의 표준 형태로

$$
\mathcal{L}^a V(x)
=
\nabla V(x)^\top b(x,a)
+
\frac{1}{2}\mathrm{Tr}\!\Big(\sigma(x,a)\sigma(x,a)^\top \nabla^2 V(x)\Big)
$$

로 쓸 수 있습니다. 따라서

- \(\sigma\equiv 0\) (결정론적 동역학)이면 2차항이 사라져 HJB는 **1차 PDE**로 정확히 환원됩니다.
- \(\sigma\neq 0\) (확률적 확산 포함)이면 2차항이 추가되고, 등방 확산이면 \(\nu\Delta V\)로 단순화됩니다(아래 3절).

$$
0
=
\max_a\Big(
r(x,a)+\mathcal{L}^a V(x)-\rho V(x)
\Big).
$$

이제 핵심은 “\(\mathcal{L}^a\) 안에 무엇이 들어있는가?” 입니다.

---

### 3. 이토(Itô) 공식이 만들어내는 2차 미분 항(라플라시안의 출현)
SDE

$$
dX_t = b\,dt + \sigma\,dW_t
$$

에 대해, 충분히 매끄러운 함수 \(V(X_t)\)에 이토 공식을 적용하면

$$
dV
=
\nabla V^\top b\,dt
+
\frac{1}{2}\mathrm{Tr}\!\Big(\sigma\sigma^\top \nabla^2 V\Big)\,dt
+
\nabla V^\top \sigma\,dW_t.
$$

여기서 마지막 항 \(\nabla V^\top \sigma\,dW_t\)는 **변동(랜덤) 항**입니다. \(\mathbb{E}[dW_t]=0\)이므로, 기대값 기반 방정식(HJB)에서는 이 항이 생성자 \(\mathcal{L}^a\)에 직접 들어가지 않습니다. 대신 확산의 효과는 \(dt\)에 곱해진 2차항으로 요약됩니다.

여기서 **2차 미분 항**이 바로

$$
\frac{1}{2}\mathrm{Tr}\!\Big(\sigma\sigma^\top \nabla^2 V\Big)
$$

입니다. 만약 확산이 등방성이라서 \(\sigma\sigma^\top = 2\nu I\) (스칼라 \(\nu>0\))라면

$$
\frac{1}{2}\mathrm{Tr}(2\nu I \nabla^2 V)
=
\nu\,\mathrm{Tr}(\nabla^2 V)
=
\nu\,\Delta V.
$$

즉, **확률적 확산을 포함하면 라플라시안 \(\Delta V\)** 가 자연스럽게 등장합니다.
이게 “확산이 곧 2차 미분(곡률)”이라는 핵심 연결고리입니다.

---

### 4. 왜 “라플라스–벨트라미(LBO)”인가? (좌표 불변성)
위의 \(\Delta\)는 유클리드 공간(\(\mathbb{R}^n\))의 라플라시안입니다.
하지만 상태/설계공간이 “굽은 공간(다양체)”로 모델링되어야 한다면, 좌표를 바꿔도 의미가 유지되는 연산자가 필요합니다.
그때 쓰는 것이 라플라스–벨트라미 연산자 \(\Delta_g\)입니다.

#### 4.1 정의(표준 형태)
리만 다양체 \((\mathcal{M},g)\)에서 스칼라 함수 \(f\)에 대한 LBO는

$$
\Delta_g f := \mathrm{div}_g(\nabla_g f)
$$

로 정의됩니다. 좌표계 \((x^1,\ldots,x^n)\)에서의 전개식은 다음입니다.

$$
\Delta_g f
=
\frac{1}{\sqrt{|g|}}\partial_i\Big(\sqrt{|g|}\,g^{ij}\partial_j f\Big).
$$

- \(g^{ij}\)는 메트릭 행렬 \(g\)의 역행렬 성분
- \(|g|\)는 \(\det(g)\)
- 반복 인덱스는 합(sum)으로 약속

유클리드 공간에서는 \(g=I\), \(|g|=1\)이므로 \(\Delta_g=\sum_i \partial_{ii}=\Delta\)로 돌아갑니다.

---

#### 4.2 정리(완전 정합): 연속시간 “벨만 원리”는 \((\rho-\nu\Delta_g)V=r\)로 정확히 쓴다
여기서는 “벨만 방정식을 라플라스–벨트라미로 대체”가 **정확히(동치로) 성립하는** 최소 조건을 정리합니다.
핵심은 **(i) 이산 시간을 없애 연속시간으로 보내고**, **(ii) 최적화(행동 선택)를 없애거나(=action이 없는 MRP) 정책을 고정**하는 것입니다.

> 요약: 제어가 없는(혹은 정책 고정된) 연속시간 MDP에서는 Bellman(동적계획 원리)이 **선형**이고, 생성자(generator)가 \( \nu\Delta_g\)이면 Bellman은 정확히 \((\rho-\nu\Delta_g)V=r\)가 됩니다.

##### 정리 4.2 (할인된 연속시간 Markov reward process의 Bellman ↔ LBO)
\((\mathcal{M},g)\)를 경계가 없는(compact) 리만 다양체로 두고, \((X_t)_{t\ge 0}\)를 \(\mathcal{M}\) 위 확산 과정이라 하자.
이 과정의 생성자(generator)가

$$
(\mathcal{L}f)(x) = \nu\,\Delta_g f(x)
$$

인 경우(예: 분산 스케일이 \(\nu\)인 Brownian motion), 보상(러닝 리워드) \(r:\mathcal{M}\to\mathbb{R}\) (유계 연속; 고전해를 원하면 \(C^2\) 정도로 충분)와 할인율 \(\rho>0\)에 대해

$$
V(x)\;:=\;\mathbb{E}_x\Big[\int_0^\infty e^{-\rho t}\,r(X_t)\,dt\Big]
$$

로 정의한 함수 \(V\)는 다음을 만족한다.

1) (Bellman / DPP) 임의의 \(\Delta t>0\)에 대해

$$
V(x)
=
\mathbb{E}_x\Big[
\int_0^{\Delta t} e^{-\rho t}\,r(X_t)\,dt
+
e^{-\rho\Delta t}\,V(X_{\Delta t})
\Big].
$$

2) (PDE) \(V\)는 (고전해의 의미로)

$$
\boxed{\;\rho V(x)-\nu\,\Delta_g V(x)=r(x)\;}
$$

를 만족한다.
또한 \(\mathcal{M}\)이 compact이고 \(r\)가 충분히 매끄럽다면, 위 PDE의 **유계 해는 유일**하다.

##### 증명
**(1) DPP(Bellman) 증명.**
\(\int_0^\infty\) 적분을 \([0,\Delta t]\)와 \([\Delta t,\infty)\)로 쪼개면

$$
\int_0^\infty e^{-\rho t}r(X_t)\,dt
=
\int_0^{\Delta t} e^{-\rho t}r(X_t)\,dt
+
e^{-\rho\Delta t}\int_0^\infty e^{-\rho s}r(X_{\Delta t+s})\,ds
$$

이다. 양변에 \(\mathbb{E}_x[\cdot]\)를 취하고, 마르코프 성질로

$$
\mathbb{E}_x\Big[\int_0^\infty e^{-\rho s}r(X_{\Delta t+s})\,ds\;\Big|\;\mathcal{F}_{\Delta t}\Big]
=V(X_{\Delta t})
$$

이므로 1)이 성립한다.

**(2) PDE 증명(“Bellman → LBO”).**
1)의 양변에서 \(V(x)\)를 이항하고 \(\Delta t\)로 나눈 뒤 \(\Delta t\to 0\) 극한을 취한다.
먼저

$$
\frac{1-e^{-\rho\Delta t}}{\Delta t}\to \rho
$$

이고, 생성자의 정의(임의의 충분히 매끄러운 \(f\)에 대해)

$$
\mathcal{L}f(x)
:=
\lim_{\Delta t\to 0}\frac{\mathbb{E}_x[f(X_{\Delta t})]-f(x)}{\Delta t}
$$

를 \(f=V\)에 적용하면

$$
\lim_{\Delta t\to 0}\frac{\mathbb{E}_x[V(X_{\Delta t})]-V(x)}{\Delta t}=\mathcal{L}V(x).
$$

또한 \(\Delta t\to 0\)에서 \(\frac{1}{\Delta t}\mathbb{E}_x[\int_0^{\Delta t} e^{-\rho t} r(X_t)\,dt]\to r(x)\) (연속성과 dominated convergence로 충분) 이다.
따라서 1)로부터

$$
0
=
r(x) - \rho V(x) + \mathcal{L}V(x)
$$

가 되고, \(\mathcal{L}=\nu\Delta_g\)를 대입하면 \(\rho V-\nu\Delta_g V=r\)가 된다.

> (엄밀한 버전 메모) 위 “\(\Delta t\to 0\)” 논리는 \(V\)가 생성자의 정의역에 있어야 한다.
> 이를 가장 깔끔하게 쓰려면 반군 \(P_t f(x):=\mathbb{E}_x[f(X_t)]\)를 두고
>
> $$
> V(x)=\int_0^\infty e^{-\rho t}\,(P_t r)(x)\,dt
> $$
>
> (즉 \(V=R_\rho r\), resolvent)로 정의한 뒤,
> \(\partial_t P_t r = \mathcal{L} P_t r\)를 이용해 적분 부분적분으로
>
> $$
> (\rho I-\mathcal{L})V=r
> $$
>
> 를 얻는 방식이 표준적이다. \(r\)가 매끄럽고 \(\mathcal{M}\)이 compact이면 elliptic regularity로 \(V\in C^2\) (더 나아가 \(C^\infty\))가 따라와 “고전해”로도 정당화된다.

**(3) 유일성(“완전 정합”을 닫는 부분).**
만약 \(u\)가 \(\rho u-\nu\Delta_g u=0\)을 만족하는 유계 \(C^2\) 함수라 하자.
\(\mathcal{M}\)이 경계가 없고 compact이면 적분 부분적분(그린 정리)로

$$
\int_{\mathcal{M}} u\,\Delta_g u\,d\mathrm{vol}_g
=
-\int_{\mathcal{M}}\|\nabla_g u\|_g^2\,d\mathrm{vol}_g
$$

이므로, 방정식에 \(u\)를 곱해 적분하면

$$
0
=
\rho\int_{\mathcal{M}} u^2\,d\mathrm{vol}_g
+
\nu\int_{\mathcal{M}}\|\nabla_g u\|_g^2\,d\mathrm{vol}_g.
$$

\(\rho,\nu>0\)이므로 두 적분항은 각각 \(0\)이어야 하고, 따라서 \(u\equiv 0\)이다.
유계 해의 차이는 동차해이므로, 해는 유일하다.

##### 구현/부호(중요)
수치 구현에서는 연산자를 보통 **양의 반정 definite(PSD)** 형태로 잡습니다.
예를 들어 코드의 `laplace_beltrami_matrix`는 그래프 라플라시안 \(L=D-W\) (PSD) 형태이며, 이때 연속체의 \(\Delta_g\)와는 부호가 반대인 근사(대략 \(\Delta_g \approx -L\))가 되는 것이 일반적입니다.
따라서 \(\rho V-\nu\Delta_g V=r\)를 그대로 이산화하면 흔히

$$
(\rho I + \nu L)\,v = r
$$

같은 **선형 시스템**으로 떨어집니다(부호 규약을 한 번만 고정하면 혼동이 사라집니다).

##### (이 레포 적용) “승률로 치환”은 \(r(x)\) 선택의 문제다
LB‑IGD에서는 설계 \(x\)의 품질을 승률/승률행렬에서 계산한 스칼라로 둡니다.
따라서 정리 4.2의 \(r(x)\)를 아래처럼 두면 “Bellman을 LBO로 대체”가 정확히 **문제 정의 차원에서** 완료됩니다.

- \(r(x)=P(x)\): \(V\)는 “확산(Δ\_g) 하에서의 할인 누적 승률”이므로, **승률 지형을 LBO로 스무딩한 값**이 된다.
- \(r(x)=-J(x)\): 여기서 \(J(x)\)는 `designer.py`가 쓰는 승률 기반 손실(밸런스/퇴화 방지/분포 정합 등)이다. 이때 \(V\)는 “스무딩된 목적(보상)”이 되어, 노이즈가 큰 \(J\)를 직접 쓰는 것보다 안정적인 스칼라 필드를 얻는다.

이산화는 위의 \((\rho I+\nu L)v=r\) 선형시스템으로 구현 가능하며, \(L\)은 (i) 설계 샘플들로 만든 그래프 라플라시안 또는 (ii) `laplace_beltrami_matrix`로 만든 커널 기반 라플라시안으로 두면 된다.

#### 4.3 (확장) “뉴런 토션/전기 방향(비가역성)”을 넣으면: \(\Delta_g\)는 유지되고 1차항(드리프트)이 추가된다
정리 4.2는 생성자가 \(\mathcal{L}=\nu\Delta_g\)인 **가역(reversible)** 확산을 가정했다.
“토션/전기 방향”을 수학적으로 넣는 가장 단순한 방법은, 생성자에 **방향성(순환/비가역)을 만드는 1차항**을 추가하는 것이다.

##### 정리 4.3 (토션 드리프트 포함 Bellman ↔ 선형 PDE)
\((\mathcal{M},g)\) 위의 마르코프 과정 \((X_t)\)의 생성자가

$$
(\mathcal{L}f)(x)=\nu\,\Delta_g f(x) + b(x)\cdot\nabla_g f(x)
$$

로 주어진다고 하자. 여기서 \(b\)는 다양체 위의 벡터장이다(해석적으로는 충분히 매끄럽다고 가정).
보상 \(r:\mathcal{M}\to\mathbb{R}\), 할인율 \(\rho>0\)에 대해

$$
V(x):=\mathbb{E}_x\Big[\int_0^\infty e^{-\rho t}r(X_t)\,dt\Big]
$$

로 정의하면, \(V\)는 다음을 만족한다.

$$
\boxed{\;\rho V(x)-\nu\,\Delta_g V(x)-b(x)\cdot\nabla_g V(x)=r(x)\;}
$$

또한 \(\rho>0\)이고 \(\mathcal{M}\)이 compact이면(표준 조건 하에서) 유계 해는 유일하다.

##### 증명
정리 4.2의 “엄밀한 버전 메모”에서 쓴 반군/레졸벤트 논리가 그대로 적용된다.
즉 \(P_t f(x)=\mathbb{E}_x[f(X_t)]\)를 두고

$$
V(x)=\int_0^\infty e^{-\rho t}\,(P_t r)(x)\,dt
$$

로 정의하면(=레졸벤트 \(R_\rho r\)), \(\partial_t P_t r=\mathcal{L}P_t r\)와 적분 부분적분으로
\((\rho I-\mathcal{L})V=r\)를 얻는다. \(\mathcal{L}=\nu\Delta_g+b\cdot\nabla_g\)를 대입하면 결론이 나온다.
유일성은 \(u\)가 \((\rho I-\mathcal{L})u=0\)이면 \(P_t u=e^{\rho t}u\)가 되어 유계성과 모순임을 이용하는 표준 최대원리/반군 논증으로 닫을 수 있다(혹은 \(b\)가 발산 0 등 추가 조건 하에서 에너지법으로도 가능).

##### (그래프 이산화) 토션은 “비대칭 성분”으로 들어가며, 선형시스템은 여전히 풀린다
그래프에서 “전기 방향/토션”을 가장 단순하게 넣는 방법은 가중치 행렬 \(W\)의 비대칭 성분을 분리하는 것이다:

$$
S=\frac{W+W^\top}{2}\quad(\text{대칭}),\qquad
A=\frac{W-W^\top}{2}\quad(\text{반대칭}).
$$

대칭 성분으로부터 PSD 라플라시안 \(L_{\text{sym}}:=D_S-S\)를 만들고, 반대칭 성분은 **드리프트(방향성) 연산자**로 본다.
그러면 “정리 4.3”의 이산 대응은 보통

$$
(\rho I+\nu L_{\text{sym}}+\kappa A)\,v=r
$$

꼴이 된다(\(\kappa\)는 토션 스케일).

**정리(가역+토션 혼합 행렬의 가역성).**
\(M:=H+K\)에서 \(H\)가 대칭 양의정부호(SPD), \(K\)가 반대칭(skew-symmetric)이면 \(M\)은 가역이다.

**증명.** \(Mx=0\)이면 \(x^\top Hx=-x^\top Kx\). 그런데 \(x^\top Kx=0\)이므로 \(x^\top Hx=0\).
SPD이므로 \(x=0\). 따라서 영공간이 자명하여 \(M\)은 가역이다. □

즉 \(\rho>0\)이면(그리고 \(L_{\text{sym}}\)이 PSD이면) 위 선형시스템은 **토션을 넣어도** 잘 정의된다.

##### (푸리에 전기 신호) 시간 의존 보상/토션은 모드별 레졸벤트로 분해된다
전기 신호를 \(t\)에 대한 푸리에 급수로 두자(예: \(r(t,x)=\sum_k r_k(x)e^{i\omega_k t}\)).
무한 지평 할인 누적값을

$$
V(t,x):=\mathbb{E}_{t,x}\Big[\int_t^\infty e^{-\rho (s-t)}\,r(s,X_s)\,ds\Big]
$$

로 정의하면(시간 의존 reward), 표준 DPP로

$$
(\partial_t+\rho-\mathcal{L})V(t,x)=r(t,x)
$$

를 얻는다. 선형이므로 푸리에 모드별로

$$
(\rho+i\omega_k-\mathcal{L})V_k(x)=r_k(x)
$$

가 되며, 각 모드의 응답을 풀어 합하면 \(V(t,x)\)가 된다.
즉 “푸리에 전기 신호”는 수학적으로 **\(\rho\to\rho+i\omega_k\)** 로 바뀐 레졸벤트를 모드별로 푸는 문제로 떨어진다.

### 5. 설계공간에서의 LBO: “진짜 \(\Delta_g\)” 대신 “차분 기반 곡률 측정”
이 프로젝트의 설계공간 \(x\)는 이산(패턴 ID, 유닛 수)과 연속(스탯) 변수가 섞여 있고, 코드에서 클램프/정수화를 거칩니다.
따라서 엄밀한 의미의 매끄러운 다양체 \((\mathcal{X},g)\)를 직접 두고 \(\Delta_g\)를 계산하기보다, 다음 전략을 씁니다.

> **승률 지형 \(P(x)\)** 가 “너무 뾰족한 방향(곡률 큰 방향)”을 탐지해, 그 방향 업데이트를 약화한다.

이를 위해 2차 중앙 차분을 사용합니다.

#### 5.1 1차원 중앙 차분 증명 스케치
테일러 전개로

$$
f(x\pm h)=f(x)\pm h f'(x)+\frac{h^2}{2}f''(x)+O(h^3).
$$

따라서

$$
f(x+h)+f(x-h)-2f(x)=h^2 f''(x)+O(h^4)
$$

이고,

$$
f''(x)\approx \frac{f(x+h)+f(x-h)-2f(x)}{h^2}.
$$

즉 “양쪽을 더하면 1차항이 상쇄되고 2차항이 남는다”는 구조가 핵심입니다.

#### 5.2 다차원에서의 방향 2차 미분(헤시안) 연결
\(f:\mathbb{R}^d\to\mathbb{R}\)가 충분히 매끄럽고, 방향 벡터 \(e\in\mathbb{R}^d\)에 대해 테일러 전개를 쓰면

$$
f(x+\sigma e)
=
f(x)+\sigma\nabla f(x)^\top e+\frac{\sigma^2}{2}e^\top H(x)e+O(\sigma^3)
$$

이며 \(H(x)=\nabla^2 f(x)\)는 헤시안입니다. 같은 방식으로 \(f(x-\sigma e)\)를 더하면

$$
f(x+\sigma e)+f(x-\sigma e)-2f(x)
=
\sigma^2 e^\top H(x)e + O(\sigma^4).
$$

따라서

$$
\frac{f(x+\sigma e)+f(x-\sigma e)-2f(x)}{\sigma^2\|e\|^2}
\approx
\frac{e^\top H(x)e}{\|e\|^2}
$$

는 “그 방향의 곡률”을 근사합니다.

#### 5.3 “라플라시안”과의 연결(등방성 방향 평균)
만약 \(e\)가 등방성(isotropic) 분포에서 뽑히면(예: 표준정규), 위의 방향 곡률의 기대값은 \(\mathrm{Tr}(H)\)와 연결됩니다.
정확한 상수는 분포/정규화에 따라 달라지지만, 요지는

$$
\mathbb{E}\Big[\frac{e^\top H e}{\|e\|^2}\Big]
\propto
\mathrm{Tr}(H)
=
\Delta f
$$

로 “방향 곡률을 평균내면 라플라시안”이 된다는 점입니다.
프로젝트에서는 이 평균을 엄밀히 쓰기보다, 샘플별 \(|\Delta P|\)를 **안정성 지표**로 사용합니다.

---

### 6. 이 프로젝트에서의 사용: ES 업데이트를 “곡률로 가중”하기
이 프로젝트의 핵심은 다음입니다.

- 외부 목적은 승률/퇴화 방지/메타(거리 분포 등)로 구성된 \(J(x)\)이고(`docs/blackbox.md`)
- ES는 \(x\pm\sigma e\)를 평가해 업데이트 방향을 얻는데
- **승률 지형 \(P(x)\)** 의 곡률이 큰 샘플(불안정한 방향)은 업데이트에 덜 반영합니다.

#### 6.1 설계공간 라플라시안(2차 차분) 근사
2팩션 승률 \(P(x)\)에 대해 코드에서는 다음을 사용합니다.

$$
\Delta P(x)
\approx
\frac{P(x+\sigma e)+P(x-\sigma e)-2P(x)}{\sigma^2\|e\|^2}.
$$

다팩션의 경우 \(P\)가 행렬 \(W_{f,g}\)이므로, 각 쌍에 대해 위를 계산해 평균(또는 평균 절대값)을 사용합니다.

#### 6.2 공통 난수(CRN)가 중요한 이유
승률 추정 \(P(x;\omega)\)는 self-play/RL 노이즈가 큽니다.
따라서 같은 방향 \(e\)에 대해 center/pos/neg를 **동일 시드(=CRN)** 로 평가하여,
\(P(x+\sigma e)-P(x-\sigma e)\) 같은 “차이”의 분산을 낮춥니다.

#### 6.3 가중치로 쓰는 방식(현재 구현)
코드(`src/core/designer.py`)는

$$
w(e)=\frac{1}{1+\lambda\,|\Delta P(x)|}
$$

형태의 단순 가중치를 두고(\(\lambda\)는 현재 1.0), ES 그라디언트 누적에 곱합니다.
직관적으로 \(|\Delta P|\)가 크면 “그 방향은 설계가 예민하다”는 뜻이므로, 업데이트를 약하게 만듭니다.

---

### 7. (부록) 그래프 라플라시안과 디리클레 에너지(토이 실험 모듈)
`src/core/lbo.py`에는 그래프 라플라시안 \(L=D-W\)와 디리클레 에너지가 구현되어 있습니다.
여기서 \(W\)는 대칭 가중치 행렬, \(D\)는 차수 행렬입니다(\(D_{ii}=\sum_j W_{ij}\)).

#### 7.1 디리클레 에너지의 비음수성(증명 스케치)

$$
u^\top (D-W)u
=
\sum_i d_i u_i^2 - \sum_{i,j} w_{ij}u_i u_j.
$$

대칭 \(w_{ij}=w_{ji}\)를 가정하면,

$$
\frac{1}{2}\sum_{i,j} w_{ij}(u_i-u_j)^2
=
\frac{1}{2}\sum_{i,j} w_{ij}(u_i^2+u_j^2-2u_i u_j)
=
\sum_i d_i u_i^2 - \sum_{i,j} w_{ij}u_i u_j
$$

이므로

$$
\boxed{u^\top L u = \frac{1}{2}\sum_{i,j} w_{ij}(u_i-u_j)^2 \ge 0}
$$

가 됩니다(가중치 \(w_{ij}\ge 0\)). 이 성질은 테스트(`tests/test.py`)에서 간단히 확인합니다.

#### 7.2 그래프 확산 \(-Lu\)는 디리클레 에너지의 경사하강(상위호환 정식화)
위의 디리클레 에너지를

$$
E(u):=\frac{1}{2}u^\top L u
$$

로 두면, 연속 시간에서의 그래프 확산(스무딩)을

$$
\frac{du}{dt} = -L u
$$

로 정의할 수 있습니다. 이때 에너지 감소는 즉시 확인됩니다:

$$
\frac{dE}{dt}
=
u^\top L \frac{du}{dt}
=
-u^\top L^2 u
\le 0.
$$

즉, \(-Lu\) 확산은 \(E(u)\)를 단조 감소시키는 **안정화(스무딩) 동역학**이며, 7.1절의 비음수성은 그 기반이 됩니다.

실제 구현에서는 보통 이산 시간(Euler) 스텝

$$
u^{k+1}=u^k - h\,L u^k
$$

를 사용합니다. 이때 \(h\)는 안정성을 위해 충분히 작아야 하며, 예를 들어 \(L\)의 최대 고유값을 \(\lambda_{\max}(L)\)라 하면 표준적인 충분조건은 \(0<h<2/\lambda_{\max}(L)\)입니다.

---

### 8. 실무 튜닝 포인트(설계/구현 관점)
- \(\sigma\)가 너무 작으면: 이산/클램프 때문에 \(x+\sigma e\)와 \(x-\sigma e\)가 같은 설계로 투영되어 차분이 0이 되기 쉽습니다.
- \(\sigma\)가 너무 크면: “국소 곡률”이 아니라 전역 비선형 변화까지 섞여 곡률 해석이 어려워집니다.
- CRN은 필수에 가깝습니다: self-play 노이즈가 크기 때문에, 같은 seed로 center/pos/neg를 묶는 것이 분산 감소에 큰 도움이 됩니다.
- 다팩션에서는 \(\Delta P\)를 행렬로 두고 요약해야 합니다(현재 구현은 쌍대결별 \(\Delta W_{i,j}\) 절대값 평균).

---

### 9. (연구 메모) “뉴런 토션/전기 방향 분해”를 이 레포에 붙일 수 있는가?
이 레포의 “LB(Laplace–Beltrami)”는 **설계공간에서의 곡률/확산(라플라시안)** 관점입니다.  
질문에서 말하는 “뉴런 토션/전기 방향 분해”는 이 레포에 **그대로 구현되어 있지는 않으며**, 먼저 “토션”을 무엇으로 정의할지부터 고정해야 합니다.

생물학 관점에서 “뉴런이 어느 뉴런하고 붙는가?”는 보통 두 층이 섞여 있습니다.

- **발달/배선(구조 형성)**: 화학적 guidance(분자 신호) + 활동 의존 정련(refinement)
- **가소성(연결 강도 변화)**: 전기적 활동(스파이크) 패턴에 따라 시냅스 강도가 변함(STDP 등)

이 레포에 “전기 신호 기반”으로 자연스럽게 붙일 수 있는 것은 두 번째(가소성)이고, 첫 번째(발달/분자 신호)는 현재 코드 범위 밖입니다.

아래는 현재 코드가 제공하는 최소 구성요소와, “연결이 강화/형성될지”를 **생물학에서 널리 쓰이는 활동-의존 가소성 수식**으로 정리한 메모입니다.

#### 9.1 이 레포에 이미 있는 “뇌 비유” 블록(토이 실험)
`src/core/experiment.py`는 아래 3가지를 묶어 “노이즈 관측 → 확산(스무딩) → 사건 시 연결 업데이트”를 보여줍니다.

- **활동 \(u\)**: 노드(뉴런)별 활동값 벡터 \(u\in\mathbb{R}^n\) (`experiment.py`의 `u`)
- **확산/스무딩**: 그래프 라플라시안 \(L=D-W\)로 \(-Lu\) 확산 (`src/core/lbo.py:laplacian_mul`)
- **연결(시냅스) 학습**: 활동 기반으로 가중치 행렬 \(W\)를 갱신 (`src/core/metric.py:update`)
- **사건 게이트(도파민 비유)**: 오차(놀람)가 급등할 때만 학습을 허용 (`src/core/dopamine.py:DopamineGate`)

즉, “뉴런이 누구랑 붙을지”를 결정하는 후보식을 붙일 위치는 현재로선 `src/core/metric.py`(연결 업데이트 규칙) 쪽이 가장 자연스럽습니다.

#### 9.2 (이미 구현됨) 활동 유사도 기반 연결 판정(헤비안 유사)
현재 `metric.py`는 활동 \(u\)만으로 “가까운 애들끼리” 연결을 강화하는 **정적 유사도 규칙**을 씁니다.

1) 활동 유사도(affinity) 정의:

$$
a_{ij}(t) = \exp\!\Big(-\frac{(u_i(t)-u_j(t))^2}{2\tau^2}\Big)
$$

2) sparsify + 업데이트(대략적 형태):

$$
W \leftarrow (1-\text{decay})W,\qquad
W \leftarrow (1-\text{lr})W + \text{lr}\cdot \mathrm{TopK}(A),\qquad
W \leftarrow \mathrm{clip}(W; 0, w_{\max})
$$

이 규칙은 “전기 신호(활동)가 비슷한 뉴런끼리 붙는다”라는 매우 단순한 가정에 해당합니다.

> 생물학적 해석 메모(정확한 동치가 아님):  
> 현재 구현은 \(W\)를 **대칭**으로 만들고(`metric._symmetrize`), `lbo.py`의 확산(-L u)을 함께 씁니다.  
> 이 조합은 “방향성 화학 시냅스”라기보다 “전기적 결합(갭정션) 같은 양방향 커플링 + 활동 유사도 기반 인접성”에 가깝게 해석하는 편이 자연스럽습니다.

#### 9.3 (후보) 생물학 기반 연결 판정의 대표식: STDP / 3-factor(도파민) 가소성
“전기 신호로 연결이 강해진다/약해진다”에서 가장 널리 알려진 최소 수식은 **STDP(스파이크 타이밍 의존 가소성)** 입니다.  
핵심은 “**pre가 먼저 쏘고 post가 뒤따라 쏠 때 강화**” 같은 시간 비대칭(인과성)입니다.

가장 단순한 형태(스파이크 시각 차 \(\Delta t=t_{\text{post}}-t_{\text{pre}}\))는:

$$
\Delta w_{ij}=
\begin{cases}
A_+\exp(-\Delta t/\tau_+) & \Delta t>0 \\
-A_-\exp(\Delta t/\tau_-) & \Delta t<0
\end{cases}
$$

여기서 \(w_{ij}\)는 “\(i\to j\)” 방향 시냅스(방향성 포함)이며, 이 규칙을 쓰면 “어느 뉴런이 어느 뉴런과 연결이 강화되는지”가 **전기 신호의 시간 상관(인과성)** 으로 판정됩니다.

다만 실무/시뮬레이션에서는 위의 “스파이크 시각 차” 공식을 그대로 쓰기보다, **trace(지연 누적) 형태**로 구현하는 경우가 많습니다.

##### 9.3.1 (구현 친화) trace 기반 STDP(이산 시간)
시간 스텝 \(t\)에서 스파이크 지시변수 \(s_i[t]\in\{0,1\}\)를 두고, pre/post trace를

$$
p_i[t+1]=\lambda_+ p_i[t] + s_i[t],\qquad
q_i[t+1]=\lambda_- q_i[t] + s_i[t]
$$

로 둡니다(\(\lambda_\pm=\exp(-\Delta t/\tau_\pm)\)). 그러면 방향 시냅스 \(i\to j\)의 업데이트를 예를 들어

$$
\Delta w_{ij}[t]
=
\eta\Big(A_+\,p_i[t]\;s_j[t]\;-\;A_-\,s_i[t]\;q_j[t]\Big)
$$

처럼 쓸 수 있습니다. (post가 쏠 때 pre-trace를 보고 강화, pre가 쏠 때 post-trace를 보고 약화)

##### 9.3.2 (보상/사건 결합) 3-factor 학습: 도파민(전역 신호) × eligibility
생물학에서는 STDP 같은 2-factor(전/후) 규칙이 **보상/사건(도파민성 신호)** 에 의해 조절되는 형태(3-factor)가 자주 사용됩니다.
가장 단순한 구조는 “eligibility trace \(e_{ij}\)”를 누적해두고, 도파민성 신호 \(\delta[t]\)가 왔을 때만 가중치를 바꾸는 형태입니다.

$$
e_{ij}[t+1]=\lambda_e e_{ij}[t] + \Big(A_+\,p_i[t]s_j[t]-A_-\,s_i[t]q_j[t]\Big),\qquad
\Delta w_{ij}[t]=\eta\,\delta[t]\;e_{ij}[t].
$$

> 코드 매핑 메모: `src/core/dopamine.py:DopamineGate`는 현재 “오차 급등”을 사건 신호로 쓰는 토이 게이트입니다.  
> 이를 \(\delta[t]\in\{0,1\}\) (gate open이면 1)로 해석하면 “3-factor의 최소 형태”로 연결할 수 있습니다.

##### 9.3.3 “붙을지”를 명시하려면(구조적 가소성 최소 규칙)
STDP는 원래 “연결 강도”를 바꾸는 규칙입니다. “붙는다/끊긴다”까지 포함하려면 보통 아래 같은 최소 규칙을 추가합니다.

- **임계치 기반 생성/제거**: \(w_{ij}\ge w_{\text{on}}\)이면 연결로 간주, \(w_{ij}\le w_{\text{off}}\)이면 제거(히스테리시스)
- **예산/희소성**: 뉴런당 상위 \(k\)개만 유지(top-k pruning) 같은 구조 제약
- **폭주 방지(homeostasis)**: synaptic scaling(행/열 정규화), 가중치 클램프, 평균 발화율 타깃 등

#### 9.4 “토션/전기 방향”을 생물학적으로 해석하면(현실적인 대응)
생물학에서 “전기 신호로 연결이 결정된다”의 핵심은 보통 **방향성(축삭→수상돌기)** 과 **시간차(\(\Delta t\))** 입니다.  
즉, “방향”은 벡터장 분해라기보다 **시냅스의 방향성 \(i\to j\)** 과 **전-후 스파이크 순서**로 나타나는 경우가 많습니다(STDP가 그 자체로 방향 판정식).

반면 “토션”은 연결 가소성에서 표준적으로 쓰이는 단일 수식이 따로 있는 개념은 아닙니다.  
다만 분석 용도로 “방향성/순환 회로가 얼마나 강한가”를 보려면, 방향 가중치 행렬 \(W\)를

$$
S=\frac{W+W^\top}{2},\qquad
A=\frac{W-W^\top}{2}
$$

로 나눠 \(A\)(비대칭 성분)를 “순환/피드백 성분”의 최소 지표로 보는 식의 **수학적 대체 정의**는 가능합니다(이는 생물학 법칙이 아니라 분석 도구).

#### 9.5 이 레포에 “붙일 수 있나?”에 대한 결론(현실적인 범위)
- **가능(토이 실험 레벨)**: `src/core/experiment.py`의 “노드 활동 \(u\) + 연결 \(W\) 학습”에  
  STDP/3-factor 같은 규칙을 넣어 “전기 신호(스파이크/사건)로 연결이 바뀐다”를 모델링하는 것은 구조적으로 가능합니다.
- **어려움(현 상태의 메인 IGD 루프)**: `main.py` → `DesignOptimizer`는 게임 설계 최적화가 목적이라,  
  뉴런 연결 규칙을 “그대로” 붙이는 건 문제 설정 자체가 달라집니다.
- **토션은(있다면) 분석 정의에 가깝다**: 생물학 법칙으로서의 “토션 식”을 찾기보다는,  
  (i) 방향 시냅스(STDP)로 먼저 모델을 세우고, (ii) 그 결과 네트워크의 순환성을 분석 지표로 정의하는 접근이 KISS합니다.
