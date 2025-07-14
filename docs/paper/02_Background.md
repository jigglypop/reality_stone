## 2. 배경: 리만 기하학과 신경 표상

### 2.1. 리만 기하학의 기초

리만 기하학은 곡률을 가진 공간을 연구하는 수학 분야로, 아인슈타인의 일반 상대성 이론부터 현대 기계학습까지 광범위하게 응용된다. 본 절에서는 RBE를 이해하는 데 필요한 핵심 개념들을 소개한다.

#### 2.1.1. 리만 다양체와 계량 텐서

**정의 2.1 (리만 다양체)**: n차원 리만 다양체 $(\mathcal{M}, g)$는 매끄러운 다양체 $\mathcal{M}$과 각 점 $p \in \mathcal{M}$에서 접공간 $T_p\mathcal{M}$의 내적을 정의하는 계량 텐서 $g_p$의 쌍이다.

계량 텐서는 국소 좌표계에서 대칭 양의 정부호 행렬 $g_{ij}$로 표현되며, 두 접벡터 $u, v \in T_p\mathcal{M}$ 사이의 내적을 다음과 같이 정의한다:

$$\langle u, v \rangle_g = \sum_{i,j} g_{ij} u^i v^j$$

#### 2.1.2. 측지선과 지수 맵

**정의 2.2 (측지선)**: 리만 다양체에서 측지선은 국소적으로 최단 거리를 실현하는 곡선이다. 측지선 방정식은:

$$\frac{d^2 x^k}{dt^2} + \sum_{i,j} \Gamma^k_{ij} \frac{dx^i}{dt} \frac{dx^j}{dt} = 0$$

여기서 $\Gamma^k_{ij}$는 Christoffel 기호이다.

**정의 2.3 (지수 맵과 로그 맵)**:
- 지수 맵 $\exp_p: T_p\mathcal{M} \rightarrow \mathcal{M}$은 접벡터를 측지선을 따라 이동시킨다.
- 로그 맵 $\log_p: \mathcal{M} \rightarrow T_p\mathcal{M}$은 지수 맵의 역함수이다.

### 2.2. 쌍곡 기하학 모델

쌍곡 공간은 일정한 음의 곡률을 가진 리만 다양체로, 계층적 데이터 표현에 특히 유용하다.

#### 2.2.1. 푸앵카레 볼 모델

n차원 푸앵카레 볼 $\mathbb{B}^n_c$는:
- 다양체: $\mathbb{B}^n_c = \{x \in \mathbb{R}^n : c\|x\|^2 < 1\}$
- 계량 텐서: $g^{\mathbb{B}}_x = \lambda_c^2(x) g^E$, 여기서 $\lambda_c(x) = \frac{2}{1-c\|x\|^2}$

거리 함수:
$$d_{\mathbb{B}}(x, y) = \frac{1}{\sqrt{c}} \text{arcosh}\left(1 + 2c\frac{\|x-y\|^2}{(1-c\|x\|^2)(1-c\|y\|^2)}\right)$$

#### 2.2.2. 로렌츠 모델 (쌍곡면)

n차원 로렌츠 모델 $\mathbb{H}^n_c$는:
- 다양체: $\mathbb{H}^n_c = \{x \in \mathbb{R}^{n+1} : \langle x, x \rangle_\mathcal{L} = -1/c, x_0 > 0\}$
- 로렌츠 내적: $\langle x, y \rangle_\mathcal{L} = -x_0 y_0 + \sum_{i=1}^n x_i y_i$

### 2.3. 리만 기하학적 뇌 표상 프레임워크

#### 2.3.1. 신경 표상의 기하학적 해석

최근 신경과학 연구는 뇌가 정보를 단순한 유클리드 공간이 아닌 곡률을 가진 리만 다양체로 표현할 가능성을 시사한다. 이 관점에서:

- **신경 활성화 패턴**은 다양체 위의 점으로 표현된다.
- **정보 처리**는 다양체 위에서의 측지선 이동으로 해석된다.
- **학습**은 다양체의 기하학적 구조(계량 텐서)를 변형시키는 과정이다.
- **기억**은 다양체 위의 특정 영역과 이들 간의 연결로 표현된다.

#### 2.3.2. 신경 리만 다양체의 수학적 모델

신경 리만 다양체 $\mathcal{M}_{\text{neural}}$은:

$$\mathcal{M}_{\text{neural}} = (\mathbb{R}^n, g_{ij}(x))$$

계량 텐서는 시냅스 가중치와 관련되어:

$$g_{ij}(x) = \delta_{ij} + \sum_{k} W_{ik} W_{jk} f'(W_k \cdot x + b_k)^2$$

여기서 $f'$은 뉴런의 활성화 함수의 도함수이다.

#### 2.3.3. 해마 인덱싱의 기하학적 해석

해마의 인덱싱 기능은 고차원 피질 표상 다양체 $\mathcal{M}_C$에서 저차원 해마 다양체 $\mathcal{M}_H$로의 측지 투영으로 모델링할 수 있다:

$$\pi: \mathcal{M}_C \rightarrow \mathcal{M}_H$$

이 투영은 다음 목적 함수를 최소화한다:

$$L(\pi) = \mathbb{E}_{x \in \mathcal{M}_C} [d_{\mathcal{M}_C}(x, \phi(\pi(x)))^2] + \lambda \cdot \text{Complexity}(\pi)$$

여기서 $\phi: \mathcal{M}_H \rightarrow \mathcal{M}_C$는 재구성 함수이다.

### 2.4. 뫼비우스 변환과 자이로벡터 공간

쌍곡 공간에서의 연산을 위해 뫼비우스 변환과 자이로벡터 공간 구조를 활용한다.

#### 2.4.1. 뫼비우스 덧셈

푸앵카레 볼에서 두 점 $x, y \in \mathbb{B}^n_c$의 뫼비우스 덧셈:

$$x \oplus_c y = \frac{(1 + 2c\langle x, y \rangle + c\|y\|^2)x + (1 - c\|x\|^2)y}{1 + 2c\langle x, y \rangle + c^2\|x\|^2\|y\|^2}$$

#### 2.4.2. 자이로벡터 공간

자이로벡터 공간 $(\mathbb{B}^n_c, \oplus_c, \otimes)$은:
- 자이로 덧셈: $\oplus_c$
- 스칼라 곱: $r \otimes x = \tanh(r \cdot \text{arctanh}(\sqrt{c}\|x\|)) \frac{x}{\sqrt{c}\|x\|}$

이러한 연산들은 쌍곡 신경망 레이어 구현의 기초가 된다.

### 2.5. 정보 기하학과 Fisher 계량

정보 기하학은 확률 분포의 공간을 리만 다양체로 다루며, Fisher 정보 행렬을 계량 텐서로 사용한다.

**정의 2.4 (Fisher 정보 행렬)**: 매개변수 $\theta$를 가진 확률 분포 $p(x|\theta)$에 대해:

$$g_{ij}^{\text{Fisher}}(\theta) = \mathbb{E}_{p(x|\theta)}\left[\frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j}\right]$$

이는 신경망의 파라미터 공간에 자연스러운 기하학적 구조를 부여하며, RBE의 이론적 기초가 된다. 