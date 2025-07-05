## 2. 수학적 배경

본 장에서는 `Reality Stone`의 이론적 토대를 이루는 리만 기하학(Riemannian Geometry)과 쌍곡 기하학의 핵심 개념들을 소개한다.

### 2.1 리만 다양체 (Riemannian Manifold)

리만 다양체는 유클리드 공간을 일반화한 개념으로, 국소적으로는 유클리드 공간과 유사하지만 전체적으로는 휘어져 있을 수 있는 공간이다.

-   **매끄러운 다양체(Smooth Manifold)**: 모든 점이 국소적으로 $n$차원 유클리드 공간 $\mathbb{R}^n$과 위상동형(homeomorphic)인 공간이다. 각 점의 근방은 좌표 차트(chart)를 통해 $\mathbb{R}^n$의 열린 집합으로 표현될 수 있다.
-   **접공간(Tangent Space)**: 다양체 $M$ 위의 한 점 $p$에서 가능한 모든 방향으로의 "미소 변위"들이 모여 형성하는 벡터 공간으로, $T_pM$으로 표기한다. $n$차원 다양체의 접공간은 $n$차원 벡터 공간이다.
-   **리만 계량(Riemannian Metric)**: 각 점 $p$의 접공간 $T_pM$에 내적(inner product) $\langle \cdot, \cdot \rangle_p$를 매끄럽게 부여하는 텐서 필드 $g$이다. 이 계량을 통해 접벡터의 길이와 두 접벡터 사이의 각도를 측정할 수 있다. 좌표계에서 계량은 행렬 $g_{ij}(x) = \langle \frac{\partial}{\partial x^i}, \frac{\partial}{\partial x^j} \rangle_p$로 표현된다.
-   **측지선(Geodesic)**: 리만 다양체 위에서 "가장 곧은" 곡선, 즉 가속도가 0인 경로를 의미한다. 이는 두 점 사이의 최단 경로가 되며, 유클리드 공간의 직선에 해당한다.
-   **곡률(Curvature)**: 다양체가 얼마나 휘어져 있는지를 측정하는 척도이다. 리만 곡률 텐서($R^i{}_{j k\ell}$), 리치 곡률($R_{j\ell}$), 스칼라 곡률($R$) 등으로 나타낸다. 곡률이 양수이면 구(sphere)처럼, 음수이면 쌍곡면처럼, 0이면 평면처럼 국소적으로 행동한다.

### 2.2 쌍곡 기하학 모델

음의 상수 곡률을 갖는 쌍곡 공간($\mathbb{H}^n$)은 여러 등거리 동형(isometrically equivalent) 모델로 표현될 수 있다.

1.  **푸앵카레 원반 모델 (Poincaré Ball Model)**
    -   **정의**: $n$차원 열린 단위 원반 $\mathbb{B}^n = \{ \mathbf{x} \in \mathbb{R}^n : \|\mathbf{x}\| < 1 \}$에 리만 계량 $g_x = \left(\frac{2}{1-\|\mathbf{x}\|^2}\right)^2 g_E$를 부여한 모델. ($g_E$는 유클리드 계량)
    -   **특징**: 각도를 보존하는 등각(conformal) 모델로, 시각화에 용이하여 딥러닝에서 널리 사용된다.
    -   **거리**: 두 점 $\mathbf{u}, \mathbf{v} \in \mathbb{B}^n$ 사이의 거리는 다음과 같다.
        $d(\mathbf{u}, \mathbf{v}) = \cosh^{-1} \left( 1 + 2 \frac{\|\mathbf{u} - \mathbf{v}\|^2}{(1 - \|\mathbf{u}\|^2)(1 - \|\mathbf{v}\|^2)} \right)$

2.  **로렌츠 모델 (Lorentz Model) / 쌍곡면 모델**
    -   **정의**: $n+1$차원 민코프스키 공간($\mathbb{R}^{1,n}$) 상의 쌍곡면 $\mathbb{H}^n = \{ \mathbf{x} \in \mathbb{R}^{n+1} : \langle \mathbf{x}, \mathbf{x} \rangle_{\mathcal{L}} = -1, x_0 > 0 \}$으로 표현. ($\langle \mathbf{x}, \mathbf{x} \rangle_{\mathcal{L}} = -x_0^2 + \sum_{i=1}^n x_i^2$)
    -   **특징**: 기하학적 연산, 특히 측지선 계산이 선형적으로 표현되어 수치적으로 안정적인 장점이 있다.

3.  **클라인 모델 (Klein Model)**
    -   **정의**: 푸앵카레 모델처럼 단위 원반으로 표현되지만, 측지선이 유클리드 직선으로 나타나는 투영 모델.
    -   **특징**: 측지선 계산이 간단하나, 등각 모델이 아니므로 각도가 왜곡된다.

### 2.3 뫼비우스 연산과 자이로벡터 공간

쌍곡 공간에서 신경망을 구성하려면 유클리드 벡터 공간의 덧셈, 스칼라 곱셈 등을 대체할 대수적 구조가 필요하다. 푸앵카레 원반 모델에서는 뫼비우스(Möbius) 연산과 자이로벡터 공간(Gyrovector Space) 이론이 그 역할을 한다. (곡률 $c=1$ 가정)

-   **뫼비우스 덧셈**: $ \mathbf{u} \oplus \mathbf{v} = \frac{(1 + 2\langle\mathbf{u}, \mathbf{v}\rangle + \|\mathbf{v}\|^2)\mathbf{u} + (1 - \|\mathbf{u}\|^2)\mathbf{v}}{1 + 2\langle\mathbf{u}, \mathbf{v}\rangle + \|\mathbf{u}\|^2\|\mathbf{v}\|^2} $
-   **뫼비우스 스칼라 곱셈**: $ r \otimes \mathbf{v} = \tanh(r \cdot \tanh^{-1}(\|\mathbf{v}\|)) \frac{\mathbf{v}}{\|\mathbf{v}\|} $
-   **지수/로그 사상**: 한 점 $x$에서의 접공간($T_x\mathbb{B}^n$)과 다양체($\mathbb{B}^n$) 사이를 매핑하는 함수로, 신경망의 선형 변환과 업데이트 규칙에 필수적이다.
    -   **로그 사상 $\log_x(y)$**: 다양체 위의 두 점 $x, y$를 $x$의 접공간에 있는 벡터로 변환한다.
    -   **지수 사상 $\exp_x(v)$**: $x$의 접공간에 있는 벡터 $v$를 다양체 위의 점으로 변환한다.

### 2.4 헬가손-푸리에 변환 (Helgason-Fourier Transform)

헬가손-푸리에 변환(HFT)은 유클리드 공간의 푸리에 변환을 대칭 공간(symmetric space), 특히 쌍곡 공간으로 일반화한 것이다. 이는 쌍곡 공간 위에서 함수를 주파수 성분으로 분해하는 강력한 도구를 제공한다.

$n$차원 쌍곡 공간 $\mathbb{H}^n$ 위의 함수 $f$에 대한 HFT는 다음과 같이 정의된다.

$ \tilde{f}(\lambda, b) = \int_{\mathbb{H}^n} f(x) e^{(-i\lambda+\rho)B(x,b)} dx $

-   $\lambda \in \mathbb{R}$: 스펙트럼(주파수) 파라미터
-   $b \in \partial\mathbb{H}^n$: 쌍곡 공간의 무한 경계 위의 점 (방향)
-   $B(x,b)$: 부제만 함수(Busemann function), $x$에서 방향 $b$까지의 부호 있는 거리
-   $\rho = (n-1)/2$: 차원에 의존하는 상수

HFT는 쌍곡 공간의 라플라스-벨트라미 연산자의 고유함수인 평면파(호로사이클 파동)를 기저로 사용하여 함수를 분해한다. 이는 `Reality Stone`의 Hyper-Butterfly 레이어와 같은 스펙트럼 방법론의 수학적 토대를 이룬다. 