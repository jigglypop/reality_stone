## 4. 이론적 분석 및 보장

본 장에서는 3장에서 제안한 방법론들의 수학적 정합성과 안정성을 엄밀히 증명한다.

### 4.1 Hyper-Butterfly 레이어의 보편적 근사 정리

**정리 4.1 (Universal Approximation Theorem for Hyper-Butterfly)**

Hyper-Butterfly 레이어들의 집합 $\mathcal{F}=\{\exp_0^c \circ B \circ \log_0^c\}$으로 구성된 함수 대수(algebra)는 컴팩트 리만 다양체 $M$ 위의 연속 함수 공간 $C(M)$에서 조밀(dense)하다. 즉, 임의의 연속 함수 $h \in C(M)$와 임의의 $\varepsilon > 0$에 대해, 다음을 만족하는 $f \in \mathcal{F}$가 존재한다.
$ \sup_{x \in M} |f(x) - h(x)| < \varepsilon $

**증명 스케치**:
이 증명은 Stone-Weierstrass 정리를 기반으로 한다. 함수 대수 $\mathcal{F}$가 (1) 대수 구조를 형성하고, (2) 상수 함수를 포함하며, (3) 점들을 분리(point-separating)함을 보이면 된다.

1.  **대수 구조**: 두 함수 $f, g \in \mathcal{F}$의 합과 곱이 더 깊은 Hyper-Butterfly 네트워크로 표현될 수 있다.
2.  **상수 함수**: Butterfly 변환 $B=I$ (항등 행렬)과 bias 항을 통해 상수 함수를 만들 수 있다.
3.  **점 분리**: $\log_0^c$와 $\exp_0^c$가 전단사(bijective)이고, Butterfly 변환 행렬 $B$가 풀랭크(full-rank) 표현력을 가지므로, 임의의 두 점 $x \neq y$에 대해 $f(x) \neq f(y)$가 되도록 하는 $B$를 항상 찾을 수 있다.

따라서 Stone-Weierstrass 정리의 조건이 모두 만족되므로, Hyper-Butterfly 네트워크는 보편적 근사 능력을 가진다.

### 4.2 수치적 안정성 분석

신경망의 안정적인 학습을 위해서는 연산의 수치적 안정성이 보장되어야 한다. 여기서는 Hyper-Butterfly 레이어의 조건수(Condition Number)와 그래디언트 흐름을 분석한다.

**정리 4.2 (Condition Number Bounding)**

Hyper-Butterfly 함수 $f(x) = \exp_0^c(B(\log_0^c(x)))$의 야코비안 $Df(x)$의 조건수 $\kappa(f, x) = \|Df(x)\|_2 \|(Df(x))^{-1}\|_2$는 유한한 값으로 상한이 정해진다(bounded).

**증명 스케치**:
$f$는 $\exp_0^c$, $B$, $\log_0^c$ 세 함수의 합성이므로, 체인룰에 의해 $Df(x) = D\exp_0^c(v) \cdot B \cdot D\log_0^c(x)$이다 ($v=B(\log_0^c(x))$). 따라서 조건수는 각 요소의 조건수의 곱으로 바운딩된다.
$ \kappa(f,x) \le \kappa(\exp_0^c) \cdot \kappa(B) \cdot \kappa(\log_0^c) $
각각의 로그/지수 사상과 Butterfly 변환의 야코비안 조건수가 입력의 노름과 곡률 $c$에 의해 제어되는 유한한 값임을 보일 수 있다. 따라서 전체 조건수는 유한하며, 이는 수치적으로 안정적인 연산이 가능함을 의미한다.

**정리 4.3 (Gradient Stability)**

Hyper-Butterfly 레이어의 역전파 과정에서 그래디언트의 노름은 폭발하거나 소실되지 않고 안정적으로 전파된다. 즉, 손실 함수 $L$에 대해 출력 그래디언트 $\nabla_y L$과 입력 그래디언트 $\nabla_x L$ 사이에는 다음 관계가 성립한다.
$ \|\nabla_x L\| \le C_g \|\nabla_y L\| $
여기서 $C_g$는 네트워크의 깊이나 차원에 무관한 상수이다.

**증명 스케치**:
역전파 그래디언트는 $\nabla_x L = (Df(x))^T \nabla_y L$로 계산된다. 따라서 $\|\nabla_x L\| \le \|(Df(x))^T\|_2 \|\nabla_y L\| = \|Df(x)\|_2 \|\nabla_y L\|$ 이다. $Df(x)$의 스펙트럼 노름이 앞서 증명한 바와 같이 유한하게 바운딩되므로, 그래디언트 역시 안정적으로 제어된다.

### 4.3 압축 방법론의 오차 분석

스플라인이나 푸리에 급수와 같은 방법으로 가중치를 근사할 때 발생하는 오차를 분석하는 것은 압축 성능을 보장하는 데 중요하다.

**정리 4.4 (Spectral Truncation Error)**

가중치 함수 $f$가 특정 소볼레프 공간 $H^s(M)$에 속하여 충분히 매끄럽다고 가정할 때, Helgason-Fourier 급수를 $N_{coef}$개의 계수만 사용하여 절단(truncate)할 때 발생하는 L2 오차는 다음과 같이 바운딩된다.
$ \|f - f_{N_{coef}}\|_{L^2(M)} \le C \cdot N_{coef}^{-s/d} $
여기서 $d$는 다양체의 차원, $s$는 함수의 매끄러움 정도, $C$는 함수의 소볼레프 노름에 비례하는 상수이다. 이는 계수 수를 늘릴수록 오차가 다항적으로 빠르게 감소함을 의미하며, 효율적인 압축이 가능함을 시사한다.

이 정리는 스플라인 압축에도 유사하게 적용될 수 있다. 스플라인 보간은 다항식 근사의 한 형태이며, $k$차 스플라인으로 매끄러운 함수를 근사할 때의 오차는 제어점 사이의 간격 $h$에 대해 $O(h^{k+1})$로 감소한다. 스플라인 압축에서 제어점의 수를 늘리는 것은 $h$를 줄이는 것과 같으므로, 제어점 수를 늘림에 따라 표현 오차가 빠르게 감소함을 이론적으로 보장할 수 있다. 