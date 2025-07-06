## 동적 곡률: 데이터에 최적화된 기하학적 학습

### 1. 개요 (Abstract)

전통적인 쌍곡 신경망 모델은 쌍곡 공간의 곡률 $c$를 고정된 하이퍼파라미터로 간주한다. 이는 모델의 표현력을 제한하고, 특정 데이터셋에 대한 최적의 기하학적 구조를 사전에 추정해야 하는 부담을 준다. 본 장에서는 **동적 곡률(Dynamic Curvature)** 이라는 새로운 접근법을 제안한다. 이 방법은 곡률 $c$를 학습 가능한 파라미터(learnable parameter)로 취급하여, 손실 함수를 최소화하는 과정에서 데이터의 내재적 특성에 가장 적합한 곡률을 모델 스스로 찾도록 한다. 이를 통해 모델의 표현력과 최종 성능을 극대화하는 것을 목표로 한다.

### 2. 동기 (Motivation)

데이터의 계층적 구조를 표현하는 데 있어 쌍곡 공간의 효과는 널리 알려져 있다. 그러나 모든 데이터, 그리고 모델의 모든 레이어가 동일한 곡률을 최적으로 요구하지는 않는다.

-   **데이터 의존성**: 어떤 데이터셋은 트리와 유사한 가파른 계층 구조를 가져 높은 곡률($c$가 큼)이 유리할 수 있는 반면, 다른 데이터셋은 유클리드 공간에 가까운 완만한 구조를 가져 낮은 곡률($c$가 0에 가까움)이 더 적합할 수 있다.
-   **레이어별 이질성**: 심층 신경망의 각 레이어는 서로 다른 수준의 추상화를 학습한다. 초기 레이어에서는 원시 데이터의 특징을 포착하기 위해 낮은 곡률이, 후반부의 고차원적인 의미 표현 레이어에서는 높은 곡률이 유리할 수 있다.

고정된 곡률은 이러한 유연성을 제공하지 못한다. 반면, 곡률 $c$를 학습 가능한 파라미터로 설정하면, 각 레이어가 데이터 처리 파이프라인의 역할에 맞춰 최적의 기하학적 공간을 동적으로 형성할 수 있다.

### 3. 수학적 정식화 (Mathematical Formulation)

동적 곡률을 구현하기 위해서는, 곡률 $c$에 대한 쌍곡 기하학의 주요 연산들의 미분 가능성(differentiability)을 확보해야 한다. 즉, 역전파 과정에서 손실 함수의 그래디언트가 곡률 $c$까지 전달될 수 있어야 한다.

#### 3.1 곡률 $c$를 포함하는 핵심 연산

푸앵카레 원반 모델(Poincaré Ball Model)에서 곡률 $c > 0$일 때, 두 점 $x, y \in \mathcal{B}_c^n$ 사이의 핵심 연산은 다음과 같이 $c$에 의존한다.

-   **뫼비우스 덧셈 (Möbius Addition)**:
    $$
    x \oplus_c y = \frac{(1 + 2c\langle x, y \rangle + c\|y\|^2)x + (1 - c\|x\|^2)y}{1 + 2c\langle x, y \rangle + c^2\|x\|^2\|y\|^2}
    $$

-   **쌍곡 거리 (Hyperbolic Distance)**:
    $$
    d_c(x, y) = \frac{2}{\sqrt{c}} \tanh^{-1}(\sqrt{c} \|-x \oplus_c y\|)
    $$

-   **지수 사상 (Exponential Map)**: 원점 $0$에서의 접공간 벡터 $v \in \mathcal{T}_0\mathcal{B}_c^n \cong \mathbb{R}^n$을 다양체로 매핑한다.
    $$
    \exp_0^c(v) = \tanh(\sqrt{c} \|v\|) \frac{v}{\sqrt{c} \|v\|}
    $$

-   **로그 사상 (Logarithmic Map)**: 다양체의 점 $x$를 원점에서의 접공간으로 매핑한다.
    $$
    \log_0^c(x) = \frac{\tanh^{-1}(\sqrt{c}\|x\|)}{\sqrt{c}\|x\|} x
    $$

#### 3.2 곡률 $c$에 대한 그래디언트 계산

손실 함수 $\mathcal{L}$을 최소화하기 위해 경사 하강법을 사용할 때, 곡률 $c$에 대한 그래디언트 $\frac{\partial \mathcal{L}}{\partial c}$를 계산해야 한다. 연쇄 법칙(chain rule)에 의해, 이는 $\mathcal{L}$이 의존하는 모든 중간 변수들을 통해 전파된다.

예를 들어, 두 점 사이의 거리에 기반한 손실 함수 $\mathcal{L} = f(d_c(x,y))$를 생각해보자. 그래디언트는 다음과 같이 계산된다.
$$
\frac{\partial \mathcal{L}}{\partial c} = \frac{\partial \mathcal{L}}{\partial d_c(x,y)} \frac{\partial d_c(x,y)}{\partial c}
$$

여기서 핵심은 $\frac{\partial d_c(x,y)}{\partial c}$를 구하는 것이다. 계산의 편의를 위해 원점과의 거리 $d_c(0, y)$를 미분해보면 다음과 같다.
$$
d_c(0, y) = \frac{2}{\sqrt{c}}\tanh^{-1}(\sqrt{c}\|y\|)
$$
$$
\frac{\partial d_c(0, y)}{\partial c} = \frac{\partial}{\partial c}(2c^{-1/2}) \tanh^{-1}(\sqrt{c}\|y\|) + 2c^{-1/2} \frac{\partial}{\partial c}\tanh^{-1}(\sqrt{c}\|y\|)
$$
$$
= -c^{-3/2} \tanh^{-1}(\sqrt{c}\|y\|) + \frac{2}{\sqrt{c}} \frac{1}{1 - c\|y\|^2} \frac{\|y\|}{2\sqrt{c}}
$$
$$
= - \frac{\tanh^{-1}(\sqrt{c}\|y\|)}{c\sqrt{c}} + \frac{\|y\|}{c(1-c\|y\|^2)}
$$

일반적인 두 점 $x, y$에 대한 거리의 미분은 뫼비우스 덧셈 항으로 인해 훨씬 복잡해지지만, 위와 같이 해석적으로 유도 가능하다. 다행히 PyTorch, TensorFlow와 같은 현대적인 자동 미분 프레임워크는 이러한 복잡한 계산을 자동으로 처리해주므로, 우리는 수식을 정확히 구현하기만 하면 된다.

### 4. 구현 전략 (Implementation Strategy)

1.  **파라미터 초기화**: 곡률 $c$는 양수여야 한다는 제약 조건이 있다 ($c > 0$).
    -   이를 보장하기 위해, 직접 $c$를 학습하는 대신 실수 값을 가지는 파라미터 $c_{raw}$를 두고, $c = \text{softplus}(c_{raw})$ 와 같이 양수 활성화 함수를 통과시켜 사용한다. Softplus 함수 $f(x) = \ln(1+e^x)$는 항상 양수를 반환하며 미분 가능하다.
    -   초기값은 $c=1.0$에 해당하는 $c_{raw} = \ln(e^{1.0}-1) \approx 0.54$로 설정하여 표준적인 쌍곡 공간에서 학습을 시작할 수 있다.

2.  **레이어별 곡률 할당**: 모델의 각 레이어 $l$마다 별도의 곡률 파라미터 $c_l$을 할당한다.
    ```python
    # 예시: PyTorch에서의 구현
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class HyperbolicLayer(nn.Module):
        def __init__(self, input_dim, output_dim, c_init=1.0):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            
            # 곡률을 위한 학습 가능한 파라미터
            c_raw_init = torch.log(torch.expm1(torch.tensor(c_init)))
            self.c_raw = nn.Parameter(torch.tensor(c_raw_init))

        def get_c(self):
            # 항상 양수를 보장
            return F.softplus(self.c_raw)

        def forward(self, x):
            c = self.get_c()
            # ... 쌍곡 연산 (로그 사상, 선형 변환, 지수 사상 등) ...
            # 모든 연산은 c에 의존하여 수행되어야 함
            return x_hyp
    ```

3.  **옵티마이저 등록**: 각 레이어의 $c_{raw}$ 파라미터는 다른 가중치들과 마찬가지로 옵티마이저에 등록되어 경사 하강법을 통해 업데이트된다.

### 5. 기대 효과 및 절충점 (Benefits and Trade-offs)

**기대 효과**:
-   **성능 향상**: 데이터와 태스크에 최적화된 기하학적 구조를 학습하여 모델의 최종 정확도를 높일 수 있다.
-   **표현력 증대**: 모델이 유클리드 공간부터 다양한 수준의 쌍곡 공간까지 넓은 범위의 기하학을 표현할 수 있게 된다.
-   **하이퍼파라미터 튜닝 감소**: 최적의 곡률 $c$를 찾는 과정을 자동화하여, 수동 튜닝의 필요성을 줄여준다.

**절충점**:
-   **학습 불안정성**: 곡률 $c$가 너무 크거나 작아지면 수치적으로 불안정한 연산(overflow/underflow)이 발생할 수 있다. 그래디언트 클리핑(gradient clipping)이나 $c$ 값의 범위를 제한하는 추가적인 장치가 필요할 수 있다.
-   **모델 복잡도 증가**: 학습할 파라미터가 레이어마다 추가되어 모델의 복잡도가 약간 증가한다.

### 6. 결론 (Conclusion)

동적 곡률은 쌍곡 신경망에 유연성을 부여하는 강력한 방법론이다. 곡률을 학습 가능한 파라미터로 전환함으로써, 모델은 데이터의 고유한 구조에 맞는 최적의 기하학을 스스로 발견할 수 있다. 이는 수동 하이퍼파라미터 튜닝의 한계를 극복하고, 모델의 표현력과 성능을 한 단계 끌어올릴 잠재력을 가진다. 학습 안정성을 확보하기 위한 장치와 함께 신중하게 구현된다면, 동적 곡률은 차세대 쌍곡 모델의 핵심 요소가 될 것이다. 