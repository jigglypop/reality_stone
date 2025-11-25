# 심층 다양체 주입: 신경망의 기하학적 수술 (Deep Manifold Injection)

## 1. 개념 (Concept)
표준 신경망은 유클리드 공간에서의 행렬 곱셈에 의존합니다. **심층 다양체 주입(Deep Manifold Injection)**은 이러한 연산을 리만 기하학적 대응물로 교체하여, AI의 "뇌세포"를 굽은 공간으로 이식하는 외과적 수술 기법입니다.

## 2. 수학적 정식화 (Mathematical Formulation)

### A. 유클리드 선형 레이어 (Euclidean Linear Layer)
표준 `nn.Linear` 또는 `Conv1D`에서:
$$ y = x W^T + b $$
이는 벡터 공간이 평평하다고 가정합니다. 모든 지점에서의 그래디언트 흐름은 균일합니다.

### B. 리만 선형 레이어 (Riemannian Linear Layer)
우리는 가중치 행렬 $W$를 단순한 매핑이 아니라, 다양체 위의 **접벡터(Tangent Vector)**를 정의하는 것으로 재해석합니다. 연산은 측지선(Geodesic)을 따르는 이동이 됩니다.

1. **로그 맵 (Logarithmic Map - Tangent Space Projection)**:
   입력 $x$를 접공간(Tangent Space) $\mathcal{T}_x M$으로 투영합니다.
   $$ v = \log_x(x) $$
   (효율성을 위해 원점 근처에서는 항등 함수로 근사합니다.)

2. **평행 이동 및 선형 변환 (Parallel Transport & Linear Transform)**:
   $$ v' = v W^T $$
   
3. **등각 보정 (Conformal Correction - The "Reality Stone" Effect)**:
   굽은 공간(예: 곡률 $c$인 푸앵카레 볼)에서 메트릭 텐서 $g_x$는 원점으로부터의 거리에 따라 스케일이 변합니다. 이를 보정하기 위해 **등각 계수(Conformal Factor) $\lambda_x$**를 적용합니다:
   
   $$ \lambda_x = \frac{2}{1 - c \|x\|^2} $$
   
   최종 출력은 다음과 같습니다:
   $$ y = v' \cdot \lambda_x + b $$

### C. 왜 작동하는가? (Why This Works)
- **동적 가중치 (Dynamic Weighting)**: $\lambda_x$ 항으로 인해, 뉴런의 유효 가중치는 입력의 "에너지(Norm)"에 따라 동적으로 변합니다.
- **계층 구조 인식 (Hierarchy Awareness)**: 경계면($\|x\| \to 1/\sqrt{c}$)에 가까운 입력은 그래디언트가 폭발적으로 증폭되어, 계층 구조의 말단(Leaf Node) 정보를 자연스럽게 인코딩합니다.

## 3. 구현 전략 (Implementation Strategy)
대상 모델(GPT-2, Llama 등)의 모든 `nn.Linear`와 `Conv1D`를 `RiemannianLinear`로 교체합니다.

```python
# 순전파(Forward Pass) 의사 코드
def forward(self, x):
    # 1. 유클리드 선형 변환 (접공간 근사)
    y_linear = F.linear(x, self.weight, self.bias)
    
    # 2. 리만 등각 보정 (Conformal Correction)
    # 입력의 에너지(Norm) 측정
    x_norm_sq = x.pow(2).sum(dim=-1)
    
    # 람다(x) 계산: 곡률 c에 따른 공간 왜곡 반영
    lambda_x = 2.0 / (1.0 - self.c * x_norm_sq)
    
    # 보정된 신호 전달
    return y_linear * lambda_x
```

이 겉보기에 단순한 스케일링이 네트워크의 가장 깊은 곳에 비유클리드 기하학을 주입합니다.
