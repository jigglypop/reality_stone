# HyperButterfly: PyTorch를 위한 효율적인 하이퍼볼릭 기하학 라이브러리

[![PyPI version](https://img.shields.io/pypi/v/hyper-butterfly.svg)](https://pypi.org/project/hyper-butterfly/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![라이선스](https://img.shields.io/badge/%EB%9D%BC%EC%9D%B4%EC%84%A0%EC%8A%A4-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🌟 개요

**HyperButterfly**는 하이퍼볼릭 공간에서의 기하학적 딥러닝을 위한 고성능 PyTorch 확장 라이브러리입니다. 리만 다양체, 특히 하이퍼볼릭 공간에서의 효율적인 연산과 Butterfly 팩터를 통한 효율적인 변환 구현을 제공합니다. CUDA 가속을 활용하여 대규모 데이터셋에서도 빠른 연산이 가능합니다.


```bash
pip cache purge && python setup.py build_ext --inplace && pip install -e .
```

```csharp

hyper_butterfly/
├─ csrc/
│  ├─ include/
│  │  └─ hyper_butterfly/
│  │     ├─ manifolds/               # manifold별 헤더
│  │     │  ├─ base.h                # 공통 추상 인터페이스
│  │     │  ├─ poincare.h
│  │     │  ├─ lorentz.h
│  │     │  └─ sphere.h
│  │     ├─ maps/                     # map 연산들 (로그, 익스펜 등)
│  │     │  ├─ base.h
│  │     │  ├─ poincare_maps.h
│  │     │  ├─ lorentz_maps.h
│  │     │  └─ sphere_maps.h
│  │     └─ extension.h              # 바인딩 노출부
│  └─ src/
│     ├─ manifolds/
│     │  ├─ poincare.cpp
│     │  ├─ lorentz.cpp
│     │  └─ sphere.cpp
│     ├─ maps/
│     │  ├─ poincare_maps_cpu.cpp
│     │  ├─ poincare_maps_cuda.cu
│     │  ├─ lorentz_maps_cpu.cpp
│     │  └─ sphere_maps_cpu.cpp
│     └─ extension.cpp
└─ python/
   ├─ manifold/                       # Python 레이어별 구현
   │  ├─ base.py                      # 인터페이스, 팩토리
   │  ├─ poincare.py
   │  ├─ lorentz.py
   │  └─ sphere.py
   ├─ layers.py                       # 하이퍼-버터플라이 레이어
   └─ utils.py

```


## ✨ 주요 기능

- 🚀 **포인카레 볼 모델**: 하이퍼볼릭 공간의 지수 맵, 로그 맵, 측지 거리 계산을 위한 최적화된 C++/CUDA 구현
- 🧮 **Butterfly 팩터**: O(N log N) 복잡도로 행렬 변환을 근사하는 효율적인 알고리즘
- 🔄 **Hyper-Butterfly 레이어**: 하이퍼볼릭 공간에서의 효율적인 신경망 레이어
- 🔍 **수치적 안정성**: 유한 조건수와 역전파 안정성 보장
- 📊 **시각화 도구**: 하이퍼볼릭 공간에서의 데이터 시각화

## 📚 수학적 원리

### 포인카레 볼 모델

곡률 $c > 0$인 $N$차원 쌍곡공간은 다음과 같이 정의됩니다:

$$\mathbb{D}^N_c = \{x \in \mathbb{R}^N : c\,\|x\|_2^2 < 1\}$$

#### 지수 맵과 로그 맵

- **지수 맵** $\exp_0^c: \mathbb{R}^N \to \mathbb{D}^N_c$:
  
  $$\exp_0^c(v) = \tanh(\sqrt{c}\,\|v\|)\;\frac{v}{\sqrt{c}\,\|v\|}$$

- **로그 맵** $\log_0^c: \mathbb{D}^N_c \to \mathbb{R}^N$:
  
  $$\log_0^c(x) = \frac{\tanh^{-1}(\sqrt{c}\,\|x\|)}{\sqrt{c}\,\|x\|}\;x$$

### Butterfly 팩터

$N=2^L$일 때, 각 단계 $\ell=1,\dots,L$의 Butterfly 팩터 $B_\ell \in \mathbb{R}^{N \times N}$는:

$$B_\ell = \bigoplus_{k=1}^{2^{L-\ell}}
\begin{pmatrix}
a_{k,\ell} & b_{k,\ell}\\
-b_{k,\ell} & a_{k,\ell}
\end{pmatrix}$$

즉, $2 \times 2$ 회전 블록이 대각선상에 반복 배치된 block-diagonal 행렬로 정의합니다.

### Hyper-Butterfly 레이어

Hyper-Butterfly 레이어는 다음 순전파로 정의됩니다:

$$\begin{aligned}
u &= \log_0^c(x)\\
v &= B_L\,B_{L-1}\,\cdots\,B_1\,u\\
y &= \exp_0^c(v)
\end{aligned}$$

### 주요 수학적 특성

1. **조건수 유한성**: 곡률 $c < 1$, 입력 $\|x\| \le R$에 대해 $cR^2 < 0.9$이면 조건수는 $\kappa(f) \le \frac{1}{1-cR^2}$로 바운딩됩니다.

2. **역전파 안정성**: 동일 조건에서 그래디언트는 $\|\nabla_x L\| \le \frac{1}{(1-cR^2)^2}\|\nabla_y L\|$를 만족합니다.

3. **보편 근사성**: Stone-Weierstrass 정리에 의해 컴팩트 리만 다양체 위의 연속 함수를 임의의 정밀도로 근사할 수 있습니다.

4. **효율적 차원 축소**: Nash 임베딩을 통해 리만 다양체를 정보 손실 없이 $O(N\log N)$ 파라미터로 표현합니다.

## 순전파 (Forward) 과정:

### 1. 로그 맵 (Log Map) - 하이퍼볼릭 → 유클리드 접공간
$$u = \log_{\mathbf{0}}^c(x) = \frac{2}{\sqrt{c}} \tanh^{-1}(\sqrt{c}||x||) \frac{x}{||x||}$$

여기서:
- $x$: 하이퍼볼릭 공간의 입력 벡터
- $c$: 하이퍼볼릭 공간의 곡률
- $u$: 접공간 벡터

### 2. 버터플라이 변환 - 유클리드 접공간에서 선형 변환
$$v = B(u, \Theta) = \prod_{l=0}^{L-1} B_l(u, \theta_l)$$

각 레이어 $l$에서 버터플라이 연산:
$$B_l(u, \theta_l)[i] = a \cdot u[i] + b \cdot u[i+2^l], & \text{if } i \bmod 2^{l+1} < 2^l c \cdot u[i-2^l] + d \cdot u[i], & \text{if } i \bmod 2^{l+1} \geq 2^l$$

여기서:
- $\Theta = \{\theta_0, \theta_1, ..., \theta_{L-1}\}$: 버터플라이 네트워크의 파라미터
- $\theta_l = \{a, b, c, d\}$: 각 $2 \times 2$ 블록의 파라미터
- $L$: 레이어 수

### 3. 지수 맵 (Exp Map) - 유클리드 접공간 → 하이퍼볼릭
$$y = \exp_{\mathbf{0}}^c(v) = \tanh\left(\sqrt{c}\frac{||v||}{2}\right) \frac{v}{\sqrt{c}||v||}$$

여기서:
- $v$: 변환된 접공간 벡터
- $y$: 하이퍼볼릭 공간의 출력 벡터



## 📦 설치 방법

```bash
git clone https://github.com/username/hyper_butterfly.git
cd hyper_butterfly
pip install -e .
```

## 🚀 빠른 시작

```python
import torch
import hyper_butterfly

# 포인카레 볼 모델에서 연산 예제
x = torch.zeros(1, 2)  # 포인카레 볼의 원점
v = torch.torch::Tensor([[0.3, 0.4]])  # 접벡터

# 지수 사상 적용
y = hyper_butterfly.exp_map(x, v)
print("원점으로부터의 지수 맵 결과:", y)

# 거리 계산
dist = hyper_butterfly.distance(x, y)
print(f"리만 거리: {dist.item():.4f}")

# Hyper-Butterfly 레이어 사용
layer = hyper_butterfly.HyperButterflyLayer(dim=8, num_layers=3, curvature=0.5)
input_data = torch.randn(8) * 0.3  # 반지름이 작은 점들
output = layer(input_data)
```

## 🧪 테스트 실행

라이브러리의 주요 기능을 테스트하려면:

```bash
python test.py
```

## 📚 주요 구현 내용

### 포인카레 볼 모델

포인카레 볼 모델은 하이퍼볼릭 공간의 등각 모델로, 다음과 같은 핵심 연산이 구현되어 있습니다:

1. **지수 맵 (Exponential Map)**:
   ```python
   # 원점에서의 지수 맵
   y = hyper_butterfly.exp_map(torch.torch::zeros_like(x), v, c=1.0)
   ```

2. **로그 맵 (Logarithmic Map)**:
   ```python
   # 원점으로의 로그 맵
   v = hyper_butterfly.log_map(torch.torch::zeros_like(y), y, c=1.0)
   ```

3. **측지 거리 (Geodesic Distance)**:
   ```python
   dist = hyper_butterfly.distance(x, y, c=1.0)
   ```

### Butterfly 팩터

Butterfly 팩터는 행렬을 효율적으로 표현하기 위한 방법으로, 다음과 같이 구현되어 있습니다:

```python
# 버터플라이 변환 레이어 적용
output = hyper_butterfly.butterfly_transform(input_data, params, layer=0)
```

### Hyper-Butterfly 레이어

Hyper-Butterfly 레이어는 하이퍼볼릭 공간에서 효율적인 신경망 레이어를 구현합니다:

```python
layer = hyper_butterfly.HyperButterflyLayer(dim=8, num_layers=3, curvature=0.5)
output = layer(input_data)
```

## 📄 논문 참조

이 구현은 "Hyper-Butterfly 네트워크: 계산적 하이퍼볼릭 기하학의 수학적 분석" 논문을 기반으로 합니다. 자세한 수학적 이론과 증명은 `hyper_butterfly.md` 문서를 참조하세요.

## 🤝 기여하기

기여는 언제나 환영합니다! 버그 리포트, 기능 요청, 풀 리퀘스트 모두 가능합니다.

## 📝 라이선스

MIT 라이선스에 따라 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
