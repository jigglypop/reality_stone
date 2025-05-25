# Reality Stone 하이퍼볼릭 신경망 라이브러리 문서

Reality Stone은 하이퍼볼릭 기하학을 활용한 신경망 라이브러리입니다. 이 문서는 src/ 디렉토리의 모든 코드를 수학적 배경과 함께 상세히 설명합니다.

## 📁 문서 구조

### 🔢 [기본 연산 (ops/)](./ops/)
하이퍼볼릭 공간에서의 기본적인 수학 연산들
- [Möbius 연산](./ops/mobius.md) - 포인카레 디스크 모델의 핵심 연산
- [Lorentz 연산](./ops/lorentz.md) - 하이퍼볼로이드 모델의 연산
- [Klein 연산](./ops/klein.md) - Klein 모델의 연산

### 🧠 [신경망 레이어 (layers/)](./layers/)
하이퍼볼릭 공간에서 동작하는 신경망 레이어들
- [Poincare Ball 레이어](./layers/poincare_ball.md) - 포인카레 볼 모델 기반 레이어
- [Lorentz 레이어](./layers/lorentz.md) - 로렌츠 모델 기반 레이어
- [Klein 레이어](./layers/klein.md) - Klein 모델 기반 레이어

### 🚀 [고급 기능 (advanced/)](./advanced/)
최첨단 하이퍼볼릭 기하학 알고리즘들
- [Chebyshev 다항식](./advanced/chebyshev.md) - 체비셰프 근사법과 하이퍼볼릭 함수
- [Laplace-Beltrami 연산자](./advanced/laplace_beltrami.md) - 매니폴드에서의 미분 연산
- [하이퍼볼릭 FFT](./advanced/hyperbolic_fft.md) - 하이퍼볼릭 공간에서의 푸리에 변환
- [동적 곡률](./advanced/dynamic_curvature.md) - 적응적 곡률 조정
- [정규화](./advanced/regularization.md) - 하이퍼볼릭 정규화 기법
- [측지선 활성화](./advanced/geodesic_activation.md) - 측지선 기반 활성화 함수
- [융합 연산](./advanced/fused_ops.md) - 최적화된 복합 연산

### 🛠️ [유틸리티 (utils/)](./utils/)
공통 유틸리티 함수들과 CUDA 헬퍼

### ⚙️ [설정 (config/)](./config/)
라이브러리 설정과 상수 정의

## 🧮 수학적 배경

### 하이퍼볼릭 기하학의 기본 개념

하이퍼볼릭 기하학은 음의 곡률을 갖는 비유클리드 기하학입니다. 계층적 구조와 트리형 데이터를 표현하는데 특히 유용합니다.

#### 1. 포인카레 디스크 모델 (Poincare Disk Model)
**정의**: 단위 원판 $\mathbb{D}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$

**계량 텐서**:
$$g_{ij} = \frac{4\delta_{ij}}{(1-\|x\|^2)^2}$$

**거리 공식**:
$$d_{\mathbb{D}}(x,y) = \text{arccosh}\left(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$$

#### 2. 로렌츠 모델 (Lorentz Model)
**정의**: 하이퍼볼로이드 $\mathbb{L}^n = \{x \in \mathbb{R}^{n+1} : \langle x,x \rangle_{\mathcal{L}} = -1, x_0 > 0\}$

**민코프스키 내적**:
$$\langle x,y \rangle_{\mathcal{L}} = -x_0y_0 + \sum_{i=1}^n x_iy_i$$

#### 3. Klein 모델 (Klein Model)
**정의**: 단위 원판에서의 직선 측지선 모델

## 🔧 구현 특징

### CPU/CUDA 하이브리드 구현
- 모든 핵심 연산이 CPU와 CUDA 버전으로 구현
- 자동 디바이스 선택과 메모리 관리
- 최적화된 CUDA 커널 설계

### 수치적 안정성
- 경계 근처에서의 수치적 불안정성 방지
- 안전한 클리핑과 정규화
- IEEE 754 표준 준수

### 확장성
- 모듈화된 설계로 새로운 연산 추가 용이
- 템플릿 기반 제네릭 프로그래밍
- Python 바인딩을 통한 편리한 사용

## 📖 사용 예제

```python
import torch
import reality_stone as rs

# 포인카레 디스크에서 두 점의 Möbius 덧셈
x = torch.randn(32, 64) * 0.1
y = torch.randn(32, 64) * 0.1
result = rs.mobius_add_cuda(x, y, c=1.0)

# 하이퍼볼릭 신경망 레이어
layer = rs.PoincareBallLayer(input_dim=64, output_dim=32)
output = layer(x)

# 고급 기능: 체비셰프 근사
coeffs = rs.chebyshev_approximation_cuda(x, order=10, curvature=1.0)
```

## 📚 참고 문헌

1. **Hyperbolic Neural Networks** - Ganea et al. (2018)
2. **Poincaré Embeddings for Learning Hierarchical Representations** - Nickel & Kiela (2017)
3. **Hyperbolic Graph Neural Networks** - Chami et al. (2019)
4. **Lorentzian Distance Learning for Hyperbolic Representations** - Law et al. (2019)

---

각 모듈의 상세한 수학적 배경과 구현 내용은 해당 섹션의 문서를 참조하세요. 