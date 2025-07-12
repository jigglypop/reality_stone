# 수학적 배경

Reality Stone에서 사용되는 하이퍼볼릭 기하학의 수학적 기초를 설명합니다.

## 하이퍼볼릭 기하학 개요

### 기본 개념

하이퍼볼릭 기하학은 **음의 곡률**을 갖는 비유클리드 기하학입니다. 유클리드 기하학과 달리 평행선 공준이 성립하지 않으며, 이는 계층적 구조를 자연스럽게 표현할 수 있는 독특한 성질을 제공합니다.

### 주요 특징

1. **음의 곡률**: 공간이 "안쪽으로 굽어있음"
2. **지수적 부피 증가**: 반지름이 증가할 때 부피가 지수적으로 증가
3. **계층적 구조**: 트리와 같은 계층적 데이터를 자연스럽게 임베딩

## 하이퍼볼릭 공간의 모델들

Reality Stone에서 지원하는 세 가지 주요 하이퍼볼릭 모델을 소개합니다.

### 1. Poincaré Ball 모델

**정의**: 단위 원판 $\mathbb{D}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$

**Riemannian 계량**:
$$g_x = \frac{4}{(1-\|x\|^2)^2} \cdot I_n$$

**거리 함수**:
$$d_{\mathbb{D}}(x,y) = \text{arccosh}\left(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$$

**특징**:
- 등각 모델 (각도 보존)
- 경계에서 무한대로 발산
- 직관적인 시각화 가능

### 2. Lorentz 모델 (하이퍼볼로이드)

**정의**: $\mathbb{L}^n = \{x \in \mathbb{R}^{n+1} : \langle x,x \rangle_{\mathcal{L}} = -1, x_0 > 0\}$

**민코프스키 내적**:
$$\langle x,y \rangle_{\mathcal{L}} = -x_0y_0 + \sum_{i=1}^n x_iy_i$$

**거리 함수**:
$$d_{\mathbb{L}}(x,y) = \text{arccosh}(-\langle x,y \rangle_{\mathcal{L}})$$

**특징**:
- 수치적으로 안정적
- 선형 대수와 잘 호환
- 차원이 하나 증가

### 3. Klein 모델

**정의**: 단위 원판에서 직선 측지선을 갖는 모델

**거리 함수**:
$$d_K(x,y) = \frac{1}{\sqrt{c}} \text{arccosh}\left(\frac{1-\langle x,y\rangle}{\sqrt{(1-\|x\|^2)(1-\|y\|^2)}}\right)$$

**특징**:
- 측지선이 유클리드 직선
- 각도 보존하지 않음
- 기하학적 직관이 뛰어남

## 좌표 변환

### Poincaré ↔ Lorentz

**Poincaré → Lorentz**:
$$\phi: \mathbb{D}^n \to \mathbb{L}^n$$
$$\phi(x) = \frac{1}{\sqrt{c}} \left(\frac{1+c\|x\|^2}{1-c\|x\|^2}, \frac{2x}{1-c\|x\|^2}\right)$$

**Lorentz → Poincaré**:
$$\phi^{-1}: \mathbb{L}^n \to \mathbb{D}^n$$
$$\phi^{-1}(x) = \frac{\sqrt{c} \cdot x_{1:n}}{1 + \sqrt{c} \cdot x_0}$$

### Poincaré ↔ Klein

**Poincaré → Klein**:
$$\psi: \mathbb{D}^n \to \mathbb{D}^n$$
$$\psi(x) = \frac{2x}{1+\|x\|^2}$$

**Klein → Poincaré**:
$$\psi^{-1}: \mathbb{D}^n \to \mathbb{D}^n$$
$$\psi^{-1}(x) = \frac{x}{1+\sqrt{1-\|x\|^2}}$$

## 핵심 연산

### Mobius 변환

Poincaré Ball에서의 핵심 연산은 Mobius 변환입니다.

**Mobius 덧셈**:
$$x \oplus_c y = \frac{(1+2c\langle x,y\rangle + c\|y\|^2)x + (1-c\|x\|^2)y}{1+2c\langle x,y\rangle + c^2\|x\|^2\|y\|^2}$$

**Mobius 스칼라 곱셈**:
$$r \otimes_c x = \frac{1}{\sqrt{c}} \tanh\left(r \tanh^{-1}(\sqrt{c}\|x\|)\right) \frac{x}{\|x\|}$$

**성질**:
- 결합법칙: $(x \oplus_c y) \oplus_c z = x \oplus_c (y \oplus_c z)$
- 항등원: $0 \oplus_c x = x$
- 역원: $x \oplus_c (-x) = 0$

### 지수 및 로그 매핑

**지수 매핑** (접선 공간 → 다양체):
$$\text{Exp}_x^c(v) = x \oplus_c \left(\tanh\left(\sqrt{c}\frac{\lambda_x^c \|v\|}{2}\right) \frac{v}{\sqrt{c}\|v\|}\right)$$

**로그 매핑** (다양체 → 접선 공간):
$$\text{Log}_x^c(y) = \frac{2}{\sqrt{c}\lambda_x^c} \tanh^{-1}(\sqrt{c}\|(-x) \oplus_c y\|) \frac{(-x) \oplus_c y}{\|(-x) \oplus_c y\|}$$

여기서 $\lambda_x^c = \frac{2}{1-c\|x\|^2}$는 공형 인수입니다.

## 신경망에서의 활용

### 하이퍼볼릭 레이어

Reality Stone의 하이퍼볼릭 레이어는 다음과 같이 정의됩니다:

$$\text{Layer}(u, v) = (1-t) \otimes_c u \oplus_c t \otimes_c v$$

여기서:
- $u, v$: 입력 벡터
- $c$: 곡률 매개변수
- $t$: 보간 비율

### 그래디언트 계산

하이퍼볼릭 공간에서의 그래디언트는 Riemannian 그래디언트로 변환됩니다:

$$\nabla_{\mathcal{R}} f = \left(\frac{1-c\|x\|^2}{2}\right)^2 \nabla_{\mathcal{E}} f$$

이는 하이퍼볼릭 공간의 계량 구조를 반영한 것입니다.

## 수치적 고려사항

### 안정성 문제

1. **경계 근처**: $\|x\| \to 1$일 때 수치적 불안정성
2. **작은 곡률**: $c \to 0$일 때 유클리드 공간으로 퇴화
3. **큰 곡률**: $c$가 클 때 그래디언트 폭발

### 해결책

```python
# 안전한 클리핑
def safe_clip(x, max_norm=0.99):
    norm = torch.norm(x, dim=-1, keepdim=True)
    return x * torch.clamp(norm, max=max_norm) / (norm + 1e-8)

# 안정적인 arccosh 계산
def stable_arccosh(x):
    return torch.log(x + torch.sqrt(torch.clamp(x**2 - 1, min=1e-8)))
```

## 이론적 배경

### 왜 하이퍼볼릭 공간인가?

1. **계층적 구조**: 트리의 자연스러운 임베딩
2. **지수적 확장**: 깊이에 따른 노드 수의 지수적 증가
3. **거리 보존**: 계층적 거리를 기하학적 거리로 보존

### 복잡도 분석

하이퍼볼릭 공간에서 $n$개 노드를 갖는 트리를 임베딩하는 데 필요한 차원:

$$d = O(\log n)$$

이는 유클리드 공간의 $O(n)$과 대비됩니다.

## 응용 분야

### 1. 자연어 처리
- 단어 임베딩 (Word2Vec의 하이퍼볼릭 버전)
- 문서 분류 (계층적 토픽 모델링)

### 2. 컴퓨터 비전
- 이미지 분류 (계층적 카테고리)
- 객체 검출 (부분-전체 관계)

### 3. 그래프 분석
- 소셜 네트워크 분석
- 지식 그래프 임베딩

### 4. 생물정보학
- 계통 발생학적 분석
- 단백질 구조 예측

## 실험적 검증

### 이론적 예측 vs 실제 결과

```python
import torch
import reality_stone as rs

# 이론적 거리 계산
def theoretical_distance(x, y, c):
    norm_x = torch.norm(x, dim=-1)
    norm_y = torch.norm(y, dim=-1)
    xy_dot = torch.sum(x * y, dim=-1)
    
    numerator = 2 * torch.norm(x - y, dim=-1)**2
    denominator = (1 - c * norm_x**2) * (1 - c * norm_y**2)
    
    return torch.acosh(1 + numerator / denominator) / torch.sqrt(torch.tensor(c))

# 실제 구현과 비교
x = torch.randn(100, 64) * 0.1
y = torch.randn(100, 64) * 0.1
c = 1.0

theoretical = theoretical_distance(x, y, c)
actual = rs.poincare_distance(x, y, c)

print(f"평균 절대 오차: {torch.mean(torch.abs(theoretical - actual)):.6f}")
```

## 🎓 추가 학습 자료

### 필수 논문
1. **Poincaré Embeddings** - Nickel & Kiela (2017)
2. **Hyperbolic Neural Networks** - Ganea et al. (2018)
3. **Hyperbolic Graph Neural Networks** - Chami et al. (2019)

### 온라인 자료
- **Hyperbolic Geometry 강의**: MIT OpenCourseWare
- **Riemannian Geometry**: Stanford CS468
- **Geometric Deep Learning**: Imperial College London

### 실습 자료
- **Geoopt**: PyTorch 기반 Riemannian 최적화 라이브러리
- **Manifold Learning**: Scikit-learn 예제들

---

이 수학적 배경 지식을 바탕으로 Reality Stone의 다양한 기능을 더 깊이 이해하고 활용할 수 있습니다! 