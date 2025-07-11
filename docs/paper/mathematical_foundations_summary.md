# Reality Stone 수학적 원리 종합 정리

## 1. 핵심 개념

Reality Stone은 **쌍곡 기하학(Hyperbolic Geometry)**을 기반으로 한 신경망 라이브러리입니다. 쌍곡 공간은 음의 곡률을 가진 비유클리드 공간으로, 계층적 구조를 효과적으로 표현할 수 있습니다.

## 2. 쌍곡 공간의 기하학적 특성

### 2.1 주요 특징
- **음의 곡률**: 평행선이 발산하는 특성
- **지수적 성장**: 반경 r인 구의 부피가 $e^r$에 비례
- **계층적 표현력**: 트리 구조를 거의 등거리적으로 임베딩 가능

### 2.2 리만 계량
쌍곡 공간의 리만 계량은 각 모델마다 다르게 표현됩니다:

**Poincaré Ball**:
$$g_x^{\mathbb{D}} = \left(\frac{2}{1 - c\|x\|^2}\right)^2 g^E$$

**Lorentz Model**:
$$g_x^{\mathbb{H}} = \text{diag}(1, -1, -1, ..., -1)$$

**Klein Model**:
$$g_x^{\mathbb{K}} = \frac{1}{1 - c\|x\|^2}I + \frac{c}{(1 - c\|x\|^2)^2}x \otimes x$$

## 3. Möbius 변환 - 쌍곡 공간의 핵심 연산

### 3.1 Möbius 덧셈
두 점 $u, v \in \mathbb{D}_c^n$의 Möbius 덧셈:

$$u \oplus_c v = \frac{(1 + 2c\langle u,v \rangle + c\|v\|^2)u + (1 - c\|u\|^2)v}{1 + 2c\langle u,v \rangle + c^2\|u\|^2\|v\|^2}$$

**기하학적 의미**: 
- 원점에서 $u$를 거쳐 $v$로 향하는 측지선을 따라 이동
- 유클리드 덧셈의 쌍곡 공간 버전

### 3.2 Möbius 스칼라 곱셈
스칼라 $r \in \mathbb{R}$과 점 $u \in \mathbb{D}_c^n$의 곱셈:

$$r \otimes_c u = \tanh\left(r \cdot \text{arctanh}(\sqrt{c}\|u\|)\right) \frac{u}{\sqrt{c}\|u\|}$$

**구현 단계**:
1. 거리 변환: $d = \text{arctanh}(\sqrt{c}\|u\|)$
2. 스칼라 곱: $d' = r \cdot d$  
3. 역변환: $\|r \otimes_c u\| = \frac{1}{\sqrt{c}}\tanh(d')$

### 3.3 Möbius 변환의 성질
- **결합법칙 부재**: $(u \oplus_c v) \oplus_c w \neq u \oplus_c (v \oplus_c w)$
- **교환법칙**: $u \oplus_c v = v \oplus_c u$ (특수한 경우만)
- **항등원**: $0 \oplus_c u = u$
- **역원**: $(-u) \oplus_c u = 0$

## 4. 측지선과 거리

### 4.1 측지선 방정식
쌍곡 공간에서 두 점을 잇는 최단 경로:

**Poincaré Ball**:
$$\gamma(t) = u \oplus_c (t \otimes_c ((-u) \oplus_c v))$$

**물리적 해석**: 
- $t = 0$: 시작점 $u$
- $t = 1$: 끝점 $v$
- $0 < t < 1$: 측지선상의 중간점

### 4.2 거리 공식

**Poincaré 거리**:
$$d_{\mathbb{D}}(u, v) = \frac{2}{\sqrt{c}} \text{arctanh}\left(\sqrt{c}\|(-u) \oplus_c v\|\right)$$

**Lorentz 거리**:
$$d_{\mathbb{H}}(u, v) = \frac{1}{\sqrt{c}} \text{arccosh}(-c\langle u, v \rangle_L)$$

**Klein 거리**:
$$d_{\mathbb{K}}(u, v) = \frac{1}{2\sqrt{c}} \ln\left(\frac{1 + \sqrt{c}\|u - v\|_K}{1 - \sqrt{c}\|u - v\|_K}\right)$$

## 5. Poincaré Ball Layer의 수학적 원리

### 5.1 측지선 보간
Poincaré ball layer는 두 특징 벡터 사이의 측지선 보간을 수행:

$$\text{PoincareBallLayer}(u, v, c, t) = ((1-t) \otimes_c u) \oplus_c (t \otimes_c v)$$

### 5.2 기하학적 해석
- **선형 보간의 일반화**: 유클리드 공간의 $(1-t)u + tv$를 쌍곡 공간으로 확장
- **측지선 혼합**: 두 표현 사이의 자연스러운 전환
- **곡률 효과**: $c$가 클수록 비선형성 증가

### 5.3 실험 결과 해석
```
t = 0.5:  97.27%  # 균등 혼합
t = 0.7:  97.48%  # 최적점 (v에 더 가중치)
t = 1.0:  97.33%  # v만 사용
```

최적 성능이 $t = 0.7$에서 나타난 이유:
- 원본 특징($u$)과 변환된 특징($v$)의 적절한 균형
- 쌍곡 공간의 기하학적 특성 활용

## 6. 접공간과 지수/로그 맵

### 6.1 지수 맵 (Exponential Map)
접공간 $T_p\mathbb{D}_c^n$에서 manifold로의 매핑:

$$\exp_p^c(v) = p \oplus_c \left(\tanh\left(\frac{\sqrt{c}\|v\|}{2}\right) \frac{v}{\sqrt{c}\|v\|}\right)$$

### 6.2 로그 맵 (Logarithmic Map)
Manifold에서 접공간으로의 매핑:

$$\log_p^c(x) = \frac{2}{\sqrt{c}} \text{arctanh}(\sqrt{c}\|(-p) \oplus_c x\|) \frac{(-p) \oplus_c x}{\|(-p) \oplus_c x\|}$$

### 6.3 응용: Hyperbolic Linear Layer
```python
def hyperbolic_linear(x, W, b, c):
    # 1. 접공간으로 매핑
    x_tangent = log_map(x, c)
    
    # 2. 선형 변환
    y_tangent = x_tangent @ W + b
    
    # 3. 다시 manifold로 매핑
    y = exp_map(y_tangent, c)
    
    return y
```

## 7. 곡률의 영향

### 7.1 곡률 파라미터 $c$의 역할
- **$c \to 0$**: 유클리드 공간에 근사
- **$c = 1$**: 표준 쌍곡 공간
- **$c > 1$**: 더 강한 쌍곡성

### 7.2 곡률과 표현력
$$\text{용량} \propto \frac{1}{\sqrt{c}}$$

작은 곡률일수록 더 넓은 공간을 표현 가능하지만, 계층적 구조 표현력은 감소

### 7.3 동적 곡률
```python
c(x) = c_0 \cdot \sigma(w^T x + b)
```
입력에 따라 곡률을 조정하여 적응적 표현 학습

## 8. 정규화 기법

### 8.1 경계 정규화
Poincaré ball의 경계 근처에서 수치적 불안정성 방지:

$$\mathcal{L}_{\text{boundary}} = \sum_{i} \lambda \cdot \max(0, \|x_i\| - (1-\epsilon)/\sqrt{c})^2$$

### 8.2 곡률 정규화
곡률이 너무 크거나 작아지는 것을 방지:

$$\mathcal{L}_{\text{curvature}} = (c - c_{\text{target}})^2$$

### 8.3 측지선 분산 정규화
점들이 측지선을 따라 고르게 분포하도록 유도:

$$\mathcal{L}_{\text{geodesic}} = \text{Var}_{i,j}(d_{\mathbb{D}}(x_i, x_j))$$

## 9. 고급 수학적 개념

### 9.1 Laplace-Beltrami 연산자
쌍곡 공간의 라플라시안:

$$\Delta_{\mathbb{H}} f = \frac{1}{\sqrt{|g|}} \partial_i\left(\sqrt{|g|}g^{ij}\partial_j f\right)$$

**Poincaré Ball에서**:
$$\Delta_{\mathbb{D}} = \left(\frac{1-c\|x\|^2}{2}\right)^2 \Delta_{\mathbb{E}}$$

### 9.2 쌍곡 푸리에 변환
$$\mathcal{F}_{\mathbb{H}}[f](\lambda) = \int_{\mathbb{H}^n} f(x) \phi_\lambda(x) d\mu_{\mathbb{H}}(x)$$

여기서 $\phi_\lambda$는 쌍곡 공간의 고유함수

### 9.3 Chebyshev 근사
쌍곡 함수의 효율적 계산:

$$\tanh(x) = \sum_{k=0}^{n} a_k T_k(x) + O(x^{n+1})$$

## 10. 구현상의 수치적 고려사항

### 10.1 수치적 안정성
- **Clipping**: $\|x\| < \frac{1-\epsilon}{\sqrt{c}}$
- **최소 분모**: $\text{denom} = \max(\text{denom}, 10^{-6})$
- **안전한 역삼각함수**: $\text{arctanh}(\min(x, 1-\epsilon))$

### 10.2 효율적 계산
- **Fused Operations**: 여러 연산을 하나의 커널로 통합
- **In-place 연산**: 메모리 효율성
- **벡터화**: SIMD 명령어 활용

## 11. 이론적 보장

### 11.1 표현력
- **Universal Approximation**: 쌍곡 신경망도 universal approximator
- **계층적 구조**: $O(\log n)$ 왜곡으로 트리 임베딩 가능

### 11.2 수렴성
- **측지선 경사하강법**: 수렴 보장
- **Riemannian SGD**: $O(1/\sqrt{T})$ 수렴률

## 12. 결론

Reality Stone의 수학적 기반은 다음과 같이 요약됩니다:

1. **엄밀한 기하학적 기초**: 쌍곡 기하학의 모든 핵심 개념 구현
2. **효율적인 연산**: Möbius 변환을 통한 쌍곡 연산
3. **수치적 안정성**: 경계 처리 및 정규화
4. **확장 가능성**: 다양한 쌍곡 모델 간 변환
5. **이론적 보장**: 표현력과 수렴성

이러한 수학적 원리들이 Reality Stone을 강력하고 신뢰할 수 있는 쌍곡 신경망 라이브러리로 만듭니다. 