# 리만 메트릭 계층 구조 (Riemannian Metric Hierarchy)

## 1. 메트릭 텐서의 수학적 정의

### 1.1 일반 리만 메트릭

리만 다양체 $(M, g)$ 에서 메트릭 텐서 $g$ 는 각 점 $p \in M$ 에서 접공간 $T_pM$ 위의 내적을 정의합니다:

$$
g_p: T_pM \times T_pM \to \mathbb{R}
$$

국소 좌표 $(x^1, \ldots, x^n)$ 에서:

$$
g_{ij}(x) = \langle \partial_i, \partial_j \rangle, \quad
ds^2 = g_{ij} \, dx^i dx^j
$$

### 1.2 Reality Stone의 4가지 메트릭 모델

#### 1.2.1 푸앵카레 메트릭 (Poincaré Metric)

**정의**: 단위 공 $\mathbb{B}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$ 위의 등각 메트릭

$$
g_{ij}^P(x) = \lambda^2(x) \, \delta_{ij}, \quad
\lambda(x) = \frac{2}{1-c\|x\|^2}
$$

**곡률**: $K = -c$ (상수 음곡률)

**거리**:

$$
d_P(x, y) = \frac{1}{\sqrt{c}} \operatorname{arcosh}\left( 1 + 2c \frac{\|x-y\|^2}{(1-c\|x\|^2)(1-c\|y\|^2)} \right)
$$

**특성**:
- 계층 구조 표현에 최적
- 지수적 공간 확장 (중심에서 멀어질수록 공간이 빠르게 늘어남)
- 경계에서 무한대 거리

#### 1.2.2 로렌츠 메트릭 (Lorentz/Hyperboloid Metric)

**정의**: 민코프스키 공간 $\mathbb{R}^{1,n}$ 의 초쌍곡면

$$
\mathcal{H}^n = \{x \in \mathbb{R}^{n+1} : x_0^2 - \sum_{i=1}^n x_i^2 = \frac{1}{c}, \, x_0 > 0\}
$$

**메트릭**: 민코프스키 내적

$$
\langle u, v \rangle_L = u_0 v_0 - \sum_{i=1}^n u_i v_i
$$

**거리**:

$$
d_L(x, y) = \frac{1}{\sqrt{c}} \operatorname{arcosh}(c \langle x, y \rangle_L)
$$

**특성**:
- 상대론적 기하학과 직접 연결
- 수치 안정성 우수 (경계 없음)
- Poincaré와 동형 (변환 가능)

#### 1.2.3 클라인 메트릭 (Klein/Projective Metric)

**정의**: 사영 모델, 측지선이 유클리드 직선

$$
g_{ij}^K(x) = \frac{1}{1-c\|x\|^2} \left( \delta_{ij} + \frac{c x_i x_j}{1-c\|x\|^2} \right)
$$

**거리**:

$$
d_K(x, y) = \frac{1}{\sqrt{c}} \operatorname{arcosh}\left( \frac{1 - c\langle x, y \rangle}{\sqrt{(1-c\|x\|^2)(1-c\|y\|^2)}} \right)
$$

**특성**:
- 측지선이 유클리드 직선 (시각화 용이)
- 경계 거리가 유한
- 일부 연산이 단순함

#### 1.2.4 대각 메트릭 (Diagonal Metric, 학습 가능)

**정의**: 각 축이 독립적으로 스케일되는 학습 가능한 메트릭

$$
g_{ij}^D(x) = w_i(x) \, \delta_{ij}
$$

여기서 $w_i(x) = \text{softplus}(\theta_i \cdot x_i) + \varepsilon$

**특성**:
- 파라미터 효율적 (차원당 1개 파라미터)
- 메트릭 학습 가능
- 데이터에 적응적

## 2. 크리스토펠 기호 (Christoffel Symbols)

### 2.1 일반 정의

$$
\Gamma^k_{ij} = \frac{1}{2} g^{k\ell} \left( \partial_i g_{j\ell} + \partial_j g_{i\ell} - \partial_\ell g_{ij} \right)
$$

### 2.2 대각 근사 (Diagonal Approximation)

Reality Stone은 계산 효율성을 위해 대각 근사를 사용합니다:

$$
g_{ij}(x) = w_i(x) \, \delta_{ij}
$$

이 경우 대부분의 크리스토펠 기호가 0이 되고, 남는 항은:

$$
\Gamma^i_{ii} = \frac{1}{2w_i(x)} \frac{dw_i(x)}{dx_i} = \frac{d}{dx_i}\left(\ln \sqrt{w_i(x)}\right)
$$

### 2.3 각 모델의 크리스토펠 기호

#### 푸앵카레:

$$
\Gamma^k_{ij} = \frac{2c}{1-c\|x\|^2} \left( \delta_{ik} x_j + \delta_{jk} x_i - \delta_{ij} x_k \right)
$$

대각 근사: $\Gamma^i_{ii} = \frac{2c x_i}{1-c\|x\|^2}$

#### 로렌츠:

민코프스키 공간은 평탄하므로 $\Gamma^k_{ij} = 0$

#### 클라인:

대각 근사: $\Gamma^i_{ii} = \frac{c x_i}{1-c\|x\|^2}$

## 3. 측지선 방정식 (Geodesic Equation)

### 3.1 일반 형태

$$
\frac{d^2 x^k}{dt^2} + \Gamma^k_{ij} \frac{dx^i}{dt} \frac{dx^j}{dt} = 0
$$

### 3.2 대각 근사에서의 단순화

대각 메트릭에서:

$$
\frac{d^2 x^i}{dt^2} + \Gamma^i_{ii} \left(\frac{dx^i}{dt}\right)^2 = 0
$$

각 좌표가 독립적으로 진화하므로 $O(n)$ 복잡도로 계산 가능.

### 3.3 수치 적분

Reality Stone은 Velocity Verlet 방법 사용:

```
v_{n+1/2} = v_n + (Δt/2) a_n
x_{n+1} = x_n + Δt v_{n+1/2}
a_{n+1} = -Γ(x_{n+1}) v_{n+1/2}²
v_{n+1} = v_{n+1/2} + (Δt/2) a_{n+1}
```

## 4. 모델 간 변환

### 4.1 푸앵카레 ↔ 로렌츠

**푸앵카레 → 로렌츠**:

$$
\Phi_P(x) = \frac{1}{\sqrt{c}(1-c\|x\|^2)} \begin{pmatrix} 1 + c\|x\|^2 \\ 2x \end{pmatrix}
$$

**로렌츠 → 푸앵카레**:

$$
\Phi_L^{-1}(x_0, x) = \frac{x}{x_0 + 1/\sqrt{c}}
$$

### 4.2 푸앵카레 ↔ 클라인

**푸앵카레 → 클라인**:

$$
\Phi_{PK}(x) = \frac{2x}{1 + c\|x\|^2}
$$

**클라인 → 푸앵카레**:

$$
\Phi_{KP}(x) = \frac{x}{1 + \sqrt{1-c\|x\|^2}}
$$

## 5. 구현 복잡도

| 연산 | 전체 메트릭 | 대각 근사 |
|------|------------|----------|
| 메트릭 계산 | $O(d^2)$ | $O(d)$ |
| 크리스토펠 기호 | $O(d^3)$ | $O(d)$ |
| 측지선 1스텝 | $O(d^3)$ | $O(d)$ |
| 역메트릭 | $O(d^3)$ | $O(d)$ |

**결론**: 대각 근사를 통해 실용적인 속도 달성 ($d=1000$에서도 실시간 가능)

## 6. 사용 지침

### 모델 선택 기준

- **푸앵카레**: 계층 데이터 (트리, 그래프, 언어)
- **로렌츠**: 수치 안정성이 중요한 경우
- **클라인**: 시각화, 직선 경로 필요 시
- **대각**: 메트릭 학습, 일반 목적

### 곡률 선택

- $c = 1.0$: 표준 (단위 곡률)
- $c < 1.0$: 완만한 곡률 (큰 스케일 데이터)
- $c > 1.0$: 강한 곡률 (작은 스케일, 깊은 계층)

## 참고문헌

1. Nickel & Kiela (2017). "Poincaré Embeddings for Learning Hierarchical Representations"
2. Ganea et al. (2018). "Hyperbolic Neural Networks"
3. Chami et al. (2019). "Hyperbolic Graph Convolutional Neural Networks"
4. Do Carmo (1992). "Riemannian Geometry"

