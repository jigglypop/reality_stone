# 09-07. Fold Consistency and Mathematical Foundations

> 이 문서는 RS-ULF의 **폴드 정합성(Fold Consistency)**, **메트릭 정확성**, 
> **곡률 해석**에 대한 수학적 기초를 정의한다.

---

## 1. Metric Tensor Definition

### 1.1 Raw Metric

Transformer의 Q, K 가중치로부터 유도되는 원시 메트릭:

$$
G_{\text{raw}} = W_Q^\top W_K \in \mathbb{R}^{d \times d}
$$

### 1.2 Symmetry Issue

**문제**: $G_{\text{raw}}$는 일반적으로 비대칭 행렬이다.

$$
G_{\text{raw}} \neq G_{\text{raw}}^\top
$$

리만 메트릭은 대칭이어야 하므로, 대칭화가 필요할 수 있다:

$$
G_{\text{sym}} = \frac{1}{2}(G_{\text{raw}} + G_{\text{raw}}^\top)
$$

**현재 구현**: RS-ULF는 대칭화 없이 $G_{\text{raw}}$를 직접 사용한다.
이는 Q-K 내적 구조를 그대로 보존하기 위함이며, 
SVD 기반 저랭크 근사에서 대칭성이 자연스럽게 확보된다.

### 1.3 Positive Definiteness

양정치(Positive Definite) 조건:

$$
x^\top G x > 0 \quad \forall x \neq 0
$$

**정규화 전략**:

$$
g_{ii} = \max(G_{ii}, \epsilon), \quad \epsilon = 10^{-6}
$$

---

## 2. SVD-based Dimension Folding

### 2.1 Low-rank Approximation

원시 메트릭 $G$에 대한 rank-$r$ SVD 근사:

$$
G \approx U_r \Sigma_r V_r^\top
$$

여기서:
- $U_r \in \mathbb{R}^{d \times r}$: 좌측 특이벡터
- $\Sigma_r = \text{diag}(\sigma_1, \ldots, \sigma_r)$: 상위 $r$개 특이값
- $V_r \in \mathbb{R}^{d \times r}$: 우측 특이벡터

### 2.2 Fold Accuracy

폴드 정확도(Fold Accuracy)는 저랭크 근사의 품질을 측정한다:

$$
\text{Accuracy} = \frac{\sum_{i=1}^{r} \sigma_i^2}{\|G\|_F^2} = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{d} \sigma_i^2}
$$

**권장 조건**:

$$
\text{Accuracy} \geq 0.90
$$

### 2.3 Reconstruction Error

복원 오차:

$$
\epsilon_{\text{recon}} = \frac{\|G - U_r \Sigma_r V_r^\top\|_F}{\|G\|_F}
$$

이는 폴드 정확도와 다음 관계를 가진다:

$$
\epsilon_{\text{recon}} = \sqrt{1 - \text{Accuracy}}
$$

---

## 3. Error-Curvature Interpretation

### 3.1 Residual Singular Values

SVD에서 버려진 특이값 집합:

$$
S_{\text{residual}} = \{\sigma_{r+1}, \sigma_{r+2}, \ldots, \sigma_d\}
$$

### 3.2 Curvature Definition

잔차 곡률(Residual Curvature):

$$
\kappa = \|S_{\text{residual}}\|_2 = \sqrt{\sum_{i=r+1}^{d} \sigma_i^2}
$$

**물리적 해석**: 

잔차 곡률은 저차원 폴딩에서 손실된 정보량을 나타내며,
리만 다양체의 **측지선 편차(Geodesic Deviation)**와 대응한다.

측지선 편차 방정식:

$$
\frac{D^2 \xi^\mu}{d\tau^2} = -R^\mu_{\nu\rho\sigma} u^\nu u^\rho \xi^\sigma
$$

RS-ULF에서의 근사:

$$
\delta = -\frac{1}{2} \kappa \|v\|^2 x
$$

### 3.3 Curvature Correction in Update

전체 업데이트 규칙:

$$
x_{t+1} = x_t + v + \delta
$$

여기서:

$$
v = -\eta \cdot g^{-1} \nabla\Phi(x) + \alpha(x - \bar{x}) + \beta L x
$$

$$
\delta_i = -\frac{1}{2} \kappa \|v_i\|^2 \cdot x_i
$$

---

## 4. Diagonal Metric Consistency

### 4.1 Column-wise Inner Product

대각 메트릭 계산:

$$
g_{ii}^{\text{diag}} = W_Q[:, i]^\top \cdot W_K[:, i]
$$

**코드 구현** (`rsulf.rs:503-506`):
```rust
for i in 0..d {
    let col_q = wq.column(i);
    let col_k = wk.column(i);
    g_diag[i] = col_q.dot(&col_k).abs();
}
```

### 4.2 Sign Handling

**문제**: 내적 $W_Q[:, i]^\top W_K[:, i]$는 음수일 수 있다.

**현재 전략**: 절대값 취함 (`abs()`)

**대안적 해석**:

$$
g_{ii} = \|W_Q[:, i]\|_2 \cdot \|W_K[:, i]\|_2 \cdot |\cos\theta_i|
$$

여기서 $\theta_i$는 두 컬럼 벡터 사이의 각도.

### 4.3 Regularization

수치 안정성을 위한 클리핑:

$$
g_{ii} = \text{clip}(g_{ii}, 10^{-6}, 10^{6})
$$

역메트릭:

$$
g^{-1}_{ii} = \frac{1}{g_{ii}}
$$

---

## 5. Potential Function Definition

### 5.1 FFN-based Potential

Transformer FFN:

$$
f(x) = W_2 \cdot \sigma(W_1 x)
$$

RS-ULF Potential:

$$
\Phi(x) = \frac{1}{2} \|f(x)\|^2
$$

### 5.2 Gradient Computation

이론적 gradient:

$$
\nabla_x \Phi(x) = J_f(x)^\top f(x)
$$

여기서 $J_f(x)$는 FFN의 Jacobian.

**SwiGLU 활성화 함수의 도함수**:

$$
\sigma(x) = x \cdot \text{sigmoid}(x)
$$

$$
\sigma'(x) = \text{sigmoid}(x) + x \cdot \text{sigmoid}(x) \cdot (1 - \text{sigmoid}(x))
$$

### 5.3 Riemannian Gradient

리만 gradient:

$$
\nabla_g \Phi(x) = g^{-1} \nabla \Phi(x)
$$

대각 메트릭의 경우:

$$
(\nabla_g \Phi)_i = \frac{(\nabla \Phi)_i}{g_{ii}}
$$

---

## 6. Consistency Conditions

### 6.1 Metric Consistency

| 조건 | 수식 | 임계값 |
|------|------|--------|
| 대칭성 오차 | $\|G - G^\top\| / \|G\|$ | < 0.3 |
| 폴드 정확도 | $\sum_{i=1}^{r} \sigma_i^2 / \|G\|_F^2$ | $\geq 0.90$ |
| 조건수 | $\sigma_1 / \sigma_r$ | < $10^6$ |

### 6.2 Gradient Consistency

FFN 출력과 Potential gradient의 정합:

$$
\cos(f(x), \nabla\Phi(x)) > 0.9
$$

### 6.3 Update Consistency

Transformer residual과 RS-ULF geodesic step의 정합:

$$
\cos(f(x), -\eta g^{-1}\nabla\Phi(x)) > 0.9
$$

---

## 7. Hyperparameter Ranges

### 7.1 Fold Parameters

| 파라미터 | 의미 | 권장 범위 | 기본값 |
|---------|------|-----------|--------|
| $r$ | target rank | $d/8 \sim d/2$ | 1024 |
| $n_{\text{oversample}}$ | SVD oversampling | 5-10 | 5 |
| $n_{\text{iter}}$ | power iteration | 1-3 | 1 |

### 7.2 Dynamics Parameters

| 파라미터 | 의미 | 권장 범위 | 기본값 |
|---------|------|-----------|--------|
| $\eta$ | gradient step size | 0.001 ~ 0.05 | 0.01 |
| $\alpha$ | diffusion coefficient | 0.001 ~ 0.05 | 0.02 |
| $\beta$ | graph Laplacian weight | 0 ~ 0.1 | 0.01 |
| $\gamma$ | Bellman discount | 0.9 ~ 0.999 | 0.99 |

### 7.3 Curvature Threshold

곡률 보정 적용 조건:

$$
|\kappa| > 10^{-6}
$$

---

## 8. Complexity-Curvature Correspondence

### 8.1 SFE Theory Connection

SFE(Suppression Field) 이론에서의 억압장 포텐셜:

$$
\Phi_{\text{supp}} = -\log P_{\text{selected}}
$$

RS-ULF에서의 대응:

$$
\Phi(x) = \frac{1}{2}\|f(x)\|^2 \approx \text{complexity density}
$$

### 8.2 Complexity-Curvature Mapping

$$
K(x) \xrightarrow{\text{fold}} \kappa \xrightarrow{\text{correction}} \delta
$$

- **Complexity** $K(x)$: 원본 메트릭의 정보량
- **Curvature** $\kappa$: SVD 잔차로부터 유도된 곡률
- **Correction** $\delta$: 2차 기하학적 보정항

### 8.3 Universal Stability Principle

RS-ULF의 곡률 억제 구조는 다양한 복잡계에서 공통으로 나타나는
안정화 원리와 대응한다:

$$
\mathcal{L} = |\nabla\phi|^2 + \lambda|\nabla^2\phi|^2
$$

- 1차 항: gradient 안정화
- 2차 항: curvature 안정화

---

## 9. Verification Checklist

### 9.1 Metric Verification

- [ ] $g_{\text{raw}} = W_Q^\top W_K$ 계산 확인
- [ ] 대각 요소가 모두 양수인지 확인
- [ ] 조건수가 $10^6$ 이하인지 확인
- [ ] $g^{-1}$ 계산 시 NaN/inf 없음

### 9.2 Fold Verification

- [ ] SVD 분해 정확성 확인
- [ ] 폴드 정확도 $\geq 0.90$
- [ ] 잔차 곡률 $\kappa$ 계산 확인

### 9.3 Dynamics Verification

- [ ] $\Phi(x_{t+1}) < \Phi(x_t)$ 비율 > 90%
- [ ] gradient 방향 정합성 확인
- [ ] 곡률 보정항 적용 확인

---

## 10. References

- `02_METRIC_AND_POTENTIAL.md`: Metric 추출 및 Potential 정의
- `01_RS_UNIFIED_FLOW_SPEC.md`: RS-ULF 전체 스펙
- `06_TRANSFORMER_TO_RSULF_CONVERSION.md`: Transformer 변환 규칙
- `Derivations_Applications/05_Neural_RealityStone_Derivation.md`: SFE 이론 연결

