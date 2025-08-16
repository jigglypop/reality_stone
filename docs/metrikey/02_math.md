# 02. Math — SPD Metric and Composition

## SPD 메트릭과 거리
- G_k ≻ 0 (SPD), Mahalanobis: d_k(x,y) = (x−y)^T G_k (x−y)
- 분해: G_k = L_k^T L_k → 전처리 z = L_k x, 이후 유클리드/코사인

## g-직교와 정합성
- R^T g R = g ⇒ 거리/코사인 불변(등거리)
- 블록-직교: g = diag(g_global, g_dept), Q_k = diag(I, R_k)

## 레이어 합성
- 레이어별 T_l = g(l)^{1/2}, 순서보존 합성 T_total = T_n … T_1
- 무손실 합성 저장 시 파라미터 압축률≈L(레이어 수)

## 지오데식
- 상수 곡률: Poincaré/Lorentz 폐형식 거리
- 일반 리만: log_p(x)로 접공간 근사 + 상위 M 재랭크
