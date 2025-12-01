# 3장. 복잡도 분석 및 메모리 수식 (Complexity & Memory Analysis)

## 1. 서론

본 장에서는 리만 지오데식 압축 아키텍처(RS-ULF)의 시간 복잡도, 공간 복잡도, 그리고 메모리 사용량을 기존 트랜스포머와 정량적으로 비교 분석한다. 모든 수식은 시퀀스 길이 $N$, 모델 차원 $D$, 압축 랭크 $R$, 레이어 수 $L$을 기준으로 한다.

## 2. 시간 복잡도 분석 (Time Complexity)

### 2.1 트랜스포머 (Baseline)
*   **Attention**: $Q K^\top$ 연산은 $O(N^2 D)$.
*   **FFN**: $O(N D^2)$.
*   **Total per Layer**: $O(N^2 D + N D^2)$.
    *   $N$이 커질수록 $N^2$ 항이 지배적이다.

### 2.2 RS-ULF (Geodesic Compression)
*   **Metric Update**:
    *   메트릭 $g$는 $R \times R$ 코어 행렬이므로, 벡터-메트릭 곱은 $O(N D R)$.
    *   $R \ll D$ (예: $R=128, D=4096$) 이므로, 이는 $O(N D)$에 가깝다.
*   **Attention 대체 (Graph Diffusion)**:
    *   그래프 라플라시안 $L$은 희소 행렬(Sparse Matrix) 혹은 윈도우 기반이므로 $O(N \cdot \text{window})$.
    *   전체 상호작용: $O(N D)$.
*   **FFN 대체 (Potential Force)**:
    *   압축된 포텐셜 미분 계산: $O(N D R)$.
*   **Total per Layer**: $O(N D R)$.
    *   **결론**: $N^2$ 항이 완전히 제거되어 $O(N)$ 선형 복잡도를 달성한다.

### 2.3 이론적 가속비 (Theoretical Speedup)
$$ \text{Speedup} = \frac{N^2 D + N D^2}{N D R} = \frac{N + D}{R} $$
$N=4096, D=4096, R=128$일 때, 약 **64배**의 이론적 연산량 감소가 발생한다.

## 3. 공간 복잡도 및 메모리 수식 (Space & Memory)

### 3.1 파라미터 메모리 (Model Weights)
*   **Transformer**:
    *   $W_Q, W_K, W_V, W_O$: $4 \times D^2$.
    *   FFN ($W_1, W_2, W_3$): $3 \times D \times 4D = 12 D^2$.
    *   Total: $\approx 16 D^2$ per layer.
*   **RS-ULF**:
    *   Global Basis $U$: $D \times R$ (전체 공유, 1회).
    *   Layer Metric/Gauge Core: $2 \times R^2$ (레이어별).
    *   Layer Potential Core: $2 \times R^2$ (레이어별).
    *   Total per Layer: $4 R^2$.
*   **압축률 (Compression Ratio)**:
    $$ \text{CR}_{param} = \frac{16 D^2}{4 R^2} = 4 \left( \frac{D}{R} \right)^2 $$
    $D=4096, R=128$일 때, 레이어당 파라미터는 약 **4000배** 감소한다. (단, Global Basis가 전체 메모리 차지)

### 3.2 KV 캐시 메모리 (Runtime Memory)
*   **Transformer (KV Cache)**:
    *   각 토큰마다 $K, V$ 벡터 저장.
    *   Memory: $2 \times N \times D \times L \times \text{Batch}$.
*   **RS-ULF (Geodesic Spline)**:
    *   제어점(Control Point)만 저장. 제어점 비율 $\rho \approx 0.1$ (10%만 저장).
    *   저장 항목: 위치 $x$ + 속도 $v$ = $2D$.
    *   Memory: $2 \times (\rho N) \times D \times L \times \text{Batch}$.
*   **압축률**:
    $$ \text{CR}_{cache} = \frac{N}{\rho N} = \frac{1}{\rho} \approx 10 $$
    KV 캐시 메모리를 **10배** 절약한다.

## 4. 메모리 대역폭 요구량 (Memory Bandwidth)

*   **Transformer**: 매 토큰 생성 시 전체 파라미터와 KV 캐시를 로드해야 함. 대역폭 병목(Memory Wall)이 주요 이슈.
*   **RS-ULF**:
    *   Global Basis는 칩 내 SRAM/L2 캐시에 상주 가능 ($D \times R \approx 1MB$).
    *   레이어별 코어 행렬($R^2$)은 극도로 작음.
    *   따라서 메인 메모리(HBM) 접근을 획기적으로 줄일 수 있음.

## 5. 요약 테이블

| 항목 | Transformer | RS-ULF (Geodesic) | 개선 효과 |
| :--- | :--- | :--- | :--- |
| **Time Complexity** | $O(N^2 D)$ | $O(N D R)$ | **Linear ($N$)** |
| **Weights (Layer)** | $16 D^2$ | $4 R^2$ | **~1000x (Layer-wise)** |
| **KV Cache** | $2 N D L$ | $0.2 N D L$ | **10x** |
| **Operation Type** | MatMul (Dense) | Metric-Vector (Structured) | **Bandwidth Efficient** |

이 분석은 RS-ULF 아키텍처가 단순한 용량 압축을 넘어, **연산 효율성과 메모리 대역폭 문제까지 근본적으로 해결**함을 보여준다.

