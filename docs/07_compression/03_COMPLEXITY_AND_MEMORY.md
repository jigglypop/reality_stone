# 3장. 복잡도 분석 및 메모리 효율 (Complexity & Memory)

## 1. 메모리 복잡도 (Space Complexity)

Reality Stone v2는 파라미터 수를 레이어 수 $L$에 비례하지 않고, **상수(Constant)에 가깝게** 유지한다.

### 1.1 이론적 모델
*   **Original Transformer**: $O(L \cdot d^2)$
    *   모든 레이어가 독립적인 $d \times d$ 행렬을 가짐.
*   **RS-ULF v2**: $O(d \cdot r + P_{hyper})$
    *   $d \cdot r$: 전역 기저 $U, V$ (레이어 수 $L$과 무관).
    *   $P_{hyper}$: 하이퍼네트워크 크기 (매우 작음, $L$에 대해 천천히 증가).

### 1.2 정량적 비교 (GPT-2 Small, $d=768, L=12$)
| 구성 요소 | Original (FP32) | RS-ULF v2 ($r=64$) | 압축률 |
| :--- | :--- | :--- | :--- |
| Attention ($W_Q, W_K, W_V, W_O$) | $\approx 300$ MB | $\approx 0.4$ MB | **1/750** |
| FFN ($W_1, W_2$) | $\approx 200$ MB | $\approx 0.2$ MB | **1/1000** |
| **Total Weights** | **$\approx 500$ MB** | **$< 1$ MB** | **~1/500** |

> **Note**: 이는 임베딩 레이어($W_{emb}$)를 제외한 수치이다. $W_{emb}$는 어휘 크기($V$)에 비례하므로 별도로 다루거나 동일한 방식으로 압축할 수 있다.

---

## 2. 연산 복잡도 (Time Complexity)

압축된 상태에서 추론을 수행할 때의 연산량(FLOPs) 변화를 분석한다.

### 2.1 행렬 곱 (Matrix Multiplication)
*   **Original**: $x \cdot W$ ($d \times d$ 행렬)
    *   FLOPs: $2 d^2$
*   **RS-ULF v2**: $x \cdot (U C V^\top)$
    *   분해 연산: $(x U) \cdot C \cdot V^\top$
    *   FLOPs: $2dr + 2r^2 + 2rd = 4dr + 2r^2$
*   **Speedup Factor**:
    $$ \frac{2d^2}{4dr} = \frac{d}{2r} $$
    *   $d=768, r=64$일 때, 이론적으로 **약 6배 가속**.
    *   $d=4096, r=64$일 때 (LLaMA급), **약 32배 가속**.

### 2.2 하이퍼네트워크 오버헤드
*   매 레이어마다 코어 행렬 $C^{(l)}$을 생성해야 한다.
*   하지만 이는 입력 토큰 수(Sequence Length)와 무관하게 **레이어당 1회**만 수행하면 되므로, 긴 시퀀스에서는 오버헤드가 0에 수렴한다.

---

## 3. 대역폭 효율 (Bandwidth Efficiency)

현대 GPU 추론의 병목은 연산(Compute)보다 메모리 대역폭(Memory Bandwidth)이다.

*   **Original**: 매 레이어마다 거대한 $W$ 행렬을 HBM에서 로드해야 함. ($500$ MB 이동)
*   **RS-ULF v2**:
    *   $U, V$ (수백 KB)는 **L2 캐시**에 상주시킬 수 있음.
    *   HBM 트래픽이 거의 발생하지 않음 (Compute Bound로 전환).
    *   이는 특히 배치 크기가 작은 실시간 추론(Latency-sensitive)에서 극적인 성능 향상을 가져온다.
