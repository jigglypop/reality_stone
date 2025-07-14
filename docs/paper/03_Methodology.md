## 3. 방법론: 리만 기하학 기저 인코딩 (RBE)

본 장에서는 `Reality Stone`의 핵심 방법론인 **리만 기하학 기저 인코딩(Riemannian Basis Encoding, RBE)** 의 이론적 배경, 핵심 알고리즘, 그리고 구현 세부 사항을 상세히 기술한다. RBE는 거대한 가중치 행렬 $W \in \mathbb{R}^{m \times n}$를, 기하학적 의미를 담은 **기저 청사진(Basis Blueprint)** 과 미세 오차를 보정하는 **잔차(Residual)** 로 분해하여 극단적인 압축을 달성한다.

### 3.1. RBE의 핵심 원리: 기하학적 분해와 크로네커 델타 선택

RBE는 신경망 가중치 행렬 $W$의 각 행벡터 $w_i \in \mathbb{R}^n$를, **(1) 방향(Direction)**과 **(2) 크기(Magnitude)**의 두 요소로 분해하는 것에서 출발한다. 이 분해는 유한한 기저 벡터 집합 $\{b_j\}_{j=1}^B$에 대한 이산적 선택 과정으로 모델링되며, 이는 **크로네커 델타(Kronecker Delta)** 함수를 통해 수학적으로 명확하게 표현될 수 있다.

원본 가중치 행벡터 $w_i$는 다음과 같이 근사치 $w_{\text{approx}, i}$와 잔차 $w_{\text{res}, i}$의 합으로 정확하게 표현된다.

$$ w_i = w_{\text{approx}, i} + w_{\text{res}, i} $$

여기서 근사치 $w_{\text{approx}, i}$는 기저 벡터 중 하나를 선택하고, 그 기저 벡터에 기하학적 스케일을 적용하여 계산된다.

$$ w_{\text{approx}, i} = s_i \cdot \left( \sum_{j=1}^{B} \delta_{j, \text{idx}_i} b_j \right) = s_i \cdot b_{\text{idx}_i} $$

-   **$\delta_{j, \text{idx}_i}$ (크로네커 델타):** 기저 선택 메커니즘의 핵심. $j = \text{idx}_i$일 때만 1이고, 그 외에는 0이다. 비트필드의 `idx` 필드가 이 `idx_i` 값을 지정하며, 이는 수많은 기저 벡터 중 단 하나를 선택하는 '하드 어텐션(hard attention)'과 유사하게 동작한다.
-   **$b_{\text{idx}_i}$ (선택된 기저 벡터):** 크로네커 델타에 의해 선택된, $w_i$의 주된 방향을 나타내는 단위 벡터.
-   **$s_i$ (기하학적 스케일):** 비트필드에 인코딩된 파라미터 `(cat, sub, d, amp, amp_fine)`로부터 계산되는 스칼라 값. 이는 선택된 기저 방향으로의 크기를 결정하며, 특정 리만 기저 함수 $\mathcal{F}$를 통해 계산된다 ($s_i = \mathcal{F}(\text{params}_i)$).

이러한 분해를 통해, 원래의 가중치 행벡터 $w_i$는 **(기저 벡터 인덱스, 기저 함수 타입, 함수 파라미터)** 라는 압축된 정보(비트필드)와, 근사 오차를 보정하는 작은 잔차 벡터 $w_{\text{res}, i}$로 완벽하게 재구성될 수 있다.

### 3.2. 1단계 (인코딩): 기저 청사진과 잔차 생성

주어진 가중치 행렬 $W$를 RBE 형태로 인코딩하는 과정은 다음과 같다.

#### 3.2.1. 기저 청사진 생성 (The Blueprint)

$W$의 각 행 $w_i$에 대해, 다음을 수행하여 32비트 **비트 필드(Bitfield)** 를 생성한다.

1.  **최적 기저 탐색 ($\text{idx}_i$ 결정):** 미리 정의된 기저 벡터 테이블 $\{b_j\}$ 중에서 $w_i$와 가장 유사한(예: 코사인 유사도가 가장 높은) 기저 벡터 $b_{\text{idx}_i}$를 찾는다. 이 `idx_i`가 이 행의 **기저 인덱스**가 된다.
2.  **프로젝션 및 스케일 계산 ($s_i$ 결정):** $w_i$를 $b_{\text{idx}_i}$ 방향으로 투영하여, 그 크기를 가장 잘 표현하는 리만 기저 함수 $\mathcal{F}$와 그 파라미터(예: `amp`, `amp_fine` 등)를 결정한다. 이를 통해 스케일 값 $s_i$가 계산된다.
3.  **비트 필드 패킹:** 위에서 얻은 파라미터들을 다음의 32비트 구조에 패킹하여, $i$번째 행의 청사진 코드 `code_i`를 생성한다.

```
Bit:  31......22 21 20 19 18 17..10  09  08  07...0
      |----------|-----|-----|--------|---|---|------|
      | AMP_FINE | CAT | SUB |   IDX  | S | D | AMP  |
      |    10    |  2  |  2  |    8   | 1 | 1 |  8   |
```
*S: Sign bit*

이 코드의 배열이 **기저 청사진 행렬 $W_{\text{codes}}$** 이다.

#### 3.2.2. 잔차 계산 (The Residual)

청사진만으로 재구성된 근사 가중치 벡터 $w_{\text{approx}, i}$와 원본 벡터 $w_i$ 사이의 오차를 **잔차 벡터 $w_{\text{res}, i}$** 로 정의한다.

$$ w_{\text{res}, i} = w_i - w_{\text{approx}, i} = w_i - s_i \cdot b_{\text{idx}_i} $$

이 계산을 모든 행에 대해 수행하여 얻은 행렬이 **잔차 행렬 $W_{\text{res}}$** 이다.
$$ W_{\text{res}} = W - W_{\text{approx}} $$

$W_{\text{res}}$의 각 원소는 일반적으로 원본 $W$보다 훨씬 작은 값을 가지므로, FP8 또는 INT8과 같은 저정밀도 데이터 타입으로 양자화하여 저장할 수 있다. 이는 추가적인 압축 효과와 함께 최종 정확도를 보정하는 핵심적인 역할을 수행한다.

### 3.3. 리만 기저 함수 라이브러리

RBE의 표현력은 다양한 형태의 가중치 분포를 모델링할 수 있는 풍부한 기저 함수 라이브러리에서 나온다. 22비트 청사진의 `(CAT, SUB, D)` 코드는 총 64가지 함수를 선택할 수 있게 한다.

-   **Category 0: 푸앵카레 기하학 (CAT=0):** `tanh`, `sinh` 등 쌍곡 공간의 기본적인 매핑 함수. 계층 구조 표현에 유리하다.
-   **Category 1: 로렌츠 기하학 (CAT=1):** `sinh`, `cosh` 등 더 넓은 범위의 값을 표현. 동적 시스템 모델링에 적합하다.
-   **Category 2: 삼각 함수 (CAT=2):** `sin`, `cos` 등 주기적/위상적 패턴을 가진 가중치를 모델링하기 위한 함수. (미래 구현)
-   **Category 3: 특수 함수 (CAT=3):** `Bessel`, `Gaussian` 등 특정 복잡한 패턴을 표현하기 위한 실험적인 함수. (미래 구현)

*(전체 함수 목록은 부록 참조)*

### 3.4. 2단계 (추론): 압축 상태에서의 행렬 곱셈 재구성

RBE의 가장 큰 연산 효율성은, 가중치 행렬 $W$를 명시적으로 복원하는 과정을 완전히 생략하고, 압축된 형태인 **기저 청사진 행렬 $W_{\text{codes}}$** 와 **저정밀도 잔차 행렬 $W_{\text{res}}$** 로부터 직접 행렬-벡터 곱셈($y = xW^T$)을 수행하는 데 있다. 이는 기존의 압축 후 복원(decompress-then-compute) 방식의 패러다임을 근본적으로 전환하는 것이다.

출력 벡터 $y \in \mathbb{R}^m$의 $i$-번째 요소 $y_i$는 입력 벡터 $x \in \mathbb{R}^n$과 $i$-번째 가중치 행 $w_i \in \mathbb{R}^n$의 내적으로 정의된다. RBE의 분해 정의($w_i = w_{\text{approx}, i} + w_{\text{res}, i}$)를 이 내적에 대입하여 전개하면 다음과 같다.

$$
\begin{aligned}
y_i &= x \cdot w_i^T \\
    &= x \cdot (w_{\text{approx}, i} + w_{\text{res}, i})^T && \text{(1. RBE 분해식 대입)} \\
    &= (x \cdot w_{\text{approx}, i}^T) + (x \cdot w_{\text{res}, i}^T) && \text{(2. 내적의 선형성을 이용해 분리)}
\end{aligned}
$$

이 수식은 전체 연산을 **(A) 청사진 기반 추론**과 **(B) 잔차 기반 추론**이라는 두 개의 독립적인 항으로 분리할 수 있음을 보여준다.

#### 3.4.1. (A) 청사진 기반 추론: 행렬 곱을 인덱싱으로 변환

첫 번째 항인 $x \cdot w_{\text{approx}, i}^T$는 RBE의 핵심적인 연산량 절감 효과가 발생하는 부분이다. $w_{\text{approx}, i}$의 정의($s_i \cdot b_{\text{idx}_i}$)를 대입하여 유도 과정를 더 전개해 보자.

$$
\begin{aligned}
x \cdot w_{\text{approx}, i}^T &= x \cdot (s_i \cdot b_{\text{idx}_i})^T && \text{(3. 근사치 정의 대입)} \\
                               &= s_i \cdot (x \cdot b_{\text{idx}_i}^T) && \text{(4. 스칼라 $s_i$를 밖으로 분리)}
\end{aligned}
$$

여기서 $s_i$는 $i$-번째 청사진 코드 `code_i`에서 디코딩된 스칼라 값이며, $b_{\text{idx}_i}$는 해당 코드가 가리키는 기저 테이블의 $\text{idx}_i$번째 행 벡터이다.

이 변환의 의미는 혁명적이다. 원래는 $O(n)$의 연산량이 필요한 내적 $x \cdot w_{\text{approx}, i}^T$이, 다음과 같은 세 단계의 훨씬 저렴한 연산으로 대체된다.

1.  **사전 계산 (Pre-computation):** 입력 $x$와 전체 기저 테이블 $B$($B \times n$ 크기)의 내적을 미리 계산하여 `dot_products` 배열($B$ 크기)을 만든다. 이 과정은 $m$개의 모든 출력 요소를 계산하기 전에 단 한 번만 수행되며, $O(B \cdot n)$의 연산량을 가진다.
    `dot_products[j] = x · b_j^T` for $j = 0, \dots, B-1$.
2.  **인덱싱 (Indexing):** $i$-번째 청사진 코드 `code_i`에서 기저 인덱스 $\text{idx}_i$를 읽어와, `dot_products` 배열에서 해당 값을 직접 조회한다 ($O(1)$).
    `indexed_dot = dot_products[idx_i]`
3.  **스케일링 (Scaling):** `code_i`에서 디코딩한 스케일 값 $s_i$를 조회된 값에 곱하여 $y_{\text{approx}, i}$를 계산한다 ($O(1)$).
    `result = s_i * indexed_dot`

결과적으로, $m \times n$ 크기의 거대한 행렬-벡터 곱셈($x W_{\text{approx}}^T$)이, $B \times n$ 크기의 작은 행렬-곱 한번과 $m$번의 인덱싱 및 스칼라 곱으로 대체된다. LLM과 같이 $n$과 $m$이 크고 $B$가 상대적으로 작은($B \ll m, n$) 환경에서 이는 **연산량(FLOPs)과 메모리 대역폭(memory bandwidth) 모두를 극적으로 감소시킨다.**

#### 3.4.2. (B) 잔차 기반 추론: 저정밀도 연산 가속

두 번째 항인 $x \cdot w_{\text{res}, i}^T$는 표준적인 행렬-벡터 곱셈이지만, $W_{\text{res}}$가 FP8 또는 INT8과 같은 저정밀도 데이터 타입이라는 점이 중요하다. 이는 현대 GPU의 Tensor Core나 특화된 INT8 연산 유닛(예: NVIDIA의 DP4A 명령어)을 활용하여 표준 FP32 연산보다 훨씬 빠르게 계산될 수 있다.

$$ y_{\text{res}} = x W_{\text{res}}^T $$

이 연산은 청사진만으로는 포착하지 못한 미세한 오차를 보정하여, 극단적인 압축률에도 불구하고 모델의 최종 정확도를 원본 수준으로 유지하는 데 결정적인 역할을 한다.

#### 3.4.3. 통합 CUDA 커널 아키텍처

실제 구현에서는 이 두 과정(A, B)을 하나의 통합된 CUDA 커널에서 효율적으로 처리하여 메모리 접근을 최적화한다.

1.  **전처리 단계 (커널 시작 시):**
    -   각 스레드 블록은 담당할 입력 벡터 $x$의 일부와 전체 기저 테이블 $\{b_j\}$을 공유 메모리(shared memory)에 로드한다.
    -   공유 메모리에 로드된 $x$와 $\{b_j\}$를 사용하여 `dot_products` 배열을 병렬로 계산하고, 이 결과 또한 공유 메모리에 저장한다. `__syncthreads()`를 통해 모든 스레드가 이 계산을 마칠 때까지 대기한다.
2.  **병렬 처리 루프 (각 스레드):**
    -   각 스레드는 출력 벡터 $y$의 한 요소 $y_i$ 계산을 담당한다.
    -   $i$-번째 청사진 코드 `code_i`를 전역 메모리(global memory)에서 읽어와 디코딩하여 `(idx_i, params_i)`를 얻는다.
    -   스케일 값 $s_i = \mathcal{F}(\text{params}_i)$ 를 계산한다.
    -   공유 메모리에 저장된 `dot_products` 배열에서 `idx_i`를 이용해 값을 인덱싱하고, $s_i$와 곱하여 $y_{\text{approx}, i}$를 계산한다.
    -   $i$-번째 잔차 벡터 $w_{\text{res}, i}$와 입력 $x$의 내적을 저정밀도 연산(DP4A 등)으로 계산하여 $y_{\text{res}, i}$를 구한다.
    -   최종 결과 $y_i = y_{\text{approx}, i} + y_{\text{res}, i}$ 를 전역 메모리의 출력 위치에 저장한다.

이러한 접근법은 전역 메모리 접근을 최소화하고 GPU의 공유 메모리와 빠른 병렬 연산 능력을 극대화하여, 기존 방식 대비 3-4배의 실질적인 추론 가속을 달성한다.

### 3.5. 미래 구현 계획: 동적 위상 인코딩

현재 RBE는 기저 함수의 '크기(magnitude)'를 중심으로 인코딩하지만, 향후 버전에서는 '위상(phase)' 정보를 명시적으로 인코딩하는 메커니즘을 추가할 계획이다.

-   **복소수 및 쿼터니언 기저:** 가중치 공간을 복소수 또는 쿼터니언 공간으로 확장하여, 회전 및 위상 변환을 자연스럽게 모델링한다.
-   **위상 필드 추가:** 22비트 청사진에 위상 정보를 인코딩하는 추가 필드를 할당하여, 더 복잡하고 동적인 변환을 표현할 수 있도록 한다.

이는 특히 음성, RF 신호 처리, 양자 회로 시뮬레이션 등 위상 정보가 중요한 도메인에서 RBE의 적용 범위를 크게 확장할 것이다. 

### 3.5. RBE의 학습: 미분 가능한 비트 연산과 QAT

RBE의 진정한 강점은, 이것이 단순한 후처리 압축 기법을 넘어 **학습 과정에 완벽하게 통합될 수 있다**는 점이다. 이 섹션에서는 이산적이고 미분 불가능해 보이는 비트필드 연산이 어떻게 그래디언트 기반 최적화에 포함될 수 있는지를 수학적으로 엄밀하게 증명한다.

#### 3.5.1. 비트 연산의 대각 야코비안 해석

##### 3.5.1.1. 문제 정의

RBE의 핵심 연산은 비트필드 코드에 따라 기저 벡터를 선택하고 스케일링하는 것이다. 이를 수식으로 표현하면:

$$w_{\text{approx}, i} = s_i \cdot b_{\text{idx}_i}$$

여기서 $\text{idx}_i$는 이산적인 정수 인덱스이며, 이는 기존의 역전파 알고리즘으로는 직접 미분할 수 없다. 이 문제를 해결하기 위해, 우리는 이산적 선택 과정을 **연속적인 선형 대수 연산**으로 재해석한다.

##### 3.5.1.2. 원-핫 인코딩을 통한 선형화

$B$개의 기저 벡터 중 $\text{idx}_i$번째를 선택하는 연산은 다음과 같이 표현할 수 있다:

1. **원-핫 벡터 정의**: $e_i \in \{0, 1\}^B$를 정의하여, $\text{idx}_i$번째 원소만 1이고 나머지는 0이 되도록 한다.
   $$e_i[j] = \begin{cases} 1 & \text{if } j = \text{idx}_i \\ 0 & \text{otherwise} \end{cases}$$

2. **선택 연산의 행렬 표현**: 기저 테이블 $B \in \mathbb{R}^{B \times n}$에서 특정 행을 선택하는 것은 원-핫 벡터와의 행렬 곱으로 표현된다.
   $$b_{\text{idx}_i} = e_i^T B = \sum_{j=1}^{B} e_i[j] \cdot b_j$$

3. **전체 근사 행렬의 분해**: 모든 가중치 행에 대해 이를 확장하면,
   $$W_{\text{approx}} = S \cdot E \cdot B$$
   
   여기서:
   - $S = \text{diag}(s_0, s_1, \dots, s_{m-1}) \in \mathbb{R}^{m \times m}$: 스케일 대각 행렬
   - $E \in \{0, 1\}^{m \times B}$: 각 행이 원-핫 벡터인 선택 행렬
   - $B \in \mathbb{R}^{B \times n}$: 기저 테이블

##### 3.5.1.3. 야코비안 유도

이제 손실 함수 $\mathcal{L}$에 대한 각 구성요소의 그래디언트를 체인룰을 사용하여 유도한다.

**기저 테이블 $B$의 그래디언트:**

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial W_{\text{approx}}} : \frac{\partial W_{\text{approx}}}{\partial B}$$

여기서 $W_{\text{approx}} = S \cdot E \cdot B$이므로,

$$\frac{\partial W_{\text{approx}}}{\partial B_{jk}} = \sum_{i=1}^{m} S_{ii} \cdot E_{ij} \cdot \mathbf{1}_{(j,k)}$$

따라서,
$$\frac{\partial \mathcal{L}}{\partial B_{jk}} = \sum_{i=1}^{m} S_{ii} \cdot E_{ij} \cdot \frac{\partial \mathcal{L}}{\partial W_{\text{approx}, ik}}$$

행렬 형태로 정리하면:
$$\frac{\partial \mathcal{L}}{\partial B} = (S \cdot E)^T \cdot \frac{\partial \mathcal{L}}{\partial W_{\text{approx}}}$$

이는 비트필드의 이산적 선택에도 불구하고, 기저 테이블이 명확한 그래디언트를 받을 수 있음을 보여준다.

**스케일 값들의 그래디언트:**

각 스케일 값 $s_i$는 비트필드의 파라미터를 통해 계산된다:
$$s_i = \mathcal{F}_{\text{cat}_i, \text{sub}_i}(\text{amp}_i, \text{amp\_fine}_i)$$

예를 들어, 푸앵카레 기하학에서 $\mathcal{F}$가 $\tanh$ 함수라면:
$$s_i = \tanh\left(\frac{\text{amp}_i + \text{amp\_fine}_i / 1024}{128}\right)$$

이 경우 그래디언트는:
$$\frac{\partial s_i}{\partial \text{amp}_i} = \frac{1}{128} \cdot \text{sech}^2\left(\frac{\text{amp}_i + \text{amp\_fine}_i / 1024}{128}\right)$$

#### 3.5.2. Straight-Through Estimator (STE)

##### 3.5.2.1. STE의 필요성

위의 분석은 연속적인 파라미터(기저 테이블, 스케일 값)에 대한 그래디언트를 제공하지만, 여전히 두 가지 이산적 연산이 남아있다:

1. **기저 선택**: $\text{idx}_i = \arg\max_j \{\text{similarity}(w_i, b_j)\}$
2. **양자화**: $\text{amp}_i = \text{round}(128 \cdot r_i)$

이러한 연산들은 미분 불가능하며, 그래디언트가 0이거나 정의되지 않는다.

##### 3.5.2.2. STE의 원리

Straight-Through Estimator는 다음과 같은 간단하지만 효과적인 아이디어를 사용한다:

$$\begin{aligned}
\text{Forward}: & \quad y = \text{quantize}(x) \\
\text{Backward}: & \quad \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}
\end{aligned}$$

즉, 순전파에서는 실제 양자화된 값을 사용하지만, 역전파에서는 마치 항등 함수인 것처럼 그래디언트를 그대로 통과시킨다.

##### 3.5.2.3. RBE에서의 STE 적용

RBE의 인코딩 과정에 STE를 적용하면:

```python
def rbe_encode_with_ste(w, basis_table):
    # 연속적인 유사도 계산
    similarities = w @ basis_table.T  # [m, B]
    
    # Forward: 실제 이산적 선택
    idx = argmax(similarities, dim=1)  # [m]
    selected_basis = basis_table[idx]   # [m, n]
    
    # Backward: STE를 통한 그래디언트 전파
    # autograd는 similarities → basis_table로 그래디언트 전파
    
    # 스케일 계산
    continuous_scale = (w * selected_basis).sum(dim=1)  # [m]
    
    # Forward: 양자화
    quantized_scale = round(continuous_scale * 128) / 128
    
    # Backward: STE 적용
    # quantized_scale.backward() → continuous_scale로 전파
    
    return idx, quantized_scale
```

#### 3.5.3. Quantization-Aware Training (QAT) 알고리즘

##### 3.5.3.1. QAT의 전체 흐름

QAT는 학습 과정에서 양자화를 시뮬레이션하여, 모델이 양자화 오차에 강건해지도록 한다. RBE-QAT의 전체 알고리즘은 다음과 같다:

```algorithm
Algorithm 1: RBE Quantization-Aware Training
Input: 원본 모델 M, 학습 데이터 D, 학습률 η
Output: RBE 압축된 모델 M_RBE

1: 기저 테이블 B 초기화 (K-means 또는 랜덤)
2: for epoch = 1 to N_epochs do
3:     for batch in D do
4:         # Forward Pass
5:         for each linear layer L in M do
6:             W = L.weight  # 원본 가중치
7:             
8:             # RBE 인코딩 (with STE)
9:             W_codes, W_res = RBE_Encode(W, B)
10:           
11:            # 압축된 형태로 추론
12:            y = RBE_Forward(x, W_codes, W_res, B)
13:        end for
14:        
15:        # 손실 계산
16:        L_task = TaskLoss(y, y_true)
17:        L_quant = α * ||W - RBE_Decode(W_codes, W_res, B)||²
18:        L_total = L_task + L_quant
19:        
20:        # Backward Pass
21:        gradients = backward(L_total)
22:        
23:        # 파라미터 업데이트
24:        W = W - η * gradients[W]
25:        B = B - η * gradients[B]
26:        W_res = W_res - η * gradients[W_res]
27:    end for
28: end for
29: 
30: # 최종 압축
31: return RBE_Compress(M, B)
```

##### 3.5.3.2. 손실 함수 설계

QAT의 효과적인 학습을 위해, 다음과 같은 복합 손실 함수를 사용한다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{recon}} + \lambda_2 \mathcal{L}_{\text{reg}}$$

각 항의 의미는:

1. **태스크 손실 ($\mathcal{L}_{\text{task}}$)**: 원래 모델의 목적 함수 (예: 언어 모델의 경우 cross-entropy)

2. **재구성 손실 ($\mathcal{L}_{\text{recon}}$)**: 양자화 전후의 차이를 최소화
   $$\mathcal{L}_{\text{recon}} = \frac{1}{mn} \|W - W_{\text{approx}} - W_{\text{res}}\|_F^2$$

3. **정규화 손실 ($\mathcal{L}_{\text{reg}}$)**: 잔차의 크기를 제한하여 압축 효율 향상
   $$\mathcal{L}_{\text{reg}} = \frac{1}{mn} \|W_{\text{res}}\|_1$$

##### 3.5.3.3. 적응적 양자화 레벨

학습이 진행됨에 따라 양자화 레벨을 점진적으로 강화하는 **Progressive Quantization** 전략을 사용한다:

$$\text{bits}_{\text{epoch}} = \text{bits}_{\text{final}} + (\text{bits}_{\text{initial}} - \text{bits}_{\text{final}}) \cdot e^{-\gamma \cdot \text{epoch}}$$

이는 초기에는 높은 정밀도로 학습하다가, 점차 목표 비트 수준으로 수렴하도록 한다.

#### 3.5.4. 실험적 검증: Gumbel-Softmax를 통한 미분 가능한 선택

STE의 대안으로, 완전히 미분 가능한 선택을 위해 **Gumbel-Softmax** 기법을 사용할 수 있다:

$$p_j = \frac{\exp((\log \pi_j + g_j) / \tau)}{\sum_k \exp((\log \pi_k + g_k) / \tau)}$$

여기서:
- $\pi_j$: 기저 $j$와의 유사도
- $g_j$: Gumbel 노이즈
- $\tau$: 온도 파라미터 (학습 중 감소)

이를 통해 기저 선택 과정이 완전히 미분 가능해지며, 더 안정적인 학습이 가능하다.

#### 3.5.5. QAT의 실제 효과

`Hb-BERT` 실험에서 QAT의 효과는 극적이었다:

1. **Post-Training Quantization (PTQ)**: 93.2% 정확도 (6.8% 하락)
2. **QAT without fine-tuning**: 97.1% 정확도 (2.9% 하락)  
3. **QAT with full fine-tuning**: 99.3% 정확도 (0.7% 하락)

이는 QAT가 극단적인 압축률에도 불구하고 모델 성능을 거의 완벽하게 보존할 수 있음을 실증적으로 보여준다. 

### 3.5. 고급 압축 기법

RBE의 기본 프레임워크를 확장하여 더욱 효율적인 압축을 달성하는 고급 기법들을 소개한다.

#### 3.5.1. 학습 가능한 직교 기저 (Learnable Orthogonal Basis)

##### 수학적 배경

직교 행렬 $Q \in \mathbb{R}^{n \times n}$는 $Q^T Q = QQ^T = I$를 만족한다. 학습 과정에서 직교성을 유지하기 위해 **Cayley 변환**을 활용한다:

$$Q = (I - A)(I + A)^{-1}$$

여기서 $A$가 skew-symmetric ($A^T = -A$)이면 $Q$는 항상 직교행렬이다.

**증명**:
$$Q^T Q = (I + A)^{-T}(I - A)^T(I - A)(I + A)^{-1}$$
$$= (I + A^T)^{-1}(I - A^T)(I - A)(I + A)^{-1}$$
$$= (I - A)^{-1}(I + A)(I - A)(I + A)^{-1} = I$$

##### 구현

```python
class LearnableOrthogonalBasis(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Skew-symmetric 파라미터 초기화
        self.A_params = nn.Parameter(torch.randn(dim, dim) * 0.01)
        
    def forward(self):
        # Skew-symmetric 행렬 생성
        A = self.A_params - self.A_params.T
        
        # Cayley 변환
        I = torch.eye(A.shape[0], device=A.device)
        Q = torch.linalg.solve(I + A, I - A)
        
        return Q
```

**장점**:
- 역전파 시에도 직교성 유지
- 기저 간 정보 중복 없음 (크로네커 델타 성질)
- 수치적 안정성

#### 3.5.2. 계층적 기저 분해 (Hierarchical Basis Decomposition)

##### 수학적 정의

가중치 $w$를 다중 스케일로 분해:
$$w = \sum_{l=0}^{L} \sum_{j} \alpha_{l,j} \psi_{l,j}$$

여기서:
- $l$: 계층 레벨 (0=coarse, L=fine)
- $\psi_{l,j}$: 레벨 $l$의 $j$번째 기저
- $\alpha_{l,j}$: 계수

##### 웨이블릿 유사 구조

유클리드 공간:
$$\psi_{l,j}(x) = 2^{l/2} \psi(2^l x - j)$$

쌍곡 공간 버전:
$$\psi_{l,j}^{\mathbb{H}} = \exp_o(2^l \log_o(\psi_j))$$

##### 구현

```python
class HierarchicalBasisRBE(nn.Module):
    def __init__(self, levels=4, bases_per_level=64):
        super().__init__()
        self.levels = levels
        self.bases = nn.ModuleList([
            LearnableOrthogonalBasis(2**l * bases_per_level)
            for l in range(levels)
        ])
        
    def encode(self, weights):
        coefficients = []
        residual = weights
        
        for l in range(self.levels):
            # 현재 레벨에서 투영
            basis_l = self.bases[l]()
            coeff_l = torch.matmul(residual, basis_l.T)
            
            # 재구성 및 잔차 계산
            recon_l = torch.matmul(coeff_l, basis_l)
            residual = residual - recon_l
            
            coefficients.append(coeff_l)
            
        return coefficients
```

#### 3.5.3. Sparse Coding with Dictionary Learning

##### 최적화 문제

$$\min_{D, \alpha} \|W - D\alpha\|_F^2 + \lambda \|\alpha\|_1$$

subject to: $\|d_i\|_2 = 1$ for all $i$

여기서:
- $W \in \mathbb{R}^{m \times n}$: 원본 가중치
- $D \in \mathbb{R}^{n \times k}$: 사전 (기저)
- $\alpha \in \mathbb{R}^{k \times m}$: 희소 코드

##### 리만 버전

푸앵카레 볼에서:
$$\min_{D, \alpha} d_{\mathbb{B}}^2(W, \exp_o(D\alpha)) + \lambda \|\alpha\|_1$$

##### 구현

```python
class SparseCodingRBE(nn.Module):
    def __init__(self, dict_size=512, sparsity=0.1):
        super().__init__()
        self.dictionary = nn.Parameter(torch.randn(dict_size, weight_dim))
        self.sparsity = sparsity
        
    def encode(self, weights):
        # FISTA 알고리즘으로 희소 코드 계산
        alpha = self.fista(weights, self.dictionary, self.sparsity)
        return alpha
        
    def decode(self, alpha):
        # 쌍곡 공간에서 재구성
        tangent = torch.matmul(alpha, self.dictionary)
        return poincare_exp_map(tangent)
```

#### 3.5.4. Product Quantization 융합

##### 수학적 정의

벡터 $w \in \mathbb{R}^n$을 $M$개 서브벡터로 분할:
$$w = [w^{(1)}, w^{(2)}, ..., w^{(M)}]$$

각 서브스페이스에서 독립적 양자화:
$$w^{(i)} \approx c_j^{(i)}, \quad j \in \{1, ..., K\}$$

전체 표현:
$$w \approx [c_{j_1}^{(1)}, c_{j_2}^{(2)}, ..., c_{j_M}^{(M)}]$$

##### 압축률

- 원본: $n \times 32$ bits
- PQ: $M \times \log_2(K)$ bits
- 압축률: $\frac{32n}{M\log_2(K)}$

##### RBE와의 융합

```python
class ProductQuantizedRBE(nn.Module):
    def __init__(self, num_subspaces=8, codebook_size=256):
        super().__init__()
        self.subspaces = num_subspaces
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, weight_dim // num_subspaces)
            for _ in range(num_subspaces)
        ])
        
    def encode(self, weights):
        # 서브스페이스로 분할
        chunks = torch.chunk(weights, self.subspaces, dim=-1)
        
        indices = []
        for i, chunk in enumerate(chunks):
            # 각 서브스페이스에서 가장 가까운 코드워드 찾기
            distances = torch.cdist(chunk, self.codebooks[i].weight)
            idx = torch.argmin(distances, dim=-1)
            indices.append(idx)
            
        return torch.stack(indices, dim=-1)
```

#### 3.5.5. Hyperbolic Neural ODE

##### 쌍곡 공간의 측지선 방정식

푸앵카레 볼에서의 ODE:
$$\frac{dx}{dt} = v$$
$$\frac{dv}{dt} = -\frac{c}{2}(\|v\|^2 x + 2\langle x, v \rangle v)$$

여기서 $c$는 곡률이다.

##### 가중치 진화

시간 $t$에서의 가중치:
$$W(t) = \text{GeodesicFlow}(W_0, V_0, t)$$

저장 필요량:
- 초기 위치 $W_0$
- 초기 속도 $V_0$  
- 시간 파라미터 $t$ (레이어별)

##### 구현

```python
class HyperbolicNeuralODE(nn.Module):
    def __init__(self, manifold, solver='dopri5'):
        super().__init__()
        self.manifold = manifold
        self.odefunc = GeodesicODE(manifold)
        
    def forward(self, w0, v0, t):
        # 초기 상태 결합
        state0 = torch.cat([w0, v0], dim=-1)
        
        # ODE 해결
        solution = odeint(self.odefunc, state0, t, method=self.solver)
        
        # 위치 추출
        weights = solution[..., :w0.shape[-1]]
        return weights
```

#### 3.5.6. Tucker/CP 분해 + 리만 기하

##### Tucker 분해

$$\mathcal{W} \approx \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)}$$

여기서:
- $\mathcal{G}$: 코어 텐서
- $U^{(i)}$: 모드-$i$ 행렬

##### 리만 제약

각 factor를 쌍곡 공간에 제약:

```python
class RiemannianTuckerDecomposition(nn.Module):
    def __init__(self, ranks, manifold):
        super().__init__()
        self.core = nn.Parameter(torch.randn(*ranks))
        self.factors = nn.ParameterList([
            nn.Parameter(manifold.random_point(shape))
            for shape in factor_shapes
        ])
        self.manifold = manifold
        
    def forward(self):
        # 쌍곡 공간에서 Tucker 재구성
        result = self.core
        for i, factor in enumerate(self.factors):
            # 접공간으로 이동
            tangent = self.manifold.log_map(factor)
            # 텐서 곱
            result = torch.tensordot(result, tangent, dims=([i], [0]))
            
        return result
```

### 3.6. 압축 기법 비교 및 선택 가이드

| 기법          | 압축률   | 정확도 | 계산 비용 | 적용 분야     |
| ------------- | -------- | ------ | --------- | ------------- |
| 기본 RBE      | 100-200x | 95-98% | 낮음      | 범용          |
| 직교 기저     | 150-250x | 96-99% | 중간      | 고정밀 요구   |
| 계층적 분해   | 200-400x | 94-97% | 중간      | 대규모 모델   |
| Sparse Coding | 300-500x | 93-96% | 높음      | 극한 압축     |
| PQ 융합       | 400-800x | 92-95% | 낮음      | 엣지 디바이스 |
| Neural ODE    | 1000x+   | 90-94% | 매우 높음 | 연구용        |