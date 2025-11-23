# Implicit Riemannian Hub Networks (IRHN): 자연스러운 위계의 창발

## 1. 개요 (Overview)

본 문서는 인위적인 계층 정의나 명시적인 허브 파라미터 없이, 데이터 간의 상호작용(Interaction)과 리만 기하학적 공간(Riemannian Geometric Space)의 특성만을 이용하여 자연스럽게 위계 구조(Hierarchy)가 창발(Emergence)하는 신경망 모델을 기술한다.

이 모델은 뇌 신경망의 자기 조직화(Self-Organization) 원리와 물리학의 포텐셜 에너지 이론에 기반한다.

## 2. 수학적 원리 (Mathematical Principles)

### 2.1. 리만 다양체와 정보 기하학 (Riemannian Manifold & Information Geometry)

데이터 공간을 일정한 음의 곡률 $-c$를 갖는 로렌츠 모델(Lorentz Model) 또는 푸앵카레 볼(Poincaré Ball) $\mathbb{D}_c^n$로 정의한다.

$$ d_{\mathbb{D}}(x, y) = \frac{1}{\sqrt{c}} \text{arcosh}\left(1 + 2c \frac{\|x-y\|^2}{(1-c\|x\|^2)(1-c\|y\|^2)}\right) $$

이 공간의 핵심 특성은 **"지수적 팽창(Exponential Expansion)"**이다. 원점(Origin)에서의 거리가 멀어질수록 공간의 부피가 지수적으로 늘어난다.
*   **원점 근처**: 공간이 좁아 모든 노드 간의 거리가 가깝다. $\rightarrow$ **전역적 연결 (Global Connection) = 상위 계층 (Root)**
*   **가장자리**: 공간이 넓어 이웃 노드 외에는 거리가 매우 멀다. $\rightarrow$ **지역적 연결 (Local Connection) = 하위 계층 (Leaf)**

즉, **데이터의 놈(Norm) $\|x\|$ 자체가 계층의 깊이(Depth)를 나타낸다.**

### 2.2. 암시적 허브 어텐션 (Implicit Hub Attention)

별도의 허브 벡터 $h_k$를 두지 않고, 배치(Batch) 내의 데이터 집합 $\mathcal{X} = \{x_1, \dots, x_B\}$ 내에서 상호 작용을 통해 허브를 정의한다.

각 데이터 포인트 $x_i$는 잠재적 그래프의 노드가 되며, 연결 강도(Adjacency) $A_{ij}$는 쌍곡 거리(Hyperbolic Distance)에 의해 결정된다.

$$ A_{ij} = \frac{\exp(-d_{\mathbb{D}}(x_i, x_j)^2 / \tau)}{\sum_k \exp(-d_{\mathbb{D}}(x_i, x_k)^2 / \tau)} $$

여기서 $\tau$는 온도 파라미터이다.

### 2.3. 정보의 흐름과 허브의 창발 (Message Passing & Emergence of Hubs)

정보(Feature) $z_i$는 연결된 노드들을 통해 전파된다. 탄젠트 공간(Tangent Space) $T_{x_i}\mathbb{D}$ 근사를 이용한 업데이트 식은 다음과 같다:

$$ z_i^{(t+1)} = \text{Exp}_{x_i} \left( \sum_{j} A_{ij} \cdot \text{Log}_{x_i}(z_j^{(t)}) \right) $$

이 과정에서 **"중심성(Centrality)"**이 높은 노드, 즉 많은 다른 노드들과 가깝게 위치한 데이터 포인트는 자연스럽게 높은 $A_{ij}$ 합을 갖게 되며, 정보의 집결지인 **"허브(Hub)"** 역할을 수행한다.
반면, 외곽에 고립된 노드는 자신의 정보만 보존하거나 가장 가까운 허브의 정보만 수신하므로 **"리프(Leaf)"** 역할을 수행한다.

이것은 **"순서(Sequence)"**나 **"위치(Geometry)"**에 따라 어떤 노드는 허브가 되고, 어떤 노드는 리프가 되는 동적인 과정이다.

### 2.4. 물리적 에너지 관점 (Physical Energy Perspective)

이 시스템은 전체 시스템의 **자유 에너지(Free Energy)**를 최소화하는 방향으로 거동한다.

$$ F = \sum_{i} \mathbb{E}_{q(z|x)} [E(x, z)] - H(q) $$

여기서 에너지 함수 $E$는 리만 거리의 제곱에 비례한다. 데이터들이 서로 당기는 인력(Attraction)과 공간의 척력(Repulsion)이 평형을 이루는 상태에서 자연스러운 클러스터링(Clustering)이 발생한다.

## 3. 모델 아키텍처 (Model Architecture)

### 3.1. 구조

1.  **Embedding Layer**: 입력을 고차원 벡터로 변환.
2.  **Hyperbolic Projection**: $\text{Exp}_0(x)$를 통해 리만 다양체로 투영.
3.  **Implicit Graph Layer**:
    *   배치 내 모든 쌍(Pairwise)의 리만 거리 계산.
    *   거리 기반 Softmax로 Adjacency Matrix $A$ 생성.
    *   $A$를 이용한 Message Passing (정보 교환).
    *   이 과정에서 데이터들이 서로를 참조하며 위계적 위치를 재조정함.
4.  **Classification Head**: 최종 위치 기반 분류.

### 3.2. 동작 예시

*   **입력**: ["신한은행 예금", "신한은행 대출", "국민은행 예금"]
*   **초기**: 무작위 위치.
*   **상호작용**:
    *   "신한은행"이라는 공통 분모를 가진 데이터끼리 거리가 가까워짐.
    *   "신한은행 예금"과 "신한은행 대출"의 중점 근처로 "신한은행"이라는 추상적 개념(실제 데이터에는 없더라도)이 형성됨.
    *   데이터들이 이 가상의 중점을 중심으로 공전(Orbit)하거나 뭉침.

## 4. 결론 (Conclusion)

이 모델은 "무엇이 허브인가"를 정의하지 않는다. 대신 **"허브가 될 수 있는 공간적 조건"**만을 리만 기하학으로 제공한다. 데이터는 그 공간 안에서 상호 작용하며 스스로 위계를 형성한다. 이는 생물학적 뇌의 가소성(Plasticity) 및 자기 조직화 임계성(Self-Organized Criticality)과 수학적으로 동치이다.

