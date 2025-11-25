# 리만 하이퍼 컴프레션: 웜홀 효과 (Riemannian Hyper-Compression)

## 1. 이론: 사르카의 구성 (Sarkar's Construction, 2011)
수학적 증명(Sarkar, 2011; Sala, 2018)에 따르면, **쌍곡 공간(Hyperbolic Space)**은 유클리드 공간에서는 노드 수에 비례하여 차원이 늘어나야 하는 트리 구조를, 단 **2차원**만으로도 왜곡 없이 임베딩할 수 있습니다.

**Reality Stone**은 이 원리를 이용하여 신경망 내부에 **"웜홀(Wormhole)"**을 생성합니다.

## 2. 아키텍처 (Architecture)

표준적인 병목(Autoencoder) 대신, **다양체 병목(Manifold Bottleneck)**을 사용합니다.

$$ \text{Input (High-Dim)} \xrightarrow{\text{Exp}_c} \text{Hyperbolic (Low-Dim)} \xrightarrow{\text{Log}_c} \text{Output (High-Dim)} $$

### 1단계: 인코더 (접공간으로)
$$ z_{tan} = x W_{enc} + b $$
$768 \to 64$ 차원으로 매핑합니다.

### 2단계: 다양체 투영 (웜홀 진입)
접벡터를 푸앵카레 볼 위로 투영합니다:
$$ z_{hyp} = \tanh(\|z_{tan}\|) \frac{z_{tan}}{\|z_{tan}\|} $$

이 64차원 굽은 공간은 수천 차원의 유클리드 공간에 맞먹는 계층 표현 용량을 가집니다.

### 3단계: 디코더 (접공간에서 복귀)
다음 레이어를 위해 다시 유클리드 공간으로 매핑합니다:
$$ y = \text{arctanh}(\|z_{hyp}\|) \frac{z_{hyp}}{\|z_{hyp}\|} W_{dec} $$

## 3. 장점 (Advantages)
- **극한 압축 (Extreme Compression)**: 히든 차원을 10배~100배 줄일 수 있습니다.
- **구조 보존 (Structure Preservation)**: 쌍곡 병목은 모델이 표면적인 패턴을 암기하는 대신, 데이터의 진정한 **계층적 본질**을 학습하도록 강제합니다.
- **노이즈 제거 (Denoising)**: 무작위 노이즈(높은 엔트로피, 비계층적 정보)는 쌍곡 투영을 통과하지 못하고 걸러집니다. 이는 강력한 기하학적 필터 역할을 합니다.
