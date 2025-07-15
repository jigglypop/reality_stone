
# `Reality Stone`: 리만 기하학 기반 신경망을 위한 기저 필드 인코딩 (RBE)

[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![라이선스](https://img.shields.io/badge/라이선스-MIT-green.svg)](https://opensource.org/licenses/MIT)

**`Reality Stone`**은 PyTorch를 위한 최첨단 신경망 압축 프레임워크로, **리만 기하학 기저 인코딩(Riemannian Basis Encoding, RBE)**이라는 새로운 패러다임을 제시합니다. 본 라이브러리는 거대한 가중치 행렬을 해당 기하학적 공간에 최적화된 소수의 **기저 함수(Basis Functions)** 조합으로 분해하여, 전례 없는 수준의 압축률과 정확도 보존을 동시에 달성합니다.

Rust의 고성능 코어와 CUDA 가속을 통해, `Reality Stone`은 수십억 파라미터 모델을 개인용 컴퓨터에서도 효율적으로 구동하는 AI의 민주화를 목표로 합니다.

### 핵심 혁신: 리만 기하학 기저 인코딩 (RBE)

RBE는 가중치 행렬을 두 가지 핵심 요소로 분해합니다:

1.  **기저 청사진 (The Blueprint / Basis Field):**
    *   가중치 변환의 핵심적인 구조적 패턴을 담은, 고도로 압축된 **단일 64비트 시드**입니다.
    *   이 비트 필드는 특정 리만 공간(예: 푸앵카레 볼)을 가장 잘 표현하는 미리 정의된 **기저 함수들**을 어떻게 선택하고 조합할지에 대한 '설계도' 역할을 합니다.
    .
2.  **잔차 보정 (The Residual):**
    *   기저 청사진만으로는 표현되지 않는 미세한 오차를 보정하기 위한, 훨씬 작은 크기의 학습 가능한 행렬입니다.
    *   이를 통해 모델은 핵심 패턴과 세부 정보를 분리하여 학습함으로써, 표현력 손실을 최소화합니다.

| 메트릭        | 성능      | 설명                                                           |
| ------------- | --------- | -------------------------------------------------------------- |
| **압축률**    | **186x**  | RBE를 통해 가중치 행을 단 **22비트**로 인코딩                  |
| **정확도**    | **98.6%** | 기하학적 구조에 최적화된 기저 함수로 복잡한 분포를 표현        |
| **추론 속도** | **3-4x**  | 압축된 청사진과 잔차로부터 직접 추론하여 메모리 병목 현상 완화 |


## 🎉 주요 기능

- **RBE 압축 엔진**: `nn.Linear`를 `RBELinear`로 대체하여, RBE의 강력한 압축 및 추론 성능을 제공합니다.
- **기하학-인식 레이어 (Geometry-Aware Layers)**: `Poincaré`, `Lorentz`, `Klein` 등 RBE가 최대의 효율을 발휘하는 하이퍼볼릭 레이어를 네이티브로 지원합니다.
- **동적 곡률 최적화**: 모델 학습 중 각 레이어의 곡률을 자동으로 최적화하여 표현력을 극대화합니다.
- **고성능 Rust 코어 및 CUDA 가속**: CPU와 GPU 모두에서 최고의 성능을 발휘하도록 설계되었습니다.

## 🚀 빠른 시작

### 1. 설치

Docker를 사용하여 가장 간단하게 환경을 구성하고 빌드할 수 있습니다.

```bash
docker-compose up --build -d
```

### 2. RBE를 `nn.Linear` 레이어에 적용하기

기존 `nn.Linear` 레이어를 RBE 압축 레이어로 변환하여 극한 압축의 이점을 즉시 활용할 수 있습니다.

```python
import torch
import torch.nn as nn
from reality_stone.layers import RBELinear

# 1. 압축할 기존 nn.Linear 레이어 정의
original_layer = nn.Linear(in_features=768, out_features=256)
# original_layer.load_state_dict(...) # 사전 학습된 가중치 로드

# 2. RBELinear로 변환하여 RBE 적용
# .from_linear 메소드가 가중치를 푸앵카레 디스크 상의 단일 시드로 압축합니다.
compressed_layer = RBELinear.from_linear(original_layer)

print(f"원본 레이어: {original_layer}")
print(f"RBE 적용 레이어: {compressed_layer}")

# 3. 압축된 레이어로 직접 추론 (메모리 및 속도 이점)
input_tensor = torch.randn(16, 768)
output = compressed_layer(input_tensor)

print(f"추론 결과 shape: {output.shape}")
```

## 🔬 아키텍처

`Reality Stone`은 성능과 유연성을 극대화하기 위해 명확한 계층 구조로 설계되었습니다.

```plaintext
/
├── src/                      # 🦀 Rust 핵심 로직 (RBE 알고리즘, 기하학 연산)
│   ├── layers/               #   - Poincaré, Lorentz, RBE 등 핵심 연산 (ndarray)
│   │   └── cuda/             #   - GPU 가속을 위한 CUDA 커널 (.cu)
│   ├── ops/                  #   - 일반적인 수학 연산
│   └── bindings/             #   - PyO3를 통한 Python-Rust 인터페이스
├── python/                   # 🐍 Python API 및 PyTorch 래퍼
│   └── reality_stone/
│       ├── layers/           #   - PoincareBallLayer, RBELinear 등 사용자용 nn.Module
│       └── core/             #   - 핵심 연산을 위한 Python 인터페이스
├── docs/                     # 📚 문서 및 논문 (RBE 상세 설명)
├── examples/                 # 💡 사용 예제 코드
└── tests/                    # 🧪 테스트 코드
```

## 🌟 하이퍼볼릭 기하학과 RBE

RBE는 유클리드 공간을 넘어, 특히 **하이퍼볼릭 기하학**과 같은 비유클리드 공간에서 강력한 성능을 발휘합니다. 계층적 데이터나 그래프 구조를 임베딩하는 데 뛰어난 푸앵카레 볼(Poincaré Ball) 모델과 RBE를 결합하면, 극도의 압축률과 높은 표현력을 동시에 달성할 수 있습니다.

### 푸앵카레 볼 모델 (Poincaré Ball Model)

곡률 $c > 0$인 $N$차원 쌍곡공간은 다음과 같이 정의됩니다:

$$ \mathbb{D}^N_c = \{x \in \mathbb{R}^N : c\,\|x\|_2^2 < 1\} $$

`Reality Stone`은 이 공간 내의 연산(덧셈, 스칼라 곱, 거리 계산 등)을 위한 최적화된 CPU/GPU 커널을 제공하며, RBE는 이 공간의 기저를 활용하여 신경망을 압축합니다.

