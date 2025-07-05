## 5. 시스템 아키텍처

`Reality Stone`은 성능, 안정성, 사용성을 모두 만족시키기 위해 세심하게 설계된 다층 아키텍처를 채택하고 있다.

```mermaid
graph TD
    subgraph Python Layer (User Facing)
        A[Python API: PyTorch-like Interface] --> B{PyO3 Bindings};
    end
    subgraph Rust Core (High Performance & Safety)
        B --> C[Core Logic & Abstractions];
        C --> D[Poincaré, Lorentz, Klein Ops];
        D --> |Optional| E[CUDA Kernels];
    end
    subgraph GPU Acceleration
        E[CUDA Kernels for Parallel Ops]
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f96,stroke:#333,stroke-width:2px
    style E fill:#9cf,stroke:#333,stroke-width:2px
```

1.  **Python API Layer**: 사용자와 직접 상호작용하는 최상위 계층. PyTorch의 `nn.Module`과 유사한 인터페이스(`PoincareLayer`, `SplineLinear` 등)를 제공하여, 기존 딥러닝 개발자들이 익숙한 환경에서 쉽게 쌍곡 신경망을 구축하고 실험할 수 있도록 설계되었다. 내부적으로는 `PyO3` 프레임워크를 통해 Rust로 구현된 고성능 함수를 호출한다.

2.  **Rust Core Layer**: 라이브러리의 심장부. 모든 핵심 수학 연산, 자료구조, 그리고 알고리즘이 Rust로 구현되어 있다. Rust의 소유권(Ownership) 시스템과 강력한 타입 시스템은 메모리 안정성을 보장하고 데이터 경쟁(data race)을 원천적으로 방지하여, 병렬 처리 환경에서도 안전하고 예측 가능한 성능을 제공한다. `src/ops` 디렉토리에는 각 쌍곡 모델에 대한 연산이, `src/layers`에는 신경망 레이어 로직이 구현되어 있다.

3.  **GPU Acceleration Layer**: 대규모 행렬 연산과 같이 계산 집약적인 작업을 가속하기 위한 CUDA 커널이 `src/ops/cuda`에 위치한다. Rust 코드는 C++ 브릿지를 통해 CUDA 커널을 효율적으로 호출하여 GPU의 대규모 병렬 처리 능력을 최대한 활용한다. 이를 통해 모델 학습과 추론 시간을 획기적으로 단축시킨다. 커널 퓨전(kernel fusion)과 같은 최적화 기법을 적용하여 메모리 접근 병목을 최소화한다. 