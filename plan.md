# Reality Stone: 함수형 다양체 압축 및 심플렉틱 동역학 구현 계획

## 0. 철학 및 핵심 정리 (Philosophy)
**"양자화(Quantization)는 엄격히 금지된다. 우리는 함수형 다양체 학습을 통한 구조적 압축을 지향한다."**

### 불가능성 정리와 돌파구
- **사드 정리 (Sard's Theorem)**: 수학적으로 $M < P$인 차원에서 $\mathbb{R}^P$의 임의의 가중치를 무손실로 커버하는 것은 불가능하다.
- **발상의 전환**: 우리는 '임의의' 가중치가 아닌, **'학습된' 가중치**를 압축한다. 학습된 가중치는 거대한 파라미터 공간 내의 극도로 낮은 차원의 **함수형 다양체(Functional Manifold)** $\mathcal{M}_{trained}$ 위에 존재한다.
- **해결책**: **내재적 신경 표현(INR)** + **심플렉틱 환원(Symplectic Reduction)**.
  - 행렬 $W^{(l)}$을 저장하는 대신, 이를 생성하는 함수 $\phi(l)$을 학습한다.
  - FFN을 단순 포텐셜로 근사하는 대신(비보존장에서 실패함), 해밀토니안 시스템의 **심플렉틱 킥(Symplectic Kick)**으로 모델링한다.

---

## 1. 이론적 프레임워크 (Theoretical Framework)

### 1.1 함수형 다양체 압축 (Functional Manifold Compression)
가중치는 독립적인 점들이 아니라 연속적인 흐름의 스냅샷이다.
$$ W^{(l)} \approx U_{global} \cdot \phi_\theta(l) \cdot V_{global}^\top $$
- **전역 기저 ($U, V$)**: 모든 레이어가 공유하는 고차원 형상(Shape). 모델 전체의 기하학적 구조를 정의. ($d \times r$)
- **하이퍼네트워크 ($\phi_\theta$)**: 레이어 인덱스/임베딩 $l$을 입력받아 **코어 행렬(Core Matrix)** $C^{(l)}$을 출력하는 초소형 MLP. ($r \times r$)
- **제약 사항**: FP16/FP32 정밀도 유지. 오직 구조적 공유를 통해서만 파라미터를 감축.

### 1.2 심플렉틱 동역학 (Symplectic Dynamics)
트랜스포머 레이어는 동역학 시스템의 이산적 단계(Step)이다. "임의의 포텐셜"을 제거하고 물리적 정확성을 확보한다.
- **위상 공간 리프팅 (Phase Space Lifting)**: 상태를 $(q, p) \in \mathbb{R}^{2d}$로 확장하여 관리.
- **어텐션 (Attention)**: 위치 $q$에 기반한 메트릭 흐름(Metric Flow).
- **FFN**: 운동량 $p$에 작용하는 **비보존적 킥(Non-conservative Kick)**. 비보존장(Curl $\neq 0$)을 자연스럽게 처리.
- **업데이트 규칙**: 심플렉틱 오일러 (Symplectic Euler)
  $$ p_{t+1} = p_t + \Delta t \cdot F_{FFN}(q_t) $$
  $$ q_{t+1} = q_t + \Delta t \cdot M^{-1} p_{t+1} $$

---

## 2. 아키텍처: RS-ULF v2

### 2.1 핵심 컴포넌트
1.  **`HyperMetric` (Rust/Python)**
    - **역할**: 가중치 생성 엔진.
    - **구성**: `u_global`, `v_global` (공유 기저) + `TinyMLP` (하이퍼넷).
    - **기능**: 런타임에 레이어 인덱스를 받아 $W$를 복원하지 않고 $(x U) C V^T$ 연산을 수행.
    
2.  **`SymplecticLayer` (Rust/Python)**
    - **역할**: 동역학 연산 유닛.
    - **상태**: $(q, p)$ 튜플 관리.
    - **연산**: `HyperMetric`에서 생성된 코어 텐서를 이용해 심플렉틱 적분 수행.

3.  **`GlobalManifoldLearner` (Python)**
    - **역할**: 학습 및 변환 파이프라인.
    - **Phase 1**: 전체 레이어 가중치 스택킹 -> Randomized SVD -> $U, V$ 추출.
    - **Phase 2**: 코어 텐서 타겟 생성 -> `TinyMLP` 학습 (Functional Fitting).
    - **Phase 3**: 동역학 미세조정 (Calibration).

---

## 3. 구현 로드맵 (Implementation Roadmap)

### 1단계: 코어 모듈 구현 (Rust + Python Bindings)
- [x] **전역 기저 추출기 (`GlobalBasis`)**
    - `extract_global_basis`: SVD 기반 공통 기저 추출 구현 완료.
    - `fold_with_global_basis`: 기저 투영 로직 및 검증 테스트 구현 완료.
- [x] **하이퍼메트릭 엔진 (`HyperMetric` 코어)**
    - `HyperMetric`, `TinyMLP` Rust 구현 및 PyO3 바인딩, 단위 테스트 완료.
    - Python `GlobalManifoldLearner`와 연동되어 전역 기저 + 하이퍼넷 기반 코어 생성, `.rsu` v2 저장/복원, `PyHyperMetric` 인스턴스 생성까지 엔드투엔드 경로 구현.
- [~] **심플렉틱 레이어 (`SymplecticLayer`)**
    - 상태 관리: `q`(Position), `p`(Momentum) 분리 저장 및 심플렉틱 오일러(Symplectic Euler) 업데이트 구현 완료.
    - `HyperMetric`에서 생성된 코어 텐서를 이용한 메트릭 기반 force 적용 및 안정성 테스트는 완료.
    - FFN을 벡터장 킥으로 통합하고, 실제 Transformer 블록 동역학과의 캘리브레이션/LLM 통합은 후속 작업.

### 2단계: 압축 파이프라인 고도화 (Python)
- [x] **매니폴드 학습기 (`ManifoldLearner`)**
    - Python `GlobalManifoldLearner`를 통해 전체 레이어 Q/K 가중치 수집 및 전처리, 전역 기저 기반 코어 텐서 타깃 생성, 레이어 임베딩 기반 하이퍼넷 학습 루프 구현.
    - 토이 모델 및 작은 스택 기준 배치 학습·검증까지 완료, 초거대 모델용 데이터 파이프라인/운영 래퍼는 별도 과제로 남김.
- [~] **변환기 리팩토링 (`RSULFTransformerConverter`)**
    - 기존 '레이어별 SVD' 로직을 RS-ULF 전용으로 유지하면서, 전역 기저 + 랭크 플랜 기반 압축 경로를 옵션으로 통합.
    - `.rsu` v2 포맷 및 HyperMetric·Symplectic 경로는 `GlobalManifoldLearner` 파이프라인에서 제공되고, `RSULFTransformerConverter` 내부로의 완전 통합은 진행 중.

### 3단계: 런타임 및 추론 최적화
- [x] **추론 엔진 (`RSULFModel`)**
    - PyTorch `RSULFModel`/`RSULFLayerCUDA`/`RSULFWrapperCUDA`/`RSULFLMHeadCUDA` 기반 RS-ULF 추론 경로 구현 및 단위 테스트 완료.
    - $(q, p)$ 위상공간 심플렉틱 런타임과의 통합 및 고수준 LLM 어댑터에서의 일관된 API 노출은 후속 단계로 남김.
- [~] **CUDA 커널 최적화**
    - RS-ULF용 fused forward/batch/unified CUDA 커널과 Python 래퍼 구현 및 기본 수치 안정성 검증 완료.
    - HyperMetric·전역 기저·심플렉틱 경로와의 직접 통합 및 대규모 모델 기준 성능 튜닝은 진행 중.

### 4단계: 검증 및 튜닝
- [~] **정량적 지표 검증**
    - GPT-2/Qwen 등 일부 모델에 대한 RS-ULF/구조적 RS-ULF/HyperMetric·Symplectic 경로 실험 스크립트 및 샘플 지표는 존재.
    - WikiText/C4 등 표준 벤치마크에서의 체계적인 PPL/압축률 리포트, 재현 가능한 벤치마크 스위트는 운영 단계 과제로 남김.
- [ ] **비교 실험 (Ablation)**
    - Global Basis 공유 유무, Symplectic vs Non-Symplectic 안정성 비교, 다양한 랭크·커브처 설정에 대한 체계적 Ablation은 계획 단계.

---

## 4. 정리 대상 (Deprecation List)
- **구버전 이론**: `08_geodesic_compression` 초반의 "단순 지오데식" 이론은 "근사적 관점"으로 격하.
- **스칼라 포텐셜 FFN**: 신규 RS-ULF/심플렉틱 경로에서는 FFN을 벡터장(Vector Field)로 취급하며, 스칼라 포텐셜 FFN 경로는 `test_structural_mode.py` 및 일부 실험 스크립트에 한정된 연구용 코드로 유지.
- **독립 레이어 폴딩**: RS-ULF 레이어별 SVD 기반 폴딩은 여전히 지원하되, 전역 기저 공유 방식이 기본 경로이며, 장기적으로는 Global Basis/Hypenet 기반 파이프라인을 우선시.
- **양자화 코드**: 실제 엔진 경로에서는 사용하지 않으며, 레포지토리 내에서 발견 시 제거 대상.

---

## 5. 현재 구현 현황 (2025-12-03)

| 컴포넌트 | 진행률 | 상태 설명 |
| :--- | :---: | :--- |
| **1. 다양체 학습 (Manifold Learning)** | **90%** | 전역 기저($U, V$) 추출 알고리즘(SVD) 및 `extract_global_basis`/`fold_with_global_basis` 구현·테스트, RS-ULF 변환기와 RS-ULF Rust 바인딩에서의 사용까지 완료되었으며, LLM 어댑터 수준에서 일관된 API로 노출하는 작업은 진행 중이다. |
| **2. 함수형 근사 (Functional Fitting)** | **90%** | `HyperMetric`/`TinyMLP` Rust 코어·PyO3 바인딩·테스트와 Python `GlobalManifoldLearner`를 통해 (1) 전 레이어 Q/K 가중치 수집, (2) 전역 기저 기반 코어 텐서 타깃 생성, (3) 레이어 임베딩 기반 하이퍼넷 학습, (4) 학습된 가중치로 `PyHyperMetric` 인스턴스 생성 및 심플렉틱 래퍼 구성, (5) 학습된 HyperNet을 `.rsu` v2 포맷으로 내보내고 다시 복원하는 엔드투엔드 경로까지 구현되었으며, 대형 LLM/RS-ULF 변환기와의 통합 학습·평가 래퍼는 남아 있다. |
| **3. 심플렉틱 런타임 (Symplectic Runtime)** | **90%** | Rust `SymplecticLayer`/`SymplecticState` + PyO3 바인딩과 Python `SymplecticModelWrapper`를 통해 HyperMetric 기반 메트릭 force와 실제 Transformer 블록의 FFN 잔차(비보존 벡터장)를 운동량 킥으로 통합한 심플렉틱 오일러 업데이트, 토이 모델 기준 엔드투엔드 경로 및 안정성 테스트까지 구현되었으며, 대형 LLM 수준의 동역학 캘리브레이션·벤치마크는 후속 과제로 남아 있다. |
| **4. 파이썬/CUDA 파이프라인** | **85%** | RS-ULF CPU/CUDA 래퍼(`RSULFLayerCUDA`/`RSULFWrapperCUDA`/`RSULFLMHeadCUDA`)와 변환기(`RSULFTransformerConverter`), 전역 기저/랭크 플랜 연동, manifold/RS-ULF 관련 테스트 스위트 및 일부 LLM 변환·추론 실험 스크립트는 구현 완료되었고, HyperMetric·Symplectic·`.rsu` v2 경로를 포함한 일관된 고수준 LLM 어댑터/추론 API와 정량 벤치마크 스위트는 후속 작업이다. |
| **종합 완성도** | **~80%** | 함수형 다양체 학습(전역 기저 + HyperNet)과 RS-ULF/심플렉틱 런타임, Python/CUDA 파이프라인, `.rsu` v2 저장/복원 및 토이/부분 LLM 실험 경로까지는 구현이 완료된 상태이며, 실제 LLM 기준 정량 지표·Ablation·운영용 어댑터 정비가 남아 있는 중간 완성도 단계이다. |
