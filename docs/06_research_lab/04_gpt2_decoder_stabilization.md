# GPT-2 RS-ULF Decoder Stabilization & Fidelity Fix

## 1. 개요
GPT-2를 RS-ULF로 변환 시, 마지막 레이어(Layer 11)의 은닉 상태(Hidden State)가 Teacher 모델과 직교(Orthogonal)에 가까워지는 현상 발생.
이로 인해 기존 `ln_f` + `lm_head`를 그대로 사용할 경우 로짓(Logit) 분포가 붕괴되고 생성된 텍스트가 깨지는 문제 해결.

## 2. 문제 상황
- **현상**: Layer 11에서 Cosine Similarity ≈ 0.03, Relative L2 Error ≈ 2.4로 구조적 불일치 심각.
- **원인**:
  1. `RSULFTransformerConverter(exact=True)` 모드에서 `global_basis`가 추출되지 않음.
  2. 기존 `fit_riemannian_decoder`는 `global_basis`가 없으면 `None`을 반환.
  3. 디코더 없이 붕괴된 은닉 상태를 그대로 출력층에 통과시켜 텍스트 생성 실패.
  4. 랭크가 낮아질수록(`r=178` 이하) 과적합 및 분포 이탈로 인해 특수문자/무의미한 단어 생성.

## 3. 해결 방안 (구현 완료 및 진행 중)

### 3.1 디코더 학습 강제 (완료)
- `fit_riemannian_decoder` 수정:
  - `global_basis`가 없어도 항등 행렬(Identity Matrix)을 기저로 사용하여 **순수 선형 디코더(Linear Decoder)**를 학습하도록 변경.
  - 이를 통해 RS-ULF 마지막 은닉 상태를 Teacher 로짓 공간으로 매핑하는 보정 레이어 역할 수행.
- **결과**: `r=356`에서 텍스트 생성 정상화 확인.

### 3.2 코드 구조 분리 (진행 중)
- **목적**: 추론(Inference) 로직과 학습(Fitting) 로직의 엄격한 분리.
- **변경**:
  - `experiments/gpt2/decoder.py`: 순수 텍스트 생성 및 추론 유틸리티 (`rsulf_generate_text`, `_sample_next_token`).
  - `experiments/gpt2/dt.py`: 디코더 학습 및 데이터 수집 로직 (`fit_riemannian_decoder`, `_collect_decoder_data`).

### 3.3 안정성 강화 (예정)
- **데이터 증강**: 디코더 학습 시 배치 크기(`num_batches`) 확대 (16 -> 64+).
- **정규화(Regularization)**: 랭크가 낮을 때(`r < 200`) 발생하는 과적합 방지를 위해 Ridge Regression 정규화 계수 조정 또는 구조 단순화.

## 4. 작업 로그
- `decoder.py`에서 학습 로직 제거 및 정리.
- `dt.py` 생성 및 학습 로직 이관.
- `main.py` 임포트 경로 수정.
